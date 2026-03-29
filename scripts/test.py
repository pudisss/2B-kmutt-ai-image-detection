#!/usr/bin/env python3
"""Evaluation / test script for the Multi-Domain AI Image Detector.

Usage
-----
# Basic evaluation with a checkpoint and test manifest:
python scripts/test.py \\
    --checkpoint checkpoints/best.pt \\
    --test-manifest /workspace/data/processed/test_manifest.csv

# Override root directory and output folder:
python scripts/test.py \\
    --checkpoint checkpoints/best.pt \\
    --test-manifest /workspace/data/processed/test_manifest.csv \\
    --root-dir /workspace/data/processed \\
    --output-dir ./eval_results

# Use a custom decision threshold:
python scripts/test.py --checkpoint checkpoints/best.pt --threshold 0.6
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MultiDomainDataset, collate_fn
from src.models.detector import load_detector
from src.training.trainer import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Multi-Domain AI Image Detector"
    )
    parser.add_argument("--checkpoint", "-ckpt", required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", "-c", default="configs/default.yaml",
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--test-manifest", default=None,
                        help="CSV manifest for test data (overrides config)")
    parser.add_argument("--root-dir", default=None,
                        help="Root directory for images (overrides config)")
    parser.add_argument("--output-dir", "-o", default="./eval_results",
                        help="Directory to save evaluation results (default: ./eval_results)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of dataloader workers")
    parser.add_argument("--device", default=None,
                        help="Device: cuda | cpu | mps")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold for fake class "
                             "(default: best_threshold saved in checkpoint)")
    return parser.parse_args()


def _print_table(title: str, metrics: dict) -> None:
    bar = "─" * max(0, 55 - len(title))
    print(f"\n── {title} {bar}")
    for key in ("accuracy", "balanced_accuracy", "auc", "precision", "recall", "f1"):
        if key in metrics:
            print(f"  {key:<22} : {metrics[key]:.4f}")


def _make_serializable(obj):
    """Recursively convert numpy/torch scalars to plain Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    if hasattr(obj, "item"):          # numpy / torch scalar
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]

    root_dir   = args.root_dir   or data_cfg["root_dir"]
    batch_size = args.batch_size or data_cfg.get("batch_size", 32)
    workers    = args.workers    or data_cfg.get("num_workers", 8)
    device     = args.device     or config.get("device", "cuda")

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device = "cpu"

    # ── Test manifest ──────────────────────────────────────────────────────
    test_manifest = args.test_manifest or data_cfg.get("test_manifest")
    if not test_manifest:
        default = Path(root_dir) / "test_manifest.csv"
        if default.exists():
            test_manifest = str(default)
        else:
            sys.exit(
                "ERROR: Provide --test-manifest or set data.test_manifest in config."
            )

    print("=" * 60)
    print("  Multi-Domain AI Image Detector — Evaluation")
    print("=" * 60)
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Test manifest : {test_manifest}")
    print(f"  Root dir      : {root_dir}")
    print(f"  Device        : {device}")

    df = pd.read_csv(test_manifest)
    print(f"  Test images   : {len(df):,}  "
          f"(real: {len(df[df['label']==0]):,} | "
          f"fake: {len(df[df['label']==1]):,})")

    # ── Load model + threshold ─────────────────────────────────────────────
    raw_ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    threshold = args.threshold or float(raw_ckpt.get("best_threshold", 0.5))
    print(f"  Threshold     : {threshold:.3f}")
    print("=" * 60)

    model = load_detector(args.checkpoint, device=device)

    # ── Dataloader ─────────────────────────────────────────────────────────
    test_dataset = MultiDomainDataset(
        df=df,
        root_dir=root_dir,
        image_size=data_cfg.get("image_size", 256),
        augment=False,
        return_metadata=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    evaluator = Evaluator(model=model, device=device)
    results   = evaluator.evaluate(
        test_loader,
        per_architecture=True,
        threshold=threshold,
    )

    # ── Print results ──────────────────────────────────────────────────────
    _print_table("Overall", results["overall"])

    if "per_architecture" in results:
        print("\n── Per-Architecture ──────────────────────────────────────────")
        rows = []
        for arch, m in sorted(results["per_architecture"].items()):
            rows.append({
                "architecture":       arch,
                "n_samples":          m.get("count", "?"),
                "accuracy":           round(m.get("accuracy", 0), 4),
                "balanced_accuracy":  round(m.get("balanced_accuracy", 0), 4),
                "auc":                round(m.get("auc", 0), 4),
            })
        arch_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
        print(arch_df.to_string(index=False))

    if "dataset" in df.columns and "per_architecture" in results:
        print("\n── Per-Dataset ───────────────────────────────────────────────")
        rows = []
        for ds_name, ds_df in df.groupby("dataset"):
            ds_archs = set(ds_df["architecture"].unique())
            arch_metrics = {
                k: v for k, v in results["per_architecture"].items()
                if k in ds_archs
            }
            if not arch_metrics:
                continue
            avg_auc = sum(m.get("auc", 0)      for m in arch_metrics.values()) / len(arch_metrics)
            avg_acc = sum(m.get("accuracy", 0) for m in arch_metrics.values()) / len(arch_metrics)
            rows.append({
                "dataset":       ds_name,
                "n_images":      len(ds_df),
                "architectures": len(ds_archs),
                "avg_accuracy":  round(avg_acc, 4),
                "avg_auc":       round(avg_auc, 4),
            })
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nClassification Report:\n{results['classification_report']}")

    # ── Save results ───────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    print(f"\nResults saved to : {out_path}")


if __name__ == "__main__":
    main()
