#!/usr/bin/env python3
"""Training script for the Multi-Domain AI Image Detector.

Usage
-----
# Train using manifest CSVs (recommended):
python scripts/train.py \\
    --train-manifest /workspace/data/processed/train_manifest.csv \\
    --test-manifest  /workspace/data/processed/test_manifest.csv

# Train using auto-split from dataset configs in default.yaml:
python scripts/train.py

# Override common hyperparameters:
python scripts/train.py --epochs 30 --batch-size 16 --lr 5e-5 --device cuda

# Resume from checkpoint:
python scripts/train.py --resume checkpoints/best.pt
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MultiDomainDataModule
from src.models.detector import create_detector
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Multi-Domain AI Image Detector"
    )
    # Config
    parser.add_argument("--config", "-c", default="configs/default.yaml",
                        help="Path to YAML config (default: configs/default.yaml)")
    # Data overrides
    parser.add_argument("--root-dir", default=None,
                        help="Override data.root_dir from config")
    parser.add_argument("--train-manifest", default=None,
                        help="CSV manifest for training (filepath, label, architecture, dataset)")
    parser.add_argument("--test-manifest", default=None,
                        help="CSV manifest for test / held-out evaluation")
    # Training overrides
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override data.batch_size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training.learning_rate")
    parser.add_argument("--device", default=None,
                        help="Device: cuda | cpu | mps")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override data.num_workers")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    # Checkpointing
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Override training.checkpoint_dir")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume training from")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    config     = load_config(args.config)
    data_cfg   = config["data"]
    model_cfg  = config["model"]
    train_cfg  = config["training"]
    log_cfg    = config.get("logging", {})

    # Apply CLI overrides
    if args.root_dir:    data_cfg["root_dir"]        = args.root_dir
    if args.epochs:      train_cfg["epochs"]         = args.epochs
    if args.batch_size:  data_cfg["batch_size"]      = args.batch_size
    if args.lr:          train_cfg["learning_rate"]  = args.lr
    if args.workers:     data_cfg["num_workers"]     = args.workers
    if args.output_dir:  train_cfg["checkpoint_dir"] = args.output_dir
    if args.seed:        config["seed"]              = args.seed

    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device = "cpu"

    seed = int(config.get("seed", 42))
    set_seed(seed)

    print("=" * 60)
    print("  Multi-Domain AI Image Detector — Training")
    print("=" * 60)
    print(f"  Config     : {args.config}")
    print(f"  Device     : {device}")
    print(f"  Seed       : {seed}")
    print(f"  Epochs     : {train_cfg.get('epochs', 50)}")
    print(f"  Batch size : {data_cfg.get('batch_size', 32)}")
    print(f"  LR         : {train_cfg.get('learning_rate', 1e-4)}")
    print("=" * 60)

    # ── Data module ────────────────────────────────────────────────────────
    train_manifest = args.train_manifest or data_cfg.get("train_manifest")
    test_manifest  = args.test_manifest  or data_cfg.get("test_manifest")

    common_dm_kwargs = dict(
        root_dir=data_cfg["root_dir"],
        image_size=data_cfg.get("image_size", 256),
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=data_cfg.get("pin_memory", True),
        balance_classes=data_cfg.get("balance_classes", True),
        val_split=data_cfg.get("val_split", 0.1),
        seed=seed,
        weighted_sampling=data_cfg.get("weighted_sampling", True),
        augmentation_config=data_cfg.get("augmentation", {}),
    )

    if train_manifest and test_manifest:
        print(f"\nUsing manifest files:")
        print(f"  Train : {train_manifest}")
        print(f"  Test  : {test_manifest}")
        dm = MultiDomainDataModule(
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            **common_dm_kwargs,
        )
    else:
        dataset_configs = data_cfg.get("datasets", [])
        if not dataset_configs:
            sys.exit(
                "ERROR: Provide --train-manifest and --test-manifest, "
                "or set data.datasets in the config file."
            )
        print("\nUsing auto model-aware split from config datasets.")
        dm = MultiDomainDataModule(
            dataset_configs=dataset_configs,
            train_models=data_cfg.get("train_models"),
            test_models=data_cfg.get("test_models"),
            **common_dm_kwargs,
        )

    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # ── Model ──────────────────────────────────────────────────────────────
    rgb_cfg    = model_cfg["rgb_branch"]
    freq_cfg   = model_cfg["frequency_branch"]
    noise_cfg  = model_cfg["noise_branch"]
    fusion_cfg = model_cfg["fusion"]
    cls_cfg    = model_cfg["classifier"]

    model = create_detector(
        rgb_model=rgb_cfg["model_name"],
        rgb_pretrained=rgb_cfg.get("pretrained", True),
        rgb_feature_dim=rgb_cfg.get("feature_dim", 512),
        freq_feature_dim=freq_cfg.get("feature_dim", 256),
        freq_resnet=freq_cfg.get("resnet_type", "resnet18"),
        noise_feature_dim=noise_cfg.get("feature_dim", 256),
        noise_resnet=noise_cfg.get("resnet_type", "resnet18"),
        fusion_type=fusion_cfg.get("type", "attention"),
        fusion_hidden_dim=fusion_cfg.get("hidden_dim", 512),
        fusion_num_heads=fusion_cfg.get("num_heads", 8),
        fusion_dropout=fusion_cfg.get("dropout", 0.3),
        classifier_hidden_dims=cls_cfg.get("hidden_dims", [256, 128]),
        classifier_dropout=cls_cfg.get("dropout", 0.5),
        num_classes=cls_cfg.get("num_classes", 2),
    )

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters : {total_params:,} total  |  {trainable_params:,} trainable")

    # ── Trainer ────────────────────────────────────────────────────────────
    early_cfg = train_cfg.get("early_stopping", {})
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.get("epochs", 50),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        optimizer=train_cfg.get("optimizer", "adamw"),
        scheduler=train_cfg.get("scheduler", "cosine"),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        min_lr=train_cfg.get("min_lr", 1e-6),
        use_amp=train_cfg.get("use_amp", True),
        gradient_clip=train_cfg.get("gradient_clip", 1.0),
        accumulation_steps=train_cfg.get("accumulation_steps", 1),
        early_stopping_patience=early_cfg.get("patience", 10),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "./checkpoints"),
        save_best_only=train_cfg.get("save_best_only", True),
        device=device,
        log_interval=log_cfg.get("log_freq", 100),
        threshold_metric=train_cfg.get("threshold_metric", "balanced_accuracy"),
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # ── Train ──────────────────────────────────────────────────────────────
    trainer.train()

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best Val AUC : {trainer.best_val_auc:.4f}")
    print(f"  Checkpoints  : {trainer.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
