#!/usr/bin/env python3
"""Download and preprocess DiffusionForensics (DIRE) dataset.

Dataset: Wang et al. 2023 "DIRE for Diffusion-Generated Image Detection"
Source:  HuggingFace — set HF_DATASET_ID below to the correct repo ID.

Common HuggingFace locations (verify before running):
  - "nebula/DiffusionForensics"
  - "ZhendongWang6/DiffusionForensics"
  Paper/code: https://github.com/ZhendongWang6/DIRE
  If per-architecture configs exist, pass --use-datasets-lib.

Output directory structure
--------------------------
<output_dir>/diffusionforensics/
    natural_images/
        train/0_real/   diff_<id>.png
        test/0_real/    diff_<id>.png
    <arch>/             e.g. ddpm/, sdv1/, midjourney/ …
        train/1_fake/   diff_<id>.png
        test/1_fake/    diff_<id>.png
    metadata.csv   ← columns: filepath,split,label,dataset,architecture

Architecture labels used
------------------------
Real : natural_images
Fake : ddpm, iddpm, adm, pndm, ldm, sdv1, sdv1_new1, sdv1_new2,
       sdv2, dalle2, midjourney, if, vqdiffusion
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Configure ──────────────────────────────────────────────────────────────
HF_DATASET_ID = "nebula/DiffusionForensics"  # VERIFY before running

ARCH_MAP = {
    "real": "natural_images", "natural": "natural_images",
    "natural_images": "natural_images", "lsun": "natural_images",
    "imagenet": "natural_images",
    "ddpm": "ddpm",   "iddpm": "iddpm",  "adm": "adm",
    "pndm": "pndm",   "ldm": "ldm",
    "stable-diffusion-v1": "sdv1", "sdv1": "sdv1", "sd_v1": "sdv1",
    "sdv1_new1": "sdv1_new1", "sdv1-new1": "sdv1_new1",
    "sdv1_new2": "sdv1_new2", "sdv1-new2": "sdv1_new2",
    "stable-diffusion-v2": "sdv2", "sdv2": "sdv2", "sd_v2": "sdv2",
    "dalle2": "dalle2", "dalle-2": "dalle2", "dall-e-2": "dalle2",
    "midjourney": "midjourney",
    "if": "if", "deepfloyd-if": "if", "deepfloyd_if": "if",
    "vqdiffusion": "vqdiffusion", "vq-diffusion": "vqdiffusion",
}

# Architectures that belong to the training split (seen models)
TRAIN_ARCHS = {"ddpm", "iddpm", "ldm", "pndm", "sdv1", "sdv1_new1", "natural_images"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_WORKERS  = 32
DEFAULT_MAX_TRAIN = 10_000  # max training images kept (balanced real/fake)
DEFAULT_MAX_TEST  = 2_000   # max test images kept (balanced real/fake)
# ───────────────────────────────────────────────────────────────────────────


def download_from_hf_snapshot(output_dir: Path, hf_token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: run  pip install huggingface_hub")

    print(f"Downloading {HF_DATASET_ID} from HuggingFace (snapshot) …")
    cache_dir = output_dir / "diffusionforensics" / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(cache_dir),
        token=hf_token,
        ignore_patterns=["*.git*", "*.md"],
    ))


# ── Scan (file-based) ──────────────────────────────────────────────────────

def _scan_file(img_path: Path, source_root: Path) -> dict | None:
    """Parse one image path into a record (runs in thread pool)."""
    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    parts = [p.lower() for p in img_path.relative_to(source_root).parts]

    split = next((s for s in ("train", "test", "val") if s in parts), None)
    if split == "val":
        split = "test"

    if "0_real" in parts or "real" in parts:
        label = 0
    elif "1_fake" in parts or "fake" in parts:
        label = 1
    else:
        label = None

    arch = None
    for part in parts:
        if part in ARCH_MAP:
            arch = ARCH_MAP[part]
            break
    if arch is None:
        return None

    if label is None:
        label = 0 if arch == "natural_images" else 1

    # Assign split from architecture if not detected from path
    if split is None:
        split = "train" if arch in TRAIN_ARCHS else "test"

    return {
        "_source_path": img_path,
        "split":        split,
        "label":        label,
        "architecture": arch,
        "suffix":       img_path.suffix.lower(),
    }


def collect_images(source_root: Path, workers: int) -> list[dict]:
    all_paths = [p for p in source_root.rglob("*") if p.is_file()]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_scan_file, p, source_root): p for p in all_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning files"):
            result = fut.result()
            if result is not None:
                records.append(result)
    return records


def sample_records(records: list[dict], max_train: int, max_test: int) -> list[dict]:
    """Randomly sample records, keeping at most max_train/2 per class for train
    and max_test/2 per class for test (balanced real/fake)."""
    buckets: dict[tuple, list] = defaultdict(list)
    for rec in records:
        buckets[(rec["split"], rec["label"])].append(rec)
    sampled: list[dict] = []
    for (split, label), recs in buckets.items():
        limit = (max_train if split == "train" else max_test) // 2
        sampled.extend(random.sample(recs, min(len(recs), limit)))
    return sampled


# ── Organise (file copy, threaded) ─────────────────────────────────────────

def _assign_output_paths(records: list[dict]) -> list[dict]:
    """Pre-assign output path for every record (single-threaded)."""
    label_dir = {0: "0_real", 1: "1_fake"}
    counters: dict[tuple, int] = defaultdict(int)

    for rec in records:
        key = (rec["split"], rec["architecture"], rec["label"])
        idx = counters[key]
        counters[key] += 1
        arch  = rec["architecture"]
        split = rec["split"]
        lbl   = rec["label"]
        rec["_rel_path"] = (
            Path("diffusionforensics") / arch / split / label_dir[lbl]
            / f"diff_{idx}{rec['suffix']}"
        )
    return records


def _copy_one(rec: dict, out_root: Path) -> dict:
    """Copy one image to its destination (runs in thread pool)."""
    dest = out_root / rec["_rel_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(rec["_source_path"], dest)
    return {
        "filepath":     str(rec["_rel_path"]).replace("\\", "/"),
        "split":        rec["split"],
        "label":        rec["label"],
        "dataset":      "DiffusionForensics",
        "architecture": rec["architecture"],
    }


def organize_images(records: list[dict], out_root: Path, workers: int) -> list[dict]:
    records = _assign_output_paths(records)
    metadata = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_copy_one, rec, out_root): i
                   for i, rec in enumerate(records)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Copying images"):
            metadata[futures[fut]] = fut.result()
    return [m for m in metadata if m is not None]


# ── datasets-library mode (PIL images from HF parquet) ────────────────────

def _save_one_pil(args: tuple) -> dict:
    """Save a PIL image to disk (runs in thread pool)."""
    img, rel_path, out_root, split, label, arch = args
    dest = out_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        img.convert("RGB").save(dest)
    return {
        "filepath":     str(rel_path).replace("\\", "/"),
        "split":        split,
        "label":        label,
        "dataset":      "DiffusionForensics",
        "architecture": arch,
    }


def download_and_save_via_datasets(
    out_root: Path, hf_token: str | None, workers: int,
    max_train: int = DEFAULT_MAX_TRAIN, max_test: int = DEFAULT_MAX_TEST,
) -> list[dict]:
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        sys.exit("ERROR: run  pip install datasets")

    cache_dir = out_root / "diffusionforensics" / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_dir = {0: "0_real", 1: "1_fake"}

    try:
        configs = get_dataset_config_names(HF_DATASET_ID, token=hf_token)
        print(f"Dataset configs found: {configs}")
    except Exception:
        configs = ["default"]

    # Collect all (image, metadata) tasks first
    tasks: list[tuple] = []
    counters: dict[tuple, int] = defaultdict(int)
    global_counts: dict[tuple, int] = defaultdict(int)   # (split, label) → kept
    limits = {"train": max_train // 2, "test": max_test // 2}

    for config in configs:
        print(f"Loading config: {config} …")
        try:
            ds = load_dataset(
                HF_DATASET_ID, config,
                token=hf_token, cache_dir=str(cache_dir),
            )
        except Exception as e:
            print(f"  Skipping '{config}': {e}")
            continue

        arch = ARCH_MAP.get(config.lower(), config.lower())

        for split_name, split_ds in ds.items():
            split = "train" if split_name.lower() == "train" else "test"

            for example in tqdm(split_ds, desc=f"  {config}/{split_name}"):
                img = (example.get("image")
                       or example.get("img")
                       or example.get("pixel_values"))
                if img is None:
                    continue

                label_raw = example.get("label", example.get("cls", -1))
                label = (int(label_raw) if isinstance(label_raw, (int, float))
                         else (0 if str(label_raw).lower() in ("real", "0") else 1))
                eff_arch = "natural_images" if label == 0 else arch

                gkey = (split, label)
                if global_counts[gkey] >= limits[split]:
                    continue
                global_counts[gkey] += 1

                key = (split, eff_arch, label)
                idx = counters[key]
                counters[key] += 1

                rel_path = (
                    Path("diffusionforensics") / eff_arch / split / label_dir[label]
                    / f"diff_{idx}.png"
                )
                tasks.append((img, rel_path, out_root, split, label, eff_arch))

    metadata = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_save_one_pil, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Saving images"):
            metadata[futures[fut]] = fut.result()

    return [m for m in metadata if m is not None]


# ── Output ─────────────────────────────────────────────────────────────────

def build_metadata_csv(metadata: list[dict], out_root: Path) -> Path:
    df = pd.DataFrame(metadata)
    out_csv = out_root / "diffusionforensics" / "metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def print_summary(df: pd.DataFrame) -> None:
    print("\n── DiffusionForensics summary ────────────────────────────────")
    print(df.groupby(["split", "architecture", "label"])
            .size().reset_index(name="count").to_string(index=False))
    print(f"\nTotal images : {len(df):,}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global HF_DATASET_ID

    parser = argparse.ArgumentParser(
        description="Download & preprocess DiffusionForensics (DIRE) dataset"
    )
    parser.add_argument("--output-dir", "-o", default="/workspace/data/processed",
                        help="Root output directory (default: /workspace/data/processed)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token (for gated datasets)")
    parser.add_argument("--source-dir", default=None,
                        help="Path to already-downloaded folder (skips HF download)")
    parser.add_argument("--hf-dataset-id", default=HF_DATASET_ID,
                        help=f"HuggingFace dataset ID (default: {HF_DATASET_ID})")
    parser.add_argument("--use-datasets-lib", action="store_true",
                        help="Use 'datasets' library instead of snapshot_download "
                             "(needed for parquet-based HF repos)")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel worker threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN,
                        help=f"Max training images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TRAIN})")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST,
                        help=f"Max test images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TEST})")
    args = parser.parse_args()

    HF_DATASET_ID = args.hf_dataset_id
    out_root = Path(args.output_dir)
    print(f"Max train : {args.max_train}  |  Max test : {args.max_test}")

    print(f"Workers : {args.workers}")

    if args.source_dir:
        source_root = Path(args.source_dir)
        print(f"Local source : {source_root}")
        records  = collect_images(source_root, args.workers)
        print(f"Images found : {len(records):,}")
        if not records:
            sys.exit("ERROR: no images found.")
        records  = sample_records(records, args.max_train, args.max_test)
        print(f"Images sampled : {len(records):,}  "
              f"(≤{args.max_train} train / ≤{args.max_test} test)")
        metadata = organize_images(records, out_root, args.workers)

    elif args.use_datasets_lib:
        metadata = download_and_save_via_datasets(
            out_root, args.hf_token, args.workers, args.max_train, args.max_test)
        if not metadata:
            sys.exit("ERROR: no images downloaded.")

    else:
        source_root = download_from_hf_snapshot(out_root, args.hf_token)
        print(f"Downloaded to : {source_root}")
        records  = collect_images(source_root, args.workers)
        print(f"Images found  : {len(records):,}")
        if not records:
            sys.exit("ERROR: no images found.")
        records  = sample_records(records, args.max_train, args.max_test)
        print(f"Images sampled  : {len(records):,}  "
              f"(≤{args.max_train} train / ≤{args.max_test} test)")
        metadata = organize_images(records, out_root, args.workers)

    csv_path = build_metadata_csv(metadata, out_root)
    print(f"\nMetadata saved to : {csv_path}")
    print_summary(pd.read_csv(csv_path))


if __name__ == "__main__":
    main()
