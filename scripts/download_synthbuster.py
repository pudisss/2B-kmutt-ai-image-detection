#!/usr/bin/env python3
"""Download and preprocess SynthBuster dataset.

Dataset: Bammey 2023 "SynthBuster: Towards Detection of Diffusion Model Deep Fakes"
Source:  HuggingFace — set HF_DATASET_ID below to the correct repo ID.

Common HuggingFace locations (verify before running):
  - "nebula/synthbuster"
  Paper: https://arxiv.org/abs/2308.01441

Dataset overview
----------------
  Real  : ~1000 photos from RAISE-1k dataset
  Fake  : ~9000 photorealistic images from 9 generators
          Adobe Firefly, DALL-E 2, Midjourney 5.1,
          SD 1.4, SD 2.1, SD XL, Stable Diffusion (unspecified), Wukong
  All images are considered "test" (evaluation-only dataset).

Output directory structure
--------------------------
<output_dir>/synthbuster/
    raise_real/
        test/0_real/   raise_real_0_real_<id>.png
    <arch>/             e.g. dalle2/, midjourney/, firefly/ …
        test/1_fake/   <arch>_1_fake_<id>.png
    synthbuster_metadata.csv   ← columns: filepath,split,label,dataset,architecture

Architecture labels used
------------------------
Real : raise_real
Fake : firefly, dalle2, midjourney, sd_14, sd_2, sd_xl, wukong
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
HF_DATASET_ID   = "nebula/synthbuster"   # VERIFY before running
DEFAULT_WORKERS = 32

# Map from HF config/folder names → canonical architecture names
ARCH_MAP = {
    # real
    "real": "raise_real", "raise": "raise_real", "raise_1k": "raise_real",
    "raise1k": "raise_real", "raise-1k": "raise_real",
    # fake
    "adobe-firefly": "firefly", "firefly": "firefly",
    "dalle-2": "dalle2", "dalle2": "dalle2", "dall-e-2": "dalle2",
    "midjourney": "midjourney", "midjourney-5.1": "midjourney",
    "stable-diffusion-1.4": "sd_14", "sd_1.4": "sd_14",
    "stable-diffusion-1": "sd_14",   "sd_14": "sd_14",
    "stable-diffusion-2": "sd_2",    "sd_2.1": "sd_2",
    "stable-diffusion-2.1": "sd_2",  "sd_2": "sd_2",
    "stable-diffusion-xl": "sd_xl",  "sdxl": "sd_xl",
    "sd_xl": "sd_xl", "stable-diffusion-xl-base-1.0": "sd_xl",
    "wukong": "wukong",
    # generic fallbacks
    "stable-diffusion": "sd_14",
}

# All SynthBuster images are test-only (no training split)
DEFAULT_SPLIT     = "test"
DEFAULT_MAX_TEST  = 2_000   # max test images kept (balanced real/fake)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# ───────────────────────────────────────────────────────────────────────────


def download_from_hf_snapshot(output_dir: Path, hf_token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: run  pip install huggingface_hub")

    print(f"Downloading {HF_DATASET_ID} from HuggingFace (snapshot) …")
    cache_dir = output_dir / "synthbuster" / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(cache_dir),
        token=hf_token,
        ignore_patterns=["*.git*", "*.md", "*.txt"],
    ))


# ── Scan local directory ───────────────────────────────────────────────────

def _scan_file(img_path: Path, source_root: Path) -> dict | None:
    """Parse one image path into a record (runs in thread pool)."""
    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    parts = [p.lower() for p in img_path.relative_to(source_root).parts]

    # Detect label from folder names
    if any(p in ("0_real", "real") for p in parts):
        label = 0
    elif any(p in ("1_fake", "fake") for p in parts):
        label = 1
    else:
        label = None

    # Detect architecture
    arch = None
    for part in reversed(parts):            # deepest folder wins
        if part in ARCH_MAP:
            arch = ARCH_MAP[part]
            break

    # If no explicit label dir, infer from architecture
    if arch is None:
        return None
    if label is None:
        label = 0 if arch == "raise_real" else 1

    return {
        "_source_path": img_path,
        "_pil_image":   None,
        "split":        DEFAULT_SPLIT,
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
            r = fut.result()
            if r is not None:
                records.append(r)
    return records


def sample_records(records: list[dict], max_test: int) -> list[dict]:
    """Randomly sample to at most max_test/2 per class (balanced real/fake).
    SynthBuster is test-only so no train split is considered."""
    buckets: dict[int, list] = defaultdict(list)
    for rec in records:
        buckets[rec["label"]].append(rec)
    sampled: list[dict] = []
    for label, recs in buckets.items():
        limit = max_test // 2
        sampled.extend(random.sample(recs, min(len(recs), limit)))
    return sampled


# ── Assign output paths (single-threaded) ─────────────────────────────────

def assign_output_paths(records: list[dict]) -> list[dict]:
    label_dir = {0: "0_real", 1: "1_fake"}
    type_str  = {0: "real",   1: "fake"}
    counters: dict[tuple, int] = defaultdict(int)

    for rec in records:
        arch  = rec["architecture"]
        lbl   = rec["label"]
        split = rec["split"]
        key   = (arch, lbl)
        idx   = counters[key]
        counters[key] += 1
        ext = rec.get("suffix", ".png")
        rec["_rel_path"] = (
            Path("synthbuster") / arch / split / label_dir[lbl]
            / f"{arch}_{lbl}_{type_str[lbl]}_{idx}{ext}"
        )
    return records


# ── Copy worker ───────────────────────────────────────────────────────────

def _copy_one(rec: dict, out_root: Path) -> dict:
    dest = out_root / rec["_rel_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(rec["_source_path"], dest)
    return {
        "filepath":     str(rec["_rel_path"]).replace("\\", "/"),
        "split":        rec["split"],
        "label":        rec["label"],
        "dataset":      "SynthBuster",
        "architecture": rec["architecture"],
    }


def organize_images(records: list[dict], out_root: Path, workers: int) -> list[dict]:
    records = assign_output_paths(records)
    metadata = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_copy_one, rec, out_root): i
                   for i, rec in enumerate(records)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Copying images"):
            metadata[futures[fut]] = fut.result()
    return [m for m in metadata if m is not None]


# ── datasets-library mode ─────────────────────────────────────────────────

def _save_pil(args: tuple) -> dict:
    """Save one PIL image (runs in thread pool)."""
    pil_img, rel_path, out_root, split, label, arch = args
    dest = out_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        pil_img.convert("RGB").save(dest)
    return {
        "filepath":     str(rel_path).replace("\\", "/"),
        "split":        split,
        "label":        label,
        "dataset":      "SynthBuster",
        "architecture": arch,
    }


def download_and_save_via_datasets(
    out_root: Path, hf_token: str | None, workers: int, max_test: int = DEFAULT_MAX_TEST
) -> list[dict]:
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        sys.exit("ERROR: run  pip install datasets")

    cache_dir = out_root / "synthbuster" / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_dir = {0: "0_real", 1: "1_fake"}
    type_str  = {0: "real",   1: "fake"}

    try:
        configs = get_dataset_config_names(HF_DATASET_ID, token=hf_token)
        print(f"Dataset configs found: {configs}")
    except Exception:
        configs = ["default"]

    counters: dict[tuple, int] = defaultdict(int)
    global_counts: dict[int, int] = defaultdict(int)   # label → kept so far
    per_class_limit = max_test // 2
    tasks = []

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
            for example in tqdm(split_ds, desc=f"  {config}/{split_name}"):
                img = (example.get("image")
                       or example.get("img")
                       or example.get("pixel_values"))
                if img is None:
                    continue

                label_raw = example.get("label", example.get("cls", -1))
                label = (int(label_raw) if isinstance(label_raw, (int, float))
                         else (0 if str(label_raw).lower() in ("real", "0") else 1))

                eff_arch = "raise_real" if label == 0 else arch
                if global_counts[label] >= per_class_limit:
                    continue
                global_counts[label] += 1
                key = (eff_arch, label)
                idx = counters[key]
                counters[key] += 1

                rel_path = (
                    Path("synthbuster") / eff_arch / DEFAULT_SPLIT / label_dir[label]
                    / f"{eff_arch}_{label}_{type_str[label]}_{idx}.png"
                )
                tasks.append((img, rel_path, out_root, DEFAULT_SPLIT, label, eff_arch))

    metadata = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_save_pil, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Saving images"):
            metadata[futures[fut]] = fut.result()

    return [m for m in metadata if m is not None]


# ── Output ─────────────────────────────────────────────────────────────────

def build_metadata_csv(metadata: list[dict], out_root: Path) -> Path:
    df = pd.DataFrame(metadata)
    out_csv = out_root / "synthbuster" / "synthbuster_metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def print_summary(df: pd.DataFrame) -> None:
    print("\n── SynthBuster summary ───────────────────────────────────────")
    print(df.groupby(["split", "architecture", "label"])
            .size().reset_index(name="count").to_string(index=False))
    print(f"\nTotal images : {len(df):,}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global HF_DATASET_ID

    parser = argparse.ArgumentParser(
        description="Download & preprocess SynthBuster dataset"
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
                        help="Use 'datasets' library instead of snapshot_download")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel worker threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST,
                        help=f"Max test images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TEST})")
    args = parser.parse_args()

    HF_DATASET_ID = args.hf_dataset_id
    out_root = Path(args.output_dir)
    print(f"Max test  : {args.max_test}")

    print(f"Workers : {args.workers}")

    if args.source_dir:
        source_root = Path(args.source_dir)
        print(f"Local source : {source_root}")
        records  = collect_images(source_root, args.workers)
        print(f"Images found : {len(records):,}")
        if not records:
            sys.exit("ERROR: no images found.")
        records  = sample_records(records, args.max_test)
        print(f"Images sampled : {len(records):,}  (≤{args.max_test} test)")
        metadata = organize_images(records, out_root, args.workers)

    elif args.use_datasets_lib:
        metadata = download_and_save_via_datasets(
            out_root, args.hf_token, args.workers, args.max_test)
        if not metadata:
            sys.exit("ERROR: no images downloaded.")

    else:
        source_root = download_from_hf_snapshot(out_root, args.hf_token)
        print(f"Downloaded to : {source_root}")
        records  = collect_images(source_root, args.workers)
        print(f"Images found  : {len(records):,}")
        if not records:
            sys.exit("ERROR: no images found.")
        records  = sample_records(records, args.max_test)
        print(f"Images sampled  : {len(records):,}  (≤{args.max_test} test)")
        metadata = organize_images(records, out_root, args.workers)

    csv_path = build_metadata_csv(metadata, out_root)
    print(f"\nMetadata saved to : {csv_path}")
    print_summary(pd.read_csv(csv_path))


if __name__ == "__main__":
    main()
