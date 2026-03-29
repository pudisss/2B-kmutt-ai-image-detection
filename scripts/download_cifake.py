#!/usr/bin/env python3
"""Download and preprocess CIFAKE dataset from Kaggle.

Dataset: Bird & Lotfi 2023 "CIFAKE: Real and AI-Generated Synthetic Images"
Source:  Kaggle — https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Setup (one-time)
----------------
1. Install the Kaggle CLI:
       pip install kaggle

2. Create a Kaggle API token at https://www.kaggle.com/settings → API → Create New Token
   This downloads  kaggle.json.  Place it at:
       ~/.kaggle/kaggle.json          (Linux/Mac)
       %USERPROFILE%\\.kaggle\\kaggle.json  (Windows)
   Then:
       chmod 600 ~/.kaggle/kaggle.json

3. Alternatively, pass credentials via environment variables:
       export KAGGLE_USERNAME=your_username
       export KAGGLE_KEY=your_api_key

Dataset overview
----------------
  60,000 REAL images (CIFAR-10)     + 60,000 FAKE (Stable Diffusion v1.4)
  Train split : 50,000 real + 50,000 fake
  Test  split : 10,000 real + 10,000 fake
  Resolution  : 32 × 32  (use --save-size 256 to upsample on save)

Kaggle structure (inside the downloaded zip)
--------------------------------------------
  train/REAL/<id>.jpg
  train/FAKE/<id>.jpg
  test/REAL/<id>.jpg
  test/FAKE/<id>.jpg

Output directory structure
--------------------------
<output_dir>/cifake/
    train/
        0_real/   cifar_real_0_real_<id>.jpg
        1_fake/   stable_diffusion_v14_1_fake_<id>.jpg
    test/
        0_real/   cifar_real_0_real_<id>.jpg
        1_fake/   stable_diffusion_v14_1_fake_<id>.jpg
    cifake_metadata.csv   ← columns: filepath,split,label,dataset,architecture

Architecture labels used
------------------------
Real : cifar_real
Fake : stable_diffusion_v14
"""

import argparse
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Configure ──────────────────────────────────────────────────────────────
KAGGLE_DATASET    = "birdy654/cifake-real-and-ai-generated-synthetic-images"
DEFAULT_WORKERS   = 32
DEFAULT_SAVE_SIZE = 32    # native resolution; set to e.g. 256 to upsample
DEFAULT_MAX_TRAIN = 10_000  # max training images kept (balanced real/fake)
DEFAULT_MAX_TEST  = 2_000   # max test images kept (balanced real/fake)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_TO_ARCH = {0: "cifar_real", 1: "stable_diffusion_v14"}
LABEL_DIR     = {0: "0_real",     1: "1_fake"}
TYPE_STR      = {0: "real",       1: "fake"}
# ───────────────────────────────────────────────────────────────────────────


# ── Download from Kaggle ───────────────────────────────────────────────────

def download_from_kaggle(output_dir: Path, kaggle_dataset: str) -> Path:
    """Download and extract the CIFAKE zip via the Kaggle API."""
    try:
        import kaggle  # noqa: F401 – triggers credential check on import
    except ImportError:
        sys.exit("ERROR: run  pip install kaggle")
    except OSError as e:
        sys.exit(
            f"ERROR: Kaggle credentials not found.\n{e}\n\n"
            "Place kaggle.json at ~/.kaggle/kaggle.json  or set "
            "KAGGLE_USERNAME / KAGGLE_KEY environment variables."
        )

    zip_dir = output_dir / "cifake" / "_kaggle_download"
    zip_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {kaggle_dataset} from Kaggle …")
    import kaggle as kg
    kg.api.dataset_download_files(
        kaggle_dataset,
        path=str(zip_dir),
        unzip=False,
        quiet=False,
    )

    # Find the downloaded zip
    zips = list(zip_dir.glob("*.zip"))
    if not zips:
        sys.exit("ERROR: no zip file found after Kaggle download.")
    zip_path = zips[0]

    extract_dir = zip_dir / "extracted"
    if not extract_dir.exists():
        print(f"Extracting {zip_path.name} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    return extract_dir


# ── Scan source directory ─────────────────────────────────────────────────

def _scan_file(img_path: Path, source_root: Path) -> dict | None:
    """Parse one image path into a metadata record (runs in thread pool)."""
    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    # Kaggle structure:  {split}/REAL/<file>  or  {split}/FAKE/<file>
    # Also handle flat:  REAL/<file>  /  FAKE/<file>
    parts = [p.lower() for p in img_path.relative_to(source_root).parts]

    split = next((s for s in ("train", "test", "val") if s in parts), None)
    if split == "val":
        split = "test"
    if split is None:
        split = "test"   # default for unknown structure

    if "real" in parts or "0_real" in parts:
        label = 0
    elif "fake" in parts or "1_fake" in parts:
        label = 1
    else:
        return None   # can't determine class — skip

    return {
        "_source_path": img_path,
        "split":        split,
        "label":        label,
        "suffix":       img_path.suffix.lower(),
    }


def collect_images(source_root: Path, workers: int) -> list[dict]:
    """Scan source directory in parallel."""
    all_paths = [p for p in source_root.rglob("*") if p.is_file()]
    records   = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_scan_file, p, source_root): p for p in all_paths}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Scanning files"):
            r = fut.result()
            if r is not None:
                records.append(r)
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


# ── Assign output paths (single-threaded, race-condition free) ─────────────

def assign_output_paths(records: list[dict]) -> list[dict]:
    counters: dict[tuple, int] = defaultdict(int)
    for rec in records:
        lbl   = rec["label"]
        split = rec["split"]
        arch  = LABEL_TO_ARCH[lbl]
        key   = (split, lbl)
        idx   = counters[key]
        counters[key] += 1
        ext = rec["suffix"]
        rec["_rel_path"] = (
            Path("cifake") / split / LABEL_DIR[lbl]
            / f"{arch}_{lbl}_{TYPE_STR[lbl]}_{idx}{ext}"
        )
        rec["_arch"] = arch
    return records


# ── Copy worker (threaded) ─────────────────────────────────────────────────

def _copy_one(args: tuple) -> dict:
    """Copy + optionally resize one image (runs in thread pool)."""
    src, rel_path, out_root, split, label, arch, save_size = args
    dest = out_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        if save_size and save_size != 32:
            # Resize on the fly using PIL
            from PIL import Image
            img = Image.open(src).convert("RGB")
            img = img.resize((save_size, save_size), Image.BILINEAR)
            img.save(dest)
        else:
            shutil.copy2(src, dest)
    return {
        "filepath":     str(rel_path).replace("\\", "/"),
        "split":        split,
        "label":        label,
        "dataset":      "CIFAKE",
        "architecture": arch,
    }


def copy_images(records: list[dict], out_root: Path,
                workers: int, save_size: int) -> list[dict]:
    """Copy all images in parallel, return metadata rows."""
    tasks = [
        (rec["_source_path"], rec["_rel_path"], out_root,
         rec["split"], rec["label"], rec["_arch"], save_size)
        for rec in records
    ]
    metadata = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_copy_one, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Copying images"):
            metadata[futures[fut]] = fut.result()
    return [m for m in metadata if m is not None]


# ── Output ─────────────────────────────────────────────────────────────────

def build_metadata_csv(metadata: list[dict], out_root: Path) -> Path:
    df = pd.DataFrame(metadata)
    out_csv = out_root / "cifake" / "cifake_metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def print_summary(df: pd.DataFrame) -> None:
    print("\n── CIFAKE summary ────────────────────────────────────────────")
    print(df.groupby(["split", "architecture", "label"])
            .size().reset_index(name="count").to_string(index=False))
    print(f"\nTotal images : {len(df):,}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & preprocess CIFAKE dataset from Kaggle"
    )
    parser.add_argument("--output-dir", "-o", default="/workspace/data/processed",
                        help="Root output directory (default: /workspace/data/processed)")
    parser.add_argument("--source-dir", default=None,
                        help="Path to already-extracted CIFAKE folder (skips Kaggle download)")
    parser.add_argument("--kaggle-dataset", default=KAGGLE_DATASET,
                        help=f"Kaggle dataset slug (default: {KAGGLE_DATASET})")
    parser.add_argument("--save-size", type=int, default=DEFAULT_SAVE_SIZE,
                        help=f"Save images at this px resolution "
                             f"(default: {DEFAULT_SAVE_SIZE}; use 256 to upsample)")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel worker threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN,
                        help=f"Max training images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TRAIN})")
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST,
                        help=f"Max test images to keep, balanced real/fake "
                             f"(default: {DEFAULT_MAX_TEST})")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    print(f"Workers   : {args.workers}")
    print(f"Save size : {args.save_size}px")
    print(f"Max train : {args.max_train}  |  Max test : {args.max_test}")

    if args.source_dir:
        source_root = Path(args.source_dir)
        print(f"Local source : {source_root}")
    else:
        source_root = download_from_kaggle(out_root, args.kaggle_dataset)
        print(f"Extracted to : {source_root}")

    records = collect_images(source_root, args.workers)
    print(f"Images found   : {len(records):,}")
    if not records:
        sys.exit("ERROR: no images found. Check the source directory structure.")

    records = sample_records(records, args.max_train, args.max_test)
    print(f"Images sampled : {len(records):,}  "
          f"(≤{args.max_train} train / ≤{args.max_test} test)")

    records  = assign_output_paths(records)
    metadata = copy_images(records, out_root, args.workers, args.save_size)

    csv_path = build_metadata_csv(metadata, out_root)
    print(f"\nMetadata saved to : {csv_path}")
    print_summary(pd.read_csv(csv_path))


if __name__ == "__main__":
    main()
