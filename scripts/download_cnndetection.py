#!/usr/bin/env python3
"""Download and preprocess CNNDetection dataset.

Dataset: Wang et al. 2020 "CNN-generated images are surprisingly easy to spot... for now"
Source:  HuggingFace — sywang/CNNDetection (official repo by paper author Sheng-Yu Wang)
  https://huggingface.co/datasets/sywang/CNNDetection

  Fallback: download original from Google Drive links in the GitHub README:
    https://github.com/peterwang512/CNNDetection
  then point --source-dir at the extracted folder.

Output directory structure
--------------------------
<output_dir>/cnndetection/
    train/
        0_real/   natural_0_real_<id>.<ext>
        1_fake/   <arch>_1_fake_<id>.<ext>
    test/
        0_real/   natural_0_real_<id>.<ext>
        1_fake/   <arch>_1_fake_<id>.<ext>
    cnndetection_metadata.csv   ← columns: filepath,split,label,dataset,architecture

Architecture labels used
------------------------
Real : natural
Fake : progan, stylegan, stylegan2, biggan, cyclegan, stargan, gaugan,
       deepfake, seeingdark, san, crn, imle, sitd
"""

import argparse
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Configure ──────────────────────────────────────────────────────────────
HF_DATASET_ID = "sywang/CNNDetection"  # official author repo (Sheng-Yu Wang / PeterWang512)

ARCH_MAP = {
    "0_real": "natural", "real": "natural",
    "1_fake": "progan",  "progan": "progan",
    "stylegan": "stylegan",  "stylegan2": "stylegan2",
    "biggan": "biggan",      "cyclegan": "cyclegan",
    "stargan": "stargan",    "gaugan": "gaugan",
    "deepfake": "deepfake",  "seeingdark": "seeingdark",
    "san": "san",            "crn": "crn",
    "imle": "imle",          "sitd": "sitd",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_WORKERS  = 32
DEFAULT_MAX_TRAIN = 10_000  # max training images kept (balanced real/fake)
DEFAULT_MAX_TEST  = 2_000   # max test images kept (balanced real/fake)
# ───────────────────────────────────────────────────────────────────────────


def _count_images(directory: Path) -> int:
    """Count image files recursively in a directory."""
    return sum(1 for p in directory.rglob("*")
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _extract_zip_with_progress(zp: Path, extract_dir: Path) -> None:
    """Extract a zip file one file at a time with a tqdm progress bar."""
    with zipfile.ZipFile(zp, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=zp.name, unit="file", leave=True):
            zf.extract(member, extract_dir)


def _extract_7z_with_progress(first_part: Path, extract_dir: Path) -> None:
    """Extract a multi-part 7z archive with a tqdm progress bar via a monitor thread."""
    try:
        import py7zr
    except ImportError:
        sys.exit("ERROR: run  pip install py7zr  to extract .7z archives")

    with py7zr.SevenZipFile(first_part, mode="r") as zf:
        total = len(zf.getnames())

    with tqdm(total=total, desc=first_part.name, unit="file", leave=True) as pbar:
        prev = 0

        def _tick() -> None:
            current = sum(1 for p in extract_dir.rglob("*") if p.is_file())
            nonlocal prev
            if current > prev:
                pbar.update(current - prev)
                prev = current

        with py7zr.SevenZipFile(first_part, mode="r") as zf:
            for _ in zf.getnames():          # tick once per file as extraction runs
                _tick()
                time.sleep(0)               # yield to allow extraction progress
            zf.extractall(path=extract_dir)  # actual extraction

        # Final flush
        _tick()


def _extract_archives(cache_dir: Path, max_train: int, max_test: int) -> Path:
    """Extract archives into an 'extracted' subfolder.

    Zip files (test/val sets) are extracted in parallel with progress bars.
    The large multi-part 7z train archive is skipped if the already-extracted
    zips already provide enough images to satisfy max_train + max_test.
    """
    extract_dir = cache_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    # ── zip files (test / val sets — extracted in parallel) ────────────────
    for zp in sorted(cache_dir.glob("*.zip")):
        marker = extract_dir / f".done_{zp.stem}"
        if marker.exists():
            print(f"  Already extracted: {zp.name}")
            continue
        print(f"  Extracting {zp.name} …")
        _extract_zip_with_progress(zp, extract_dir)
        marker.touch()

    # ── check if we already have enough images from the zips ───────────────
    current_count = _count_images(extract_dir)
    needed = max_train + max_test
    if current_count >= needed:
        print(f"  {current_count:,} images already extracted — "
              f"skipping large .7z train archive (need only {needed:,}).")
        return extract_dir

    # ── multi-part 7z train archive (can be 90+ GB) ────────────────────────
    for first_part in sorted(cache_dir.glob("*.7z.001")):
        stem   = first_part.name.replace(".7z.001", "")
        marker = extract_dir / f".done_{stem}_7z"
        if marker.exists():
            print(f"  Already extracted: {first_part.name} (multi-part)")
            continue
        print(f"  Extracting {first_part.name} (multi-part 7z, this may take a while) …")
        _extract_7z_with_progress(first_part, extract_dir)
        marker.touch()

    return extract_dir


def download_from_hf(output_dir: Path, hf_token: str | None,
                     max_train: int = DEFAULT_MAX_TRAIN,
                     max_test: int = DEFAULT_MAX_TEST) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: run  pip install huggingface_hub")

    cache_dir = output_dir / "cnndetection" / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Skip re-download if archives are already present
    if not any(cache_dir.glob("*.zip")) and not any(cache_dir.glob("*.7z.*")):
        print(f"Downloading {HF_DATASET_ID} from HuggingFace …")
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=hf_token,
            ignore_patterns=["*.git*", "*.md", "*.txt", "*.json"],
        )
    else:
        print(f"Archives already present in {cache_dir}, skipping download.")

    print("Extracting archives …")
    return _extract_archives(cache_dir, max_train, max_test)


# ── Scan ───────────────────────────────────────────────────────────────────

def _scan_file(img_path: Path, source_root: Path) -> dict | None:
    """Parse a single image path into a metadata record (runs in thread pool)."""
    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    parts = [p.lower() for p in img_path.relative_to(source_root).parts]

    split = next((s for s in ("train", "test", "val") if s in parts), "test")
    if split == "val":
        split = "test"

    label_folder = next((p for p in parts if p in ("0_real", "1_fake", "real", "fake")), None)
    if label_folder in ("0_real", "real"):
        label = 0
    elif label_folder in ("1_fake", "fake"):
        label = 1
    else:
        label = int(any(p in ARCH_MAP and ARCH_MAP[p] != "natural" for p in parts))

    # Walk parts from deepest to find architecture name
    arch = "unknown"
    for part in reversed(parts):
        if part in ARCH_MAP:
            arch = ARCH_MAP[part]
            break

    if label == 0:
        arch = "natural"

    return {
        "_source_path": img_path,
        "split":        split,
        "label":        label,
        "architecture": arch,
        "suffix":       img_path.suffix.lower(),
    }


def collect_images(source_root: Path, workers: int) -> list[dict]:
    """Scan source_root in parallel and return list of image records."""
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


# ── Organise ───────────────────────────────────────────────────────────────

def _assign_output_paths(records: list[dict]) -> list[dict]:
    """Pre-assign canonical output path to every record (single-threaded, O(n))."""
    label_dir = {0: "0_real", 1: "1_fake"}
    type_str  = {0: "real",   1: "fake"}
    counters: dict[tuple, int] = defaultdict(int)

    for rec in records:
        key = (rec["split"], rec["label"], rec["architecture"])
        idx = counters[key]
        counters[key] += 1

        arch  = rec["architecture"]
        split = rec["split"]
        lbl   = rec["label"]

        filename = f"{arch}_{lbl}_{type_str[lbl]}_{idx}{rec['suffix']}"
        rec["_rel_path"] = Path("cnndetection") / split / label_dir[lbl] / filename

    return records


def _copy_one(rec: dict, out_root: Path) -> dict:
    """Copy a single image to its canonical destination (runs in thread pool)."""
    dest = out_root / rec["_rel_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(rec["_source_path"], dest)
    return {
        "filepath":     str(rec["_rel_path"]).replace("\\", "/"),
        "split":        rec["split"],
        "label":        rec["label"],
        "dataset":      "CNNDetection",
        "architecture": rec["architecture"],
    }


def organize_images(records: list[dict], out_root: Path, workers: int) -> list[dict]:
    """Copy all images in parallel, return metadata rows."""
    records = _assign_output_paths(records)

    metadata = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_copy_one, rec, out_root): i
                   for i, rec in enumerate(records)}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Copying images"):
            idx = futures[fut]
            metadata[idx] = fut.result()

    return [m for m in metadata if m is not None]


# ── Output ─────────────────────────────────────────────────────────────────

def build_metadata_csv(metadata: list[dict], out_root: Path) -> Path:
    df = pd.DataFrame(metadata)
    out_csv = out_root / "cnndetection" / "cnndetection_metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def print_summary(df: pd.DataFrame) -> None:
    print("\n── CNNDetection summary ──────────────────────────────────────")
    print(df.groupby(["split", "architecture", "label"])
            .size().reset_index(name="count").to_string(index=False))
    print(f"\nTotal images : {len(df):,}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    global HF_DATASET_ID

    parser = argparse.ArgumentParser(
        description="Download & preprocess CNNDetection dataset"
    )
    parser.add_argument("--output-dir", "-o", default="/workspace/data/processed",
                        help="Root output directory (default: /workspace/data/processed)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token (for gated datasets)")
    parser.add_argument("--source-dir", default=None,
                        help="Path to already-downloaded folder (skips HF download)")
    parser.add_argument("--hf-dataset-id", default=HF_DATASET_ID,
                        help=f"HuggingFace dataset ID (default: {HF_DATASET_ID})")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel worker threads (default: {DEFAULT_WORKERS})")
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

    if args.source_dir:
        source_root = Path(args.source_dir)
        print(f"Using local source : {source_root}")
    else:
        source_root = download_from_hf(out_root, args.hf_token, args.max_train, args.max_test)
        print(f"Downloaded to      : {source_root}")

    print(f"Workers            : {args.workers}")

    records = collect_images(source_root, args.workers)
    print(f"Images found       : {len(records):,}")
    if not records:
        sys.exit("ERROR: no images found — check the path or dataset structure.")

    records = sample_records(records, args.max_train, args.max_test)
    print(f"Images sampled     : {len(records):,}  "
          f"(≤{args.max_train} train / ≤{args.max_test} test)")

    metadata = organize_images(records, out_root, args.workers)

    csv_path = build_metadata_csv(metadata, out_root)
    print(f"Metadata saved to  : {csv_path}")
    print_summary(pd.read_csv(csv_path))


if __name__ == "__main__":
    main()
