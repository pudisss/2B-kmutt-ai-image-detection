#!/usr/bin/env python3
"""Generate train and test manifest CSV files from per-dataset metadata.

This script implements the MANIFEST MODE of the training pipeline.

Two pipeline modes
------------------
  Auto mode     Training reads each dataset's metadata CSV directly and
                applies ModelAwareSplitter on-the-fly (controlled by
                dataset_configs + train_models + test_models in config.yaml).

  Manifest mode This script is run ONCE to combine all metadata CSVs into
                a fixed  train_manifest.csv  and  test_manifest.csv.
                Training then uses those frozen files for full reproducibility.

Usage
-----
  # Generate with defaults (reads configs/default.yaml for split assignments)
  python scripts/generate_manifests.py --data-dir /workspace/data/processed

  # Override output location
  python scripts/generate_manifests.py \\
      --data-dir /workspace/data/processed \\
      --output-dir src/data

  # Limit samples per class (for quick experiments)
  python scripts/generate_manifests.py \\
      --data-dir /workspace/data/processed \\
      --samples-per-class 5000

  # Add a dataset that isn't in the default config
  python scripts/generate_manifests.py \\
      --data-dir /workspace/data/processed \\
      --extra-metadata cifake/cifake_metadata.csv \\
      --extra-metadata synthbuster/synthbuster_metadata.csv

Expected metadata CSV columns (produced by download_*.py scripts)
------------------------------------------------------------------
  filepath, split, label, dataset, architecture
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Project root (this script lives in scripts/, project root is one level up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG    = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT    = PROJECT_ROOT / "src" / "data"
DEFAULT_WORKERS   = 32
DEFAULT_VAL_SPLIT = 0.1
SEED              = 42

# ── Architecture → split assignment ───────────────────────────────────────
# Architectures not listed here will be assigned to TRAIN by default.

# Fake architectures used only for testing (unseen during training)
DEFAULT_TEST_FAKE_ARCHS = {
    # CNNDetection – unseen GANs
    "stylegan2", "seeingdark", "san", "crn", "imle", "sitd",
    # DiffusionForensics – unseen diffusion models
    "sdv2", "sdv1_new2", "adm", "dalle2", "midjourney", "if", "vqdiffusion",
    # SynthBuster – all unseen (evaluation-only dataset)
    "raise_real", "firefly", "sd_14", "sd_2", "sd_xl", "wukong",
}

# Fake architectures used for training (seen during training)
DEFAULT_TRAIN_FAKE_ARCHS = {
    # CNNDetection – seen GANs
    "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "deepfake",
    # DiffusionForensics – seen diffusion models
    "ddpm", "iddpm", "ldm", "pndm", "sdv1", "sdv1_new1",
    # CIFAKE – used for training (small, 32×32 synthetic)
    "stable_diffusion_v14",
}

# Real architectures — split 80 % train / 20 % test
REAL_ARCHS = {"natural", "natural_images", "cifar_real", "raise_real"}
# ───────────────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Load individual metadata CSVs in parallel ─────────────────────────────

def _load_one_csv(args: tuple) -> pd.DataFrame | None:
    """Load and validate a single metadata CSV (runs in thread pool)."""
    csv_path, data_root = args
    full_path = data_root / csv_path
    if not full_path.exists():
        print(f"  [WARN] Not found: {full_path}")
        return None
    df = pd.read_csv(full_path)
    # Drop unnamed index columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    required = {"filepath", "label", "architecture"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] {csv_path} missing columns: {missing} — skipping")
        return None
    if "dataset" not in df.columns:
        df["dataset"] = full_path.parent.name
    if "split" not in df.columns:
        df["split"] = "unknown"
    # Normalise
    df["architecture"] = df["architecture"].astype(str).str.lower().str.strip()
    df["label"]        = df["label"].astype(int)
    df["full_csv_path"] = str(full_path)
    print(f"  Loaded {len(df):>7,} rows  ← {full_path.relative_to(data_root)}")
    return df


def load_all_metadata(metadata_paths: list[str], data_root: Path, workers: int) -> pd.DataFrame:
    print(f"Loading {len(metadata_paths)} metadata file(s) with {workers} workers …")
    args_list = [(p, data_root) for p in metadata_paths]
    dfs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_load_one_csv, a): a for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Reading CSVs"):
            df = fut.result()
            if df is not None:
                dfs.append(df)
    if not dfs:
        sys.exit("ERROR: No valid metadata files loaded.")
    combined = pd.concat(dfs, ignore_index=True)
    combined.drop(columns=["full_csv_path"], inplace=True, errors="ignore")
    return combined


# ── Split assignment ───────────────────────────────────────────────────────

def assign_splits(
    df: pd.DataFrame,
    train_fake: set[str],
    test_fake: set[str],
    real_archs: set[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each row to train or test and return (train_df, test_df)."""

    def _assign(row) -> str:
        arch  = row["architecture"]
        label = row["label"]

        if label == 0:
            return "real_pool"   # real images will be split 80/20 below

        if arch in test_fake:
            return "test"
        if arch in train_fake:
            return "train"
        # Default: assign unknown fake archs to train with a warning
        return "train_unknown"

    df = df.copy()
    df["_split_assign"] = df.apply(_assign, axis=1)

    unknown = df[df["_split_assign"] == "train_unknown"]["architecture"].unique()
    if len(unknown):
        print(f"  [WARN] Unknown fake architectures assigned to TRAIN: {list(unknown)}")
    df.loc[df["_split_assign"] == "train_unknown", "_split_assign"] = "train"

    # Split real images 80 / 20
    real_pool  = df[df["_split_assign"] == "real_pool"].copy()
    real_train, real_test = train_test_split(
        real_pool, test_size=0.2, random_state=seed, shuffle=True
    )
    df.loc[real_train.index, "_split_assign"] = "train"
    df.loc[real_test.index,  "_split_assign"] = "test"

    train_df = df[df["_split_assign"] == "train"].drop(columns=["_split_assign"])
    test_df  = df[df["_split_assign"] == "test"].drop(columns=["_split_assign"])

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ── Optional balancing ─────────────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    """Down-sample each class to samples_per_class (balanced per architecture for fakes)."""
    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    # Sample real
    if len(real_df) > samples_per_class:
        real_df = real_df.sample(n=samples_per_class, random_state=seed)

    # Sample fake — balanced across architectures
    archs   = fake_df["architecture"].unique()
    per_arch = samples_per_class // len(archs)
    remainder = samples_per_class % len(archs)
    parts = []
    for i, arch in enumerate(archs):
        arch_df = fake_df[fake_df["architecture"] == arch]
        n = per_arch + (1 if i < remainder else 0)
        if len(arch_df) > n:
            arch_df = arch_df.sample(n=n, random_state=seed + i)
        parts.append(arch_df)
    fake_df = pd.concat(parts, ignore_index=True)

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Validate integrity ────────────────────────────────────────────────────

def validate_manifests(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_fake_archs = set(train_df[train_df["label"] == 1]["architecture"].unique())
    test_fake_archs  = set(test_df[test_df["label"] == 1]["architecture"].unique())
    overlap = train_fake_archs & test_fake_archs
    if overlap:
        print(f"  [WARN] Fake architecture overlap between train and test: {overlap}")
        print("         This is expected if the same arch has different splits.")
    else:
        print("  [OK] No fake architecture overlap between train and test.")


# ── Save manifests in parallel ────────────────────────────────────────────

def _write_csv(args: tuple) -> None:
    df, path = args
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_manifests(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    workers: int,
) -> tuple[Path, Path]:
    train_path = output_dir / "train_manifest.csv"
    test_path  = output_dir / "test_manifest.csv"

    with ThreadPoolExecutor(max_workers=min(2, workers)) as ex:
        futs = [
            ex.submit(_write_csv, (train_df, train_path)),
            ex.submit(_write_csv, (test_df,  test_path)),
        ]
        for fut in futs:
            fut.result()

    return train_path, test_path


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\n── {name} manifest ──────────────────────────────────────")
        print(df.groupby(["dataset", "architecture", "label"])
                .size().reset_index(name="count").to_string(index=False))
        real = (df["label"] == 0).sum()
        fake = (df["label"] == 1).sum()
        print(f"  Total: {len(df):,}  (real: {real:,}, fake: {fake:,})")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/test manifest CSVs from per-dataset metadata"
    )
    parser.add_argument("--data-dir", "-d", required=True,
                        help="Root directory containing per-dataset folders "
                             "(e.g. /workspace/data/processed)")
    parser.add_argument("--output-dir", "-o", default=str(DEFAULT_OUTPUT),
                        help=f"Where to write manifest CSVs (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG),
                        help=f"Path to config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--extra-metadata", action="append", default=[],
                        metavar="REL_PATH",
                        help="Additional metadata CSV relative to --data-dir "
                             "(can be repeated, e.g. --extra-metadata cifake/cifake_metadata.csv)")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Cap each class (real/fake) to this many samples in TRAIN "
                             "(default: use all)")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT,
                        help=f"Fraction of train set to reserve for validation "
                             f"(default: {DEFAULT_VAL_SPLIT}; informational only — "
                             "MultiDomainDataModule handles this at runtime)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel worker threads (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    data_root  = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Data root  : {data_root}")
    print(f"Output dir : {output_dir}")
    print(f"Workers    : {args.workers}")

    # Collect metadata file paths from config
    metadata_paths: list[str] = []
    if Path(args.config).exists():
        cfg = load_config(Path(args.config))
        for ds in cfg.get("data", {}).get("datasets", []):
            if "metadata" in ds:
                metadata_paths.append(ds["metadata"])
    else:
        print(f"[WARN] Config not found at {args.config} — using defaults only")
        # Defaults for all four datasets
        metadata_paths = [
            "cnndetection/cnndetection_metadata.csv",
            "diffusionforensics/metadata.csv",
            "cifake/cifake_metadata.csv",
            "synthbuster/synthbuster_metadata.csv",
        ]

    # Add any extra paths
    for extra in args.extra_metadata:
        if extra not in metadata_paths:
            metadata_paths.append(extra)

    # Load and combine
    combined_df = load_all_metadata(metadata_paths, data_root, args.workers)
    print(f"\nTotal rows loaded : {len(combined_df):,}")

    # Assign train / test splits
    print("\nAssigning train/test splits …")
    train_df, test_df = assign_splits(
        combined_df,
        train_fake=DEFAULT_TRAIN_FAKE_ARCHS,
        test_fake=DEFAULT_TEST_FAKE_ARCHS,
        real_archs=REAL_ARCHS,
        seed=args.seed,
    )

    # Optional class balancing for train
    if args.samples_per_class:
        print(f"Balancing train to {args.samples_per_class:,} samples/class …")
        train_df = balance_classes(train_df, args.samples_per_class, args.seed)

    # Validate
    print("Validating manifest integrity …")
    validate_manifests(train_df, test_df)

    # Save
    print("Saving manifests …")
    train_path, test_path = save_manifests(train_df, test_df, output_dir, args.workers)

    print(f"\nTrain manifest : {train_path}  ({len(train_df):,} rows)")
    print(f"Test  manifest : {test_path}  ({len(test_df):,} rows)")

    print_summary(train_df, test_df)

    print(
        f"\nDone. To use these manifests in training:\n"
        f"  python scripts/train.py --use-manifests "
        f"--train-manifest {train_path} --test-manifest {test_path} "
        f"--data-dir {data_root}"
    )


if __name__ == "__main__":
    main()
