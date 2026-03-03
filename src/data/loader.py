"""
src/data/loader.py
==================
Data loading utilities for the FIG-Loneliness dataset.

This dataset must be cloned from:
https://huggingface.co/datasets/FIG-Loneliness/FIG-Loneliness

If not found locally, it will be cloned automatically.
"""

import subprocess
from pathlib import Path
from ..config import DATASET_REPO_URL, DEFAULT_LOCAL_PATH


def _clone_dataset(destination: Path):
    """Clone dataset repository using git."""
    print("Dataset folder not found.")
    print("Cloning FIG-Loneliness dataset from HuggingFace...")

    try:
        subprocess.run(["git", "clone", DATASET_REPO_URL, str(destination)], check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Git is not installed or not available in PATH.\n"
            "Install Git or manually clone the dataset:\n"
            f"git clone {DATASET_REPO_URL}"
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to clone dataset repository.")

    print("Dataset cloned successfully.")


def load_data():
    """
    Load FIG-Loneliness dataset.

    Workflow:
        1. If dataset folder does not exist → clone repo
        2. Load splits using load_from_disk()
    """

    try:
        from datasets import load_from_disk
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required.\n" "Install with: pip install datasets"
        )

    root = Path(DEFAULT_LOCAL_PATH)

    # ─────────────────────────────────────────────
    # STEP 1: Clone if missing
    # ─────────────────────────────────────────────
    if not root.exists():
        _clone_dataset(root)

    # ─────────────────────────────────────────────
    # STEP 2: Validate structure
    # ─────────────────────────────────────────────
    train_path = root / "train_set"
    dev_path = root / "dev_set"
    test_path = root / "test_set"

    if not (train_path.exists() and dev_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            "Dataset folder exists but does not contain expected subfolders:\n"
            "  - train_set/\n"
            "  - dev_set/\n"
            "  - test_set/\n"
            "Ensure the repository was cloned correctly."
        )

    # ─────────────────────────────────────────────
    # STEP 3: Load splits
    # ─────────────────────────────────────────────
    print("Loading FIG-Loneliness dataset from local repository...")

    dataset = {
        "train": load_from_disk(str(train_path)),
        "validation": load_from_disk(str(dev_path)),
        "test": load_from_disk(str(test_path)),
    }

    print("Dataset loaded successfully.")
    for split_name, split_data in dataset.items():
        print(
            f"  {split_name:<10} : {len(split_data):>6} samples | "
            f"columns: {split_data.column_names}"
        )

    return dataset


def inspect_dataset(dataset) -> None:
    """Print human-readable dataset summary."""
    print("=" * 70)
    print("FIG-Loneliness Dataset Summary")
    print("=" * 70)

    for split_name, split_data in dataset.items():
        print(f"\n── Split: {split_name} ({len(split_data):,} samples) ──")
        print(f"   Columns : {split_data.column_names}")
        print(f"   Features: {split_data.features}")

    if "train" in dataset:
        print("\n── First training example ──")
        example = dataset["train"][0]
        for k, v in example.items():
            snippet = str(v)[:120] + ("…" if len(str(v)) > 120 else "")
            print(f"   {k:25s}: {snippet}")

    print("=" * 70)


def dataset_to_dataframe(dataset) -> dict:
    """
    Convert DatasetDict to dictionary of pandas DataFrames.
    """
    dfs = {}

    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()
        df["split"] = split_name
        dfs[split_name] = df
        print(f"Split {split_name}: {len(df)} rows, {len(df.columns)} columns")

    return dfs
