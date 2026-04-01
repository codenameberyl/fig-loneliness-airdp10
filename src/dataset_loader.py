

import logging
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from huggingface_hub import snapshot_download

from .config import DATASET_DIR, DATASET_REPO, DATASET_SPLITS
from .results import save_json, record_step

logger = logging.getLogger(__name__)



def load_dataset():
    
    repo_id = "FIG-Loneliness/FIG-Loneliness"
    
    # Use huggingface_hub to download 
    logger.info(f"Downloading dataset from {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(DATASET_DIR),
            allow_patterns=["*.arrow", "*.json", "*.jsonl"],
        )
        logger.info(f"Dataset downloaded to {DATASET_DIR}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    # Validate that the expected split directories exist
    missing = [
        folder
        for folder in DATASET_SPLITS.values()
        if not (DATASET_DIR / folder).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Expected dataset splits not found under {DATASET_DIR}: {missing}"
        )

    # Load all splits from disk
    splits = {}
    for split_name, folder in DATASET_SPLITS.items():
        split_path = DATASET_DIR / folder
        logger.info(f"Loading split '{split_name}' from {split_path}...")
        try:
            splits[split_name] = load_from_disk(str(split_path))
            logger.info(
                f"  → {len(splits[split_name]):,} samples, "
                f"columns: {splits[split_name].column_names}"
            )
        except Exception as e:
            logger.error(f"Failed to load split '{split_name}': {e}")
            raise

    dataset = DatasetDict(splits)

    # Save summary for API / EDA
    summary = {
        "splits": {
            name: {
                "n_samples": len(ds),
                "columns": ds.column_names,
                "n_lonely": sum(v[1] for v in ds["lonely"]),
                "n_non_lonely": sum(v[0] for v in ds["lonely"]),
            }
            for name, ds in dataset.items()
        },
        "total_samples": sum(len(ds) for ds in dataset.values()),
        "label_map": {"0": "non_lonely", "1": "lonely"},
    }
    save_json("dataset_summary.json", summary)
    record_step("load_dataset", meta={"total_samples": summary["total_samples"]})

    logger.info(
        f"Dataset loaded — "
        f"train: {len(dataset['train']):,}, "
        f"val: {len(dataset['validation']):,}, "
        f"test: {len(dataset['test']):,}"
    )
    return dataset
