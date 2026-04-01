import logging
import time

from datasets import load_from_disk

from src.results import cache_exists, load_joblib
from src.config import RESULTS_SUBDIRS
from src.dataset_loader import load_dataset
from src.eda import run_eda
from src.preprocessing import preprocess_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PREPROCESS_CACHE = RESULTS_SUBDIRS["cache"] / "preprocessed_dataset"


def _banner(stage: str) -> None:
    width = 60
    logger.info("─" * width)
    logger.info(f"  STAGE: {stage.upper()}")
    logger.info("─" * width)


def main():
    logger.info("=" * 60)
    logger.info("  FIG-Loneliness NLP Pipeline")
    logger.info("=" * 60)

    t0 = time.time()

    dataset = None
    processed = None
    features_bundle = None
    all_results = None
    best_rep = None
    best_model = None

    #  Load
    _banner("load")
    dataset = load_dataset()

    #  Preprocess
    _banner("preprocess")
    processed = preprocess_dataset(dataset)

    #  EDA
    _banner("eda")
    if processed is None:
        if PREPROCESS_CACHE.exists():
            processed = load_from_disk(PREPROCESS_CACHE)
        else:
            processed = preprocess_dataset(load_dataset())
    run_eda(processed)

   

    # Done
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete in {elapsed:.1f}s")
    if best_rep and best_model:
        logger.info(f"  Best model : {best_rep} / {best_model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
