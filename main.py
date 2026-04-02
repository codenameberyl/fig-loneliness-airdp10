import logging
import time

from datasets import load_from_disk

from src.results import cache_exists, load_joblib
from src.config import RESULTS_SUBDIRS
from src.dataset_loader import load_dataset
from src.eda import run_eda
import run_evaluation
from src.features import build_features
from src.models import train_and_compare
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

    # Features
    _banner("features")
    if processed is None:
        if PREPROCESS_CACHE.exists():
            processed = load_from_disk(str(PREPROCESS_CACHE))
        else:
            raise RuntimeError("Preprocessed dataset not found.")
    features_bundle = build_features(processed)

    # Train
    _banner("train")
    if features_bundle is None:
        if cache_exists("features_all_representations.joblib"):
            features_bundle = load_joblib("features_all_representations.joblib")
        else:
            raise RuntimeError("Feature bundle not found.")
    if processed is None:
        processed = load_from_disk(str(PREPROCESS_CACHE))

    best_rep, best_model, all_results = train_and_compare(
        features_bundle, processed
    )

    # Evaluate
    _banner("evaluate")
    if all_results is None:
        if cache_exists("all_model_results.joblib"):
            all_results = load_joblib("all_model_results.joblib")
            best = max(all_results, key=lambda r: r.get("f1", 0))
            best_rep, best_model = best["representation"], best["model"]
        else:
            raise RuntimeError("Model results not found.")
    if features_bundle is None:
        features_bundle = load_joblib("features_all_representations.joblib")
    if processed is None:
        processed = load_from_disk(str(PREPROCESS_CACHE))

    run_evaluation(all_results, features_bundle, processed, best_rep, best_model)


    # Done
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete in {elapsed:.1f}s")
    if best_rep and best_model:
        logger.info(f"  Best model : {best_rep} / {best_model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
