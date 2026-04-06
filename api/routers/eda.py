"""
EDA router — serve exploratory analysis results and plot images.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..schemas import (
    DatasetSummaryResponse,
    FeatureStats,
    LengthStatsResponse,
    NGramsResponse,
    PlotListResponse,
)
from src.results import (
    get_pipeline_state,
    json_exists,
    list_plots,
    load_json,
    plot_path,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eda", tags=["EDA"])


def _require(name: str):
    if not json_exists(name):
        raise HTTPException(
            status_code=404,
            detail=f"Result '{name}' not found. Run the pipeline first.",
        )
    return load_json(name)


@router.get("/summary")
async def get_eda_summary():
    """Return the combined EDA summary."""
    return _require("eda_summary.json")


@router.get("/dataset", response_model=DatasetSummaryResponse)
async def get_dataset_summary():
    """Return dataset split sizes and class balance."""
    data = _require("dataset_summary.json")
    # Pass raw dict — SplitStats accepts both n_* and plain field names
    return DatasetSummaryResponse(**data)


@router.get("/class_distribution")
async def get_class_distribution():
    return _require("eda_class_distribution.json")


@router.get("/length_stats")
async def get_length_stats():
    raw = _require("eda_length_stats.json")
    # Convert nested dicts to schema objects
    result = {}
    for feat, values in raw.items():
        result[feat] = FeatureStats(**values)
    return result


@router.get("/linguistic_stats")
async def get_linguistic_stats():
    return _require("eda_linguistic_stats.json")


@router.get("/ngrams")
async def get_ngrams():
    return _require("eda_ngrams.json")


@router.get("/pos_distribution")
async def get_pos_distribution():
    return _require("eda_pos_distribution.json")


@router.get("/preprocessing/samples")
async def get_preprocessing_samples(
    split: str = "train",
    label: str = "both",
    n: int = 10,
):
    """
    Return sample rows from the preprocessed dataset.

    Parameters
    ----------
    split : "train" | "validation" | "test"
    label : "both" | "lonely" | "non_lonely"
    n     : max samples per class to return (capped at 50)
    """
    data = _require("preprocessing_samples.json")

    if split not in data:
        raise HTTPException(
            status_code=404,
            detail=f"Split '{split}' not found. Available: {list(data.keys())}",
        )

    n = min(n, 50)
    split_data = data[split]

    if label == "lonely":
        return {"split": split, "samples": split_data.get("lonely", [])[:n]}
    elif label == "non_lonely":
        return {"split": split, "samples": split_data.get("non_lonely", [])[:n]}
    else:
        return {
            "split": split,
            "lonely": split_data.get("lonely", [])[:n],
            "non_lonely": split_data.get("non_lonely", [])[:n],
        }


@router.get("/preprocessing/summary")
async def get_preprocessing_summary():
    """Return preprocessing metadata — feature columns, split sizes, step timestamp."""
    state = get_pipeline_state()
    preprocess_meta = state.get("preprocess", {})
    return {
        "status": preprocess_meta.get("status"),
        "timestamp": preprocess_meta.get("timestamp"),
        "feature_columns": preprocess_meta.get("feature_columns", []),
        "train_size": preprocess_meta.get("train_size"),
        "val_size": preprocess_meta.get("val_size"),
        "test_size": preprocess_meta.get("test_size"),
    }


@router.get("/plots", response_model=PlotListResponse)
async def list_eda_plots():
    """List all available EDA plot filenames."""
    all_plots = list_plots()
    eda_plots = [p for p in all_plots if p.startswith("eda_")]
    return PlotListResponse(plots=eda_plots)


@router.get("/plots/{plot_name}")
async def get_plot(plot_name: str):
    """Serve a plot image by filename."""
    p = plot_path(plot_name)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found.")
    return FileResponse(str(p), media_type="image/png")
