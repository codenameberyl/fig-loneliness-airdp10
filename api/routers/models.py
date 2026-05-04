"""
Models router — serve training results, model comparison, and evaluation reports.
"""
 
import logging
 
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
 
from ..schemas import ModelResult, ModelResultsResponse, PlotListResponse
from src.results import (
    json_exists,
    list_plots,
    load_json,
    plot_path,
    sanitise_for_json,
    RESULTS_SUBDIRS,
)
 
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])
 
 
def _require(name: str):
    if not json_exists(name):
        raise HTTPException(
            status_code=404,
            detail=f"Result '{name}' not found. Run the pipeline first.",
        )
    return load_json(name)
 
 
@router.get("/results", response_model=ModelResultsResponse)
async def get_model_results():
    """Return validation metrics for all representation × model combinations."""
    results_raw = _require("all_model_results.json")
    results = [ModelResult(**r) for r in results_raw]
 
    if not results:
        raise HTTPException(status_code=404, detail="No model results found.")
 
    best = max(results, key=lambda r: r.f1 or 0)
    return ModelResultsResponse(
        results=results,
        best_representation=best.representation,
        best_model=best.model,
        best_f1=best.f1 or 0.0,
    )
 
 
@router.get("/best_per_representation")
async def get_best_per_representation():
    return _require("best_per_representation.json")
 
 
@router.get("/test_report")
async def get_test_report():
    return sanitise_for_json(_require("test_evaluation_report.json"))
 
 
@router.get("/features_metadata")
async def get_features_metadata():
    return _require("features_metadata.json")
 
 
@router.get("/confusion_matrix/{model_key}")
async def get_confusion_matrix(model_key: str):
    return _require(f"eval_confusion_{model_key}.json")
 
 
@router.get("/roc_curve/{model_key}")
async def get_roc_curve(model_key: str):
    return sanitise_for_json(_require(f"eval_roc_{model_key}.json"))
 
 
@router.get("/full_report/{model_key}")
async def get_full_report(model_key: str):
    return sanitise_for_json(_require(f"eval_full_{model_key}.json"))


# Interpretability
@router.get("/interpretability/summary")
async def get_interpretability_summary():
    """Return cross-representation interpretability summary (RQ4)."""
    return _require("interpretability_summary.json")
 
 
@router.get("/interpretability/coefficients/{representation}")
async def get_lr_coefficients(representation: str):
    """Top-50 LR coefficients for linguistic_only, tfidf, or tfidf_ling."""
    return sanitise_for_json(
        _require(f"interpretability_{representation}_lr_coefficients.json")
    )
 
 
@router.get("/interpretability/attention")
async def get_distilbert_attention():
    """DistilBERT average last-layer attention weights per class."""
    return sanitise_for_json(_require("interpretability_distilbert_attention.json"))


# Error analysis
@router.get("/error_analysis/summary")
async def get_error_analysis_summary():
    """
    Return the aggregated error analysis summary — FP/FN counts per model,
    confusion overlap across models, and qualitative linguistic patterns.
    """
    return sanitise_for_json(_require("error_analysis_summary.json"))
 
 
@router.get("/error_analysis/{model_key}")
async def get_error_analysis_for_model(model_key: str):
    """
    Return detailed FP/FN examples for a specific model.
 
    model_key format: {representation}_{model}
    e.g. tfidf_logistic_regression, sbert_random_forest
    """
    return sanitise_for_json(_require(f"error_analysis_{model_key}.json"))
 
 
@router.get("/error_analysis_keys")
async def list_error_analysis_keys():
    """List all model keys that have a saved error analysis JSON."""
    json_dir = RESULTS_SUBDIRS["json"]
    keys = [
        p.stem.replace("error_analysis_", "")
        for p in json_dir.glob("error_analysis_*.json")
        if p.stem != "error_analysis_summary"
    ]
    return {"keys": sorted(keys)}


# Utility
@router.get("/confusion_matrix_keys")
async def list_confusion_matrix_keys():
    json_dir = RESULTS_SUBDIRS["json"]
    keys = [
        p.stem.replace("eval_confusion_", "")
        for p in json_dir.glob("eval_confusion_*.json")
    ]
    return {"keys": sorted(keys)}
 
 
@router.get("/roc_curve_keys")
async def list_roc_curve_keys():
    json_dir = RESULTS_SUBDIRS["json"]
    keys = [p.stem.replace("eval_roc_", "") for p in json_dir.glob("eval_roc_*.json")]
    return {"keys": sorted(keys)}
 
 
@router.get("/plots", response_model=PlotListResponse)
async def list_eval_plots():
    all_plots = list_plots()
    eval_plots = [p for p in all_plots if p.startswith("eval_") or p.startswith("error_")]
    return PlotListResponse(plots=eval_plots)


@router.get("/plots/{plot_name}")
async def get_plot(plot_name: str):
    p = plot_path(plot_name)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found.")
    return FileResponse(str(p), media_type="image/png")
