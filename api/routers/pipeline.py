"""
Pipeline router — trigger and monitor pipeline execution steps.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas import PipelineRunRequest, PipelineStatusResponse, PipelineStepStatus
from src.dataset_loader import load_dataset
from src.eda import run_eda
from src.evaluation import run_evaluation
from src.features import build_features
from src.interpretability import run_interpretability
import src.models as _m
from src.models import train_and_compare
from src.preprocessing import preprocess_dataset
from src.results import ensure_results, get_pipeline_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

_executor = ThreadPoolExecutor(max_workers=1)
_running = False


def _run_full_pipeline():
    global _running
    _running = True
    try:
        ensure_results()

        dataset = load_dataset()
        processed = preprocess_dataset(dataset)
        run_eda(processed)
        features_bundle = build_features(processed)

        best_rep, best_model, all_results = train_and_compare(
            features_bundle, processed
        )

        run_evaluation(all_results, features_bundle, processed, best_rep, best_model)

        run_interpretability(all_results, features_bundle, processed)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        _running = False


@router.post("/run")
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Trigger the full NLP pipeline in the background."""
    global _running
    if _running:
        raise HTTPException(status_code=409, detail="Pipeline is already running.")

    background_tasks.add_task(
        _run_full_pipeline,
    )
    return {"message": "Pipeline started.", "running": True}


@router.get("/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    """Return the current pipeline execution state."""
    state = get_pipeline_state()

    all_steps = [
        "load_dataset",
        "preprocess",
        "eda",
        "build_features",
        "train_models",
        "evaluation",
        "interpretability",
    ]
    completed = [s for s in all_steps if state.get(s, {}).get("status") == "done"]
    pending = [s for s in all_steps if s not in completed]

    return PipelineStatusResponse(
        steps={k: PipelineStepStatus(**v) for k, v in state.items()},
        completed_steps=completed,
        pending_steps=pending,
    )


@router.get("/running")
async def is_running():
    return {"running": _running}
