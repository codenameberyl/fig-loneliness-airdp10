"""
Predict router — single-text inference endpoint.
"""

import logging

from fastapi import APIRouter, HTTPException

from ..schemas import PredictRequest, PredictResponse
from src.inference import get_predictor, InvalidInputError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify a text as expressing loneliness self-disclosure or not.
 
    Returns label (0/1), label_name, confidence, the threshold applied,
    and the representation/model used.
 
    Raises 400 for invalid/empty input, 503 if the pipeline has not been run.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(request.text)
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e) + " Run the pipeline first via POST /api/pipeline/run.",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
 
    return PredictResponse(**result, input_text=request.text)


@router.post("/batch")
async def predict_batch(texts: list[str]):
    """
    Classify a list of texts.
    
    Returns a list of prediction dicts. Invalid individual texts return an
    error entry rather than failing the whole batch.
    """
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch.")
    try:
        predictor = get_predictor()
        results = [predictor.predict(t) for t in texts]
        return results
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
