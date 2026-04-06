"""
Pydantic schemas for the FIG-Loneliness API.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    force_reprocess: bool = Field(False, description="Re-run even if cached results exist")
    eval_on_test: bool = Field(True, description="Evaluate best model on test split")
    skip_bert: bool = Field(False, description="Skip DistilBERT fine-tuning")


class PipelineStepStatus(BaseModel):
    status: str
    timestamp: Optional[str] = None


class PipelineStatusResponse(BaseModel):
    steps: dict[str, PipelineStepStatus]
    completed_steps: list[str]
    pending_steps: list[str]


# ──────────────────────────────────────────────
# Dataset / EDA
# ──────────────────────────────────────────────

class SplitStats(BaseModel):
    """Flexible schema — accepts both n_* keys (from dataset_loader JSON) and plain keys."""
    total: int = None
    lonely: int = None
    non_lonely: int = None
    lonely_pct: Optional[float] = None
    columns: list


class DatasetSummaryResponse(BaseModel):
    splits: Optional[dict[str, SplitStats]] = None
    total_samples: Optional[int] = None
    label_map: Optional[dict[str, str]] = None


class DescriptiveStats(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float


class FeatureStats(BaseModel):
    non_lonely: DescriptiveStats
    lonely: DescriptiveStats


class LengthStatsResponse(BaseModel):
    word_count: FeatureStats
    char_count: FeatureStats
    sentence_count: FeatureStats


class NGramEntry(BaseModel):
    term: str
    count: int


class NGramsResponse(BaseModel):
    non_lonely_unigrams: list[NGramEntry]
    lonely_unigrams: list[NGramEntry]
    non_lonely_bigrams: list[NGramEntry]
    lonely_bigrams: list[NGramEntry]


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class ModelResult(BaseModel):
    representation: str
    model: str
    split: str = "validation"
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    # Optional test metrics
    test_accuracy: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    test_roc_auc: Optional[float] = None


class ModelResultsResponse(BaseModel):
    results: list[ModelResult]
    best_representation: str
    best_model: str
    best_f1: float


class RepresentationSummary(BaseModel):
    representation: str
    best_model: str
    f1: float
    roc_auc: Optional[float] = None
    accuracy: Optional[float] = None
    n_features: Optional[int] = None


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to classify")


class PredictResponse(BaseModel):
    label: int = Field(..., description="0 = non-lonely, 1 = lonely")
    label_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    representation: str
    model: str
    input_text: str


# ──────────────────────────────────────────────
# General
# ──────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    version: str
    pipeline_state: dict[str, Any]


class PlotListResponse(BaseModel):
    plots: list[str]


class ErrorResponse(BaseModel):
    detail: str
