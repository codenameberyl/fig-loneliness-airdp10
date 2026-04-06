"""
FIG-Loneliness NLP API

Endpoints
─────────
GET  /api/status
POST /api/pipeline/run
GET  /api/pipeline/status
GET  /api/pipeline/running

GET  /api/eda/summary
GET  /api/eda/dataset
GET  /api/eda/class_distribution
GET  /api/eda/length_stats
GET  /api/eda/linguistic_stats
GET  /api/eda/ngrams
GET  /api/eda/pos_distribution
GET  /api/eda/plots
GET  /api/eda/plots/{plot_name}

GET  /api/models/results
GET  /api/models/best_per_representation
GET  /api/models/test_report
GET  /api/models/features_metadata
GET  /api/models/plots
GET  /api/models/plots/{plot_name}

POST /api/predict
POST /api/predict/batch
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import eda, models, pipeline, predict
from src.results import ensure_results, get_pipeline_state
from src.config import API_PREFIX, API_TITLE, API_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "REST API for the FIG-Loneliness NLP pipeline. "
        "Exposes dataset statistics, EDA results, model comparison metrics, "
        "evaluation plots, and real-time inference."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────

app.include_router(pipeline.router, prefix=API_PREFIX)
app.include_router(eda.router, prefix=API_PREFIX)
app.include_router(models.router, prefix=API_PREFIX)
app.include_router(predict.router, prefix=API_PREFIX)


# ──────────────────────────────────────────────
# Root / status
# ──────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    ensure_results()
    logger.info(f"{API_TITLE} v{API_VERSION} started.")


@app.get(f"{API_PREFIX}/status")
async def status():
    return {
        "status": "ok",
        "title": API_TITLE,
        "version": API_VERSION,
        "pipeline_state": get_pipeline_state(),
    }


@app.get("/")
async def root():
    return {
        "message": f"Welcome to the {API_TITLE}",
        "docs": "/docs",
        "status": f"{API_PREFIX}/status",
    }
