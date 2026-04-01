

import json
import math
import joblib
import logging
import numpy as np
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

from .config import RESULTS_SUBDIRS

logger = logging.getLogger(__name__)


# Initialisation
def ensure_results() -> None:
    #Create all results subdirectories if they don't exist
    for name, path in RESULTS_SUBDIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured results dir: {path}")


# JSON helpers
def save_json(name: str, data: Any) -> Path:
    #Persist *data* as pretty-printed JSON under results/json/
    ensure_results()
    path = RESULTS_SUBDIRS["json"] / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_serialise)
    logger.info(f"Saved JSON → {path}")
    return path


def load_json(name: str) -> Any:
    #Load a JSON result by filename
    path = RESULTS_SUBDIRS["json"] / name
    if not path.exists():
        raise FileNotFoundError(f"JSON result not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_exists(name: str) -> bool:
    return (RESULTS_SUBDIRS["json"] / name).exists()


# Joblib (binary cache) helpers
def save_joblib(name: str, obj: Any) -> Path:
    #Persist a Python object via joblib under results/cache/
    ensure_results()
    path = RESULTS_SUBDIRS["cache"] / name
    joblib.dump(obj, path)
    logger.info(f"Cached object → {path}")
    return path


def load_joblib(name: str) -> Any:
    #Load a joblib result by filename.
    path = RESULTS_SUBDIRS["cache"] / name
    if not path.exists():
        raise FileNotFoundError(f"Joblib result not found: {path}")
    return joblib.load(path)


def cache_exists(name: str) -> bool:
    return (RESULTS_SUBDIRS["cache"] / name).exists()


# Plot helpers
def plot_path(name: str) -> Path:
    #Return the full path for a named plot file (does NOT create file)
    ensure_results()
    return RESULTS_SUBDIRS["plots"] / name


def list_plots() -> list[str]:
    #Return filenames of all saved plots
    d = RESULTS_SUBDIRS["plots"]
    if not d.exists():
        return []
    return [p.name for p in sorted(d.glob("*.png")) + sorted(d.glob("*.svg"))]


# Pipeline state tracking
_STATE_FILE = "pipeline_state.json"


def record_step(step: str, status: str = "done", meta: Optional[dict] = None) -> None:
    #Record that a pipeline step completed (or failed)
    state = load_json(_STATE_FILE) if json_exists(_STATE_FILE) else {}
    state[step] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(meta or {}),
    }
    save_json(_STATE_FILE, state)


def get_pipeline_state() -> dict:
    return load_json(_STATE_FILE) if json_exists(_STATE_FILE) else {}


def step_done(step: str) -> bool:
    state = get_pipeline_state()
    return state.get(step, {}).get("status") == "done"


# Internal helpers
def _sanitise_float(v: float) -> Any:
    #Return None for non-finite floats so JSON stays compliant."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _json_serialise(obj: Any) -> Any:
    #Fallback JSON serialiser — handles numpy types and non-finite floats
    if isinstance(obj, float):
        return _sanitise_float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return _sanitise_float(v)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serialise(v) for v in obj]
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def sanitise_for_json(obj: Any) -> Any:

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise_for_json(v) for v in obj]
    return obj
