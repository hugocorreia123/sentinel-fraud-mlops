"""Load the registered champion from MLflow at service startup.

Pulls the latest version of `sentinel-champion` and the tuned threshold from
the artifacts directory. Holds them in process memory; no per-request loading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "champion" / "artifacts" / "threshold" / "threshold_sweep.json"
)

MLFLOW_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentinel-champion"


@dataclass
class LoadedModel:
    booster: lgb.Booster
    name: str
    version: str
    stage: str
    feature_names: list[str]
    n_features: int
    threshold: float
    loaded_at: str  # ISO timestamp


def load_champion() -> LoadedModel:
    """Pull the latest registered champion + its tuned threshold."""
    mlflow.set_tracking_uri(MLFLOW_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
    logger.info(f"Loading champion from {model_uri}")
    booster: lgb.Booster = mlflow.lightgbm.load_model(model_uri)

    # Get registered model metadata for version + stage
    client = mlflow.MlflowClient()
    latest_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not latest_versions:
        raise RuntimeError(f"No versions found for model '{REGISTERED_MODEL_NAME}'")
    # Pick the highest version number
    latest = max(latest_versions, key=lambda v: int(v.version))

    # Load tuned threshold; default to 0.5 if missing
    threshold = 0.5
    if THRESHOLD_PATH.exists():
        data = json.loads(THRESHOLD_PATH.read_text())
        threshold = float(data["best"]["threshold"])
        logger.info(f"Loaded cost-aware threshold: {threshold:.4f}")
    else:
        logger.warning(f"Threshold file not found at {THRESHOLD_PATH}; defaulting to 0.5")

    feature_names = list(booster.feature_name())

    return LoadedModel(
        booster=booster,
        name=REGISTERED_MODEL_NAME,
        version=str(latest.version),
        stage=latest.current_stage or "None",
        feature_names=feature_names,
        n_features=len(feature_names),
        threshold=threshold,
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )