"""Load the registered champion from MLflow at service startup.

Pulls the latest version of `sentinel-champion` and the tuned threshold from
the artifacts directory. Holds them in process memory; no per-request loading.
"""
from __future__ import annotations

import tempfile

from mlflow.artifacts import download_artifacts

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
    logger.info(f"Downloading champion artifacts from {model_uri}")
    local_dir = Path(download_artifacts(artifact_uri=model_uri,
                                          dst_path=tempfile.mkdtemp()))
    files = list(local_dir.rglob("*"))
    logger.info(f"Downloaded {len(files)} files to {local_dir}")
    candidates = [p for p in files if p.suffix in (".lgb", ".txt") and "model" in p.stem]
    if not candidates:
        raise FileNotFoundError(
            f"No LightGBM model file found in {local_dir}. Files: {files}"
        )
    model_file = candidates[0]
    logger.info(f"Loading booster from {model_file}")
    booster: lgb.Booster = lgb.Booster(
        model_file=str(model_file),
        params={"num_threads": 1, "predict_disable_shape_check": True},
    )

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

import torch
from models.challenger.model import FraudMLP


@dataclass
class LoadedChallenger:
    """The PyTorch challenger plus its scaler + feature order."""
    model: FraudMLP
    name: str
    version: str
    feature_names: list[str]  # ordered numeric feature names (no one-hot type cols)
    scaler_mean: torch.Tensor
    scaler_std: torch.Tensor
    device: torch.device
    loaded_at: str


def load_challenger() -> LoadedChallenger:
    """Load the challenger from the saved torch artifact + scaler."""
    artifact_path = PROJECT_ROOT / "models" / "challenger" / "artifacts" / "model.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Challenger artifact not found at {artifact_path}. "
            "Run `uv run python -m models.challenger.train` first."
        )

    # Load to CPU at inference — MPS for batch training, CPU is fine for one tx
    device = torch.device("cpu")
    blob = torch.load(artifact_path, map_location=device, weights_only=False)

    numeric_cols: list[str] = blob["numeric_cols"]
    model = FraudMLP(n_numeric_features=len(numeric_cols)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    # Get registered version from MLflow
    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='sentinel-challenger'")
    latest = max(versions, key=lambda v: int(v.version)) if versions else None

    return LoadedChallenger(
        model=model,
        name="sentinel-challenger",
        version=str(latest.version) if latest else "local",
        feature_names=numeric_cols,
        scaler_mean=torch.from_numpy(blob["scaler_mean"]).to(device),
        scaler_std=torch.from_numpy(blob["scaler_std"]).to(device),
        device=device,
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )