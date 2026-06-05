"""Load the registered champion from MLflow at service startup.

Pulls the latest version of `sentinel-champion` and the tuned threshold from
the artifacts directory. Holds them in process memory; no per-request loading.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import json
import lightgbm as lgb
from loguru import logger

MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "mlflow")

# Only import MLflow if we actually need it
if MODEL_SOURCE == "mlflow":
    import mlflow
    import mlflow.lightgbm
    from mlflow.artifacts import download_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAMPION_THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "champion" / "artifacts" / "threshold" / "threshold_sweep.json"
)
CHALLENGER_THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "challenger" / "artifacts" / "threshold" / "threshold_sweep.json"
)

MLFLOW_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentinel-champion"

CHAMPION_SNAPSHOT_DIR = PROJECT_ROOT / "models" / "champion" / "snapshot"
CHALLENGER_SNAPSHOT_DIR = PROJECT_ROOT / "models" / "challenger" / "snapshot"


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


def _load_champion_from_disk() -> LoadedModel:
    """Load champion from the local snapshot bundle (no MLflow needed)."""
    booster_path = CHAMPION_SNAPSHOT_DIR / "model.lgb"
    threshold_path = CHAMPION_SNAPSHOT_DIR / "threshold.json"
    info_path = CHAMPION_SNAPSHOT_DIR / "info.json"

    if not booster_path.exists():
        raise FileNotFoundError(
            f"Champion snapshot not found at {booster_path}. "
            "Run `uv run python -m scripts.snapshot_models` first."
        )

    logger.info(f"Loading champion from local snapshot: {booster_path}")
    booster: lgb.Booster = lgb.Booster(
        model_file=str(booster_path),
        params={"num_threads": 1, "predict_disable_shape_check": True},
    )

    threshold = 0.5
    if threshold_path.exists():
        data = json.loads(threshold_path.read_text())
        threshold = float(data["best"]["threshold"])
        logger.info(f"Loaded champion threshold from snapshot: {threshold:.4f}")

    info: dict = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())

    feature_names = list(booster.feature_name())
    return LoadedModel(
        booster=booster,
        name=info.get("name", "sentinel-champion"),
        version=str(info.get("version", "local")),
        stage="local",
        feature_names=feature_names,
        n_features=len(feature_names),
        threshold=threshold,
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

def load_champion() -> LoadedModel:
    """Pull the latest registered champion + its tuned threshold."""
    if MODEL_SOURCE == "local":
        return _load_champion_from_disk()
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
    if CHAMPION_THRESHOLD_PATH.exists():
        data = json.loads(CHAMPION_THRESHOLD_PATH.read_text())
        threshold = float(data["best"]["threshold"])
        logger.info(f"Loaded cost-aware threshold: {threshold:.4f}")
    else:
        logger.warning(f"Threshold file not found at {CHAMPION_THRESHOLD_PATH}; defaulting to 0.5")

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
    feature_names: list[str]
    scaler_mean: torch.Tensor
    scaler_std: torch.Tensor
    device: torch.device
    loaded_at: str
    threshold: float

def _load_challenger_from_disk() -> LoadedChallenger:
    """Load challenger from the local snapshot bundle (no MLflow needed)."""
    pt_path = CHALLENGER_SNAPSHOT_DIR / "model.pt"
    threshold_path = CHALLENGER_SNAPSHOT_DIR / "threshold.json"
    info_path = CHALLENGER_SNAPSHOT_DIR / "info.json"

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Challenger snapshot not found at {pt_path}. "
            "Run `uv run python -m scripts.snapshot_models` first."
        )

    device = torch.device("cpu")
    blob = torch.load(pt_path, map_location=device, weights_only=False)
    numeric_cols: list[str] = blob["numeric_cols"]
    model = FraudMLP(n_numeric_features=len(numeric_cols)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    threshold = 0.5
    if threshold_path.exists():
        data = json.loads(threshold_path.read_text())
        threshold = float(data["best"]["threshold"])
        logger.info(f"Loaded challenger threshold from snapshot: {threshold:.4f}")

    info: dict = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())

    return LoadedChallenger(
        model=model,
        name=info.get("name", "sentinel-challenger"),
        version=str(info.get("version", "local")),
        feature_names=numeric_cols,
        scaler_mean=torch.from_numpy(blob["scaler_mean"]).to(device),
        scaler_std=torch.from_numpy(blob["scaler_std"]).to(device),
        device=device,
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        threshold=threshold,
    )

def load_challenger() -> LoadedChallenger:
    """Load the challenger from the saved torch artifact + scaler."""
    if MODEL_SOURCE == "local":
        return _load_challenger_from_disk()
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

    # Load challenger's tuned threshold; default to 0.5 if not yet tuned
    threshold = 0.5
    if CHALLENGER_THRESHOLD_PATH.exists():
        data = json.loads(CHALLENGER_THRESHOLD_PATH.read_text())
        threshold = float(data["best"]["threshold"])
        logger.info(f"Loaded challenger threshold: {threshold:.4f}")
    else:
        logger.warning(
            f"Challenger threshold file not found at {CHALLENGER_THRESHOLD_PATH}; "
            "defaulting to 0.5"
        )

    return LoadedChallenger(
        model=model,
        name="sentinel-challenger",
        version=str(latest.version) if latest else "local",
        feature_names=numeric_cols,
        scaler_mean=torch.from_numpy(blob["scaler_mean"]).to(device),
        scaler_std=torch.from_numpy(blob["scaler_std"]).to(device),
        device=device,
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        threshold=threshold,
    )