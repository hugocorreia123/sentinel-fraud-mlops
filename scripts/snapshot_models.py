"""Snapshot model artifacts from local disk to a self-contained bundle.

Reads from:
  models/champion/artifacts/model.txt            -> renamed to model.lgb
  models/champion/artifacts/threshold/threshold_sweep.json -> extract best threshold
  models/challenger/artifacts/model.pt           -> as is
  models/challenger/artifacts/threshold/threshold_sweep.json -> extract best threshold

Writes self-contained bundles to:
  models/champion/snapshot/{model.lgb, threshold.json, info.json}
  models/challenger/snapshot/{model.pt, threshold.json, scaler.json, feature_names.json, info.json}

These bundles let the inference service load models without MLflow,
which is required for the HF Spaces deployment.

The challenger scaler + feature_names are embedded INSIDE the .pt file (the
training script saves them as a checkpoint dict). The loader unpacks them
at inference time; nothing extra needed here.

Usage:
    uv run python -m scripts.snapshot_models
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_SRC = PROJECT_ROOT / "models" / "champion" / "artifacts"
CHALLENGER_SRC = PROJECT_ROOT / "models" / "challenger" / "artifacts"
CHAMPION_DEST = PROJECT_ROOT / "models" / "champion" / "snapshot"
CHALLENGER_DEST = PROJECT_ROOT / "models" / "challenger" / "snapshot"


def extract_best_threshold(sweep_path: Path) -> float:
    data = json.loads(sweep_path.read_text())
    return float(data["best"]["threshold"])


def snapshot_champion() -> None:
    CHAMPION_DEST.mkdir(parents=True, exist_ok=True)

    # Booster: LightGBM accepts both .lgb and .txt. We have model.txt.
    booster_src = CHAMPION_SRC / "model.txt"
    if not booster_src.exists():
        # If model.lgb already exists (from earlier snapshot attempt), use it
        alt = CHAMPION_DEST / "model.lgb"
        if not alt.exists():
            raise FileNotFoundError(f"No champion booster at {booster_src}")
    else:
        shutil.copy(booster_src, CHAMPION_DEST / "model.lgb")
        logger.info(f"Copied {booster_src.name} -> {CHAMPION_DEST / 'model.lgb'}")

    # Threshold: extract from sweep JSON
    sweep_path = CHAMPION_SRC / "threshold" / "threshold_sweep.json"
    if sweep_path.exists():
        threshold = extract_best_threshold(sweep_path)
        threshold_json = {"best": {"threshold": threshold}}
        (CHAMPION_DEST / "threshold.json").write_text(json.dumps(threshold_json, indent=2))
        logger.info(f"Extracted champion threshold: {threshold}")
    else:
        logger.warning(f"No sweep file at {sweep_path}; service will default to 0.5")

    # Info
    info = {"name": "sentinel-champion", "version": "2", "framework": "lightgbm"}
    (CHAMPION_DEST / "info.json").write_text(json.dumps(info, indent=2))
    logger.success(f"Champion snapshot ready at {CHAMPION_DEST}")


def snapshot_challenger() -> None:
    CHALLENGER_DEST.mkdir(parents=True, exist_ok=True)

    pt_src = CHALLENGER_SRC / "model.pt"
    if not pt_src.exists():
        raise FileNotFoundError(f"No challenger .pt at {pt_src}")

    # Copy as is — the training script stores everything we need inside the checkpoint
    shutil.copy(pt_src, CHALLENGER_DEST / "model.pt")
    logger.info(f"Copied {pt_src.name} -> {CHALLENGER_DEST / 'model.pt'}")

    # Inspect to confirm the checkpoint shape (no error if it's a state_dict only)
    state = torch.load(CHALLENGER_DEST / "model.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        top_keys = sorted(state.keys())
        logger.info(f"Challenger .pt top-level keys: {top_keys}")
    else:
        logger.warning(
            "Challenger .pt is not a dict — may be a bare state_dict. "
            "Loader will need to handle this case."
        )

    # Threshold
    sweep_path = CHALLENGER_SRC / "threshold" / "threshold_sweep.json"
    if sweep_path.exists():
        threshold = extract_best_threshold(sweep_path)
        threshold_json = {"best": {"threshold": threshold}}
        (CHALLENGER_DEST / "threshold.json").write_text(json.dumps(threshold_json, indent=2))
        logger.info(f"Extracted challenger threshold: {threshold}")

    info = {"name": "sentinel-challenger", "version": "3", "framework": "pytorch"}
    (CHALLENGER_DEST / "info.json").write_text(json.dumps(info, indent=2))
    logger.success(f"Challenger snapshot ready at {CHALLENGER_DEST}")


def main() -> int:
    snapshot_champion()
    print()
    snapshot_challenger()
    print()

    logger.info("Final snapshot contents:")
    for d in [CHAMPION_DEST, CHALLENGER_DEST]:
        logger.info(f"  {d.relative_to(PROJECT_ROOT)}/")
        for f in sorted(d.iterdir()):
            logger.info(f"    {f.name} ({f.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())