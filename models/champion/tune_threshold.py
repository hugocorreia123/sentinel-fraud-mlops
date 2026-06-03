"""Tune the cost-aware decision threshold for the registered champion.

Pulls the latest version of `sentinel-champion` from the MLflow registry,
scores the val set, finds the threshold that minimizes expected cost,
and persists the result.

Usage:
    uv run python -m models.champion.tune_threshold
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
from loguru import logger

from models.threshold import CostConfig, tune_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "champion" / "artifacts" / "threshold"

MLFLOW_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentinel-champion"

# Cost matrix: fraud-team intuition.
# Missing a fraud (FN) costs ~100x a false alarm (FP).
COST_CONFIG = CostConfig(fn_cost=100.0, fp_cost=1.0)


def main() -> int:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sentinel-champion-lightgbm")

    # Load the latest registered champion
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
    logger.info(f"Loading model from registry: {model_uri}")
    model = mlflow.lightgbm.load_model(model_uri)

    # Load val
    val_df = pl.read_parquet(PROCESSED_DIR / "paysim_val_features.parquet")
    feature_cols = [c for c in val_df.columns if c not in {"isFraud", "step"}]
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["isFraud"].to_numpy()
    logger.info(f"Val: {X_val.shape}, fraud rate {y_val.mean() * 100:.3f}%")

    val_proba = model.predict(X_val)

    with mlflow.start_run(run_name="champion-threshold-tuning") as run:
        mlflow.set_tag("phase", "2.3")
        mlflow.set_tag("model_role", "champion-threshold")
        mlflow.log_param("registered_model", REGISTERED_MODEL_NAME)
        best = tune_threshold(
            y_true=y_val,
            y_proba=val_proba,
            cost=COST_CONFIG,
            artifacts_dir=ARTIFACTS_DIR,
            log_to_mlflow=True,
        )

        print()
        print("=" * 70)
        print("Cost-aware threshold tuning — champion")
        print("=" * 70)
        print(f"  Cost matrix: FN={COST_CONFIG.fn_cost}, FP={COST_CONFIG.fp_cost} "
              f"(missing fraud costs {COST_CONFIG.fn_cost / COST_CONFIG.fp_cost:.0f}x a false positive)")
        print()
        print(f"  Optimal threshold: {best.threshold:.4f}  "
              f"(vs naive default of 0.5)")
        print(f"  Expected cost at optimum: {best.expected_cost:.1f} units")
        print()
        print(f"  At this threshold (val set):")
        print(f"    Precision: {best.precision:.4f}")
        print(f"    Recall:    {best.recall:.4f}")
        print(f"    F1:        {best.f1:.4f}")
        print(f"    TP: {best.true_positives}, "
              f"FP: {best.false_positives}, "
              f"FN: {best.false_negatives}")
        print()
        print(f"  Plot:    {ARTIFACTS_DIR / 'cost_curve.png'}")
        print(f"  Sweep:   {ARTIFACTS_DIR / 'threshold_sweep.json'}")
        print(f"  MLflow:  {MLFLOW_URI}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())