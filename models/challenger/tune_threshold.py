"""Tune the cost-aware decision threshold for the registered challenger.

The challenger MLP was trained with class-weighted BCE (pos_weight ~1219),
which pushes its raw probabilities upward to fight imbalance. Champion's
threshold (0.999) is calibrated for tree-based outputs and is wrong for the
MLP — applying it makes the challenger BLOCK nearly everything.

This script computes the cost-minimizing threshold for the challenger on val,
which lets the shadow comparison evaluate the models fairly.

Usage:
    uv run python -m models.challenger.tune_threshold
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
from loguru import logger

from apps.inference.model_loader import load_challenger
from models.threshold import CostConfig, tune_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "challenger" / "artifacts" / "threshold"

TARGET = "isFraud"
TYPE_COLS = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
             "type_PAYMENT", "type_TRANSFER"]

COST_CONFIG = CostConfig(fn_cost=100.0, fp_cost=1.0)


def score_val(challenger) -> tuple[np.ndarray, np.ndarray]:
    """Run challenger over the val parquet, return (y_proba, y_true)."""
    val = pl.read_parquet(PROCESSED_DIR / "paysim_val_features.parquet")
    y_true = val[TARGET].to_numpy().astype(np.int32)

    # Numeric features in the challenger's expected order
    X_num = val.select(challenger.feature_names).to_numpy().astype(np.float32)
    # Type as a single index column reconstructed from the one-hot
    x_type = np.argmax(val.select(TYPE_COLS).to_numpy(), axis=1).astype(np.int64)

    # Standardize using the scaler saved with the model
    scaler_mean = challenger.scaler_mean.cpu().numpy()
    scaler_std = challenger.scaler_std.cpu().numpy()
    X_std = (X_num - scaler_mean) / scaler_std

    # Score in batches; model is small but val is big
    batch = 8192
    probas = np.empty(len(y_true), dtype=np.float32)
    challenger.model.eval()
    with torch.no_grad():
        for i in range(0, len(y_true), batch):
            sl = slice(i, i + batch)
            xn = torch.from_numpy(X_std[sl]).to(challenger.device)
            xt = torch.from_numpy(x_type[sl]).to(challenger.device)
            logits = challenger.model(xn, xt)
            probas[sl] = torch.sigmoid(logits).cpu().numpy()

    return probas, y_true


def main() -> int:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("sentinel-challenger-mlp")

    logger.info("Loading challenger…")
    challenger = load_challenger()
    logger.info(f"Loaded {challenger.name} v{challenger.version}")

    logger.info("Scoring val with challenger…")
    val_proba, y_val = score_val(challenger)
    logger.info(
        f"Val: {len(y_val):,} rows, "
        f"fraud rate {y_val.mean() * 100:.3f}%, "
        f"proba range [{val_proba.min():.4f}, {val_proba.max():.4f}], "
        f"mean {val_proba.mean():.4f}"
    )

    with mlflow.start_run(run_name="challenger-threshold-tuning") as run:
        mlflow.set_tag("phase", "4.3")
        mlflow.set_tag("model_role", "challenger-threshold")
        mlflow.log_param("registered_model", challenger.name)

        best = tune_threshold(
            y_true=y_val,
            y_proba=val_proba,
            cost=COST_CONFIG,
            artifacts_dir=ARTIFACTS_DIR,
            log_to_mlflow=True,
        )

        print()
        print("=" * 70)
        print("Cost-aware threshold tuning — challenger")
        print("=" * 70)
        print(f"  Cost matrix: FN={COST_CONFIG.fn_cost}, FP={COST_CONFIG.fp_cost}")
        print()
        print(f"  Optimal threshold: {best.threshold:.4f}")
        print(f"  Expected cost: {best.expected_cost:.1f} units")
        print()
        print(f"  At this threshold (val):")
        print(f"    Precision: {best.precision:.4f}")
        print(f"    Recall:    {best.recall:.4f}")
        print(f"    F1:        {best.f1:.4f}")
        print(f"    TP: {best.true_positives},  "
              f"FP: {best.false_positives},  "
              f"FN: {best.false_negatives}")
        print()
        print(f"  Plot:    {ARTIFACTS_DIR / 'cost_curve.png'}")
        print(f"  Sweep:   {ARTIFACTS_DIR / 'threshold_sweep.json'}")
        print(f"  MLflow:  http://127.0.0.1:5000/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())