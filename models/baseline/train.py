"""Baseline LightGBM — fast, default hyperparameters, no tuning.

Purpose: produce one honest holdout metric to anchor every later improvement.
Phase 2 will train the real champion (LightGBM with tuned cost-aware threshold)
and challenger (PyTorch tabular NN). Both must beat this baseline to justify
their existence.

Run:
    uv run python -m models.baseline.train
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "baseline" / "artifacts"

TARGET = "isFraud"
# Features the model will see (everything except the target itself + step,
# which we keep for the velocity calc but don't want as a direct feature).
EXCLUDE_FROM_FEATURES = {TARGET, "step"}

# Default LightGBM params. Deliberately mild — no tuning yet.
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "is_unbalance": True,  # critical for highly imbalanced fraud data
    "seed": 42,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a split and return (X, y, feature_names)."""
    path = PROCESSED_DIR / f"paysim_{name}_features.parquet"
    df = pl.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    X = df.select(feature_cols).to_numpy()
    y = df[TARGET].to_numpy()
    return X, y, feature_cols


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading splits…")
    X_train, y_train, feature_names = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_holdout, y_holdout, _ = load_split("holdout")

    logger.info(
        f"Train: {X_train.shape}, fraud rate {y_train.mean()*100:.3f}%   |   "
        f"Val: {X_val.shape}, fraud rate {y_val.mean()*100:.3f}%   |   "
        f"Holdout: {X_holdout.shape}, fraud rate {y_holdout.mean()*100:.3f}%"
    )

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, feature_name=feature_names)

    logger.info(f"Training LightGBM (up to {NUM_BOOST_ROUND} rounds, "
                f"early stop after {EARLY_STOPPING_ROUNDS}) …")
    t0 = perf_counter()
    model = lgb.train(
        DEFAULT_PARAMS,
        train_ds,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    train_seconds = perf_counter() - t0
    logger.success(f"Trained in {train_seconds:.1f}s. Best iteration: {model.best_iteration}.")

    # Holdout evaluation — touched exactly once.
    logger.info("Scoring holdout…")
    holdout_proba = model.predict(X_holdout, num_iteration=model.best_iteration)
    holdout_auc = roc_auc_score(y_holdout, holdout_proba)
    holdout_pr_auc = average_precision_score(y_holdout, holdout_proba)

    # Also score val for the report
    val_proba = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_proba)
    val_pr_auc = average_precision_score(y_val, val_proba)

    # Precision@K for fraud-team operations: what fraction of the top-K riskiest
    # transactions are actually fraud? This is the metric a fraud reviewer cares
    # about — they can only look at so many cases per day.
    precision_at_k = {}
    for k in (100, 500, 1000):
        top_k_idx = np.argsort(holdout_proba)[-k:]
        precision_at_k[k] = float(y_holdout[top_k_idx].mean())

    # Top features by gain
    importance = model.feature_importance(importance_type="gain")
    top_features = sorted(
        zip(feature_names, importance, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    # Report
    print()
    print("=" * 70)
    print("Baseline LightGBM — holdout metrics")
    print("=" * 70)
    print(f"  Val      ROC-AUC: {val_auc:.4f}    PR-AUC: {val_pr_auc:.4f}")
    print(f"  Holdout  ROC-AUC: {holdout_auc:.4f}    PR-AUC: {holdout_pr_auc:.4f}")
    print()
    print("  Precision @ K (holdout):")
    for k, p in precision_at_k.items():
        print(f"    top {k:>4}: {p*100:.2f}% fraud")
    print()
    print("  Top 10 features by gain:")
    for name, imp in top_features:
        print(f"    {name:<28} {imp:>14,.0f}")
    print("=" * 70)

    # Persist model + metrics
    model_path = ARTIFACTS_DIR / "model.txt"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    model.save_model(str(model_path), num_iteration=model.best_iteration)

    metrics = {
        "model": "lightgbm_baseline",
        "train_seconds": round(train_seconds, 2),
        "best_iteration": model.best_iteration,
        "val": {"roc_auc": val_auc, "pr_auc": val_pr_auc},
        "holdout": {"roc_auc": holdout_auc, "pr_auc": holdout_pr_auc},
        "precision_at_k": precision_at_k,
        "top_features": [{"name": n, "gain": float(g)} for n, g in top_features],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.success(f"Saved model → {model_path.name}, metrics → {metrics_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())