"""Champion: tuned LightGBM with Optuna, tracked + registered in MLflow.

This is the model that gets deployed to production. Every Optuna trial is a
separate MLflow run; the winning model is registered as `sentinel-champion`
so the inference service (Phase 3) can load it by name+stage rather than by
file path.

Run:
    uv run python -m models.champion.train --trials 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import polars as pl
from loguru import logger
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "champion" / "artifacts"

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "sentinel-champion-lightgbm"
REGISTERED_MODEL_NAME = "sentinel-champion"

TARGET = "isFraud"
EXCLUDE_FROM_FEATURES = {TARGET, "step"}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = PROCESSED_DIR / f"paysim_{name}_features.parquet"
    df = pl.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    X = df.select(feature_cols).to_numpy()
    y = df[TARGET].to_numpy()
    return X, y, feature_cols


def suggest_params(trial: optuna.Trial) -> dict:
    """Optuna search space. Mid-sized — enough to find real improvements,
    bounded enough to converge in 30 trials."""
    return {
        "objective": "binary",
        "metric": "average_precision",  # PR-AUC: the right metric for imbalanced fraud
        "verbosity": -1,
        "is_unbalance": True,
        "seed": 42,
        # Searched:
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: list[str],
) -> float:
    """Train one model with the trial's suggested hyperparameters.
    Return val PR-AUC — Optuna maximises this."""
    params = suggest_params(trial)

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, feature_name=feature_names)

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        ],
    )
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_pr_auc = average_precision_score(y_val, val_pred)
    # Stash best_iteration so the final fit can reuse it without re-tuning.
    trial.set_user_attr("best_iteration", model.best_iteration)
    return val_pr_auc


def final_fit_and_register(
    best_params: dict,
    best_iteration: int,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_holdout: np.ndarray, y_holdout: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Refit at the best params, evaluate on holdout, register in MLflow."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    full_params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "is_unbalance": True,
        "seed": 42,
        **best_params,
    }

    with mlflow.start_run(run_name="champion-final") as run:
        # Log params
        mlflow.log_params(full_params)
        mlflow.log_param("best_iteration", best_iteration)
        mlflow.set_tag("phase", "2.2")
        mlflow.set_tag("model_role", "champion")
        mlflow.set_tag("model_family", "lightgbm")

        # Train at best params on train, with val for early stopping (same as trials)
        train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, feature_name=feature_names)

        t0 = perf_counter()
        model = lgb.train(
            full_params,
            train_ds,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            ],
        )
        train_seconds = perf_counter() - t0
        mlflow.log_metric("train_seconds", train_seconds)

        # Evaluate
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        holdout_pred = model.predict(X_holdout, num_iteration=model.best_iteration)

        val_metrics = {
            "val_roc_auc": float(roc_auc_score(y_val, val_pred)),
            "val_pr_auc": float(average_precision_score(y_val, val_pred)),
        }
        holdout_metrics = {
            "holdout_roc_auc": float(roc_auc_score(y_holdout, holdout_pred)),
            "holdout_pr_auc": float(average_precision_score(y_holdout, holdout_pred)),
        }
        # Operational precision@k
        for k in (100, 500, 1000):
            top_k_idx = np.argsort(holdout_pred)[-k:]
            holdout_metrics[f"holdout_precision_at_{k}"] = float(y_holdout[top_k_idx].mean())

        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(holdout_metrics)

        # Log model + register
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Persist locally too
        model.save_model(str(ARTIFACTS_DIR / "model.txt"),
                         num_iteration=model.best_iteration)
        (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(
            {"params": full_params, **val_metrics, **holdout_metrics,
             "train_seconds": train_seconds, "best_iteration": model.best_iteration,
             "mlflow_run_id": run.info.run_id},
            indent=2,
        ))

        print()
        print("=" * 70)
        print("Champion (tuned LightGBM) — final results")
        print("=" * 70)
        print(f"  Train time: {train_seconds:.1f}s   Best iter: {model.best_iteration}")
        print(f"  Val      ROC-AUC: {val_metrics['val_roc_auc']:.4f}    "
              f"PR-AUC: {val_metrics['val_pr_auc']:.4f}")
        print(f"  Holdout  ROC-AUC: {holdout_metrics['holdout_roc_auc']:.4f}    "
              f"PR-AUC: {holdout_metrics['holdout_pr_auc']:.4f}")
        print("  Precision @ K (holdout):")
        for k in (100, 500, 1000):
            print(f"    top {k:>4}: {holdout_metrics[f'holdout_precision_at_{k}']*100:.2f}% fraud")
        print(f"  MLflow run: {MLFLOW_URI}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")
        print(f"  Registered as: {REGISTERED_MODEL_NAME}")
        print("=" * 70)

        return {"run_id": run.info.run_id, **val_metrics, **holdout_metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Loading splits…")
    X_train, y_train, feature_names = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_holdout, y_holdout, _ = load_split("holdout")
    logger.info(f"Train {X_train.shape}, Val {X_val.shape}, Holdout {X_holdout.shape}")

    logger.info(f"Starting Optuna study: {args.trials} trials")
    mlflow_cb = MLflowCallback(
        tracking_uri=MLFLOW_URI,
        metric_name="val_pr_auc",
    )
    study = optuna.create_study(
        direction="maximize",
        study_name="sentinel-champion-tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val, feature_names),
        n_trials=args.trials,
        callbacks=[mlflow_cb],
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    logger.success(
        f"Best trial: #{best_trial.number}, val_pr_auc={best_trial.value:.4f}"
    )
    logger.info(f"Best params: {best_trial.params}")

    # Final fit + registry
    final_fit_and_register(
        best_params=best_trial.params,
        best_iteration=best_trial.user_attrs["best_iteration"],
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_holdout=X_holdout, y_holdout=y_holdout,
        feature_names=feature_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())