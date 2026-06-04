"""Train the PyTorch tabular challenger and register it in MLflow.

The architecture, loss, optimizer, and schedule are deliberately different
from the LightGBM champion — diversity is the value-add.

Usage:
    uv run python -m models.challenger.train --epochs 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import mlflow
import mlflow.pytorch
import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.challenger.model import FraudMLP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "challenger" / "artifacts"

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "sentinel-challenger-mlp"
REGISTERED_MODEL_NAME = "sentinel-challenger"

TARGET = "isFraud"
TYPE_COLS = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
# Exclude velocity features: in PaySim training data they are nearly constant
# (mean ~1.0, std ~0.03), so the scaler maps any production-time 0 to z ≈ -31,
# saturating the BatchNorm + sigmoid. Trees handle this fine; MLPs don't.
VELOCITY_COLS = {"velocity_count_1h", "velocity_count_6h", "velocity_count_24h"}
EXCLUDE = {TARGET, "step", *TYPE_COLS, *VELOCITY_COLS}

BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (X_numeric, x_type_index, y, numeric_feature_names)."""
    df = pl.read_parquet(PROCESSED_DIR / f"paysim_{name}_features.parquet")
    numeric_cols = [c for c in df.columns if c not in EXCLUDE]
    X_numeric = df.select(numeric_cols).to_numpy().astype(np.float32)

    # Reconstruct type index from the one-hot columns
    type_onehot = df.select(TYPE_COLS).to_numpy()
    x_type = np.argmax(type_onehot, axis=1).astype(np.int64)

    y = df[TARGET].to_numpy().astype(np.float32)
    return X_numeric, x_type, y, numeric_cols


def normalize(X_train: np.ndarray, X_val: np.ndarray, X_holdout: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard scaler: fit on train, apply to all."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # protect against zero-variance columns

    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_holdout - mean) / std,
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def make_loader(X_num, x_type, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X_num),
        torch.from_numpy(x_type),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probas = []
    labels = []
    for x_num, x_type, y in loader:
        x_num = x_num.to(device)
        x_type = x_type.to(device)
        logits = model(x_num, x_type)
        probas.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probas), np.concatenate(labels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = select_device()
    logger.info(f"Device: {device}")

    logger.info("Loading splits…")
    X_train_num, x_train_type, y_train, numeric_cols = load_split("train")
    X_val_num, x_val_type, y_val, _ = load_split("val")
    X_holdout_num, x_holdout_type, y_holdout, _ = load_split("holdout")
    logger.info(f"Numeric features: {len(numeric_cols)}, train shape: {X_train_num.shape}")

    # Normalize
    X_train_num, X_val_num, X_holdout_num, scaler_mean, scaler_std = normalize(
        X_train_num, X_val_num, X_holdout_num
    )

    train_loader = make_loader(X_train_num, x_train_type, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val_num, x_val_type, y_val, BATCH_SIZE, shuffle=False)
    holdout_loader = make_loader(X_holdout_num, x_holdout_type, y_holdout, BATCH_SIZE, shuffle=False)

    # Class-weighted loss — fraud is ~0.1% of train
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)],
        dtype=torch.float32,
        device=device,
    )
    logger.info(f"pos_weight: {pos_weight.item():.1f}")

    model = FraudMLP(n_numeric_features=len(numeric_cols)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with mlflow.start_run(run_name="challenger-mlp") as run:
        mlflow.set_tag("phase", "2.4")
        mlflow.set_tag("model_role", "challenger")
        mlflow.set_tag("model_family", "pytorch_mlp")
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": args.epochs,
            "n_params": n_params,
            "pos_weight": pos_weight.item(),
            "device": str(device),
        })

        best_val_pr = -1.0
        best_state = None
        history = []
        t0 = perf_counter()

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for x_num, x_type, y in train_loader:
                x_num = x_num.to(device)
                x_type = x_type.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x_num, x_type)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            train_loss = epoch_loss / max(n_batches, 1)
            val_proba, val_labels = predict_proba(model, val_loader, device)
            val_pr = average_precision_score(val_labels, val_proba)
            val_roc = roc_auc_score(val_labels, val_proba)

            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_pr_auc": val_pr, "val_roc_auc": val_roc})
            logger.info(
                f"Epoch {epoch:>3}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_PR-AUC={val_pr:.4f}  val_ROC-AUC={val_roc:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_pr_auc": val_pr, "val_roc_auc": val_roc},
                step=epoch,
            )

            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        train_seconds = perf_counter() - t0
        logger.success(f"Trained in {train_seconds:.1f}s. Best val PR-AUC: {best_val_pr:.4f}")
        mlflow.log_metric("train_seconds", train_seconds)

        # Restore best weights for final eval
        assert best_state is not None
        model.load_state_dict(best_state)

        val_proba, _ = predict_proba(model, val_loader, device)
        holdout_proba, _ = predict_proba(model, holdout_loader, device)

        val_metrics = {
            "val_roc_auc": float(roc_auc_score(y_val, val_proba)),
            "val_pr_auc": float(average_precision_score(y_val, val_proba)),
        }
        holdout_metrics = {
            "holdout_roc_auc": float(roc_auc_score(y_holdout, holdout_proba)),
            "holdout_pr_auc": float(average_precision_score(y_holdout, holdout_proba)),
        }
        for k in (100, 500, 1000):
            top_k = np.argsort(holdout_proba)[-k:]
            holdout_metrics[f"holdout_precision_at_{k}"] = float(y_holdout[top_k].mean())

        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(holdout_metrics)

        # Persist & register
        torch.save({"state_dict": model.state_dict(),
                    "scaler_mean": scaler_mean, "scaler_std": scaler_std,
                    "numeric_cols": numeric_cols},
                   ARTIFACTS_DIR / "model.pt")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(
            {**val_metrics, **holdout_metrics, "train_seconds": train_seconds,
             "history": history, "mlflow_run_id": run.info.run_id},
            indent=2,
        ))

        print()
        print("=" * 70)
        print("Challenger (PyTorch tabular MLP) — final results")
        print("=" * 70)
        print(f"  Train time: {train_seconds:.1f}s   Device: {device}")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())