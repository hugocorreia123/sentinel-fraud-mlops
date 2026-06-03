"""Cost-aware threshold tuning for fraud classifiers.

Default 0.5 thresholds assume symmetric error costs. Fraud is asymmetric:
a missed fraud (FN) costs ~100× a false positive (FP). The optimal threshold
minimizes expected € cost, not error rate.

This module:
  1. Sweeps thresholds 0.001 -> 0.999 on val predictions
  2. Computes expected cost at each threshold using a configurable cost matrix
  3. Picks the cost-minimizing threshold
  4. Logs the analysis to MLflow

Usage:
    from models.threshold import tune_threshold, CostConfig
    result = tune_threshold(y_true, y_proba, CostConfig(fn_cost=100, fp_cost=1))
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from loguru import logger
from sklearn.metrics import confusion_matrix


@dataclass
class CostConfig:
    """Cost matrix in arbitrary currency units.

    Defaults reflect industry-typical fraud asymmetry: a missed fraud
    is ~100× more costly than a false positive.
    """
    fn_cost: float = 100.0  # cost of missing a fraud
    fp_cost: float = 1.0    # cost of a false alarm
    tp_cost: float = 0.0    # cost of correctly catching fraud (usually 0)
    tn_cost: float = 0.0    # cost of correctly clearing a tx (usually 0)


@dataclass
class ThresholdResult:
    """Result of a threshold sweep."""
    threshold: float
    expected_cost: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost: CostConfig,
) -> ThresholdResult:
    """Compute confusion matrix + expected cost at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    expected_cost = (
        fn * cost.fn_cost
        + fp * cost.fp_cost
        + tp * cost.tp_cost
        + tn * cost.tn_cost
    )
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ThresholdResult(
        threshold=float(threshold),
        expected_cost=float(expected_cost),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost: CostConfig,
    num_thresholds: int = 200,
) -> list[ThresholdResult]:
    """Sweep thresholds and return all results."""
    # Concentrate samples near the action — fraud predictions cluster near 0
    # so most of the interesting decisions are at low thresholds.
    thresholds = np.unique(np.concatenate([
        np.linspace(0.001, 0.05, num_thresholds // 2),  # dense low range
        np.linspace(0.05, 0.999, num_thresholds // 2),  # sparse high range
    ]))
    return [evaluate_at_threshold(y_true, y_proba, t, cost) for t in thresholds]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost: CostConfig,
) -> tuple[ThresholdResult, list[ThresholdResult]]:
    """Find the threshold that minimizes expected cost."""
    sweep = sweep_thresholds(y_true, y_proba, cost)
    best = min(sweep, key=lambda r: r.expected_cost)
    return best, sweep


def plot_cost_curve(
    sweep: list[ThresholdResult],
    best: ThresholdResult,
    cost: CostConfig,
    out_path: Path,
) -> None:
    """Plot expected cost vs threshold, with the optimum marked."""
    thresholds = [r.threshold for r in sweep]
    costs = [r.expected_cost for r in sweep]
    recalls = [r.recall for r in sweep]
    precisions = [r.precision for r in sweep]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: cost curve
    axes[0].plot(thresholds, costs, label="Expected cost", color="#B91C1C", linewidth=2)
    axes[0].axvline(best.threshold, linestyle="--", color="#15803D", linewidth=1.5,
                    label=f"Optimal threshold = {best.threshold:.4f}")
    axes[0].set_ylabel("Expected cost (units)")
    axes[0].set_title(
        f"Cost-aware threshold tuning  "
        f"(FN cost = {cost.fn_cost}, FP cost = {cost.fp_cost})"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: precision and recall vs threshold
    axes[1].plot(thresholds, precisions, label="Precision", color="#1D4ED8", linewidth=2)
    axes[1].plot(thresholds, recalls, label="Recall", color="#B45309", linewidth=2)
    axes[1].axvline(best.threshold, linestyle="--", color="#15803D", linewidth=1.5)
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost: CostConfig | None = None,
    artifacts_dir: Path | None = None,
    log_to_mlflow: bool = True,
) -> ThresholdResult:
    """Find optimal threshold, log to MLflow, save plot."""
    cost = cost or CostConfig()
    best, sweep = find_optimal_threshold(y_true, y_proba, cost)

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Cost curve plot
        plot_path = artifacts_dir / "cost_curve.png"
        plot_cost_curve(sweep, best, cost, plot_path)
        # Full sweep as JSON for reproducibility
        sweep_path = artifacts_dir / "threshold_sweep.json"
        sweep_path.write_text(json.dumps(
            {"cost_config": asdict(cost),
             "best": best.as_dict(),
             "sweep": [r.as_dict() for r in sweep]},
            indent=2,
        ))

    if log_to_mlflow and mlflow.active_run() is not None:
        mlflow.log_metrics({
            "threshold_optimal": best.threshold,
            "threshold_expected_cost": best.expected_cost,
            "threshold_precision": best.precision,
            "threshold_recall": best.recall,
            "threshold_f1": best.f1,
            "threshold_tp": best.true_positives,
            "threshold_fp": best.false_positives,
            "threshold_fn": best.false_negatives,
        })
        mlflow.log_params({
            "fn_cost": cost.fn_cost,
            "fp_cost": cost.fp_cost,
        })
        if artifacts_dir is not None:
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="threshold")

    logger.info(
        f"Optimal threshold: {best.threshold:.4f}  "
        f"(precision={best.precision:.3f}, recall={best.recall:.3f}, "
        f"FN={best.false_negatives}, FP={best.false_positives})"
    )
    return best