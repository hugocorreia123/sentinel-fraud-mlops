"""FGSM adversarial robustness eval for champion vs challenger.

Generates small gradient-based perturbations to fraud transactions and tests
whether each model still detects the fraud. Tree-based models are typically
robust to gradient attacks (no gradient → can't directly optimize); neural nets
are not.

Usage:
    uv run python -m adversarial.fgsm_attack --n-samples 100 --epsilon 0.05
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import polars as pl
import torch
from loguru import logger

from apps.inference.model_loader import load_challenger, load_champion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "adversarial" / "reports"

TARGET = "isFraud"
TYPE_COLS = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
             "type_PAYMENT", "type_TRANSFER"]


def load_fraud_sample(n: int, seed: int = 42) -> tuple[pl.DataFrame, np.ndarray]:
    """Sample n confirmed-fraud rows from the holdout split."""
    df = pl.read_parquet(PROCESSED_DIR / "paysim_holdout_features.parquet")
    fraud = df.filter(pl.col(TARGET) == 1).sample(n=n, seed=seed)
    type_idx = np.argmax(fraud.select(TYPE_COLS).to_numpy(), axis=1).astype(np.int64)
    return fraud, type_idx


# Features the attacker can plausibly control. Balances are bookkeeping facts
# the attacker can't fake unilaterally, so we don't perturb them.
ATTACKER_CONTROLLABLE = {
    "amount", "log_amount",
    "amount_to_balance_ratio",
    "amount_vs_rolling_mean",
}


def fgsm_perturb(model, x_num: torch.Tensor, x_type: torch.Tensor,
                 epsilon: float, feature_names: list[str]) -> torch.Tensor:
    """Single-step FGSM, restricted to features an attacker could realistically modify.
    
    Balance fields (`oldbalanceOrg` etc.) are bookkeeping facts the system
    enforces; they cannot be unilaterally tampered with. We mask the gradient
    to zero on those columns so the attack only optimizes features the
    attacker actually controls (transaction amount, ratios).
    """
    x_num = x_num.clone().detach().requires_grad_(True)
    model.eval()
    logits = model(x_num, x_type)
    loss = logits.sum()
    loss.backward()
    
    # Build a mask: 1 for attacker-controllable features, 0 elsewhere
    mask = torch.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name in ATTACKER_CONTROLLABLE:
            mask[i] = 1.0
    
    perturbation = epsilon * x_num.grad.sign() * mask
    return (x_num - perturbation).detach()


def evaluate_attack(args) -> dict:
    logger.info(f"Loading models…")
    champion = load_champion()
    challenger = load_challenger()
    logger.info(
        f"  Champion: threshold {champion.threshold:.4f} "
        f"({champion.n_features} features)"
    )
    logger.info(
        f"  Challenger: threshold {challenger.threshold:.4f} "
        f"({len(challenger.feature_names)} numeric features)"
    )

    logger.info(f"Sampling {args.n_samples} fraud transactions from holdout…")
    fraud_df, type_idx = load_fraud_sample(args.n_samples)

    # Build champion-format input matrix (23 features, includes one-hot type)
    champ_features = champion.feature_names
    X_champ = fraud_df.select(champ_features).to_numpy().astype(np.float32)
    champ_probs = champion.booster.predict(
        X_champ, num_iteration=champion.booster.best_iteration
    )
    champ_baseline_detected = (champ_probs >= champion.threshold).sum()
    logger.info(
        f"Champion baseline: detects {champ_baseline_detected}/{args.n_samples} "
        f"({champ_baseline_detected/args.n_samples*100:.1f}%) of these fraud transactions"
    )

    # Build challenger-format input (15 numeric features, type as index)
    chall_features = challenger.feature_names
    X_chall_num = fraud_df.select(chall_features).to_numpy().astype(np.float32)
    scaler_mean = challenger.scaler_mean.cpu().numpy()
    scaler_std = challenger.scaler_std.cpu().numpy()
    X_chall_std = (X_chall_num - scaler_mean) / scaler_std

    x_num = torch.from_numpy(X_chall_std).float()
    x_type = torch.from_numpy(type_idx)

    with torch.no_grad():
        baseline_logits = challenger.model(x_num, x_type)
        baseline_probs = torch.sigmoid(baseline_logits).numpy()
    chall_baseline_detected = (baseline_probs >= challenger.threshold).sum()
    logger.info(
        f"Challenger baseline: detects {chall_baseline_detected}/{args.n_samples} "
        f"({chall_baseline_detected/args.n_samples*100:.1f}%) of these fraud transactions"
    )

    logger.info(f"Running FGSM attack with epsilon={args.epsilon} (std space)…")
    x_num_adv_std = fgsm_perturb(challenger.model, x_num, x_type, args.epsilon, chall_features)

    # Score the attacked inputs against the CHALLENGER (the model FGSM targeted)
    with torch.no_grad():
        adv_logits = challenger.model(x_num_adv_std, x_type)
        adv_probs = torch.sigmoid(adv_logits).numpy()
    chall_post_detected = (adv_probs >= challenger.threshold).sum()
    chall_flipped = chall_baseline_detected - chall_post_detected
    logger.info(
        f"Challenger after attack: detects {chall_post_detected}/{args.n_samples} "
        f"({chall_post_detected/args.n_samples*100:.1f}%); flipped {chall_flipped}"
    )

    # Un-standardize the perturbed numeric features so we can score with champion
    x_num_adv_raw = x_num_adv_std.numpy() * scaler_std + scaler_mean

   # Per-feature perturbation magnitude in raw space (saved to report, not logged)
    raw_perturb = x_num_adv_raw - X_chall_num
    per_feature_max_abs = np.abs(raw_perturb).max(axis=0)

    # Champion baseline already evaluated. Now apply the SAME perturbations to
    # the corresponding columns in the champion's input and re-score.
    X_champ_adv = X_champ.copy()
    feature_index = {name: i for i, name in enumerate(champ_features)}
    for j, name in enumerate(chall_features):
        if name in feature_index:
            X_champ_adv[:, feature_index[name]] = x_num_adv_raw[:, j]


    champ_adv_probs = champion.booster.predict(
        X_champ_adv, num_iteration=champion.booster.best_iteration
    )
    champ_post_detected = (champ_adv_probs >= champion.threshold).sum()
    champ_flipped = champ_baseline_detected - champ_post_detected
    logger.info(
        f"Champion after attack (using challenger-derived perturbation): "
        f"detects {champ_post_detected}/{args.n_samples} "
        f"({champ_post_detected/args.n_samples*100:.1f}%); flipped {champ_flipped}"
    )

    # Perturbation magnitude in standardized space
    perturb_l2_std = float(
        np.linalg.norm(x_num_adv_std.numpy() - X_chall_std, axis=1).mean()
    )

    report = {
        "n_samples": args.n_samples,
        "epsilon_std_space": args.epsilon,
        "perturbation_l2_std_mean": perturb_l2_std,
        "per_feature_max_abs_raw": {
            name: float(v) for name, v in zip(chall_features, per_feature_max_abs)
        },
        "champion": {
            "threshold": champion.threshold,
            "baseline_detected": int(champ_baseline_detected),
            "post_attack_detected": int(champ_post_detected),
            "flipped": int(champ_flipped),
            "attack_success_rate": float(champ_flipped) / max(int(champ_baseline_detected), 1),
            "note": "Perturbation was optimized against challenger; champion just receives the perturbed inputs",
        },
        "challenger": {
            "threshold": challenger.threshold,
            "baseline_detected": int(chall_baseline_detected),
            "post_attack_detected": int(chall_post_detected),
            "flipped": int(chall_flipped),
            "attack_success_rate": float(chall_flipped) / max(int(chall_baseline_detected), 1),
            "note": "FGSM optimized against this model",
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        help="FGSM step sizes to sweep (in standardized feature space)",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_reports = []
    for eps in args.epsilons:
        logger.info(f"\n{'='*70}\nRunning eps={eps}\n{'='*70}")
        single_args = argparse.Namespace(n_samples=args.n_samples, epsilon=eps)
        report = evaluate_attack(single_args)
        all_reports.append(report)

    # Save combined sweep
    out_path = REPORTS_DIR / "fgsm_sweep.json"
    out_path.write_text(json.dumps(all_reports, indent=2))

    # Summary table
    print()
    print("=" * 90)
    print(f"FGSM robustness sweep — n={args.n_samples} fraud samples")
    print("=" * 90)
    print(
        f"{'epsilon':>10} {'perturb L2':>12} "
        f"{'champ baseline':>15} {'champ attacked':>15} "
        f"{'chall baseline':>15} {'chall attacked':>15}"
    )
    print("-" * 90)
    for r in all_reports:
        print(
            f"{r['epsilon_std_space']:>10.4f} "
            f"{r['perturbation_l2_std_mean']:>12.4f} "
            f"{r['champion']['baseline_detected']:>15d} "
            f"{r['champion']['post_attack_detected']:>15d} "
            f"{r['challenger']['baseline_detected']:>15d} "
            f"{r['challenger']['post_attack_detected']:>15d}"
        )
    print("-" * 90)
    print(f"Sweep saved to: {out_path}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())