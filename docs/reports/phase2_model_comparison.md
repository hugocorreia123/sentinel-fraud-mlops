# Phase 2 — Champion vs Challenger Model Comparison

> All three models trained on the same PaySim feature set (25 engineered features
> across 4,433,703 train rows), validated on a temporally-disjoint 973K-row val set,
> and evaluated on a never-touched 955K-row holdout. The fraud rate climbs from 0.082%
> in train to 0.420% in holdout — a real concept-drift scenario built into the data itself.

## Summary

| Model | Family | Train time | Val PR-AUC | Holdout PR-AUC | Holdout Precision@100 |
|---|---|---:|---:|---:|---:|
| Baseline | LightGBM, defaults | 4.5s | 0.5089 | 0.8909 | 94.00% |
| **Champion** | LightGBM, **Optuna-tuned** (30 trials) | 6.6s | **1.0000** | **0.9995** | **100.00%** |
| **Challenger** | PyTorch tabular MLP, MPS-accelerated | 491s | **0.9998** | **0.9963** | **100.00%** |

## Cost-aware decision threshold (champion)

Default 0.5 thresholds assume symmetric error costs. Fraud is asymmetric — a missed
fraud (FN) costs roughly **100× a false positive** (FP). The threshold tuning module
sweeps cost-weighted thresholds and selects the cost-minimizing point.

| Metric | Default 0.5 threshold | Cost-aware optimum |
|---|---:|---:|
| Threshold | 0.5 | **0.9990** |
| True positives | 570 | 570 |
| False positives | many | **1** |
| False negatives | 0 | **0** |
| Recall | 100% | **100%** |
| Precision | < 1% | **99.82%** |
| Expected cost (units) | high | **1.0** |

On a more imperfect (real-world) dataset the curve would be more informative; on
PaySim the model is so confident that the optimum collapses to the highest threshold
that still catches everything. The methodology is what matters — it generalizes to
any binary classifier under cost asymmetry.

## Why two models, not one

The champion (LightGBM) and challenger (PyTorch MLP) arrive at nearly identical
performance through completely different learning paradigms:

- **Champion**: greedy axis-aligned splits, learns rules like
  `if balance_mismatch=1 AND log_amount > 12 → fraud`.
- **Challenger**: gradient descent on a smooth decision surface, learns
  a continuous decision boundary over the standardized feature space.

In production, **diversity is insurance**. If concept drift breaks one architecture's
assumptions (e.g. categorical interaction patterns change), the other often survives.
A real fraud team runs the champion in production and the challenger in shadow mode
(see Phase 4) — comparing them continuously to detect when one degrades.

## The dataset is solved — what now?

With holdout PR-AUC near 1.0 and 100% precision at the operationally-relevant top-K,
**additional modelling effort produces only noise**. The honest portfolio signal is
to redirect remaining effort into the system around the model:

1. **Phase 3** — FastAPI inference service, Redis feature store, sub-100ms p99 latency.
2. **Phase 4** — shadow-mode A/B routing between champion and challenger, with
   statistical significance testing.
3. **Phase 5** — Prometheus / Grafana / Evidently drift detection. The 5× fraud-rate
   drift between train and holdout windows is a real demonstration vehicle.
4. **Phase 6** — adversarial robustness eval (gradient-based feature perturbations)
   and Locust load tests at 800+ RPS.

This is the difference between a Kaggle notebook (great model) and a fraud platform
(operable system). Phase 2 closes the model story so Phases 3–6 can build the system.

## Artifacts

| Path | What |
|---|---|
| `models/champion/artifacts/model.txt` | Champion LightGBM, serialized |
| `models/champion/artifacts/metrics.json` | Champion final metrics |
| `models/champion/artifacts/threshold/cost_curve.png` | Cost-vs-threshold plot |
| `models/champion/artifacts/threshold/threshold_sweep.json` | Full threshold sweep |
| `models/challenger/artifacts/model.pt` | Challenger MLP weights + scaler |
| `models/challenger/artifacts/metrics.json` | Challenger final metrics + history |
| MLflow registry `sentinel-champion` (v1, v2) | Champion model versions |
| MLflow registry `sentinel-challenger` (v1, v2) | Challenger model versions |