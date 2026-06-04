# Phase 4 — Shadow Mode + A/B Routing: A Real Production Bug

> Shadow mode caught a calibration bug that holdout metrics couldn't.
> Champion and challenger had near-identical PR-AUC on the holdout set, but
> diverged catastrophically on production-shape traffic. This report walks
> through the discovery, root-cause analysis, and fix.

## The setup

The Sentinel inference service runs both models on every request: champion
(LightGBM) returns the decision served to the client; challenger (PyTorch
tabular MLP) is scored in shadow mode. Both predictions are written to a
SQLite log (`data/predictions.db`) for offline analysis. Traffic split between
models is controlled by `CHALLENGER_TRAFFIC_PCT` and uses a stable SHA-256
hash of `request_id` modulo 100 — so the same request always routes to the
same model.

## What we found

After 30 seconds of mixed traffic (90% clean PAYMENT, 5% clean CASH_OUT,
5% fraud TRANSFER) at 800 RPS:

| Confusion cell | Count |
|---|---:|
| champion=APPROVE, challenger=APPROVE | 2 |
| champion=APPROVE, challenger=BLOCK | **22,638** |
| champion=BLOCK, challenger=BLOCK | 1,208 |

Agreement rate: **5.07%**. The challenger was blocking *everything* except a
handful of TRANSFERs that even the champion agreed were fraud.

Phase 2 holdout metrics for the same challenger:
- PR-AUC: 0.9963
- Precision@100: 100%
- Holdout ROC-AUC: 0.9993

The metrics looked great. The system behaved nothing like the metrics said.

## Root cause

Drilling into the actual standardized inputs the MLP was seeing on a clean
PAYMENT (amount=50, balance=1000):
velocity_count_1h:   -31.19 std away from training mean
velocity_count_6h:   -31.14 std away
velocity_count_24h:  -31.11 std away
Three inputs were ~31 standard deviations below zero. The cause:

1. In PaySim training data, **velocity counters are near-constant** —
   most accounts make at most one transaction per window. The scaler learned
   mean ≈ 1.001, std ≈ 0.032 for all three velocity features.
2. At inference time, Locust generates traffic from fresh accounts not in
   the Redis feature store, so velocity features default to 0.
3. The scaling produces `(0 - 1.001) / 0.032 ≈ -31`.
4. Inside the MLP's first BatchNorm + ReLU layer, three -31 inputs produce
   a strong out-of-distribution signal. The model interprets this as a
   strong fraud indicator and saturates the output near 1.0.

LightGBM (the champion) is **scale-invariant** — its splits don't care about
z-scores. The MLP (challenger) is exquisitely sensitive.

## The deeper lesson

The holdout metric (PR-AUC 0.9963) wasn't lying. It was answering the wrong
question. PR-AUC measures *ranking* on the holdout set, where velocity
features have the same near-constant distribution as in training. The model
correctly rank-ordered val and holdout fraud. What broke was **probability
calibration under feature distribution shift** — exactly what every fraud
team will tell you can't be detected with offline metrics alone.

**Shadow mode exists precisely to catch this class of bug**, and it did.

## The fix

Retrained the challenger excluding the three velocity columns. The
distribution-shift trap is gone; for a tabular MLP, three nearly-constant
features were pure noise anyway. The new model has 15 numeric features
instead of 18 and the holdout PR-AUC *improved* from 0.9963 to 0.9984.

| | Challenger v2 (with velocity) | Challenger v3 (without velocity) |
|---|---:|---:|
| Numeric features | 18 | 15 |
| Holdout PR-AUC | 0.9963 | **0.9984** |
| Holdout ROC-AUC | 0.9993 | 0.9999 |
| Precision@100 | 100% | 100% |
| Tuned threshold | 0.9894 | **0.9990** (matches champion) |
| Threshold-tuning FP | 59 | 1 (matches champion) |

The two models are now operating in the same probability regime.

## After the fix

Re-ran shadow mode at 50/50 A/B with the same Locust traffic profile:

| Confusion cell | Count |
|---|---:|
| champion=APPROVE, challenger=APPROVE | 23,609 |
| champion=APPROVE, challenger=BLOCK | 0 |
| champion=BLOCK, challenger=APPROVE | 0 |
| champion=BLOCK, challenger=BLOCK | 1,205 |

**Agreement rate: 100.000% on 24,814 decisions.** Probability correlation:
**0.9999**. The two completely independent learning paradigms produce
byte-identical decisions on the realistic traffic mix.

On a real card-not-present dataset there would be meaningful disagreement,
and McNemar's test (the binomial significance test we implemented) would
flag whether the disagreement is statistically significant. On PaySim the
dataset is too clean for ambiguity. The methodology is proven; the test
artifact exists; both will activate the moment we point Sentinel at less
saturated data.

## Service performance under shadow scoring

Locust 30s, 30 concurrent clients, both models scored per request:

| Metric | Phase 3 (champion only) | Phase 4 (champion + shadow) |
|---|---:|---:|
| Sustained RPS | 1,371 | 832 |
| p50 latency | 25ms | 28ms |
| p95 latency | 82ms | 40ms |
| p99 latency | 96ms | 110ms |
| Failures | 0 | 0 |

The extra PyTorch forward pass costs ~40% throughput but a real fraud team
runs shadow scoring on dedicated infrastructure anyway. Per-request p50
barely moved.

## Artifacts

| Path | What |
|---|---|
| `apps/inference/prediction_log.py` | SQLite logger with WAL + thread lock |
| `apps/inference/routing.py` | Stable SHA-256 hash routing |
| `apps/inference/model_loader.py` | Loads both models, per-model thresholds |
| `models/challenger/tune_threshold.py` | Reuses Phase 2 cost-aware module on the MLP |
| `scripts/analyze_shadow.py` | Agreement rate + McNemar's test |
| `data/predictions.db` | Live prediction log (gitignored) |
| MLflow `sentinel-challenger` v3 | Velocity-free retrain |
