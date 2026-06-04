"""Prometheus metrics for the inference service.

Exported on /metrics; scraped by Prometheus, visualized in Grafana (Phase 5).
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

# Request counts
PREDICTIONS_TOTAL = Counter(
    "sentinel_predictions_total",
    "Total number of /predict requests handled",
    ["decision"],  # APPROVE | BLOCK
)

# End-to-end latency histogram (in seconds, Prometheus convention)
PREDICTION_LATENCY = Histogram(
    "sentinel_prediction_latency_seconds",
    "End-to-end latency of /predict requests",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Model fraud probability distribution — lets Grafana plot the score distribution
FRAUD_PROBABILITY = Histogram(
    "sentinel_fraud_probability",
    "Distribution of fraud probabilities returned",
    buckets=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)

# Redis lookup latency
REDIS_LATENCY = Histogram(
    "sentinel_redis_lookup_latency_seconds",
    "Latency of Redis velocity feature lookups",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
)

# Process-level info
MODEL_LOADED = Gauge(
    "sentinel_model_loaded",
    "1 if a champion model is loaded, else 0",
)

REDIS_REACHABLE = Gauge(
    "sentinel_redis_reachable",
    "1 if Redis is reachable, else 0",
)