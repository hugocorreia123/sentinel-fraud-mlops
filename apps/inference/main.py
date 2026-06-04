"""Sentinel — real-time fraud-detection inference service.

POST /predict: score one transaction; return APPROVE/BLOCK with probability.
GET  /health:  liveness probe (model loaded? redis reachable?).
GET  /model/info: introspection — registered model name, version, features.

Run locally:
    uv run uvicorn apps.inference.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import redis
from fastapi import FastAPI, HTTPException
from loguru import logger

from apps.inference.model_loader import LoadedModel, load_champion
from apps.inference.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

# ----- module-level state, populated at startup --------------------
SERVICE_STARTED_AT = time.time()
MODEL: LoadedModel | None = None
REDIS: redis.Redis | None = None

TYPE_INDEX = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model + connect to Redis at startup; clean up on shutdown."""
    global MODEL, REDIS
    logger.info("Sentinel inference service starting…")

    MODEL = load_champion()
    logger.success(
        f"Champion loaded: {MODEL.name} v{MODEL.version}, "
        f"{MODEL.n_features} features, threshold={MODEL.threshold:.4f}"
    )

    REDIS = redis.Redis(host="localhost", port=6379, decode_responses=True,
                        socket_connect_timeout=1, socket_timeout=1)
    try:
        REDIS.ping()
        logger.success("Connected to Redis feature store")
    except Exception as e:
        logger.warning(f"Redis unreachable at startup: {e}")
        REDIS = None  # service still works; velocity features default to 0

    yield

    logger.info("Sentinel inference service shutting down")


app = FastAPI(
    title="Sentinel — Fraud Detection API",
    version="0.1.0",
    description="Real-time fraud scoring with cost-aware decision threshold.",
    lifespan=lifespan,
)


# ============================================================
#  /health
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    redis_ok = False
    if REDIS is not None:
        try:
            REDIS.ping()
            redis_ok = True
        except Exception:
            redis_ok = False

    status = "ok" if (MODEL is not None and redis_ok) else "degraded"
    if MODEL is None:
        status = "down"

    return HealthResponse(
        status=status,
        model_loaded=MODEL is not None,
        redis_reachable=redis_ok,
        uptime_seconds=round(time.time() - SERVICE_STARTED_AT, 2),
    )


# ============================================================
#  /model/info
# ============================================================
@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return ModelInfoResponse(
        name=MODEL.name,
        version=MODEL.version,
        stage=MODEL.stage,
        framework="lightgbm",
        n_features=MODEL.n_features,
        feature_names=MODEL.feature_names,
        threshold=MODEL.threshold,
        loaded_at=MODEL.loaded_at,
    )


# ============================================================
#  /predict
# ============================================================
def _fetch_velocity(name_orig: str) -> tuple[int, int, int]:
    """Look up (v1h, v6h, v24h) from Redis. Returns zeros if unavailable."""
    if REDIS is None:
        return 0, 0, 0
    try:
        vals = REDIS.hmget(f"sentinel:velocity:{name_orig}", "v1h", "v6h", "v24h")
        return tuple(int(v) if v is not None else 0 for v in vals)  # type: ignore[return-value]
    except Exception as e:
        logger.warning(f"Redis lookup failed for {name_orig}: {e}")
        return 0, 0, 0


def _build_feature_vector(tx: TransactionRequest,
                          velocity: tuple[int, int, int]) -> np.ndarray:
    """Replicate the offline feature engineering for one transaction.

    Order MUST match the training feature order (MODEL.feature_names).
    """
    v1h, v6h, v24h = velocity

    # Static features
    log_amount = np.log1p(tx.amount)
    orig_balance_delta = tx.oldbalanceOrg - tx.newbalanceOrig
    dest_balance_delta = tx.newbalanceDest - tx.oldbalanceDest
    balance_mismatch = int(
        abs((tx.oldbalanceOrg - tx.amount) - tx.newbalanceOrig) > 0.01
    )
    drained_to_zero = int(tx.newbalanceOrig == 0 and tx.oldbalanceOrg > 0)
    dest_was_empty = int(tx.oldbalanceDest == 0)
    amount_to_balance_ratio = (
        tx.amount / tx.oldbalanceOrg if tx.oldbalanceOrg > 0 else 0.0
    )
    dest_is_merchant = int(tx.nameDest.startswith("M"))

    # One-hot type
    type_one_hot = [0] * 5
    type_one_hot[TYPE_INDEX[tx.type]] = 1

    # Build features in the SAME order as training. The model's
    # booster.feature_name() gives us the canonical order.
    feature_map = {
        "amount": tx.amount,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "log_amount": log_amount,
        "orig_balance_delta": orig_balance_delta,
        "dest_balance_delta": dest_balance_delta,
        "balance_mismatch": balance_mismatch,
        "drained_to_zero": drained_to_zero,
        "dest_was_empty": dest_was_empty,
        "amount_to_balance_ratio": amount_to_balance_ratio,
        "dest_is_merchant": dest_is_merchant,
        "type_CASH_IN": type_one_hot[0],
        "type_CASH_OUT": type_one_hot[1],
        "type_DEBIT": type_one_hot[2],
        "type_PAYMENT": type_one_hot[3],
        "type_TRANSFER": type_one_hot[4],
        "velocity_count_1h": v1h,
        "velocity_count_6h": v6h,
        "velocity_count_24h": v24h,
        # Behavioural features — at inference we don't have history,
        # so we approximate. Real systems compute these from a feature store.
        "rolling_mean_amount": tx.amount,  # fallback: assume mean = current
        "amount_vs_rolling_mean": 1.0,
    }

    assert MODEL is not None
    try:
        vec = np.array([feature_map[name] for name in MODEL.feature_names],
                       dtype=np.float32)
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature {e} expected by model but not built at inference",
        )
    return vec.reshape(1, -1)


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    request_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    velocity = _fetch_velocity(tx.nameOrig)
    X = _build_feature_vector(tx, velocity)
    proba = float(MODEL.booster.predict(X, num_iteration=MODEL.booster.best_iteration)[0])
    decision = "BLOCK" if proba >= MODEL.threshold else "APPROVE"

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictionResponse(
        decision=decision,
        fraud_probability=round(proba, 6),
        threshold_used=MODEL.threshold,
        model_name=MODEL.name,
        model_version=MODEL.version,
        latency_ms=latency_ms,
        features_used=MODEL.n_features,
        request_id=request_id,
    )