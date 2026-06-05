"""Sentinel — real-time fraud-detection inference service.

POST /predict: score one transaction; return APPROVE/BLOCK with probability.
GET  /health:  liveness probe (model loaded? redis reachable?).
GET  /model/info: introspection — registered model name, version, features.

Run locally:
    uv run uvicorn apps.inference.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("MLFLOW_DISABLE_ENV_CREATION", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LIGHTGBM_EXEC_THREADS", "1")

import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import redis
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from apps.inference.prediction_log import PredictionLog, PredictionRecord
from apps.inference.routing import pick_serving_model

from apps.inference.metrics import (
    FRAUD_PROBABILITY,
    MODEL_LOADED,
    PREDICTIONS_TOTAL,
    PREDICTION_LATENCY,
    REDIS_LATENCY,
    REDIS_REACHABLE,
)
from apps.inference.model_loader import (
    LoadedChallenger,
    LoadedModel,
    load_challenger,
    load_champion,
)
from apps.inference.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

# ----- module-level state, populated at startup --------------------
SERVICE_STARTED_AT = time.time()
MODEL: LoadedModel | None = None
CHALLENGER: LoadedChallenger | None = None
REDIS: redis.Redis | None = None

TYPE_INDEX = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model + connect to Redis at startup; clean up on shutdown."""
    global MODEL, CHALLENGER, REDIS
    logger.info("Sentinel inference service starting…")

    MODEL = load_champion()
    MODEL_LOADED.set(1)
    logger.success(
        f"Champion loaded: {MODEL.name} v{MODEL.version}, "
        f"{MODEL.n_features} features, threshold={MODEL.threshold:.4f}"
    )

    try:
        CHALLENGER = load_challenger()
        logger.success(
            f"Challenger loaded: {CHALLENGER.name} v{CHALLENGER.version}, "
            f"{len(CHALLENGER.feature_names)} numeric features"
        )
    except Exception as e:
        logger.warning(f"Challenger unavailable: {e}")
        CHALLENGER = None

    REDIS = redis.Redis(host="localhost", port=6379, decode_responses=True,
                        socket_connect_timeout=1, socket_timeout=1)
    try:
        REDIS.ping()
        REDIS_REACHABLE.set(1)
        logger.success("Connected to Redis feature store")
    except Exception as e:
        logger.warning(f"Redis unreachable at startup: {e}")
        REDIS_REACHABLE.set(0)
        REDIS = None

    global PRED_LOG
    PRED_LOG = PredictionLog()
    logger.success(f"Prediction log ready ({PRED_LOG.count()} existing rows)")

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

    status = "ok" if MODEL is not None else "down"

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
#  Feature engineering at inference time
# ============================================================
def _fetch_velocity(name_orig: str) -> tuple[int, int, int]:
    """Look up (v1h, v6h, v24h) from Redis. Returns zeros if unavailable."""
    if REDIS is None:
        return 0, 0, 0
    try:
        with REDIS_LATENCY.time():
            vals = REDIS.hmget(f"sentinel:velocity:{name_orig}", "v1h", "v6h", "v24h")
        return tuple(int(v) if v is not None else 0 for v in vals)  # type: ignore[return-value]
    except Exception as e:
        logger.warning(f"Redis lookup failed for {name_orig}: {e}")
        return 0, 0, 0


def _build_full_feature_map(tx: TransactionRequest,
                             velocity: tuple[int, int, int]) -> dict[str, float]:
    """Compute the union of all feature values needed by champion or challenger."""
    v1h, v6h, v24h = velocity
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

    type_one_hot = [0] * 5
    type_one_hot[TYPE_INDEX[tx.type]] = 1

    return {
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
        "rolling_mean_amount": tx.amount,
        "amount_vs_rolling_mean": 1.0,
    }


def _build_feature_vector(tx: TransactionRequest,
                          velocity: tuple[int, int, int]) -> np.ndarray:
    """Build the LightGBM input row in the model's expected feature order."""
    feature_map = _build_full_feature_map(tx, velocity)
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


def _challenger_score(tx: TransactionRequest,
                       velocity: tuple[int, int, int]) -> float | None:
    """Score one transaction with the challenger MLP. Returns None if unavailable."""
    if CHALLENGER is None:
        return None
    try:
        feature_map = _build_full_feature_map(tx, velocity)
        numeric = [feature_map[name] for name in CHALLENGER.feature_names]
        x_num = torch.tensor([numeric], dtype=torch.float32, device=CHALLENGER.device)
        x_num = (x_num - CHALLENGER.scaler_mean) / CHALLENGER.scaler_std

        type_index = TYPE_INDEX[tx.type]
        x_type = torch.tensor([type_index], dtype=torch.long, device=CHALLENGER.device)

        with torch.no_grad():
            logits = CHALLENGER.model(x_num, x_type)
            proba = torch.sigmoid(logits).item()
        return float(proba)
    except Exception as e:
        logger.warning(f"Challenger scoring failed: {e}")
        return None


# ============================================================
#  /predict
# ============================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    request_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    with PREDICTION_LATENCY.time():
        velocity = _fetch_velocity(tx.nameOrig)

        # Champion (always scored)
        X = _build_feature_vector(tx, velocity)
        champion_proba = float(
            MODEL.booster.predict(X, num_iteration=MODEL.booster.best_iteration)[0]
        )
        champion_decision = "BLOCK" if champion_proba >= MODEL.threshold else "APPROVE"

        # Challenger (shadow-scored if loaded)
        challenger_proba = _challenger_score(tx, velocity)
        challenger_decision: str | None = None
        if challenger_proba is not None and CHALLENGER is not None:
            challenger_decision = (
                "BLOCK" if challenger_proba >= CHALLENGER.threshold else "APPROVE"
            )

        # Decide whose answer to serve
        served_by = pick_serving_model(request_id)
        if served_by == "challenger" and challenger_proba is not None:
            served_proba = challenger_proba
            served_decision = challenger_decision  # type: ignore[assignment]
            served_model_name = CHALLENGER.name  # type: ignore[union-attr]
            served_model_version = CHALLENGER.version  # type: ignore[union-attr]
        else:
            served_by = "champion"  # fall back if challenger unavailable
            served_proba = champion_proba
            served_decision = champion_decision
            served_model_name = MODEL.name
            served_model_version = MODEL.version

    FRAUD_PROBABILITY.observe(served_proba)
    PREDICTIONS_TOTAL.labels(decision=served_decision).inc()

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Log both predictions for offline analysis
    if PRED_LOG is not None:
        try:
            PRED_LOG.write(PredictionRecord(
                request_id=request_id,
                ts_unix=time.time(),
                served_by=served_by,
                decision_served=served_decision,
                champion_proba=champion_proba,
                challenger_proba=challenger_proba,
                champion_decision=champion_decision,
                challenger_decision=challenger_decision,
                threshold_used=MODEL.threshold,
                latency_ms=latency_ms,
                tx_type=tx.type,
                tx_amount=tx.amount,
                name_orig=tx.nameOrig,
            ))
        except Exception as e:
            logger.warning(f"Prediction log write failed: {e}")

    served_threshold = (
        CHALLENGER.threshold
        if served_by == "challenger" and CHALLENGER is not None
        else MODEL.threshold
    )
    return PredictionResponse(
        decision=served_decision,
        fraud_probability=round(served_proba, 6),
        threshold_used=served_threshold,
        model_name=served_model_name,
        model_version=served_model_version,
        latency_ms=latency_ms,
        features_used=MODEL.n_features,
        request_id=request_id,
    )


# ============================================================
#  /metrics  — Prometheus text format
# ============================================================
@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================
#  Middleware — add a request-id header to every response
# ============================================================
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response