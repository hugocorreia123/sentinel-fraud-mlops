"""Pydantic schemas for the inference API.

These are the contract between the inference service and any client.
Validated at every request; clear errors are returned for malformed input.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class TransactionRequest(BaseModel):
    """One PaySim-shaped transaction submitted for scoring."""
    step: int = Field(..., ge=0, le=10_000, description="Hour of simulation")
    type: TransactionType
    amount: float = Field(..., ge=0)
    nameOrig: str = Field(..., min_length=1, max_length=64,
                          description="Originator account ID")
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str = Field(..., min_length=1, max_length=64,
                          description="Destination account ID")
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 500,
                "type": "TRANSFER",
                "amount": 5000.0,
                "nameOrig": "C1234567890",
                "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C0987654321",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 5000.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """The decision returned for one transaction."""
    decision: Literal["APPROVE", "BLOCK"]
    fraud_probability: float = Field(..., ge=0, le=1)
    threshold_used: float = Field(..., ge=0, le=1)
    model_name: str
    model_version: str
    latency_ms: float
    features_used: int
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    redis_reachable: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    stage: str
    framework: str
    n_features: int
    feature_names: list[str]
    threshold: float
    loaded_at: str