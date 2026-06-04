"""Traffic routing between champion and challenger.

Uses a stable hash of the request_id so that repeated requests with the
same id always land in the same bucket (important for reproducible A/B).

Configured via `CHALLENGER_TRAFFIC_PCT` env var:
  0   -> shadow mode (challenger scores everything but never serves)
  5   -> 5% of traffic served by challenger
  100 -> challenger serves everything
"""
from __future__ import annotations

import hashlib
import os
from typing import Literal

ServedBy = Literal["champion", "challenger"]


def _challenger_pct() -> int:
    """How much traffic the challenger should *serve* (not just shadow-score)."""
    raw = os.environ.get("CHALLENGER_TRAFFIC_PCT", "0")
    try:
        pct = int(raw)
    except ValueError:
        pct = 0
    return max(0, min(100, pct))


def pick_serving_model(request_id: str) -> ServedBy:
    """Decide which model's prediction to return to the client."""
    pct = _challenger_pct()
    if pct <= 0:
        return "champion"
    if pct >= 100:
        return "challenger"
    bucket = int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 100
    return "challenger" if bucket < pct else "champion"