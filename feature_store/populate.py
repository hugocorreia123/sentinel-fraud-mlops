"""Populate Redis with per-originator velocity features.

Reads the processed training-window data, computes the velocity counters
(transactions per originator in last 1h / 6h / 24h) as of the holdout's
start time, and writes them to Redis. The FastAPI inference service in
Phase 3.2 reads these on every request.

Key schema:
    sentinel:velocity:{nameOrig}  -> Hash with fields v1h, v6h, v24h

Usage:
    uv run python -m feature_store.populate
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import redis
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
KEY_PREFIX = "sentinel:velocity"
TTL_SECONDS = 7 * 24 * 3600  # 7 days

# Cutoff step for the snapshot: end of val window, i.e. just before holdout begins.
# At inference time we treat this as "now".
CUTOFF_STEP = 378

VELOCITY_WINDOWS = [1, 6, 24]


def compute_velocity_snapshot(cutoff_step: int) -> pl.DataFrame:
    """For each nameOrig, count tx in (cutoff_step - window, cutoff_step]."""
    # Read both train and val — together they contain everything up to cutoff.
    train = pl.read_parquet(PROCESSED_DIR / "paysim_train.parquet")
    val = pl.read_parquet(PROCESSED_DIR / "paysim_val.parquet")
    df = pl.concat([train, val]).filter(pl.col("step") < cutoff_step)
    logger.info(f"Computing velocity from {len(df):,} rows up to step {cutoff_step}")

    # Per-originator counts per window
    out = df.select("nameOrig").unique()
    for w in VELOCITY_WINDOWS:
        counts = (
            df.filter(pl.col("step") > cutoff_step - w)
            .group_by("nameOrig")
            .agg(pl.len().alias(f"v{w}h"))
        )
        out = out.join(counts, on="nameOrig", how="left").with_columns(
            pl.col(f"v{w}h").fill_null(0).cast(pl.Int32)
        )
    return out


def populate(df: pl.DataFrame) -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()  # crash early if Redis isn't reachable

    pipe = r.pipeline(transaction=False)
    written = 0
    batch_size = 5000

    for row in df.iter_rows(named=True):
        key = f"{KEY_PREFIX}:{row['nameOrig']}"
        pipe.hset(key, mapping={
            "v1h": int(row["v1h"]),
            "v6h": int(row["v6h"]),
            "v24h": int(row["v24h"]),
        })
        pipe.expire(key, TTL_SECONDS)
        written += 1
        if written % batch_size == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
            logger.info(f"  Written {written:,} keys…")

    pipe.execute()
    logger.success(f"Populated Redis with {written:,} originator velocity keys")

    # Verify a couple of keys — pick from the END (most recent writes)
    sample = df.tail(3)
    for row in sample.iter_rows(named=True):
        key = f"{KEY_PREFIX}:{row['nameOrig']}"
        stored = r.hgetall(key)
        expected = {"v1h": row["v1h"], "v6h": row["v6h"], "v24h": row["v24h"]}
        logger.info(f"  Sample {key}: stored={stored}, expected={expected}")


def main() -> int:
    snapshot = compute_velocity_snapshot(CUTOFF_STEP)
    logger.info(f"Snapshot shape: {snapshot.shape}")
    populate(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
