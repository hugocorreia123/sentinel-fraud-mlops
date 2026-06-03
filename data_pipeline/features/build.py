"""Feature engineering for PaySim transactions.

Three feature families:

  1. Static     — derived from a single row (log-transforms, balance deltas,
                  one-hot encoded transaction type, missing-balance flags).
  2. Velocity   — count of transactions per originator in last 1 / 6 / 24 steps.
                  At training time we compute these in a single rolling pass.
                  At inference time these features come from Redis (Phase 3).
  3. Behavioural — current amount vs originator's recent rolling mean.

All features are computed within each split, so there is no train-val leakage.
Output: parquet files in data/processed/ with `_features` suffix.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

VELOCITY_WINDOWS = [1, 6, 24]  # in steps (PaySim: 1 step = 1 hour)
TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def add_static_features(df: pl.DataFrame) -> pl.DataFrame:
    """Static features — depend only on the current row."""
    return df.with_columns(
        # Log amount compresses the heavy tail typical of fraud amounts.
        log_amount=(pl.col("amount") + 1).log(),

        # Balance deltas: real change in originator/destination balances.
        # Fraudulent CASH_OUTs often drain accounts to zero.
        orig_balance_delta=(pl.col("oldbalanceOrg") - pl.col("newbalanceOrig")),
        dest_balance_delta=(pl.col("newbalanceDest") - pl.col("oldbalanceDest")),

        # Suspicious flag: balance did not change the expected amount.
        # If oldbalanceOrg - amount != newbalanceOrig, something is off.
        balance_mismatch=((pl.col("oldbalanceOrg") - pl.col("amount")
                          - pl.col("newbalanceOrig")).abs() > 0.01).cast(pl.Int8),

        # "Drained to zero" flag — common fraud signature on TRANSFER+CASH_OUT.
        drained_to_zero=((pl.col("newbalanceOrig") == 0)
                         & (pl.col("oldbalanceOrg") > 0)).cast(pl.Int8),

        # Missing-balance flags — dest balance is 0 for merchant accounts.
        dest_was_empty=(pl.col("oldbalanceDest") == 0).cast(pl.Int8),

        # Ratio of amount to originator balance — high values are riskier.
        amount_to_balance_ratio=pl.when(pl.col("oldbalanceOrg") > 0)
            .then(pl.col("amount") / pl.col("oldbalanceOrg"))
            .otherwise(0.0),

        # Destination is a merchant (account name starts with "M" in PaySim).
        dest_is_merchant=pl.col("nameDest").str.starts_with("M").cast(pl.Int8),
    ).with_columns(
        # One-hot transaction type — separate column per type for tree models.
        [(pl.col("type") == t).cast(pl.Int8).alias(f"type_{t}") for t in TX_TYPES]
    )


def add_velocity_features(df: pl.DataFrame) -> pl.DataFrame:
    """Per-originator transaction counts in sliding windows of N steps."""
    df = df.sort(["nameOrig", "step"])

    for window in VELOCITY_WINDOWS:
        df = df.with_columns(
            # Count of prior transactions by this originator in the last `window` steps.
            # Uses a self-join trick: for each row, count earlier rows by same nameOrig
            # whose step is within (current_step - window, current_step].
            pl.col("step").rolling_max(window_size=window).over("nameOrig").alias(
                f"_dummy_{window}"  # placeholder, replaced below
            )
        )

    # Velocity: for each row, count transactions by nameOrig where step is in
    # (current.step - window, current.step]. We compute via window function.
    for window in VELOCITY_WINDOWS:
        df = df.with_columns(
            velocity_count=pl.col("step")
                .map_elements(lambda s: 1, return_dtype=pl.Int32)
                .cum_sum()
                .over("nameOrig")
                .alias(f"velocity_count_{window}h")
        )

    # Cleaner approach: explicit windowed count via group_by + join would be O(N²);
    # for ~6M rows, use the cumulative trick: count of prior tx by same originator,
    # then subtract count from `window` steps ago. We approximate with rank-based
    # counts since PaySim has 1-hour granularity.

    # Simpler, correct version using join:
    for w in VELOCITY_WINDOWS:
        velocity = (
            df.group_by(["nameOrig", "step"])
            .agg(tx_in_step=pl.len())
            .sort(["nameOrig", "step"])
            .with_columns(
                cumulative=pl.col("tx_in_step").cum_sum().over("nameOrig"),
            )
        )
        # For each (nameOrig, step), velocity_w = cumulative_at_step - cumulative_at_(step - w)
        # Implement via a self-join on the offset.
        velocity_offset = velocity.select(
            pl.col("nameOrig"),
            (pl.col("step") + w).alias("step"),
            pl.col("cumulative").alias("prior_cumulative"),
        )
        velocity = velocity.join(
            velocity_offset, on=["nameOrig", "step"], how="left"
        ).with_columns(
            (pl.col("cumulative") - pl.col("prior_cumulative").fill_null(0))
                .alias(f"velocity_count_{w}h")
        ).select("nameOrig", "step", f"velocity_count_{w}h")

        df = df.join(velocity, on=["nameOrig", "step"], how="left")

    # Drop placeholder columns
    df = df.drop([c for c in df.columns if c.startswith("_dummy_")])
    df = df.drop("velocity_count")

    return df


def add_behavioural_features(df: pl.DataFrame) -> pl.DataFrame:
    """Behavioural features — current amount vs originator's history."""
    df = df.sort(["nameOrig", "step"])
    # Rolling mean amount per originator over their last 5 transactions.
    df = df.with_columns(
        rolling_mean_amount=pl.col("amount")
            .rolling_mean(window_size=5, min_samples=1)
            .over("nameOrig")
    )
    df = df.with_columns(
        amount_vs_rolling_mean=pl.when(pl.col("rolling_mean_amount") > 0)
            .then(pl.col("amount") / pl.col("rolling_mean_amount"))
            .otherwise(1.0)
    )
    return df


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_static_features(df)
    df = add_velocity_features(df)
    df = add_behavioural_features(df)
    # Drop columns we don't want the model to use directly.
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "type"]
    df = df.drop([c for c in drop_cols if c in df.columns])
    return df


def main() -> int:
    for split in ["train", "val", "holdout"]:
        in_path = PROCESSED_DIR / f"paysim_{split}.parquet"
        out_path = PROCESSED_DIR / f"paysim_{split}_features.parquet"

        logger.info(f"Building features for {split}: {in_path.name}")
        df = pl.read_parquet(in_path)
        logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        df = build_features(df)

        df.write_parquet(out_path, compression="zstd", compression_level=3)
        logger.success(
            f"  Wrote {out_path.name}: {len(df):,} rows, "
            f"{len(df.columns)} columns, "
            f"{out_path.stat().st_size / (1024**2):.1f} MB"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())