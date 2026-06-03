"""Load and split PaySim into train/val/holdout with strict temporal order.

The dataset's `step` column is hour-of-simulation (0 to ~744 for ~31 days).
We split on step ranges so the model is never trained on data that occurs
*after* what it's being evaluated on. This is the single most important
discipline in fraud modelling — random splits silently leak future signal.

Splits:
    train    : step in [0,   500)   -> ~21 days,  first 80%
    val      : step in [500, 620)   -> ~5 days,   next 15%
    holdout  : step in [620, 744]   -> ~5 days,   last 5%

The val set is for hyperparameter tuning and threshold selection.
The holdout set is the final unbiased measurement, touched ONCE at the end.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "paysim.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Polars schema. Explicit > inferred. Errors loudly if PaySim changes shape.
SCHEMA: dict[str, pl.DataType] = {
    "step": pl.Int32,
    "type": pl.Categorical,
    "amount": pl.Float64,
    "nameOrig": pl.Utf8,
    "oldbalanceOrg": pl.Float64,
    "newbalanceOrig": pl.Float64,
    "nameDest": pl.Utf8,
    "oldbalanceDest": pl.Float64,
    "newbalanceDest": pl.Float64,
    "isFraud": pl.Int8,
    "isFlaggedFraud": pl.Int8,
}

# Empirically calibrated for ~70/15/15 row split.
# PaySim's `step` is not uniformly distributed: bursty hours mean a naive
# step-range split produces highly imbalanced row counts. These boundaries
# come from inspecting cumulative row counts (see scripts/inspect_splits.py).
TRAIN_END = 323   # train: step in [0, 323)    -> ~70% of rows
VAL_END = 378     # val:   step in [323, 378)  -> ~15% of rows
# holdout: step in [378, max]                  -> ~15% of rows


@dataclass(frozen=True)
class SplitStats:
    name: str
    rows: int
    fraud_rows: int
    fraud_rate: float
    step_min: int
    step_max: int


def load_raw() -> pl.DataFrame:
    """Read the raw CSV with explicit schema."""
    logger.info(f"Loading {RAW_CSV}")
    df = pl.read_csv(RAW_CSV, schema=SCHEMA)
    logger.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")
    return df


def split_temporal(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split by `step` ranges. Strict temporal order, no overlap, no random shuffle."""
    splits = {
        "train":   df.filter(pl.col("step") < TRAIN_END),
        "val":     df.filter((pl.col("step") >= TRAIN_END) & (pl.col("step") < VAL_END)),
        "holdout": df.filter(pl.col("step") >= VAL_END),
    }

    # Verify the split is exhaustive and non-overlapping.
    total = sum(len(s) for s in splits.values())
    assert total == len(df), f"Split lost rows: {total} != {len(df)}"

    return splits


def summarise(name: str, df: pl.DataFrame) -> SplitStats:
    fraud_rows = int(df["isFraud"].sum())
    return SplitStats(
        name=name,
        rows=len(df),
        fraud_rows=fraud_rows,
        fraud_rate=fraud_rows / len(df) if len(df) else 0.0,
        step_min=int(df["step"].min()),
        step_max=int(df["step"].max()),
    )


def write_splits(splits: dict[str, pl.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        out_path = PROCESSED_DIR / f"paysim_{name}.parquet"
        df.write_parquet(out_path, compression="zstd", compression_level=3)
        logger.success(
            f"Wrote {name:>7} -> {out_path.name}  "
            f"({len(df):>10,} rows, {out_path.stat().st_size / (1024**2):.1f} MB)"
        )


def main() -> int:
    df = load_raw()
    splits = split_temporal(df)

    # Diagnostic report
    print()
    print(f"{'Split':<10} {'Rows':>12} {'Fraud':>10} {'Rate':>8}  {'Step range':>14}")
    print("-" * 60)
    for name in ["train", "val", "holdout"]:
        stats = summarise(name, splits[name])
        print(
            f"{stats.name:<10} {stats.rows:>12,} {stats.fraud_rows:>10,} "
            f"{stats.fraud_rate * 100:>7.3f}%  "
            f"[{stats.step_min:>3}, {stats.step_max:>3}]"
        )
    print()

    write_splits(splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())