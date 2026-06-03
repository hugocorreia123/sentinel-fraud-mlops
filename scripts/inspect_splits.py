"""Inspect the temporal distribution of PaySim. Used to calibrate split boundaries.

Run this whenever the underlying dataset changes, to verify the TRAIN_END / VAL_END
constants in data_pipeline/ingestion/load.py still produce a ~70/15/15 split.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> int:
    df = pl.scan_parquet(str(PROCESSED_DIR / "paysim_*.parquet")).collect()

    by_step = (
        df.group_by("step")
        .agg(rows=pl.len(), fraud=pl.col("isFraud").sum())
        .sort("step")
        .with_columns(
            cum_rows=pl.col("rows").cum_sum(),
            cum_pct=(pl.col("rows").cum_sum() / pl.col("rows").sum() * 100),
        )
    )

    print(f"Total rows:  {len(df):,}")
    print(f"Total fraud: {int(df['isFraud'].sum()):,}")
    print()

    for threshold in (70, 85):
        crossing = by_step.filter(pl.col("cum_pct") >= threshold).head(1)
        if not crossing.is_empty():
            row = crossing.row(0, named=True)
            print(
                f"  {threshold}% boundary: step={row['step']:>3}  "
                f"({row['cum_rows']:>10,} cumulative rows, {row['cum_pct']:.2f}%)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())