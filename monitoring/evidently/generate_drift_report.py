"""Generate an Evidently data-drift report.

Compares the live prediction log (data/predictions.db) against the training
reference distribution (data/processed/paysim_train_features.parquet).

Saves an HTML report and a JSON summary to monitoring/evidently/reports/.

Usage:
    uv run python -m monitoring.evidently.generate_drift_report
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PARQUET = PROJECT_ROOT / "data" / "processed" / "paysim_train_features.parquet"
PRED_DB = PROJECT_ROOT / "data" / "predictions.db"
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "evidently" / "reports"

# Features we monitor for drift. Excludes target, type one-hots (categorical handled
# via tx_type from the prediction log), and velocity (low signal on PaySim).
DRIFT_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "log_amount", "orig_balance_delta", "dest_balance_delta",
    "balance_mismatch", "drained_to_zero", "dest_was_empty",
    "amount_to_balance_ratio", "dest_is_merchant",
    "rolling_mean_amount", "amount_vs_rolling_mean",
]


def load_reference() -> pd.DataFrame:
    """Sample of training data — the reference distribution."""
    df = pl.read_parquet(TRAIN_PARQUET)
    # Sample 50k rows; full 4.4M would make the report huge
    sample = df.sample(n=min(50_000, len(df)), seed=42)
    cols = [c for c in DRIFT_FEATURES if c in sample.columns]
    return sample.select(cols).to_pandas()


def load_current() -> pd.DataFrame:
    """Reconstruct feature distribution from the live prediction log.

    The DB only stores raw inputs (tx_type, tx_amount, name_orig). We
    rebuild the engineered features from them to make a fair comparison.
    """
    conn = sqlite3.connect(str(PRED_DB))
    rows = pd.read_sql("SELECT tx_type, tx_amount FROM predictions", conn)
    conn.close()
    if rows.empty:
        raise RuntimeError("No predictions logged yet. Run some traffic first.")

    # Drop in synthetic balances to allow feature engineering. We don't have the
    # original balances in the prediction log (we don't store PII in production
    # either), so reconstructed features will be partial. Drift on `amount` and
    # `log_amount` is the most informative signal anyway.
    df = pd.DataFrame({
        "amount": rows["tx_amount"],
        "log_amount": (rows["tx_amount"] + 1).apply(lambda x: x and (x ** 0)) * 0,  # placeholder
    })
    import numpy as np
    df["log_amount"] = np.log1p(rows["tx_amount"])
    return df


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading reference (training) distribution…")
    ref = load_reference()
    logger.info(f"Reference: {ref.shape}, columns: {list(ref.columns)[:5]}...")

    logger.info("Loading current (live) distribution from prediction log…")
    cur = load_current()
    logger.info(f"Current: {cur.shape}")

    # Keep only the columns we have on both sides
    common = [c for c in cur.columns if c in ref.columns]
    if not common:
        raise RuntimeError("No overlapping columns between reference and current.")
    logger.info(f"Drift will be computed on: {common}")

    ref_aligned = ref[common]
    cur_aligned = cur[common]

    logger.info("Running Evidently drift report…")
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(reference_data=ref_aligned, current_data=cur_aligned)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    html_path = REPORTS_DIR / f"drift_{ts}.html"
    json_path = REPORTS_DIR / f"drift_{ts}.json"
    latest_html = REPORTS_DIR / "drift_latest.html"
    latest_json = REPORTS_DIR / "drift_latest.json"

    snapshot.save_html(str(html_path))
    snapshot_dict = snapshot.dict()
    json_path.write_text(json.dumps(snapshot_dict, indent=2, default=str))

    # Also update "latest" symlinks/copies
    latest_html.write_bytes(html_path.read_bytes())
    latest_json.write_text(json_path.read_text())

    # Try to extract drift summary from the snapshot
    n_drifted = 0
    drifted_features = []
    try:
        for metric in snapshot_dict.get("metrics", []):
            mid = metric.get("metric_id", "")
            if "DriftedColumnsCount" in mid or "drift_share" in str(mid).lower():
                value = metric.get("value", {})
                if isinstance(value, dict):
                    n_drifted = int(value.get("count", value.get("share", 0)))
            if metric.get("drift_detected") or metric.get("status") == "drift":
                drifted_features.append(metric.get("column_name", "?"))
    except Exception as e:
        logger.warning(f"Could not parse drift summary: {e}")

    print()
    print("=" * 70)
    print("Evidently drift report")
    print("=" * 70)
    print(f"  Reference: {ref_aligned.shape[0]:,} rows (training sample)")
    print(f"  Current:   {cur_aligned.shape[0]:,} rows (live predictions)")
    print(f"  Features analyzed: {len(common)}")
    print()
    print(f"  HTML report: {html_path}")
    print(f"  JSON summary: {json_path}")
    print(f"  Latest copies: {latest_html.name} + {latest_json.name}")
    print()
    print(f"  Open in browser:")
    print(f"    open {latest_html}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())