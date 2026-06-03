"""Verify the PaySim dataset is in place.

PaySim is a synthetic mobile-money transactions dataset with ~6.36M rows
and real fraud labels. Industry-standard for fraud-MLOps work because it
has plain English column names (not anonymised) and clear adversarial signal.

Download manually (one-time, ~180 MB zip → 470 MB CSV):
    1. Visit https://www.kaggle.com/datasets/ealaxi/paysim1
    2. Click Download (top right). No phone verification needed for datasets.
    3. unzip ~/Downloads/archive.zip -d data/raw/
    4. mv "data/raw/PS_*.csv" data/raw/paysim.csv

Usage:
    uv run python scripts/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "paysim.csv"


def main() -> int:
    if not CSV_PATH.exists():
        logger.error(f"PaySim CSV not found at {CSV_PATH}")
        logger.error("Download it from https://www.kaggle.com/datasets/ealaxi/paysim1")
        logger.error("and place at data/raw/paysim.csv (see docstring above).")
        return 1

    size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
    logger.success(f"PaySim dataset present: {CSV_PATH} ({size_mb:.1f} MB)")

    with CSV_PATH.open() as f:
        line_count = sum(1 for _ in f)
    logger.success(f"Row count (incl. header): {line_count:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())