"""SQLite-backed log of every prediction made by the service.

Used by the analysis script (`scripts/analyze_shadow.py`) to compare
champion vs challenger predictions, compute agreement rate, and assess
statistical significance of any disagreement.

The log is asynchronous-friendly: writes use a single connection guarded
by a thread lock, so concurrent uvicorn requests don't corrupt the DB.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id           TEXT PRIMARY KEY,
    ts_unix              REAL NOT NULL,
    served_by            TEXT NOT NULL,           -- 'champion' | 'challenger'
    decision_served      TEXT NOT NULL,           -- 'APPROVE'  | 'BLOCK'
    champion_proba       REAL,
    challenger_proba     REAL,
    champion_decision    TEXT,
    challenger_decision  TEXT,
    threshold_used       REAL NOT NULL,
    latency_ms           REAL,
    tx_type              TEXT,
    tx_amount            REAL,
    name_orig            TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts_unix);
CREATE INDEX IF NOT EXISTS idx_predictions_served_by ON predictions(served_by);
"""


@dataclass
class PredictionRecord:
    request_id: str
    ts_unix: float
    served_by: str
    decision_served: str
    champion_proba: float | None
    challenger_proba: float | None
    champion_decision: str | None
    challenger_decision: str | None
    threshold_used: float
    latency_ms: float
    tx_type: str
    tx_amount: float
    name_orig: str


class PredictionLog:
    """Thread-safe SQLite logger for predictions."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path), check_same_thread=False, isolation_level=None
        )
        # WAL improves concurrent read+write performance significantly
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        logger.info(f"Prediction log opened at {db_path}")

    def write(self, record: PredictionRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    request_id, ts_unix, served_by, decision_served,
                    champion_proba, challenger_proba,
                    champion_decision, challenger_decision,
                    threshold_used, latency_ms,
                    tx_type, tx_amount, name_orig
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.request_id, record.ts_unix, record.served_by,
                    record.decision_served, record.champion_proba,
                    record.challenger_proba, record.champion_decision,
                    record.challenger_decision, record.threshold_used,
                    record.latency_ms, record.tx_type, record.tx_amount,
                    record.name_orig,
                ),
            )

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM predictions")
            return cur.fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()