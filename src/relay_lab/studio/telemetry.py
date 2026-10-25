"""Local, payload-free exchange measurements. Spacetime remains the task journal.

Timing becomes known after acknowledgement, so it cannot be in that transaction.
This separate measurement log survives Studio restarts and deduplicates retries.
Losing it must not turn a committed robot message into an apparent failed action.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class TelemetryStore:
    def __init__(self, storage):
        self.path = Path(storage) / "telemetry.sqlite3"
        self.lock = threading.RLock()
        self.unavailable_reason = None
        try:
            with self._connect() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS studio_exchange (
                    session_id TEXT NOT NULL, exchange INTEGER NOT NULL,
                    recorded_at REAL NOT NULL, mode TEXT NOT NULL,
                    direction TEXT NOT NULL, source TEXT NOT NULL,
                    elapsed_ms REAL, database_ms REAL, llm_ms REAL,
                    bytes INTEGER NOT NULL, readback_verified INTEGER NOT NULL,
                    actions_executed INTEGER NOT NULL,
                    PRIMARY KEY (session_id, exchange))""")
                db.execute(
                    "CREATE INDEX IF NOT EXISTS exchange_time ON studio_exchange(recorded_at)"
                )
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error):
            self.unavailable_reason = "Local measurement storage is unavailable"

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _timing(value):
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value)
        return None

    def record(self, session_id, direction, result):
        # Recovered results have no original after-ACK measurements. Never
        # relabel recovery/retry latency as execution latency.
        if result.get("recovered") or not result.get("readback_verified"):
            return False
        row = (
            session_id,
            result["exchange"],
            time.time(),
            result["mode"],
            direction,
            result["source"],
            *(self._timing(result.get(key)) for key in ("elapsed_ms", "database_ms", "llm_ms")),
            result["bytes"],
            1,
            int(result.get("actions_executed") is True),
        )
        try:
            with self.lock, self._connect() as db:
                written = db.execute(
                    "INSERT OR IGNORE INTO studio_exchange VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                ).rowcount
            self.unavailable_reason = None
            return bool(written)
        except (OSError, sqlite3.Error):
            self.unavailable_reason = "Some committed exchanges could not be measured locally"
            return False

    def snapshot(self, limit=500):
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("Telemetry window must be 1–500 exchanges")
        rows, total = [], 0
        try:
            with self.lock, self._connect() as db:
                total = db.execute("SELECT count(*) FROM studio_exchange").fetchone()[0]
                rows = [
                    dict(r)
                    for r in db.execute(
                        "SELECT * FROM studio_exchange ORDER BY recorded_at DESC, session_id, exchange DESC LIMIT ?",
                        (limit,),
                    )
                ]
            for row in rows:
                for key in ("readback_verified", "actions_executed"):
                    row[key] = bool(row[key])
        except (OSError, sqlite3.Error):
            self.unavailable_reason = "Local measurement storage is unavailable"
        return {
            "rows": rows,
            "total_rows": total,
            "window_limit": limit,
            "unavailable_reason": self.unavailable_reason,
        }
