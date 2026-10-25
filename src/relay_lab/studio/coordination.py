"""Durable advisory deadlines and measured maintenance beside the task journal.

Robot authority remains in TaskService. A failed projection never changes the
outcome of an already committed task. Its bounded local outbox retries metadata
and measurements only; it never contains a robot command or credential.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager

from relay_lab.spacetime import SpacetimeError, _identifier


def _seconds(value):
    if isinstance(value, dict):
        value = value.get("__timestamp_micros_since_unix_epoch__", 0)
    if isinstance(value, list):
        value = value[0] if value else 0
    return int(value or 0) / 1_000_000


class CoordinationService:
    def __init__(self, studio, registry, *, durable=None):
        self.durable = durable
        self.studio, self.registry = studio, registry
        self.workspace_id = _identifier(registry.registry_id, "workspace")
        self.path = studio.storage / "coordination.sqlite3"
        self.stop = threading.Event()
        self.wakeup = threading.Event()
        self.thread = None
        self.flush_lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.available = False
        self.error = "Coordination has not connected yet"
        self.write_error = None
        self.last_success_at = None
        self.dropped_measurements = 0
        self._defaults_ready = False
        self._reconciled = False
        self._reconciliation_generation = 0
        self._revoked_after = ""
        self.workers = None
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS pending (
                key TEXT PRIMARY KEY, kind TEXT NOT NULL, revision INTEGER NOT NULL,
                payload TEXT NOT NULL, pending INTEGER NOT NULL, updated REAL NOT NULL
            )""")
        self.path.chmod(0o600)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _client(self):
        if self.stop.is_set():
            raise SpacetimeError("Coordination is shutting down")
        return self.studio.factory().client

    def _rows(self, table, limit=256):
        # Table names are internal constants; workspace identities are validated.
        return self._client()._sql(
            f"SELECT * FROM {table} WHERE workspace_id = '{self.workspace_id}' LIMIT {limit}"
        )

    def _call(self, reducer, args):
        self._client()._request(
            f"/call/{reducer}?confirmed=true",
            json.dumps(args, separators=(",", ":"), allow_nan=False).encode(),
            content_type="application/json",
        )

    def _healthy(self, *, read_only=False):
        with self.state_lock:
            if not read_only:
                self.write_error = None
            self.available, self.error, self.last_success_at = True, self.write_error, time.time()

    def _failed(self, *, read_only=False):
        with self.state_lock:
            self.available = False
            self.error = "Spacetime coordination is unavailable; pending metadata will retry"
            if not read_only:
                self.write_error = self.error

    def _needs_reconciliation(self):
        # A failed enqueue may race an already running repair. The generation
        # prevents that older repair from acknowledging work it never observed.
        with self.state_lock:
            self._reconciliation_generation += 1
            self._reconciled = False
            self._revoked_after = ""
        self.wakeup.set()

    def _queue(self, key, kind, payload, revision=0):
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._db() as db:
            # Serialize read/compare/write across request and reconciliation threads.
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM pending WHERE key=?", (key,)).fetchone()
            if old and (old["revision"] > revision or old["payload"] == body):
                return
            capacity = 10000 if kind == "studio_record_metric" else 12000
            if old is None and db.execute("SELECT count(*) FROM pending").fetchone()[0] >= capacity:
                # Metrics may be omitted visibly; no task commit is rolled back.
                raise ValueError("Coordination outbox capacity reached")
            db.execute(
                "INSERT INTO pending VALUES (?,?,?,?,1,?) ON CONFLICT(key) DO UPDATE SET "
                "kind=excluded.kind,revision=excluded.revision,payload=excluded.payload,"
                "pending=1,updated=excluded.updated",
                (key, kind, revision, body, time.time()),
            )
        self.wakeup.set()

    def observe_task(self, goal, *, metrics=False):
        try:
            task_id = _identifier(goal["id"], "task")
            self._queue(
                "task-" + task_id,
                "studio_sync_task",
                [
                    self.workspace_id,
                    task_id,
                    goal["revision"],
                    goal["status"],
                    goal.get("timeout_seconds", 0),
                ],
                goal["revision"],
            )
            if metrics:
                if goal["revision"] == 0:
                    self.record_metric(task_id + "-created", "task_created", 1)
                terminal = {
                    "succeeded": "task_completed",
                    "failed": "task_failed",
                    "cancelled": "task_cancelled",
                }.get(goal["status"])
                if terminal:
                    self.record_metric(task_id + "-terminal", terminal, 1)
            return True
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
            self._needs_reconciliation()
            self._failed()
            return False

    def record_metric(self, event_id, metric, value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return False
        event_id = hashlib.sha256(event_id.encode()).hexdigest()
        try:
            self._queue(
                "metric-" + event_id,
                "studio_record_metric",
                [self.workspace_id, event_id, metric, float(value)],
            )
            return True
        except (OSError, sqlite3.Error, ValueError):
            self.dropped_measurements += 1
            self._failed()
            return False

    def record_exchange(self, session_id, result):
        if result.get("recovered") or not result.get("readback_verified"):
            return
        for source, metric in (
            ("elapsed_ms", "exchange_latency_ms"),
            ("database_ms", "database_latency_ms"),
            ("llm_ms", "llm_latency_ms"),
            ("bytes", "exchange_bytes"),
        ):
            self.record_metric(
                f"{session_id}-{result['exchange']}-{source}", metric, result.get(source)
            )

    def heartbeat(self, worker, lease_seconds=45):
        try:
            self._call(
                "studio_heartbeat",
                [self.workspace_id, worker["id"], worker["agent_id"], lease_seconds],
            )
            rows = self._rows("studio_presence")
            found = next((r for r in rows if r["worker_id"] == worker["id"]), None)
            if not found or found["status"] != "online":
                raise SpacetimeError("Worker presence readback failed")
            self._healthy(read_only=True)
        except Exception:
            self._failed()
            raise

    def revoke(self, worker_id):
        try:
            worker_id = _identifier(worker_id, "worker")
            self._queue(
                "revoke-" + worker_id, "studio_revoke_worker", [self.workspace_id, worker_id]
            )
            return True
        except (OSError, sqlite3.Error, ValueError):
            self._needs_reconciliation()
            self._failed()
            return False

    def _defaults(self):
        # Manager updates use the same governance boundary: startup may create
        # missing jobs but must never overwrite a concurrently paused job.
        while not self.stop.is_set():
            if self.registry.lock.acquire(timeout=0.2):
                break
        else:
            return
        try:
            jobs = {j["job_id"] for j in self._rows("studio_job", 8)}
            for job, schedule in (("health", "hourly"), ("rollup", "hourly"), ("cleanup", "daily")):
                if job not in jobs:
                    self._call("studio_configure_job", [self.workspace_id, job, True, schedule, 7])
            self._defaults_ready = True
        finally:
            self.registry.lock.release()

    def _reconcile(self):
        # Repair projection after a crash between the authoritative commit and
        # enqueue, without inventing historical measurements.
        with self.state_lock:
            generation = self._reconciliation_generation
            revoked_after = self._revoked_after
        owner = (
            "studio-goals-"
            + hashlib.sha256(
                json.dumps(self.workspace_id, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:12]
        )
        rows = self._client()._sql(
            f"SELECT run_id FROM factory_run WHERE worker_id = '{owner}' LIMIT 100"
        )
        for row in rows:
            if self.stop.is_set():
                return
            task_id = _identifier(row["run_id"], "task")
            found = self._client()._sql(
                f"SELECT state_json, revision FROM factory_run WHERE run_id = '{task_id}'"
            )
            if not found:
                continue
            state = json.loads(found[0]["state_json"])
            if (
                state.get("registry_id") == self.workspace_id
                and state.get("task") == "studio-objective-v1"
                and not self.observe_task({**state["goal"], "revision": found[0]["revision"]})
            ):
                return
        # Local credential revocation commits before its optional native mirror.
        # Read only bounded ID pages: tokens and full worker records never enter
        # the coordination outbox. Advance between flushes so large old histories
        # cannot monopolize a pass or grow the outbox without draining it.
        revoked = self.workers.revoked_ids(after=revoked_after, limit=100) if self.workers else []
        for worker_id in revoked:
            if self.stop.is_set() or not self.revoke(worker_id):
                return
        with self.state_lock:
            if generation == self._reconciliation_generation:
                self._revoked_after = revoked[-1] if len(revoked) == 100 else ""
                self._reconciled = len(revoked) < 100

    def flush(self, limit=32):
        # The outbox is part of the Spacetime checkpoint on ephemeral hosts.
        # Serialize the whole background pass with requests and stream auth.
        if self.stop.is_set():
            return
        if self.durable is not None and not self.durable.holder:
            # Never wait indefinitely behind a foreground request during shutdown.
            # DurableState uses an RLock, so its request can re-enter once acquired.
            while not self.stop.is_set():
                if self.durable.lock.acquire(timeout=0.2):
                    break
            else:
                return
            try:
                if self.stop.is_set():
                    return
                with self.durable.request():
                    self._flush(limit)
                    if self.stop.is_set():
                        # Abort the checkpoint; DurableState only releases its lease.
                        raise SpacetimeError("Coordination is shutting down")
            except Exception:  # noqa: BLE001 - the pending checkpoint remains authoritative
                self._failed()
            finally:
                self.durable.lock.release()
            return
        self._flush(limit)

    def _flush(self, limit=32):
        if not self.flush_lock.acquire(blocking=False):
            return
        try:
            if not self._defaults_ready:
                self._defaults()
            if not self._reconciled:
                self._reconcile()
            with self._db() as db:
                rows = db.execute(
                    "SELECT * FROM pending WHERE pending=1 "
                    "ORDER BY CASE WHEN kind='studio_record_metric' THEN 1 ELSE 0 END,updated LIMIT ?",
                    (limit,),
                ).fetchall()
            for row in rows:
                if self.stop.is_set():
                    return
                self._call(row["kind"], json.loads(row["payload"]))
                with self._db() as db:
                    db.execute(
                        "UPDATE pending SET pending=0 WHERE key=? AND payload=?",
                        (row["key"], row["payload"]),
                    )
            with self._db() as db:
                db.execute(
                    "DELETE FROM pending WHERE kind='studio_record_metric' "
                    "AND pending=0 AND updated<?",
                    (time.time() - 7 * 86400,),
                )
            self._healthy(read_only=not self._reconciled)
        except Exception:  # noqa: BLE001 - preserve metadata and redact transport errors
            self._failed()
        finally:
            self.flush_lock.release()

    def _run(self):
        while not self.stop.is_set():
            self.wakeup.clear()
            self.flush()
            self.wakeup.wait(10)

    def start(self):
        with self.state_lock:
            if self.thread is not None:
                return
            self.thread = threading.Thread(
                target=self._run, name="studio-coordination", daemon=True
            )
            self.thread.start()

    def request_stop(self):
        self.stop.set()
        self.wakeup.set()

    def close(self):
        self.request_stop()
        if self.thread is not None:
            # Each HTTP operation is bounded. Stop checks prevent beginning
            # another request after shutdown, including during reconciliation.
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise RuntimeError("Coordination writer did not finish shutdown")

    def configure_job(self, value):
        job = value.get("job_id")
        schedule = value.get("schedule")
        enabled = value.get("enabled")
        if job not in ("health", "rollup", "cleanup") or schedule not in ("hourly", "daily"):
            raise ValueError("Choose health, rollup or cleanup, scheduled hourly or daily in UTC")
        days = value.get("retention_days")
        if type(enabled) is not bool or (
            days is not None and (type(days) is not int or not 1 <= days <= 90)
        ):
            raise ValueError("Enabled must be a boolean and retention must be 1–90 days")
        with self.registry.lock:
            if days is None:
                previous = next(
                    (j for j in self._rows("studio_job", 8) if j["job_id"] == job), None
                )
                days = previous["retention_days"] if previous else 7
            self._call("studio_configure_job", [self.workspace_id, job, enabled, schedule, days])
            rows = self._rows("studio_job", 8)
            row = next((j for j in rows if j["job_id"] == job), None)
            if row is None or any(
                row[k] != v
                for k, v in (("enabled", enabled), ("schedule", schedule), ("retention_days", days))
            ):
                raise SpacetimeError("Scheduled job configuration readback failed")
        return self.snapshot()

    def run_job(self, value):
        job = value.get("job_id")
        if job not in ("health", "rollup", "cleanup"):
            raise ValueError("Unknown maintenance job")
        # No automatic retry on an uncertain manual invocation.
        self._call("studio_run_job", [self.workspace_id, job])
        return self.snapshot()

    def snapshot(self, realtime=None):
        result = {
            "enabled": True,
            "available": False,
            "presence": [],
            "attention": [],
            "jobs": [],
            "rollups": [],
            "timezone": "UTC",
            "deadline_behavior": "advisory_only",
            "dropped_measurements": self.dropped_measurements,
        }
        try:
            result["presence"] = [
                dict(
                    r,
                    expires_at=_seconds(r["lease_expires_us"]),
                    last_seen_at=_seconds(r["updated_at"]),
                )
                for r in self._rows("studio_presence")
            ]
            result["attention"] = [
                dict(r, created_at=_seconds(r["created_at"]), updated_at=_seconds(r["updated_at"]))
                for r in self._rows("studio_attention", 1024)
            ]
            result["attention"].sort(key=lambda r: r["updated_at"], reverse=True)
            result["attention"] = result["attention"][:100]
            result["jobs"] = [
                dict(
                    r,
                    label={
                        "health": "Health checks",
                        "rollup": "Hourly metrics",
                        "cleanup": "Retention cleanup",
                    }[r["job_id"]],
                    next_run_at=_seconds(r["next_run_us"]),
                    last_run_at=_seconds(r["last_run_us"]),
                    last_status="error"
                    if r["last_error"]
                    else "completed"
                    if r["run_count"]
                    else "scheduled",
                )
                for r in self._rows("studio_job", 8)
            ]
            # SQL does not support ORDER BY. Project the small numeric summary
            # before sorting all bounded rows; full keys/workspace IDs could
            # exceed the HTTP client's response cap at the 8192-row limit.
            result["rollups"] = self._client()._sql(
                "SELECT metric,bucket_us,sample_count,total,minimum,maximum "
                f"FROM studio_rollup WHERE workspace_id = '{self.workspace_id}' LIMIT 8192"
            )
            result["rollups"].sort(key=lambda r: int(r["bucket_us"]), reverse=True)
            result["rollups"] = result["rollups"][:100]
            self._healthy(read_only=True)
        except Exception:  # noqa: BLE001 - keep credentials and DB diagnostics server-side
            self._failed(read_only=True)
        try:
            with self._db() as db:
                result["pending_updates"] = db.execute(
                    "SELECT count(*) FROM pending WHERE pending=1"
                ).fetchone()[0]
        except (OSError, sqlite3.Error):
            result["pending_updates"] = None
            self._failed()
        with self.state_lock:
            result.update(
                available=self.available, error=self.error, last_success_at=self.last_success_at
            )
        result["realtime"] = realtime.status() if realtime is not None else {"state": "unavailable"}
        return result
