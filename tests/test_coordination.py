"""Metadata failures never rewrite task authority or invent measurements."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from relay_lab.spacetime import SpacetimeError
from relay_lab.studio.coordination import CoordinationService, _seconds
from relay_lab.studio.workers import WorkerStore


class Client:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.jobs = []
        self.after_call = None

    def _request(self, path, body, **kwargs):
        if self.fail:
            raise SpacetimeError("private token must not escape")
        args = json.loads(body)
        self.calls.append((path, args))
        if "studio_configure_job" in path:
            self.jobs = [row for row in self.jobs if row["job_id"] != args[1]]
            self.jobs.append(
                {
                    "workspace_id": args[0],
                    "job_id": args[1],
                    "enabled": args[2],
                    "schedule": args[3],
                    "retention_days": args[4],
                    "next_run_us": 0,
                    "last_run_us": 0,
                    "run_count": 0,
                    "last_error": "",
                }
            )
        if self.after_call:
            hook, self.after_call = self.after_call, None
            hook()
        return b""

    def _sql(self, query):
        if self.fail:
            raise SpacetimeError("private token must not escape")
        if "studio_job" in query:
            return self.jobs
        return []


@pytest.fixture
def service(tmp_path):
    client = Client()
    studio = SimpleNamespace(storage=tmp_path, factory=lambda: SimpleNamespace(client=client))
    value = CoordinationService(
        studio, SimpleNamespace(registry_id="registry-unit", lock=threading.RLock())
    )
    return value, client


def goal(revision=0, status="queued"):
    return {
        "id": "goal-0123456789abcdef",
        "revision": revision,
        "status": status,
        "timeout_seconds": 1800,
    }


def test_failed_metadata_flush_survives_restart_without_replaying_actions(service):
    coordinator, client = service
    client.fail = True
    coordinator.observe_task(goal(), metrics=True)
    coordinator.flush()
    status = coordinator.snapshot()
    assert status["pending_updates"] == 2
    assert not status["available"] and "private token" not in json.dumps(status)
    restored = CoordinationService(coordinator.studio, coordinator.registry)
    client.fail = False
    restored.flush()
    assert restored.snapshot()["pending_updates"] == 0
    assert all("claim" not in path and "report" not in path for path, _ in client.calls)


def test_acknowledging_old_outbox_value_does_not_drop_new_revision(service):
    coordinator, client = service
    coordinator._defaults_ready = coordinator._reconciled = True
    coordinator.observe_task(goal())
    client.after_call = lambda: coordinator.observe_task(goal(1, "running"))
    coordinator.flush()
    assert coordinator.snapshot()["pending_updates"] == 1
    coordinator.flush()
    assert [args[2] for path, args in client.calls if "studio_sync_task" in path] == [0, 1]
    assert coordinator.snapshot()["pending_updates"] == 0


def test_old_revision_cannot_replace_new_task_and_metrics_are_deduped(service):
    coordinator, client = service
    coordinator.observe_task(goal(3, "cancelled"), metrics=True)
    coordinator.observe_task(goal(1, "running"))
    coordinator.observe_task(goal(3, "cancelled"), metrics=True)
    coordinator.flush()
    calls = [args for path, args in client.calls if "studio_sync_task" in path]
    assert len(calls) == 1 and calls[0][2:4] == [3, "cancelled"]
    assert len([path for path, _ in client.calls if "studio_record_metric" in path]) == 1


def test_disabled_jobs_stay_disabled_on_restart(service):
    coordinator, client = service
    client.jobs = [{"job_id": name, "enabled": False} for name in ("health", "rollup", "cleanup")]
    coordinator.flush()
    assert not client.calls


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1])
def test_invalid_measurements_are_not_queued(service, value):
    coordinator, client = service
    assert not coordinator.record_metric("event", "exchange_latency_ms", value)
    coordinator.flush()
    assert not any("studio_record_metric" in path for path, _ in client.calls)


def test_recovered_exchanges_have_no_fabricated_latency(service):
    coordinator, client = service
    coordinator.record_exchange("s", {"recovered": True, "readback_verified": True})
    coordinator.record_exchange("s", {"readback_verified": False})
    coordinator.flush()
    assert not any("studio_record_metric" in path for path, _ in client.calls)


def test_heartbeat_requires_confirmed_presence_readback(service):
    coordinator, _ = service
    with pytest.raises(SpacetimeError, match="readback"):
        coordinator.heartbeat({"id": "worker-a", "agent_id": "left"})
    assert not coordinator.available


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": "true"},
        {"schedule": "every-second"},
        {"retention_days": 0},
        {"job_id": "run-robot"},
    ],
)
def test_job_configuration_validation_precedes_write(service, changes):
    coordinator, client = service
    with pytest.raises(ValueError):
        coordinator.configure_job(
            {"job_id": "health", "schedule": "hourly", "enabled": True, **changes}
        )
    assert not client.calls


def test_timestamp_decoding():
    assert _seconds(2_000_000) == 2
    assert _seconds([2_000_000]) == 2
    assert _seconds({"__timestamp_micros_since_unix_epoch__": 2_000_000}) == 2


def test_snapshot_keeps_recent_metrics_after_many_retained_buckets(service):
    coordinator, client = service
    original = client._sql

    def query(sql):
        if "studio_rollup" in sql:
            limit = int(sql.rsplit("LIMIT ", 1)[1])
            return [{"bucket_us": bucket * 1_000_000} for bucket in range(600)][:limit]
        return original(sql)

    client._sql = query
    rows = coordinator.snapshot()["rollups"]
    assert len(rows) == 100
    assert rows[0]["bucket_us"] == 599_000_000
    assert rows[-1]["bucket_us"] == 500_000_000


def test_concurrent_outbox_writers_preserve_highest_revision(service):
    coordinator, _ = service
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda revision: coordinator.observe_task(goal(revision, "running")),
                list(range(100)) + list(reversed(range(100))),
            )
        )
    with coordinator._db() as db:
        row = db.execute("SELECT revision FROM pending").fetchone()
    assert row["revision"] == 99


def test_schedule_save_preserves_existing_retention(service):
    coordinator, client = service
    coordinator.configure_job(
        {"job_id": "cleanup", "schedule": "daily", "enabled": True, "retention_days": 30}
    )
    coordinator.configure_job({"job_id": "cleanup", "schedule": "hourly", "enabled": False})
    assert client.calls[-1][1][2:] == [False, "hourly", 30]


def test_defaults_cannot_overwrite_manager_choice(service):
    coordinator, client = service
    with coordinator.registry.lock:
        coordinator.start()
        coordinator.configure_job({"job_id": "health", "schedule": "daily", "enabled": False})
    coordinator.close()
    assert next(row for row in client.jobs if row["job_id"] == "health")["enabled"] is False


def test_close_interrupts_defaults_waiting_for_governance_lock(service):
    coordinator, _ = service
    with coordinator.registry.lock:
        coordinator.start()
        coordinator.close()
    assert not coordinator.thread.is_alive()


def test_metric_capacity_reserves_space_for_task_metadata(service):
    coordinator, _ = service
    with coordinator._db() as db:
        db.executemany(
            "INSERT INTO pending VALUES (?, 'studio_record_metric', 0, '[]', 1, 0)",
            [(f"metric-{i}",) for i in range(10000)],
        )
    assert not coordinator.record_metric("another", "exchange_latency_ms", 1)
    coordinator.observe_task(goal())
    with coordinator._db() as db:
        assert (
            db.execute("SELECT count(*) FROM pending WHERE kind='studio_sync_task'").fetchone()[0]
            == 1
        )


def test_successful_snapshot_does_not_hide_failed_metadata_write(service):
    coordinator, client = service
    coordinator._defaults_ready = coordinator._reconciled = True
    coordinator.observe_task(goal())

    def fail(*args, **kwargs):
        raise SpacetimeError("private message")

    client._request = fail
    coordinator.flush()
    snapshot = coordinator.snapshot()
    assert snapshot["available"] and snapshot["pending_updates"] == 1
    assert "will retry" in snapshot["error"] and "private" not in json.dumps(snapshot)


def worker_store(coordinator, rows):
    access = SimpleNamespace(authorize_actor=lambda actor, *_: actor, audit=lambda *_: None)
    workers = WorkerStore(coordinator.studio.storage, access, coordinator.registry, coordinator.studio)
    with workers._db() as db:
        for worker_id, active, workspace in rows:
            db.execute(
                "INSERT INTO runtime_worker VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (worker_id, "Test runtime", "left", "model-a", 1, workspace, "manager",
                 "private-hash-" + worker_id, 1, 2, None, active, "", "simulation"),
            )
    return workers


def test_restart_repairs_revocation_committed_before_outbox_enqueue(service):
    coordinator, client = service
    workers = worker_store(coordinator, [("worker-crash", 1, coordinator.workspace_id)])
    # This is the crash window: local revocation committed, native mirror absent.
    workers.revoke({"id": "manager"}, "worker-crash")
    assert not client.calls
    restarted = CoordinationService(coordinator.studio, coordinator.registry)
    restarted.workers = workers
    restarted.flush()
    revocations = [args for path, args in client.calls if "studio_revoke_worker" in path]
    assert revocations == [[coordinator.workspace_id, "worker-crash"]]
    assert "private-hash" not in json.dumps(client.calls)
    assert restarted._reconciled


def test_failed_revocation_enqueue_retries_from_local_authority(service, monkeypatch):
    coordinator, client = service
    coordinator.workers = worker_store(
        coordinator, [("worker-retry", 0, coordinator.workspace_id)]
    )
    coordinator._reconciled = True
    original = coordinator._queue

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(coordinator, "_queue", original)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(coordinator, "_queue", fail_once)
    assert not coordinator.revoke("worker-retry")
    assert not coordinator._reconciled
    coordinator.flush()
    assert coordinator._reconciled
    assert any(args == [coordinator.workspace_id, "worker-retry"] for _, args in client.calls)


def test_concurrent_enqueue_failure_cannot_be_acknowledged_by_older_repair(service, monkeypatch):
    coordinator, client = service
    workers = worker_store(coordinator, [("worker-later", 1, coordinator.workspace_id)])
    coordinator.workers = workers
    original_page, original_queue = workers.revoked_ids, coordinator._queue

    def page_then_revoke(**kwargs):
        old_page = original_page(**kwargs)
        workers.revoke({"id": "manager"}, "worker-later")
        monkeypatch.setattr(coordinator, "_queue", lambda *_a, **_k: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ))
        assert not coordinator.revoke("worker-later")
        monkeypatch.setattr(coordinator, "_queue", original_queue)
        monkeypatch.setattr(workers, "revoked_ids", original_page)
        return old_page

    monkeypatch.setattr(workers, "revoked_ids", page_then_revoke)
    coordinator.flush()
    assert not coordinator._reconciled
    assert "will retry" in coordinator.snapshot()["error"]
    coordinator.flush()
    assert coordinator._reconciled
    assert any(args == [coordinator.workspace_id, "worker-later"] for _, args in client.calls)


def test_revoked_history_reconciliation_is_paged_and_workspace_scoped(service):
    coordinator, client = service
    revoked = [f"worker-{index:04d}" for index in range(201)]
    coordinator.workers = worker_store(coordinator, [
        *((worker_id, 0, coordinator.workspace_id) for worker_id in revoked),
        ("worker-active", 1, coordinator.workspace_id),
        ("worker-foreign", 0, "different-workspace"),
    ])
    with pytest.raises(ValueError, match="1–100"):
        coordinator.workers.revoked_ids(limit=101)
    assert len(coordinator.workers.revoked_ids()) == 100
    coordinator.flush(limit=200)
    assert not coordinator._reconciled
    coordinator.flush(limit=200)
    assert not coordinator._reconciled
    coordinator.flush(limit=200)
    assert coordinator._reconciled
    assert [args[1] for path, args in client.calls if "studio_revoke_worker" in path] == revoked


def test_failed_task_repair_keeps_reconciliation_pending(service, monkeypatch):
    coordinator, client = service
    coordinator._defaults_ready = True
    original_sql, original_queue = client._sql, coordinator._queue

    def task_rows(query):
        if "SELECT run_id FROM factory_run" in query:
            return [{"run_id": goal()["id"]}]
        if "SELECT state_json, revision FROM factory_run" in query:
            return [{"revision": 0, "state_json": json.dumps({
                "registry_id": coordinator.workspace_id, "task": "studio-objective-v1",
                "goal": goal(),
            })}]
        return original_sql(query)

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(coordinator, "_queue", original_queue)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(client, "_sql", task_rows)
    monkeypatch.setattr(coordinator, "_queue", fail_once)
    coordinator.flush()
    assert not coordinator._reconciled
    coordinator.flush()
    assert coordinator._reconciled
    assert any("studio_sync_task" in path for path, _ in client.calls)


def test_shutdown_interrupts_waiting_durable_checkpoint_without_late_writes(service):
    from contextlib import contextmanager

    coordinator, client = service
    lock = threading.RLock()
    waiting = threading.Event()
    opened = []

    class Lock:
        def acquire(self, **kwargs):
            waiting.set()
            return lock.acquire(**kwargs)

        def release(self):
            lock.release()

    @contextmanager
    def request():
        opened.append(True)
        yield

    coordinator.durable = SimpleNamespace(lock=Lock(), holder=None, request=request)
    with lock:
        coordinator.start()
        assert waiting.wait(1)
        coordinator.close()
        assert not coordinator.thread.is_alive()
        assert not opened and not client.calls
    coordinator.flush()
    assert not opened and not client.calls


def test_shutdown_during_durable_pass_aborts_checkpoint(service):
    from contextlib import contextmanager

    coordinator, client = service
    outcomes = []

    @contextmanager
    def request():
        try:
            yield
        except Exception:
            outcomes.append("released")
            raise
        else:
            outcomes.append("committed")

    coordinator.durable = SimpleNamespace(lock=threading.RLock(), holder=None, request=request)
    client.after_call = coordinator.request_stop
    coordinator.flush()
    assert outcomes == ["released"]
    assert len(client.calls) == 1
