"""Opt-in real local DB checks. Publish the module to a disposable DB first.

RELAY_COORDINATION_TEST_DATABASE and RELAY_COORDINATION_TEST_TOKEN_FILE are
required; these tests never publish, clear a database, or execute robot actions.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from relay_lab.spacetime import SpacetimeClient, SpacetimeConfig, SpacetimeError, _read_token
from relay_lab.studio.coordination import CoordinationService
from relay_lab.studio.realtime import RealtimeHub


@pytest.fixture
def live(tmp_path):
    database = os.environ.get("RELAY_COORDINATION_TEST_DATABASE")
    token_file = os.environ.get("RELAY_COORDINATION_TEST_TOKEN_FILE")
    if not database or not token_file:
        pytest.skip("Explicit disposable Spacetime database and owner token file required")
    client = SpacetimeClient(
        SpacetimeConfig(
            os.environ.get("RELAY_COORDINATION_TEST_SERVER", "http://127.0.0.1:3000"),
            database,
            _read_token(Path(token_file)),
        )
    )
    workspace = "test-" + uuid.uuid4().hex[:16]
    service = SimpleNamespace(storage=tmp_path, factory=lambda: SimpleNamespace(client=client))
    coordinator = CoordinationService(
        service, SimpleNamespace(registry_id=workspace, lock=threading.RLock())
    )
    yield coordinator, client, workspace
    coordinator.close()


def wait_for(read, match, seconds=7):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = read()
        if match(value):
            return value
        time.sleep(0.1)
    raise AssertionError("Native scheduler did not reach the expected state")


def test_native_expiry_runs_without_any_dashboard_process(live):
    c, _, _ = live
    c.heartbeat({"id": "worker-a", "agent_id": "left"}, lease_seconds=1)
    assert c.thread is None  # No Python background scheduler or dashboard is running.
    rows = wait_for(lambda: c._rows("studio_presence"), lambda r: r and r[0]["status"] == "stale")
    assert rows[0]["generation"] == 1
    assert c._rows("studio_attention")[0]["kind"] == "worker_stale"


def test_renewal_cancels_stale_timer_and_revocation_cannot_resurrect(live):
    c, _, workspace = live
    worker = {"id": "worker-a", "agent_id": "left"}
    c.heartbeat(worker, lease_seconds=1)
    c.heartbeat(worker, lease_seconds=4)
    time.sleep(1.3)
    assert c._rows("studio_presence")[0]["status"] == "online"
    c._call("studio_revoke_worker", [workspace, "worker-a"])
    with pytest.raises(SpacetimeError):
        c.heartbeat(worker)
    assert c._rows("studio_presence")[0]["status"] == "revoked"


def test_task_deadline_alerts_and_terminal_cancels_watchdog(live):
    c, _, workspace = live
    c._call("studio_sync_task", [workspace, "task-1", 0, "queued", 1])
    wait_for(
        lambda: c._rows("studio_attention"), lambda r: any(x["kind"] == "task_overdue" for x in r)
    )
    c._call("studio_sync_task", [workspace, "task-1", 1, "cancelled", 1])
    assert c._rows("studio_attention")[0]["status"] == "resolved"
    c._call("studio_sync_task", [workspace, "task-1", 0, "queued", 1])
    assert c._rows("studio_task_watch")[0]["status"] == "cancelled"


def test_native_rollup_continuation_and_dedup(live):
    c, _, workspace = live
    c._call("studio_configure_job", [workspace, "rollup", True, "hourly", 7])
    # One scheduled continuation must finish this >256 record backlog without
    # any dashboard process. Numeric samples are synthetic TEST measurements.
    for i in range(260):
        c._call("studio_record_metric", [workspace, f"e-{i}", "exchange_latency_ms", 10.0])
    c._call("studio_record_metric", [workspace, "e-0", "exchange_latency_ms", 10.0])
    c._call("studio_run_job", [workspace, "rollup"])
    rows = wait_for(
        lambda: c._rows("studio_rollup"), lambda r: sum(int(x["sample_count"]) for x in r) == 260
    )
    assert sum(x["total"] for x in rows) == 2600
    assert int(c._rows("studio_job")[0]["run_count"]) >= 2
    c._call("studio_configure_job", [workspace, "rollup", False, "daily", 7])
    assert c._rows("studio_job")[0]["next_run_us"] == 0


def test_real_sdk_subscription_receives_db_change_and_reconnects(live):
    c, _, workspace = live

    async def run():
        hub = RealtimeHub(c.studio)
        queue = hub.subscribe()
        try:
            for _ in range(100):
                if hub.state == "ready":
                    break
                await asyncio.sleep(0.1)
            assert hub.state == "ready", hub.snapshot()
            while not queue.empty():
                queue.get_nowait()
            await asyncio.to_thread(
                c._call, "studio_heartbeat", [workspace, "worker-push", "left", 45]
            )
            event = await asyncio.wait_for(queue.get(), 3)
            assert "workers" in event["topics"]
            assert workspace not in json.dumps(event) and "worker-push" not in json.dumps(event)
            process = hub.process
            process.terminate()
            await process.wait()
            for _ in range(100):
                if hub.process is not None and hub.process is not process and hub.state == "ready":
                    break
                await asyncio.sleep(0.1)
            assert hub.process is not process and hub.state == "ready"
            assert (await asyncio.wait_for(queue.get(), 2))["event"] == "sync"
        finally:
            hub.unsubscribe(queue)
            await hub.close()

    asyncio.run(run())
