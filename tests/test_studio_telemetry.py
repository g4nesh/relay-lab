"""Post-ACK telemetry must be durable without becoming part of action success."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_studio_permissions_http import LLM, Journal

from relay_lab.studio.service import StudioService, example_workspace
from relay_lab.studio.telemetry import TelemetryStore


def result(**changes):
    return {
        "exchange": 1,
        "mode": "mediated",
        "source": "manual",
        "elapsed_ms": 13.5,
        "database_ms": 3.5,
        "llm_ms": 10.0,
        "bytes": 456,
        "readback_verified": True,
        "actions_executed": False,
        "recovered": False,
        "input_reports": {"left": "private observation text"},
        "messages": [{"payload": {"secret": "private-message"}}],
        **changes,
    }


def request():
    return {
        "request_id": "same-captured-observation",
        "source": "manual",
        "reports": {"left": "Local path clear", "right": "Local obstacle; please hold"},
    }


def connected_service(tmp_path):
    journal, llm = Journal(), LLM()
    service = StudioService(tmp_path, coordinator_factory=journal.coordinator, llm=llm)
    workspace = {**example_workspace(), "mode": "mediated"}
    session_id = service.connect(workspace)["session_id"]
    return service, journal, llm, session_id


def test_telemetry_persists_across_store_restart_without_message_payloads(tmp_path):
    store = TelemetryStore(tmp_path)
    assert store.record("studio-example", "both", result())
    reopened = TelemetryStore(tmp_path)
    snapshot = reopened.snapshot()
    assert snapshot["total_rows"] == 1
    assert snapshot["unavailable_reason"] is None
    row = snapshot["rows"][0]
    assert row["readback_verified"] is True and row["actions_executed"] is False
    assert row["elapsed_ms"] == 13.5
    assert row["recorded_at"] > 0
    assert "private-message" not in json.dumps(snapshot)
    assert "private observation" not in json.dumps(snapshot)
    assert "messages" not in row and "input_reports" not in row
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_retry_keeps_original_measurement_and_concurrent_duplicates_are_unique(tmp_path):
    store = TelemetryStore(tmp_path)
    assert store.record("studio-example", "both", result())
    assert not store.record("studio-example", "both", result(elapsed_ms=99999))
    with ThreadPoolExecutor(max_workers=4) as pool:
        written = list(
            pool.map(
                lambda _: store.record("studio-example", "both", result(exchange=2)), range(12)
            )
        )
    assert sum(written) == 1
    snapshot = store.snapshot()
    assert snapshot["total_rows"] == 2
    original = next(r for r in snapshot["rows"] if r["exchange"] == 1)
    assert original["elapsed_ms"] == 13.5


@pytest.mark.parametrize("changes", [{"recovered": True}, {"readback_verified": False}])
def test_recovered_and_unverified_results_are_not_fresh_measurements(tmp_path, changes):
    store = TelemetryStore(tmp_path)
    assert not store.record("studio-example", "both", result(**changes))
    assert store.snapshot()["total_rows"] == 0


def test_missing_nonfinite_negative_timings_are_unknown_not_zero(tmp_path):
    store = TelemetryStore(tmp_path)
    store.record(
        "studio-example", "both", result(elapsed_ms=float("nan"), database_ms=-1, llm_ms=None)
    )
    store.record(
        "studio-example",
        "both",
        result(exchange=2, elapsed_ms=float("inf"), database_ms=True, llm_ms=0),
    )
    rows = sorted(store.snapshot()["rows"], key=lambda row: row["exchange"])
    assert (
        rows[0]["elapsed_ms"] is None
        and rows[0]["database_ms"] is None
        and rows[0]["llm_ms"] is None
    )
    assert (
        rows[1]["elapsed_ms"] is None and rows[1]["database_ms"] is None and rows[1]["llm_ms"] == 0
    )


@pytest.mark.parametrize("limit", [0, -1, 501, 1.5, True])
def test_snapshot_limit_is_bounded(tmp_path, limit):
    with pytest.raises(ValueError):
        TelemetryStore(tmp_path).snapshot(limit)


def test_snapshot_returns_recent_window_and_lifetime_row_count(tmp_path):
    store = TelemetryStore(tmp_path)
    for tick in range(1, 5):
        store.record("studio-example", "both", result(exchange=tick))
    snapshot = store.snapshot(limit=2)
    assert snapshot["total_rows"] == 4
    assert len(snapshot["rows"]) == 2
    assert [row["exchange"] for row in snapshot["rows"]] == [4, 3]


def test_missing_storage_is_reported_without_raising(tmp_path):
    store = TelemetryStore(tmp_path / "does-not-exist")
    assert store.unavailable_reason
    assert not store.record("studio-example", "both", result())
    assert store.snapshot()["rows"] == []
    assert store.snapshot()["unavailable_reason"]


def test_real_exchange_records_once_and_duplicate_does_not_reinfer(tmp_path):
    service, journal, llm, sid = connected_service(tmp_path)
    first = service.exchange(sid, request())
    second = service.exchange(sid, request())
    assert first == second
    assert journal.commits == 1 and llm.calls == 2
    assert service.telemetry.snapshot()["total_rows"] == 1
    assert service.telemetry.snapshot()["rows"][0]["elapsed_ms"] == first["elapsed_ms"]


def test_lost_telemetry_and_recovered_retry_do_not_fabricate_timing(tmp_path):
    service, journal, llm, sid = connected_service(tmp_path)
    service.exchange(sid, request())
    service.telemetry.path.unlink()
    recovered = StudioService(tmp_path, coordinator_factory=journal.coordinator, llm=llm)
    recovered.session(sid)
    replay = recovered.exchange(sid, request())
    assert replay["recovered"] is True
    assert replay["elapsed_ms"] is None and replay["database_ms"] is None
    assert recovered.telemetry.snapshot()["total_rows"] == 0
    assert journal.commits == 1 and llm.calls == 2


def test_measurement_disk_failure_cannot_fail_a_committed_exchange(tmp_path, monkeypatch):
    service, journal, llm, sid = connected_service(tmp_path)

    def fail_connect():
        raise sqlite3.OperationalError("private-location-must-not-leak")

    monkeypatch.setattr(service.telemetry, "_connect", fail_connect)
    response = service.exchange(sid, request())
    assert response["readback_verified"] is True
    assert service.sessions[sid]["failed"] is False
    assert service.sessions[sid]["tick"] == 1
    assert journal.commits == 1 and llm.calls == 2
    assert service.telemetry.unavailable_reason
    assert "private-location" not in service.telemetry.unavailable_reason
    assert service.exchange(sid, request()) == response
