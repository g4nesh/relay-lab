"""Analytics grain, identity matching, truthful missingness and privacy contracts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.analytics import (
    EFFORT_DEFAULTS,
    AnalyticsService,
    distribution,
    estimate_effort,
    summarize_factory,
    summarize_learned,
)

ROOT = Path(__file__).resolve().parents[1]
MANAGER = {"id": "manager-secret-id", "role": "manager", "username": "private-manager"}
EMPLOYEE = {"id": "employee-secret-id", "role": "employee", "username": "private-employee"}


class Registry:
    def __init__(self):
        self.actors = []

    def catalog(self, actor):
        self.actors.append(actor)
        return {
            "models": [
                {
                    "status": status,
                    "created_by": "private-author",
                    "spec": {"source": "private-host-token"},
                }
                for status in ("draft", "released", "retired")
            ]
        }


class Studio:
    def __init__(self, telemetry=None, histories=None):
        self.telemetry = telemetry
        self.histories = histories or {}
        self.reads = []

    def list_sessions(self):
        return {
            "sessions": [
                {
                    "session_id": key,
                    "status": "running",
                    "name": "Private project",
                    "updated_at": index,
                }
                for index, key in enumerate(self.histories)
            ]
        }

    def history(self, session_id):
        self.reads.append(session_id)
        return {"results": self.histories[session_id]}


def read(path):
    return json.loads((ROOT / path).read_text())


def sample(exchange=1, **overrides):
    return {
        "exchange": exchange,
        "session_id": "studio-001",
        "mode": "latent",
        "bytes": 100,
        "elapsed_ms": 10.0,
        "database_ms": 6.0,
        "llm_ms": 0,
        "source": "captured",
        "readback_verified": True,
        "actions_executed": False,
        **overrides,
    }


def test_raw_percentiles_and_missing_values():
    assert distribution([0, 10, 20, 30]) == pytest.approx(
        {"n": 4, "p50": 15.0, "p95": 28.5, "mean": 15.0}
    )
    assert distribution([None, -1, float("nan"), float("inf"), True])["n"] == 0
    assert distribution([])["p95"] is None


def test_effort_is_an_explicit_scenario_and_keeps_negative_savings():
    result = estimate_effort(EFFORT_DEFAULTS)
    assert result["edges"] == 15
    assert result["manual_hours"] == 102
    assert result["relay_hours"] == 20
    assert result["hours_saved"] == 82
    assert result["cost_saved"] == 8200
    assert result["measured"] is False
    slower = estimate_effort({**EFFORT_DEFAULTS, "relay_per_robot_hours": 100})
    assert slower["hours_saved"] < 0
    zero = estimate_effort(
        {k: v if k in ("robots", "tasks") else 0 for k, v in EFFORT_DEFAULTS.items()}
    )
    assert zero["savings_percent"] is None


@pytest.mark.parametrize("invalid", [True, -1, float("nan"), float("inf"), "2", 100001])
def test_effort_rejects_invalid_numbers(invalid):
    with pytest.raises(ValueError):
        estimate_effort({**EFFORT_DEFAULTS, "hourly_rate": invalid})


def test_effort_rejects_fractional_counts_and_unknown_fields():
    with pytest.raises(ValueError):
        estimate_effort({**EFFORT_DEFAULTS, "robots": 1.1})
    with pytest.raises(ValueError):
        estimate_effort({**EFFORT_DEFAULTS, "extra": 1})
    with pytest.raises(ValueError):
        estimate_effort({})


def test_factory_counts_raw_runs_and_does_not_claim_a_database_speedup():
    source = read("examples/factory-communication/results.json")
    # Precomputed aggregate is deliberately corrupt. We must derive from runs.
    source["aggregate"]["none"]["successful_tasks"] = 999
    result = summarize_factory(source)
    assert result["pairing_verified"] is True
    assert result["paired_cases"] == 12
    assert result["attempts"] == 36
    rows = {r["mode"]: r for r in result["rows"]}
    assert (rows["text"]["successes"], rows["none"]["successes"]) == (12, 2)
    assert rows["none"]["failures"] == 10
    assert rows["text"]["total_routes"] == 36
    assert rows["text"]["payload_bytes"] == 468
    assert result["backend"] == "memory"
    assert result["comparison"]["time_saved_seconds"] is None
    assert result["comparison"]["success_gain_pp"]["text"] == pytest.approx(83.333333)
    assert rows["text"]["wilson95"][0] < 1


@pytest.mark.parametrize(
    "mutation", ["identity", "missing", "duplicate", "audit", "codec", "sensor", "schedule"]
)
def test_mismatched_cohorts_do_not_produce_lift(mutation):
    source = read("examples/factory-communication/results.json")
    if mutation == "identity":
        source["runs"][0]["actor_sha256"] = "another-policy"
    elif mutation == "missing":
        source["runs"].pop()
    elif mutation == "duplicate":
        source["runs"].append(copy.deepcopy(source["runs"][0]))
    elif mutation == "codec":
        source["runs"][0]["communication"]["model"]["checkpoint_sha256"] = "another-codec"
    elif mutation == "sensor":
        source["runs"][0]["communication"]["observation_provenance"][0]["source_frame_sha256"] = (
            "other-frame"
        )
    elif mutation == "schedule":
        source["runs"][0]["communication"]["period_ticks"] += 1
    else:
        source["paired_checks"][0]["identical_private_sensor_inputs"] = False
    result = summarize_factory(source)
    assert result["pairing_verified"] is False
    assert result["comparison"]["success_gain_pp"] == {}
    assert result["attempts"] == len(source["runs"])


def test_learned_decisions_and_live_failures_stay_separate():
    result = summarize_learned(
        read("examples/cloth-coordination/learned-latent/results.json"),
        read("examples/cloth-coordination/learned-latent/models/evaluation.json"),
        read("examples/cloth-coordination/learned-latent/transport/manifest.json"),
    )
    rows = {r["mode"]: r for r in result["static"]}
    assert rows["none"]["hidden_accuracy"] == 0.5
    assert rows["embeddings"]["hidden_accuracy"] == 1.0
    assert rows["embeddings"]["decisions"] == 120
    assert rows["embeddings"]["hidden_decisions"] == 80
    assert result["live"]["full_task_success"] is False
    assert result["live"]["grasp_count"] == 0
    assert result["live"]["missed_hazard_holds"] == 12
    stressed = next(
        r
        for r in result["transport"]
        if r["mode"] == "embeddings" and r["scenario"] == "delayed_dropped"
    )
    assert stressed["hidden_correct"] == 158
    assert stressed["hidden_decisions"] == 320
    assert stressed["hidden_accuracy"] == 0.49375


def test_operations_deduplicate_journal_and_preserve_unknown_latency(tmp_path):
    telemetry = SimpleNamespace(snapshot=lambda **_: {"rows": [sample()], "total_rows": 1})
    studio = Studio(
        telemetry, {"studio-001": [sample(), sample(2, elapsed_ms=None, database_ms=None)]}
    )
    result = AnalyticsService(tmp_path, studio, Registry()).snapshot(EMPLOYEE)
    ops = result["operations"]
    assert ops["exchanges"] == 2
    assert ops["modes"][0]["elapsed_ms"] == {"n": 1, "p50": 10.0, "p95": 10.0, "mean": 10.0}
    assert ops["coverage"]["timing_coverage"] == 0.5
    assert ops["modes"][0]["llm_ms"]["n"] == 2
    assert ops["archive"]["included_in_live_totals"] is False


def test_archive_is_not_live_and_no_private_catalog_or_session_data_leaks():
    registry = Registry()
    result = AnalyticsService(ROOT, Studio(histories={"session-secret": []}), registry).snapshot(
        EMPLOYEE
    )
    assert result["operations"]["exchanges"] == 0
    assert result["operations"]["archive"]["exchanges"] == 6
    assert result["governance"]["models"] == {"total": 3, "draft": 1, "released": 1, "retired": 1}
    serialized = json.dumps(result)
    for secret in (
        "private-author",
        "private-host-token",
        "session-secret",
        "Private project",
        "private-employee",
        "employee-secret-id",
    ):
        assert secret not in serialized
    assert all(source["sha256"] for source in result["sources"] if source["status"] == "available")
    assert result["cloth"]["diagnostics"]["pooled_with_primary"] is False


def test_inventory_and_history_are_bounded(tmp_path):
    studio = Studio(histories={f"session-{i}": [sample(j) for j in range(60)] for i in range(250)})
    result = AnalyticsService(tmp_path, studio, Registry()).snapshot(MANAGER)
    assert len(studio.reads) == 20
    assert result["operations"]["sessions"] == 200
    assert result["operations"]["exchanges"] == 500
    assert result["operations"]["coverage"]["session_inventory_may_be_truncated"] is True


def test_auth_is_required_even_on_cache_hit_and_catalog_is_refreshed(tmp_path):
    registry = Registry()
    service = AnalyticsService(tmp_path, Studio(), registry)
    first = service.snapshot(MANAGER)
    first["sources"].clear()
    assert service.snapshot(EMPLOYEE)["sources"]
    assert registry.actors == [MANAGER, EMPLOYEE]
    with pytest.raises(AuthenticationError):
        service.snapshot(None)
    with pytest.raises(AuthenticationError):
        service.snapshot({"id": "user", "role": "admin"})


def test_source_symlink_cannot_escape_workspace(tmp_path):
    other = tmp_path / "outside.json"
    other.write_text('{"password":"must-not-leak"}')
    workspace = tmp_path / "workspace"
    target = workspace / "examples/factory-communication/results.json"
    target.parent.mkdir(parents=True)
    target.symlink_to(other)
    result = AnalyticsService(workspace, Studio(), Registry()).snapshot(MANAGER)
    assert result["factory"]["status"] == "unavailable"
    assert "must-not-leak" not in json.dumps(result)


def test_live_unavailable_is_explicit(tmp_path):
    class Offline(Studio):
        def list_sessions(self):
            raise RuntimeError("private-token-must-not-escape")

    result = AnalyticsService(tmp_path, Offline(), Registry()).snapshot(EMPLOYEE)
    assert result["operations"]["status"] == "unavailable"
    assert result["headline"]["value"] is None
    assert "private-token-must-not-escape" not in json.dumps(result)


def journal_message(tick, sender="left", mode="latent", provenance="captured foundation features"):
    return {
        "message_id": f"{sender}-{tick}",
        "sender": sender,
        "recipient": "right" if sender == "left" else "left",
        "payload": {
            "exchange": tick,
            "mode": mode,
            "provenance": provenance,
            "context": "private older report",
            "vector": [0.1, 0.2],
        },
    }


def test_restart_inspection_counts_observed_exchanges_without_inventing_timings(tmp_path):
    class Inspected(Studio):
        def history(self, session_id):
            return {
                "tick": 5,
                "results": [],
                "failed": False,
                "status": "inspect_only",
                "messages": [
                    journal_message(1),
                    journal_message(1, "right"),
                    journal_message(2),
                    journal_message(2),
                    journal_message(6),
                    journal_message(0),
                ],
            }

    service = AnalyticsService(
        tmp_path, Inspected(histories={"private-session-id": []}), Registry()
    )
    result = service.snapshot(EMPLOYEE)
    ops = result["operations"]
    assert ops["exchanges"] == 2
    assert ops["coverage"]["journal_only_exchanges"] == 2
    assert ops["coverage"]["timed_exchanges"] == 0
    assert ops["modes"][0]["llm_ms"]["n"] == 0
    assert ops["modes"][0]["verified_readbacks"] == 0
    assert ops["modes"][0]["elapsed_ms"]["p50"] is None
    assert ops["failed_sessions"] == 0
    assert "private older report" not in json.dumps(result)
    assert "private-session-id" not in json.dumps(result)


def test_telemetry_and_result_take_priority_over_older_journal_group(tmp_path):
    class Inspected(Studio):
        def history(self, session_id):
            return {
                "tick": 3,
                "results": [sample(2, elapsed_ms=30)],
                "failed": False,
                "messages": [journal_message(i) for i in (1, 2, 3)],
            }

    telemetry = SimpleNamespace(snapshot=lambda **_: {"rows": [sample()], "total_rows": 1})
    result = AnalyticsService(
        tmp_path, Inspected(telemetry, {"studio-001": []}), Registry()
    ).snapshot(EMPLOYEE)
    ops = result["operations"]
    assert ops["exchanges"] == 3
    assert ops["coverage"]["journal_only_exchanges"] == 1
    assert ops["coverage"]["timed_exchanges"] == 2
    assert ops["modes"][0]["elapsed_ms"]["mean"] == 20
    assert ops["modes"][0]["verified_readbacks"] == 2


def test_message_reconstruction_is_limited_by_committed_tick_and_history_window(tmp_path):
    class Inspected(Studio):
        def history(self, session_id):
            return {
                "tick": 100,
                "results": [],
                "failed": False,
                "messages": [journal_message(i) for i in range(1, 103)],
            }

    result = AnalyticsService(
        tmp_path, Inspected(histories={"studio-001": []}), Registry()
    ).snapshot(EMPLOYEE)
    assert result["operations"]["exchanges"] == 50
    assert result["operations"]["coverage"]["journal_only_exchanges"] == 50


def test_worker_failure_is_reconciled_and_unknown_health_is_not_zero(tmp_path):
    class Inspected(Studio):
        def history(self, session_id):
            if session_id == "unknown":
                raise RuntimeError("unreadable")
            return {"tick": 0, "results": [], "messages": [], "failed": session_id == "failed"}

    studio = Inspected(histories={"failed": [], "healthy": [], "unknown": []})
    result = AnalyticsService(tmp_path, studio, Registry()).snapshot(EMPLOYEE)["operations"]
    assert result["failed_sessions"] is None
    assert result["known_failed_sessions"] == 1
    assert result["coverage"]["session_health_checked"] == 2
    assert result["coverage"]["session_health_unknown"] == 1
    assert result["coverage"]["failed_session_count_complete"] is False
    studio.histories.pop("unknown")
    complete = AnalyticsService(tmp_path, studio, Registry()).snapshot(EMPLOYEE)["operations"]
    assert complete["failed_sessions"] == 1
    assert complete["coverage"]["failed_session_count_complete"] is True
