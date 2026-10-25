"""A real task journal drives the pair; fake physics isolates dispatch guarantees."""

import json
import time
from pathlib import Path

import pytest
from test_studio_tasks import EMPLOYEE, MANAGER, Journal, update
from test_studio_worker import Client

from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.service import StudioService, validate_workspace
from relay_lab.studio.simulation_recipe import (
    OBJECTIVE,
    PINNED_HASHES,
    SUCCESS_CRITERIA,
    recipe_workspace,
)
from relay_lab.studio.tasks import TaskService
from relay_lab.studio.team_worker import SimulationStopped, TeamRuntime
from relay_lab.studio.worker import AgentHTTPError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pair(tmp_path):
    journal = Journal()
    studio = StudioService(tmp_path / "studio", coordinator_factory=journal.factory)
    studio.save(recipe_workspace())
    registry = ModelRegistry(studio.storage)
    for model in registry.sync_workspace(MANAGER, studio.workspace())["models"]:
        registry.transition(MANAGER, model["id"], "released", model["revision"])
    tasks = TaskService(studio, registry)
    release = registry.authorize_workspace(EMPLOYEE, studio.workspace())
    workers = {
        m["spec"]["id"]: {
            "id": "worker-" + ("1" if m["spec"]["id"] == "left" else "2") * 16,
            "agent_id": m["spec"]["id"], "model_id": m["id"],
            "release_revision": m["revision"], "registry_id": registry.registry_id,
            "execution_kind": "simulation",
        }
        for m in release["models"]
    }
    task = tasks.create(EMPLOYEE, {
        "request_id": "recipe", "objective": OBJECTIVE,
        "success_criteria": SUCCESS_CRITERIA, "target": "both",
    })
    clients = {arm: Client(tasks, worker) for arm, worker in workers.items()}
    yield tasks, task, clients, tmp_path
    studio.close()


def success():
    return {"status": "succeeded", "success": True, "metrics": {"success": True}, "ticks": 1126}


def progress(index=0):
    return {
        "status": "running", "tick": index * 45,
        "phases": {arm: f"phase_{index}" for arm in ("left", "right")},
        "phase_indices": {arm: index for arm in ("left", "right")},
        "total_phases": 16, "hold_reasons": {},
    }


def runtime(pair, runner, **kwargs):
    _, task, clients, tmp = pair
    return TeamRuntime(
        clients, task["id"], tmp / "rollout", ROOT, tmp / "state", runner=runner,
        verifier=kwargs.pop("verifier", lambda *args: {"verified": True}), **kwargs,
    )


def test_recipe_is_valid_and_explicit_about_controller_identity():
    workspace = recipe_workspace()
    assert validate_workspace(workspace, complete=True) == workspace
    assert all(spec["family"] == "custom" for spec in workspace["agents"])
    assert all("state-assisted" in spec["action_contract"] for spec in workspace["agents"])
    assert {s["id"]: s["source"]["revision"] for s in workspace["agents"]} == PINNED_HASHES


def test_one_world_two_scoped_claims_measured_success_and_durable_result(pair):
    tasks, task, clients, tmp = pair
    calls = []

    def run(config, output, **hooks):
        calls.append(config)
        assert config["coordination"] == "spacetime" and config["planner"] == "llm"
        assert hooks["expected_checkpoint_hashes"] == PINNED_HASHES
        assert not hooks["should_stop"]()
        output.mkdir()
        hooks["on_progress"](progress())
        hooks["on_progress"](progress(1))
        return success()

    worker = runtime(pair, run)
    try:
        result = worker.run()
        assert result["status"] == "succeeded" and result["success"] is True
        assert result["verified_physical_outcome"] is False
        assert result["verification"]["verified"]
        saved = tasks.get(EMPLOYEE, task["id"])
        assert saved["status"] == "succeeded"
        assert len({a["worker_id"] for a in saved["assignments"].values()}) == 2
        assert saved["revision"] == 8  # Two claims, four phase reports, two final reports.
        assert json.loads((tmp / "state/progress.json").read_text())["tick"] == 45
        assert worker.run() == result
        assert len(calls) == 1
        assert all(c.sent_reports[-1]["status"] == "succeeded" for c in clients.values())
    finally:
        worker.close()


@pytest.mark.parametrize("metric_success,verified", [(False, True), (True, False)])
def test_success_requires_measured_geometry_and_independent_replay(pair, metric_success, verified):
    tasks, task, _, _ = pair
    value = success()
    value["metrics"]["success"] = metric_success
    verifier_calls = []
    worker = runtime(
        pair, lambda *args, **kwargs: value,
        verifier=lambda *args: verifier_calls.append(1) or {"verified": verified},
    )
    try:
        result = worker.run()
        assert result["status"] == "failed" and not result["success"]
        assert tasks.get(EMPLOYEE, task["id"])["status"] == "failed"
        assert len(verifier_calls) == int(metric_success)
    finally:
        worker.close()


def test_ambiguous_report_retries_same_envelope_without_repeating_world(pair):
    _, _, clients, _ = pair
    clients["left"].after_report = lambda: (_ for _ in ()).throw(AgentHTTPError(503))
    calls = []
    worker = runtime(pair, lambda *a, **k: calls.append(1) or success())
    try:
        assert worker.run()["status"] == "succeeded"
        assert calls == [1]
        sent = clients["left"].sent_reports
        assert len(sent) == 2 and sent[0] == sent[1]
    finally:
        worker.close()


def test_peer_revision_conflict_rebases_result_without_repeating_physics(pair):
    tasks, task, clients, _ = pair

    def peer_report():
        current = tasks.get(EMPLOYEE, task["id"])
        tasks.report(clients["right"].identity, task["id"], update(
            current, "peer-update", status="running", progress=90,
            summary="Peer state changed before left report", evidence=[],
        ))

    clients["left"].before_report = peer_report
    calls = []
    worker = runtime(pair, lambda *a, **k: calls.append(1) or success())
    try:
        assert worker.run()["status"] == "succeeded"
        sent = clients["left"].sent_reports
        assert len(sent) == 2 and sent[0]["request_id"] != sent[1]["request_id"]
        assert sent[0]["evidence"] == sent[1]["evidence"]
        assert calls == [1]
    finally:
        worker.close()


def test_cancelled_task_stops_runtime_without_success_report(pair):
    tasks, task, clients, _ = pair

    def run(*args, **hooks):
        current = tasks.get(EMPLOYEE, task["id"])
        tasks.cancel(MANAGER, task["id"], update(current, "stop", reason="Operator stopped"))
        deadline = time.monotonic() + 2
        while not hooks["should_stop"]() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hooks["should_stop"]()
        return {"status": "cancelled", "success": False, "metrics": {}}

    worker = runtime(pair, run, poll_seconds=0.01)
    try:
        assert worker.run()["status"] == "cancelled"
        assert tasks.get(EMPLOYEE, task["id"])["status"] == "cancelled"
        assert not any(c.sent_reports for c in clients.values())
    finally:
        worker.close()


def test_monitor_loses_authorization_and_holds_next_step(pair):
    _, _, clients, _ = pair

    def run(*args, **hooks):
        old = clients["left"].request

        def revoked(path, value=None):
            if path == "/api/agent/heartbeat":
                raise AgentHTTPError(403)
            return old(path, value)

        clients["left"].request = revoked
        deadline = time.monotonic() + 2
        while not hooks["should_stop"]() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hooks["should_stop"]()
        return success()  # Even a stale successful result must not award completion.

    worker = runtime(pair, run, poll_seconds=0.01)
    try:
        assert worker.run()["success"] is False
        assert not any(c.sent_reports for c in clients.values())
    finally:
        worker.close()


def test_interrupted_world_cannot_restart_or_be_claimed_twice(pair):
    tasks, task, _, _ = pair
    calls = []

    def crash(*args, **kwargs):
        calls.append(1)
        raise KeyboardInterrupt

    worker = runtime(pair, crash)
    try:
        with pytest.raises(KeyboardInterrupt):
            worker.run()
        assert worker.state["phase"] == "executing"
    finally:
        worker.close()
    restarted = runtime(pair, crash)
    try:
        result = restarted.run()
        assert result["status"] == "interrupted"
        assert calls == [1]
        assert tasks.get(EMPLOYEE, task["id"])["revision"] == 2
    finally:
        restarted.close()


def test_journal_and_engine_locks_prevent_two_worlds(pair):
    _, task, clients, tmp = pair
    worker = runtime(pair, lambda *a, **k: success(), engine_lock_file=tmp / "engine.lock")
    try:
        with pytest.raises(RuntimeError, match="running process"):
            runtime(pair, lambda *a, **k: success())
        with pytest.raises(RuntimeError, match="workspace engine"):
            TeamRuntime(clients, task["id"], tmp / "other-output", ROOT, tmp / "other-state",
                        engine_lock_file=tmp / "engine.lock")
    finally:
        worker.close()


def test_task_or_model_mismatch_fails_before_actions(pair):
    _, _, clients, _ = pair
    original = clients["left"].request

    def altered(path, value=None):
        result = original(path, value)
        if path.endswith("/" + pair[1]["id"]):
            result["assignments"]["right"]["spec"]["source"]["revision"] = "0" * 64
        return result

    clients["left"].request = altered
    calls = []
    worker = runtime(pair, lambda *a, **k: calls.append(1) or success())
    try:
        assert worker.run()["status"] == "failed"
        assert calls == []
    finally:
        worker.close()


def test_checkpoint_bytes_are_verified_before_model_execution(pair, tmp_path):
    calls = []
    worker = runtime(pair, lambda *a, **k: calls.append(1) or success())
    worker.root = tmp_path / "bad-root"
    checkpoint = worker.root / "checkpoints/cloth-policies/left.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unapproved weights")
    try:
        assert worker.run()["status"] == "failed"
        assert calls == []
    finally:
        worker.close()


def test_cancel_during_verification_cannot_publish_success(pair):
    tasks, task, clients, _ = pair

    def verify(*args):
        current = tasks.get(EMPLOYEE, task["id"])
        tasks.cancel(MANAGER, task["id"], update(current, "cancel-replay", reason="Stop replay"))
        raise SimulationStopped("Replay stopped")

    worker = runtime(pair, lambda *a, **k: success(), verifier=verify)
    try:
        assert worker.run()["status"] == "cancelled"
        assert not any(c.sent_reports for c in clients.values())
    finally:
        worker.close()


def test_unconfirmed_final_report_resumes_reporting_without_reexecuting_world(pair):
    _, _, clients, _ = pair
    original = clients["left"].request
    calls = []

    def disconnected(path, value=None):
        if path.endswith("/report"):
            raise AgentHTTPError(503)
        return original(path, value)

    clients["left"].request = disconnected
    worker = runtime(pair, lambda *a, **k: calls.append(1) or success())
    try:
        result = worker.run()
        assert result["status"] == "report_pending" and not result["success"]
        assert worker.state["phase"] == "reporting"
        pending_id = worker.state["pending"]["payload"]["request_id"]
    finally:
        worker.close()
    clients["left"].request = original
    restarted = runtime(pair, lambda *a, **k: calls.append(1) or success())
    try:
        assert restarted.run()["status"] == "succeeded"
        assert calls == [1]
        assert clients["left"].sent_reports[0]["request_id"] == pending_id
    finally:
        restarted.close()


def test_cancel_between_final_arm_reports_prevents_global_success(pair):
    tasks, task, clients, _ = pair

    def cancel():
        current = tasks.get(EMPLOYEE, task["id"])
        tasks.cancel(MANAGER, task["id"], update(current, "cancel-final", reason="Cancel now"))

    clients["left"].after_report = cancel
    worker = runtime(pair, lambda *a, **k: success())
    try:
        result = worker.run()
        assert result["status"] == "cancelled" and not result["success"]
        assert tasks.get(EMPLOYEE, task["id"])["status"] == "cancelled"
        assert not clients["right"].sent_reports
    finally:
        worker.close()


def test_stale_authority_stops_without_waiting_for_blocked_network(pair):
    worker = runtime(pair, lambda *a, **k: success(), freshness_seconds=0.1)
    try:
        worker.last_authority = time.monotonic() - 1
        assert worker.should_stop()
        assert "stale" in worker.stop_reason
    finally:
        worker.close()
