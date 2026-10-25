"""Fixed simulation launch, lifecycle and artifact boundaries through real auth."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from test_studio_permissions_http import Bridge
from test_studio_tasks import Journal

from relay_lab.spacetime import SpacetimeError
from relay_lab.studio import simulations
from relay_lab.studio.simulation_recipe import (
    MAX_STATE_BYTES,
    OBJECTIVE,
    RECIPE_ID,
    SUCCESS_CRITERIA,
    recipe_workspace,
)
from relay_lab.studio.tasks import TaskConflict, TaskService
from relay_lab.studio.workers import WorkerStore


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    result = Bridge(tmp_path)
    result.monkeypatch = monkeypatch
    result.task_journal = Journal()
    result.service.factory = result.task_journal.factory
    result.tasks = TaskService(result.service, result.registry)
    result.workers = WorkerStore(
        result.service.storage, result.access, result.registry, result.service
    )
    monkeypatch.setattr(simulations.SimulationService, "_watch", lambda self: None)
    monkeypatch.setattr(simulations.SimulationService, "prerequisites", lambda self: (True, None))
    monkeypatch.setattr(simulations.SimulationService, "_preflight", lambda self: None)
    result.simulations = simulations.SimulationService(
        result.service,
        result.access,
        result.registry,
        result.tasks,
        result.workers,
    )
    result.service.simulations = result.simulations
    result.spawns = []

    def spawn(command, **kwargs):
        process = SimpleNamespace(returncode=None, poll=lambda: process.returncode)
        result.spawns.append((command, kwargs, process))
        return process

    monkeypatch.setattr(simulations.subprocess, "Popen", spawn)
    yield result
    result.simulations.close()


def approve_recipe(bridge):
    assert (
        bridge.request(
            "/api/workspace", method="POST", actor=bridge.manager, value=recipe_workspace()
        )[0]
        == 200
    )
    records = bridge.request("/api/models", actor=bridge.manager)[2]["models"]
    for record in records:
        if record["spec"] in recipe_workspace()["agents"]:
            assert (
                bridge.request(
                    f"/api/models/{record['id']}/release",
                    method="POST",
                    actor=bridge.manager,
                    value={"expected_revision": record["revision"], "reason": "Approved recipe"},
                )[0]
                == 200
            )


def launch(bridge, request_id="fold-one", **changes):
    return bridge.request(
        "/api/simulations",
        method="POST",
        actor=bridge.manager,
        value={"request_id": request_id, **changes},
    )


def completed(bridge):
    approve_recipe(bridge)
    status, _, run = launch(bridge)
    assert status == 201, run
    directory = bridge.simulations.storage / run["goal_id"]
    (directory / "rollout").mkdir()
    (directory / "rollout/overview.mp4").write_bytes(b"0123456789")
    (directory / "state/result.json").write_text(json.dumps({"status": "failed", "success": False}))
    bridge.spawns[0][2].returncode = 0
    bridge.simulations._reap()
    return run, directory


def test_routes_require_identity_manager_permission_and_csrf(bridge):
    assert bridge.request("/api/simulations")[0] == 401
    assert bridge.request("/api/simulations", actor=bridge.employee)[0] == 200
    assert (
        bridge.request(
            "/api/simulations", method="POST", actor=bridge.employee, value={"request_id": "fold"}
        )[0]
        == 403
    )
    assert (
        bridge.request(
            "/api/simulations",
            method="POST",
            actor=bridge.manager,
            value={"request_id": "fold"},
            headers={"X-Relay-Token": "wrong"},
        )[0]
        == 403
    )
    assert not bridge.spawns


def test_fixed_recipe_requires_exact_approved_pair(bridge):
    assert launch(bridge)[0] == 403
    bridge.save_and_release()
    assert launch(bridge)[0] == 403
    approve_recipe(bridge)
    assert launch(bridge, command="python arbitrary.py")[0] == 400
    assert not bridge.spawns


def test_start_is_idempotent_and_allows_only_one_active_world(bridge):
    approve_recipe(bridge)
    status, _, run = launch(bridge)
    assert status == 201, run
    assert launch(bridge)[2]["goal_id"] == run["goal_id"]
    assert launch(bridge, request_id="second-world")[0] == 409
    assert len(bridge.spawns) == 1
    goal = bridge.tasks.get(bridge.manager["user"], run["goal_id"])
    assert goal["objective"] == OBJECTIVE and goal["success_criteria"] == SUCCESS_CRITERIA
    assert set(goal["assignments"]) == {"left", "right"}
    command, _, _ = bridge.spawns[0]
    assert "relay_lab.studio.team_worker" in command
    assert not any("relay_worker_" in part for part in command)
    directory = bridge.simulations.storage / run["goal_id"]
    assert (directory / "left.token").stat().st_mode & 0o777 == 0o600
    assert len(bridge.workers.list(bridge.manager["user"])["workers"]) == 2


def test_history_limit_keeps_newest_and_all_unfinished_shutdowns(bridge):
    for number in range(105):
        directory = bridge.simulations.storage / f"goal-{number:016x}"
        directory.mkdir()
        value = {
            "goal_id": directory.name,
            "registry_id": bridge.registry.registry_id,
            "created_at": number,
            "status": "failed",
            "workers": [],
        }
        if number == 0:
            value["status"] = "running"
        if number == 1:
            value["cleanup_pending"] = True
        (directory / "manifest.json").write_text(json.dumps(value))
    manifests = bridge.simulations._manifests()
    assert len(manifests) == 102
    assert manifests[0]["goal_id"] == "goal-0000000000000068"
    assert {row["goal_id"] for row in manifests[-2:]} == {
        "goal-0000000000000000",
        "goal-0000000000000001",
    }
    approve_recipe(bridge)
    assert launch(bridge)[0] == 409
    assert not bridge.spawns


def test_hosted_recipe_is_unavailable_and_cannot_spawn(bridge, monkeypatch):
    monkeypatch.undo()
    bridge.simulations.hosted = True
    status, _, value = bridge.request("/api/simulations", actor=bridge.manager)
    assert status == 200 and value["available"] is False
    assert "local" in value["reason"]
    assert launch(bridge)[0] == 403
    assert not bridge.spawns


def test_native_readiness_requires_both_pinned_base_models_and_sidecar(bridge, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setenv("HF_HOME", str(bridge.service.root / "empty-model-cache"))
    monkeypatch.setattr(
        simulations,
        "recipe_workspace",
        lambda: {
            "agents": [{"family": "smolvla"}, {"family": "xvla"}],
        },
    )
    available, reason = bridge.simulations.prerequisites()
    assert not available
    assert "HuggingFaceVLA/smolvla_libero" in reason
    assert "lerobot/xvla-folding" in reason
    assert "X-VLA native adapter weights" in reason


def test_two_smol_instances_require_their_base_assets_but_no_xvla(bridge, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setenv("HF_HOME", str(bridge.service.root / "empty-model-cache"))
    monkeypatch.setattr(
        simulations,
        "recipe_workspace",
        lambda: {
            "agents": [{"family": "smolvla"}, {"family": "smolvla"}],
        },
    )
    monkeypatch.setattr(simulations.platform, "system", lambda: "Linux")
    monkeypatch.setattr(simulations.platform, "machine", lambda: "x86_64")
    available, reason = bridge.simulations.prerequisites()
    assert not available
    assert "Apple Silicon" in reason and "smolvla_libero" in reason
    assert "X-VLA" not in reason and "xvla-folding" not in reason


def test_spawn_failure_retires_credentials_and_deletes_token_files(bridge, monkeypatch):
    approve_recipe(bridge)

    def fail(*args, **kwargs):
        raise OSError("process creation unavailable")

    monkeypatch.setattr(simulations.subprocess, "Popen", fail)
    assert launch(bridge)[0] == 503
    workers = bridge.workers.list(bridge.manager["user"])["workers"]
    assert len(workers) == 2 and all(not row["active"] for row in workers)
    run = bridge.simulations.list(bridge.manager["user"])["runs"][0]
    assert run["status"] == "failed"
    assert bridge.tasks.get(bridge.manager["user"], run["goal_id"])["status"] == "cancelled"
    assert not list((bridge.simulations.storage / run["goal_id"]).glob("*.token"))


def test_restart_interrupts_old_world_and_does_not_resume(bridge):
    approve_recipe(bridge)
    _, _, run = launch(bridge)
    restarted = simulations.SimulationService(
        bridge.service, bridge.access, bridge.registry, bridge.tasks, bridge.workers
    )
    try:
        record = restarted.get(bridge.manager["user"], run["goal_id"])
        assert record["status"] == "interrupted"
        assert bridge.tasks.get(bridge.manager["user"], run["goal_id"])["status"] == "cancelled"
        assert len(bridge.spawns) == 1
        assert all(
            not row["active"] for row in bridge.workers.list(bridge.manager["user"])["workers"]
        )
        assert not list((restarted.storage / run["goal_id"]).glob("*.token"))
        assert (
            restarted.start(
                bridge.manager["user"], {"request_id": "fold-one"}, origin="http://127.0.0.1:8770"
            )["status"]
            == "interrupted"
        )
        assert len(bridge.spawns) == 1
    finally:
        restarted.close()


def test_cancellation_uses_shared_task_and_retires_runtime_when_it_exits(bridge):
    approve_recipe(bridge)
    _, _, run = launch(bridge)
    task = bridge.tasks.get(bridge.manager["user"], run["goal_id"])
    status, _, cancelled = bridge.request(
        f"/api/tasks/{run['goal_id']}/cancel",
        method="POST",
        actor=bridge.manager,
        value={
            "request_id": "operator-stop",
            "expected_revision": task["revision"],
            "reason": "Stop the current simulation",
        },
    )
    assert status == 200 and cancelled["status"] == "cancelled"
    directory = bridge.simulations.storage / run["goal_id"]
    (directory / "state/result.json").write_text(
        json.dumps({"status": "cancelled", "success": False})
    )
    bridge.spawns[0][2].returncode = 0
    bridge.simulations._reap()
    assert bridge.simulations.get(bridge.manager["user"], run["goal_id"])["status"] == "cancelled"
    assert all(not row["active"] for row in bridge.workers.list(bridge.manager["user"])["workers"])


@pytest.mark.parametrize("receipt", [None, {"status": "succeeded", "success": True}])
def test_failed_process_cannot_be_presented_as_success(bridge, receipt):
    approve_recipe(bridge)
    _, _, run = launch(bridge)
    directory = bridge.simulations.storage / run["goal_id"]
    if receipt:
        (directory / "state/result.json").write_text(json.dumps(receipt))
    bridge.spawns[0][2].returncode = 1
    bridge.simulations._reap()
    value = bridge.simulations.get(bridge.manager["user"], run["goal_id"])
    assert value["status"] == "failed"
    assert not (value["result"] or {}).get("success", False)


def test_unconfirmed_cleanup_holds_new_launch_until_backend_reconciliation(bridge, monkeypatch):
    approve_recipe(bridge)
    _, _, run = launch(bridge)
    original = bridge.tasks.cancel_managed

    def unavailable(*args):
        raise OSError("Spacetime unavailable")

    monkeypatch.setattr(bridge.tasks, "cancel_managed", unavailable)
    bridge.spawns[0][2].returncode = 1
    bridge.simulations._reap()
    value = bridge.simulations.list(bridge.manager["user"])
    assert value["runs"][0]["cleanup_pending"] and not value["available"]
    assert launch(bridge, "next-world")[0] == 409
    assert all(
        not worker["active"] for worker in bridge.workers.list(bridge.manager["user"])["workers"]
    )
    manifest = bridge.simulations._manifests()[0]
    monkeypatch.setattr(bridge.tasks, "cancel_managed", original)
    bridge.simulations._cleanup(manifest, "A different retry message must not replace the intent")
    assert not bridge.simulations.get(bridge.manager["user"], run["goal_id"])["cleanup_pending"]
    task = bridge.tasks.get(bridge.manager["user"], run["goal_id"])
    assert task["status"] == "cancelled" and task["cancel_reason"] == manifest["cleanup_reason"]


def test_replay_verification_is_active_and_never_a_completed_task(bridge):
    approve_recipe(bridge)
    _, _, run = launch(bridge)
    directory = bridge.simulations.storage / run["goal_id"]
    (directory / "state/result.json").write_text(
        json.dumps({"status": "verifying", "success": False})
    )
    value = bridge.simulations.list(bridge.manager["user"])
    assert value["active"]["status"] == "verifying"
    assert not value["active"]["result"]["success"]
    assert launch(bridge, "another-world")[0] == 409


def test_zero_exit_without_geometry_and_verification_is_not_success(bridge):
    approve_recipe(bridge)
    _, _, run = launch(bridge)
    directory = bridge.simulations.storage / run["goal_id"]
    (directory / "state/result.json").write_text(
        json.dumps({"status": "succeeded", "success": True})
    )
    bridge.spawns[0][2].returncode = 0
    bridge.simulations._reap()
    value = bridge.simulations.get(bridge.manager["user"], run["goal_id"])
    assert value["status"] == "failed" and value["result"]["success"] is False


def test_artifact_auth_range_and_private_file_allowlist(bridge):
    run, directory = completed(bridge)
    url = f"/api/simulations/{run['goal_id']}/artifacts/overview.mp4"
    assert bridge.request(url)[0] == 401
    status, headers, body = bridge.request(
        url, actor=bridge.employee, headers={"Range": "bytes=2-5"}
    )
    assert status == 206 and body == b"2345"
    assert headers["Content-Range"] == "bytes 2-5/10"
    assert headers["Cache-Control"] == "private, no-store"
    assert bridge.request(url, actor=bridge.manager, headers={"Range": "bytes=-3"})[2] == b"789"
    assert bridge.request(url, actor=bridge.manager, headers={"Range": "bytes=99-"})[0] == 416
    (directory / "rollout/worker.log").write_text("private")
    assert bridge.request(url.replace("overview.mp4", "worker.log"), actor=bridge.manager)[0] == 400
    assert bridge.request(url.replace("overview.mp4", "../left.token"), actor=bridge.manager)[
        0
    ] in (400, 404)


def test_artifact_rejects_invalid_id_empty_oversized_and_symlink_files(bridge, tmp_path):
    run, directory = completed(bridge)
    actor = bridge.manager["user"]
    with pytest.raises(ValueError):
        bridge.simulations.artifact(actor, "../outside", "overview.mp4")
    path = directory / "rollout/overview.mp4"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        bridge.simulations.artifact(actor, run["goal_id"], "overview.mp4")
    with path.open("wb") as stream:
        stream.truncate(simulations.MAX_ARTIFACT_BYTES + 1)
    with pytest.raises(ValueError):
        bridge.simulations.artifact(actor, run["goal_id"], "overview.mp4")
    path.unlink()
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"secret")
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        bridge.simulations.artifact(actor, run["goal_id"], "overview.mp4")


def test_artifact_rejects_symlink_rollout_directory(bridge, tmp_path):
    run, directory = completed(bridge)
    path = directory / "rollout"
    (path / "overview.mp4").unlink()
    path.rmdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "overview.mp4").write_bytes(b"private content")
    path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        bridge.simulations.artifact(bridge.manager["user"], run["goal_id"], "overview.mp4")


def managed_goal(bridge, *, claim=False):
    approve_recipe(bridge)
    _, _, run = launch(bridge)
    directory = bridge.simulations.storage / run["goal_id"]
    workers = {
        arm: bridge.workers.authenticate((directory / f"{arm}.token").read_text())
        for arm in ("left", "right")
    }
    goal = bridge.tasks.get(bridge.manager["user"], run["goal_id"])
    if claim:
        for arm, worker in workers.items():
            goal = bridge.tasks.claim(
                worker,
                goal["id"],
                {
                    "request_id": f"claim-{arm}",
                    "expected_revision": goal["revision"],
                },
            )
    return goal, workers


@pytest.mark.parametrize("claim", [False, True])
def test_managed_cancel_stops_queued_or_running_pair(bridge, claim):
    goal, workers = managed_goal(bridge, claim=claim)
    result = bridge.tasks.cancel_managed(
        goal["id"], [w["id"] for w in workers.values()] if claim else [], "Simulation host stopped"
    )
    assert result["status"] == "cancelled"
    assert result["cancel_reason"] == "Simulation host stopped"
    assert all(a["status"] == "cancelled" for a in result["assignments"].values())
    assert result["events"][-1]["summary"] == "Simulation runtime stopped; see the task reason."
    state = bridge.task_journal.runs[goal["id"]]["state"]
    assert list(state["receipts"].values())[-1]["identity"] == "simulation-supervisor"


def test_managed_cancel_refuses_another_runtime_and_non_simulation_claim(bridge):
    goal, workers = managed_goal(bridge, claim=True)
    with pytest.raises(PermissionError, match="not owned"):
        bridge.tasks.cancel_managed(goal["id"], [workers["left"]["id"]], "Not our right worker")
    assert bridge.tasks.get(bridge.manager["user"], goal["id"])["status"] == "running"
    # The API already fixes this field at authenticated claim time; corrupt it
    # here only to exercise the trusted supervisor's additional scope check.
    bridge.task_journal.runs[goal["id"]]["state"]["goal"]["assignments"]["left"][
        "execution_kind"
    ] = "observation"
    with pytest.raises(PermissionError, match="not owned"):
        bridge.tasks.cancel_managed(goal["id"], [w["id"] for w in workers.values()], "Stop")


def test_managed_cancel_works_after_release_retirement_and_never_overwrites_terminal(bridge):
    goal, workers = managed_goal(bridge, claim=True)
    left = workers["left"]
    bridge.registry.transition(
        bridge.manager["user"], left["model_id"], "retired", left["release_revision"]
    )
    result = bridge.tasks.cancel_managed(goal["id"], [w["id"] for w in workers.values()], "Retired")
    assert result["status"] == "cancelled"
    assert (
        bridge.tasks.cancel_managed(
            goal["id"], [w["id"] for w in workers.values()], "Another reason"
        )
        == result
    )


@pytest.mark.parametrize("point", ["prepare", "commit"])
def test_managed_cancel_reconciles_uncertain_write_without_repeating_it(bridge, point):
    goal, workers = managed_goal(bridge, claim=True)
    worker_ids = [w["id"] for w in workers.values()]
    setattr(bridge.task_journal, "fail_after_" + point, True)
    with pytest.raises(SpacetimeError):
        bridge.tasks.cancel_managed(goal["id"], worker_ids, "Host stopped")
    result = bridge.tasks.cancel_managed(goal["id"], worker_ids, "Host stopped")
    assert result["status"] == "cancelled" and result["revision"] == goal["revision"] + 1
    assert bridge.task_journal.commits == 3  # Two claims, one cancellation.


def test_pending_managed_cancel_cannot_reuse_request_with_different_reason(bridge):
    goal, workers = managed_goal(bridge, claim=True)
    worker_ids = [w["id"] for w in workers.values()]
    bridge.task_journal.fail_after_prepare = True
    with pytest.raises(SpacetimeError):
        bridge.tasks.cancel_managed(goal["id"], worker_ids, "Original stop reason")
    with pytest.raises(TaskConflict, match="reused"):
        bridge.tasks.cancel_managed(goal["id"], worker_ids, "Changed stop reason")
    assert (
        bridge.tasks.cancel_managed(goal["id"], worker_ids, "Original stop reason")["status"]
        == "cancelled"
    )


def test_managed_cancel_has_no_browser_or_worker_route(bridge):
    goal, workers = managed_goal(bridge)
    payload = {"worker_ids": [w["id"] for w in workers.values()], "reason": "Untrusted request"}
    assert (
        bridge.request(
            f"/api/tasks/{goal['id']}/cancel_managed",
            method="POST",
            actor=bridge.manager,
            value=payload,
        )[0]
        == 404
    )
    directory = bridge.simulations.storage / goal["id"]
    assert (
        bridge.request(
            f"/api/agent/tasks/{goal['id']}/cancel_managed",
            method="POST",
            value=payload,
            headers={"Authorization": "Bearer " + (directory / "left.token").read_text()},
        )[0]
        == 404
    )
    assert bridge.tasks.get(bridge.manager["user"], goal["id"])["status"] == "queued"


def _write_receipt_json(path, value):
    path.write_text(json.dumps(value))


def _native_receipt(bridge, *, reports=True, physics_valid=True):
    """Real scoped task reports; synthetic evidence never executes a model/world."""
    from relay_lab.studio import simulation_recipe

    model_bytes = {arm: f"synthetic-native-{arm}".encode() for arm in ("left", "right")}
    model_hashes = {arm: hashlib.sha256(raw).hexdigest() for arm, raw in model_bytes.items()}
    bridge.monkeypatch.setattr(simulation_recipe, "PINNED_HASHES", model_hashes)
    goal, workers = managed_goal(bridge, claim=True)
    if reports:
        for arm, worker in workers.items():
            goal = bridge.tasks.report(
                worker,
                goal["id"],
                {
                    "request_id": f"complete-{arm}",
                    "expected_revision": goal["revision"],
                    "status": "succeeded",
                    "progress": 100,
                    "summary": "Synthetic test evidence confirms the requested simulation.",
                    "evidence": ["fixture://bound-native-replay"],
                },
            )
    directory = bridge.simulations.storage / goal["id"]
    rollout = directory / "rollout"
    (rollout / "verification").mkdir(parents=True)
    (rollout / "models").mkdir()
    for arm, raw in model_bytes.items():
        (rollout / f"models/{arm}.pt").write_bytes(raw)
    summary = {
        "status": "succeeded",
        "success": True,
        "ticks": 10,
        "metrics": {"success": True, "physics_valid": physics_valid},
        "models": {
            arm: {"foundation_vla": True, "checkpoint_sha256": digest}
            for arm, digest in model_hashes.items()
        },
        "training_provenance": "x" * (300 * 1024),
    }
    _write_receipt_json(rollout / "summary.json", summary)
    _write_receipt_json(rollout / "final.json", {"tick": 10, "metrics": summary["metrics"]})
    (rollout / "events.jsonl.gz").write_bytes(b"synthetic bound action trace; not a real rollout")
    hashes = {
        name: hashlib.sha256((rollout / name).read_bytes()).hexdigest()
        for name in ("summary.json", "final.json", "events.jsonl.gz")
    }
    audit = {
        "verified": True,
        "source_sha256": hashes,
        "model_snapshot_sha256": {
            f"{arm}_adapter_sha256": digest for arm, digest in model_hashes.items()
        },
    }
    _write_receipt_json(rollout / "native-audit.json", audit)
    verification = {
        "kind": "independent_mujoco_action_reexecution",
        "verified": True,
        "final_metrics": summary["metrics"],
        "source_artifact_sha256": hashes,
        "native_commands": {
            "verified": True,
            "same_source_artifacts": True,
            "audit_sha256": hashlib.sha256(
                (rollout / "native-audit.json").read_bytes()
            ).hexdigest(),
        },
    }
    _write_receipt_json(rollout / "verification/verification.json", verification)
    result = {
        "task_id": goal["id"],
        "recipe_id": RECIPE_ID,
        "status": "succeeded",
        "success": True,
        "summary": summary,
        "verification": verification,
    }
    _write_receipt_json(directory / "state/result.json", result)
    assert 256 * 1024 < (directory / "state/result.json").stat().st_size <= MAX_STATE_BYTES
    return SimpleNamespace(
        goal=goal,
        workers=workers,
        directory=directory,
        rollout=rollout,
        result=result,
        audit=audit,
        verification=verification,
    )


def _rebind_receipt_fixture(fixture):
    """Keep synthetic hashes consistent to isolate a deeper validation failure."""
    _write_receipt_json(fixture.rollout / "summary.json", fixture.result["summary"])
    hashes = {
        name: hashlib.sha256((fixture.rollout / name).read_bytes()).hexdigest()
        for name in ("summary.json", "final.json", "events.jsonl.gz")
    }
    fixture.audit["source_sha256"] = hashes
    _write_receipt_json(fixture.rollout / "native-audit.json", fixture.audit)
    fixture.verification["source_artifact_sha256"] = hashes
    fixture.verification["native_commands"]["audit_sha256"] = hashlib.sha256(
        (fixture.rollout / "native-audit.json").read_bytes()
    ).hexdigest()
    _write_receipt_json(fixture.rollout / "verification/verification.json", fixture.verification)
    _write_receipt_json(fixture.directory / "state/result.json", fixture.result)


def _old_missing_receipt(bridge, fixture):
    manifest = json.loads((fixture.directory / "manifest.json").read_text())
    manifest.update(
        status="failed", reason=simulations.MISSING_RECEIPT_REASON, cleanup_pending=False
    )
    _write_receipt_json(fixture.directory / "manifest.json", manifest)
    # The historical worker already exited; recovery cannot restart it.
    bridge.simulations.processes.pop(fixture.goal["id"])
    bridge.spawns[0][2].returncode = 0
    return manifest


def test_large_native_receipt_is_visible_while_verifying_and_reaps_success(bridge):
    fixture = _native_receipt(bridge)
    receipt = fixture.directory / "state/result.json"
    _write_receipt_json(receipt, {**fixture.result, "status": "verifying", "success": False})
    visible = bridge.simulations.get(bridge.manager["user"], fixture.goal["id"])
    assert visible["status"] == "verifying"
    assert visible["result"]["summary"] == fixture.result["summary"]
    assert visible["result"]["success"] is False
    _write_receipt_json(receipt, fixture.result)
    commits = bridge.task_journal.commits
    bridge.spawns[0][2].returncode = 0
    bridge.simulations._reap()
    visible = bridge.simulations.get(bridge.manager["user"], fixture.goal["id"])
    assert visible["status"] == "succeeded" and visible["result"]["success"] is True
    assert bridge.tasks.get(bridge.manager["user"], fixture.goal["id"])["status"] == "succeeded"
    assert bridge.task_journal.commits == commits and len(bridge.spawns) == 1


def test_receipt_over_shared_state_bound_is_not_read_or_recovered(bridge):
    fixture = _native_receipt(bridge)
    fixture.result["padding"] = "x" * MAX_STATE_BYTES
    _write_receipt_json(fixture.directory / "state/result.json", fixture.result)
    assert bridge.simulations.get(bridge.manager["user"], fixture.goal["id"])["result"] is None
    bridge.spawns[0][2].returncode = 0
    bridge.simulations._reap()
    manifest = json.loads((fixture.directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["reason"] == simulations.MISSING_RECEIPT_REASON
    assert bridge.simulations._recover_missing_receipt(manifest) is False


def test_startup_recovers_only_local_receipt_with_bound_evidence_and_both_reports(bridge):
    fixture = _native_receipt(bridge)
    old = _old_missing_receipt(bridge, fixture)
    before = json.dumps(bridge.task_journal.runs, sort_keys=True)
    commits = bridge.task_journal.commits
    restarted = simulations.SimulationService(
        bridge.service, bridge.access, bridge.registry, bridge.tasks, bridge.workers
    )
    try:
        value = restarted.get(bridge.manager["user"], fixture.goal["id"])
        assert value["status"] == "succeeded" and value["result"]["success"] is True
        path = fixture.directory / "manifest.json"
        recovered = json.loads(path.read_text())
        assert recovered["receipt_recovery"]["previous_status"] == old["status"]
        assert recovered["receipt_recovery"]["previous_reason"] == old["reason"]
        assert bridge.task_journal.commits == commits
        assert json.dumps(bridge.task_journal.runs, sort_keys=True) == before
        assert len(bridge.spawns) == 1 and not restarted.processes
        content = path.read_bytes()
        assert restarted._recover_missing_receipt(recovered) is False
        assert path.read_bytes() == content
    finally:
        restarted.close()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_receipt",
        "small_receipt",
        "wrong_reason",
        "cancelled_manifest",
        "active_manifest",
        "cleanup_pending",
        "wrong_task",
        "wrong_recipe",
        "failed_receipt",
        "missing_summary",
        "different_summary",
        "missing_replay",
        "false_replay",
        "false_native_commands",
        "unbound_native_commands",
        "missing_audit",
        "false_audit",
        "tampered_audit",
        "tampered_final",
        "tampered_events",
        "wrong_workers",
        "non_simulation_worker",
        "cancelled_task",
        "unreported_task",
        "pending_snapshot",
        "backend_unavailable",
        "invalid_physics",
    ],
)
def test_old_missing_receipt_cannot_recover_without_every_required_proof(
    bridge, monkeypatch, failure
):
    fixture = _native_receipt(
        bridge,
        reports=failure not in {"cancelled_task", "unreported_task"},
        physics_valid=failure != "invalid_physics",
    )
    manifest = _old_missing_receipt(bridge, fixture)
    receipt = fixture.directory / "state/result.json"
    replay = fixture.rollout / "verification/verification.json"
    if failure == "missing_receipt":
        receipt.unlink()
    elif failure == "small_receipt":
        fixture.result["summary"].pop("training_provenance")
    elif failure == "wrong_reason":
        manifest["reason"] = "An actual native policy failed."
    elif failure in {"cancelled_manifest", "active_manifest"}:
        manifest["status"] = "cancelled" if failure == "cancelled_manifest" else "running"
    elif failure == "cleanup_pending":
        manifest["cleanup_pending"] = True
    elif failure == "wrong_task":
        fixture.result["task_id"] = "goal-0000000000000000"
    elif failure == "wrong_recipe":
        fixture.result["recipe_id"] = "another-recipe"
    elif failure == "failed_receipt":
        fixture.result.update(status="failed", success=False)
    elif failure == "missing_summary":
        (fixture.rollout / "summary.json").unlink()
    elif failure == "different_summary":
        _write_receipt_json(fixture.rollout / "summary.json", {"success": True})
    elif failure == "missing_replay":
        replay.unlink()
    elif failure == "false_replay":
        fixture.verification["verified"] = False
    elif failure == "false_native_commands":
        fixture.verification["native_commands"]["verified"] = False
    elif failure == "unbound_native_commands":
        fixture.verification["native_commands"]["same_source_artifacts"] = False
    elif failure == "missing_audit":
        (fixture.rollout / "native-audit.json").unlink()
    elif failure == "false_audit":
        fixture.audit["verified"] = False
        path = fixture.rollout / "native-audit.json"
        _write_receipt_json(path, fixture.audit)
        fixture.verification["native_commands"]["audit_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    elif failure.startswith("tampered_"):
        name = {
            "tampered_audit": "native-audit.json",
            "tampered_final": "final.json",
            "tampered_events": "events.jsonl.gz",
        }[failure]
        with (fixture.rollout / name).open("ab") as stream:
            stream.write(b" ")
    elif failure == "wrong_workers":
        manifest["workers"][0] = "worker-0000000000000000"
    elif failure == "non_simulation_worker":
        bridge.task_journal.runs[fixture.goal["id"]]["state"]["goal"]["assignments"]["right"][
            "execution_kind"
        ] = "external"
    elif failure == "cancelled_task":
        bridge.tasks.cancel(
            bridge.manager["user"],
            fixture.goal["id"],
            {
                "request_id": "cancel-before-recovery",
                "expected_revision": fixture.goal["revision"],
                "reason": "Operator cancelled the task.",
            },
        )
    elif failure == "pending_snapshot":
        bridge.task_journal.runs[fixture.goal["id"]]["phase"] = "executing"
    elif failure == "backend_unavailable":

        def unavailable(*args):
            raise SpacetimeError("Authoritative task read unavailable")

        monkeypatch.setattr(bridge.tasks, "_load", unavailable)
    if failure != "missing_receipt":
        _write_receipt_json(receipt, fixture.result)
    if failure in {
        "false_replay",
        "false_native_commands",
        "unbound_native_commands",
        "false_audit",
    }:
        _write_receipt_json(replay, fixture.verification)
    _write_receipt_json(fixture.directory / "manifest.json", manifest)
    before = (fixture.directory / "manifest.json").read_bytes()
    commits = bridge.task_journal.commits
    assert bridge.simulations._recover_missing_receipt(manifest) is False
    assert (fixture.directory / "manifest.json").read_bytes() == before
    assert bridge.task_journal.commits == commits and len(bridge.spawns) == 1


@pytest.mark.parametrize(
    "failure",
    [
        "missing_model",
        "changed_model",
        "audit_pin",
        "summary_pin",
        "assignment_pin",
        "final_metrics",
        "assignment_status",
    ],
)
def test_recovery_requires_current_model_and_physics_evidence_to_match_authority(bridge, failure):
    fixture = _native_receipt(bridge)
    manifest = _old_missing_receipt(bridge, fixture)
    model = fixture.rollout / "models/right.pt"
    if failure == "missing_model":
        model.unlink()
    elif failure == "changed_model":
        model.write_bytes(b"different native adapter")
    elif failure == "audit_pin":
        fixture.audit["model_snapshot_sha256"]["right_adapter_sha256"] = "0" * 64
    elif failure == "summary_pin":
        fixture.result["summary"]["models"]["right"]["checkpoint_sha256"] = "0" * 64
    elif failure == "assignment_pin":
        assignment = bridge.task_journal.runs[fixture.goal["id"]]["state"]["goal"]["assignments"][
            "right"
        ]
        assignment["spec"]["source"]["revision"] = "0" * 64
    elif failure == "assignment_status":
        bridge.task_journal.runs[fixture.goal["id"]]["state"]["goal"]["assignments"]["right"][
            "status"
        ] = "running"
    elif failure == "final_metrics":
        final = json.loads((fixture.rollout / "final.json").read_text())
        final["metrics"]["success"] = False
        _write_receipt_json(fixture.rollout / "final.json", final)
    _rebind_receipt_fixture(fixture)
    before = (fixture.directory / "manifest.json").read_bytes()
    commits = bridge.task_journal.commits
    assert bridge.simulations._recover_missing_receipt(manifest) is False
    assert (fixture.directory / "manifest.json").read_bytes() == before
    assert bridge.task_journal.commits == commits and len(bridge.spawns) == 1


@pytest.mark.parametrize("artifact", ["native-audit.json", "summary.json"])
def test_recovery_rejects_evidence_replaced_after_capture(bridge, monkeypatch, artifact):
    fixture = _native_receipt(bridge)
    manifest = _old_missing_receipt(bridge, fixture)
    load = bridge.tasks._load
    replacements = []

    def replace_after_capture(task_id):
        # The authority lookup follows captured summary/audit validation.
        path = fixture.rollout / artifact
        with path.open("ab") as stream:
            stream.write(b" ")
        replacements.append(artifact)
        return load(task_id)

    monkeypatch.setattr(bridge.tasks, "_load", replace_after_capture)
    before = (fixture.directory / "manifest.json").read_bytes()
    commits = bridge.task_journal.commits
    assert bridge.simulations._recover_missing_receipt(manifest) is False
    assert replacements == [artifact]
    assert (fixture.directory / "manifest.json").read_bytes() == before
    assert bridge.task_journal.commits == commits and len(bridge.spawns) == 1
