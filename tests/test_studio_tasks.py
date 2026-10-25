"""Objective dispatch uses a durable journal; tests never move hardware."""

import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from relay_lab.factory.coordination import MessageQueue
from relay_lab.spacetime import SpacetimeError
from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.service import StudioService, example_workspace
from relay_lab.studio.tasks import TaskConflict, TaskService

MANAGER = {"id": "owner", "role": "manager", "active": True}
EMPLOYEE = {"id": "operator", "role": "employee", "active": True}


class Journal:
    """Fresh coordinator handles share committed state, matching a restart."""

    def __init__(self):
        self.runs, self.batches, self.queues = {}, {}, {}
        self.commits = 0
        self.snapshots = 0
        self.fail_after_prepare = False
        self.fail_after_commit = False
        self.corrupt_readback = False

    def factory(self):
        journal = self

        class Coordinator:
            def __init__(self):
                self.client = self

            def _sql(self, query):
                match = re.search(r"run_id = '([^']+)'", query)
                owner = re.search(r"worker_id = '([^']+)'", query)
                return [
                    {k: journal.runs[task_id][k] for k in ("run_id", "revision", "phase")}
                    for task_id in journal.runs
                    if (not match or match[1] == task_id)
                    and (not owner or journal.runs[task_id]["worker_id"] == owner[1])
                ]

            def create(self, run_id, worker_id, robots, jobs, resources, *, max_ticks, state):
                self.run_id, self.worker_id = run_id, worker_id
                assert run_id not in journal.runs
                journal.runs[run_id] = {
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "revision": 0,
                    "phase": "ready",
                    "tick": 0,
                    "status": "running",
                    "state": copy.deepcopy(state),
                }
                journal.queues[run_id] = MessageQueue({r["robot_id"] for r in robots})

            def snapshot(self):
                journal.snapshots += 1
                if self.run_id not in journal.runs:
                    raise SpacetimeError("Factory run not found")
                result = copy.deepcopy(journal.runs[self.run_id])
                if journal.corrupt_readback and result["revision"]:
                    result["state"]["goal"]["objective"] = "CORRUPT"
                return result

            def prepare(self, revision, commands, messages):
                run = journal.runs[self.run_id]
                assert run["revision"] == revision and commands == []
                assert run["phase"] == "ready"
                for message in messages:
                    journal.queues[self.run_id].publish(message, revision)
                journal.batches[(self.run_id, revision)] = {
                    "prepare_json": json.dumps({"commands": [], "messages": messages}),
                    "result_json": "",
                }
                run["phase"] = "executing"
                if journal.fail_after_prepare:
                    journal.fail_after_prepare = False
                    raise SpacetimeError("Lost prepare acknowledgement")

            def commit(self, revision, state, outcomes, *, messages):
                run = journal.runs[self.run_id]
                assert run["revision"] == revision and run["phase"] == "executing"
                assert state["tick"] == revision + 1
                assert state["jobs"] == run["state"]["jobs"] and state["success"] is False
                for message in messages:
                    journal.queues[self.run_id].publish(message, revision + 1)
                result = {"state": state, "outcomes": outcomes, "messages": messages}
                journal.batches[(self.run_id, revision)]["result_json"] = json.dumps(result)
                run.update(
                    state=copy.deepcopy(state),
                    tick=revision + 1,
                    revision=revision + 1,
                    phase="ready",
                )
                journal.commits += 1
                if journal.fail_after_commit:
                    journal.fail_after_commit = False
                    raise SpacetimeError("Lost commit acknowledgement")

            def _batch(self, revision):
                return copy.deepcopy(journal.batches.get((self.run_id, revision)))

        return Coordinator()


@pytest.fixture
def setup(tmp_path):
    journal = Journal()
    studio = StudioService(tmp_path, coordinator_factory=journal.factory)
    studio.save(example_workspace())
    registry = ModelRegistry(studio.storage)
    for model in registry.sync_workspace(MANAGER, studio.workspace())["models"]:
        registry.transition(MANAGER, model["id"], "released", model["revision"])
    service = TaskService(studio, registry)
    snapshot = registry.authorize_workspace(MANAGER, studio.workspace())
    workers = {
        m["spec"]["id"]: {
            "id": f"runtime-{m['spec']['id']}",
            "agent_id": m["spec"]["id"],
            "model_id": m["id"],
            "release_revision": m["revision"],
            "registry_id": registry.registry_id,
        }
        for m in snapshot["models"]
    }
    return service, studio, registry, journal, workers


def create(service, target="both", **changes):
    return service.create(
        EMPLOYEE,
        {
            "request_id": "create-1",
            "objective": "Fold the shirt.\nWait if your peer is blocked.",
            "success_criteria": ["Both sleeves folded", "No collision"],
            "target": target,
            **changes,
        },
    )


def update(task, request_id="update-1", **changes):
    return {"request_id": request_id, "expected_revision": task["revision"], **changes}


def claim(service, worker, task):
    return service.claim(worker, task["id"], update(task, f"claim-{worker['agent_id']}"))


def report(service, worker, task, request_id="report", **changes):
    return service.report(
        worker,
        task["id"],
        update(
            task,
            request_id,
            status="succeeded",
            progress=100,
            summary="Observed requested fold complete.",
            evidence=["camera://frame-42"],
            **changes,
        ),
    )


def test_objective_is_exact_and_both_assignments_must_succeed(setup):
    service, _, _, journal, workers = setup
    task = create(service)
    assert task["revision"] == 0 and task["status"] == "queued"
    assert all(a["instruction"] == task["objective"] for a in task["assignments"].values())
    assert service.inbox(workers["left"])["tasks"][0]["id"] == task["id"]
    task = claim(service, workers["left"], task)
    task = claim(service, workers["right"], task)
    task = report(service, workers["left"], task, "left-success")
    assert task["status"] == "running"
    assert task["assignments"]["left"]["evidence"] == ["camera://frame-42"]
    task = report(service, workers["right"], task, "right-success")
    assert task["status"] == "succeeded" and task["success_basis"] == "agent_reported"
    assert not task["verified_physical_outcome"]
    assert journal.runs[task["id"]]["state"]["success"] is False
    assert service.inbox(workers["left"])["tasks"][0]["status"] == "succeeded"
    packets = journal.queues[task["id"]].events.values()
    assert any(
        m["recipient"] == "broadcast" and m["payload"].get("status") == "succeeded" for m in packets
    )


def test_single_target_does_not_assign_an_unrequested_robot(setup):
    service, _, _, _, workers = setup
    task = create(service, "right")
    assert list(task["assignments"]) == ["right"]
    assert service.inbox(workers["left"])["tasks"] == []
    with pytest.raises(PermissionError):
        claim(service, workers["left"], task)
    task = claim(service, workers["right"], task)
    assert report(service, workers["right"], task)["status"] == "succeeded"


def test_creation_idempotency_and_changed_content_rejected(setup):
    service, _, _, journal, _ = setup
    task = create(service)
    assert create(service) == task and len(journal.runs) == 1
    with pytest.raises(TaskConflict, match="reused"):
        create(service, objective="Another task")


@pytest.mark.parametrize("point", ["prepare", "commit"])
def test_lost_ack_recovers_same_request_after_server_restart_without_double_claim(setup, point):
    service, studio, registry, journal, workers = setup
    task = create(service)
    request = update(task, "claim-once")
    setattr(journal, "fail_after_" + point, True)
    with pytest.raises(SpacetimeError):
        service.claim(workers["left"], task["id"], request)
    restarted = TaskService(studio, registry)
    assert restarted.get(EMPLOYEE, task["id"])["pending_update"] == (point == "prepare")
    recovered = restarted.claim(workers["left"], task["id"], request)
    assert recovered["revision"] == 1 and journal.commits == 1
    assert restarted.claim(workers["left"], task["id"], request) == recovered
    assert journal.commits == 1


def test_pending_update_rejects_another_request_but_can_be_cancelled(setup):
    service, _, _, journal, workers = setup
    task = create(service)
    journal.fail_after_prepare = True
    original = update(task, "claim-pending")
    with pytest.raises(SpacetimeError):
        service.claim(workers["left"], task["id"], original)
    with pytest.raises(TaskConflict, match="pending"):
        service.claim(workers["right"], task["id"], update(task, "other-claim"))
    cancelled = service.cancel(EMPLOYEE, task["id"], update(task, "cancel", reason="Stop"))
    assert cancelled["status"] == "cancelled" and journal.commits == 1
    assert service.claim(workers["left"], task["id"], original)["status"] == "cancelled"
    with pytest.raises(TaskConflict, match="terminal"):
        claim(service, workers["right"], cancelled)


def test_competing_runtime_cannot_take_an_assignment_and_claim_is_idempotent(setup):
    service, _, _, journal, workers = setup
    task = create(service)
    request = update(task)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(service.claim, workers["left"], task["id"], request) for _ in range(2)
        ]
        first, second = [future.result() for future in futures]
    assert first == second and journal.commits == 1
    rival = {**workers["left"], "id": "another-runtime"}
    with pytest.raises(TaskConflict, match="already owns"):
        service.claim(rival, task["id"], update(first, "steal"))
    assert service.inbox(rival)["tasks"] == []


def test_stale_update_and_conflicting_retry_rejected(setup):
    service, _, _, _, workers = setup
    task = create(service)
    claimed = claim(service, workers["left"], task)
    with pytest.raises(TaskConflict, match="revision"):
        claim(service, workers["right"], task)
    with pytest.raises(TaskConflict, match="reused"):
        service.claim(workers["left"], task["id"], update(claimed, "claim-left"))


def test_report_requires_claim_and_cannot_forge_peer_worker_or_release(setup):
    service, _, _, _, workers = setup
    task = create(service)
    with pytest.raises(PermissionError, match="Claim"):
        report(service, workers["left"], task)
    for field, value in (
        ("agent_id", "other"),
        ("model_id", "wrong"),
        ("release_revision", 999),
        ("registry_id", "wrong"),
    ):
        with pytest.raises(PermissionError):
            claim(service, {**workers["left"], field: value}, task)


def test_retirement_blocks_runtime_but_not_history_or_cancellation(setup):
    service, _, registry, _, workers = setup
    task = claim(service, workers["left"], create(service))
    worker = workers["left"]
    registry.transition(MANAGER, worker["model_id"], "retired", worker["release_revision"])
    with pytest.raises(PermissionError):
        report(service, worker, task)
    assert service.get(EMPLOYEE, task["id"])["status"] == "running"
    assert service.inbox(worker)["tasks"] == []
    assert service.cancel(EMPLOYEE, task["id"], update(task, "stop"))["status"] == "cancelled"


def test_other_registry_cannot_inspect_goal(setup, tmp_path):
    service, studio, _, _, _ = setup
    task = create(service)
    other = TaskService(studio, ModelRegistry(tmp_path / "other"))
    with pytest.raises(PermissionError):
        other.get(MANAGER, task["id"])
    assert other.list(MANAGER)["tasks"] == []


def test_only_creator_or_manager_can_cancel(setup):
    service, _, _, _, _ = setup
    task = create(service)
    with pytest.raises(PermissionError):
        service.cancel({**EMPLOYEE, "id": "another"}, task["id"], update(task))
    assert service.cancel(MANAGER, task["id"], update(task))["status"] == "cancelled"


@pytest.mark.parametrize(
    "changes",
    [
        {"progress": float("nan")},
        {"progress": -1},
        {"progress": 101},
        {"progress": True},
        {"status": "succeeded", "progress": 99},
        {"status": "done"},
        {"evidence": ["x"] * 9},
        {"summary": "x" * 1001},
    ],
)
def test_invalid_outcomes_do_not_prepare_a_batch(setup, changes):
    service, _, _, journal, workers = setup
    task = claim(service, workers["left"], create(service))
    value = update(task, "bad", status="running", progress=10, summary="Working", evidence=[])
    value.update(changes)
    with pytest.raises(ValueError):
        service.report(workers["left"], task["id"], value)
    assert len(journal.batches) == 1 and journal.runs[task["id"]]["phase"] == "ready"


def test_blocked_feedback_and_recovery_are_visible_to_peer(setup):
    service, _, _, _, workers = setup
    task = claim(service, workers["left"], create(service))
    task = service.report(
        workers["left"],
        task["id"],
        update(
            task,
            "obstacle",
            status="blocked",
            progress=20,
            summary="Obstacle on shared path",
            evidence=[],
        ),
    )
    assert task["status"] == "blocked"
    peer = service.inbox(workers["right"])["tasks"][0]
    assert peer["assignments"]["left"]["summary"] == "Obstacle on shared path"
    task = service.report(
        workers["left"],
        task["id"],
        update(task, "clear", status="running", progress=25, summary="Path now clear", evidence=[]),
    )
    assert task["status"] == "running"


def test_exact_readback_failure_never_confirms_dispatch(setup):
    service, _, _, journal, workers = setup
    task = create(service)
    journal.corrupt_readback = True
    with pytest.raises(SpacetimeError, match="readback"):
        claim(service, workers["left"], task)


def test_history_is_bounded_and_oversize_payload_rejected_before_prepare(setup):
    service, _, _, journal, workers = setup
    task = claim(service, workers["left"], create(service))
    for index in range(20):
        task = service.report(
            workers["left"],
            task["id"],
            update(
                task, f"r{index}", status="running", progress=index, summary="Moving", evidence=[]
            ),
        )
    assert len(task["events"]) == task["event_history_limit"] == 16
    before = journal.commits
    with pytest.raises((ValueError, SpacetimeError)):
        service.report(workers["left"], task["id"], update(task, "large", blob="x" * 65536))
    assert journal.commits == before


def test_sql_task_identifier_validation_precedes_access(setup):
    service, _, _, _, _ = setup
    with pytest.raises(ValueError):
        service.get(EMPLOYEE, "goal-' OR TRUE")


def test_worker_receipts_survive_peer_revision_changes_and_are_scoped(setup):
    service, studio, registry, _, workers = setup
    task = create(service)
    task = claim(service, workers["left"], task)
    task = claim(service, workers["right"], task)
    request = update(
        task, "report-once", status="running", progress=20, summary="Checked left side", evidence=[]
    )
    task = service.report(workers["left"], task["id"], request)
    task = service.report(
        workers["right"],
        task["id"],
        update(
            task,
            "right-later",
            status="running",
            progress=30,
            summary="Checked right side",
            evidence=[],
        ),
    )
    restarted = TaskService(studio, registry)
    observed = restarted.worker_get(workers["left"], task["id"])
    assert {r["request_id"] for r in observed["request_receipts"]} == {"claim-left", "report-once"}
    assert observed["assignments"]["left"]["last_request_id"] == "report-once"
    assert service.report(workers["left"], task["id"], request)["revision"] == task["revision"]
    assert observed["events"][-1]["request_id"] == "right-later"
    with pytest.raises(PermissionError, match="owns"):
        restarted.worker_get({**workers["left"], "id": "rival"}, task["id"])


def test_workspace_capacity_cannot_silently_hide_new_tasks(setup, tmp_path):
    service, studio, _, journal, _ = setup
    first = create(service)
    template = copy.deepcopy(journal.runs[first["id"]])
    for index in range(99):
        task_id = f"goal-{index:016x}"
        row = copy.deepcopy(template)
        row["run_id"] = task_id
        row["state"]["goal"]["id"] = task_id
        journal.runs[task_id] = row
    with pytest.raises(TaskConflict, match="100-task"):
        create(service, request_id="too-many")
    assert len(service.list(MANAGER)["tasks"]) == 100
    assert create(service)["id"] == first["id"]  # Retries still resolve at capacity.
    other_registry = ModelRegistry(tmp_path / "other")
    for model in other_registry.sync_workspace(MANAGER, studio.workspace())["models"]:
        other_registry.transition(MANAGER, model["id"], "released", model["revision"])
    other = TaskService(studio, other_registry)
    second = create(other)
    assert other.worker_id != service.worker_id
    assert len(other.list(MANAGER)["tasks"]) == 1
    assert second["id"] != first["id"]


def test_cancelled_assignment_remains_visible_to_its_running_worker(setup):
    service, _, _, _, workers = setup
    task = claim(service, workers["left"], create(service))
    task = service.cancel(EMPLOYEE, task["id"], update(task, "stop"))
    assert service.inbox(workers["left"])["tasks"][0]["status"] == "cancelled"
    assert service.inbox(workers["right"])["tasks"] == []
    with pytest.raises(TaskConflict, match="terminal"):
        report(service, workers["left"], task)


def test_list_uses_authoritative_metadata_but_only_reloads_changed_snapshots(setup):
    service, _, _, journal, workers = setup
    first = create(service)
    second = create(service, request_id="second")
    service.list(EMPLOYEE)
    warmed = journal.snapshots
    listed = service.list(EMPLOYEE)
    assert journal.snapshots == warmed and len(listed["tasks"]) == 2
    listed["tasks"][0]["objective"] = "Client mutation"
    assert all(t["objective"] != "Client mutation" for t in service.list(EMPLOYEE)["tasks"])
    claim(service, workers["left"], first)
    before = journal.snapshots
    changed = service.list(EMPLOYEE)
    assert journal.snapshots == before + 1
    assert next(t for t in changed["tasks"] if t["id"] == first["id"])["revision"] == 1
    journal.fail_after_prepare = True
    with pytest.raises(SpacetimeError):
        claim(service, workers["right"], second)
    before = journal.snapshots
    pending = service.list(EMPLOYEE)
    assert journal.snapshots == before + 1
    assert next(t for t in pending["tasks"] if t["id"] == second["id"])["pending_update"]


def test_full_update_budget_retains_all_receipts_and_still_allows_unicode_cancellation(setup):
    from relay_lab.factory.coordination import canonical

    service, _, _, journal, workers = setup
    task = create(service)
    task = claim(service, workers["left"], task)
    task = claim(service, workers["right"], task)
    for index in range(118):
        worker = workers["left" if index % 2 == 0 else "right"]
        request_id = f"{index:03d}" + "r" * 77
        task = service.report(
            worker,
            task["id"],
            update(
                task,
                request_id,
                status="running",
                progress=50,
                summary="Working on assigned objective",
                evidence=[],
            ),
        )
    assert task["revision"] == 120
    before = copy.deepcopy(journal.runs[task["id"]]["state"]["receipts"])
    assert len(before) == 120
    with pytest.raises(TaskConflict, match="budget reached"):
        service.report(
            workers["left"],
            task["id"],
            update(
                task,
                "over-budget",
                status="running",
                progress=50,
                summary="Would exceed tick budget",
                evidence=[],
            ),
        )
    reason = "🛑" * 1000
    task = service.cancel(EMPLOYEE, task["id"], update(task, "cancel" + "c" * 74, reason=reason))
    assert task["status"] == "cancelled" and task["cancel_reason"] == reason
    after = journal.runs[task["id"]]["state"]["receipts"]
    assert all(after[key] == value for key, value in before.items())
    for batch in journal.batches.values():
        assert len(canonical(json.loads(batch["prepare_json"])).encode()) <= 65536
        assert len(canonical(json.loads(batch["result_json"])).encode()) <= 65536


def test_maximum_text_reports_fail_before_prepare_and_cancellation_remains_available(setup):
    from relay_lab.factory.coordination import canonical

    service, _, _, journal, workers = setup
    task = create(
        service,
        objective="o" * 2000,
        title="t" * 100,
        success_criteria=["c" * 500 for _ in range(8)],
    )
    task = claim(service, workers["left"], task)
    task = claim(service, workers["right"], task)
    rejected = False
    for index in range(118):
        worker = workers["left" if index % 2 == 0 else "right"]
        value = update(
            task,
            f"{index:03d}" + "r" * 77,
            status="running",
            progress=50,
            summary="s" * 1000,
            evidence=["e" * 500 for _ in range(8)],
        )
        before = copy.deepcopy(journal.runs[task["id"]])
        try:
            task = service.report(worker, task["id"], value)
        except ValueError as error:
            assert "payload budget" in str(error)
            assert journal.runs[task["id"]] == before
            assert journal.runs[task["id"]]["phase"] == "ready"
            rejected = True
            break
    assert rejected  # This rich payload exhausts the byte budget before the tick budget.
    receipts = copy.deepcopy(journal.runs[task["id"]]["state"]["receipts"])
    task = service.cancel(EMPLOYEE, task["id"], update(task, "cancel-rich", reason="🛑" * 1000))
    assert task["status"] == "cancelled"
    assert all(
        journal.runs[task["id"]]["state"]["receipts"][key] == value
        for key, value in receipts.items()
    )
    assert len(canonical(journal.runs[task["id"]]["state"]).encode()) <= 65536


def test_oversize_unicode_report_and_creation_leave_no_prepared_state(setup):
    service, _, _, journal, workers = setup
    with pytest.raises(ValueError, match="payload budget"):
        create(service, objective="🛑" * 2000)
    assert journal.runs == {}
    task = claim(service, workers["left"], create(service))
    before = copy.deepcopy(journal.runs[task["id"]])
    with pytest.raises(ValueError, match="payload budget"):
        service.report(
            workers["left"],
            task["id"],
            update(
                task,
                "large-report",
                status="running",
                progress=50,
                summary="🛑" * 1000,
                evidence=["🛑" * 500 for _ in range(8)],
            ),
        )
    assert journal.runs[task["id"]] == before
    task = service.cancel(EMPLOYEE, task["id"], update(task, "stop-large", reason="🛑" * 1000))
    assert task["status"] == "cancelled"


def test_maximum_unicode_cancellation_can_supersede_large_pending_report(setup):
    service, _, _, journal, workers = setup
    task = claim(
        service,
        workers["left"],
        create(service, objective="o" * 2000, success_criteria=["c" * 500 for _ in range(8)]),
    )
    request = update(
        task,
        "pending-rich",
        status="running",
        progress=40,
        summary="s" * 1000,
        evidence=["e" * 500 for _ in range(8)],
    )
    journal.fail_after_prepare = True
    with pytest.raises(SpacetimeError):
        service.report(workers["left"], task["id"], request)
    task = service.cancel(EMPLOYEE, task["id"], update(task, "stop-pending", reason="🛑" * 1000))
    assert task["status"] == "cancelled"
    receipt = next(
        r
        for r in service.worker_get(workers["left"], task["id"])["request_receipts"]
        if r["request_id"] == "pending-rich"
    )
    assert receipt["cancelled_before_commit"] is True
