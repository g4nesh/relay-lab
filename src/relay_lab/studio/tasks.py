"""Released, durable objectives and explicit external-agent acknowledgements.

The existing factory reducer journal carries goal state and peer feedback. Its
physical process stages remain untouched: a worker's success report is not
simulator or hardware attestation. No VLA, endpoint, or robot is invoked here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time
from functools import wraps

from relay_lab.factory.coordination import canonical
from relay_lab.spacetime import SpacetimeError

from .governance import _actor, _text

WORKER = "studio-goals"
TERMINAL = {"succeeded", "failed", "cancelled"}
MAX_UPDATES = 120
HISTORY_LIMIT = 16
LIST_LIMIT = 100
MAX_ENVELOPE_BYTES = 64 * 1024
# Leave room for an operator's maximum Unicode cancellation reason and receipts.
ACTIVE_ENVELOPE_BYTES = 48 * 1024


class TaskConflict(ValueError):
    """Refresh the current task revision before proposing another update."""


def _locked(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        # Match the server's lock order, including release retirement.
        with self.registry.lock, self.lock:
            result = method(self, *args, **kwargs)
            if self.coordination is not None and method.__name__ in (
                "create", "claim", "report", "cancel", "supervisor_cancel"
            ) and isinstance(result, dict) and "id" in result:
                self.coordination.observe_task(result, metrics=True)
            return result

    return run


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"goal-[a-f0-9]{16}", value):
        raise ValueError("Unknown task identifier")
    return value


def _request(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a task request object")  # noqa: TRY004
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", request_id):
        raise ValueError("A request_id of 1–80 letters, digits, underscores or hyphens is required")
    canonical(value)
    return request_id


def _signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _json_size(value):
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _strings(value, label, *, maximum=8, length=500, required=False):
    if not isinstance(value, list) or len(value) > maximum or (required and not value):
        raise ValueError(f"{label} must contain {'1' if required else '0'}–{maximum} text items")
    return [_text(item, label, length) for item in value]


class TaskService:
    def __init__(self, studio_service, registry):
        self.studio = studio_service
        self.registry = registry
        self.lock = threading.RLock()
        self.worker_id = WORKER + "-" + _signature(registry.registry_id)[:12]
        self._cache = {}
        self.coordination = None

    def _coordinator(self, task_id):
        coordinator = self.studio.factory()
        coordinator.run_id, coordinator.worker_id = _id(task_id), self.worker_id
        return coordinator

    def _load(self, task_id):
        coordinator = self._coordinator(task_id)
        snapshot = coordinator.snapshot()
        state = snapshot["state"]
        if (
            snapshot.get("worker_id") != self.worker_id
            or state.get("task") != "studio-objective-v1"
            or state.get("registry_id") != self.registry.registry_id
        ):
            raise PermissionError("This task does not belong to this workspace registry")
        return coordinator, snapshot

    @staticmethod
    def _public(snapshot):
        goal = copy.deepcopy(snapshot["state"]["goal"])
        goal.update(
            revision=snapshot["revision"],
            backend="SpacetimeDB",
            readback_verified=True,
            pending_update=snapshot["phase"] == "executing",
            success_basis="agent_reported",
            verified_physical_outcome=False,
            model_release=copy.deepcopy(snapshot["state"]["model_release"]),
            event_history_limit=HISTORY_LIMIT,
            update_budget=MAX_UPDATES,
            updates_remaining=max(0, MAX_UPDATES - snapshot["revision"]),
        )
        return goal

    def _ids(self):
        coordinator = self.studio.factory()
        return coordinator.client._sql(
            f"SELECT run_id, revision, phase FROM factory_run "
            f"WHERE worker_id = '{self.worker_id}' LIMIT {LIST_LIMIT}"
        )

    def _all(self):
        goals = []
        rows = self._ids()
        active_ids = {row["run_id"] for row in rows}
        self._cache = {key: value for key, value in self._cache.items() if key in active_ids}
        for row in rows:
            # Fetch bounded individual rows rather than an unbounded state-JSON response.
            metadata = (row["revision"], row["phase"])
            cached = self._cache.get(row["run_id"])
            if cached and cached[0] == metadata:
                goals.append(copy.deepcopy(cached[1]))
                continue
            try:
                _, snapshot = self._load(row["run_id"])
            except PermissionError:
                continue
            goal = self._public(snapshot)
            if (snapshot["revision"], snapshot["phase"]) == metadata:
                # Only cache the revision just verified against authoritative metadata.
                self._cache[row["run_id"]] = (metadata, copy.deepcopy(goal))
            goals.append(goal)
        goals.sort(key=lambda goal: (goal["updated_at"], goal["id"]), reverse=True)
        return goals

    @_locked
    def list(self, actor):
        _actor(actor)
        return {"tasks": self._all(), "backend": "SpacetimeDB", "limit": LIST_LIMIT}

    @_locked
    def get(self, actor, task_id):
        _actor(actor)
        return self._public(self._load(task_id)[1])

    @_locked
    def create(self, actor, value):
        _actor(actor)
        request_id = _request(value)
        # Preserve the exact objective for the runtime; validation may not rewrite it.
        objective = value.get("objective")
        _text(objective, "Objective", 2000)
        criteria = _strings(value.get("success_criteria"), "Success criteria", required=True)
        title = _text(value.get("title") or objective.strip().splitlines()[0][:100], "Title", 100)
        target = value.get("target", "both")
        if target not in ("left", "right", "both"):
            raise ValueError("Target must be left, right, or both")
        normalized = {
            "title": title,
            "objective": objective,
            "success_criteria": criteria,
            "target": target,
        }
        timeout = value.get("timeout_seconds", 1800)
        if type(timeout) is not int or not 60 <= timeout <= 86400:
            raise ValueError("Task attention deadline must be 60–86400 seconds")
        # Keep old creation receipts compatible when no deadline was specified.
        if "timeout_seconds" in value:
            normalized["timeout_seconds"] = timeout
        signature = _signature(normalized)
        task_id = "goal-" + _signature([self.registry.registry_id, actor["id"], request_id])[:16]
        coordinator = self._coordinator(task_id)
        existing = coordinator.client._sql(
            f"SELECT run_id FROM factory_run WHERE run_id = '{task_id}'"
        )
        if existing:
            _, snapshot = self._load(task_id)
            if snapshot["state"].get("create_signature") != signature:
                raise TaskConflict("This creation request_id was reused for different task content")
            return self._public(snapshot)
        if len(self._ids()) >= LIST_LIMIT:
            raise TaskConflict(
                "This workspace has reached its 100-task history capacity; no task was queued. "
                "Export this deployment and provision a new workspace registry before adding tasks."
            )
        workspace = self.studio.validate(self.studio.workspace(), complete=True)
        release = self.registry.authorize_workspace(actor, workspace)
        now = time.time()
        assignments = {}
        for model in release["models"]:
            spec = model["spec"]
            if target != "both" and target != spec["id"]:
                continue
            assignments[spec["id"]] = {
                "agent_id": spec["id"],
                "name": spec["name"],
                "model_id": model["id"],
                "release_revision": model["revision"],
                "spec": copy.deepcopy(spec),
                "instruction": objective,
                "status": "queued",
                "worker_id": None,
                "execution_kind": None,
                "last_request_id": None,
                "progress": 0,
                "summary": "Waiting for an authenticated runtime to claim this task.",
                "evidence": [],
                "updated_at": now,
            }
        goal = {
            "id": task_id,
            **normalized,
            "timeout_seconds": timeout,
            "status": "queued",
            "created_by": actor["id"],
            "created_at": now,
            "updated_at": now,
            "workspace_name": workspace["name"],
            "communication_mode": workspace["mode"],
            "dispatch_protocol": "structured-objective-v1",
            "assignments": assignments,
            "events": [
                {
                    "revision": 0,
                    "type": "created",
                    "at": now,
                    "summary": "Objective queued for the selected released agents.",
                }
            ],
        }
        state = {
            "tick": 0,
            "success": False,
            "task": "studio-objective-v1",
            "jobs": {"objective": {"job_id": "objective", "stage": "registered"}},
            "registry_id": self.registry.registry_id,
            "model_release": release,
            "create_signature": signature,
            "goal": goal,
            "receipts": {},
        }
        if _json_size(state) > ACTIVE_ENVELOPE_BYTES - 4096:
            raise ValueError(
                "Combined objective, success criteria and model metadata exceed the task payload "
                "budget; shorten the task text or model metadata before dispatching."
            )
        canonical(state)
        coordinator.create(
            task_id,
            self.worker_id,
            [{"robot_id": a["id"], "capabilities": ["communicate"]} for a in workspace["agents"]],
            list(state["jobs"].values()),
            [],
            max_ticks=MAX_UPDATES + 1,
            state=state,
        )
        committed = coordinator.snapshot()
        if committed["state"] != state:
            raise SpacetimeError("Task creation readback differs from the submitted objective")
        return self._public(committed)

    def _worker(self, worker, state):
        if (
            not isinstance(worker, dict)
            or not isinstance(worker.get("id"), str)
            or not worker["id"]
        ):
            raise PermissionError("An authenticated agent runtime is required")
        assignment = state["goal"]["assignments"].get(worker.get("agent_id"))
        if (
            not assignment
            or worker.get("registry_id") != state["registry_id"]
            or worker.get("model_id") != assignment["model_id"]
            or type(worker.get("release_revision")) is not int
            or worker["release_revision"] != assignment["release_revision"]
        ):
            raise PermissionError("This runtime is not pinned to the task's assigned release")
        self.registry.authorize_snapshot(
            {"id": worker["id"], "active": True, "role": "employee"}, state["model_release"]
        )
        return assignment

    @_locked
    def inbox(self, worker):
        tasks = []
        for task in self._all():
            state = {
                "goal": task,
                "model_release": task["model_release"],
                "registry_id": task["model_release"]["registry_id"],
            }
            try:
                assignment = self._worker(worker, state)
            except PermissionError:
                continue
            if assignment["worker_id"] == worker["id"] or (
                assignment["worker_id"] is None and task["status"] not in TERMINAL
            ):
                tasks.append(task)
        return {"tasks": tasks, "backend": "SpacetimeDB", "limit": LIST_LIMIT}

    @_locked
    def worker_get(self, worker, task_id):
        _, snapshot = self._load(task_id)
        assignment = self._worker(worker, snapshot["state"])
        if assignment["worker_id"] not in (None, worker["id"]):
            raise PermissionError("Another runtime owns this agent assignment")
        task = self._public(snapshot)
        task["request_receipts"] = [
            {k: v for k, v in receipt.items() if k not in ("identity", "signature")}
            for receipt in snapshot["state"]["receipts"].values()
            if receipt.get("identity") == worker["id"]
        ]
        return task

    def _begin(self, task_id, value, identity, operation):
        request_id = _request(value)
        if type(value.get("expected_revision")) is not int or value["expected_revision"] < 0:
            raise ValueError("A nonnegative expected_revision is required")
        coordinator, snapshot = self._load(task_id)
        signature = _signature({"operation": operation, "identity": identity, "value": value})
        key = _signature([identity, request_id])
        receipt = snapshot["state"]["receipts"].get(key)
        if receipt:
            if receipt["signature"] != signature:
                raise TaskConflict("This request_id was reused for a different task update")
            return coordinator, snapshot, key, signature, True
        if snapshot["revision"] != value["expected_revision"]:
            raise TaskConflict("Task revision changed; refresh before updating")
        if snapshot["state"]["goal"]["status"] in TERMINAL:
            raise TaskConflict("This task is terminal; create a new task")
        if snapshot["revision"] >= MAX_UPDATES and operation != "cancel":
            raise TaskConflict("Task update budget reached; cancel this task and create a new one")
        return coordinator, snapshot, key, signature, False

    @staticmethod
    def _message(task_id, revision, kind, payload, *, intent=False):
        return {
            "message_id": f"{task_id}-{revision}-{'intent' if intent else 'feedback'}",
            "sender": "mediator",
            "recipient": "judge" if intent else "broadcast",
            "sequence": revision * 2 + (0 if intent else 1),
            "expires_tick": revision + (1 if intent else 2),
            "priority": 9 if kind == "blocked" else 5,
            "confidence": 1.0,
            "kind": kind,
            "correlation_id": None,
            "payload": payload,
        }

    def _commit(
        self,
        coordinator,
        snapshot,
        state,
        key,
        signature,
        operation,
        *,
        request_id,
        identity,
        agent_id=None,
    ):
        revision = snapshot["revision"]
        state["tick"] = revision + 1
        now = time.time()
        goal = state["goal"]
        goal["updated_at"] = now
        summary = (
            goal["assignments"][agent_id]["summary"]
            if agent_id
            else (
                "Simulation runtime stopped; see the task reason."
                if operation == "cancelled" and identity == "simulation-supervisor"
                else "Task cancelled by an operator; see the task cancellation reason."
                if operation == "cancelled"
                else "Task state updated."
            )
        )
        event = {
            "revision": revision + 1,
            "type": operation,
            "at": now,
            "agent_id": agent_id,
            "summary": summary,
            "request_id": request_id,
        }
        goal["events"] = [*goal["events"], event][-HISTORY_LIMIT:]
        state["receipts"][key] = {
            "signature": signature,
            "revision": revision + 1,
            "request_id": request_id,
            "identity": identity,
            "operation": operation,
        }
        message = self._message(
            goal["id"],
            revision,
            "blocked" if goal["status"] == "blocked" else "outcome",
            {
                "protocol": "studio-objective-v1",
                "task_id": goal["id"],
                "status": goal["status"],
                "event": event,
                # The full instructions, evidence and exact release specs already
                # live in the same committed state. Do not multiply their size in
                # every packet; peers read that state through their scoped inbox.
                "assignments": {
                    slot: {"status": item["status"], "progress": item["progress"]}
                    for slot, item in goal["assignments"].items()
                },
            },
        )
        result = {"state": state, "outcomes": [], "messages": [message]}
        intent = self._message(
            goal["id"],
            revision,
            "intent",
            {
                "protocol": "studio-objective-transition-v1",
                "key": key,
                "signature": signature,
                "result": result,
            },
            intent=True,
        )
        replay = False
        if snapshot["phase"] == "executing":
            batch = coordinator._batch(revision)
            try:
                pending = json.loads(batch["prepare_json"])["messages"][0]["payload"]
                if pending["protocol"] != "studio-objective-transition-v1":
                    raise ValueError
                if pending["key"] == key and pending["signature"] == signature:
                    result = pending["result"]
                    replay = True
                elif pending["key"] == key:
                    raise TaskConflict("This request_id was reused for a different pending update")
                elif operation == "cancelled":
                    # No external action is dispatched until commit + exact readback.
                    # A cancellation may supersede an uncommitted assignment/report.
                    state["receipts"][pending["key"]] = {
                        **pending["result"]["state"]["receipts"][pending["key"]],
                        "revision": revision + 1,
                        "cancelled_before_commit": True,
                    }
                else:
                    raise TaskConflict(
                        "An update is pending; retry its original request or cancel the task"
                    )
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                if isinstance(exc, TaskConflict):
                    raise
                raise TaskConflict(
                    "Pending task update cannot be verified; no runtime action authorized"
                ) from exc
        # Trim display history only, never request receipts or the latest report.
        # Every earlier transition also remains in its immutable factory batch.
        if not replay:
            limit = MAX_ENVELOPE_BYTES if operation == "cancelled" else ACTIVE_ENVELOPE_BYTES
            envelope = {"commands": [], "messages": [intent]}
            while _json_size(envelope) > limit and len(goal["events"]) > 1:
                goal["events"].pop(0)
            if _json_size(envelope) > limit:
                raise ValueError(
                    "Task update exceeds its payload budget; shorten report text/evidence or "
                    "cancel this task. No update was prepared."
                )
            canonical(envelope)
        canonical(result)
        if snapshot["phase"] != "executing":
            coordinator.prepare(revision, [], messages=[intent])
        coordinator.commit(revision, result["state"], [], messages=result["messages"])
        committed = coordinator.snapshot()
        batch = coordinator._batch(revision)
        if (
            committed["state"] != result["state"]
            or committed["revision"] != revision + 1
            or not batch
            or json.loads(batch["result_json"]) != result
        ):
            raise SpacetimeError("Task update exact readback failed; retry the same request_id")
        return self._public(committed)

    @_locked
    def claim(self, worker, task_id, value):
        coordinator, snapshot, key, signature, duplicate = self._begin(
            task_id, value, worker.get("id"), "claim"
        )
        assignment = self._worker(worker, snapshot["state"])
        if duplicate:
            return self._public(snapshot)
        if assignment["worker_id"] not in (None, worker["id"]):
            raise TaskConflict("Another runtime already owns this assignment")
        if assignment["status"] in TERMINAL:
            raise TaskConflict("This agent assignment is terminal")
        # A restarted runtime retains its existing claim without another dispatch.
        if assignment["worker_id"] == worker["id"] and snapshot["phase"] == "ready":
            return self._public(snapshot)
        state = copy.deepcopy(snapshot["state"])
        state["goal"]["assignments"][worker["agent_id"]].update(
            worker_id=worker["id"],
            execution_kind=worker.get("execution_kind", "external"),
            last_request_id=value["request_id"],
            status="running",
            updated_at=time.time(),
            summary="Authenticated runtime claimed the objective.",
        )
        self._status(state["goal"])
        return self._commit(
            coordinator,
            snapshot,
            state,
            key,
            signature,
            "claimed",
            request_id=value["request_id"],
            identity=worker["id"],
            agent_id=worker["agent_id"],
        )

    @staticmethod
    def _status(goal):
        statuses = [assignment["status"] for assignment in goal["assignments"].values()]
        goal["status"] = (
            "succeeded"
            if all(status == "succeeded" for status in statuses)
            else "failed"
            if "failed" in statuses
            else "blocked"
            if "blocked" in statuses
            else "running"
            if any(status != "queued" for status in statuses)
            else "queued"
        )

    @_locked
    def report(self, worker, task_id, value):
        coordinator, snapshot, key, signature, duplicate = self._begin(
            task_id, value, worker.get("id"), "report"
        )
        assignment = self._worker(worker, snapshot["state"])
        if duplicate:
            return self._public(snapshot)
        if assignment["worker_id"] != worker["id"]:
            raise PermissionError("Claim this assignment before reporting outcomes")
        if assignment["status"] in TERMINAL:
            raise TaskConflict("This agent assignment is terminal")
        status = value.get("status")
        if status not in ("running", "blocked", "succeeded", "failed"):
            raise ValueError("Report status must be running, blocked, succeeded, or failed")
        progress = value.get("progress")
        if (
            type(progress) not in (int, float)
            or not math.isfinite(progress)
            or not 0 <= progress <= 100
        ):
            raise ValueError("Progress must be a finite percentage from 0 to 100")
        if status == "succeeded" and progress != 100:
            raise ValueError("A successful assignment must report 100 percent progress")
        summary = _text(value.get("summary"), "Outcome summary", 1000)
        evidence = _strings(value.get("evidence", []), "Evidence", length=500)
        state = copy.deepcopy(snapshot["state"])
        state["goal"]["assignments"][worker["agent_id"]].update(
            status=status,
            progress=progress,
            summary=summary,
            evidence=evidence,
            last_request_id=value["request_id"],
            updated_at=time.time(),
        )
        self._status(state["goal"])
        return self._commit(
            coordinator,
            snapshot,
            state,
            key,
            signature,
            "reported",
            request_id=value["request_id"],
            identity=worker["id"],
            agent_id=worker["agent_id"],
        )

    @_locked
    def cancel(self, actor, task_id, value):
        _actor(actor)
        coordinator, snapshot, key, signature, duplicate = self._begin(
            task_id, value, actor["id"], "cancel"
        )
        if actor["role"] != "manager" and snapshot["state"]["goal"]["created_by"] != actor["id"]:
            raise PermissionError("Only this task's creator or a manager can cancel it")
        if duplicate:
            return self._public(snapshot)
        reason = _text(
            value.get("reason", "Cancelled by an operator."), "Cancellation reason", 1000
        )
        state = copy.deepcopy(snapshot["state"])
        state["goal"].update(status="cancelled", cancel_reason=reason)
        for assignment in state["goal"]["assignments"].values():
            if assignment["status"] not in TERMINAL:
                assignment.update(
                    status="cancelled",
                    summary="Cancelled by an operator; see the task reason.",
                    updated_at=time.time(),
                )
        return self._commit(
            coordinator,
            snapshot,
            state,
            key,
            signature,
            "cancelled",
            request_id=value["request_id"],
            identity=actor["id"],
        )

    @_locked
    def cancel_managed(self, task_id, worker_ids, reason):
        """Stop one supervisor-owned pair without impersonating a former manager.

        Internal lifecycle cleanup only: there is deliberately no HTTP route.
        The caller supplies IDs from its private launch manifest, including an
        empty list if startup failed before credential creation. Already-claimed
        assignments must belong to those exact simulation credentials. Approval
        retirement cannot prevent the owner from recording a stop.
        """
        if (
            not isinstance(worker_ids, (list, tuple, set))
            or len(worker_ids) > 2
            or any(
                not isinstance(worker, str) or re.fullmatch(r"worker-[a-f0-9]{16}", worker) is None
                for worker in worker_ids
            )
        ):
            raise ValueError("Managed cleanup requires the run's scoped worker identifiers")
        reason = _text(reason, "Cancellation reason", 1000)
        _, current = self._load(task_id)
        goal = current["state"]["goal"]
        if set(goal["assignments"]) != {"left", "right"} or any(
            assignment["execution_kind"] not in (None, "simulation")
            or assignment["worker_id"] is not None
            and assignment["worker_id"] not in worker_ids
            for assignment in goal["assignments"].values()
        ):
            raise PermissionError("This task is not owned by the managed simulation runtimes")
        if goal["status"] in TERMINAL:
            return self._public(current)
        identity = "simulation-supervisor"
        value = {
            "request_id": f"managed-stop-{task_id}-{current['revision']}",
            "expected_revision": current["revision"],
            "reason": reason,
        }
        coordinator, snapshot, key, signature, duplicate = self._begin(
            task_id, value, identity, "cancel"
        )
        if duplicate:
            return self._public(snapshot)
        state = copy.deepcopy(snapshot["state"])
        state["goal"].update(status="cancelled", cancel_reason=reason)
        for assignment in state["goal"]["assignments"].values():
            if assignment["status"] not in TERMINAL:
                assignment.update(
                    status="cancelled",
                    summary="Simulation runtime stopped; see the task reason.",
                    updated_at=time.time(),
                )
        return self._commit(
            coordinator,
            snapshot,
            state,
            key,
            signature,
            "cancelled",
            request_id=value["request_id"],
            identity=identity,
        )
