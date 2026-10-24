"""Factory command barriers and bounded, causally linked robot communication.

The memory implementation is a deterministic reference for tests and explicit
``--coordination memory`` runs. It is never an outage fallback. The trusted
simulator supplies observations; the coordinator verifies their correspondence
to authorized skills but does not claim hardware-grade attestation.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from relay_lab.spacetime import SpacetimeConflict

STAGES = (
    "queued",
    "loaded",
    "at_cell",
    "in_fixture",
    "processed",
    "outbound_loaded",
    "at_dispatch",
    "completed",
)
STAGE_CAPABILITIES = (
    "transfer",
    "navigate",
    "transfer",
    "process",
    "transfer",
    "navigate",
    "transfer",
)


class CoordinationError(SpacetimeConflict):
    """A request violates the factory protocol; physics must not advance."""


def canonical(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 65_536:
        raise CoordinationError("Factory payload exceeds 64 KiB")
    return encoded


def require(condition: Any, message: str) -> None:
    if not condition:
        raise CoordinationError(message)


def _identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and all(char.isascii() and (char.isalnum() or char in "_-") for char in value)
    )


class MessageQueue:
    """Delivery queue with idempotency, causal ACKs, expiration and priority aging."""

    def __init__(self, participants: set[str], limit: int = 64):
        self.participants = participants | {"mediator", "judge"}
        self.limit = limit
        self.events: dict[str, dict] = {}
        self.sequence: dict[str, int] = {}
        self.acknowledged: set[str] = set()
        self.created: dict[str, int] = {}

    def publish(self, message: dict, tick: int) -> None:
        canonical(message)
        ident, sender, recipient = (message.get(k) for k in ("message_id", "sender", "recipient"))
        require(_identifier(ident), "Invalid message ID")
        prior = self.events.get(ident)
        if prior is not None:
            require(prior == message, "Conflicting duplicate message")
            return
        require(sender in self.participants, "Unknown message sender")
        require(recipient in self.participants or recipient == "broadcast", "Unknown recipient")
        seq = message.get("sequence")
        require(
            type(seq) is int and seq == self.sequence.get(sender, -1) + 1,
            "Message sequence is stale or out of order",
        )
        expiry = message.get("expires_tick")
        require(
            type(expiry) is int and tick < expiry <= tick + 10_000, "Message expired or invalid TTL"
        )
        priority = message.get("priority", 0)
        require(type(priority) is int and 0 <= priority <= 10, "Invalid message priority")
        confidence = message.get("confidence", 1.0)
        require(
            isinstance(confidence, (int, float))
            and math.isfinite(confidence)
            and 0 <= confidence <= 1,
            "Invalid confidence",
        )
        kind = message.get("kind")
        require(
            kind in {"intent", "belief", "request", "ack", "outcome", "blocked"}, "Invalid kind"
        )
        require(isinstance(message.get("payload"), dict), "Message payload must be an object")
        correlation = message.get("correlation_id")
        require(correlation is None or correlation in self.events, "Unknown causal parent")
        if kind == "ack":
            require(correlation in self.events, "An ACK requires its causal parent")
            parent = self.events[correlation]
            require(
                parent["recipient"] == sender and recipient == parent["sender"], "Invalid ACK scope"
            )
            require(parent["expires_tick"] > tick, "Cannot ACK an expired message")
            self.acknowledged.add(correlation)
        else:
            targets = self.participants - {sender} if recipient == "broadcast" else {recipient}
            for target in targets:
                require(len(self.pending(target, tick)) < self.limit, "Recipient queue is full")
        self.events[ident] = copy.deepcopy(message)
        self.created[ident] = tick
        self.sequence[sender] = seq

    def pending(self, recipient: str, tick: int) -> list[dict]:
        live = [
            m
            for mid, m in self.events.items()
            if m["recipient"] in {recipient, "broadcast"}
            and m["sender"] != recipient
            and m["kind"] != "ack"
            and mid not in self.acknowledged
            and m["expires_tick"] > tick
        ]
        # Unbounded aging eventually outranks a newly arrived priority-10 item.
        live.sort(
            key=lambda m: (
                -(m.get("priority", 0) + (tick - self.created[m["message_id"]]) // 4),
                self.created[m["message_id"]],
                m["message_id"],
            )
        )
        return copy.deepcopy(live)


class MemoryCoordinator:
    """Serializable reference state machine; each rejected operation is atomic."""

    @property
    def backend_info(self):
        return {"backend": "memory", "durable": False, "trusted_worker": True}

    def __init__(self):
        self.run: dict | None = None
        self.batches: dict[int, dict] = {}
        self.commands: dict[str, dict] = {}
        self.resources: dict[str, dict] = {}
        self.queue: MessageQueue | None = None

    def create(
        self,
        run_id: str,
        worker_id: str,
        robots: list[dict],
        jobs: list[dict],
        resources: list[str],
        max_ticks: int = 1000,
        state: dict | None = None,
    ) -> dict:
        spec = {"robots": robots, "jobs": jobs, "resources": resources, "max_ticks": max_ticks}
        canonical(spec)
        require(_identifier(run_id) and _identifier(worker_id), "Invalid run or worker ID")
        if self.run:
            require(
                self.run["run_id"] == run_id
                and self.run["worker_id"] == worker_id
                and self.run["spec"] == spec,
                "Conflicting run creation",
            )
            initial = state or {
                "tick": 0,
                "jobs": {j["job_id"]: dict(j) for j in jobs},
                "success": False,
            }
            require(
                self.run["revision"] == 0
                and self.run["phase"] == "ready"
                and self.run["state"] == initial,
                "Existing execution cannot be resumed by creating a new world",
            )
            return self.snapshot()
        ids = [r.get("robot_id") for r in robots]
        require(
            1 <= len(ids) <= 32 and len(set(ids)) == len(ids) and all(map(_identifier, ids)),
            "Invalid robot roster",
        )
        require(
            all(isinstance(r.get("capabilities"), list) and r["capabilities"] for r in robots),
            "Missing robot capabilities",
        )
        require(
            len(set(resources)) == len(resources) and all(map(_identifier, resources)),
            "Invalid resource registry",
        )
        require(type(max_ticks) is int and 1 <= max_ticks <= 100_000, "Invalid tick budget")
        job_ids = [j.get("job_id") for j in jobs]
        require(
            1 <= len(jobs) <= 64
            and len(set(job_ids)) == len(job_ids)
            and all(map(_identifier, job_ids)),
            "Invalid jobs",
        )
        initial = state or {
            "tick": 0,
            "jobs": {j["job_id"]: dict(j) for j in jobs},
            "success": False,
        }
        self.run = {
            "run_id": run_id,
            "worker_id": worker_id,
            "spec": copy.deepcopy(spec),
            "revision": 0,
            "tick": 0,
            "phase": "ready",
            "status": "running",
            "state": copy.deepcopy(initial),
        }
        self.resources = {
            r: {"holder": "", "command_id": "", "expires_tick": 0}
            for r in dict.fromkeys([*resources, *ids])
        }
        self.queue = MessageQueue(set(ids))
        return self.snapshot()

    def snapshot(self) -> dict:
        require(self.run is not None, "Run not created")
        return copy.deepcopy({**self.run, "resources": self.resources})

    def _transaction_copy(self, revision: int):
        # Old immutable batch payloads are an audit journal, not mutable working
        # state. Copying every prior camera/pose snapshot on every tick would be
        # quadratic in both work and allocations for a factory-length rollout.
        clone = copy.copy(self)
        clone.run = copy.deepcopy(self.run)
        clone.batches = self.batches.copy()
        if revision in clone.batches:
            clone.batches[revision] = copy.deepcopy(clone.batches[revision])
        clone.commands = {key: value.copy() for key, value in self.commands.items()}
        clone.resources = copy.deepcopy(self.resources)
        if self.queue is not None:
            clone.queue = copy.copy(self.queue)
            clone.queue.events = self.queue.events.copy()
            clone.queue.sequence = self.queue.sequence.copy()
            clone.queue.acknowledged = self.queue.acknowledged.copy()
            clone.queue.created = self.queue.created.copy()
        return clone

    def prepare(
        self, revision: int, commands: list[dict], messages: list[dict] | None = None
    ) -> list[dict]:
        # Validate on a copy so a rejected batch cannot leave partial locks or ACKs.
        clone = self._transaction_copy(revision)
        result = clone._prepare(revision, commands, messages or [])
        self.__dict__.update(clone.__dict__)
        return result

    def _prepare(self, revision: int, commands: list[dict], messages: list[dict]) -> list[dict]:
        require(self.run is not None and self.queue is not None, "Run not created")
        body = {"commands": commands, "messages": messages}
        canonical(body)
        if revision in self.batches:
            require(self.batches[revision]["prepare"] == body, "Conflicting duplicate batch")
            require(
                self.run["status"] == "running"
                and self.run["phase"] == "executing"
                and self.batches[revision]["result"] is None,
                "Committed batch cannot authorize physics twice",
            )
            return copy.deepcopy(commands)
        require(self.run["phase"] == "ready" and self.run["status"] == "running", "Run not ready")
        require(
            type(revision) is int and revision == self.run["revision"], "Stale or future revision"
        )
        require(len(commands) <= len(self.run["spec"]["robots"]), "Too many commands")
        robots = {r["robot_id"]: r for r in self.run["spec"]["robots"]}
        for message in messages:
            self.queue.publish(message, self.run["tick"])
        seen_robots, seen_jobs = set(), set()
        active = {cid for cid, c in self.commands.items() if c["status"] == "running"}
        require(
            active <= {c.get("command_id") for c in commands}, "Running commands cannot be omitted"
        )
        for command in commands:
            cid, robot = command.get("command_id"), command.get("robot_id")
            require(_identifier(cid), "Invalid command ID")
            prior = self.commands.get(cid)
            if prior is not None:
                require(
                    prior["status"] == "running"
                    and {k: v for k, v in prior.items() if k != "status"} == command,
                    "Terminal or changed command cannot be replayed",
                )
            require(robot in robots and robot not in seen_robots, "Unknown or duplicate robot")
            require(
                command.get("capability") in robots[robot]["capabilities"], "Unsupported capability"
            )
            require(isinstance(command.get("payload", {}), dict), "Invalid command payload")
            seen_robots.add(robot)
            job = command.get("job_id") or ""
            jobs = self.run["state"]["jobs"]
            if job:
                require(job in jobs and job not in seen_jobs, "Unknown job or conflicting custody")
                holder = jobs[job].get("custodian")
                if prior is None and holder in robots and holder != robot:
                    handoff = command.get("payload", {}).get("handoff_message_id")
                    parent = self.queue.events.get(handoff, {})
                    require(
                        command["capability"] == "transfer"
                        and handoff in self.queue.acknowledged
                        and parent.get("kind") == "request"
                        and parent.get("sender") == robot
                        and parent.get("recipient") == holder
                        and parent.get("payload", {}).get("job_id") == job
                        and parent.get("payload", {}).get("command_id") == cid
                        and (prior is not None or parent.get("expires_tick", 0) > self.run["tick"]),
                        "Robot custody transfer requires an acknowledged handoff",
                    )
                require(
                    command.get(
                        "expected_stage",
                        command.get("payload", {}).get("expected_stage", jobs[job]["stage"]),
                    )
                    == jobs[job]["stage"],
                    "Stale job stage",
                )
                seen_jobs.add(job)
            lease = command.get("lease_ticks", 1)
            require(type(lease) is int and 1 <= lease <= 1000, "Invalid lease duration")
            resource_ids = command.get("resources", [])
            require(
                isinstance(resource_ids, list) and len(set(resource_ids)) == len(resource_ids),
                "Invalid resources",
            )
            payload = command.get("payload", {})
            required = set()
            if command["capability"] in {"transfer", "process"}:
                required.update(
                    value
                    for value in (payload.get("source"), payload.get("target"))
                    if value in self.resources
                )
            elif command["capability"] == "navigate":
                required.update(
                    value
                    for value in (payload.get("target"), "aisle-main")
                    if value in self.resources
                )
            cargo = jobs[job].get("cargo_id") if job else None
            if cargo in self.resources:
                required.add(cargo)
            require(
                required <= set(resource_ids),
                "Command omitted required station, aisle, or cargo resources",
            )
            # Every actor is reserved even if an ill-formed client omits its own
            # ID from resources. A peer cannot move while a transfer holds it.
            resource_ids = list(dict.fromkeys([robot, *resource_ids]))
            for resource in resource_ids:
                require(resource in self.resources, "Unknown resource")
                lock = self.resources[resource]
                require(
                    not lock["holder"]
                    or lock["expires_tick"] <= self.run["tick"]
                    or lock["holder"] == robot,
                    "Resource leased to another robot",
                )
                self.resources[resource] = {
                    "holder": robot,
                    "command_id": cid,
                    "expires_tick": self.run["tick"] + lease,
                }
            handoff = command.get("payload", {}).get("handoff_message_id")
            if handoff:
                require(handoff in self.queue.acknowledged, "Handoff has not been acknowledged")
            self.commands[cid] = {**copy.deepcopy(command), "status": "authorized"}
        self.batches[revision] = {"prepare": copy.deepcopy(body), "result": None}
        self.run["phase"] = "executing"
        return copy.deepcopy(commands)

    def commit(
        self, revision: int, state: dict, outcomes: list[dict], messages: list[dict] | None = None
    ) -> dict:
        clone = self._transaction_copy(revision)
        clone._commit(revision, state, outcomes, messages or [])
        self.__dict__.update(clone.__dict__)
        return self.snapshot()

    def _commit(
        self, revision: int, state: dict, outcomes: list[dict], messages: list[dict]
    ) -> None:
        require(self.run is not None and self.queue is not None, "Run not created")
        body = {"state": state, "outcomes": outcomes, "messages": messages}
        canonical(body)
        batch = self.batches.get(revision)
        require(batch is not None, "No authorized batch")
        if batch["result"] is not None:
            require(batch["result"] == body, "Conflicting duplicate observed result")
            return
        require(
            self.run["phase"] == "executing" and revision == self.run["revision"],
            "Commit revision mismatch",
        )
        require(state.get("tick") == self.run["tick"] + 1, "Observed tick mismatch")
        commands = {c["command_id"]: c for c in batch["prepare"]["commands"]}
        require(len(outcomes) == len(commands), "Missing command results")
        seen = set()
        for result in outcomes:
            cid = result.get("command_id")
            require(cid in commands and cid not in seen, "Unknown or duplicate command result")
            seen.add(cid)
            command = commands[cid]
            require(result.get("robot_id") == command["robot_id"], "Outcome robot mismatch")
            require(
                result.get("status") in {"running", "completed", "failed"}, "Invalid outcome status"
            )
            require(isinstance(result.get("observed"), dict), "Missing observed evidence")
            self.commands[cid]["status"] = result["status"]
            if result["status"] != "running":
                for lock in self.resources.values():
                    if lock["command_id"] == cid:
                        lock.update(holder="", command_id="", expires_tick=0)
        expected_jobs = {j["job_id"] for j in self.run["spec"]["jobs"]}
        require(set(state.get("jobs", {})) == expected_jobs, "Observed job inventory mismatch")
        for jid, job in state["jobs"].items():
            prior = self.run["state"]["jobs"][jid]
            associated = [r for r in outcomes if commands[r["command_id"]].get("job_id") == jid]
            if job["stage"] != prior["stage"]:
                require(
                    len(associated) == 1 and associated[0]["status"] == "completed",
                    "Unobserved job stage change",
                )
                result = associated[0]
                command = commands[result["command_id"]]
                require(
                    prior["stage"] in STAGES[:-1]
                    and job["stage"] == STAGES[STAGES.index(prior["stage"]) + 1]
                    and command["capability"] == STAGE_CAPABILITIES[STAGES.index(prior["stage"])],
                    "Invalid factory process transition",
                )
                require(
                    command.get("next_stage", command.get("payload", {}).get("next_stage"))
                    == job["stage"],
                    "Unauthorized job stage transition",
                )
                require(
                    result["observed"].get("stage") == job["stage"],
                    "Job stage disagrees with evidence",
                )
            require(
                job.get("custodian")
                in (
                    None,
                    "",
                    *(r["robot_id"] for r in self.run["spec"]["robots"]),
                    *self.resources,
                ),
                "Unknown job custodian",
            )
            if job.get("custodian") != prior.get("custodian"):
                require(
                    len(associated) == 1
                    and associated[0]["observed"].get("custodian") == job.get("custodian"),
                    "Unobserved custody change",
                )
                command = commands[associated[0]["command_id"]]
                payload = command.get("payload", {})
                require(
                    command["capability"] == "transfer", "Only a transfer may change cargo custody"
                )
                require(
                    job.get("custodian")
                    in (
                        None,
                        "",
                        command["robot_id"],
                        payload.get("source"),
                        payload.get("target"),
                    ),
                    "Custody recipient is outside the authorized transfer",
                )
        success = all(j.get("stage") == "completed" for j in state["jobs"].values())
        require(
            state.get("success") is success, "Claimed success disagrees with observed job stages"
        )
        for message in messages:
            self.queue.publish(message, state["tick"])
        batch["result"] = copy.deepcopy(body)
        status = (
            "succeeded"
            if success
            else "exhausted"
            if state["tick"] >= self.run["spec"]["max_ticks"]
            else "running"
        )
        self.run.update(
            state=copy.deepcopy(state),
            tick=state["tick"],
            revision=revision + 1,
            status=status,
            phase="ready" if status == "running" else "terminal",
        )

    def abort(self, reason: str) -> dict:
        require(self.run is not None, "Run not created")
        require(isinstance(reason, str) and 0 < len(reason.strip()) <= 2048, "Invalid abort reason")
        if self.run["status"] == "failed":
            require(self.run.get("reason") == reason, "Conflicting duplicate abort")
            return self.snapshot()
        require(self.run["status"] == "running", "Cannot abort terminal run")
        self.run.update(status="failed", phase="terminal", reason=reason)
        self.run["state"]["abort_reason"] = reason
        for lock in self.resources.values():
            lock.update(holder="", command_id="", expires_tick=0)
        for command in self.commands.values():
            if command["status"] in {"authorized", "running"}:
                command["status"] = "aborted"
        return self.snapshot()

    def pending(self, recipient: str) -> list[dict]:
        require(self.run is not None and self.queue is not None, "Run not created")
        return self.queue.pending(recipient, self.run["tick"])
