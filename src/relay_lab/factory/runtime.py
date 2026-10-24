"""One trusted simulation worker; Spacetime authorizes every physics tick."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path

from .contracts import CAPABILITIES
from .planner import FactoryPlanner
from .scene import ROBOT_SPECS, STATIONS
from .world import FactoryWorld


class FactoryRun:
    def __init__(
        self,
        *,
        seed=7,
        orders=3,
        planner="llm",
        coordination="spacetime",
        max_ticks=6000,
        output=None,
        world=None,
        coordinator=None,
        model=None,
    ):
        if type(orders) is not int or not 1 <= orders <= 6:
            raise ValueError("Orders must be an integer from 1 to 6")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be a nonnegative 32-bit integer")
        if type(max_ticks) is not int or not 1 <= max_ticks <= 100_000:
            raise ValueError("Invalid tick budget")
        if coordination not in ("spacetime", "memory"):
            raise ValueError("Unknown coordination backend")
        self.world = world or FactoryWorld(seed=seed, batch_size=orders)
        self.planner = model or FactoryPlanner(planner)
        if coordinator is None:
            if coordination == "spacetime":
                from .spacetime import SpacetimeCoordinator

                coordinator = SpacetimeCoordinator()
            else:
                from .coordination import MemoryCoordinator

                coordinator = MemoryCoordinator()
        self.coordinator = coordinator
        self.run_id = "factory-" + uuid.uuid4().hex[:16]
        self.worker_id = "worker-" + uuid.uuid4().hex[:12]
        self.seed, self.orders, self.max_ticks = seed, orders, max_ticks
        self.output = Path(output or Path("runs/factory") / self.run_id).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = (self.output / "events.jsonl").open("w")
        self.state = self.world.snapshot()
        self.scene = self.world.scene()
        self.authority = None
        self.active = {}
        self.sequences = defaultdict(int)
        self.messages, self.events = deque(maxlen=240), deque(maxlen=240)
        self.feedback = deque(maxlen=16)
        self.command_count = self.completed_commands = self.failed_commands = 0
        self.status, self.error, self.reason = "ready", None, "Awaiting production order"
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self._poisoned = False
        self._closed = False
        self.no_progress_ticks = 0
        self._view = {}
        self._refresh()

    def _write(self, kind, value):
        self.log.write(
            json.dumps({"type": kind, "run_id": self.run_id, **value}, separators=(",", ":")) + "\n"
        )
        self.log.flush()

    def initialize(self):
        resources = sorted(
            set(STATIONS)
            | set(ROBOT_SPECS)
            | {"aisle-main"}
            | {c["id"] for c in self.state["cargo"]}
        )
        self.authority = self.coordinator.create(
            self.run_id,
            self.worker_id,
            [{"robot_id": key, "capabilities": list(value)} for key, value in CAPABILITIES.items()],
            [
                {"job_id": key, "stage": value["stage"], "custodian": value.get("custodian")}
                for key, value in self.state["jobs"].items()
            ],
            resources,
            self.max_ticks,
            self.state,
        )
        self.status = "running"
        self._write(
            "created",
            {"scene": self.scene, "state": self.state, "backend": self.coordinator.backend_info},
        )
        self._refresh()

    def _message(self, sender, recipient, kind, payload, *, correlation=None, priority=5):
        sequence = self.sequences[sender]
        self.sequences[sender] += 1
        return {
            "message_id": f"{sender}-{sequence}",
            "sender": sender,
            "recipient": recipient,
            "sequence": sequence,
            "kind": kind,
            "correlation_id": correlation,
            "expires_tick": self.state["tick"] + 300,
            "priority": priority,
            "confidence": 1.0,
            "payload": payload,
        }

    def _new_command(self, candidate, reason):
        self.command_count += 1
        command = {
            "command_id": f"cmd-{self.command_count:04d}",
            "robot_id": candidate["robot_id"],
            "capability": candidate["capability"],
            "job_id": candidate.get("job_id"),
            "resources": candidate["resources"],
            "lease_ticks": 8,
            "payload": {
                key: candidate.get(key)
                for key in ("kind", "source", "target", "cargo_id", "expected_stage", "next_stage")
            },
        }
        robot, job = command["robot_id"], candidate.get("job_id")
        messages = []
        # Readiness comes from world.candidates(): station proximity, idle receiver,
        # actual custody, and free destination. ACKs are worker sensor reports.
        if candidate["kind"] == "transfer":
            peers = [
                p
                for p in (candidate.get("source"), candidate.get("target"))
                if p in CAPABILITIES and p != robot
            ]
            for peer in dict.fromkeys(peers):
                request = self._message(
                    robot,
                    peer,
                    "request",
                    {
                        "job_id": job,
                        "text": f"Request custody transfer of {candidate['cargo_id']} from {candidate['source']} to {candidate['target']}.",
                        "command_id": command["command_id"],
                        "subgoal": candidate.get("next_stage"),
                    },
                    priority=8,
                )
                ack = self._message(
                    peer,
                    robot,
                    "ack",
                    {
                        "job_id": job,
                        "text": "Docking, stationary state and transfer readiness verified by simulator.",
                        "evidence": "world affordance checks",
                    },
                    correlation=request["message_id"],
                    priority=8,
                )
                messages.extend([request, ack])
                if peer == candidate.get("source"):
                    command["payload"]["handoff_message_id"] = request["message_id"]
        messages.append(
            self._message(
                robot,
                "broadcast",
                "intent",
                {
                    "job_id": job,
                    "text": candidate["description"],
                    "subgoal": candidate.get("next_stage") or candidate.get("target"),
                    "reason": reason,
                    "command_id": command["command_id"],
                    "resources": candidate["resources"],
                },
            )
        )
        return command, messages

    def _remember(self, messages, tick):
        for message in messages:
            entry = {
                **message,
                "id": message["message_id"],
                "tick": tick,
                "text": message["payload"].get("text", ""),
                "task_id": message["payload"].get("job_id"),
            }
            self.messages.append(entry)

    def tick(self):
        if self._poisoned:
            raise RuntimeError(
                "Worker is stopped after an unconfirmed or failed execution; create a new run"
            )
        if self.status in ("succeeded", "exhausted", "failed", "cancelled"):
            return
        try:
            if self.authority is None:
                self.initialize()
            used = {
                resource for command in self.active.values() for resource in command["resources"]
            }
            candidates = [
                c for c in self.world.candidates() if not used.intersection(c["resources"])
            ]
            outgoing = []
            if candidates:
                self.status = "planning"
                self._refresh()
                # Consume authoritative peer messages, not only locally generated text.
                pending = self.coordinator.pending("mediator")
                feedback = list(self.feedback) + [
                    {
                        "kind": m["kind"],
                        "from": m["sender"],
                        "text": m["payload"].get("text", ""),
                        "subgoal": m["payload"].get("subgoal"),
                    }
                    for m in pending[:6]
                ]
                decision = self.planner.choose(self.state, candidates, feedback)
                self.reason = decision.reason
                self._write(
                    "decision",
                    {"tick": self.state["tick"], "reason": decision.reason, **decision.trace},
                )
                for candidate in decision.candidates:
                    command, messages = self._new_command(candidate, decision.reason)
                    self.active[command["command_id"]] = command
                    outgoing.extend(messages)
            self.status = "running"
            revision = self.authority["revision"]
            commands = self.coordinator.prepare(revision, list(self.active.values()), outgoing)
            result = self.world.step(commands)  # Exactly once; no retry of physical work.
            state, outcomes = result["snapshot"], result["outcomes"]
            self._write(
                "execution_observed",
                {
                    "revision": revision,
                    "tick": state["tick"],
                    "committed": False,
                    "state": state,
                    "outcomes": outcomes,
                    "events": result["events"],
                },
            )
            messages = []
            terminal = [o for o in outcomes if o["status"] != "running"]
            for outcome in terminal:
                command = self.active[outcome["command_id"]]
                fact = {
                    "command_id": outcome["command_id"],
                    "job_id": command.get("job_id"),
                    "text": f"{outcome['robot_id']} {outcome['status']}: {command['capability']}; observed stage {outcome['observed'].get('stage') or 'carrier positioned'}. {outcome['detail']}",
                    "observed": outcome["observed"],
                }
                self.feedback.append(fact)
                messages.append(
                    self._message(
                        outcome["robot_id"],
                        "mediator",
                        "outcome" if outcome["status"] == "completed" else "blocked",
                        fact,
                        priority=7,
                    )
                )
            authority = self.coordinator.commit(revision, state, outcomes, messages)
            self.state, self.authority = state, authority
            self._remember(outgoing, state["tick"] - 1)
            self._remember(messages, state["tick"])
            self.events.extend(result["events"])
            for outcome in terminal:
                del self.active[outcome["command_id"]]
                self.completed_commands += outcome["status"] == "completed"
                self.failed_commands += outcome["status"] == "failed"
            self.no_progress_ticks = (
                0 if self.active or terminal or candidates else self.no_progress_ticks + 1
            )
            self.status = authority["status"]
            self._write(
                "tick",
                {
                    "tick": state["tick"],
                    "state": state,
                    "outcomes": outcomes,
                    "messages": outgoing + messages,
                    "events": result["events"],
                },
            )
            if self.failed_commands:
                raise RuntimeError(
                    "A motion skill failed; stopped for inspection of observed evidence"
                )
            if self.no_progress_ticks >= 20:
                raise RuntimeError(
                    "No feasible progress for 20 ticks; stopped instead of repeatedly replanning"
                )
            self._refresh()
        except Exception as error:
            self.fail(error)
            raise

    def fail(self, error):
        self._poisoned, self.status, self.error = True, "failed", str(error)
        self._write("failure", {"tick": self.state["tick"], "error": self.error})
        abort = getattr(self.coordinator, "abort", None)
        if abort and self.authority:
            try:
                abort(self.error)
            except Exception as abort_error:  # noqa: BLE001 - best-effort failure reporting
                self._write("abort_unconfirmed", {"error": str(abort_error)})
        self._refresh()

    def _refresh(self):
        resources = self.authority["resources"] if self.authority else {}
        view = {
            **deepcopy(self.state),
            "run_id": self.run_id,
            "status": self.status,
            "sim_time": self.state.get("time", 0),
            "scene": self.scene,
            "stations": self.scene["stations"],
            "orders": list(self.state["jobs"].values()),
            "mission": {"completed": self.state["completed"], "total": self.orders},
            "resources": [
                {"id": key, "label": key, "owner": v["holder"] or None, **v}
                for key, v in resources.items()
            ],
            "messages": list(self.messages),
            "events": list(self.events),
            "error": self.error,
            "planner": {**self.planner.info, "reason": self.reason},
            "coordination": {
                **self.coordinator.backend_info,
                "connected": self.authority is not None and not self._poisoned,
            },
            "metrics": {
                "commands": self.command_count,
                "completed_commands": self.completed_commands,
                "failed_commands": self.failed_commands,
                "messages": sum(self.sequences.values()),
                "wall_seconds": round(time.monotonic() - self.started, 2),
            },
            "output": str(self.output),
        }
        with self.lock:
            self._view = view

    def view(self):
        with self.lock:
            return deepcopy(self._view)

    def close(self, cancelled=False):
        if self._closed:
            return
        self._closed = True
        if cancelled and self.status not in ("succeeded", "exhausted", "failed"):
            self.status = "cancelled"
            if self.authority and hasattr(self.coordinator, "abort"):
                try:
                    self.coordinator.abort("Cancelled by user")
                except Exception as abort_error:  # noqa: BLE001 - cancellation cannot restart physics
                    self._write("abort_unconfirmed", {"error": str(abort_error)})
        self._refresh()
        (self.output / "summary.json").write_text(json.dumps(self.view(), indent=2))
        self.log.close()
        self.planner.close()
        self.world.close()


def run_factory(**kwargs):
    run = FactoryRun(**kwargs)
    try:
        while run.status not in ("succeeded", "exhausted", "failed"):
            run.tick()
        return run.view()
    finally:
        run.close()
