"""Turn locally observed execution into small, transport-independent status events.

The tracker reports named *instruction* outcomes, never whole-episode success.
It has no access to the private packing order, full simulator state, or model
predictions. Relay readiness requires an observed release, stationary local
object observations, and a clear picker. A gripper command alone proves nothing.
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import dist, hypot
from typing import Literal

from .types import AGENTS, Action, AgentId, Instruction, Observation

EventKind = Literal["started", "holding", "transfer_ready", "completed", "blocked"]


@dataclass(frozen=True)
class ExecutionEvent:
    task_id: str
    agent_id: AgentId
    event_id: str
    instruction_id: str
    tick: int
    kind: EventKind
    object_id: str | None
    target_id: str | None
    reason: str
    evidence: dict
    schema: str = "relay-lab.execution.v1"
    scope: str = "instruction"

    def to_dict(self) -> dict:
        """Return JSON-safe fields suitable for a local log or uAgents message."""
        return asdict(self)


@dataclass
class _InstructionState:
    instruction_id: str
    instruction: Instruction
    completed: bool = False


@dataclass
class _Placement:
    state: _InstructionState
    release_tick: int
    stationary_ticks: int = 0


class ExecutionFeedback:
    """One tracker per robot and episode; call once after every simulator step.

    ``events`` may be the simulator event list: other robots' events and global
    success are ignored. Evidence comes from the supplied robot's local views.
    ``action`` is optional and only diagnoses requested versus executed motion.
    Relay stationarity means two consecutive local 100-ms observations moving
    no more than 1 mm each, not a claim about a real force-controlled grasp.
    """

    def __init__(
        self,
        agent_id: AgentId,
        task_id: str,
        *,
        stalled_ticks: int = 20,
        settle_ticks: int = 2,
    ) -> None:
        if agent_id not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_id!r}")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        for name, value in (("stalled_ticks", stalled_ticks), ("settle_ticks", settle_ticks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.agent_id = agent_id
        self.task_id = task_id
        self.stalled_ticks = stalled_ticks
        self.settle_ticks = settle_ticks
        self.reset()

    def reset(self) -> None:
        self._state: _InstructionState | None = None
        self._instruction_count = 0
        self._event_count = 0
        self._last_tick = -1
        self._stalled = 0
        self._pending: _Placement | None = None
        self._emitted: set[tuple] = set()

    def update(
        self,
        before: Observation,
        after: Observation,
        instruction: Instruction,
        events: Iterable[Mapping] = (),
        action: Action | None = None,
    ) -> list[ExecutionEvent]:
        if before.agent_id != self.agent_id or after.agent_id != self.agent_id:
            raise ValueError("Feedback received another agent's observation")
        if after.tick <= before.tick or after.tick < self._last_tick:
            raise ValueError("Feedback requires observations in increasing tick order")
        if after.tick == self._last_tick:
            return []  # Retry of an already processed transition.
        if before.tick < self._last_tick:
            raise ValueError("Feedback transitions must not overlap")
        self._last_tick = after.tick
        output: list[ExecutionEvent] = []
        key = (instruction.kind, instruction.object_id, instruction.target_id)
        previous = self._state.instruction if self._state else None
        if previous is None or key != (previous.kind, previous.object_id, previous.target_id):
            self._instruction_count += 1
            self._state = _InstructionState(
                f"{self.agent_id}:{self._instruction_count}", instruction
            )
            self._stalled = 0
            if instruction.kind != "wait":
                self._emit(
                    output,
                    self._state,
                    after.tick,
                    "started",
                    "local_instruction_active",
                    {"instruction_kind": instruction.kind, "observed_tick": after.tick},
                )
        state = self._state
        own = [event for event in events if event.get("agent") == self.agent_id]

        if after.holding is not None and before.holding != after.holding:
            self._emit(
                output,
                state,
                after.tick,
                "holding",
                "observed_attachment",
                {
                    "holding": after.holding,
                    "matches_instruction": after.holding == instruction.object_id,
                },
                object_id=after.holding,
            )
        if instruction.object_id and after.holding not in (None, instruction.object_id):
            self._emit(
                output,
                state,
                after.tick,
                "blocked",
                "wrong_object_held",
                {"expected_object": instruction.object_id, "holding": after.holding},
            )
        elif (
            instruction.kind == "pick"
            and instruction.object_id is not None
            and after.holding == instruction.object_id
            and any(
                event.get("type") == "grasp" and event.get("object") == instruction.object_id
                for event in own
            )
        ):
            state.completed = True
            self._emit(
                output,
                state,
                after.tick,
                "completed",
                "requested_object_grasped",
                {"holding": after.holding, "simulator_event": "grasp"},
            )

        if (
            instruction.kind == "place"
            and instruction.object_id is not None
            and before.holding == instruction.object_id
            and after.holding is None
            and any(
                event.get("type") == "release" and event.get("object") == instruction.object_id
                for event in own
            )
        ):
            self._pending = _Placement(state, after.tick)
        self._check_placement(before, after, output)

        delta = [float(end - start) for start, end in zip(before.ee_position, after.ee_position)]
        moved = dist(before.ee_position, after.ee_position)
        observed_change = moved > 0.001 or before.holding != after.holding
        if instruction.kind == "wait" or state.completed or observed_change:
            self._stalled = 0
        else:
            self._stalled += after.tick - before.tick
        if self._stalled >= self.stalled_ticks:
            limited = action is not None and any(
                abs(float(requested) - actual) > 1e-6
                for requested, actual in zip(action.delta, delta)
            )
            evidence = {"stationary_ticks": self._stalled, "executed_delta": delta}
            if action is not None:
                evidence["requested_delta"] = [float(value) for value in action.delta]
            self._emit(
                output,
                state,
                after.tick,
                "blocked",
                "motion_limited" if limited else "no_observed_progress",
                evidence,
            )
        if instruction.kind == "pick" and any(event.get("type") == "grasp_missed" for event in own):
            self._emit(
                output,
                state,
                after.tick,
                "blocked",
                "grasp_missed",
                {"holding": after.holding, "simulator_event": "grasp_missed"},
            )
        return output

    def _check_placement(
        self, before: Observation, after: Observation, output: list[ExecutionEvent]
    ) -> None:
        pending = self._pending
        if pending is None:
            return
        state = pending.state
        instruction = state.instruction
        color, target_id = instruction.object_id, instruction.target_id
        if after.holding == color:
            self._pending = None
            return
        if target_id and target_id.startswith("slot_") and self.agent_id == "packer":
            try:
                slot = int(target_id.removeprefix("slot_"))
            except ValueError:
                return
            if 0 <= slot < len(after.packed) and after.packed[slot] == color:
                state.completed = True
                self._emit(
                    output,
                    state,
                    after.tick,
                    "completed",
                    "requested_object_settled_in_slot",
                    {"slot": slot, "packed_object": color, "release_tick": pending.release_tick},
                )
                self._pending = None
            return
        if target_id != "relay" or self.agent_id != "picker":
            return
        position = after.objects.get(color)
        previous = before.objects.get(color)
        target = after.targets.get("relay")
        # Visibility excludes objects held by the peer. A disappearing object
        # cannot justify announcing a still-available handoff.
        if position is None or target is None:
            self._pending = None
            return
        at_relay = hypot(position[0] - target[0], position[1] - target[1]) < 0.055
        resting_height = 0.015 < position[2] < 0.065
        stationary = (
            previous is not None
            and after.tick == before.tick + 1
            and after.tick > pending.release_tick
            and dist(previous, position) <= 0.001
        )
        pending.stationary_ticks = (
            pending.stationary_ticks + 1 if at_relay and resting_height and stationary else 0
        )
        picker_clear = (
            hypot(after.ee_position[0] - target[0], after.ee_position[1] - target[1]) >= 0.10
            or after.ee_position[2] >= 0.18
        )
        if pending.stationary_ticks >= self.settle_ticks and picker_clear:
            evidence = {
                "release_tick": pending.release_tick,
                "object_position": [float(value) for value in position],
                "stationary_ticks": pending.stationary_ticks,
                "picker_position": [float(value) for value in after.ee_position],
                "picker_clear": True,
            }
            state.completed = True
            self._emit(
                output, state, after.tick, "transfer_ready", "released_settled_clear", evidence
            )
            self._emit(output, state, after.tick, "completed", "relay_placement_verified", evidence)
            self._pending = None

    def _emit(
        self,
        output: list[ExecutionEvent],
        state: _InstructionState,
        tick: int,
        kind: EventKind,
        reason: str,
        evidence: dict,
        *,
        object_id: str | None = None,
    ) -> None:
        object_id = object_id if object_id is not None else state.instruction.object_id
        key = (state.instruction_id, kind, reason, object_id)
        if key in self._emitted:
            return
        self._emitted.add(key)
        self._event_count += 1
        output.append(
            ExecutionEvent(
                task_id=self.task_id,
                agent_id=self.agent_id,
                event_id=f"{self.task_id}:{self.agent_id}:{self._event_count}",
                instruction_id=state.instruction_id,
                tick=tick,
                kind=kind,
                object_id=object_id,
                target_id=state.instruction.target_id,
                reason=reason,
                evidence=evidence,
            )
        )
