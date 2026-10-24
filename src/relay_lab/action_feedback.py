"""Ground model requests in local observations before the next decision.

The action is a proposal after the embodiment adapter, not proof of execution.
Only before/after robot observations establish motion, aperture and attachment.
This module never reads the environment, global task outcome or private order.
Peer events stay explicitly attributed reports, regardless of their transport.
"""

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import dist, isfinite

from .feedback import ExecutionEvent
from .types import AGENTS, Action, AgentId, Instruction, Observation, Vec3

_KINDS = {"started", "holding", "transfer_ready", "completed", "blocked"}
_REASONS = {
    "wrong_object_held": "a different object is held than the instruction requested",
    "motion_limited": "requested motion was not fully observed",
    "no_observed_progress": "no local instruction progress was observed",
    "grasp_missed": "the simulator reported a missed grasp",
    "requested_object_grasped": "the requested object attachment was verified",
    "requested_object_settled_in_slot": "placement in the requested slot was verified",
    "released_settled_clear": "release, settling and clearance were verified",
    "relay_placement_verified": "placement at the relay was verified",
    "observed_attachment": "an object attachment was observed",
}
_PEER_KINDS = {
    "started": "started an instruction",
    "holding": "an observed attachment",
    "transfer_ready": "a transfer ready after release and settling",
    "completed": "a completed local instruction",
    "blocked": "a blocked local instruction",
}


def _bounded(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _identifier(value: object) -> str:
    """Identifiers are data, never free-form instructions copied into prompts."""
    if not isinstance(value, str) or not value:
        return "none"
    cleaned = "".join(char for char in value if char.isascii() and (char.isalnum() or char in "_-"))
    return cleaned[:32] or "unnamed"


def _vector(value: Iterable[float], name: str) -> Vec3:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _format_vector(value: Vec3) -> str:
    return "(" + ",".join(f"{item:+.3f}" for item in value) + ")m"


def _mapping(event: Mapping | ExecutionEvent) -> Mapping:
    return event.to_dict() if isinstance(event, ExecutionEvent) else event


@dataclass(frozen=True)
class ActionReport:
    """A single local control transition, suitable for SDK messages and JSONL."""

    task_id: str
    agent_id: AgentId
    before_tick: int
    tick: int
    requested_delta: Vec3
    observed_delta: Vec3
    gripper_command: int
    gripper_opening_before: float
    gripper_opening_after: float
    holding_before: str | None
    holding_after: str | None
    motion_limited: bool
    local_events: tuple[str, ...]
    feedback: str
    critical: bool
    schema: str = "relay-lab.action-feedback.v1"

    def to_dict(self) -> dict:
        data = asdict(self)
        for name in ("requested_delta", "observed_delta", "local_events"):
            data[name] = list(data[name])
        return data


class ActionFeedback:
    """Small per-robot memory of measured action effects and delivered reports.

    ``update`` accepts only this robot's observations. ``receive_peer`` accepts
    task-matching events from the other robot and returns whether newly delivered
    evidence warrants discarding a queued action chunk. Call ``context`` before
    the next prediction: newest own facts come first, then peer reports, then old
    transitions if the character budget permits. No RGB or private order is kept.
    """

    def __init__(
        self,
        agent_id: AgentId,
        task_id: str,
        *,
        history_size: int = 3,
        max_chars: int = 850,
        stalled_ticks: int = 20,
        stall_cooldown_ticks: int = 20,
    ) -> None:
        if agent_id not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_id!r}")
        if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 128:
            raise ValueError("task_id must contain 1 to 128 characters")
        if (
            isinstance(history_size, bool)
            or not isinstance(history_size, int)
            or not 1 <= history_size <= 16
        ):
            raise ValueError("history_size must be between 1 and 16")
        if (
            isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
            or not 256 <= max_chars <= 4096
        ):
            raise ValueError("max_chars must be between 256 and 4096")
        for name, value in (
            ("stalled_ticks", stalled_ticks),
            ("stall_cooldown_ticks", stall_cooldown_ticks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.agent_id = agent_id
        self.task_id = task_id
        self.history_size = history_size
        self.max_chars = max_chars
        self.stalled_ticks = stalled_ticks
        self.stall_cooldown_ticks = stall_cooldown_ticks
        self.reset()

    def reset(self) -> None:
        self._history: deque[ActionReport] = deque(maxlen=self.history_size)
        self._peer_reports: deque[str] = deque(maxlen=self.history_size)
        self._seen_peer: deque[str] = deque(maxlen=64)
        self._last_tick = -1
        self._instruction_key: tuple | None = None
        self._stationary_ticks = 0
        self._stall_notified = False
        self._last_stall_notice = -self.stall_cooldown_ticks
        self._wrong_grasp: tuple | None = None
        self._blocked_conditions: set[str] = set()

    def receive_peer(self, events: Iterable[Mapping | ExecutionEvent], tick: int) -> bool:
        """Remember delivered peer evidence; ignore other tasks, future or old IDs.

        The transport is responsible for authenticating the actual sender. Event
        bodies still receive scope checks here and arbitrary evidence/reason text
        is never inserted into the model's instruction context.
        """
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        critical = False
        for raw in events:
            event = _mapping(raw)
            if not isinstance(event, Mapping):
                continue
            source, kind = event.get("agent_id"), event.get("kind")
            event_tick, event_id = event.get("tick"), event.get("event_id")
            if (
                event.get("task_id") != self.task_id
                or source not in AGENTS
                or source == self.agent_id
                or not isinstance(kind, str)
                or kind not in _KINDS
                or event.get("scope", "instruction") != "instruction"
                or isinstance(event_tick, bool)
                or not isinstance(event_tick, int)
                or not 0 <= event_tick <= tick
                or not isinstance(event_id, str)
                or not 1 <= len(event_id) <= 256
                or event_id in self._seen_peer
            ):
                continue
            self._seen_peer.append(event_id)
            obj, target = _identifier(event.get("object_id")), _identifier(event.get("target_id"))
            text = (
                f"Peer {source} reports {_PEER_KINDS[kind]} for {obj}"
                f" at {target} (observed tick {event_tick}, delivered {tick})."
            )
            self._peer_reports.append(text)
            critical |= kind in {"holding", "transfer_ready", "blocked"}
        return critical

    def update(
        self,
        before: Observation,
        after: Observation,
        instruction: Instruction,
        action: Action,
        *,
        execution_events: Iterable[Mapping | ExecutionEvent] = (),
        peer_events: Iterable[Mapping | ExecutionEvent] = (),
    ) -> ActionReport:
        if before.agent_id != self.agent_id or after.agent_id != self.agent_id:
            raise ValueError("Action feedback received another agent's observation")
        if any(
            isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
            for tick in (before.tick, after.tick)
        ):
            raise ValueError("Observation ticks must be nonnegative integers")
        if after.tick <= before.tick or after.tick < self._last_tick:
            raise ValueError("Action feedback requires increasing observation ticks")
        if after.tick == self._last_tick:
            return self._history[-1]
        if before.tick < self._last_tick:
            raise ValueError("Action feedback transitions must not overlap")
        start = _vector(before.ee_position, "before position")
        end = _vector(after.ee_position, "after position")
        requested = _vector(action.delta, "requested delta")
        observed = tuple(float(finish - begin) for begin, finish in zip(start, end))
        apertures = (float(before.gripper_opening), float(after.gripper_opening))
        if not all(isfinite(value) and value >= 0 for value in apertures):
            raise ValueError("Observed gripper openings must be finite and nonnegative")
        if action.gripper not in (-1, 0, 1):
            raise ValueError("gripper must be -1, 0, or 1")
        # This detects a discrepancy, not its cause. Norm limiting, a workspace
        # boundary or a different backend could all explain the same observation.
        limited = dist(requested, observed) > 0.001
        holding_changed = before.holding != after.holding
        wrong_grasp = instruction.object_id is not None and after.holding not in (
            None,
            instruction.object_id,
        )
        instruction_key = (instruction.kind, instruction.object_id, instruction.target_id)
        instruction_changed = instruction_key != self._instruction_key
        # A discrepancy is useful context, but norm/workspace limiting alone is
        # not a reason to discard every action chunk. Replan on an observed
        # change of state, a new failure, or one sustained-stall notice instead.
        wrong_grasp_key = (instruction_key, after.holding) if wrong_grasp else None
        critical = holding_changed or (wrong_grasp and wrong_grasp_key != self._wrong_grasp)
        self._wrong_grasp = wrong_grasp_key
        progress = (
            dist(start, end) > 0.001 or holding_changed or abs(apertures[1] - apertures[0]) > 0.001
        )
        active = instruction.kind != "wait" and not (
            instruction.kind == "pick"
            and instruction.object_id is not None
            and after.holding == instruction.object_id
        )
        if instruction_changed or progress or not active or after.tick != before.tick + 1:
            self._stationary_ticks = 0
            self._stall_notified = False
            self._blocked_conditions.clear()
        if active and not progress and after.tick == before.tick + 1:
            self._stationary_ticks += 1
        self._instruction_key = instruction_key
        stalled = self._stationary_ticks >= self.stalled_ticks
        reported_stall = False
        local_events = []
        for raw in execution_events:
            event = _mapping(raw)
            if not isinstance(event, Mapping) or (
                event.get("task_id") != self.task_id
                or event.get("agent_id") != self.agent_id
                or event.get("tick") != after.tick
                or not isinstance(event.get("kind"), str)
                or event.get("kind") not in _KINDS
                or event.get("scope", "instruction") != "instruction"
            ):
                continue
            kind, reason = event.get("kind"), event.get("reason")
            if kind == "blocked":
                if isinstance(reason, str) and reason in {
                    "motion_limited",
                    "no_observed_progress",
                }:
                    reported_stall = True
                else:
                    # Repeated reports of the same unresolved failure are
                    # evidence, not new failures. Progress or a new subtask
                    # rearms the condition; attachment changes always interrupt.
                    condition = (
                        reason if isinstance(reason, str) and reason in _REASONS else "unknown"
                    )
                    critical |= condition not in self._blocked_conditions
                    self._blocked_conditions.add(condition)
            description = _REASONS.get(reason) if isinstance(reason, str) else None
            if description is None and kind == "blocked":
                description = "the local instruction was reported blocked"
            if description and description not in local_events and len(local_events) < 4:
                local_events.append(description)
        if (
            active
            and (stalled or reported_stall)
            and not self._stall_notified
            and after.tick - self._last_stall_notice >= self.stall_cooldown_ticks
        ):
            critical = True
            self._stall_notified = True
            self._last_stall_notice = after.tick
            if stalled and len(local_events) < 4:
                local_events.append(
                    "no end-effector, jaw or attachment change was observed for "
                    f"{self._stationary_ticks} consecutive ticks"
                )
        command = {-1: "close", 0: "keep", 1: "open"}[action.gripper]
        parts = [
            (
                f"Tick {after.tick}, {self.agent_id}: observed move {_format_vector(observed)}; "
                f"holding {_identifier(before.holding)} -> {_identifier(after.holding)}; "
                f"jaw {apertures[0]:.3f} -> {apertures[1]:.3f}m."
            ),
            f"Requested move {_format_vector(requested)}, gripper {command}.",
        ]
        if limited:
            parts.append("Observed motion differs from the request; cause unconfirmed.")
        if wrong_grasp:
            parts.append(
                f"Held object differs from requested {_identifier(instruction.object_id)}."
            )
        parts.extend(f"Local evidence: {description}." for description in local_events)
        if action.gripper == -1 and after.holding is None:
            parts.append("No attachment observed after close request.")
        feedback = _bounded(" ".join(parts), self.max_chars)
        report = ActionReport(
            task_id=self.task_id,
            agent_id=self.agent_id,
            before_tick=before.tick,
            tick=after.tick,
            requested_delta=requested,
            observed_delta=observed,
            gripper_command=int(action.gripper),
            gripper_opening_before=apertures[0],
            gripper_opening_after=apertures[1],
            holding_before=before.holding,
            holding_after=after.holding,
            motion_limited=limited,
            local_events=tuple(local_events),
            feedback=feedback,
            critical=bool(critical),
        )
        self._history.append(report)
        self._last_tick = after.tick
        self.receive_peer(peer_events, after.tick)
        return report

    def context(self) -> str:
        """Bounded evidence for the next action, with newest own facts first."""
        newest = list(reversed(self._history))
        pieces = [newest[0].feedback] if newest else []
        pieces.extend(reversed(self._peer_reports))
        pieces.extend(report.feedback for report in newest[1:])
        return _bounded("\n".join(pieces), self.max_chars)
