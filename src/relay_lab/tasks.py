"""Bounded task intentions shared by chat, runners, and policy feedback.

This is a task adapter boundary, not a general natural-language planner. Another
task needs an adapter plus its validated parameters and final outcome checks;
the ACP session and grounded feedback machinery can be reused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .types import AGENTS, COLORS, AgentId

if TYPE_CHECKING:
    from .environment import RelayEnv
    from .policies import RuleOrchestrator

MAX_TASK_TEXT = 256
RELAY_COMMAND = re.compile(r"run relay(?: seed ([0-9]{1,10}))?")


@dataclass(frozen=True)
class TaskSpec:
    kind: str = "relay"
    order: tuple[str, ...] | None = None
    goal: str = field(init=False)

    def __post_init__(self):
        if self.kind != "relay":
            raise ValueError("Only the relay task adapter is available")
        if self.order is not None and (
            not isinstance(self.order, tuple)
            or len(self.order) != len(COLORS)
            or any(type(color) is not str for color in self.order)
            or set(self.order) != set(COLORS)
        ):
            raise ValueError("Order must contain coral, mint, and gold exactly once")
        goal = "Pack all three parcels into their assigned tray slots using the shared relay."
        if self.order is not None:
            goal = (
                f"Pack {', then '.join(self.order)} into tray slots 1, 2, and 3 "
                "using the shared relay."
            )
        object.__setattr__(self, "goal", goal)


class TaskAdapter(Protocol):
    """The task-specific environment, planner, and local goal boundary."""

    kind: str
    description: str
    spec: TaskSpec

    def __init__(self, spec: TaskSpec | None = None): ...

    def parse(self, text: str) -> TaskSpec | None: ...

    def create_environment(self, seed: int) -> RelayEnv: ...

    def create_planners(
        self, require_transfer_ready: bool = False
    ) -> dict[AgentId, RuleOrchestrator]: ...

    def goal_for(self, agent: AgentId) -> str: ...


class RelayTaskAdapter:
    kind = "relay"
    description = (
        "Two simulated robots relay coral, mint, and gold parcels into three tray slots. "
        "Choose all three colors exactly once with `pack coral, mint, gold` or "
        "`pack coral then mint then gold`; `run relay [seed N]` uses the seeded order."
    )

    def __init__(self, spec: TaskSpec | None = None):
        self.spec = spec or TaskSpec()
        if self.spec.kind != self.kind:
            raise ValueError("Relay adapter requires a relay task")

    def create_environment(self, seed: int) -> RelayEnv:
        from .environment import RelayEnv

        return RelayEnv(seed=seed, order=self.spec.order)

    def create_planners(
        self, require_transfer_ready: bool = False
    ) -> dict[AgentId, RuleOrchestrator]:
        from .policies import RuleOrchestrator

        return {
            agent: RuleOrchestrator(agent, require_transfer_ready=require_transfer_ready)
            for agent in AGENTS
        }

    def goal_for(self, agent: AgentId) -> str:
        if agent == "picker":
            return (
                "Move the parcel requested by the packer to the shared relay, then clear the relay."
            )
        if agent == "packer":
            return self.spec.goal
        raise ValueError(f"Unknown agent: {agent!r}")

    def parse(self, text: str) -> TaskSpec | None:
        legacy = RELAY_COMMAND.fullmatch(text)
        if legacy:
            if legacy[1] and int(legacy[1]) > 2**31 - 1:
                return None
            return TaskSpec()
        # Politeness and task labels are accepted only around the explicit pack
        # grammar. Nothing after the three colors is executed or interpreted.
        text = re.sub(r"^(?:please\s+|task:\s*)", "", text, count=1)
        colors = "|".join(COLORS)
        match = re.fullmatch(
            rf"pack\s+({colors})(?:\s*,\s*|\s+then\s+)"
            rf"({colors})(?:\s*,\s*|\s+then\s+)({colors})\.?",
            text,
        )
        if not match:
            return None
        order = tuple(match.groups())
        if len(set(order)) != len(COLORS):
            return None
        return TaskSpec(order=order)


TASK_ADAPTERS: dict[str, type[TaskAdapter]] = {"relay": RelayTaskAdapter}


def get_task_adapter(spec: TaskSpec) -> TaskAdapter:
    """Resolve one explicitly registered environment adapter."""
    try:
        adapter = TASK_ADAPTERS[spec.kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported task kind: {spec.kind!r}") from exc
    return adapter(spec)


def parse_task(text: str) -> TaskSpec | None:
    """Return a supported task, or None without guessing at unsupported intent."""
    if not isinstance(text, str) or len(text) > MAX_TASK_TEXT:
        return None
    text = text.strip().lower()
    for adapter_type in TASK_ADAPTERS.values():
        task = adapter_type().parse(text)
        if task is not None:
            return task
    return None


def task_capabilities() -> str:
    available = " ".join(
        f"{adapter.kind}: {adapter.description}" for adapter in TASK_ADAPTERS.values()
    )
    return (
        "Available task adapters: " + available + " "
        "A local LLM mediates robot subgoals using observed outcome text, while SpacetimeDB "
        "synchronizes instructions, action proposals and observed reports. "
        "Arbitrary tasks and physical hardware are unsupported."
    )
