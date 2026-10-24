"""The small boundary between the simulator, orchestrator, and action policy.

Positions are world-frame metres. No policy receives the environment itself.
"""

from dataclasses import dataclass, field
from math import pi
from typing import Literal

import numpy as np

AgentId = Literal["picker", "packer"]
AGENTS: tuple[AgentId, ...] = ("picker", "packer")
COLORS = ("coral", "mint", "gold")
Vec3 = tuple[float, float, float]
MAX_DELTA = 0.035
MIN_HEIGHT = 0.035
SAFE_HEIGHT = 0.30


@dataclass(frozen=True)
class Observation:
    tick: int
    agent_id: AgentId
    ee_position: Vec3
    holding: str | None
    objects: dict[str, Vec3]
    targets: dict[str, Vec3]
    packed: tuple[str | None, ...] = ()
    order: tuple[str, ...] | None = None
    rgb: np.ndarray | None = field(default=None, repr=False, compare=False)
    wrist_rgb: np.ndarray | None = field(default=None, repr=False, compare=False)
    # Fixed downward tool frame and measured inner jaw span, in metres.
    ee_orientation: Vec3 = (pi, 0.0, 0.0)
    gripper_opening: float = 0.08


@dataclass(frozen=True)
class Instruction:
    kind: Literal["pick", "place", "wait"]
    text: str
    object_id: str | None = None
    target_id: str | None = None
    # Bounded, observed execution context from the coordinator. This is separate
    # from the task text so routine updates need not discard an action chunk.
    feedback: str = ""
    replan: bool = False

    @property
    def model_prompt(self) -> str:
        if not self.feedback:
            return self.text
        return f"{self.text}\nExecution feedback: {self.feedback}"


@dataclass(frozen=True)
class Message:
    sender: AgentId
    recipient: AgentId
    text: str
    sent_tick: int
    deliver_tick: int


@dataclass(frozen=True)
class Decision:
    instruction: Instruction
    message: str | None = None


@dataclass(frozen=True)
class Action:
    delta: Vec3 = (0.0, 0.0, 0.0)
    # -1 closes, +1 opens, 0 keeps the current state.
    gripper: int = 0
