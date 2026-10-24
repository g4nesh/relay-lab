"""Separate task reasoning from low-level actions, without a trained-model claim.

RuleOrchestrator and ScriptedPolicy are deterministic debugging baselines. The
HTTP policy is a custom adapter boundary: arbitrary VLA checkpoints cannot be
plugged in without adapting their observations, action space, and control rate.
"""

import base64
import io
import json
import math
from collections import deque
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from PIL import Image

from .types import (
    AGENTS,
    COLORS,
    MAX_DELTA,
    MIN_HEIGHT,
    SAFE_HEIGHT,
    Action,
    AgentId,
    Decision,
    Instruction,
    Message,
    Observation,
    Vec3,
)


class ActionPolicy(Protocol):
    """One policy instance per robot; inputs contain only that robot's view."""

    needs_rgb: bool

    def reset(self) -> None: ...

    def act(self, observation: Observation, instruction: Instruction) -> Action: ...


class RuleOrchestrator:
    """A tiny role-based planner using local observations and explicit messages.

    Only the packer sees the private color order. A wrong cube blocks the single
    relay slot deliberately: this makes missing coordination easy to diagnose.
    The baseline does not solve recovery or infer the order from a global state.
    """

    def __init__(
        self, agent_id: AgentId, request_interval: int = 30, *, require_transfer_ready: bool = False
    ) -> None:
        if agent_id not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_id!r}")
        if request_interval < 1:
            raise ValueError("request_interval must be positive")
        self.agent_id = agent_id
        self.request_interval = request_interval
        self.require_transfer_ready = require_transfer_ready
        self.history: deque[tuple[int, str]] = deque(maxlen=64)
        self.reset()

    def reset(self) -> None:
        self.history.clear()
        self._requested: str | None = None
        self._request_sent_tick = -self.request_interval
        self._last_received_tick = -1
        self._ready_objects: set[str] = set()

    def decide(
        self,
        observation: Observation,
        inbox: list[Message],
        execution_events: list[dict] | None = None,
    ) -> Decision:
        if observation.agent_id != self.agent_id:
            raise ValueError("Orchestrator received another agent's observation")
        for event in execution_events or []:
            # The transport verifies sender and task identity before this boundary.
            if (
                self.agent_id == "packer"
                and event.get("agent_id") == "picker"
                and event.get("kind") == "holding"
                and event.get("object_id") in COLORS
            ):
                # Re-grasping ends the previous verified handoff. A later release
                # must settle and clear the picker before collection is allowed.
                self._ready_objects.discard(event["object_id"])
            if (
                self.agent_id == "packer"
                and event.get("agent_id") == "picker"
                and event.get("kind") == "transfer_ready"
                and event.get("target_id") == "relay"
                and event.get("object_id") in COLORS
                and event.get("evidence", {}).get("picker_clear") is True
            ):
                self._ready_objects.add(event["object_id"])
        for message in sorted(inbox, key=lambda value: (value.sent_tick, value.deliver_tick)):
            if message.recipient != self.agent_id or message.deliver_tick > observation.tick:
                continue
            self.history.append((observation.tick, f"received {message.text[:280]}"))
            if (
                self.agent_id == "picker"
                and message.sender == "packer"
                and message.text in {f"request:{color}" for color in COLORS}
                and message.sent_tick >= self._last_received_tick
            ):
                self._requested = message.text.split(":", 1)[1]
                self._last_received_tick = message.sent_tick
        decision = (
            self._picker(observation) if self.agent_id == "picker" else self._packer(observation)
        )
        self.history.append((observation.tick, decision.instruction.text))
        return decision

    def _picker(self, observation: Observation) -> Decision:
        if observation.holding is not None:
            return Decision(
                Instruction(
                    "place",
                    f"Place {observation.holding} on the shared relay pad.",
                    object_id=observation.holding,
                    target_id="relay",
                )
            )
        if any(abs(position[0]) <= 0.08 for position in observation.objects.values()):
            return Decision(Instruction("wait", "Keep the relay clear while the packer collects."))
        source = {color for color, position in observation.objects.items() if position[0] < -0.12}
        wanted = self._requested or next((color for color in COLORS if color in source), None)
        if wanted is not None and wanted in source:
            return Decision(Instruction("pick", f"Pick the {wanted} cube from the source.", wanted))
        return Decision(Instruction("wait", "Wait for a new color request from the packer."))

    def _packer(self, observation: Observation) -> Decision:
        if observation.order is None:
            return Decision(Instruction("wait", "Wait for a private packing order."))
        missing = next(
            (
                index
                for index in range(len(observation.order))
                if index >= len(observation.packed) or observation.packed[index] is None
            ),
            None,
        )
        if missing is None:
            return Decision(Instruction("wait", "Packing complete; return home."))
        wanted = observation.order[missing]
        message = None
        if (
            wanted != self._requested
            or observation.tick - self._request_sent_tick >= self.request_interval
        ):
            message = f"request:{wanted}"
            self._requested = wanted
            self._request_sent_tick = observation.tick
        if observation.holding is not None:
            self._ready_objects.discard(observation.holding)
            if observation.holding != wanted:
                return Decision(
                    Instruction("wait", "Holding the wrong cube; recovery is needed."), message
                )
            return Decision(
                Instruction(
                    "place",
                    f"Place {wanted} in packing slot {missing + 1}.",
                    object_id=wanted,
                    target_id=f"slot_{missing}",
                ),
                message,
            )
        position = observation.objects.get(wanted)
        if position is not None and abs(position[0]) <= 0.08:
            if self.require_transfer_ready and wanted not in self._ready_objects:
                return Decision(
                    Instruction(
                        "wait", f"Wait for the picker to verify {wanted} is ready and clear."
                    ),
                    message,
                )
            return Decision(
                Instruction("pick", f"Collect {wanted} from the relay pad.", wanted), message
            )
        return Decision(Instruction("wait", f"Wait for {wanted} at the relay pad."), message)


class ScriptedPolicy:
    """A Cartesian waypoint controller for testing orchestration, not a VLA.

    It lifts before translating, descends to the grasp/place point, then changes
    the gripper. Each returned translation has norm at most MAX_DELTA metres.
    """

    needs_rgb = False

    def __init__(self, agent_id: AgentId | None = None) -> None:
        if agent_id is not None and agent_id not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_id!r}")
        self.agent_id = agent_id
        self.reset()

    def reset(self) -> None:
        self._key: tuple[str, str | None, str | None] | None = None
        self._phase = 0

    def _move(
        self, current: np.ndarray, destination: Vec3 | np.ndarray, gripper: int = 0
    ) -> Action | None:
        delta = np.asarray(destination, dtype=float) - current
        distance = float(np.linalg.norm(delta))
        if distance <= 0.003:
            return None
        delta *= min(1.0, MAX_DELTA / distance)
        return Action(tuple(float(value) for value in delta), gripper)

    def act(self, observation: Observation, instruction: Instruction) -> Action:
        if self.agent_id is not None and observation.agent_id != self.agent_id:
            raise ValueError("Policy received another agent's observation")
        if instruction.kind not in ("pick", "place", "wait"):
            raise ValueError(f"Unsupported instruction: {instruction.kind!r}")
        key = (instruction.kind, instruction.object_id, instruction.target_id)
        if key != self._key:
            self._key = key
            self._phase = 0
        current = np.asarray(observation.ee_position, dtype=float)
        if instruction.kind == "pick":
            # A stale pick instruction must not open a hand that already holds a cube.
            if observation.holding is not None:
                return Action()
            target = observation.objects.get(instruction.object_id)
            # Dynamic cubes settle slightly below the end-effector workspace.
            # Approach the reachable grasp height instead of waiting forever for
            # an unreachable z coordinate at the end of the descent phase.
            if target is not None:
                target = (target[0], target[1], max(MIN_HEIGHT, target[2]))
        elif instruction.kind == "place":
            if observation.holding is None:
                return Action()
            target = observation.targets.get(instruction.target_id)
        else:
            default_home = (-0.54 if observation.agent_id == "picker" else 0.48, 0.0, SAFE_HEIGHT)
            target = observation.targets.get("home", default_home)
        if target is None:
            return Action()
        travel_height = max(SAFE_HEIGHT, target[2])
        transit_gripper = 1 if instruction.kind == "pick" else 0
        # Advancing already-reached waypoints in one call avoids artificial pauses.
        while self._phase < 3:
            if self._phase == 0:
                waypoint = (current[0], current[1], travel_height)
            elif self._phase == 1:
                waypoint = (target[0], target[1], travel_height)
            else:
                waypoint = target
            action = self._move(current, waypoint, transit_gripper)
            if action is not None:
                return action
            self._phase += 1
        return Action(gripper={"pick": -1, "place": 1, "wait": 0}[instruction.kind])


class HttpVLAPolicy:
    """Call a user-supplied VLA adapter using Relay Lab's explicit JSON contract.

    POST contains agent_id, session_id, instruction text/metadata, rgb_png_base64,
    and the robot's local observation. The adapter must isolate state by session_id.
    Return {"action": {"delta": [dx,dy,dz],
    "gripper": -1|0|1}}. Deltas are world-frame metres, bounded by MAX_DELTA.
    A real model needs an adapter; this class does not load or train a checkpoint.
    """

    needs_rgb = True

    def __init__(self, endpoint: str, agent_id: AgentId, timeout: float = 10.0) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("VLA endpoint must be an http:// or https:// URL")
        if agent_id not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_id!r}")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("VLA timeout must be a finite positive number")
        self.endpoint = endpoint
        self.agent_id = agent_id
        self.timeout = timeout
        self.reset()

    def reset(self) -> None:
        """Start a fresh remote context without requiring a second HTTP endpoint."""
        self.session_id = str(uuid4())

    def act(self, observation: Observation, instruction: Instruction) -> Action:
        if observation.agent_id != self.agent_id:
            raise ValueError("HTTP policy received another agent's observation")
        if observation.rgb is None:
            raise ValueError("HTTP VLA policy requires this agent's rendered RGB observation")
        rgb = observation.rgb
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB observation must be an H x W x 3 uint8 array")
        png = io.BytesIO()
        Image.fromarray(rgb).save(png, format="PNG")
        payload = {
            "schema": "relay-lab.action.v1",
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "control_dt": 0.1,
            "action_bounds": {"max_delta_norm": MAX_DELTA, "gripper_values": [-1, 0, 1]},
            "instruction": instruction.model_prompt,
            "execution_feedback": instruction.feedback,
            "replan": instruction.replan,
            "instruction_metadata": {
                "kind": instruction.kind,
                "object_id": instruction.object_id,
                "target_id": instruction.target_id,
            },
            "rgb_png_base64": base64.b64encode(png.getvalue()).decode("ascii"),
            "observation": {
                "tick": observation.tick,
                "ee_position": observation.ee_position,
                "holding": observation.holding,
                "objects": observation.objects,
                "targets": observation.targets,
                "packed": observation.packed,
                "order": observation.order,
            },
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(1_048_577)
            if len(body) > 1_048_576:
                raise ValueError("Response exceeds the 1 MiB limit")
            data = json.loads(body)
            action = data["action"]
            delta = action["delta"]
            gripper = action["gripper"]
            if (
                not isinstance(delta, list)
                or len(delta) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    for value in delta
                )
            ):
                raise ValueError("action.delta must contain three finite numbers")
            if math.sqrt(sum(value * value for value in delta)) > MAX_DELTA + 1e-9:
                raise ValueError(f"action.delta norm must be at most {MAX_DELTA} metres")
            if (
                isinstance(gripper, bool)
                or not isinstance(gripper, int)
                or gripper not in (-1, 0, 1)
            ):
                raise ValueError("action.gripper must be an integer: -1, 0, or 1")
            return Action(tuple(float(value) for value in delta), gripper)
        except HTTPError as error:
            raise RuntimeError(f"{self.agent_id} VLA adapter returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(
                f"{self.agent_id} VLA adapter request failed (timeout {self.timeout}s): {error}"
            ) from error
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            raise RuntimeError(
                f"{self.agent_id} VLA adapter returned an invalid action: {error}"
            ) from error
