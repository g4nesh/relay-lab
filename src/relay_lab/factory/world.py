"""Bounded factory skills over a shared, observed contact world.

The five embodiments support a task-trained RGB/language/action motor policy
and an explicitly selected model-based reference. Learned execution never
calls the reference trajectory generator or receives numeric simulator goals.
Payloads are MuJoCo rigid bodies while unowned; a grasp is a model-commanded,
proximity-checked idealized attachment. All ownership changes and task successes
are measured here, never inferred from planner text or a viewer animation.
"""

from __future__ import annotations

import io
import math
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import mujoco
import numpy as np

from .scene import (
    CARGO_SIZE,
    FLOOR_SIZE,
    NAVIGATION,
    OBSTACLES,
    ROBOT_SPECS,
    STATIONS,
    mujoco_xml,
    pallet_slot,
    scene_spec,
)


@dataclass
class Robot:
    id: str
    kind: str
    position: np.ndarray
    ee: np.ndarray
    yaw: float
    speed: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    holding: str | None = None
    gripper_opening: float = 1.0
    state: str = "idle"
    command: dict | None = None
    phase: str = "idle"
    waypoints: list[np.ndarray] = field(default_factory=list)
    elapsed: float = 0.0
    progress: float = 0.0
    blocked_elapsed: float = 0.0
    joints: dict = field(default_factory=dict)
    joint_points: list = field(default_factory=list)
    segment_origin: np.ndarray | None = None
    segment_target: np.ndarray | None = None
    segment_elapsed: float = 0.0


class FactoryWorld:
    """A reproducible 24×14 m pilot line with five distinct embodiments.

    ``step`` accepts only already-committed commands. Repeating an identical
    command ID is idempotent. A conflicting ID, overlapping command, unsafe
    transfer or missing physical precondition produces an observed rejection.
    An additional local lane interlock prevents collisions even if an external
    coordinator supplies conflicting commands. The external backend should
    reserve the corresponding resources *before* committing commands.
    """

    GRASP_TOLERANCE = 0.065
    DOCK_TOLERANCE = 0.10
    PROCESS_SECONDS = 3.0
    COMMAND_TIMEOUT = 100.0
    BLOCKED_TIMEOUT = 12.0

    def __init__(
        self,
        seed: int = 7,
        dt: float = 0.1,
        batch_size: int = 3,
        cargo_ids: list[str] | None = None,
        motor_policy: Any | None = None,
        pose_jitter: float = 0.0,
    ):
        if not math.isfinite(dt) or dt < 0.01 or dt > 0.2:
            raise ValueError("dt must be between 0.01 and 0.2 seconds")
        if not 1 <= batch_size <= 6:
            raise ValueError("batch_size must be between one and six")
        if not math.isfinite(pose_jitter) or not 0 <= pose_jitter <= 0.04:
            raise ValueError("pose_jitter must be between zero and 0.04 metres")
        self.seed, self.dt = int(seed), float(dt)
        self.motor_policy = motor_policy
        self._motor_actions: list[dict] = []
        self._motor_stats = {
            "inferences": 0,
            "safety_edits": 0,
            "grasp_attempts": 0,
            "release_attempts": 0,
        }
        self._peer_embeddings: dict[str, list[dict]] = {}
        self._sensor_lock = threading.Lock()
        self._sensor_pngs: dict[str, bytes] = {}
        self._camera_cache: dict[str, dict] = {}
        self._embedding_records: dict[str, dict] = {}
        self._last_commands: dict[str, dict] = {}
        self.cargo_ids = cargo_ids or [f"part-{index + 1:03}" for index in range(batch_size)]
        if len(self.cargo_ids) != batch_size or len(set(self.cargo_ids)) != batch_size:
            raise ValueError("cargo IDs must be unique and match batch_size")
        self.model = mujoco.MjModel.from_xml_string(mujoco_xml(self.cargo_ids))
        self.data = mujoco.MjData(self.model)
        self.tick = 0
        self._commands: dict[str, dict] = {}
        self._command_results: dict[str, dict] = {}
        self._events: list[dict] = []
        self._lane_owner: str | None = None
        self._motion_blocks: set[str] = set()
        self.robots = {
            key: Robot(
                key,
                value["kind"],
                np.array(value["position"], dtype=float),
                np.array(value["ee_position"], dtype=float),
                value["yaw"],
            )
            for key, value in ROBOT_SPECS.items()
        }
        self.cargo: dict[str, dict] = {}
        self._qpos: dict[str, int] = {}
        self._qvel: dict[str, int] = {}
        for index, cargo_id in enumerate(self.cargo_ids):
            self._qpos[cargo_id] = int(self.model.joint(f"{cargo_id}-joint").qposadr[0])
            self._qvel[cargo_id] = int(self.model.joint(f"{cargo_id}-joint").dofadr[0])
            self.cargo[cargo_id] = {
                "id": cargo_id,
                "job_id": f"job-{index + 1:03}",
                "index": index,
                "owner": None,
                "location": "receive",
                "processed": False,
                "delivered": False,
                "stage": "queued",
            }
        self._decks = {
            key: int(self.model.body(f"{key}-deck").mocapid[0])
            for key, value in ROBOT_SPECS.items()
            if value["kind"] == "amr"
        }
        if pose_jitter:
            rng = np.random.default_rng(self.seed)
            for cargo_id in self.cargo_ids:
                address = self._qpos[cargo_id]
                self.data.qpos[address : address + 2] += rng.uniform(
                    -pose_jitter, pose_jitter, size=2
                )
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)
        for robot in self.robots.values():
            self._update_joints(robot)

    @property
    def success(self) -> bool:
        return all(
            item["delivered"]
            and item["processed"]
            and item["owner"] is None
            and np.linalg.norm(
                self._cargo_position(item["id"]) - np.array(pallet_slot(item["index"]))
            )
            < 0.065
            and np.linalg.norm(self.data.qvel[self._qvel[item["id"]] : self._qvel[item["id"]] + 3])
            < 0.08
            for item in self.cargo.values()
        )

    def scene(self) -> dict:
        scene = scene_spec()
        if self.motor_policy is not None:
            scene["physics"]["robots"] = (
                "learned RGB/language motor actions through bounded kinematic actuators"
            )
            scene["physics"]["learned_control"] = True
        return scene

    def snapshot(self) -> dict:
        robot_states = []
        for robot in self.robots.values():
            robot_states.append(
                {
                    "id": robot.id,
                    "kind": robot.kind,
                    "type": robot.kind,
                    "label": ROBOT_SPECS[robot.id]["label"],
                    "position": self._vec(robot.position),
                    "yaw": round(robot.yaw, 6),
                    "ee_position": self._vec(robot.ee),
                    "velocity": self._vec(robot.velocity),
                    "speed": round(robot.speed, 5),
                    "holding": robot.holding,
                    "gripper_opening": robot.gripper_opening,
                    "state": robot.state,
                    "status": robot.state,
                    "phase": robot.phase,
                    "active_command": robot.command.get("command_id") if robot.command else None,
                    "command": deepcopy(robot.command),
                    "progress": round(robot.progress, 4),
                    "joints": deepcopy(robot.joints),
                    "joint_points": deepcopy(robot.joint_points),
                    "controller": (
                        "learned vision-language-action motor policy"
                        if self.motor_policy is not None
                        else ROBOT_SPECS[robot.id]["controller"]
                    ),
                }
            )
        cargo_states = []
        for cargo_id, item in self.cargo.items():
            pose = self._cargo_position(cargo_id)
            vstart = self._qvel[cargo_id]
            cargo_states.append(
                {
                    **deepcopy(item),
                    "position": self._vec(pose),
                    "size": list(CARGO_SIZE),
                    "speed": round(float(np.linalg.norm(self.data.qvel[vstart : vstart + 3])), 5),
                }
            )
        return {
            "tick": self.tick,
            "time": round(self.tick * self.dt, 3),
            "dt": self.dt,
            "robots": robot_states,
            "cargo": cargo_states,
            "success": self.success,
            "jobs": {
                item["job_id"]: {
                    **item,
                    "id": item["job_id"],
                    "cargo_id": item["id"],
                    "custodian": item["owner"],
                    "color": ["teal", "orange", "blue"][item["index"] % 3],
                }
                for item in cargo_states
            },
            "completed": sum(item["delivered"] for item in self.cargo.values()),
            "total": len(self.cargo),
            "interlocks": {"aisle-main": self._lane_owner},
            "motor_policy": {
                "mode": "vla" if self.motor_policy is not None else "reference",
                "metrics": deepcopy(self._motor_stats),
                "model": deepcopy(getattr(self.motor_policy, "model_info", None)),
            },
            "physics": {
                "engine": "MuJoCo",
                "controller": "learned vision-language-action"
                if self.motor_policy is not None
                else "model-based skills",
                "grasp": "proximity-checked idealized attachment",
                "learned": self.motor_policy is not None,
            },
        }

    state = snapshot

    def candidates(self) -> list[dict]:
        """Return physically feasible skills; a mediator chooses among these.

        This is an affordance boundary, not a generated execution trace. New
        candidates appear only after measured custody/stage/docking changes.
        Resource reservations and acknowledgements are the backend's job.
        """
        result = []

        def add(robot_id, kind, item=None, source=None, target=None, next_stage=None, resources=()):
            robot = self.robots[robot_id]
            if robot.command:
                return
            job_id = item["job_id"] if item else None
            cargo_id = item["id"] if item else None
            stage = item["stage"] if item else None
            candidate_id = ":".join(
                str(value or "empty") for value in (robot_id, kind, cargo_id, target, stage)
            )
            result.append(
                {
                    "candidate_id": candidate_id,
                    "robot_id": robot_id,
                    "kind": kind,
                    "capability": kind,
                    "job_id": job_id,
                    "cargo_id": cargo_id,
                    "source": source,
                    "target": target,
                    "source_id": source,
                    "target_id": target,
                    "expected_stage": stage,
                    "next_stage": next_stage,
                    "resources": list(dict.fromkeys([robot_id, *resources])),
                    "description": f"{robot_id} {kind} {cargo_id or 'empty carrier'} from {source or 'current pose'} to {target or 'current station'}",
                }
            )

        def docked(robot_id, station):
            robot = self.robots[robot_id]
            return (
                not robot.command
                and np.linalg.norm(robot.position[:2] - np.array(STATIONS[station]["position"][:2]))
                < self.DOCK_TOLERANCE
            )

        inbound, outbound = self.robots["amr-01"], self.robots["amr-02"]
        queued = [
            item
            for item in self.cargo.values()
            if item["stage"] == "queued" and item["owner"] is None
        ]
        unfinished = any(not item["delivered"] for item in self.cargo.values())
        if queued and not inbound.holding and not inbound.command and not docked("amr-01", "load"):
            add("amr-01", "navigate", target="load", resources=("aisle-main", "load"))
        if (
            unfinished
            and not outbound.holding
            and not outbound.command
            and not docked("amr-02", "process-out")
        ):
            add("amr-02", "navigate", target="process-out", resources=("aisle-main", "process-out"))
        fixture_free = not any(
            item["location"] == "fixture" and item["owner"] is None for item in self.cargo.values()
        )
        for item in self.cargo.values():
            stage = item["stage"]
            if (
                stage == "queued"
                and item["owner"] is None
                and docked("amr-01", "load")
                and not inbound.holding
            ):
                add(
                    "humanoid-01",
                    "transfer",
                    item,
                    "receive",
                    "amr-01",
                    "loaded",
                    ("receive", "load", "amr-01", item["id"]),
                )
            elif stage == "loaded" and item["owner"] == "amr-01":
                add(
                    "amr-01",
                    "navigate",
                    item,
                    "load",
                    "process-in",
                    "at_cell",
                    ("aisle-main", "process-in", item["id"]),
                )
            elif (
                stage == "at_cell"
                and item["owner"] == "amr-01"
                and fixture_free
                and docked("amr-01", "process-in")
            ):
                add(
                    "arm-01",
                    "transfer",
                    item,
                    "amr-01",
                    "fixture",
                    "in_fixture",
                    ("amr-01", "process-in", "fixture", item["id"]),
                )
            elif stage == "in_fixture" and item["owner"] is None and item["location"] == "fixture":
                add(
                    "arm-01",
                    "process",
                    item,
                    "fixture",
                    "fixture",
                    "processed",
                    ("fixture", item["id"]),
                )
            elif (
                stage == "processed"
                and item["owner"] is None
                and docked("amr-02", "process-out")
                and not outbound.holding
            ):
                add(
                    "arm-01",
                    "transfer",
                    item,
                    "fixture",
                    "amr-02",
                    "outbound_loaded",
                    ("fixture", "process-out", "amr-02", item["id"]),
                )
            elif stage == "outbound_loaded" and item["owner"] == "amr-02":
                add(
                    "amr-02",
                    "navigate",
                    item,
                    "process-out",
                    "dispatch",
                    "at_dispatch",
                    ("aisle-main", "dispatch", item["id"]),
                )
            elif (
                stage == "at_dispatch"
                and item["owner"] == "amr-02"
                and docked("amr-02", "dispatch")
            ):
                add(
                    "gantry-01",
                    "transfer",
                    item,
                    "amr-02",
                    "pallet",
                    "completed",
                    ("amr-02", "dispatch", "pallet", item["id"]),
                )
        return result

    def step(self, commands: list[dict] | dict | None = None) -> dict:
        self._events = []
        self._motor_actions = []
        submitted = []
        if commands:
            if isinstance(commands, dict):
                commands = [commands] if "robot_id" in commands else list(commands.values())
            for command in commands:
                submitted.append(asdict(command) if is_dataclass(command) else deepcopy(command))
                self._accept(submitted[-1])
        for robot in self.robots.values():
            self._advance(robot)
        self._physics()
        self.tick += 1
        for robot in self.robots.values():
            self._update_joints(robot)
        snapshot = self.snapshot()
        outcomes = []
        ids = {command.get("command_id", command.get("id")) for command in submitted}
        ids.update(event["command_id"] for event in self._events if event.get("command_id"))
        ids.update(robot.command["command_id"] for robot in self.robots.values() if robot.command)
        for command_id in sorted(ids - {None}):
            if command_id in self._command_results:
                outcomes.append(deepcopy(self._command_results[command_id]))
            elif command_id in self._commands:
                command = self._commands[command_id]
                item = self.cargo.get(command.get("cargo_id"), {})
                outcomes.append(
                    {
                        "command_id": command_id,
                        "robot_id": command["robot_id"],
                        "status": "running",
                        "detail": "skill executing",
                        "observed": {"stage": item.get("stage"), "custodian": item.get("owner")},
                    }
                )
        return {
            "snapshot": snapshot,
            "state": snapshot,
            "events": deepcopy(self._events),
            "outcomes": outcomes,
            "motor_actions": deepcopy(self._motor_actions),
        }

    def _event(self, kind: str, robot: Robot | None = None, **payload: Any) -> None:
        command = robot.command if robot else None
        self._events.append(
            {
                "type": kind,
                "kind": kind,
                "tick": self.tick,
                "robot_id": robot.id if robot else payload.pop("robot_id", None),
                "command_id": command.get("command_id")
                if command
                else payload.pop("command_id", None),
                **payload,
            }
        )

    def _accept(self, command: dict) -> None:
        command = deepcopy(command)
        payload = command.pop("payload", {})
        command.update(payload)
        if "capability" in command:
            command.setdefault("kind", command["capability"])
        if "source" in command:
            command.setdefault("source_id", command["source"])
        if "target" in command:
            command.setdefault("target_id", command["target"])
        robot_id = command.get("robot_id")
        command_id = command.get("command_id", command.get("id"))
        command["command_id"] = command_id
        if not isinstance(command_id, str) or not command_id or robot_id not in self.robots:
            self._event(
                "command_rejected",
                reason="invalid command or robot ID",
                command_id=command_id,
                robot_id=robot_id,
            )
            return
        if command_id in self._commands:
            if command != self._commands[command_id]:
                self._event(
                    "command_rejected",
                    reason="command ID reused with changed payload",
                    command_id=command_id,
                    robot_id=robot_id,
                )
            return
        robot = self.robots[robot_id]
        if robot.command:
            self._event(
                "command_rejected",
                reason="robot already executing another command",
                command_id=command_id,
                robot_id=robot_id,
            )
            return
        kind = command.get("kind", command.get("action"))
        command["kind"] = kind
        allowed = ROBOT_SPECS[robot_id]["capabilities"] + ["wait"]
        if kind not in allowed:
            self._event(
                "command_rejected",
                reason=f"{robot.kind} cannot execute {kind}",
                command_id=command_id,
                robot_id=robot_id,
            )
            return
        # Record every accepted command before starting. Retrying never executes
        # a physical skill twice, including after an observed failure.
        self._commands[command_id] = deepcopy(command)
        robot.command = command
        self._last_commands[robot.id] = deepcopy(command)
        robot.state, robot.phase = "working", "starting"
        robot.elapsed = robot.progress = robot.blocked_elapsed = 0.0
        robot.speed = 0.0
        try:
            self._start(robot)
        except (ValueError, KeyError) as exc:
            self._finish(robot, False, str(exc))
            return
        self._event(
            "command_started",
            robot,
            action=kind,
            cargo_id=command.get("cargo_id"),
            target_id=command.get("target_id"),
        )

    def _start(self, robot: Robot) -> None:
        command = robot.command
        kind = command["kind"]
        cargo_id = command.get("cargo_id")
        if kind in ("pick", "place", "transfer", "process") and cargo_id not in self.cargo:
            raise ValueError("unknown cargo")
        for other in self.robots.values():
            if other.id != robot.id and other.command:
                if cargo_id and other.command.get("cargo_id") == cargo_id:
                    raise ValueError("cargo is already reserved by an executing skill")
                if kind == "navigate" and other.command.get("target_id") == robot.id:
                    raise ValueError("carrier is reserved for an executing transfer")
        if cargo_id in self.cargo and command.get("expected_stage") not in (
            None,
            self.cargo[cargo_id]["stage"],
        ):
            raise ValueError("cargo stage changed before command execution")
        if kind == "navigate":
            target_id = command.get("target_id", command.get("target"))
            target = self._navigation_target(target_id, robot)
            robot.command["target_id"] = target_id
            if robot.holding is None:
                # Cargo on a mobile deck is owned by that carrier while driving.
                deck_items = [item for item in self.cargo.values() if item["owner"] == robot.id]
                if deck_items:
                    robot.holding = deck_items[0]["id"]
            if self._lane_owner not in (None, robot.id):
                robot.phase = "waiting_for_aisle"
            else:
                self._reserve_lane(robot)
            if self.motor_policy is None:
                y = NAVIGATION["spine_y"]
                points = [
                    np.array([robot.position[0], y, 0.0]),
                    np.array([target[0], y, 0.0]),
                    target,
                ]
                robot.waypoints = [
                    point for point in points if np.linalg.norm(point - robot.position) > 0.001
                ]
                if not robot.waypoints:
                    robot.waypoints = [target]
            else:
                robot.waypoints = []
        elif kind == "wait":
            robot.phase = "waiting"
        elif kind in ("pick", "transfer"):
            item = self.cargo[cargo_id]
            owner = item["owner"]
            if robot.holding:
                raise ValueError("gripper is already occupied")
            if owner and (owner not in self.robots or self.robots[owner].kind != "amr"):
                raise ValueError("cargo is already held by another manipulator")
            if owner and self.robots[owner].command:
                raise ValueError("carrier must finish navigation before transfer")
            if (
                command.get("source_id")
                and item["location"] != command["source_id"]
                and owner != command["source_id"]
            ):
                raise ValueError("cargo is not at the requested source")
            source = self._cargo_position(cargo_id)
            self._check_reach(robot, source)
            if kind == "transfer":
                target = self._placement_target(robot, command)
                self._check_reach(robot, target)
            robot.phase = "approach_pick"
            robot.waypoints = (
                self._lifted_path(robot.ee, source, robot.kind) if self.motor_policy is None else []
            )
        elif kind == "place":
            if robot.holding != cargo_id:
                raise ValueError("robot does not hold this cargo")
            target = self._placement_target(robot, command)
            self._check_reach(robot, target)
            robot.phase = "approach_place"
            robot.waypoints = (
                self._lifted_path(robot.ee, target, robot.kind) if self.motor_policy is None else []
            )
        elif kind == "process":
            item = self.cargo[cargo_id]
            if item["owner"] is not None or item["location"] != "fixture":
                raise ValueError("processing requires released cargo in the fixture")
            if (
                np.linalg.norm(
                    self._cargo_position(cargo_id) - np.array(STATIONS["fixture"]["position"])
                )
                > 0.07
            ):
                raise ValueError("fixture alignment failed")
            if item["processed"]:
                raise ValueError("cargo is already processed")
            target = self._cargo_position(cargo_id) + np.array([0.0, 0.0, 0.30])
            robot.phase = "approach_process"
            robot.waypoints = (
                self._lifted_path(robot.ee, target, robot.kind) if self.motor_policy is None else []
            )

    def _advance(self, robot: Robot) -> None:
        if self.motor_policy is not None:
            self._advance_learned(robot)
            return
        if not robot.command:
            robot.velocity[:] = 0
            return
        robot.elapsed += self.dt
        robot.progress = min(0.95, robot.elapsed / 30.0)
        if robot.elapsed > float(robot.command.get("timeout", self.COMMAND_TIMEOUT)):
            self._finish(robot, False, "skill execution timeout")
            return
        if robot.phase == "waiting_for_aisle":
            if self._lane_owner not in (None, robot.id):
                robot.state = "waiting"
                return
            self._reserve_lane(robot)
        if robot.phase == "waiting":
            if robot.elapsed >= float(robot.command.get("duration", self.dt)):
                self._finish(robot, True)
            return
        if robot.phase in ("settling", "processing"):
            robot.blocked_elapsed += self.dt
            if robot.phase == "settling":
                cargo_id = robot.command["cargo_id"]
                if robot.blocked_elapsed >= 0.35 and self._settled_at_target(robot):
                    self._finish(robot, True)
                elif robot.blocked_elapsed > 3.0:
                    self._finish(robot, False, "released cargo did not settle at target")
            elif robot.blocked_elapsed >= self.PROCESS_SECONDS:
                cargo_id = robot.command["cargo_id"]
                item = self.cargo[cargo_id]
                if (
                    item["owner"]
                    or item["location"] != "fixture"
                    or np.linalg.norm(
                        self._cargo_position(cargo_id) - np.array(STATIONS["fixture"]["position"])
                    )
                    > 0.07
                ):
                    self._finish(robot, False, "cargo moved during processing")
                else:
                    item["processed"] = True
                    item["stage"] = "processed"
                    self._event(
                        "process_verified",
                        robot,
                        cargo_id=cargo_id,
                        dwell_seconds=round(robot.blocked_elapsed, 2),
                    )
                    self._finish(robot, True)
            return
        if robot.waypoints:
            if not self._move(robot):
                return
            robot.waypoints.pop(0)
            if robot.waypoints:
                return
        if robot.phase == "navigating":
            target = robot.command["target_id"]
            if robot.holding:
                item = self.cargo[robot.holding]
                item["location"] = robot.id
            self._event("docked", robot, target_id=target, position=self._vec(robot.position))
            self._finish(robot, True)
        elif robot.phase == "approach_pick":
            self._grasp(robot)
        elif robot.phase == "approach_place":
            self._release(robot)
        elif robot.phase == "approach_process":
            robot.phase = "processing"
            robot.blocked_elapsed = 0.0
            self._event("process_started", robot, cargo_id=robot.command["cargo_id"])

    def set_peer_embeddings(self, records_by_robot: dict[str, list[dict]]) -> None:
        """Replace per-receiver peer context with already-committed DB records.

        Embeddings never confer control authority. Expired/future/self records
        cannot influence the next neural action, even if supplied by a caller.
        """
        checked = {}
        for receiver, records in records_by_robot.items():
            if receiver not in self.robots or not isinstance(records, list):
                raise ValueError("Peer embeddings require a known receiver and a list")
            eligible = []
            for record in records[:4]:
                vector = np.asarray(record.get("vector"), dtype=float)
                if vector.shape != (32,) or not np.isfinite(vector).all():
                    raise ValueError("Peer vector must contain 32 finite values")
                if not 0.998 <= float(np.dot(vector, vector)) <= 1.002:
                    raise ValueError("Peer vector must be unit normalized")
                if (
                    record.get("robot_id") == receiver
                    or record.get("robot_id") not in self.robots
                    or record.get("tick", self.tick + 1) > self.tick
                    or record.get("expires_tick", 0) <= self.tick
                ):
                    continue
                eligible.append(deepcopy(record))
            checked[receiver] = eligible
        self._peer_embeddings = checked

    def _motor_observation(self, robot: Robot) -> dict:
        from PIL import Image

        from .vision import make_observation

        records = [
            record
            for record in self._peer_embeddings.get(robot.id, [])
            if record["expires_tick"] > self.tick and record["tick"] <= self.tick
        ]
        command = robot.command or {}
        intended_peer = (
            command.get("target_id")
            if robot.phase == "approach_place"
            else command.get("source_id")
        )
        matching = [record for record in records if record["robot_id"] == intended_peer]
        peer = matching[0] if matching else records[0] if records else None
        cache = self._camera_cache.get(robot.id)
        camera_key = (command.get("command_id"), robot.phase)
        refresh = (
            cache is None
            or cache["key"] != camera_key
            or (robot.command is not None and self.tick - cache["tick"] >= 5)
        )
        observation = make_observation(
            self.snapshot(),
            robot.id,
            robot.command,
            robot.phase,
            rgb=None if refresh else cache["rgb"],
            peer_embedding=deepcopy(peer),
        )
        if refresh:
            self._camera_cache[robot.id] = {
                "key": camera_key,
                "tick": self.tick,
                "rgb": observation["rgb"].copy(),
            }
        # The camera builder owns scene geometry. The actor receives only RGB,
        # own proprioception, language and bounded committed peer latents.
        allowed = {
            "robot_id",
            "kind",
            "rgb",
            "instruction",
            "phase",
            "proprio",
            "tick",
            "dt",
            "peer_embedding",
            "command_id",
        }
        if set(observation) - allowed:
            raise ValueError("Visual policy observation leaked undeclared scene state")
        observation["command_id"] = command.get("command_id")
        observation["peer_record_ids"] = [peer["embedding_id"]] if peer else []
        observation["rgb_tick"] = self._camera_cache[robot.id]["tick"]
        observation["proprio"]["gripper_opening"] = robot.gripper_opening
        rgb = np.asarray(observation["rgb"])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Motor camera must provide uint8 RGB")
        if refresh:
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            with self._sensor_lock:
                self._sensor_pngs[robot.id] = buffer.getvalue()
        return observation

    def sensor_png(self, robot_id: str) -> bytes | None:
        """Read the producer's last RGB frame; HTTP threads never render GL."""
        if robot_id not in self.robots:
            raise ValueError("Unknown robot camera")
        with self._sensor_lock:
            return self._sensor_pngs.get(robot_id)

    def embedding_observations(self) -> list[dict]:
        """Encode current post-step observations for durable peer publication."""
        if self.motor_policy is None:
            return []
        records = []
        for robot in self.robots.values():
            observation = self._motor_observation(robot)
            encoded = self.motor_policy.embed(observation)
            vector = np.asarray(encoded["embedding"], dtype=float)
            if vector.shape != (32,) or not np.isfinite(vector).all():
                raise ValueError("Learned encoder returned invalid latent")
            if not 0.998 <= float(np.dot(vector, vector)) <= 1.002:
                raise ValueError("Learned encoder returned a non-unit latent")
            space_id = encoded.get("space_id", encoded.get("latent_space"))
            if not isinstance(space_id, str) or not space_id.strip():
                raise ValueError("Learned encoder must identify its latent space")
            if (
                "latent_space" in encoded
                and "space_id" in encoded
                and encoded["latent_space"] != encoded["space_id"]
            ):
                raise ValueError("Learned encoder returned conflicting latent spaces")
            command = robot.command
            decoder = encoded.get("decoded", {})
            if not isinstance(decoder, dict) or not {"text", "confidence", "version"} <= set(
                decoder
            ):
                raise ValueError("Learned decoder must report text, confidence and version")
            record = {
                "robot_id": robot.id,
                "tick": self.tick,
                "command_id": command.get("command_id") if command else None,
                "job_id": command.get("job_id") if command else None,
                "vector": vector.tolist(),
                "dimension": 32,
                "space_id": space_id,
                "checkpoint_sha256": encoded["checkpoint_sha256"],
                "decoder": {
                    key: deepcopy(decoder[key]) for key in ("text", "confidence", "version")
                },
            }
            records.append(record)
            self._embedding_records[robot.id] = deepcopy(record)
        return records

    def _advance_learned(self, robot: Robot) -> None:
        """Apply only actual network motor proposals; task logic merely judges."""
        if robot.command is None:
            robot.velocity[:] = 0
            robot.speed = 0.0
            return
        robot.elapsed += self.dt
        robot.progress = min(0.95, robot.elapsed / 30.0)
        if robot.elapsed > float(robot.command.get("timeout", self.COMMAND_TIMEOUT)):
            self._finish(robot, False, "learned motor skill execution timeout")
            return
        if robot.phase == "waiting_for_aisle":
            if self._lane_owner not in (None, robot.id):
                robot.state = "waiting"
                robot.velocity[:] = 0
                return
            self._reserve_lane(robot)
        observation = self._motor_observation(robot)
        prediction = self.motor_policy.act(observation)
        self._motor_stats["inferences"] += 1
        raw_delta = np.asarray(prediction.get("delta"), dtype=float)
        yaw_delta = prediction.get("yaw_delta", 0.0)
        gripper = prediction.get("gripper", 0.0)
        if (
            raw_delta.shape != (3,)
            or not np.isfinite(raw_delta).all()
            or not isinstance(yaw_delta, (int, float))
            or not math.isfinite(yaw_delta)
            or not isinstance(gripper, (int, float))
            or not math.isfinite(gripper)
        ):
            raise ValueError("VLA returned an invalid motor action")
        before = robot.position.copy() if robot.kind == "amr" else robot.ee.copy()
        before_yaw = robot.yaw
        command_id = robot.command["command_id"]
        phase = robot.phase
        edits = self._apply_learned_motion(robot, raw_delta, float(yaw_delta))
        robot.gripper_opening = (
            0.0 if gripper < -0.5 else 1.0 if gripper > 0.5 else robot.gripper_opening
        )
        cargo_id = robot.command.get("cargo_id") if robot.command else None
        if robot.command and phase == "approach_pick" and gripper < -0.5:
            self._motor_stats["grasp_attempts"] += 1
            if np.linalg.norm(self._cargo_position(cargo_id) - robot.ee) <= self.GRASP_TOLERANCE:
                self._grasp(robot)
            else:
                edits.append("grasp_outside_contact_tolerance")
        elif robot.command and phase == "approach_place" and gripper > 0.5:
            self._motor_stats["release_attempts"] += 1
            try:
                target = self._placement_target(robot, robot.command)
                if np.linalg.norm(robot.ee - target) <= self.GRASP_TOLERANCE:
                    self._release(robot)
                else:
                    edits.append("release_outside_contact_tolerance")
            except ValueError as error:
                self._finish(robot, False, str(error))
        after = robot.position.copy() if robot.kind == "amr" else robot.ee.copy()
        self._motor_stats["safety_edits"] += bool(edits)
        self._motor_actions.append(
            {
                "robot_id": robot.id,
                "command_id": command_id,
                "tick": self.tick,
                "phase": phase,
                "instruction": observation["instruction"],
                "peer_record_ids": observation["peer_record_ids"],
                "raw_action": {
                    "delta": raw_delta.tolist(),
                    "yaw_delta": float(yaw_delta),
                    "gripper": float(gripper),
                },
                "applied_action": {
                    "delta": self._vec(after - before),
                    "yaw_delta": round(robot.yaw - before_yaw, 7),
                    "gripper_opening": robot.gripper_opening,
                },
                "safety_edits": edits,
                "provenance": deepcopy(prediction.get("provenance", {})),
            }
        )
        if robot.command is None:
            return
        if robot.phase == "navigating":
            target = self._navigation_target(robot.command["target_id"], robot)
            if (
                np.linalg.norm(robot.position[:2] - target[:2]) <= self.DOCK_TOLERANCE
                and robot.speed < 0.06
            ):
                self._event(
                    "docked",
                    robot,
                    target_id=robot.command["target_id"],
                    position=self._vec(robot.position),
                )
                self._finish(robot, True)
        elif robot.phase == "approach_process":
            target = self._cargo_position(cargo_id) + np.array([0, 0, 0.30])
            if np.linalg.norm(robot.ee - target) <= 0.08 and robot.speed < 0.08:
                robot.phase = "processing"
                robot.blocked_elapsed = 0.0
                self._event("process_started", robot, cargo_id=cargo_id)
        elif robot.phase == "processing":
            item = self.cargo[cargo_id]
            target = self._cargo_position(cargo_id) + np.array([0, 0, 0.30])
            if item["owner"] or item["location"] != "fixture":
                self._finish(robot, False, "cargo moved during learned processing")
            elif np.linalg.norm(robot.ee - target) > 0.08 or robot.speed > 0.08:
                robot.blocked_elapsed = 0.0
            else:
                robot.blocked_elapsed += self.dt
                if robot.blocked_elapsed >= self.PROCESS_SECONDS:
                    item["processed"] = True
                    self._event(
                        "process_verified",
                        robot,
                        cargo_id=cargo_id,
                        dwell_seconds=robot.blocked_elapsed,
                    )
                    self._finish(robot, True)
        elif robot.phase == "settling":
            robot.blocked_elapsed += self.dt
            if robot.blocked_elapsed >= 0.35 and self._settled_at_target(robot):
                self._finish(robot, True)
            elif robot.blocked_elapsed > 3:
                self._finish(robot, False, "learned release did not settle at target")
        elif robot.phase == "waiting" and robot.elapsed >= float(
            robot.command.get("duration", self.dt)
        ):
            self._finish(robot, True)

    def _apply_learned_motion(
        self, robot: Robot, requested_delta: np.ndarray, yaw_delta: float
    ) -> list[str]:
        """Actuator/geometry limits only; no target tracking or waypoint actions."""
        edits = []
        spec = ROBOT_SPECS[robot.id]
        velocity = requested_delta / self.dt
        if robot.kind == "amr" and abs(velocity[2]) > 1e-9:
            velocity[2] = 0.0
            edits.append("planar_drive")
        speed = float(np.linalg.norm(velocity))
        if speed > spec["max_speed"]:
            velocity *= spec["max_speed"] / speed
            edits.append("speed_limit")
        change = velocity - robot.velocity
        change_norm = float(np.linalg.norm(change))
        max_change = spec["max_acceleration"] * self.dt
        if change_norm > max_change:
            velocity = robot.velocity + change * (max_change / change_norm)
            edits.append("acceleration_limit")
        current = robot.position if robot.kind == "amr" else robot.ee
        proposed = current + velocity * self.dt
        if robot.kind == "amr":
            collision = self._navigation_collision(robot, current, proposed)
        else:
            collision = None
            try:
                self._check_reach(robot, proposed)
            except ValueError:
                collision = "workspace_envelope"
            if not 0.25 <= proposed[2] <= 3.1:
                collision = "vertical_envelope"
            if not collision:
                for obstacle in OBSTACLES:
                    padding = (
                        np.array(CARGO_SIZE[:2]) / 2 if robot.holding else np.array([0.025, 0.025])
                    )
                    low = np.array(obstacle["center"]) - np.array(obstacle["half_size"]) - padding
                    high = np.array(obstacle["center"]) + np.array(obstacle["half_size"]) + padding
                    clearance = 0.12 if robot.holding else 0.025
                    if (
                        self._segment_aabb(current[:2], proposed[:2], low, high)
                        and min(current[2], proposed[2]) - clearance < obstacle["height"] - 0.008
                    ):
                        collision = obstacle["id"]
                        break
        if collision:
            velocity[:] = 0
            proposed = current.copy()
            edits.append(f"collision_stop:{collision}")
            robot.state = "blocked"
            robot.blocked_elapsed += self.dt
            if robot.blocked_elapsed >= self.BLOCKED_TIMEOUT:
                self._finish(robot, False, f"learned motion blocked: {collision}")
        else:
            robot.state = "working"
            # Processing and settling use this counter for measured dwell.
            if robot.phase not in ("processing", "settling"):
                robot.blocked_elapsed = 0.0
        if robot.kind == "amr":
            robot.position[:] = proposed
            robot.ee[:] = proposed + np.array([0, 0, 0.62])
            applied_yaw = float(np.clip(yaw_delta, -1.2 * self.dt, 1.2 * self.dt))
            if abs(applied_yaw - yaw_delta) > 1e-9:
                edits.append("turn_rate_limit")
            robot.yaw += applied_yaw
        else:
            robot.ee[:] = proposed
            if abs(yaw_delta) > 1e-9:
                edits.append("end_effector_orientation_fixed")
        robot.velocity = velocity
        robot.speed = float(np.linalg.norm(velocity))
        return edits

    def _move(self, robot: Robot) -> bool:
        target = robot.waypoints[0]
        current = robot.position if robot.kind == "amr" else robot.ee
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance < 1e-7:
            robot.speed = 0.0
            robot.velocity[:] = 0
            return True
        spec = ROBOT_SPECS[robot.id]
        acceleration = spec["max_acceleration"]
        if robot.kind == "amr":
            desired_yaw = math.atan2(delta[1], delta[0])
            yaw_error = (desired_yaw - robot.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_error) > 0.001:
                # Waypoints are full stops, so a differential drive can turn in
                # place instead of sliding sideways through an instantaneous corner.
                robot.yaw += float(np.clip(yaw_error, -1.2 * self.dt, 1.2 * self.dt))
                robot.speed = 0.0
                robot.velocity[:] = 0.0
                return False
        # Each straight segment uses an analytic trapezoidal velocity profile.
        # Sampling that profile keeps both speed and finite-difference
        # acceleration bounded, including the last fractional timestep.
        if robot.segment_target is None or not np.array_equal(target, robot.segment_target):
            robot.segment_origin = current.copy()
            robot.segment_target = target.copy()
            robot.segment_elapsed = 0.0
        segment = target - robot.segment_origin
        length = float(np.linalg.norm(segment))
        ramp_time = min(spec["max_speed"] / acceleration, math.sqrt(length / acceleration))
        peak_speed = acceleration * ramp_time
        cruise_time = max(0.0, (length - peak_speed * ramp_time) / max(peak_speed, 1e-12))
        duration = 2 * ramp_time + cruise_time
        next_time = min(duration, robot.segment_elapsed + self.dt)
        if next_time <= ramp_time:
            displacement = 0.5 * acceleration * next_time**2
        elif next_time <= ramp_time + cruise_time:
            displacement = 0.5 * acceleration * ramp_time**2 + peak_speed * (next_time - ramp_time)
        else:
            displacement = length - 0.5 * acceleration * (duration - next_time) ** 2
        proposed = robot.segment_origin + segment / max(length, 1e-12) * displacement
        travelled = float(np.linalg.norm(proposed - current))
        robot.speed = travelled / self.dt
        if robot.kind == "amr":
            collision = self._navigation_collision(robot, current, proposed)
            if collision:
                robot.speed = 0.0
                robot.velocity[:] = 0
                robot.state = "blocked"
                robot.blocked_elapsed += self.dt
                if robot.id not in self._motion_blocks:
                    self._motion_blocks.add(robot.id)
                    self._event("collision_blocked", robot, reason=collision)
                if robot.blocked_elapsed >= self.BLOCKED_TIMEOUT:
                    self._finish(robot, False, f"navigation blocked: {collision}")
                return False
            robot.position[:] = proposed
            robot.ee[:] = proposed + np.array([0, 0, 0.62])
        else:
            robot.ee[:] = proposed
        robot.segment_elapsed = next_time
        robot.velocity = delta / distance * (travelled / self.dt)
        robot.state = "working"
        robot.blocked_elapsed = 0.0
        self._motion_blocks.discard(robot.id)
        arrived = next_time >= duration
        if arrived:
            robot.segment_target = None
            robot.segment_origin = None
        return arrived

    def _grasp(self, robot: Robot) -> None:
        cargo_id = robot.command["cargo_id"]
        item = self.cargo[cargo_id]
        source = self._cargo_position(cargo_id)
        measured_distance = float(np.linalg.norm(source - robot.ee))
        if measured_distance > self.GRASP_TOLERANCE:
            self._finish(robot, False, "cargo moved outside grasp tolerance")
            return
        previous = item["owner"]
        if previous:
            other = self.robots[previous]
            if other.kind != "amr" or other.command or other.holding != cargo_id:
                self._finish(robot, False, "source custody changed before grasp")
                return
            other.holding = None
        item["owner"] = robot.id
        item["location"] = robot.id
        robot.holding = cargo_id
        self._event(
            "custody_acquired",
            robot,
            cargo_id=cargo_id,
            previous_owner=previous,
            measured_distance=round(measured_distance, 5),
        )
        if robot.command["kind"] == "pick":
            self._finish(robot, True)
        else:
            try:
                target = self._placement_target(robot, robot.command)
                robot.phase = "approach_place"
                robot.waypoints = (
                    self._lifted_path(robot.ee, target, robot.kind)
                    if self.motor_policy is None
                    else []
                )
                robot.speed = 0.0
            except ValueError as exc:
                self._finish(robot, False, str(exc))

    def _release(self, robot: Robot) -> None:
        command = robot.command
        cargo_id = command["cargo_id"]
        if robot.holding != cargo_id or self.cargo[cargo_id]["owner"] != robot.id:
            self._finish(robot, False, "custody lost before release")
            return
        try:
            target = self._placement_target(robot, command)
        except ValueError as exc:
            self._finish(robot, False, str(exc))
            return
        if np.linalg.norm(robot.ee - target) > self.GRASP_TOLERANCE:
            self._finish(robot, False, "target moved before release")
            return
        target_id = command["target_id"]
        item = self.cargo[cargo_id]
        if target_id in self.robots:
            receiver = self.robots[target_id]
            item["owner"] = target_id
            receiver.holding = cargo_id
            # Custody changes now; task stage changes on verified completion.
        else:
            item["owner"] = None

        item["location"] = target_id
        robot.holding = None
        # No position update occurs here: the payload remains at its observed
        # handoff pose, then receives gravity/contact or follows the deck.
        self._event(
            "custody_released",
            robot,
            cargo_id=cargo_id,
            receiver=target_id,
            position=self._vec(self._cargo_position(cargo_id)),
        )
        if target_id in self.robots:
            # A rigid deck attachment has no settling period. Commit the
            # verified receiving custody and completed transfer together;
            # otherwise the next tick would falsely look like a new takeback.
            self._sync_owned_cargo()
            self._finish(robot, True)
        else:
            robot.phase = "settling"
            robot.blocked_elapsed = 0.0
            robot.velocity[:] = 0.0

    def _settled_at_target(self, robot: Robot) -> bool:
        command = robot.command
        cargo_id, target_id = command["cargo_id"], command["target_id"]
        target = self._placement_target(robot, command, check_occupancy=False)
        position = self._cargo_position(cargo_id)
        qvel = self._qvel[cargo_id]
        speed = float(np.linalg.norm(self.data.qvel[qvel : qvel + 3]))
        if np.linalg.norm(position - target) > 0.065 or speed > 0.08:
            return False
        if target_id == "pallet" or target_id.startswith("pallet-"):
            item = self.cargo[cargo_id]
            if not item["processed"] or item["owner"] is not None:
                return False
            item["delivered"] = True
            item["stage"] = "completed"
            self._event(
                "delivery_verified",
                robot,
                cargo_id=cargo_id,
                slot=item["index"],
                position=self._vec(position),
            )
        return True

    def _finish(self, robot: Robot, success: bool, reason: str = "") -> None:
        command = robot.command
        item = self.cargo.get(command.get("cargo_id"), {})
        if success and item:
            next_stage = None  # Outcomes derive from physical skill semantics, never model text.
            if not next_stage:
                if command["kind"] == "navigate":
                    next_stage = {"process-in": "at_cell", "dispatch": "at_dispatch"}.get(
                        command.get("target_id")
                    )
                elif command["kind"] in ("transfer", "place"):
                    next_stage = {
                        "amr-01": "loaded",
                        "fixture": "in_fixture",
                        "amr-02": "outbound_loaded",
                        "pallet": "completed",
                    }.get(command.get("target_id"))
                elif command["kind"] == "process":
                    next_stage = "processed"
            if next_stage:
                item["stage"] = next_stage
        self._command_results[command["command_id"]] = {
            "command_id": command["command_id"],
            "robot_id": robot.id,
            "status": "completed" if success else "failed",
            "detail": reason or "verified skill completion",
            "observed": {
                "stage": item.get("stage"),
                "custodian": item.get("owner"),
                "cargo_id": item.get("id"),
                "position": self._vec(self._cargo_position(item["id"]))
                if item
                else self._vec(robot.position),
            },
        }
        self._event(
            "command_completed" if success else "command_failed",
            robot,
            action=robot.command["kind"],
            cargo_id=robot.command.get("cargo_id"),
            target_id=robot.command.get("target_id"),
            success=success,
            reason=reason,
        )
        if self._lane_owner == robot.id:
            self._lane_owner = None
            self._event("resource_released", robot, resource_id="aisle-main")
        robot.command = None
        robot.state = "idle" if success else "fault"
        robot.phase = "idle" if success else "failed"
        robot.progress = 1.0 if success else robot.progress
        robot.speed = 0.0
        robot.velocity[:] = 0.0
        robot.waypoints = []
        robot.segment_target = None
        robot.segment_origin = None
        self._motion_blocks.discard(robot.id)

    def _reserve_lane(self, robot: Robot) -> None:
        self._lane_owner = robot.id
        robot.phase = "navigating"
        robot.state = "working"
        self._event("resource_acquired", robot, resource_id="aisle-main")

    def _navigation_target(self, target_id: str, robot: Robot) -> np.ndarray:
        if target_id in NAVIGATION["docks"]:
            return np.array([*STATIONS[target_id]["position"][:2], 0.0])
        if target_id in ("home", robot.id + "-home"):
            return np.array(ROBOT_SPECS[robot.id]["position"], dtype=float)
        raise ValueError("navigation target must be a marked dock or own home")

    def _placement_target(
        self, robot: Robot, command: dict, check_occupancy: bool = True
    ) -> np.ndarray:
        target_id = command.get("target_id", command.get("target"))
        command["target_id"] = target_id
        cargo_id = command["cargo_id"]
        item = self.cargo[cargo_id]
        if target_id in self.robots:
            receiver = self.robots[target_id]
            if receiver.kind != "amr":
                raise ValueError("place into a stationary fixture or AMR deck")
            if receiver.command or receiver.speed > 0.01:
                raise ValueError("receiver must be stopped and docked")
            required_dock = "load" if robot.kind == "humanoid" else "process-out"
            expected = np.array(STATIONS[required_dock]["position"][:2])
            if np.linalg.norm(receiver.position[:2] - expected) > self.DOCK_TOLERANCE:
                raise ValueError(f"receiver is not at {required_dock}")
            if check_occupancy and receiver.holding not in (None, cargo_id):
                raise ValueError("receiver deck is occupied")
            return receiver.ee.copy()
        if target_id == "fixture":
            if robot.kind != "arm":
                raise ValueError("only assembly arm can access fixture")
            if check_occupancy and any(
                other["id"] != cargo_id
                and other["location"] == "fixture"
                and other["owner"] is None
                for other in self.cargo.values()
            ):
                raise ValueError("fixture is occupied")
            return np.array(STATIONS["fixture"]["position"])
        if isinstance(target_id, str) and (
            target_id == "pallet" or target_id.startswith("pallet-")
        ):
            if robot.kind != "gantry":
                raise ValueError("only gantry can access pallet")
            if not item["processed"]:
                raise ValueError("unprocessed cargo cannot be dispatched")
            return np.array(pallet_slot(item["index"]))
        raise ValueError("unknown or unsupported placement target")

    @staticmethod
    def _lifted_path(current: np.ndarray, target: np.ndarray, kind: str) -> list[np.ndarray]:
        clearance = 1.75 if kind == "gantry" else (1.5 if kind == "humanoid" else 1.9)
        height = max(clearance, current[2], target[2] + 0.25)
        waypoints = [
            np.array([current[0], current[1], height]),
            np.array([target[0], target[1], height]),
            target.copy(),
        ]
        return [point for point in waypoints if np.linalg.norm(point - current) > 1e-7] or [
            target.copy()
        ]

    @staticmethod
    def _check_reach(robot: Robot, target: np.ndarray) -> None:
        if robot.kind == "gantry":
            if not (
                5.5 <= target[0] <= 8.7 and 0.7 <= target[1] <= 3.3 and 0.25 <= target[2] <= 3.1
            ):
                raise ValueError("target is outside the gantry envelope")
        else:
            shoulder = robot.position + np.array([0, 0, 1.3 if robot.kind == "humanoid" else 1.0])
            if np.linalg.norm(target - shoulder) > ROBOT_SPECS[robot.id]["reach"]:
                raise ValueError("target is outside manipulator reach")

    def _navigation_collision(self, robot: Robot, start: np.ndarray, end: np.ndarray) -> str | None:
        radius = ROBOT_SPECS[robot.id]["radius"]
        for point in (start, (start + end) / 2, end):
            if (
                abs(point[0]) > FLOOR_SIZE[0] / 2 - radius
                or abs(point[1]) > FLOOR_SIZE[1] / 2 - radius
            ):
                return "floor boundary"
        for obstacle in OBSTACLES:
            low = np.array(obstacle["center"]) - np.array(obstacle["half_size"]) - radius
            high = np.array(obstacle["center"]) + np.array(obstacle["half_size"]) + radius
            if self._segment_aabb(start[:2], end[:2], low, high):
                return obstacle["id"]
        for other in self.robots.values():
            if other.id == robot.id or other.kind not in ("amr", "humanoid"):
                continue
            separation = radius + ROBOT_SPECS[other.id]["radius"] + 0.06
            segment = end[:2] - start[:2]
            fraction = np.clip(
                np.dot(other.position[:2] - start[:2], segment)
                / max(float(np.dot(segment, segment)), 1e-12),
                0,
                1,
            )
            if np.linalg.norm(start[:2] + fraction * segment - other.position[:2]) < separation:
                return other.id
        return None

    @staticmethod
    def _segment_aabb(
        start: np.ndarray, end: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> bool:
        lower, upper = 0.0, 1.0
        delta = end - start
        for axis in range(2):
            if abs(delta[axis]) < 1e-12:
                if start[axis] < low[axis] or start[axis] > high[axis]:
                    return False
                continue
            values = sorted(
                ((low[axis] - start[axis]) / delta[axis], (high[axis] - start[axis]) / delta[axis])
            )
            lower, upper = max(lower, values[0]), min(upper, values[1])
            if lower > upper:
                return False
        return True

    def _physics(self) -> None:
        for robot_id, mocap_id in self._decks.items():
            robot = self.robots[robot_id]
            self.data.mocap_pos[mocap_id] = robot.position + np.array([0, 0, 0.46])
            self.data.mocap_quat[mocap_id] = [
                math.cos(robot.yaw / 2),
                0,
                0,
                math.sin(robot.yaw / 2),
            ]
        substeps = max(1, round(self.dt / float(self.model.opt.timestep)))
        for _ in range(substeps):
            self._sync_owned_cargo()
            mujoco.mj_step(self.model, self.data)
        self._sync_owned_cargo()
        mujoco.mj_forward(self.model, self.data)

    def _sync_owned_cargo(self) -> None:
        for cargo_id, item in self.cargo.items():
            if item["owner"]:
                owner = self.robots[item["owner"]]
                address, velocity = self._qpos[cargo_id], self._qvel[cargo_id]
                self.data.qpos[address : address + 3] = owner.ee
                self.data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
                self.data.qvel[velocity : velocity + 6] = 0

    def _cargo_position(self, cargo_id: str) -> np.ndarray:
        address = self._qpos[cargo_id]
        return self.data.qpos[address : address + 3].copy()

    def _update_joints(self, robot: Robot) -> None:
        if robot.kind == "amr":
            robot.joints = {"wheel_angle": round(robot.elapsed * robot.speed / 0.13, 5)}
            robot.joint_points = []
            return
        if robot.kind == "gantry":
            robot.joints = {
                "bridge_x": round(robot.ee[0], 5),
                "trolley_y": round(robot.ee[1], 5),
                "hoist": round(3.65 - robot.ee[2], 5),
                "gripper": robot.gripper_opening
                if self.motor_policy is not None
                else (0.0 if robot.holding else 1.0),
            }
            robot.joint_points = [self._vec([robot.ee[0], robot.ee[1], 3.65]), self._vec(robot.ee)]
            return
        shoulder = robot.position + np.array([0, 0, 1.32 if robot.kind == "humanoid" else 1.0])
        vector = robot.ee - shoulder
        yaw = math.atan2(vector[1], vector[0])
        horizontal = math.hypot(vector[0], vector[1])
        length1, length2 = (0.90, 0.90) if robot.kind == "humanoid" else (1.45, 1.35)
        distance = float(np.linalg.norm(vector))
        reach = max(abs(length1 - length2) + 1e-5, min(distance, length1 + length2 - 1e-5))
        elbow = math.acos(
            np.clip(
                (reach * reach - length1 * length1 - length2 * length2) / (2 * length1 * length2),
                -1,
                1,
            )
        )
        shoulder_angle = math.atan2(vector[2], horizontal) + math.atan2(
            length2 * math.sin(elbow), length1 + length2 * math.cos(elbow)
        )
        elbow_point = shoulder + np.array(
            [
                math.cos(yaw) * length1 * math.cos(shoulder_angle),
                math.sin(yaw) * length1 * math.cos(shoulder_angle),
                length1 * math.sin(shoulder_angle),
            ]
        )
        robot.joints = {
            "base_yaw": round(yaw, 5),
            "shoulder": round(shoulder_angle, 5),
            "elbow": round(-elbow, 5),
            "wrist": round(elbow - shoulder_angle, 5),
            "gripper": robot.gripper_opening
            if self.motor_policy is not None
            else (0.0 if robot.holding else 1.0),
        }
        robot.joint_points = [self._vec(shoulder), self._vec(elbow_point), self._vec(robot.ee)]

    @staticmethod
    def _vec(value: Any) -> list[float]:
        return [round(float(component), 6) for component in value]

    def close(self) -> None:
        """Release learned-camera GL resources on the producer's owning thread."""
        if self.motor_policy is not None:
            from .vision import close_renderers

            close_renderers()
            close = getattr(self.motor_policy, "close", None)
            if close is not None:
                close()
