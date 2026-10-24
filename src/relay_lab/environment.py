"""An original, small two-gripper MuJoCo handoff environment.

The physics is real for released cubes; grasping is deliberately idealized. A
close command attaches a nearby visible cube to the Cartesian end effector,
then kinematically carries it until release. There is no arm IK, learned action
model, force-controlled grasp, or gripper collision model. This keeps the first
communication experiment reproducible and runnable without training data.
"""

from pathlib import Path
from typing import ClassVar

import mujoco
import numpy as np

from .types import (
    AGENTS,
    COLORS,
    MAX_DELTA,
    MIN_HEIGHT,
    SAFE_HEIGHT,
    Action,
    AgentId,
    Observation,
    Vec3,
)
from .visuals import sync_gantry


class RelayEnv:
    """Two local views, one shared relay, and a packing order private to packer.

    One control tick advances 100 ms of physics. ``state()`` and the overview
    camera are spectator/debug interfaces and must never be passed to a policy.
    ``observe()`` is the policy boundary, including its masked RGB camera.
    """

    MAX_DELTA = MAX_DELTA
    SAFE_HEIGHT = SAFE_HEIGHT
    GRASP_RADIUS = 0.055
    SUBSTEPS = 50
    HOME: ClassVar[dict[AgentId, Vec3]] = {
        "picker": (-0.54, 0.0, SAFE_HEIGHT),
        "packer": (0.48, 0.0, SAFE_HEIGHT),
    }
    TARGETS: ClassVar[dict[str, Vec3]] = {
        "relay": (0.0, 0.0, 0.035),
        "slot_0": (0.48, -0.22, 0.035),
        "slot_1": (0.48, 0.0, 0.035),
        "slot_2": (0.48, 0.22, 0.035),
    }
    BOUNDS: ClassVar[dict[AgentId, tuple[Vec3, Vec3]]] = {
        "picker": ((-0.72, -0.38, MIN_HEIGHT), (0.06, 0.38, 0.42)),
        "packer": ((-0.06, -0.38, MIN_HEIGHT), (0.72, 0.38, 0.42)),
    }

    def __init__(self, seed: int = 0, order: tuple[str, ...] | None = None):
        if order is not None and (len(order) != 3 or set(order) != set(COLORS)):
            raise ValueError("order must contain coral, mint, and gold exactly once")
        self._requested_order = tuple(order) if order is not None else None
        scene = Path(__file__).parent / "assets" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self._renderers = {}
        self._closed = False
        self._qpos = {}
        self._qvel = {}
        self._object_geoms = {}
        self._mocap = {}
        self._fingers = {}
        for color in COLORS:
            joint = self.model.joint(f"{color}_joint")
            self._qpos[color] = int(joint.qposadr[0])
            self._qvel[color] = int(joint.dofadr[0])
            self._object_geoms[color] = self.model.geom(f"{color}_geom").id
        for agent in AGENTS:
            self._mocap[agent] = int(self.model.body(f"{agent}_gripper").mocapid[0])
            self._fingers[agent] = tuple(
                self.model.geom(f"{agent}_finger_{side}").id for side in ("l", "r")
            )
        self.reset(seed)

    def reset(self, seed: int | None = None) -> dict[AgentId, Observation]:
        """Restart a deterministic episode; omitted seed repeats the last seed."""
        self._ensure_open()
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        self.tick = 0
        self._holding = {agent: None for agent in AGENTS}
        random_order = tuple(str(color) for color in rng.permutation(COLORS))
        # Consume the same RNG values so task changes preserve the source scene.
        self.order = self._requested_order or random_order
        sources = np.array([(-0.50, -0.23, 0.035), (-0.58, 0.0, 0.035), (-0.50, 0.23, 0.035)])
        sources = sources[rng.permutation(len(sources))]
        sources[:, :2] += rng.uniform(-0.018, 0.018, size=(3, 2))
        for color, position in zip(COLORS, sources):
            self._set_object_position(color, position)
        for agent in AGENTS:
            self.data.mocap_pos[self._mocap[agent]] = self.HOME[agent]
            self._set_fingers(agent, closed=False)
        mujoco.mj_forward(self.model, self.data)
        # Let the 5 mm initial clearance settle without advancing control time.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        return {agent: self.observe(agent) for agent in AGENTS}

    @property
    def packed(self) -> tuple[str | None, ...]:
        """Released, settled cube identity at each numbered tray slot."""
        slots: list[str | None] = []
        for slot in range(3):
            target = np.asarray(self.TARGETS[f"slot_{slot}"])
            candidates = []
            for color in COLORS:
                if self._owner(color) is not None:
                    continue
                position = self._object_position(color)
                speed = np.linalg.norm(self.data.qvel[self._qvel[color] : self._qvel[color] + 3])
                distance = np.linalg.norm(position[:2] - target[:2])
                if distance < 0.055 and 0.015 < position[2] < 0.065 and speed < 0.15:
                    candidates.append(color)
            slots.append(candidates[0] if len(candidates) == 1 else None)
        return tuple(slots)

    @property
    def success(self) -> bool:
        return self.packed == self.order

    def observe(self, agent_id: AgentId, rgb: bool = False) -> Observation:
        self._ensure_open()
        self._check_agent(agent_id)
        objects = {
            color: self._vec(self._object_position(color))
            for color in COLORS
            if self._visible(color, agent_id)
        }
        targets = {"relay": self.TARGETS["relay"], "home": self.HOME[agent_id]}
        if agent_id == "packer":
            targets.update({key: pos for key, pos in self.TARGETS.items() if key != "relay"})
        return Observation(
            tick=self.tick,
            agent_id=agent_id,
            ee_position=self._vec(self.data.mocap_pos[self._mocap[agent_id]]),
            holding=self._holding[agent_id],
            objects=objects,
            targets=targets,
            packed=self.packed if agent_id == "packer" else (),
            order=self.order if agent_id == "packer" else None,
            rgb=self.render(camera=agent_id, width=256, height=256) if rgb else None,
            wrist_rgb=self.render(camera=f"{agent_id}_wrist", width=256, height=256)
            if rgb
            else None,
            gripper_opening=float(2 * (self.model.geom_pos[self._fingers[agent_id][1], 0] - 0.009)),
        )

    def step(self, actions: dict[str, Action]) -> dict:
        """Move both grippers, release, then grasp, then advance the physics.

        Missing agents wait. Deltas are norm-limited before workspace clipping.
        A handoff requires a release followed by a fresh close command; one
        agent cannot steal a cube held by the other. Releases are processed
        before closes, so a same-tick handoff at the relay is well-defined.
        """
        self._ensure_open()
        unknown = set(actions) - set(AGENTS)
        if unknown:
            raise ValueError(f"Unknown agents: {sorted(unknown)}")
        # Validate all inputs before changing either robot's state.
        validated = {}
        for agent in AGENTS:
            action = actions.get(agent, Action())
            delta = np.asarray(action.delta, dtype=float)
            if delta.shape != (3,) or not np.all(np.isfinite(delta)):
                raise ValueError("Action.delta must contain three finite numbers")
            if action.gripper not in (-1, 0, 1):
                raise ValueError("Action.gripper must be -1, 0, or 1")
            length = np.linalg.norm(delta)
            if length > MAX_DELTA:
                delta = delta * (MAX_DELTA / length)
            validated[agent] = (delta, action.gripper)
        before = self.packed
        previously_successful = self.success
        events = []
        for agent, (delta, _) in validated.items():
            lower, upper = self.BOUNDS[agent]
            mocap = self._mocap[agent]
            self.data.mocap_pos[mocap] = np.clip(self.data.mocap_pos[mocap] + delta, lower, upper)
        self._sync_held_objects()
        for agent, (_, gripper) in validated.items():
            if gripper == 1:
                self._set_fingers(agent, closed=False)
                color = self._holding[agent]
                if color is not None:
                    self._holding[agent] = None
                    events.append({"type": "release", "agent": agent, "object": color})
        for agent, (_, gripper) in validated.items():
            if gripper != -1 or self._holding[agent] is not None:
                continue
            self._set_fingers(agent, closed=True)
            ee = self.data.mocap_pos[self._mocap[agent]]
            candidates = [
                (np.linalg.norm(self._object_position(color) - ee), color)
                for color in COLORS
                if self._owner(color) is None
                and self._visible(color, agent)
                and self._in_workspace(self._object_position(color), agent)
            ]
            distance, color = min(candidates, default=(float("inf"), None))
            if distance <= self.GRASP_RADIUS:
                self._holding[agent] = color
                events.append({"type": "grasp", "agent": agent, "object": color})
            else:
                events.append({"type": "grasp_missed", "agent": agent})
        for _ in range(self.SUBSTEPS):
            self._sync_held_objects()
            mujoco.mj_step(self.model, self.data)
        self._sync_held_objects()
        mujoco.mj_forward(self.model, self.data)
        self.tick += 1
        after = self.packed
        for slot, (old, new) in enumerate(zip(before, after)):
            if old != new:
                if old is not None:
                    events.append({"type": "unpacked", "slot": slot, "object": old})
                if new is not None:
                    events.append({"type": "packed", "slot": slot, "object": new})
        success = self.success
        if success and not previously_successful:
            events.append({"type": "success"})
        return {"events": events, "success": success}

    def state(self) -> dict:
        """JSON-compatible full state for saved rollouts and spectator replay."""
        self._ensure_open()
        return {
            "tick": self.tick,
            "seed": self.seed,
            "success": self.success,
            "order": list(self.order),
            "packed": list(self.packed),
            "objects": {color: self._object_position(color).tolist() for color in COLORS},
            "object_holders": {color: self._owner(color) for color in COLORS},
            "robots": {
                agent: {
                    "position": self.data.mocap_pos[self._mocap[agent]].tolist(),
                    "holding": self._holding[agent],
                }
                for agent in AGENTS
            },
            "targets": {name: list(position) for name, position in self.TARGETS.items()},
        }

    def render(self, camera: str = "overview", width: int = 960, height: int = 640) -> np.ndarray:
        """Render RGB lazily. Agent cameras mask every locally hidden cube.

        ``overview`` is explicitly a privileged spectator view. To run on a
        Linux GPU server, select the MuJoCo EGL backend before importing Python
        modules (MUJOCO_GL=egl); no graphics backend is needed unless rendering.
        """
        self._ensure_open()
        camera_agent = camera.removesuffix("_wrist")
        if camera not in ("overview", *AGENTS, "picker_wrist", "packer_wrist"):
            raise ValueError(
                "camera must be overview, picker, packer, picker_wrist, or packer_wrist"
            )
        if not (1 <= width <= 1280 and 1 <= height <= 960):
            raise ValueError("Render dimensions must fit the 1280 x 960 framebuffer")
        # Reuse the policy-sized and spectator-sized renderers across frames.
        if (width, height) not in self._renderers:
            self._renderers[width, height] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._renderers[width, height]
        option = mujoco.MjvOption()
        option.geomgroup[5] = 0
        originals = self.model.geom_group.copy()
        try:
            if camera_agent in AGENTS:
                for color, geom_id in self._object_geoms.items():
                    if not self._visible(color, camera_agent):
                        self.model.geom_group[geom_id] = 5
            sync_gantry(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            renderer.update_scene(self.data, camera=camera, scene_option=option)
            return renderer.render().copy()
        finally:
            self.model.geom_group[:] = originals

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _visible(self, color: str, agent: AgentId) -> bool:
        owner = self._owner(color)
        if owner is not None:
            return owner == agent
        position = self._object_position(color)
        relay = np.linalg.norm(position[:2]) <= 0.085 and position[2] < 0.12
        local = position[0] < -0.06 if agent == "picker" else position[0] > 0.06
        return bool(relay or local)

    def _owner(self, color: str) -> AgentId | None:
        return next((agent for agent in AGENTS if self._holding[agent] == color), None)

    def _in_workspace(self, position: np.ndarray, agent: AgentId) -> bool:
        lower, upper = (np.asarray(bound, dtype=float) for bound in self.BOUNDS[agent])
        # Settled cube centers lie 5 mm below the nominal end-effector minimum.
        lower[2] -= 0.015
        return bool(np.all(position >= lower) and np.all(position <= upper))

    def _object_position(self, color: str) -> np.ndarray:
        start = self._qpos[color]
        return self.data.qpos[start : start + 3].copy()

    def _set_object_position(self, color: str, position):
        start = self._qpos[color]
        velocity = self._qvel[color]
        self.data.qpos[start : start + 3] = position
        self.data.qpos[start + 3 : start + 7] = (1, 0, 0, 0)
        self.data.qvel[velocity : velocity + 6] = 0

    def _sync_held_objects(self):
        for agent, color in self._holding.items():
            if color is not None:
                self._set_object_position(color, self.data.mocap_pos[self._mocap[agent]])

    def _set_fingers(self, agent: AgentId, closed: bool):
        gap = 0.039 if closed else 0.052
        left, right = self._fingers[agent]
        self.model.geom_pos[left, 0] = -gap
        self.model.geom_pos[right, 0] = gap

    @staticmethod
    def _vec(position) -> tuple[float, float, float]:
        return tuple(float(value) for value in position)

    @staticmethod
    def _check_agent(agent):
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent: {agent}")

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("This RelayEnv has been closed")
