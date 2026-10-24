"""Actual MuJoCo RGB sensors for the original, fiducial-equipped factory.

The sensor rig mirrors measured factory geometry and physical tag locations.
Every tag is present independently of the instruction.  It is deliberately a
simplified inspection-camera scene, not the browser's decorative PBR renderer.
Only the rendered RGB reaches the learned policy; coordinates are used here as
simulator camera input and, separately, as supervised training labels.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import threading

import numpy as np

from .scene import OBSTACLES, ROBOT_SPECS, STATIONS, pallet_slot

ROBOT_IDS = tuple(ROBOT_SPECS)
TARGETS = (
    *(f"part-{i:03}" for i in range(1, 7)),
    "load",
    "process-in",
    "process-out",
    "dispatch",
    "fixture",
    "amr-01",
    "amr-02",
    *(f"pallet-{i:03}" for i in range(1, 7)),
    "humanoid-01",
    "arm-01",
    "gantry-01",
)
# Constant-norm colours avoid an anti-aliased bright tag looking like a
# different darker tag. Every ID has a distinct hue; no tag depends on a task.
_COLOR_RAYS = np.asarray(
    [colorsys.hsv_to_rgb(i / len(TARGETS), 0.86, 1.0) for i in range(len(TARGETS))]
)
PALETTE = (0.63 * _COLOR_RAYS / np.linalg.norm(_COLOR_RAYS, axis=1, keepdims=True)).astype(
    np.float32
)
TAG_OFFSETS = {
    name: np.array([0.0, 0.0, 0.13])
    if name.startswith("part-")
    else np.array([0.25, 0.22, 0.12])
    if name.startswith("amr-")
    else np.array([0.14, -0.12, 0.05])
    if name.startswith("pallet-")
    else np.array([0.40, -0.35, 0.45])
    if name in ("humanoid-01", "arm-01", "gantry-01")
    else np.array([-0.30, -0.25, 0.04])
    for name in TARGETS
}
VIEW_HEIGHT, VIEW_WIDTH = 128, 256
IMAGE_SHAPE = (VIEW_HEIGHT, VIEW_WIDTH * 2, 3)


def camera_center(robot_id):
    if ROBOT_SPECS[robot_id]["kind"] == "amr":
        return np.array([0.0, 0.0, 1.0]), 14.0
    p = np.array(ROBOT_SPECS[robot_id]["position"], dtype=float)
    return np.array([p[0], 2.0, 1.0]), 4.0


def sensor_spec():
    """Canonical numerical sensor contract bound into every checkpoint."""
    return {
        "version": "factory-inspection-rgb-v2",
        "image_shape": list(IMAGE_SHAPE),
        "targets": list(TARGETS),
        "palette": PALETTE.tolist(),
        "tag_offsets": {name: value.tolist() for name, value in TAG_OFFSETS.items()},
        "tag_radius": 0.065,
        "material": {"emission": 1, "specular": 0, "shininess": 0},
        "cameras": {
            robot: {
                "center": camera_center(robot)[0].tolist(),
                "span": camera_center(robot)[1],
                "top_offset": [0, 0, 100],
                "oblique_offset": [0, -80, 60],
                "top_xyaxes": [1, 0, 0, 0, 1, 0],
                "oblique_xyaxes": [1, 0, 0, 0, 0.6, 0.8],
            }
            for robot in ROBOT_IDS
        },
        "obstacles": OBSTACLES,
    }


def sensor_signature():
    return hashlib.sha256(
        json.dumps(sensor_spec(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class FactoryCameraRig:
    """Two fixed calibrated perspective RGB cameras for every robot.

    Neutral workcell surfaces and small coloured spherical inspection tags are
    real MuJoCo geometry.  The tags represent physical local reference points,
    including empty docking beds.  Segmentation is exposed only for training.
    """

    def __init__(self):
        import mujoco

        self.mujoco = mujoco
        static = []
        for item in OBSTACLES:
            x, y = item["center"]
            sx, sy = item["half_size"]
            h = item["height"]
            static.append(
                f'<geom type="box" pos="{x} {y} {h / 2}" size="{sx} {sy} {h / 2}" rgba=".20 .22 .24 1" contype="0" conaffinity="0"/>'
            )
        tags = []
        for index, name in enumerate(TARGETS):
            # Emissive tags make calibrated colour IDs robust to local lighting.
            tags.append(
                f'<body name="tag-{name}" mocap="true" pos="0 0 -20"><geom name="tag-{name}" type="sphere" size=".065" material="mat-{index}" contype="0" conaffinity="0"/></body>'
            )
        materials = "".join(
            f'<material name="mat-{i}" rgba="{" ".join(map(str, c))} 1" emission="1" specular="0" shininess="0"/>'
            for i, c in enumerate(PALETTE)
        )
        cameras = []
        for robot_id in ROBOT_IDS:
            center, span = camera_center(robot_id)
            x, y, z = center
            fovy = math.degrees(2 * math.atan(span / 200))
            cameras += [
                f'<camera name="{robot_id}-top" pos="{x} {y} {z + 100}" xyaxes="1 0 0 0 1 0" fovy="{fovy}"/>',
                f'<camera name="{robot_id}-front" pos="{x} {y - 80} {z + 60}" xyaxes="1 0 0 0 .6 .8" fovy="{fovy}"/>',
            ]
        xml = f'<mujoco><visual><global offwidth="256" offheight="128"/><quality shadowsize="1024"/><headlight ambient=".4 .4 .4" diffuse="0 0 0" specular="0 0 0"/><map znear=".001" zfar="200"/></visual><asset>{materials}</asset><worldbody><geom type="plane" size="14 8 .1" rgba=".09 .10 .12 1"/>{"".join(static)}{"".join(tags)}{"".join(cameras)}</worldbody></mujoco>'
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=VIEW_HEIGHT, width=VIEW_WIDTH)
        self.mocap = {name: int(self.model.body(f"tag-{name}").mocapid[0]) for name in TARGETS}
        self.geoms = {name: self.model.geom(f"tag-{name}").id for name in TARGETS}
        self.positions = {}

    def set_positions(self, positions):
        self.positions = {
            name: np.asarray(p, dtype=float) for name, p in positions.items() if name in self.mocap
        }
        for name, mocap_id in self.mocap.items():
            self.data.mocap_pos[mocap_id] = (
                self.positions[name] + TAG_OFFSETS[name]
                if name in self.positions
                else [0.0, 0.0, -20.0]
            )
        self.mujoco.mj_forward(self.model, self.data)

    def render(self, robot_id, segmentation=False):
        if robot_id not in ROBOT_IDS:
            raise ValueError("Unknown factory camera")
        views, masks = [], []
        for view in ("top", "front"):
            self.renderer.update_scene(self.data, camera=f"{robot_id}-{view}")
            views.append(self.renderer.render().copy())
            if segmentation:
                self.renderer.enable_segmentation_rendering()
                raw = self.renderer.render().copy()
                self.renderer.disable_segmentation_rendering()
                mask = np.zeros(raw.shape[:2], dtype=np.uint8)
                for index, name in enumerate(TARGETS, 1):
                    mask[
                        (raw[..., 0] == self.geoms[name])
                        & (raw[..., 1] == int(self.mujoco.mjtObj.mjOBJ_GEOM))
                    ] = index
                masks.append(mask)
        rgb = np.concatenate(views, axis=1)
        return (rgb, np.concatenate(masks, axis=1)) if segmentation else rgb

    def close(self):
        self.renderer.close()


def marker_positions(snapshot):
    """Camera-scene construction only. Never called inside policy inference."""
    positions = {key: value["position"] for key, value in STATIONS.items() if key in TARGETS}
    positions.update({f"pallet-{i + 1:03}": pallet_slot(i) for i in range(6)})
    for robot in snapshot["robots"]:
        positions[robot["id"]] = robot["ee_position"]
    for cargo in snapshot["cargo"]:
        positions[cargo["id"]] = cargo["position"]
    return positions


def local_marker_positions(snapshot, robot_id):
    """Limit dynamic observations to each robot's physical work area.

    Fixed station/slot beacons are a shared static navigation map. Cargo and
    peer tags are observed only nearby; the task's private inspection camera
    is separate and never becomes a globally visible scene marker.
    """
    positions = marker_positions(snapshot)
    robot = next(item for item in snapshot["robots"] if item["id"] == robot_id)
    if robot["kind"] == "amr":
        center, radius = np.asarray(robot["position"][:2]), 2.5
    else:
        center = np.asarray([robot["position"][0], 2.0])
        radius = {"humanoid": 2.5, "arm": 3.0, "gantry": 3.0}[robot["kind"]]
    dynamic = set(ROBOT_IDS) | {cargo["id"] for cargo in snapshot["cargo"]}
    return {
        name: pose
        for name, pose in positions.items()
        if name not in dynamic
        or name == robot_id
        or np.linalg.norm(np.asarray(pose[:2]) - center) <= radius
    }


_LOCAL = threading.local()


def render_observation(snapshot, robot_id):
    rig = getattr(_LOCAL, "rig", None)
    if rig is None:
        rig = FactoryCameraRig()
        _LOCAL.rig = rig
    positions = (
        local_marker_positions(snapshot, robot_id)
        if snapshot.get("task") == "dispatch-routing"
        else marker_positions(snapshot)
    )
    rig.set_positions(positions)
    return rig.render(robot_id)


def close_renderers():
    rig = getattr(_LOCAL, "rig", None)
    if rig is not None:
        rig.close()
        del _LOCAL.rig


def instruction_for(robot, command, phase):
    cargo_id = command.get("cargo_id", "")
    target = command.get("target_id", command.get("target", ""))
    if phase in ("navigating", "waiting_for_aisle"):
        return f"navigate to {target}"
    if phase == "approach_pick":
        return f"pick {cargo_id}"
    if phase == "approach_place":
        if target == "pallet":
            selected_slot = command.get("pallet_slot")
            if selected_slot is not None:
                if type(selected_slot) is not int or not 0 <= selected_slot < 6:
                    raise ValueError("Invalid local pallet choice")
                target = f"pallet-{selected_slot + 1:03}"
            else:
                target = f"pallet-{int(cargo_id.split('-')[-1]):03}"
        return f"place {cargo_id} on {target}"
    if phase in ("approach_process", "processing"):
        return f"process {cargo_id}"
    return "wait"


def make_observation(snapshot, robot_id, command=None, phase=None, rgb=None, peer_embedding=None):
    robot = next(item for item in snapshot["robots"] if item["id"] == robot_id)
    command = command or robot.get("command") or {}
    phase = phase or robot.get("phase", "idle")
    return {
        "robot_id": robot_id,
        "kind": robot["kind"],
        "rgb": render_observation(snapshot, robot_id) if rgb is None else rgb,
        "instruction": instruction_for(robot, command, phase),
        "phase": phase,
        "command_id": command.get("command_id"),
        "proprio": {
            key: robot[key] for key in ("position", "ee_position", "yaw", "velocity", "holding")
        },
        "tick": snapshot["tick"],
        "dt": snapshot.get("dt", 0.1),
        "peer_embedding": peer_embedding,
    }
