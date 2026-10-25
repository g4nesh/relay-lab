"""Real MuJoCo flex cloth with two motor-driven articulated manipulators.

Grasping uses a distance-gated connect constraint on a marked hem/sleeve tab.
This is explicitly an idealized point clamp, not simulated frictional pinch.
All remaining cloth vertices move solely through MuJoCo dynamics and contacts.
Cloth contacts the table, obstacle and itself; arm-shell/cloth and arm-arm
collision are disabled. This isolates the explicitly idealized point clamp
from an uncalibrated finger friction model. Arm/table contacts remain active.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import presentation
from .scene import AGENTS, build_scene, cotton_texture


class ClothWorld:
    CONTROL_DT = 0.04
    SUBSTEPS = 10
    MAX_MOVE = 0.018
    GRASP_RADIUS = 0.035
    MAX_VALID_CLOTH_SPEED = 5.0
    HOME: ClassVar = {"left": (-0.38, -0.28, 0.28), "right": (0.38, -0.28, 0.28)}
    PHYSICS_CONFIG: ClassVar = {
        "engine": "MuJoCo 3.17",
        "garment": "single-layer T-shirt silhouette, 271 vertices / 460 triangles",
        "fixture": "six explicit upper-torso vacuum-table connect constraints, released before scoring",
        "grasp": "distance-gated idealized marked-tab vertex clamp; no frictional pinch or layer inference",
        "cloth_self_collision": True,
        "cloth_table_collision": True,
        "cloth_obstacle_collision": True,
        "cloth_arm_shell_collision": False,
        "arm_arm_collision": False,
        "arm_table_collision": True,
        "max_valid_cloth_speed_m_s": 5.0,
        "material_calibration": "illustrative; not calibrated to a real textile",
        "collision_masks": {"cloth": [4, 1], "arms": [2, 1], "table": [1, 3], "obstacle": [1, 3]},
        "cloth_contact_radius_m": 0.0018,
        "elastic_thickness_m": 0.0004,
        "actuation": "six revolute position motors per arm, analytic Cartesian IK, gravity compensation",
    }
    PHASES = (
        "approach_sleeve_above",
        "approach_sleeve",
        "grasp_sleeve",
        "lift_sleeve",
        "carry_sleeve",
        "fold_sleeve",
        "release_sleeve",
        "retreat_sleeve",
        "approach_hem_above",
        "approach_hem",
        "grasp_hem",
        "lift_hem",
        "carry_hem",
        "fold_hem",
        "release_hem",
        "settle",
    )

    def __init__(
        self, seed: int = 0, disturbances: bool = True, obstacle_schedule=None, fixture: bool = True
    ):
        self.seed = int(seed)
        self.fixture = bool(fixture)
        self.disturbances = bool(disturbances)
        self.obstacle_schedule = (
            obstacle_schedule
            if obstacle_schedule is not None
            else ((110, 210, "left"), (300, 400, "right"))
        )
        self.xml, self.rest_vertices, self.triangles, self.landmarks = build_scene()
        self.model = mujoco.MjModel.from_xml_string(
            self.xml, assets={"cotton.png": cotton_texture(), **presentation.studio_assets()}
        )
        self._studio = presentation.StudioOverlay(self.model)
        self.data = mujoco.MjData(self.model)
        self._renderers = {}
        self._closed = False
        self._joints = {
            a: np.array([self.model.joint(f"{a}_j{j}").qposadr[0] for j in range(6)], dtype=int)
            for a in AGENTS
        }
        self._dofs = {
            a: np.array([self.model.joint(f"{a}_j{j}").dofadr[0] for j in range(6)], dtype=int)
            for a in AGENTS
        }
        self._sites = {a: self.model.site(f"{a}_ee").id for a in AGENTS}
        self._eq = {a: self.model.equality(f"grasp_{a}").id for a in AGENTS}
        self._bodies = np.asarray(
            [self.model.body(f"shirt_{i}").id for i in range(len(self.rest_vertices))]
        )
        self._cloth_dofs = np.asarray(
            [
                [int(self.model.jnt_dofadr[int(self.model.body_jntadr[b])]) + k for k in range(3)]
                for b in self._bodies
            ]
        )
        self._obstacle_mocap = int(self.model.body("obstacle").mocapid[0])
        self._fixture_eq = [self.model.equality(f"vacuum_{j}").id for j in range(6)]
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(self.seed)
        self.tick = 0
        self._stable_ticks = 0
        self.peak_cloth_speed = 0.0
        self.peak_edge_strain = 0.0
        self.holding = {a: None for a in AGENTS}
        self._targets = {a: np.asarray(v, dtype=float) for a, v in self.HOME.items()}
        self._grip = {a: False for a in AGENTS}
        self._obstacle_active = False
        self._obstacle_side = None
        self.events = []
        self._jitter = rng.uniform(-0.012, 0.012, size=2)
        self.fixture_active = self.fixture
        for eq in self._fixture_eq:
            body = int(self.model.eq_obj1id[eq])
            node = int(np.flatnonzero(self._bodies == body)[0])
            anchor = self.rest_vertices[node] + [*self._jitter, 0]
            anchor[2] = 0.0018
            self.model.eq_data[eq, :6] = [0, 0, 0, *anchor]
            self.data.eq_active[eq] = self.fixture_active
        for i, body in enumerate(self._bodies):
            addr = int(self.model.jnt_qposadr[int(self.model.body_jntadr[body])])
            self.data.qpos[addr : addr + 2] = self._jitter
            p = self.rest_vertices[i]
            self.data.qpos[addr + 2] = 0.002 * np.sin(p[0] * 24 + self.seed) * np.cos(p[1] * 18)
        for a in AGENTS:
            q = self.inverse_kinematics(a, self._targets[a])
            self.data.qpos[self._joints[a]] = q
            self.data.ctrl[(0 if a == "left" else 6) : (6 if a == "left" else 12)] = q
        mujoco.mj_forward(self.model, self.data)
        for _ in range(80):
            self._compensate_gravity()
            mujoco.mj_step(self.model, self.data)
        self._sync_markers()
        self._initial_vertices = self.vertices.copy()
        return self.snapshot()

    def release_fixture(self):
        """Release all six explicit vacuum-table constraints before evaluation."""
        self.fixture_active = False
        for eq in self._fixture_eq:
            self.data.eq_active[eq] = False
        self.events.append({"tick": self.tick, "event": "vacuum_fixture_released"})

    @property
    def vertices(self):
        return self.data.flexvert_xpos[: len(self.rest_vertices)].copy()

    def ee(self, agent):
        self._check_agent(agent)
        return self.data.site_xpos[self._sites[agent]].copy()

    @staticmethod
    def inverse_kinematics(agent, target):
        sign = -1 if agent == "left" else 1
        target = np.asarray(target, dtype=float)
        dx, dy = target[0] - sign * 0.76, target[1] - 0.10
        if agent == "right":
            dx, dy = -dx, -dy
        radial = np.hypot(dx, dy)
        dz = target[2] + 0.117 - 0.225
        distance = np.clip(np.hypot(radial, dz), 0.08, 0.845)
        q0 = np.arctan2(dy, dx)
        q1 = -np.arctan2(dz, radial) - np.arccos(
            np.clip((0.45**2 + distance**2 - 0.40**2) / (0.9 * distance), -1, 1)
        )
        q2 = np.pi - np.arccos(np.clip((0.45**2 + 0.40**2 - distance**2) / (0.36), -1, 1))
        return np.array([q0, q1, q2, -(q1 + q2), 0.0, 0.0])

    def _compensate_gravity(self):
        self.data.qfrc_applied[:] = 0
        for a in AGENTS:
            self.data.qfrc_applied[self._dofs[a]] = self.data.qfrc_bias[self._dofs[a]]

    def _sync_markers(self):
        vertices = self.vertices
        for a in AGENTS:
            for n, i in self.landmarks[a].items():
                mid = int(self.model.body(f"marker_{a}_{n}").mocapid[0])
                self.data.mocap_pos[mid] = vertices[i] + np.array([0, 0, 0.002])
        mujoco.mj_forward(self.model, self.data)

    def _update_obstacle(self):
        active = None
        if self.disturbances:
            for start, end, agent in self.obstacle_schedule:
                if start <= self.tick < end:
                    active = (start, end, agent)
                    break
        was_active = self._obstacle_active
        self._obstacle_active = active is not None
        if active:
            start, end, agent = active
            self._obstacle_side = agent
            sign = -1 if agent == "left" else 1
            ramp = min(1.0, (self.tick - start + 1) / 25, (end - self.tick) / 25)
            self.data.mocap_pos[self._obstacle_mocap] = [sign * (0.67 - 0.39 * ramp), -0.04, 0.18]
        else:
            self._obstacle_side = None
            self.data.mocap_pos[self._obstacle_mocap] = [2.3, 0, 0.18]
        if self._obstacle_active != was_active:
            self.events.append(
                {
                    "tick": self.tick,
                    "event": "obstacle_entered" if self._obstacle_active else "obstacle_cleared",
                    "side": self._obstacle_side,
                }
            )

    def _set_grip(self, agent, closed):
        eq = self._eq[agent]
        if not closed:
            if self.holding[agent] is not None:
                self.events.append(
                    {
                        "tick": self.tick,
                        "event": "released",
                        "agent": agent,
                        "vertex": self.holding[agent],
                    }
                )
            self.data.eq_active[eq] = False
            self.holding[agent] = None
            return
        if self.holding[agent] is not None:
            return
        vertices = self.vertices
        distances = np.linalg.norm(vertices - self.ee(agent), axis=1)
        eligible = [self.landmarks[agent][name] for name in ("hem", "sleeve")]
        i = min(eligible, key=lambda index: distances[index])
        if distances[i] <= self.GRASP_RADIUS and i not in self.holding.values():
            self.model.eq_obj2id[eq] = int(self._bodies[i])
            self.model.eq_data[eq, :6] = [0, 0, -0.052, 0, 0, 0]
            self.data.eq_active[eq] = True
            self.holding[agent] = i
            self.events.append(
                {
                    "tick": self.tick,
                    "event": "grasped",
                    "agent": agent,
                    "vertex": i,
                    "distance": float(distances[i]),
                }
            )

    def step(self, actions: Mapping[str, Mapping[str, Any]] | None = None):
        if self._closed:
            raise RuntimeError("world is closed")
        actions = actions or {}
        for a, action in actions.items():
            self._check_agent(a)
            target = np.asarray(action.get("target", self._targets[a]), dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("target must be a finite xyz vector")
            target = np.clip(target, [-0.59, -0.49, 0.008], [0.59, 0.42, 0.52])
            delta = target - self._targets[a]
            self._targets[a] += delta * min(1.0, self.MAX_MOVE / max(np.linalg.norm(delta), 1e-12))
            self._grip[a] = bool(action.get("grip", self._grip[a]))
        self._update_obstacle()
        for a in AGENTS:
            self._set_grip(a, self._grip[a])
            q = self.inverse_kinematics(a, self._targets[a])
            self.data.ctrl[(0 if a == "left" else 6) : (6 if a == "left" else 12)] = q
            for suffix, direction in (("l", -1), ("r", 1)):
                self.model.geom(f"{a}_finger_{suffix}").pos[0] = direction * (
                    0.012 if self._grip[a] else 0.022
                )
        previous_time = float(self.data.time)
        for _ in range(self.SUBSTEPS):
            self._compensate_gravity()
            mujoco.mj_step(self.model, self.data)
            self.peak_cloth_speed = max(self.peak_cloth_speed, self._cloth_speed())
            strain = np.max(np.abs(self.data.flexedge_length / self.model.flexedge_length0 - 1))
            self.peak_edge_strain = max(self.peak_edge_strain, float(strain))
        for a in AGENTS:
            self._set_grip(a, self._grip[a])
        self.tick += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("cloth dynamics became nonfinite")
        if not np.isclose(self.data.time - previous_time, self.CONTROL_DT, atol=1e-9):
            raise RuntimeError("MuJoCo reset during a physics step; episode is invalid")
        self._sync_markers()
        settled = (
            self._cloth_speed() < 0.12
            and all(v is None for v in self.holding.values())
            and not self.fixture_active
        )
        self._stable_ticks = self._stable_ticks + 1 if settled else 0
        return self.snapshot()

    def observe(self, agent: str, rgb: bool = False):
        self._check_agent(agent)
        vertices = self.vertices
        side = vertices[:, 0] <= 0.05 if agent == "left" else vertices[:, 0] >= -0.05
        landmark_positions = {
            name: vertices[i].tolist() for name, i in self.landmarks[agent].items() if side[i]
        }
        obstacle = None
        if self._obstacle_active and agent == self._obstacle_side:
            obstacle = {
                "position": self.data.mocap_pos[self._obstacle_mocap].tolist(),
                "radius": 0.065,
                "height": 0.15,
                "active": True,
            }
        result = {
            "tick": self.tick,
            "agent": agent,
            "ee_position": self.ee(agent).tolist(),
            "grasped": self.holding[agent] is not None,
            "grasp_target": (
                "none"
                if self.holding[agent] is None
                else next(
                    (
                        name
                        for name in ("hem", "sleeve")
                        if self.landmarks[agent][name] == self.holding[agent]
                    ),
                    "unknown",
                )
            ),
            "landmarks": landmark_positions,
            "obstacle": obstacle,
        }
        if rgb:
            result["rgb"] = self.render(agent, width=128, height=128)
        return result

    def reference_target(self, agent: str, phase: str, height: float = 0.20):
        """Teaching labels only; runtime learned controllers must not call this.

        These geometric labels do not establish that a complete fold succeeds.
        """
        self._check_agent(agent)
        if phase not in self.PHASES:
            raise ValueError(f"unknown phase {phase}")
        points = self.vertices
        sleeve = points[self.landmarks[agent]["sleeve"]]
        hem = points[self.landmarks[agent]["hem"]]
        shoulder = points[self.landmarks[agent]["shoulder"]]
        sign = -1 if agent == "left" else 1
        if phase == "approach_sleeve_above":
            target = sleeve + [0, 0, 0.17]
        elif phase in ("approach_sleeve", "grasp_sleeve"):
            target = (
                sleeve + [0, 0, 0.008]
                if self.holding[agent] is None
                else self._targets[agent].copy()
            )
        elif phase == "lift_sleeve":
            target = np.array([sign * 0.44, 0.16, 0.14]) + [*self._jitter, 0]
        elif phase == "carry_sleeve":
            target = np.array([sign * 0.04, 0.16, 0.14]) + [*self._jitter, 0]
        elif phase in ("fold_sleeve", "release_sleeve"):
            target = np.array([sign * 0.04, 0.16, 0.018]) + [*self._jitter, 0]
        elif phase == "retreat_sleeve":
            target = np.array([sign * 0.04, 0.16, 0.30]) + [*self._jitter, 0]
        elif phase == "approach_hem_above":
            target = hem + [0, 0, 0.17]
        elif phase in ("approach_hem", "grasp_hem"):
            target = (
                hem + [0, 0, 0.008] if self.holding[agent] is None else self._targets[agent].copy()
            )
        elif phase == "lift_hem":
            target = np.array([sign * 0.24, -0.04, max(0.25, height)]) + [*self._jitter, 0]
        elif phase == "carry_hem":
            target = np.array([shoulder[0], shoulder[1] - 0.12, 0.18])
        elif phase in ("fold_hem", "release_hem"):
            target = shoulder + [0, -0.04, 0.020]
        else:
            target = np.asarray(self.HOME[agent])
        return {
            "target": target.tolist(),
            "grip": phase
            in (
                "grasp_sleeve",
                "lift_sleeve",
                "carry_sleeve",
                "fold_sleeve",
                "grasp_hem",
                "lift_hem",
                "carry_hem",
                "fold_hem",
            ),
        }

    def _footprint(self, vertices):
        # Union of projected physical triangles, not convex hull or summed areas
        # (which would double-count overlapping layers). 3.9 mm raster cells.
        scale = 256.0
        image = Image.new("1", (512, 512))
        draw = ImageDraw.Draw(image)
        pixel_points = (vertices[:, :2] + 1.0) * scale
        for triangle in self.triangles:
            draw.polygon([tuple(v) for v in pixel_points[triangle]], fill=1)
        occupied = np.asarray(image, dtype=bool)
        shoulders = np.asarray([vertices[self.landmarks[a]["shoulder"]] for a in AGENTS])
        center_x = float(np.mean(shoulders[:, 0]))
        shoulder_y = float(np.mean(shoulders[:, 1]))
        lo = np.floor((np.array([center_x - 0.26, shoulder_y - 0.32]) + 1) * scale).astype(int)
        hi = np.ceil((np.array([center_x + 0.26, shoulder_y + 0.04]) + 1) * scale).astype(int)
        lo = np.clip(lo, 0, 511)
        hi = np.clip(hi, 0, 512)
        inside = int(occupied[lo[1] : hi[1], lo[0] : hi[0]].sum())
        area = float(occupied.sum()) / scale**2
        target_area = 0.52 * 0.36
        return {
            "projected_area_m2": area,
            "target_coverage": inside / scale**2 / target_area,
            "inside_target_fraction": inside / max(1, int(occupied.sum())),
            "span_x_m": float(np.ptp(vertices[:, 0])),
            "span_y_m": float(np.ptp(vertices[:, 1])),
        }

    def _cloth_speed(self):
        return float(np.max(np.linalg.norm(self.data.qvel[self._cloth_dofs], axis=1)))

    def metrics(self):
        p = self.vertices
        hem_errors = {
            a: float(
                np.linalg.norm(
                    p[self.landmarks[a]["hem"]][:2] - p[self.landmarks[a]["shoulder"]][:2]
                )
            )
            for a in AGENTS
        }
        sleeve_errors = {
            a: float(max(0.0, abs(p[self.landmarks[a]["sleeve"]][0]) - 0.24)) for a in AGENTS
        }
        finite = bool(np.isfinite(p).all())
        on_table = bool(
            np.all((p[:, 2] > -0.02) & (np.abs(p[:, 0]) < 0.96) & (np.abs(p[:, 1]) < 0.64))
        )
        speed = self._cloth_speed()
        footprint = self._footprint(p)
        # Predeclared coarse fold criterion: reject collapsed bundles even if
        # selected landmarks happen to coincide. No phase counters or rewards.
        geometry_ok = (
            footprint["projected_area_m2"] >= 0.075
            and footprint["target_coverage"] >= 0.40
            and footprint["inside_target_fraction"] >= 0.75
            and 0.30 <= footprint["span_x_m"] <= 0.65
            and 0.20 <= footprint["span_y_m"] <= 0.45
        )
        physics_valid = finite and self.peak_cloth_speed <= self.MAX_VALID_CLOTH_SPEED
        folded = (
            physics_valid
            and geometry_ok
            and not self.fixture_active
            and on_table
            and max(hem_errors.values()) < 0.095
            and max(sleeve_errors.values()) < 0.06
            and max(p[:, 2]) < 0.12
            and speed < 0.12
            and self._stable_ticks >= 15
            and all(v is None for v in self.holding.values())
        )
        return {
            "success": bool(folded),
            "physics_valid": bool(physics_valid),
            "peak_cloth_speed_m_s": self.peak_cloth_speed,
            "peak_edge_strain": self.peak_edge_strain,
            "footprint": footprint,
            "fold_geometry_ok": bool(geometry_ok),
            "hem_alignment_m": hem_errors,
            "sleeve_overhang_m": sleeve_errors,
            "max_height_m": float(p[:, 2].max()),
            "max_speed_m_s": speed,
            "stable_ticks": self._stable_ticks,
            "on_table": on_table,
            "finite": finite,
        }

    def snapshot(self):
        return {
            "tick": self.tick,
            "time": float(self.data.time),
            "physics_config": {**self.PHYSICS_CONFIG, "vacuum_fixture_enabled": self.fixture},
            "fixture_active": self.fixture_active,
            "vertices": self.vertices.tolist(),
            "triangles": self.triangles.tolist(),
            "robots": {
                a: {
                    "ee": self.ee(a).tolist(),
                    "holding": self.holding[a],
                    "joint_positions": self.data.qpos[self._joints[a]].tolist(),
                }
                for a in AGENTS
            },
            "obstacle": {
                "active": self._obstacle_active,
                "side": self._obstacle_side,
                "position": self.data.mocap_pos[self._obstacle_mocap].tolist(),
                "radius": 0.065,
                "height": 0.15,
            },
            "metrics": self.metrics(),
        }

    def render(
        self, camera: str = "overview", width: int = 960, height: int = 720, *, annotate=True
    ):
        """Render a camera. Arm cameras are the unmodified policy observation path;
        presentation cameras use the studio look in ``presentation.py`` and, when
        ``annotate`` is true, label the interfering object while it is present."""
        if self._closed:
            raise RuntimeError("world is closed")
        if camera in presentation.PRESENTATION_CAMERAS:
            return self._render_studio(camera, width, height, annotate)
        key = width, height
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def _render_studio(self, camera, width, height, annotate=True):
        scale = max(
            1,
            min(
                presentation.SUPERSAMPLE,
                self.model.vis.global_.offwidth // width,
                self.model.vis.global_.offheight // height,
            ),
        )
        key = "studio", width * scale, height * scale
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(
                self.model, height=height * scale, width=width * scale
            )
        renderer = self._renderers[key]
        with self._studio.applied():
            renderer.update_scene(
                self.data, camera=camera, scene_option=presentation.studio_scene_option()
            )
            presentation.configure_studio_scene(renderer.scene)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            try:
                depth = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()
        image = presentation.grade(rgb, depth, (width, height))
        if annotate and self._obstacle_active:
            image = self._label_obstacle(image, camera)
        return image

    def _label_obstacle(self, image, camera):
        h, w = image.shape[:2]
        cam = self.model.camera(camera).id
        position = self.data.mocap_pos[self._obstacle_mocap]
        top = position + [0, 0, 0.075]  # cylinder half-height
        anchor = presentation.project(
            top, self.data.cam_xpos[cam], self.data.cam_xmat[cam], self.model.cam_fovy[cam], w, h
        )
        # Fade in while the object slides from the table edge to its stop.
        opacity = float(np.clip((0.67 - abs(position[0])) / 0.13, 0, 1))
        side = self._obstacle_side
        return presentation.annotate(
            image,
            anchor,
            "Interfering object",
            f"detected only by the {side} arm",
            side=-1 if side == "left" else 1,
            opacity=opacity,
        )

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._closed = True

    @staticmethod
    def _check_agent(agent):
        if agent not in AGENTS:
            raise ValueError(f"unknown arm {agent}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
