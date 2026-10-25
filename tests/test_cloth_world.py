"""Physics and information boundaries for the original textile workcell."""

import numpy as np
import pytest

from relay_lab.cloth.scene import shirt_mesh
from relay_lab.cloth.world import ClothWorld


def test_t_shirt_is_connected_triangle_flex_with_neckline():
    vertices, triangles, landmarks = shirt_mesh()
    assert vertices.shape == (271, 3)
    assert triangles.shape == (460, 3)
    assert np.max(np.abs(vertices[:, 0])) == pytest.approx(0.44)
    assert not np.any((vertices[:, 1] > 0.28) & (np.abs(vertices[:, 0]) < 0.08))
    used = set(triangles.ravel())
    assert used == set(range(len(vertices)))
    for agent in ("left", "right"):
        assert set(landmarks[agent]) == {"hem", "sleeve", "shoulder"}


def test_real_flex_settles_on_table_and_arms_have_actuated_joints():
    with ClothWorld(seed=7, disturbances=False) as world:
        assert world.model.nflex == 1
        assert world.model.flex_dim[0] == 2
        assert world.model.nu == 12
        assert world.model.nv > 800
        assert world.vertices[:, 2].min() > -0.002
        assert world.vertices[:, 2].max() < 0.01
        assert world.metrics()["on_table"]
        assert not world.metrics()["success"]
        assert world.data.ncon > 0
        assert world.model.flex_selfcollide[0] != 0
        for agent in ("left", "right"):
            np.testing.assert_allclose(world.ee(agent), world.HOME[agent], atol=0.002)


def test_seeded_initial_conditions_and_private_obstacle_are_deterministic():
    with ClothWorld(seed=11, obstacle_schedule=((0, 4, "left"),)) as world:
        before = world.vertices.copy()
        world.step()
        left, right = world.observe("left"), world.observe("right")
        assert left["obstacle"]["active"]
        assert right["obstacle"] is None
        assert "vertices" not in left and "robots" not in left
        assert all(xyz[0] <= 0.05 for xyz in left["landmarks"].values())
        assert all(xyz[0] >= -0.05 for xyz in right["landmarks"].values())
        world.reset(11)
        np.testing.assert_allclose(world.vertices, before, atol=1e-9)
        world.reset(12)
        assert not np.allclose(world.vertices, before)


def test_invalid_actions_fail_without_advancing_physics():
    with ClothWorld(disturbances=False) as world:
        with pytest.raises(ValueError, match="finite"):
            world.step({"left": {"target": [float("nan"), 0, 0]}})
        with pytest.raises(ValueError, match="unknown arm"):
            world.observe("third")
        assert world.tick == 0
        assert world.holding == {"left": None, "right": None}
        world.step({"left": {"grip": True}})
        assert world.holding["left"] is None  # No remote/teleporting grasps.


def test_point_clamp_lifts_actual_cloth_and_release_removes_constraint():
    with ClothWorld(seed=3, disturbances=False) as world:
        target = world.vertices[world.landmarks["left"]["hem"]] + [0, 0, 0.008]
        for _ in range(35):
            world.step({"left": {"target": target, "grip": False}})
        world.step({"left": {"target": target, "grip": True}})
        held = world.holding["left"]
        assert held == world.landmarks["left"]["hem"]
        assert world.observe("left")["grasp_target"] == "hem"
        z_before = world.vertices[held, 2]
        lift = target + [0, 0, 0.16]
        for _ in range(25):
            world.step({"left": {"target": lift, "grip": True}})
        assert world.vertices[held, 2] > z_before + 0.10
        assert np.linalg.norm(world.vertices[held] - world.ee("left")) < 0.025
        world.step({"left": {"grip": False}})
        assert world.holding["left"] is None
        assert world.observe("left")["grasp_target"] == "none"
        assert not world.data.eq_active[world._eq["left"]]
        assert not world.metrics()["success"]


def test_collapsed_bundle_cannot_pass_fold_quality():
    with ClothWorld(disturbances=False) as world:
        world.data.flexvert_xpos[:] = [0.0, 0.15, 0.015]
        world.data.qvel[:] = 0
        metrics = world.metrics()
        assert max(metrics["hem_alignment_m"].values()) == 0
        assert not metrics["success"]
        assert not metrics["fold_geometry_ok"]
        assert metrics["footprint"]["projected_area_m2"] < 0.001


def test_cloth_obstacle_contact_is_real_and_collision_masks_are_disclosed():
    import mujoco

    with ClothWorld(disturbances=False) as world:
        world.data.mocap_pos[world._obstacle_mocap] = [-0.24, -0.28, 0.03]
        mujoco.mj_forward(world.model, world.data)
        obstacle = world.model.geom("obstacle_geom").id
        contacts = [c for c in world.data.contact if obstacle in c.geom and 0 in c.flex]
        assert contacts  # True flex/rigid collision, not a drawn obstacle.
        config = world.snapshot()["physics_config"]
        assert config["cloth_obstacle_collision"]
        assert not config["cloth_arm_shell_collision"]
        assert not config["arm_arm_collision"]
        assert not world.metrics()["success"]


def test_vacuum_fixture_is_explicit_and_must_release_before_scoring():
    with ClothWorld(disturbances=False, fixture=True) as world:
        assert world.fixture_active
        assert all(world.data.eq_active[e] for e in world._fixture_eq)
        assert not world.metrics()["success"]
        world.release_fixture()
        assert not world.fixture_active
        assert all(not world.data.eq_active[e] for e in world._fixture_eq)
        assert world.events[-1]["event"] == "vacuum_fixture_released"
        assert not world.metrics()["success"]
