"""Information isolation and label alignment in the original adapter diagnostic."""

import numpy as np
import pytest

from relay_lab.cloth.latent_dataset import (
    DEFAULT_EPISODES,
    array_hash,
    capture_counterfactuals,
    frozen_obstacle,
    validate_episodes,
)
from relay_lab.cloth.world import ClothWorld


def _render_hazard(world, camera, width, height):
    # A deliberately global renderer makes hidden-geometry leakage observable.
    # The private-camera boundary must actively remove that geometry.
    present = abs(world.data.mocap_pos[world._obstacle_mocap, 0]) < 1
    return np.full((height, width, 3), 240 if present else 12, dtype=np.uint8)


def test_paired_counterfactuals_share_physics_and_require_peer_information(monkeypatch):
    monkeypatch.setattr(ClothWorld, "render", _render_hazard)
    with ClothWorld(seed=10010, disturbances=False) as world:
        before = (world.data.qpos.copy(), world.data.qvel.copy(), world.vertices.copy())
        rows, rgb = capture_counterfactuals(
            world, split="train", seed=10010, phase="approach_sleeve_above", snapshot_index=0
        )
        assert len(rows) == len(rgb) == 6
        assert len({r["pair_id"] for r in rows}) == 3
        assert len({r["physical_state_sha256"] for r in rows}) == 1
        for agent in ("left", "right"):
            by_case = {r["counterfactual"]: r for r in rows if r["agent"] == agent}
            clear = by_case["clear"]
            own = by_case[f"{agent}_obstacle"]
            peer = by_case[f"{'right' if agent == 'left' else 'left'}_obstacle"]
            assert clear["observation"] == peer["observation"]
            assert clear["rgb_sha256"] == peer["rgb_sha256"]
            assert clear["rgb_sha256"] != own["rgb_sha256"]
            assert own["observation"]["obstacle"]["active"]
            assert "peer_obstacle" not in peer["observation"]
            assert not clear["labels"]["coordinated_hold"]
            assert peer["labels"]["peer_obstacle"]
            assert clear["action"]["target"] != peer["action"]["target"]
            for held in (own, peer):
                assert held["action"] == {
                    "target": held["observation"]["ee_position"],
                    "grip": held["observation"]["grasped"],
                }
            assert clear["rgb_sha256"] == array_hash(rgb[clear["rgb_index"]])
        assert world.tick == 0
        assert not world._obstacle_active and world._obstacle_side is None
        for old, new in zip(
            before, (world.data.qpos, world.data.qvel, world.vertices), strict=True
        ):
            np.testing.assert_array_equal(old, new)
        repeated_rows, repeated_rgb = capture_counterfactuals(
            world, split="train", seed=10010, phase="approach_sleeve_above", snapshot_index=0
        )
        assert rows == repeated_rows
        np.testing.assert_array_equal(rgb, repeated_rgb)


def test_counterfactual_restores_geometry_on_failure():
    with ClothWorld(seed=10010, disturbances=False) as world:
        mocap = world.data.mocap_pos.copy()
        with pytest.raises(RuntimeError, match="render failed"), frozen_obstacle(world, "left"):
            assert world.observe("left")["obstacle"]["active"]
            raise RuntimeError("render failed")
        np.testing.assert_array_equal(mocap, world.data.mocap_pos)
        assert world.observe("left")["obstacle"] is None


def test_episode_splits_are_disjoint_and_reject_cross_split_seed_reuse():
    assert validate_episodes(DEFAULT_EPISODES) == DEFAULT_EPISODES
    for bad in (
        (("train", 104),),
        (("test", 10010),),
        (("dev", 104),),
        (("train", 10010), ("train", 10010)),
        (("unknown", 10010),),
    ):
        with pytest.raises(ValueError):
            validate_episodes(bad)
