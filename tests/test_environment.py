"""Check actual simulator boundaries, not just the scripted happy path."""

import numpy as np
import pytest

from relay_lab.environment import RelayEnv
from relay_lab.types import MAX_DELTA, Action


def test_seed_reset_and_private_views():
    with RelayEnv(seed=7) as env:
        initial = env.state()
        picker, packer = env.observe("picker"), env.observe("packer")
        assert picker.order is None and picker.packed == ()
        assert set(picker.objects) == {"coral", "mint", "gold"}
        assert packer.order == env.order and packer.objects == {}
        assert "slot_0" not in picker.targets
        assert "slot_0" in packer.targets
        env.step({"picker": Action((0.02, 0, 0))})
        env.reset(7)
        assert env.state() == initial
        # Returned observations are copies, so policy-side edits cannot alter the world.
        picker.targets["relay"] = (999, 999, 999)
        assert env.observe("picker").targets["relay"][0] == 0


def test_action_validation_is_atomic_and_motion_is_bounded():
    with RelayEnv() as env:
        before = env.state()
        with pytest.raises(ValueError, match="finite"):
            env.step({"picker": Action((0.02, 0, 0)), "packer": Action((np.nan, 0, 0))})
        assert env.state() == before
        origin = np.array(env.observe("picker").ee_position)
        env.step({"picker": Action((100, 100, 100))})
        after = np.array(env.observe("picker").ee_position)
        assert np.linalg.norm(after - origin) <= MAX_DELTA + 1e-9
        for _ in range(35):
            env.step({"picker": Action((1, 0, 0))})
        assert env.observe("picker").ee_position[0] <= 0.06


def test_grasp_requires_proximity_and_cannot_steal():
    with RelayEnv() as env:
        result = env.step({"picker": Action(gripper=-1)})
        assert env.observe("picker").holding is None
        assert any(event["type"] == "grasp_missed" for event in result["events"])
        # Move through the public action API to a source cube.
        target = np.array(env.observe("picker").objects["coral"])
        target[2] = 0.035
        for _ in range(25):
            current = np.array(env.observe("picker").ee_position)
            env.step({"picker": Action(tuple(target - current))})
        env.step({"picker": Action(gripper=-1), "packer": Action(gripper=-1)})
        assert env.observe("picker").holding == "coral"
        assert env.observe("packer").holding is None
        assert "coral" not in env.observe("packer").objects
        assert not env.success
