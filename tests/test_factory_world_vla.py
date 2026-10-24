"""Learned execution boundary tests; stubs prove routing, not model competence."""

from copy import deepcopy
from typing import ClassVar

import numpy as np
import pytest

from relay_lab.factory.world import FactoryWorld


class Motor:
    model_info: ClassVar[dict] = {"model_id": "test-neural-boundary", "checkpoint_sha256": "a" * 64}

    def __init__(self, delta=(0, 0, 0), yaw_delta=0, gripper=0):
        self.delta, self.yaw_delta, self.gripper = delta, yaw_delta, gripper
        self.observations = []
        self.closed = False

    def act(self, observation):
        self.observations.append(deepcopy(observation))
        return {
            "delta": list(self.delta),
            "yaw_delta": self.yaw_delta,
            "gripper": self.gripper,
            "embedding": [1.0] + [0.0] * 31,
            "provenance": deepcopy(self.model_info),
        }

    def embed(self, observation):
        self.observations.append(deepcopy(observation))
        return {
            "embedding": [1.0] + [0.0] * 31,
            "checkpoint_sha256": "a" * 64,
            "latent_space": "factory-vla-v1",
            "decoded": {"text": "test observation", "confidence": 1.0, "version": "test"},
        }

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def camera_stub(monkeypatch):
    """Keep safety/control tests independent of graphics drivers and trained weights."""
    from relay_lab.factory import vision

    def observe(snapshot, robot_id, command=None, phase=None, rgb=None, peer_embedding=None):
        robot = next(robot for robot in snapshot["robots"] if robot["id"] == robot_id)
        return {
            "robot_id": robot_id,
            "kind": robot["kind"],
            "rgb": np.zeros((16, 32, 3), dtype=np.uint8) if rgb is None else rgb,
            "instruction": vision.instruction_for(robot, command or {}, phase),
            "phase": phase,
            "proprio": {
                key: robot[key] for key in ("position", "ee_position", "yaw", "velocity", "holding")
            },
            "tick": snapshot["tick"],
            "dt": snapshot["dt"],
            "peer_embedding": peer_embedding,
        }

    monkeypatch.setattr(vision, "make_observation", observe)
    monkeypatch.setattr(vision, "close_renderers", lambda: None)


def command(robot="amr-01", kind="navigate", target="load", cargo=None, ident="test"):
    return {
        "command_id": ident,
        "robot_id": robot,
        "kind": kind,
        "target_id": target,
        "cargo_id": cargo,
        "job_id": "job-001" if cargo else None,
    }


def test_zero_network_actions_never_run_reference_motion_or_waypoints(monkeypatch):
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    before = world.robots["amr-01"].position.copy()

    def forbidden(*args):
        raise AssertionError("Reference motor path was invoked")

    monkeypatch.setattr(world, "_move", forbidden)
    monkeypatch.setattr(world, "_lifted_path", forbidden)
    for _ in range(30):
        result = world.step([command()])
    assert np.array_equal(world.robots["amr-01"].position, before)
    assert world.robots["amr-01"].waypoints == []
    assert len(motor.observations) == 30
    assert result["outcomes"][0]["status"] == "running"
    assert result["snapshot"]["physics"]["learned"] is True
    assert result["motor_actions"][0]["applied_action"]["delta"] == [0, 0, 0]


def test_actor_input_has_rgb_language_own_proprio_and_no_scene_coordinates():
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    world.step([command()])
    observation = motor.observations[-1]
    assert set(observation) == {
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
        "peer_record_ids",
        "rgb_tick",
    }
    assert set(observation["proprio"]) == {
        "position",
        "ee_position",
        "yaw",
        "velocity",
        "holding",
        "gripper_opening",
    }
    assert observation["instruction"] == "navigate to load"
    assert observation["command_id"] == "test"
    assert observation["rgb"].dtype == np.uint8
    assert world.sensor_png("amr-01").startswith(b"\x89PNG")


def test_neural_delta_is_applied_with_recorded_actuator_limits():
    motor = Motor(delta=(0, 10, 3), yaw_delta=2)
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    before = world.robots["amr-01"].position.copy()
    result = world.step([command()])
    robot = world.robots["amr-01"]
    displacement = robot.position - before
    assert displacement[1] > 0
    assert displacement[2] == 0
    assert np.linalg.norm(displacement) / world.dt <= 0.65 * world.dt + 1e-8
    trace = result["motor_actions"][0]
    assert trace["raw_action"]["delta"] == [0, 10, 3]
    assert set(trace["safety_edits"]) >= {
        "planar_drive",
        "speed_limit",
        "acceleration_limit",
        "turn_rate_limit",
    }
    assert trace["provenance"]["checkpoint_sha256"] == "a" * 64
    assert trace["applied_action"]["yaw_delta"] == pytest.approx(0.12)


def test_grasp_and_release_require_predicted_gripper_even_at_contact(monkeypatch):
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    carrier = world.robots["amr-01"]
    carrier.position[:] = [-6, 2, 0]
    carrier.ee[:] = [-6, 2, 0.62]
    hand = world.robots["humanoid-01"]
    hand.ee[:] = world._cargo_position("part-001")

    def forbidden(*args):
        raise AssertionError("Learned manipulation invoked scripted trajectory")

    monkeypatch.setattr(world, "_lifted_path", forbidden)
    task = command("humanoid-01", "transfer", "amr-01", "part-001")
    world.step([task])
    assert world.cargo["part-001"]["owner"] is None
    motor.gripper = -1
    world.step([task])
    assert world.cargo["part-001"]["owner"] == "humanoid-01"
    assert hand.phase == "approach_place"
    assert not hand.waypoints
    hand.ee[:] = carrier.ee
    hand.velocity[:] = 0
    world.step([task])
    assert world.cargo["part-001"]["owner"] == "humanoid-01"
    motor.gripper = 1
    result = world.step([task])
    assert world.cargo["part-001"]["owner"] == "amr-01"
    assert result["outcomes"][0]["status"] == "completed"
    assert world.cargo["part-001"]["stage"] == "loaded"


def test_matching_peer_record_reaches_network_unchanged_and_expired_records_do_not():
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    peer = {
        "robot_id": "amr-02",
        "embedding_id": "amr-02-0",
        "tick": 0,
        "expires_tick": 15,
        "space_id": "factory-vla-v1",
        "checkpoint_sha256": "a" * 64,
        "vector": [1.0] + [0.0] * 31,
    }
    world.set_peer_embeddings({"amr-01": [peer]})
    result = world.step([command()])
    assert motor.observations[-1]["peer_embedding"] == peer
    assert result["motor_actions"][0]["peer_record_ids"] == ["amr-02-0"]
    expired = {**peer, "expires_tick": world.tick}
    world.set_peer_embeddings({"amr-01": [expired]})
    world.step([command()])
    assert motor.observations[-1]["peer_embedding"] is None
    assert motor.observations[-1]["peer_record_ids"] == []


def test_embedding_records_are_fresh_poststep_for_all_five_robots():
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    world.step([command()])
    records = world.embedding_observations()
    assert {record["robot_id"] for record in records} == set(world.robots)
    assert all(record["tick"] == 1 for record in records)
    assert all(len(record["vector"]) == 32 for record in records)
    active = next(record for record in records if record["robot_id"] == "amr-01")
    assert active["command_id"] == "test"
    assert all(record["command_id"] is None for record in records if record["robot_id"] != "amr-01")
    assert all(observation["tick"] == 1 for observation in motor.observations[-5:])


def test_encoder_identity_is_explicit_and_never_relabelled(monkeypatch):
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    encoded = motor.embed(world._motor_observation(world.robots["amr-01"]))
    encoded["latent_space"] = "another-model-space"
    monkeypatch.setattr(motor, "embed", lambda observation: deepcopy(encoded))
    assert all(
        record["space_id"] == "another-model-space" for record in world.embedding_observations()
    )
    encoded["space_id"] = "conflicting-space"
    with pytest.raises(ValueError, match="conflicting latent spaces"):
        world.embedding_observations()
    del encoded["space_id"]
    del encoded["latent_space"]
    with pytest.raises(ValueError, match="identify its latent space"):
        world.embedding_observations()


def test_sensor_png_is_encoded_only_on_camera_refresh(monkeypatch):
    from PIL import Image

    saves = []
    save = Image.Image.save

    def recording_save(image, *args, **kwargs):
        saves.append(True)
        return save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", recording_save)
    world = FactoryWorld(batch_size=1, motor_policy=Motor())
    for _ in range(5):
        world.step([command()])
    assert len(saves) == 1
    world.step([command()])
    assert len(saves) == 2


def test_decoder_metadata_is_not_fabricated_for_plain_text(monkeypatch):
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    encoded = motor.embed(world._motor_observation(world.robots["amr-01"]))
    encoded["decoded"] = "unscored prediction"
    monkeypatch.setattr(motor, "embed", lambda observation: encoded)
    with pytest.raises(ValueError, match="report text, confidence and version"):
        world.embedding_observations()


def test_invalid_network_output_is_fail_closed_without_reference_fallback():
    world = FactoryWorld(batch_size=1, motor_policy=Motor(delta=(float("nan"), 0, 0)))
    before = world.robots["amr-01"].position.copy()
    with pytest.raises(ValueError, match="invalid motor action"):
        world.step([command()])
    assert np.array_equal(world.robots["amr-01"].position, before)
    assert world.tick == 0


def test_seeded_cargo_jitter_changes_camera_scene_without_changing_targets():
    first = FactoryWorld(seed=2, batch_size=6, pose_jitter=0.02)
    again = FactoryWorld(seed=2, batch_size=6, pose_jitter=0.02)
    other = FactoryWorld(seed=3, batch_size=6, pose_jitter=0.02)
    assert first.snapshot() == again.snapshot()
    assert first.snapshot()["cargo"] != other.snapshot()["cargo"]
    assert first.scene()["stations"] == other.scene()["stations"]


def test_camera_and_actor_resources_close_with_world():
    motor = Motor()
    world = FactoryWorld(batch_size=1, motor_policy=motor)
    world.close()
    assert motor.closed
