"""Inference-boundary and learned-modality checks for the original factory VLA."""

import numpy as np
import pytest

from relay_lab.factory.vision import IMAGE_SHAPE, instruction_for
from relay_lab.factory.vla import (
    DEFAULT_CHECKPOINT,
    TARGETS,
    FactoryVLABackend,
    language_features,
)


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("torch")
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("Train or download the original factory checkpoint first")
    return FactoryVLABackend()


def observation(robot_id="humanoid-01", kind="humanoid", instruction="wait", xyz=(-7.0, 2.0, 1.5)):
    return {
        "robot_id": robot_id,
        "kind": kind,
        "instruction": instruction,
        "phase": "idle",
        "tick": 5,
        "dt": 0.1,
        "command_id": "test-command",
        "proprio": {
            "position": [xyz[0], xyz[1], 0.0],
            "ee_position": list(xyz),
            "yaw": 0.0,
            "velocity": [0.0, 0.0, 0.0],
            "holding": None,
        },
    }


def test_language_disambiguates_same_cargo_and_different_destination(backend):
    torch = backend.torch
    with torch.inference_mode():
        first = (
            backend.network.language(
                torch.from_numpy(language_features("place part-001 on amr-01"))[None]
            )
            .argmax(-1)
            .item()
        )
        second = (
            backend.network.language(
                torch.from_numpy(language_features("place part-001 on pallet-003"))[None]
            )
            .argmax(-1)
            .item()
        )
    assert TARGETS[first] == "amr-01"
    assert TARGETS[second] == "pallet-003"


def test_embedding_is_learned_shared_unit_vector_and_pose_sensitive(backend):
    a = backend.embed(observation())
    b = backend.embed(observation(xyz=(-6.7, 2.0, 1.5)))
    assert len(a["embedding"]) == 32
    assert abs(np.linalg.norm(a["embedding"]) - 1) < 1e-5
    assert a["space_id"] == "factory-vla-v1"
    assert len(a["checkpoint_sha256"]) == 64
    assert not np.allclose(a["embedding"], b["embedding"])
    assert a["decoded"]["pose"][0] < b["decoded"]["pose"][0]
    assert 0 <= a["decoded"]["confidence"] <= 1


def test_blank_image_ablation_fails_closed(backend):
    backend.reset()
    obs = observation(instruction="pick part-001")
    obs["rgb"] = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not visible"):
        backend.act(obs)


def test_missing_language_or_visual_modality_is_rejected(backend):
    backend.reset()
    obs = observation(instruction="pick part-001")
    with pytest.raises(ValueError, match="requires RGB"):
        backend.act(obs)
    obs["instruction"] = ""
    with pytest.raises(ValueError, match="requires an instruction"):
        backend.act(obs)


def test_peer_embedding_changes_decoder_driven_motor_goal(backend):
    """The latent channel must affect inference, rather than merely be logged."""
    torch = backend.torch
    obs = observation(instruction="place part-001 on amr-01")
    words = torch.from_numpy(language_features(obs["instruction"]))[None]
    with torch.inference_mode():
        kinds = backend.network.kind(words).softmax(-1)
    backend.cache[obs["robot_id"]] = {
        "key": (obs["instruction"], obs["command_id"]),
        "goal": torch.tensor([[-6.0, 2.0, 0.62]]),
        "kind": kinds,
        "target": "amr-01",
        "routing": [],
        "mass": [10, 10],
        "tick": 0,
    }
    unshared = backend.act(obs)
    peer = backend.embed(observation("amr-01", "amr", xyz=(-5.8, 2.0, 0.62)))
    obs["peer_embedding"] = peer
    shared = backend.act(obs)
    assert shared["provenance"]["peer_fusion"]
    assert not np.allclose(
        shared["provenance"]["goal_estimate"], unshared["provenance"]["goal_estimate"]
    )
    assert not np.allclose(shared["delta"], unshared["delta"])
    peer["checkpoint_sha256"] = "0" * 64
    incompatible = backend.act(obs)
    assert not incompatible["provenance"]["peer_fusion"]
    assert np.allclose(incompatible["delta"], unshared["delta"])


def test_corrupt_proprioception_is_rejected(backend):
    obs = observation()
    obs["proprio"]["ee_position"][0] = float("nan")
    with pytest.raises(ValueError, match="finite local proprioception"):
        backend.act(obs)


def test_slot_instructions_name_target_without_numeric_coordinates():
    text = instruction_for({}, {"cargo_id": "part-006", "target_id": "pallet"}, "approach_place")
    assert text == "place part-006 on pallet-006"


def test_zeroed_motor_weights_remove_the_actual_motor_command(backend):
    torch = backend.torch
    clone = FactoryVLABackend()
    obs = observation(instruction="pick part-001")
    words = torch.from_numpy(language_features(obs["instruction"]))[None]
    with torch.inference_mode():
        kinds = clone.network.kind(words).softmax(-1)
    clone.cache[obs["robot_id"]] = {
        "key": (obs["instruction"], obs["command_id"]),
        "goal": torch.tensor([[-7.7, 1.3, 1.02]]),
        "kind": kinds,
        "target": "part-001",
        "routing": [],
        "mass": [10, 10],
        "tick": 0,
    }
    learned = clone.act(obs)
    with torch.no_grad():
        for parameter in clone.network.control.parameters():
            parameter.zero_()
    ablated = clone.act(obs)
    assert np.linalg.norm(learned["delta"]) > 0.001
    assert np.allclose(ablated["delta"], [0, 0, 0])


def test_image_translation_changes_the_learned_visual_goal(backend):
    from pathlib import Path

    data_path = Path("datasets/factory-vla/original.npz")
    if not data_path.is_file():
        pytest.skip("Original RGB verification scenes are optional")
    with np.load(data_path) as data:
        rgb = data["images"][0].copy()
    shifted = np.concatenate(
        [np.roll(rgb[:, :256], 12, axis=1), np.roll(rgb[:, 256:], 12, axis=1)], axis=1
    )
    torch = backend.torch
    with torch.inference_mode():
        images = torch.from_numpy(np.stack([rgb, shifted])).permute(0, 3, 1, 2).float() / 255
        poses, mass, _ = backend.network.locate(images, torch.tensor([0, 0]))
    visible = mass.min(-1).values.min(0).values.numpy() > 1
    assert visible.sum() >= 5
    displacement = (poses[1, :, 0] - poses[0, :, 0]).numpy()[visible]
    assert np.median(displacement) > 0.20


def test_checkpoint_cannot_silently_use_changed_sensor_geometry(tmp_path, backend):
    payload = backend.torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    payload["sensor_signature"] = "0" * 64
    incompatible = tmp_path / "incompatible.pt"
    backend.torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="sensor geometry do not match"):
        FactoryVLABackend(incompatible)
