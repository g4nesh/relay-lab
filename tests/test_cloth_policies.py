import copy

import numpy as np
import pytest

from relay_lab.cloth.policies import PHASES, ClothPolicy, observation_features
from relay_lab.cloth.pretrained import _bounded_target


def observation(agent="left"):
    sign = -1 if agent == "left" else 1
    return {
        "agent": agent,
        "ee_position": [sign * 0.38, -0.28, 0.28],
        "grasped": False,
        "landmarks": {
            "sleeve": [sign * 0.44, 0.16, 0.008],
            "hem": [sign * 0.24, -0.32, 0.008],
            "shoulder": [sign * 0.24, 0.28, 0.008],
        },
        "rgb": np.full((64, 64, 3), 120, dtype=np.uint8),
    }


@pytest.mark.parametrize("agent", ["left", "right"])
def test_learned_policy_respects_instruction_and_local_landmark(agent):
    policy = ClothPolicy(agent)
    obs = observation(agent)
    sleeve = policy.act(obs, "approach sleeve")["target"]
    hem = policy.act(obs, "approach hem")["target"]
    assert np.linalg.norm(np.asarray(sleeve) - np.asarray(hem)) > 0.35
    shifted = copy.deepcopy(obs)
    shifted["landmarks"]["hem"][0] += 0.06
    changed = policy.act(shifted, "approach_hem")["target"]
    assert changed[0] - hem[0] == pytest.approx(0.06, abs=0.002)


@pytest.mark.parametrize("agent", ["left", "right"])
def test_carry_tracks_locally_observed_shoulder_after_garment_moves(agent):
    policy = ClothPolicy(agent)
    obs = observation(agent)
    first = policy.act(obs, "carry_hem")["target"]
    obs["landmarks"]["shoulder"][1] += 0.17
    second = policy.act(obs, "carry_hem")["target"]
    assert second[1] - first[1] == pytest.approx(0.17, abs=0.002)


@pytest.mark.parametrize("agent", ["left", "right"])
def test_missing_local_landmark_does_not_become_zero_world_coordinate(agent):
    policy = ClothPolicy(agent)
    obs = observation(agent)
    obs["landmarks"].pop("hem")
    action = policy.act(obs, "approach_hem")
    assert action["target"] == pytest.approx(obs["ee_position"])
    assert policy.last_prediction["coordinator_hold"]
    assert len(policy.last_prediction["raw"]) == 4


def test_policy_checkpoints_are_distinct_and_provenance_honest():
    left, right = ClothPolicy("left"), ClothPolicy("right")
    assert left.checkpoint_sha256 != right.checkpoint_sha256
    assert left.model_info["architecture"] != right.model_info["architecture"]
    assert not left.model_info["foundation_vla"]
    assert not right.model_info["foundation_vla"]
    assert not left.model_info["training"]["external_demonstrations"]


@pytest.mark.parametrize("agent", ["left", "right"])
def test_hold_is_logged_interlock_after_real_model_inference(agent):
    policy = ClothPolicy(agent)
    obs = observation(agent)
    action = policy.act(obs, "lift_hem", hold=True)
    assert action["target"] == pytest.approx(obs["ee_position"])
    assert policy.calls == 1
    assert policy.last_prediction["coordinator_hold"]
    assert np.linalg.norm(np.asarray(policy.last_prediction["raw"][:3]) - obs["ee_position"]) > 0.1


def test_policy_does_not_read_peer_or_oracle_fields():
    policy = ClothPolicy("left")
    obs = observation()
    first = policy.act(obs, "carry_hem")
    obs.update({"peer": object(), "oracle": object(), "all_vertices": object()})
    assert policy.act(obs, "carry_hem") == first


def test_disappearing_fold_target_holds_instead_of_using_hidden_shoulder():
    policy = ClothPolicy("left")
    obs = observation()
    policy.act(obs, "settle")
    obs["landmarks"].pop("shoulder")
    action = policy.act(obs, "carry_hem")
    assert action["target"] == pytest.approx(obs["ee_position"])
    assert policy.last_prediction["coordinator_hold"]


def test_rejects_foreign_observation_and_bad_inputs():
    policy = ClothPolicy("left")
    with pytest.raises(ValueError, match="another arm"):
        policy.act(observation("right"), "settle")
    with pytest.raises(ValueError, match="unsupported"):
        policy.act(observation(), "teleport folded cloth")
    with pytest.raises(ValueError, match="clearance"):
        observation_features(observation(), np.nan)


@pytest.mark.parametrize("agent", ["left", "right"])
def test_all_known_phase_actions_are_finite(agent):
    policy = ClothPolicy(agent)
    for phase in PHASES:
        action = policy.act(observation(agent), phase)
        assert np.isfinite(action["target"]).all()
        assert isinstance(action["grip"], bool)


def test_pretrained_transfer_action_is_bounded_without_replacing_direction():
    ee = np.asarray([0.0, 0.0, 0.2])
    target = np.asarray(_bounded_target(ee, [5, 0, 0.2]))
    assert np.linalg.norm(target - ee) <= 0.018 + 1e-10
    assert target[0] > 0
    assert target[1] == 0


def test_pretrained_transfer_rejects_nonfinite_output():
    with pytest.raises(ValueError, match="invalid"):
        _bounded_target([0, 0, 0.2], [np.nan, 0, 0.2])


@pytest.mark.parametrize("agent", ["left", "right"])
def test_frozen_policy_matches_reachable_world_arc_without_runtime_teacher(agent):
    from relay_lab.cloth.world import ClothWorld

    with ClothWorld(seed=21007, disturbances=False) as world:
        obs = world.observe(agent)
        obs["rgb"] = observation(agent)["rgb"]
        policy = ClothPolicy(agent)
        for phase in ("lift_hem", "carry_hem", "fold_hem", "release_hem"):
            prediction = policy.act(obs, phase, clearance=0.23)
            reference = world.reference_target(agent, phase, height=0.23)
            assert prediction["target"] == pytest.approx(reference["target"], abs=0.002)
            assert prediction["grip"] == reference["grip"]
