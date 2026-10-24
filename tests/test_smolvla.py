"""Adapter invariants use a fake backend; these do not claim model/task success."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from relay_lab.smolvla import SmolVLABackend, SmolVLAPolicy
from relay_lab.types import MAX_DELTA, Instruction, Observation


class RecordingBackend:
    last_inference_seconds = 0.125
    seed = 7

    def __init__(self, chunks=None):
        self.calls = []
        self.noise_seeds = []
        self.chunks = chunks or [np.tile([0.2, 0, 0, 0, 0, 0, -1], (8, 1))]

    def predict(self, *args, noise_seed=None):
        self.calls.append(args)
        self.noise_seeds.append(noise_seed)
        return self.chunks[min(len(self.calls) - 1, len(self.chunks) - 1)]


def observation(agent_id="picker", **changes):
    return replace(
        Observation(
            tick=3,
            agent_id=agent_id,
            ee_position=(-0.35, 0.2, 0.1),
            holding=None,
            objects={"hidden": (50, 60, 70)},
            targets={"private": (1, 2, 3)},
            order=("gold", "coral", "mint"),
            rgb=np.zeros((16, 16, 3), dtype=np.uint8),
            wrist_rgb=np.full((16, 16, 3), 200, dtype=np.uint8),
            gripper_opening=0.08,
        ),
        **changes,
    )


PICK = Instruction("pick", "Pick the coral cube.", "coral")


def test_backend_receives_only_local_vision_proprioception_and_text():
    backend = RecordingBackend()
    policy = SmolVLAPolicy("picker", backend)
    original = observation()
    # No oracle fields even exist on this boundary object. Accessing them fails.
    local_only = SimpleNamespace(
        **{
            name: getattr(original, name)
            for name in (
                "agent_id",
                "ee_position",
                "ee_orientation",
                "gripper_opening",
                "rgb",
                "wrist_rgb",
            )
        }
    )
    policy.act(local_only, PICK)
    front, wrist, state, text = backend.calls[0]
    assert len(backend.calls[0]) == 4
    assert front is original.rgb and wrist is original.wrist_rgb
    np.testing.assert_allclose(state, [0, 0.2, 1, np.pi, 0, 0, 0.04, -0.04], atol=1e-7)
    assert text == PICK.text


def test_agents_have_independent_queues_and_reset():
    backend = RecordingBackend(
        [
            np.tile([0.2, 0, 0, 0, 0, 0, -1], (8, 1)),
            np.tile([-0.2, 0, 0, 0, 0, 0, 1], (8, 1)),
        ]
    )
    picker, packer = (SmolVLAPolicy(agent, backend) for agent in ("picker", "packer"))
    assert picker.act(observation(), PICK).delta[0] > 0
    assert packer.act(observation("packer"), PICK).delta[0] < 0
    assert picker.act(observation(), PICK).delta[0] > 0
    picker.reset()
    assert packer.act(observation("packer"), PICK).delta[0] < 0
    assert len(backend.calls) == 2
    picker.act(observation(), PICK)
    assert len(backend.calls) == 3


def test_agent_noise_is_independent_of_peer_calls_and_reset_repeats_sequence():
    isolated_backend, shared_backend = RecordingBackend(), RecordingBackend()
    isolated = SmolVLAPolicy("picker", isolated_backend, action_horizon=1)
    picker = SmolVLAPolicy("picker", shared_backend, action_horizon=1)
    packer = SmolVLAPolicy("packer", shared_backend, action_horizon=1)
    isolated_seeds, interleaved_seeds, peer_seeds = [], [], []
    for peer_calls in (0, 3, 1):
        isolated.act(observation(), PICK)
        isolated_seeds.append(isolated.last_prediction["noise_seed"])
        for _ in range(peer_calls):
            packer.act(observation("packer"), PICK)
            peer_seeds.append(packer.last_prediction["noise_seed"])
        picker.act(observation(), PICK)
        interleaved_seeds.append(picker.last_prediction["noise_seed"])
    assert isolated_seeds == interleaved_seeds
    assert len(set(isolated_seeds)) == 3
    assert set(isolated_seeds).isdisjoint(peer_seeds)
    picker.reset()
    picker.act(observation(), PICK)
    assert picker.last_prediction["noise_seed"] == isolated_seeds[0]


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "place"},
        {"object_id": "mint"},
        {"target_id": "relay"},
        {"text": "Another task."},
    ],
)
def test_any_instruction_change_discards_old_chunk(changes):
    backend = RecordingBackend()
    policy = SmolVLAPolicy("picker", backend)
    policy.act(observation(), PICK)
    policy.act(observation(), replace(PICK, **changes))
    assert len(backend.calls) == 2
    assert policy.last_prediction["chunk_index"] == 0


def test_replans_after_execution_horizon_with_latest_observation():
    backend = RecordingBackend()
    policy = SmolVLAPolicy("picker", backend, action_horizon=2)
    policy.act(observation(), PICK)
    policy.act(observation(), PICK)
    assert len(backend.calls) == 1
    policy.act(observation(ee_position=(-0.3, 0.2, 0.1)), PICK)
    assert len(backend.calls) == 2
    assert backend.calls[-1][2][0] == pytest.approx(0.05)


@pytest.mark.parametrize("grip,expected", [(1, -1), (-1, 1), (0, 0), (0.2, 0), (-0.2, 0)])
def test_osc_action_mapping_bounds_gripper_sign_and_audit(grip, expected):
    raw = np.array([[4, 2, -3, 0.2, -0.4, 0.7, grip]], dtype=float)
    backend = RecordingBackend([raw])
    policy = SmolVLAPolicy("picker", backend)
    action = policy.act(observation(), PICK)
    assert np.linalg.norm(action.delta) == pytest.approx(MAX_DELTA)
    np.testing.assert_allclose(action.delta, np.array([1, 1, -1]) * MAX_DELTA / np.sqrt(3))
    assert action.gripper == expected
    assert policy.last_prediction["raw_osc_action"] == raw[0].tolist()
    assert policy.last_prediction["discarded_rotation"] == [0.2, -0.4, 0.7]
    assert policy.last_prediction["clipped_osc_action"][:3] == [1, 1, -1]


def test_wait_is_an_actual_model_prediction():
    backend = RecordingBackend()
    policy = SmolVLAPolicy("picker", backend)
    action = policy.act(observation(), Instruction("wait", "Wait at the relay pad."))
    assert len(backend.calls) == 1
    assert action.delta[0] > 0  # No rule-based hold/home override.


@pytest.mark.parametrize(
    "changes",
    [
        {"rgb": None},
        {"wrist_rgb": None},
        {"rgb": np.zeros((8, 8, 3), dtype=float)},
    ],
)
def test_missing_or_invalid_local_images_fail(changes):
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="RGB"):
        SmolVLAPolicy("picker", backend).act(observation(**changes), PICK)
    assert not backend.calls


@pytest.mark.parametrize("chunk", [np.zeros((2, 6)), np.zeros((0, 7)), np.full((2, 7), np.nan)])
def test_invalid_model_output_fails_without_scripted_fallback(chunk):
    with pytest.raises(RuntimeError, match="no fallback"):
        SmolVLAPolicy("picker", RecordingBackend([chunk])).act(observation(), PICK)


def test_cross_agent_observation_rejected():
    with pytest.raises(ValueError, match="another agent"):
        SmolVLAPolicy("picker", RecordingBackend()).act(observation("packer"), PICK)


def test_feedback_reaches_next_inference_without_resetting_every_action_chunk():
    backend = RecordingBackend()
    policy = SmolVLAPolicy("picker", backend, action_horizon=2)
    policy.act(observation(), PICK)
    update = replace(PICK, feedback="Observed no parcel attached after closing the gripper.")
    policy.act(observation(), update)
    assert len(backend.calls) == 1
    assert (
        policy.last_prediction["model_prompt"] == PICK.text
    )  # Actual chunk input, not latest text.
    policy.act(observation(), update)
    assert len(backend.calls) == 2
    assert backend.calls[-1][-1] == update.model_prompt
    assert "no parcel attached" in policy.last_prediction["model_prompt"]


def test_critical_feedback_discards_stale_model_actions():
    backend = RecordingBackend()
    policy = SmolVLAPolicy("picker", backend, action_horizon=5)
    policy.act(observation(), PICK)
    correction = replace(PICK, feedback="Observed mint attached; coral was requested.", replan=True)
    policy.act(observation(), correction)
    assert len(backend.calls) == 2
    assert backend.calls[-1][-1] == correction.model_prompt
    assert policy.last_prediction["chunk_index"] == 0
    assert policy.last_prediction["feedback_replan"]


def guarded_backend(token_budget):
    """Only the lightweight prompt boundary exists; numerical work must not run."""
    backend = SmolVLABackend.__new__(SmolVLABackend)
    backend._config = SimpleNamespace(tokenizer_max_length=token_budget)
    # A deterministic tokenizer makes the appended newline a separate token,
    # exposing the exact-boundary case without loading transformer assets.
    backend._tokenizer = Mock(side_effect=lambda text: {"input_ids": list(text)})
    backend._torch = Mock()
    backend._torch.from_numpy.side_effect = AssertionError("numerical work started")
    backend._preprocess = Mock(side_effect=AssertionError("preprocessing started"))
    backend._model = Mock()
    backend._model.predict_action_chunk.side_effect = AssertionError("model inference started")
    backend.inference_calls = 0
    return backend


@pytest.mark.parametrize("prompt", ["pick", "pick\n"])
def test_backend_budget_counts_normalized_newline_before_numerical_work(prompt):
    backend = guarded_backend(token_budget=4)
    obs = observation()
    with pytest.raises(ValueError, match="exceeds the language budget"):
        backend.predict(obs.rgb, obs.wrist_rgb, np.zeros(8), prompt)
    assert backend.last_model_prompt == "pick\n"
    assert backend.last_prompt_tokens == 5
    backend._tokenizer.assert_called_once_with("pick\n")
    backend._torch.from_numpy.assert_not_called()
    backend._preprocess.assert_not_called()
    backend._model.predict_action_chunk.assert_not_called()
    assert backend.inference_calls == 0


def test_backend_rejects_oversized_feedback_without_truncating_or_running_model():
    backend = guarded_backend(token_budget=32)
    obs = observation()
    instruction = replace(PICK, feedback="Observed result. " * 50)
    with pytest.raises(ValueError, match="shorten feedback"):
        backend.predict(obs.rgb, obs.wrist_rgb, np.zeros(8), instruction.model_prompt)
    assert backend.last_model_prompt == instruction.model_prompt + "\n"
    assert backend.last_prompt_tokens > 32
    assert backend._tokenizer.call_args.args == (instruction.model_prompt + "\n",)
    backend._torch.from_numpy.assert_not_called()
    backend._preprocess.assert_not_called()
    backend._model.predict_action_chunk.assert_not_called()


def test_backend_accepts_a_prompt_exactly_at_the_normalized_token_budget():
    backend = guarded_backend(token_budget=5)
    obs = observation()
    # Reaching numerical work proves the exact-budget prompt was accepted.
    # Stop there deliberately; this is not a checkpoint/inference test.
    with pytest.raises(AssertionError, match="numerical work started"):
        backend.predict(obs.rgb, obs.wrist_rgb, np.zeros(8), "pick")
    assert backend.last_prompt_tokens == 5
    assert backend.last_model_prompt == "pick\n"
