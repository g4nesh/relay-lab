"""Runtime boundaries for the learned policy; task success needs live evaluation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from relay_lab.relay_vla import (
    WORDS,
    RelayVLABackend,
    RelayVLAPolicy,
    language_features,
    make_network,
)
from relay_lab.types import MAX_DELTA, Instruction


@pytest.fixture
def checkpoint(tmp_path):
    network = make_network()
    with torch.no_grad():
        for p in network.parameters():
            p.zero_()
        network.control[-1].bias[:] = torch.tensor([2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    path = tmp_path / "model.pt"
    torch.save(
        {
            "format": "relay-vla-v1",
            "words": list(WORDS),
            "state_dict": network.state_dict(),
            "training": {"seed": 1},
        },
        path,
    )
    return path


def observation():
    # No objects, targets, holding, order, or environment exists at this boundary.
    return SimpleNamespace(
        agent_id="picker", rgb=np.zeros((32, 32, 3), np.uint8), ee_position=(-0.54, 0, 0.3)
    )


def test_language_input_distinguishes_object_and_slot():
    assert not np.array_equal(language_features("Pick coral"), language_features("Pick mint"))
    assert not np.array_equal(
        language_features("Place gold in slot 1"), language_features("Place gold in slot 2")
    )
    assert np.array_equal(language_features("Pick coral."), language_features("PICK coral!"))
    with pytest.raises(ValueError):
        language_features("")


def test_inference_is_neural_bounded_and_has_no_oracle_input(checkpoint):
    backend = RelayVLABackend(checkpoint, device="auto")
    policy = RelayVLAPolicy("picker", backend)
    action = policy.act(observation(), Instruction("pick", "Pick coral.", "untrusted-metadata"))
    assert action.delta == pytest.approx((MAX_DELTA, 0, 0))
    assert action.gripper == -1
    assert backend.model_info["policy"] == "relay-vla"
    assert backend.inference_calls == 1
    assert policy.last_prediction["model_prompt"] == "Pick coral."


def test_wait_does_not_substitute_a_scripted_action(checkpoint):
    policy = RelayVLAPolicy("picker", RelayVLABackend(checkpoint))
    action = policy.act(observation(), Instruction("wait", "Wait at home."))
    assert action.delta[0] > 0
    assert action.gripper == -1  # Exactly the neural head output.


def test_feedback_text_does_not_contaminate_language_and_replan_refreshes_goal(checkpoint):
    backend = RelayVLABackend(checkpoint)
    policy = RelayVLAPolicy("picker", backend)
    original = backend.network.goal
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    backend.network.goal = counted
    instruction = Instruction("pick", "Pick coral.")
    policy.act(observation(), instruction)
    policy.act(observation(), replace(instruction, feedback="Previously gold and mint"))
    assert len(calls) == 1
    policy.act(observation(), replace(instruction, replan=True))
    assert len(calls) == 2
    assert policy.last_prediction["model_prompt"] == "Pick coral."


def test_invalid_output_has_no_fallback(checkpoint):
    backend = RelayVLABackend(checkpoint)
    with torch.no_grad():
        backend.network.control[-1].bias[0] = float("nan")
    with pytest.raises(RuntimeError, match="no fallback"):
        RelayVLAPolicy("picker", backend).act(observation(), Instruction("pick", "Pick coral."))


def test_missing_checkpoint_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train"):
        RelayVLABackend(tmp_path / "missing.pt")
