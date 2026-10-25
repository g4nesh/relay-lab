"""Adapter-contract checks without downloading or loading foundation weights."""

from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.pretrained import (
    XVLA_DOMAIN_ID,
    XVLAClothProbe,
    _ensure_xvla_image_normalization,
    _tokenization_evidence,
    _xvla_state,
)


@pytest.mark.parametrize("agent,offset", [("left", 0), ("right", 10)])
def test_xvla_local_pose_uses_valid_softfold_rotation_and_no_peer_state(agent, offset):
    state = _xvla_state([0.1, 0.2, 0.3], agent, True)
    pose = state[offset : offset + 10]
    assert pose[:3] == pytest.approx([0.1, 0.2, 0.3])
    columns = pose[3:9].reshape(3, 2)
    assert columns.T @ columns == pytest.approx(np.eye(2))
    assert np.cross(columns[:, 0], columns[:, 1]) == pytest.approx([0, 0, -1])
    assert pose[9] == 1
    assert np.count_nonzero(state[10 - offset : 20 - offset]) == 0


def test_xvla_rejects_nonfinite_proprio_before_inference():
    with pytest.raises(ValueError, match="proprioception"):
        _xvla_state([np.nan, 0.2, 0.3], "right", False)


def test_xvla_legacy_pipeline_image_normalization_is_repaired_exactly_once():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot.policies.xvla.processor_xvla")
    from lerobot.lerobot_types import TransitionKey

    pipeline = SimpleNamespace(steps=[])
    _ensure_xvla_image_normalization(pipeline)
    _ensure_xvla_image_normalization(pipeline)
    assert len(pipeline.steps) == 1
    pixels = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1).expand(1, 3, 2, 2)
    source = {TransitionKey.OBSERVATION: {"observation.images.image": pixels}}
    output = pipeline.steps[0](source)[TransitionKey.OBSERVATION]["observation.images.image"]
    expected = (np.array([0.0, 0.5, 1.0]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    assert output[0, :, 0, 0].numpy() == pytest.approx(expected)
    assert pixels[0, :, 0, 0].numpy() == pytest.approx([0.0, 0.5, 1.0])
    pipeline.steps.append(pipeline.steps[0])
    with pytest.raises(ValueError, match="exactly once"):
        _ensure_xvla_image_normalization(pipeline)


def test_xvla_domain_reaches_model_after_legacy_processor_drops_metadata():
    torch = pytest.importorskip("torch")
    seen = {}
    policy = XVLAClothProbe.__new__(XVLAClothProbe)
    policy.agent, policy.device, policy.seed = "right", "cpu", 7
    policy.torch, policy.calls, policy.seconds = torch, 0, 0.0
    policy.preprocess = lambda batch: {
        "observation.state": batch["observation.state"][None],
        "observation.language.tokens": torch.tensor([[7, 8]]),
    }
    policy.tokenizer = SimpleNamespace(
        decode=lambda ids, **kwargs: "Fold the T-shirt neatly using your arm."
    )
    policy.postprocess = lambda chunk: chunk

    def predict(batch):
        seen.update(batch)
        result = torch.zeros(1, 1, 20)
        result[0, 0, 10:13] = torch.tensor([0.1, 0.2, 0.3])
        result[0, 0, 19] = 1
        return result

    policy.model = SimpleNamespace(predict_action_chunk=predict)
    action = policy.act(
        {
            "agent": "right",
            "ee_position": [0.1, 0.2, 0.3],
            "grasped": False,
            "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        }
    )
    assert XVLA_DOMAIN_ID == 5
    assert seen["domain_id"].tolist() == [5]
    assert seen["observation.state"].shape == (1, 20)
    assert torch.count_nonzero(seen["observation.state"][0, :10]) == 0
    assert action["grip"]
    assert policy.last_prediction["tokenization"]["full_prompt_roundtrip_verified"]


def test_tokenization_audit_rejects_lost_peer_facts():
    torch = pytest.importorskip("torch")
    prepared = {"observation.language.tokens": torch.tensor([[1, 2]])}
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "Fold the shirt.")
    with pytest.raises(ValueError, match="coordination prompt"):
        _tokenization_evidence(prepared, tokenizer, "Fold the shirt. Peer obstacle present.")
    assert _tokenization_evidence(prepared, tokenizer, " Fold the shirt.\n")[
        "full_prompt_roundtrip_verified"
    ]
