"""Feature boundary contracts; actual cached-model smoke is recorded separately."""

from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.features import FEATURE_SCHEMA, FrozenVLAFeatureExtractor, _pool_hidden

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "hidden,mask,width",
    [
        (torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), 4),
        (torch.ones(2, 2, 3), torch.ones(2, 2, dtype=torch.bool), 3),
        (torch.ones(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool), 3),
        (torch.ones(1, 2, 3), torch.ones(1, 2), 3),
        (torch.ones(1, 2, 3), torch.ones(1, 1, dtype=torch.bool), 3),
    ],
)
def test_pool_rejects_wrong_feature_boundary(hidden, mask, width):
    with pytest.raises(ValueError, match="feature tensor"):
        _pool_hidden(hidden, mask, width)


def test_pool_ignores_padding_but_rejects_nonfinite_valid_tokens():
    hidden = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [float("nan"), 0.0]]])
    valid = torch.tensor([[True, True, False]])
    pooled = _pool_hidden(hidden, valid, 2)
    assert pooled == pytest.approx([3, 6])
    assert pooled.dtype == np.float32
    assert not pooled.flags.writeable
    with pytest.raises(ValueError, match="Nonfinite"):
        _pool_hidden(hidden, torch.ones_like(valid), 2)


def _xvla_probe():
    class Core(torch.nn.Module):
        def forward_vlm(self, tokens, images, mask):
            # Two image tokens, two valid language tokens, one padding token.
            assert tokens.shape == (1, 3)
            assert mask.tolist() == [[True, False, False]]
            values = torch.tensor([1.0, 2.0, 3.0, 4.0, 1000.0])
            return {"vlm_features": values[None, :, None].expand(1, 5, 1024)}

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def _prepare_images(self, batch):
            assert torch.count_nonzero(batch["observation.state"]) == 0
            return batch["observation.images.image"], torch.tensor([[True, False, False]])

        def predict_action_chunk(self, batch):
            pytest.fail("Feature extraction must not call the native action head")

    def preprocess(batch):
        return {
            **batch,
            "observation.language.tokens": torch.tensor([[1, 2, 0]]),
            "observation.language.attention_mask": torch.tensor([[True, True, False]]),
        }

    return SimpleNamespace(
        agent="right",
        model=Policy(),
        torch=torch,
        preprocess=preprocess,
        tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "Fold the shirt."),
        model_info={
            "model_id": "lerobot/xvla-folding",
            "revision": "c87d6e8fcf47fb19d6aac8b67d35dd20d74bbc8d",
        },
    )


def test_xvla_extracts_real_boundary_and_excludes_numeric_state_and_padding():
    probe = _xvla_probe()
    extractor = FrozenVLAFeatureExtractor("xvla", "right", probe=probe)
    observation = {
        "agent": "right",
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "ee_position": [float("nan")] * 3,
        "peer_secret": "Never passed to the model",
    }
    features = extractor.extract(observation, "Fold the shirt.")
    assert features.vector.shape == (1024,)
    assert features.vector == pytest.approx(np.full(1024, 2.5))
    assert features.metadata["schema"] == FEATURE_SCHEMA
    assert features.metadata["pooled_tokens"] == 4
    assert features.metadata["foundation_features"]
    assert features.metadata["backbone_frozen"]
    assert not features.metadata["native_action_head_executed"]
    assert not features.metadata["observation_numeric_state_used"]
    assert len(features.metadata["vector_sha256"]) == 64
    assert features.metadata["tokenization"]["full_prompt_roundtrip_verified"]
    assert all(not p.requires_grad for p in probe.model.parameters())


def test_extractor_fails_closed_on_checkpoint_identity_arm_and_token_loss():
    probe = _xvla_probe()
    probe.model_info["revision"] = "unverified"
    with pytest.raises(ValueError, match="pinned feature contract"):
        FrozenVLAFeatureExtractor("xvla", "right", probe=probe)
    with pytest.raises(ValueError, match="another arm"):
        FrozenVLAFeatureExtractor("xvla", "left", probe=_xvla_probe())
    extractor = FrozenVLAFeatureExtractor("xvla", "right", probe=_xvla_probe())
    observation = {"agent": "right", "rgb": np.zeros((16, 16, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="coordination prompt"):
        extractor.extract(observation, "Fold the shirt. Peer sees an obstacle.")
    assert extractor.calls == 0
    with pytest.raises(ValueError, match="Wrong arm"):
        extractor.extract({**observation, "agent": "left"}, "Fold the shirt.")
    extractor.model.train()
    with pytest.raises(RuntimeError, match="evaluation-mode frozen"):
        extractor.extract(observation, "Fold the shirt.")


def test_smol_boundary_excludes_state_and_pad_tokens():
    pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla")

    class VLM(torch.nn.Module):
        def forward(self, **kwargs):
            assert kwargs["use_cache"]
            # Image/language tokens must never attend to the compatibility state.
            assert not bool(kwargs["attention_mask"][0, :2, 2].any())
            values = torch.tensor([1.0, 3.0, 100.0, 1000.0])
            return [values[None, :, None].expand(1, 4, 960), None], None

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm_with_expert = VLM()

        def embed_prefix(self, *args, **kwargs):
            return (
                torch.zeros(1, 4, 960),
                torch.tensor([[True, True, True, False]]),
                torch.tensor([[False, False, True, False]]),
            )

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()

        def _prepare_batch(self, batch):
            return batch

        def prepare_images(self, batch):
            assert torch.count_nonzero(batch["observation.state"]) == 0
            return [], []

        def prepare_state(self, batch):
            return batch["observation.state"]

    class Tokenizer:
        def __call__(self, prompt):
            return {"input_ids": [1, 2]}

        def decode(self, *args, **kwargs):
            return "Fold the shirt."

    backend = SimpleNamespace(
        _model=Policy(),
        _torch=torch,
        _tokenizer=Tokenizer(),
        _config=SimpleNamespace(tokenizer_max_length=512),
        _preprocess=lambda batch: {
            **batch,
            "observation.language.tokens": torch.tensor([[1, 2]]),
            "observation.language.attention_mask": torch.tensor([[True, True]]),
        },
    )
    probe = SimpleNamespace(
        agent="left",
        backend=backend,
        model_info={
            "model_id": "HuggingFaceVLA/smolvla_libero",
            "revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        },
    )
    extractor = FrozenVLAFeatureExtractor("smolvla", "left", probe=probe)
    features = extractor.extract(
        {"agent": "left", "rgb": np.zeros((16, 16, 3), dtype=np.uint8)}, "Fold the shirt."
    )
    assert features.vector.shape == (960,)
    assert features.vector == pytest.approx(np.full(960, 2.0))
    assert features.metadata["pooled_tokens"] == 2
    assert features.metadata["native_action_head_executed"] is False
