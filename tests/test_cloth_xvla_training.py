"""Native gradient and artifact boundaries without foundation weight loading."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.xvla_adaptation import CHANNELS, load_adapter, native_contract, save_adapter
from relay_lab.cloth.xvla_training import (
    SCHEMA,
    configure_native_gradients,
    load_caches,
    load_training_source,
    native_forward,
    native_loss,
    training_lineage,
)


def test_pointwise_lineage_reads_explicit_nested_sources_and_rejects_missing_seed():
    assert training_lineage(
        {
            "training": {
                "sources": [{"source": {"seed": 10004}}],
                "accumulated_training_seeds": [10005],
            }
        }
    ) == {10004, 10005}
    with pytest.raises(ValueError, match="seed identity"):
        training_lineage({"training": {"training_seeds": [10004], "sources": [{}]}})


def test_xvla_training_source_accepts_only_right_counterfactual_corrections(tmp_path, monkeypatch):
    manifest = {"schema": "relay.cloth.native-action-corrections.v1", "agent": "right"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "relay_lab.cloth.action_dataset.load_corrections",
        lambda _: (manifest, {"right": {"tick": np.array([200])}}),
    )
    assert "right" in load_training_source(tmp_path)[1]
    manifest["agent"] = "left"
    with pytest.raises(ValueError, match="right-arm"):
        load_training_source(tmp_path)


def test_recovery_cache_keeps_physical_tick_separate_from_source_row(tmp_path):
    archive = tmp_path / "features.npz"
    np.savez_compressed(
        archive,
        vlm_features=np.zeros((1, 2, 1024), np.float32),
        aux_visual_inputs=np.zeros((1, 2, 1024), np.float32),
        proprio=np.zeros((1, 20), np.float32),
        targets=np.zeros((1, 4), np.float32),
        tick=np.array([928], np.int64),
        source_index=np.array([3], np.int64),
        phase=np.array([0], np.int64),
    )
    (tmp_path / "cache.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "native_contract": native_contract(),
                "split": "train",
                "seed": 10009,
                "samples": 1,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )
    _, arrays = load_caches([tmp_path])
    assert arrays["tick"].tolist() == [928]
    assert arrays["source_index"].tolist() == [3]


def test_training_lineage_preserves_prior_adapter_exposure():
    assert training_lineage(None) == set()
    assert training_lineage({"training": {"episode_seed": 10004}}) == {10004}
    assert training_lineage(
        {
            "training": {
                "training_seeds": [10007],
                "accumulated_training_seeds": [10004, 10005],
                "sources": [{"seed": 10007}],
            }
        }
    ) == {10004, 10005, 10007}
    with pytest.raises(ValueError, match="lineage"):
        training_lineage({"training": {}})


def test_native_gradients_change_only_domain_five_permitted_parameters():
    torch = pytest.importorskip("torch")
    from lerobot.policies.xvla.action_hub import EE6DActionSpace
    from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer

    torch.manual_seed(71)
    transformer = SoftPromptedTransformer(
        hidden_size=8,
        multi_modal_input_size=8,
        depth=2,
        num_heads=2,
        num_domains=30,
        len_soft_prompts=2,
    )
    before = {name: value.detach().clone() for name, value in transformer.named_parameters()}
    groups, handles = configure_native_gradients(transformer)
    model = SimpleNamespace(transformer=transformer, action_space=EE6DActionSpace())
    optimizer = torch.optim.Adam(groups, lr=1e-3, weight_decay=0)
    batch = {
        "vlm_features": torch.randn(2, 3, 8),
        "aux_visual_inputs": torch.randn(2, 2, 8),
        "proprio": torch.zeros(2, 20),
    }
    prediction = native_forward(model, batch, torch.randn(1, 30, 20))
    loss, _ = native_loss(prediction, torch.tensor([[0.1, 0.2, 0.3, 1.0], [0.2, 0.3, 0.4, 0.0]]))
    loss.backward()
    assert transformer.action_encoder.fc.weight.grad[5].abs().sum() > 0
    assert transformer.soft_prompt_hub.weight.grad[5].abs().sum() > 0
    for parameter in (
        transformer.action_encoder.fc.weight,
        transformer.soft_prompt_hub.weight,
        transformer.action_decoder.fc.weight,
    ):
        assert parameter.grad[:5].count_nonzero() == 0
        assert parameter.grad[6:].count_nonzero() == 0
    decoder_grad = transformer.action_decoder.fc.weight.grad[5].reshape(8, 20)
    unused = [i for i in range(20) if i not in CHANNELS]
    assert decoder_grad[:, unused].count_nonzero() == 0
    optimizer.step()
    for name, parameter in transformer.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, before[name])
        else:
            assert torch.equal(parameter[:5], before[name][:5])
            assert torch.equal(parameter[6:], before[name][6:])
    assert not transformer.training
    assert not any(block.training for block in transformer.blocks)
    for handle in handles:
        handle.remove()


def test_native_loss_has_no_fabricated_peer_or_future_action_labels():
    torch = pytest.importorskip("torch")
    prediction = torch.zeros(1, 30, 20, requires_grad=True)
    target = torch.tensor([[0.1, 0.2, 0.3, 1.0]])
    loss, _ = native_loss(prediction, target)
    loss.backward()
    assert prediction.grad[0, 1:].count_nonzero() == 0
    unused = [i for i in range(20) if i not in CHANNELS]
    assert prediction.grad[0, 0, unused].count_nonzero() == 0
    assert prediction.grad[0, 0, list(CHANNELS)].count_nonzero() == 4


def test_optional_native_input_weights_are_strictly_bound_in_existing_artifact(tmp_path):
    pytest.importorskip("safetensors")
    inputs = {
        "action_encoder.weight": np.zeros((72, 1024), np.float32),
        "action_encoder.bias": np.zeros(1024, np.float32),
        "soft_prompt_hub.weight": np.zeros(32768, np.float32),
    }
    meta = save_adapter(
        tmp_path / "adapted",
        np.zeros((1024, 20), np.float32),
        np.zeros(20, np.float32),
        dataset_sha256="a" * 64,
        training={},
        native_inputs=inputs,
    )
    assert meta["trainable_parameters"] == 111620
    assert meta["adapted_native_inputs"] == ["action_encoder", "soft_prompt_hub"]
    loaded, _, _ = load_adapter(tmp_path / "adapted", expected_sha256=meta["decoder_sha256"])
    assert loaded == meta
    inputs["action_encoder.bias"][0] = np.nan
    with pytest.raises(ValueError, match="native input"):
        save_adapter(
            tmp_path / "invalid",
            np.zeros((1024, 20), np.float32),
            np.zeros(20, np.float32),
            dataset_sha256="a" * 64,
            training={},
            native_inputs=inputs,
        )
    assert not (tmp_path / "invalid").exists()
