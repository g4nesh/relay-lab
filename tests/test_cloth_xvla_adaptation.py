"""Native-boundary contracts without loading foundation-model weights."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.pretrained import XVLA_ID, XVLA_REVISION, XVLA_WEIGHT_SHA256
from relay_lab.cloth.xvla_adaptation import (
    CHANNELS,
    FEATURE_WIDTH,
    NativeXVLA,
    decoder_metrics,
    evaluate_cache,
    fit_cache,
    fit_native_columns,
    load_adapter,
    measured_local_state,
    native_contract,
    save_adapter,
    supervised_targets,
)


def observation():
    return {
        "agent": "right",
        "tick": 0,
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "ee_position": [0.1, 0.2, 0.3],
        "ee_rotation": np.diag([1, -1, -1]),
        "gripper_closed": True,
        "grasped": False,
    }


def test_measured_pose_does_not_fabricate_peer_or_tool_rotation():
    state = measured_local_state(observation())
    assert np.count_nonzero(state[:10]) == 0
    assert state[10:13] == pytest.approx([0.1, 0.2, 0.3])
    assert state[13:19].tolist() == [1, 0, 0, -1, 0, 0]
    assert state[19] == 1
    for replacement in (None, np.zeros((3, 3)), np.diag([1, 1, -1])):
        with pytest.raises(ValueError, match="measured"):
            measured_local_state({**observation(), "ee_rotation": replacement})
    with pytest.raises(ValueError, match="right arm"):
        measured_local_state({**observation(), "agent": "left"})


def test_native_t1_targets_are_absolute_endpoints_and_grip_logits():
    result = supervised_targets([[0.3, -0.2, 0.1], [0.1, 0.2, 0.3]], [1, 0])
    assert result[:, :3].tolist() == [[0.3, -0.2, 0.1], [0.1, 0.2, 0.3]]
    assert result[:, 3].tolist() == [5, -5]
    with pytest.raises(ValueError):
        supervised_targets([[0.1, 0.2, np.nan]], [1])


def test_ridge_updates_only_executed_columns_and_learns_bias():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(16, FEATURE_WIDTH))
    w = rng.normal(size=(FEATURE_WIDTH, 20)).astype(np.float32) * 0.001
    b = rng.normal(size=20).astype(np.float32) * 0.001
    y = np.column_stack([x[:, :3] * 0.1 + 0.25, np.where(x[:, 3] > 0, 5, -5)])
    fitted_w, fitted_b = fit_native_columns(x, y, w, b, ridge=1e-5)
    untouched = [i for i in range(20) if i not in CHANNELS]
    assert np.array_equal(fitted_w[:, untouched], w[:, untouched])
    assert np.array_equal(fitted_b[untouched], b[untouched])
    metrics = decoder_metrics(x, y, fitted_w, fitted_b)
    assert metrics["xyz_mae_metres"] < 1e-5
    assert metrics["grip_accuracy"] == 1
    assert not np.array_equal(fitted_w[:, CHANNELS], w[:, CHANNELS])
    with pytest.raises(ValueError):
        fit_native_columns(x, y, w, b, ridge=0)


def test_artifact_pins_tensors_and_preserves_native_dimensions(tmp_path):
    pytest.importorskip("safetensors")
    w = np.zeros((FEATURE_WIDTH, 20), np.float32)
    b = np.zeros(20, np.float32)
    meta = save_adapter(
        tmp_path / "adapted", w, b, dataset_sha256="a" * 64, training={"split": "train"}
    )
    loaded_meta, loaded_w, loaded_b = load_adapter(
        tmp_path / "adapted", expected_sha256=meta["decoder_sha256"]
    )
    assert loaded_meta["trainable_parameters"] == 4100
    assert np.array_equal(w, loaded_w) and np.array_equal(b, loaded_b)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_adapter(tmp_path / "adapted", expected_sha256="0" * 64)
    path = tmp_path / "adapted" / "native_decoder.safetensors"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_adapter(tmp_path / "adapted")


def fake_probe():
    torch = pytest.importorskip("torch")
    from torch import nn

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Embedding(30, FEATURE_WIDTH * 20)
            self.bias = nn.Embedding(30, 20)
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.bias.weight)
            with torch.no_grad():
                self.bias.weight[5, 10:13] = torch.tensor([0.1, 0.2, 0.3])
                self.bias.weight[5, 19] = 5

        def forward(self, hidden, domain):
            return (
                hidden @ self.fc(domain).reshape(1, FEATURE_WIDTH, 20) + self.bias(domain)[:, None]
            )

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_denoising_steps=10)
            self.model = nn.Module()
            self.model.config = self.config
            self.model.transformer = nn.Module()
            self.model.transformer.action_decoder = Decoder()
            self.calls = 0
            self.seen = None

        def predict_action_chunk(self, batch):
            self.calls += 1
            self.seen = batch
            hidden = torch.randn(1, 30, FEATURE_WIDTH)
            result = self.model.transformer.action_decoder(hidden, batch["domain_id"])
            result[..., [9, 19]] = torch.sigmoid(result[..., [9, 19]])
            return result

    return SimpleNamespace(
        agent="right",
        device="cpu",
        torch=torch,
        calls=0,
        seconds=0,
        model=Policy(),
        model_info={
            "model_id": XVLA_ID,
            "revision": XVLA_REVISION,
            "checkpoint_sha256": XVLA_WEIGHT_SHA256,
        },
        preprocess=lambda b: {
            "observation.state": b["observation.state"][None],
            "observation.language.tokens": torch.tensor([[7]]),
        },
        tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: "Fold the right sleeve."),
        postprocess=lambda output: output,
    )


def test_cache_and_inference_execute_native_predict_chunk_with_same_noise():
    probe = fake_probe()
    policy = NativeXVLA(probe=probe)
    assert policy.model_info["native_action_head_executed"] is False
    first_chunk, first_hidden = policy.predict(
        observation(), "Fold the right sleeve.", capture_features=True
    )
    second_chunk, second_hidden = policy.predict(
        observation(), "Fold the right sleeve.", capture_features=True
    )
    assert probe.model.calls == 2
    assert policy.model_info["native_action_head_executed"] is True
    assert np.array_equal(first_hidden, second_hidden)
    assert np.array_equal(first_chunk, second_chunk)
    assert first_hidden.shape == (FEATURE_WIDTH,)
    assert first_chunk.shape == (30, 20)
    assert probe.model.seen["domain_id"].tolist() == [5]
    assert probe.model.seen["observation.state"][0, :10].count_nonzero() == 0
    assert probe.model.config.num_denoising_steps == 1
    assert not probe.model.model.transformer.action_decoder._forward_pre_hooks
    result = policy.act(observation(), "Fold the right sleeve.")
    assert result == {"target": pytest.approx([0.1, 0.2, 0.3]), "grip": True}
    assert probe.model.calls == 3
    assert policy.last_prediction["native_action_head_executed"]


def test_loaded_adapter_changes_native_parameters_and_rejects_unapproved_columns(tmp_path):
    probe = fake_probe()
    original = NativeXVLA(probe=probe)
    w, b = original.decoder_parameters()
    b[19] = -5
    save_adapter(tmp_path / "good", w, b, dataset_sha256="a" * 64, training={})
    adapted = NativeXVLA(probe=fake_probe(), adapter=tmp_path / "good")
    assert not adapted.act(observation(), "Fold the right sleeve.")["grip"]
    assert adapted.probe.model.calls == 1
    w[0, 0] = 1
    save_adapter(tmp_path / "bad", w, b, dataset_sha256="a" * 64, training={})
    with pytest.raises(ValueError, match="unapproved"):
        NativeXVLA(probe=fake_probe(), adapter=tmp_path / "bad")


def test_native_multimodal_artifact_installs_original_modules_and_own_prompt_row(tmp_path):
    probe = fake_probe()
    torch = probe.torch
    transformer = probe.model.model.transformer
    transformer.vlm_proj = torch.nn.Linear(1024, 1024)
    transformer.soft_prompt_hub = torch.nn.Embedding(30, 32768)
    transformer.norm = torch.nn.LayerNorm(1024)
    original_other_prompts = transformer.soft_prompt_hub.weight[:5].detach().clone()
    original = NativeXVLA(probe=probe)
    weight, bias = original.decoder_parameters()
    multimodal = {
        "vlm_proj.weight": np.full((1024, 1024), 0.125, np.float32),
        "vlm_proj.bias": np.full(1024, 0.25, np.float32),
        "soft_prompt_hub.weight": np.full(32768, 0.375, np.float32),
    }
    norm = {
        "norm.weight": np.full(1024, 0.5, np.float32),
        "norm.bias": np.full(1024, 0.625, np.float32),
    }
    metadata = save_adapter(
        tmp_path / "multimodal",
        weight,
        bias,
        dataset_sha256="a" * 64,
        training={},
        native_multimodal=multimodal,
        native_norm=norm,
    )
    policy = NativeXVLA(
        probe=probe, adapter=tmp_path / "multimodal", expected_sha256=metadata["decoder_sha256"]
    )
    assert np.array_equal(
        transformer.vlm_proj.weight.detach().numpy(), multimodal["vlm_proj.weight"]
    )
    assert np.array_equal(transformer.vlm_proj.bias.detach().numpy(), multimodal["vlm_proj.bias"])
    assert np.array_equal(transformer.norm.weight.detach().numpy(), norm["norm.weight"])
    assert np.array_equal(
        transformer.soft_prompt_hub.weight[5].detach().numpy(), multimodal["soft_prompt_hub.weight"]
    )
    assert torch.equal(transformer.soft_prompt_hub.weight[:5], original_other_prompts)
    assert policy.model_info["adapted_native_multimodal"] == ["vlm_proj", "soft_prompt_hub"]


def test_native_absolute_action_reuse_expires_on_horizon_instruction_and_hold():
    probe = fake_probe()
    policy = NativeXVLA(probe=probe)
    seen = []
    probe.tokenizer.decode = lambda ids, **kw: seen[-1]
    original_preprocess = probe.preprocess

    def prepare(batch):
        seen.append(batch["task"])
        return original_preprocess(batch)

    probe.preprocess = prepare
    # Native xyz is used as an absolute endpoint; no 18 mm per-inference clamp.
    with probe.torch.no_grad():
        probe.model.model.transformer.action_decoder.bias.weight[5, 10] = 0.4
    first = policy.act(observation(), "lift_sleeve")
    assert first["target"][0] == pytest.approx(0.4)
    assert seen[-1] == "Fold the T-shirt. Right arm: lift the grasped sleeve."
    reused = policy.act({**observation(), "tick": 7}, "lift_sleeve")
    assert reused == first and probe.model.calls == 1
    assert policy.last_prediction["action_age_ticks"] == 7
    policy.act({**observation(), "tick": 8}, "lift_sleeve")
    assert probe.model.calls == 2
    policy.act({**observation(), "tick": 9}, "carry_sleeve")
    assert probe.model.calls == 3
    stopped = policy.act({**observation(), "tick": 10}, "carry_sleeve", hold=True)
    assert stopped["target"] == observation()["ee_position"]
    assert stopped["grip"] and policy.last_prediction["coordinator_hold"]
    assert policy.last_prediction["native_action_head_executed"] is False
    assert probe.model.calls == 3
    policy.act({**observation(), "tick": 11}, "carry_sleeve")
    assert probe.model.calls == 4
    with pytest.raises(ValueError, match="backward"):
        policy.act({**observation(), "tick": 5}, "carry_sleeve")


def test_native_action_frame_cannot_silently_change_in_postprocessing():
    probe = fake_probe()
    probe.postprocess = lambda output: output + 0.1
    policy = NativeXVLA(probe=probe)
    with pytest.raises(RuntimeError, match="action frame"):
        policy.predict(observation(), "Fold the right sleeve.")


def test_configurable_horizon_is_bounded_and_reported():
    with pytest.raises(ValueError, match="horizon"):
        NativeXVLA(probe=fake_probe(), action_horizon=0)
    with pytest.raises(ValueError, match="horizon"):
        NativeXVLA(probe=fake_probe(), action_horizon=True)
    probe = fake_probe()
    policy = NativeXVLA(probe=probe, action_horizon=1)
    policy.act(observation(), "Fold the right sleeve.")
    policy.act({**observation(), "tick": 1}, "Fold the right sleeve.")
    assert probe.model.calls == 2
    assert policy.model_info["action_horizon"] == 1


def test_cache_fit_reports_interpolation_separately_from_rollout_success(tmp_path):
    pytest.importorskip("safetensors")
    rng = np.random.default_rng(19)
    features = rng.normal(size=(20, FEATURE_WIDTH)).astype(np.float32)
    targets = np.column_stack([features[:, :3] * 0.1, np.sign(features[:, 3]) * 5]).astype(
        np.float32
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / "features.npz"
    np.savez_compressed(
        archive,
        features=features,
        targets=targets,
        original_weight=np.zeros((FEATURE_WIDTH, 20), np.float32),
        original_bias=np.zeros(20, np.float32),
        phase=np.zeros(20, dtype=int),
    )
    metadata = {
        **native_contract(),
        "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "episode_manifest_sha256": "a" * 64,
        "episode_seed": 10004,
        "noise_device": "cpu",
    }
    (cache / "cache.json").write_text(json.dumps(metadata))
    trained = fit_cache(cache, tmp_path / "trained", ridge=1e-4)
    assert trained["training"]["closed_loop_success"] is None
    assert trained["training"]["within_training_episode_interpolation"]["samples"] == 4
    assert trained["training"]["fit"]["xyz_mae_metres"] < 1e-5
    metrics = evaluate_cache(cache, tmp_path / "trained")
    assert metrics["held_out_episode"] is False
    assert metrics["closed_loop_task_success"] is None
    assert metrics["overall"]["samples"] == 20
    save_adapter(
        tmp_path / "changed-norm",
        np.zeros((1024, 20), np.float32),
        np.zeros(20, np.float32),
        dataset_sha256="a" * 64,
        training={"noise_device": "cpu"},
        native_norm={
            "norm.weight": np.ones(1024, np.float32),
            "norm.bias": np.zeros(1024, np.float32),
        },
    )
    with pytest.raises(ValueError, match="feature-producing modules"):
        evaluate_cache(cache, tmp_path / "changed-norm")
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        evaluate_cache(cache, tmp_path / "trained")
