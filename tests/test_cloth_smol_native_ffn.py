"""The fast training boundary must remain the original native expert module."""

from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.smol_adaptation import (
    FINAL_FORMAT,
    FINAL_PARAMETER_SHAPES,
    RIGHT_YAW_CONTRACT,
    fixed_noise,
    load_native_adapter,
    save_native_adapter,
    training_normalization,
    validate_native_parameters,
)
from relay_lab.cloth.smol_native_ffn import (
    capture_final_residual,
    enable_final_training,
    final_native_velocity,
    frozen_upstream_identity,
    unused_phase_samples,
)


def final_backend():
    torch = pytest.importorskip("torch")

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(480, 1280, bias=False)
            self.up_proj = torch.nn.Linear(480, 1280, bias=False)
            self.down_proj = torch.nn.Linear(1280, 480, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    layer = torch.nn.Module()
    layer.post_attention_layernorm = torch.nn.RMSNorm(480)
    layer.mlp = MLP()
    expert = torch.nn.Module()
    expert.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(31)] + [layer])
    expert.norm = torch.nn.RMSNorm(480)
    flow = torch.nn.Module()
    flow.vlm_with_expert = torch.nn.Module()
    flow.vlm_with_expert.lm_expert = expert
    flow.action_out_proj = torch.nn.Linear(480, 32)
    flow.frozen_preceding_layer = torch.nn.Linear(480, 480)
    model = torch.nn.Module()
    model.model = flow
    return SimpleNamespace(_torch=torch, _model=model, device="cpu")


def test_original_final_block_token_cache_matches_full_native_sequence_and_removes_hook(
    monkeypatch,
):
    from relay_lab.cloth import smol_native_ffn

    backend = final_backend()
    torch = backend._torch
    residual = torch.randn(1, 50, 480)

    def direct(*_args, **_kwargs):
        with torch.no_grad():
            velocity = final_native_velocity(backend, residual)
            return (fixed_noise(torch, "cpu") - velocity)[0, :, :7].numpy(), None

    monkeypatch.setattr(smol_native_ffn, "native_predict", direct)
    cached, error = capture_final_residual(backend, None, None, None, None)
    assert np.allclose(cached, residual[0, 0].numpy())
    assert error < 2e-4
    assert not backend._model.model.vlm_with_expert.lm_expert.layers[
        31
    ].post_attention_layernorm._forward_pre_hooks


def test_native_final_parameters_receive_gradient_with_prior_layers_frozen():
    backend = final_backend()
    torch = backend._torch
    parameters = enable_final_training(backend)
    assert sum(p.numel() for p in parameters) == 1_859_552
    assert not backend._model.model.frozen_preceding_layer.weight.requires_grad
    original = (
        backend._model.model.vlm_with_expert.lm_expert.layers[31]
        .mlp.down_proj.weight.detach()
        .clone()
    )
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    loss = final_native_velocity(backend, torch.randn(4, 1, 480)).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(
        original, backend._model.model.vlm_with_expert.lm_expert.layers[31].mlp.down_proj.weight
    )
    assert backend._model.model.frozen_preceding_layer.weight.grad is None


def test_v3_serialization_only_accepts_original_final_modules(tmp_path):
    torch = pytest.importorskip("torch")
    tensors = {name: torch.zeros(shape) for name, shape in FINAL_PARAMETER_SHAPES.items()}
    assert validate_native_parameters(tensors, include_final=True) is tensors
    with pytest.raises(ValueError, match="unapproved"):
        validate_native_parameters(tensors)
    with pytest.raises(ValueError, match="unapproved"):
        validate_native_parameters(
            {**tensors, "auxiliary_policy.weight": torch.zeros(7, 480)}, include_final=True
        )
    path = tmp_path / "native.pt"
    stats = training_normalization(np.zeros((3, 8)), np.zeros((3, 7)))
    digest = save_native_adapter(
        path,
        "left",
        {
            "weight": np.zeros((32, 480), dtype=np.float32),
            "bias": np.zeros(32, dtype=np.float32),
            "fit": {"horizon": 1},
        },
        stats,
        {"train_seed": 10004},
        native_parameters=tensors,
    )
    loaded = load_native_adapter(path, "left", expected_sha256=digest)
    assert loaded["format"] == FINAL_FORMAT
    assert set(loaded["native_parameters"]) == set(FINAL_PARAMETER_SHAPES)


def test_native_residual_cache_identity_excludes_only_trainable_downstream_modules():
    torch = pytest.importorskip("torch")
    parameters = {name: torch.zeros(shape) for name, shape in FINAL_PARAMETER_SHAPES.items()}
    adapter = {"native_parameters": parameters, "noise_seed": 2718}
    original = frozen_upstream_identity(adapter)
    parameters["vlm_with_expert.lm_expert.layers.31.mlp.down_proj.weight"].add_(1)
    assert frozen_upstream_identity(adapter) == original
    parameters["state_proj.bias"].add_(1)
    assert frozen_upstream_identity(adapter) != original


def test_unused_frame_diagnostic_never_reuses_selected_training_frames_or_holds():
    from relay_lab.cloth.smol_native_training import select_phase_samples

    arrays = {"phase": np.repeat(np.arange(16), 24), "hold": np.zeros(384, bool)}
    arrays["hold"][::24] = True
    training = set(select_phase_samples(arrays, 10))
    diagnostic = unused_phase_samples(arrays, training_per_phase=10)
    assert len(diagnostic) == 32
    assert not training & set(diagnostic)
    assert not arrays["hold"][diagnostic].any()
    assert np.array_equal(np.bincount(arrays["phase"][diagnostic]), np.full(16, 2))


def test_right_cache_uses_right_camera_proprio_language_and_rejects_cross_arm_reuse(
    monkeypatch, tmp_path
):
    import json

    from relay_lab.cloth import action_dataset, smol_native_ffn

    def rows(x):
        return {
            "rgb": np.full((2, 2, 2, 3), int(x * 100), np.uint8),
            "ee_position": np.full((2, 3), x, np.float32),
            "ee_rotation": np.repeat(np.eye(3)[None], 2, axis=0),
            "gripper_closed": np.zeros(2, bool),
            "target": np.full((2, 3), x + 0.01, np.float32),
            "grip": np.array([False, True]),
            "phase": np.array([0, 1]),
            "tick": np.array([0, 1]),
            "hold": np.zeros(2, bool),
        }

    episodes = {"left": rows(0.1), "right": rows(0.2)}
    for index, yaw in enumerate((np.pi - 0.02, -np.pi + 0.02)):
        episodes["right"]["ee_rotation"][index] = [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    manifest = {"schema": "test-expert", "seed": 10007, "success": True}
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(action_dataset, "load_episode", lambda _path: (manifest, episodes))
    seen = []

    def capture(_backend, rgb, state, prompt, _normalization):
        seen.append((rgb.copy(), state.copy(), prompt))
        return np.zeros(480, np.float32), 0.0

    monkeypatch.setattr(smol_native_ffn, "capture_final_residual", capture)
    policy = SimpleNamespace(
        agent="right",
        backend=SimpleNamespace(model_id="pinned-model", revision="pinned-revision"),
        adapter={
            "noise_seed": 2718,
            "normalization": {"orientation_contract": RIGHT_YAW_CONTRACT},
            "sha256": "initializer",
        },
    )
    output = tmp_path / "cache"
    result = smol_native_ffn.build_final_cache(policy, source, output, per_phase=1)
    assert result["agent"] == "right"
    assert len(seen) == 2
    assert all(np.all(rgb == 20) and np.allclose(state[:3], 0.2) for rgb, state, _ in seen)
    assert all("Right arm:" in prompt for _, _, prompt in seen)
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as cached:
        np.testing.assert_allclose(cached["actions"][:, :3], 0.21)
        np.testing.assert_allclose(cached["actions"][:, 5], [-0.02, 0.02], atol=2e-7)
        assert np.array_equal(cached["actions"][:, 6], [-1, 1])
    policy.agent = "left"
    with pytest.raises(ValueError, match="different source/model/contract"):
        smol_native_ffn.build_final_cache(policy, source, output, per_phase=1)
    monkeypatch.setattr(
        action_dataset, "load_episode", lambda _path: (manifest, {"left": episodes["left"]})
    )
    with pytest.raises(ValueError, match="requested arm"):
        smol_native_ffn.load_source(source, "right")
