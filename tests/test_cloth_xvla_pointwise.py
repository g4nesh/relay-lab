from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.xvla_adaptation import _read_adapter_tensors, save_adapter
from relay_lab.cloth.xvla_pointwise import (
    condition_norm2,
    configure_pointwise,
    forward_pointwise,
    matched_motion_labels,
    native_residual,
    representative_indices,
    sample_balanced,
    whiten_native_fc1,
)


@pytest.mark.parametrize("freeze_norm2", [False, True])
def test_original_pointwise_gradient_boundary_and_output_are_exact(freeze_norm2):
    torch = pytest.importorskip("torch")
    from lerobot.policies.xvla.soft_transformer import DomainAwareLinear, TransformerBlock

    torch.manual_seed(712)
    block = TransformerBlock(8, 2, mlp_ratio=4).eval()
    parts = SimpleNamespace(
        block=block, norm=torch.nn.LayerNorm(8), decoder=DomainAwareLinear(8, 20, num_domains=6)
    )
    prefix = torch.randn(3, 12, 8)
    residual = native_residual(block, prefix).detach()
    expected = parts.decoder(parts.norm(block(prefix)[:, :1]), torch.full((3,), 5))
    torch.testing.assert_close(forward_pointwise(parts, residual), expected, atol=2e-7, rtol=0)
    before = {name: value.detach().clone() for name, value in block.state_dict().items()}
    original_w = parts.decoder.fc.weight.detach().clone()
    parameters, hooks = configure_pointwise(parts, freeze_norm2=freeze_norm2)
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    forward_pointwise(parts, residual)[:, :, [10, 11, 12, 19]].square().mean().backward()
    assert block.attn.qkv.weight.grad is None
    assert block.norm1.weight.grad is None
    assert (block.norm2.weight.grad is None) is freeze_norm2
    assert block.mlp.fc1.weight.grad.abs().sum() > 0
    assert parts.norm.weight.grad.abs().sum() > 0
    optimizer.step()
    for name, value in block.state_dict().items():
        if name.startswith(("attn.", "norm1.")) or (freeze_norm2 and name.startswith("norm2.")):
            assert torch.equal(value, before[name])
    unused = [i for i in range(20) if i not in [10, 11, 12, 19]]
    assert torch.equal(
        parts.decoder.fc.weight[5].view(8, 20)[:, unused], original_w[5].view(8, 20)[:, unused]
    )
    assert torch.equal(parts.decoder.fc.weight[:5], original_w[:5])
    for hook in hooks:
        hook.remove()


def test_native_final_norm_is_strict_and_hash_bound(tmp_path):
    pytest.importorskip("safetensors")
    norm = {"norm.weight": np.ones(1024, np.float32), "norm.bias": np.zeros(1024, np.float32)}
    meta = save_adapter(
        tmp_path / "norm",
        np.zeros((1024, 20), np.float32),
        np.zeros(20, np.float32),
        dataset_sha256="a" * 64,
        training={},
        native_norm=norm,
    )
    loaded, tensors = _read_adapter_tensors(
        tmp_path / "norm", expected_sha256=meta["decoder_sha256"]
    )
    assert loaded["adapted_native_final_norm"] is True
    assert np.array_equal(tensors["norm.weight"], norm["norm.weight"])
    with pytest.raises(ValueError, match="LayerNorm"):
        save_adapter(
            tmp_path / "bad",
            np.zeros((1024, 20), np.float32),
            np.zeros(20, np.float32),
            dataset_sha256="a" * 64,
            training={},
            native_norm={**norm, "extra.weight": np.zeros(1)},
        )


def test_motion_join_uses_exact_row_identity_not_sparse_physical_tick(monkeypatch):
    rows = {
        "targets": np.array([[0.1, 0.2, 0.3, 1], [0.2, 0.3, 0.4, 0]], np.float32),
        "phase": np.array([1, 1]),
        "tick": np.array([100, 205]),
        "source_index": np.array([0, 1]),
    }
    proprio = np.zeros((2, 20), np.float32)
    proprio[:, 10:13] = rows["targets"][:, :3] - 0.01
    source = {"seed": 10009, "archive_sha256": "a" * 64}
    monkeypatch.setattr(
        "relay_lab.cloth.xvla_pointwise.load_caches",
        lambda _: ([source], {**rows, "proprio": proprio}),
    )
    own, motion = matched_motion_labels({"source": source}, rows, "unused")
    assert np.array_equal(own, proprio[:, 10:13])
    np.testing.assert_allclose(motion, np.sqrt(3) * 0.01, atol=1e-7)
    with pytest.raises(ValueError, match="row mismatch"):
        matched_motion_labels(
            {"source": source}, {**rows, "source_index": rows["source_index"][::-1]}, "unused"
        )
    with pytest.raises(ValueError, match="identity"):
        matched_motion_labels({"source": {**source, "seed": 3}}, rows, "unused")


def test_balanced_sampling_guarantees_half_movement_rows():
    data = {"phase": np.repeat(np.arange(4), 20), "motion": np.zeros(80)}
    data["motion"][[0, 20, 40, 60]] = 0.1
    ids = sample_balanced(np.random.default_rng(4), data, 64)
    assert len(ids) == 64
    assert np.all(data["motion"][ids[:32]] > 0.03)
    assert set(data["phase"][ids[32:]]) == {0, 1, 2, 3}


def test_conditioning_is_folded_into_only_original_norm2_affine():
    torch = pytest.importorskip("torch")
    from lerobot.policies.xvla.soft_transformer import TransformerBlock

    torch.manual_seed(11)
    parts = SimpleNamespace(block=TransformerBlock(8, 2).eval())
    rows = torch.randn(1, 1, 8) + torch.randn(32, 1, 8) * 0.02
    with torch.no_grad():
        original = parts.block.norm2(rows).clone()
        expected = (original - original.mean((0, 1))) / original.std(
            (0, 1), unbiased=False
        ).clamp_min(0.001)
    before = {name: value.clone() for name, value in parts.block.state_dict().items()}
    result = condition_norm2(parts, rows)
    torch.testing.assert_close(parts.block.norm2(rows), expected, atol=1e-4, rtol=0)
    assert result["new_runtime_modules"] == 0
    for name, value in parts.block.state_dict().items():
        if not name.startswith("norm2."):
            assert torch.equal(value, before[name])


def test_whitening_changes_only_original_fc1_with_exact_affine_effect():
    torch = pytest.importorskip("torch")
    from lerobot.policies.xvla.soft_transformer import TransformerBlock

    torch.manual_seed(12)
    parts = SimpleNamespace(block=TransformerBlock(8, 2).eval())
    rows = torch.randn(16, 1, 8) * 0.02 + torch.randn(1, 1, 8)
    before = {name: value.clone() for name, value in parts.block.state_dict().items()}
    result = whiten_native_fc1(parts, rows)
    assert result["new_runtime_modules"] == 0
    assert result["affine_equivalence_max_error"] < 1e-3
    for name, value in parts.block.state_dict().items():
        if not name.startswith("mlp.fc1."):
            assert torch.equal(value, before[name])


def test_proof_indices_include_phase_boundaries_and_never_exceed_source_rows():
    phases = np.repeat(np.arange(16), [7] * 16)
    indices = representative_indices(phases)
    assert len(indices) == 32
    assert set(np.arange(16) * 7) <= set(indices)
    assert max(indices) < len(phases)
    assert representative_indices(np.array([0, 1, 2])) == [0, 1, 2]
