"""Check cached native state gradients and strict frozen-parameter boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.smol_native_training import (
    adaptation_lineage,
    enable_native_projection_training,
    native_cached_velocity,
    phase_balanced_schedule,
    select_phase_samples,
    validate_cache_source,
)


def test_cached_state_path_retains_gradients_without_mutating_frozen_prefix():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class Backbone:
        attention_mode = "cross_attn"

        def forward(self, **kwargs):
            assert self.attention_mode == "self_attn"
            assert kwargs["inputs_embeds"][1] is None
            assert kwargs["attention_mask"].shape == (1, 1, 3)
            assert kwargs["position_ids"].tolist() == [[2]]
            value = kwargs["inputs_embeds"][0][:, None]
            cache = kwargs["past_key_values"]
            cache.update(value, value * 2, 0)
            return [kwargs["inputs_embeds"][0], None], cache

    class Flow:
        state_proj = torch.nn.Linear(32, 1)
        vlm_with_expert = Backbone()

        def denoise_step(self, **kwargs):
            assert kwargs["prefix_pad_masks"].shape == (1, 3)
            assert kwargs["timestep"].tolist() == [1]
            value = kwargs["past_key_values"].layers[0].values[:, :, -1:]
            return value.reshape(1, 1, 1).expand(1, 50, 32)

    flow = Flow()
    backend = SimpleNamespace(_torch=torch, device="cpu", _model=SimpleNamespace(model=flow))
    prefix = torch.arange(2, dtype=torch.float32).reshape(1, 1, 2, 1)
    cached = {
        "pairs": [(prefix, prefix.clone())],
        "prefix_mask": torch.ones((1, 2), dtype=torch.bool),
        "state": torch.ones((1, 32)),
    }
    velocity = native_cached_velocity(backend, cached)
    changed = {**cached, "state": torch.full((1, 32), 2.0)}
    second = native_cached_velocity(backend, changed)
    repeated = native_cached_velocity(backend, cached)
    assert not torch.equal(velocity, second)
    assert torch.equal(velocity, repeated)
    velocity.square().mean().backward()
    assert flow.state_proj.weight.grad is not None
    assert torch.count_nonzero(flow.state_proj.weight.grad) > 0
    assert torch.equal(cached["pairs"][0][0], prefix)
    assert cached["pairs"][0][0].shape[-2] == 2
    assert flow.vlm_with_expert.attention_mode == "cross_attn"


def test_projection_training_freezes_backbone_and_original_transformer_expert():
    torch = pytest.importorskip("torch")
    flow = torch.nn.Module()
    flow.state_proj = torch.nn.Linear(32, 960)
    flow.action_in_proj = torch.nn.Linear(32, 480)
    flow.action_time_mlp_in = torch.nn.Linear(960, 480)
    flow.action_time_mlp_out = torch.nn.Linear(480, 480)
    flow.action_out_proj = torch.nn.Linear(480, 32)
    flow.vlm_with_expert = torch.nn.Linear(10, 10)
    model = torch.nn.Module()
    model.model = flow
    selected = enable_native_projection_training(SimpleNamespace(_model=model))
    assert sum(p.numel() for p in selected) == 755072
    assert all(p.requires_grad for p in selected)
    assert not any(p.requires_grad for p in flow.vlm_with_expert.parameters())


def test_phase_sampling_preserves_distinct_phases_and_excludes_holds():
    arrays = {"phase": np.repeat(np.arange(16), 5), "hold": np.zeros(80, dtype=bool)}
    arrays["hold"][::5] = True
    selected = select_phase_samples(arrays, per_phase=2)
    assert len(selected) == 32
    assert set(arrays["phase"][selected]) == set(range(16))
    assert not arrays["hold"][selected].any()


def test_warm_start_lineage_rejects_previous_training_and_normalization_validation_overlap():
    adapter = {"training": {"train_seeds": [10004, 10005], "normalization_training_seed": 10003}}
    current = [{"seed": 10007, "manifest_sha256": "new"}]
    for seed in (10003, 10004, 10005, 10007):
        with pytest.raises(ValueError, match="lineage overlaps"):
            adaptation_lineage(adapter, current, [{"seed": seed}])
    lineage = adaptation_lineage(adapter, current, [{"seed": 10008}])
    assert lineage["adaptation_train_seeds"] == [10003, 10004, 10005, 10007]


def test_arm_provenance_rejects_legacy_left_cache_for_right_adaptation():
    adapter = {"agent": "right", "training": {"train_seeds": [10004]}}
    right = {"agent": "right", "seed": 10007, "manifest_sha256": "right-source"}
    validation = {"agent": "right", "seed": 10008}
    lineage = adaptation_lineage(adapter, [right], [validation])
    assert lineage["adaptation_train_seeds"] == [10004, 10007]
    for invalid in ({k: v for k, v in right.items() if k != "agent"}, {**right, "agent": "left"}):
        with pytest.raises(ValueError, match="different arm"):
            adaptation_lineage(adapter, [invalid], [validation])
    with pytest.raises(ValueError, match="different arm"):
        adaptation_lineage(adapter, [right], [{**validation, "agent": "left"}])


def test_cache_reuse_verifies_source_identity_weights_and_equivalence(tmp_path):
    import hashlib

    from relay_lab.cloth.smol_adaptation import BASE_WEIGHT_SHA256
    from relay_lab.cloth.smol_native_training import CACHE_CONTRACT

    source = tmp_path / "manifest.json"
    source.write_text('{"seed": 10004}')
    backend = SimpleNamespace(model_id="pinned-model", revision="pinned-revision")
    valid = {
        "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_id": backend.model_id,
        "revision": backend.revision,
        "base_weight_sha256": BASE_WEIGHT_SHA256,
        "cache_contract": CACHE_CONTRACT,
        "normalization": {"source": "train"},
        "equivalence": {"verified": True, "max_normalized_action_difference": 1e-5},
    }
    assert validate_cache_source(valid, tmp_path, backend, valid["normalization"]) == valid
    with pytest.raises(ValueError, match="different arm"):
        validate_cache_source(valid, tmp_path, backend, valid["normalization"], agent="right")
    right = {**valid, "agent": "right"}
    assert (
        validate_cache_source(right, tmp_path, backend, valid["normalization"], agent="right")
        == right
    )
    for field, value in (
        ("base_weight_sha256", "changed"),
        ("revision", "other"),
        ("equivalence", {"verified": False}),
    ):
        with pytest.raises(ValueError):
            validate_cache_source(
                {**valid, field: value}, tmp_path, backend, valid["normalization"]
            )
    source.write_text('{"seed": 10009}')
    with pytest.raises(ValueError, match="source or pinned"):
        validate_cache_source(valid, tmp_path, backend, valid["normalization"])


def test_phase_balancing_handles_uneven_corrective_dataset_without_validation():
    phases = np.array([0] * 3 + [2] * 28 + [7] * 2)
    schedule = phase_balanced_schedule(phases, steps=12, batch_size=4)
    assert schedule.shape == (12, 4)
    assert np.array_equal(schedule, phase_balanced_schedule(phases, steps=12, batch_size=4))
    selected = phases[schedule].flatten()
    assert {int(phase): int((selected == phase).sum()) for phase in np.unique(phases)} == {
        0: 16,
        2: 16,
        7: 16,
    }
