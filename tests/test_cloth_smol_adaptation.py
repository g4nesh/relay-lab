"""Native-head fitting contracts; no pretrained downloads or neural fallback."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth.smol_adaptation import (
    ACTION_DIM,
    CHUNK_SIZE,
    HIDDEN_DIM,
    NATIVE_ACTION_DIM,
    NATIVE_PARAMETER_SHAPES,
    RIGHT_YAW_CONTRACT,
    SmolNativePolicy,
    fit_native_head,
    install_native_head,
    load_native_adapter,
    native_predict,
    orientation_contract,
    prepare_right_initializer,
    referenced_orientation,
    save_native_adapter,
    training_normalization,
    unnormalize,
    validate_native_parameters,
    validate_normalization,
    velocity_labels,
    verify_settled_rebind,
)


def normalization():
    return training_normalization(np.zeros((3, 8)), np.zeros((3, 7)))


def backend():
    torch = pytest.importorskip("torch")
    projection = torch.nn.Linear(HIDDEN_DIM, NATIVE_ACTION_DIM)
    model = SimpleNamespace(model=SimpleNamespace(action_out_proj=projection), calls=0)

    def predict(batch, noise):
        model.calls += 1
        model.last_batch = batch
        features = torch.ones((1, CHUNK_SIZE, HIDDEN_DIM))
        features[:, :, :8] = batch["observation.state"][:, None]
        return (noise - projection(features))[:, :, :ACTION_DIM]

    model.predict_action_chunk = predict
    return SimpleNamespace(
        _torch=torch,
        device="cpu",
        _model=model,
        _config=SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=NATIVE_ACTION_DIM,
            num_steps=1,
            tokenizer_max_length=128,
        ),
        _tokenizer=lambda text: {"input_ids": [1, 2]},
        # Deliberately wrong LIBERO normalization: adapter must replace it.
        _preprocess=lambda batch: {**batch, "observation.state": torch.full((1, 8), 999.0)},
    )


def test_training_normalization_roundtrips_and_constant_axes_stay_finite():
    states = np.array([[0] * 8, [2] * 8], dtype=np.float32)
    actions = np.array([[0, 1, 2, 3, 4, 5, -1], [2, 1, 4, 3, 4, 5, 1]], dtype=np.float32)
    stats = training_normalization(states, actions)
    assert stats["state"]["mean"] == [1] * 8
    assert stats["action"]["std"][1] == 1
    assert np.isfinite(unnormalize(np.zeros((50, 7)), stats["action"])).all()
    with pytest.raises(ValueError):
        validate_normalization({**stats, "source": "test_and_train"})


def test_right_yaw_chart_is_continuous_without_changing_xyz_grip_or_legacy_values():
    states = np.arange(24, dtype=np.float32).reshape(3, 8)
    states[:, 5] = [np.pi - 0.02, -np.pi + 0.02, np.pi]
    actions = states[:, :7].copy()
    assert np.array_equal(referenced_orientation(states), states)
    for raw in (states, actions):
        referenced = referenced_orientation(raw, RIGHT_YAW_CONTRACT)
        np.testing.assert_allclose(referenced[:, 5], [-0.02, 0.02, 0], atol=2e-7)
        untouched = [i for i in range(raw.shape[1]) if i != 5]
        assert np.array_equal(referenced[:, untouched], raw[:, untouched])
        assert not np.shares_memory(referenced, raw)
    stats = training_normalization(states, actions, orientation=RIGHT_YAW_CONTRACT)
    assert stats["orientation_contract"] == RIGHT_YAW_CONTRACT
    assert abs(stats["state"]["mean"][5]) < 2e-7
    assert stats["state"]["std"][5] < 0.02
    with pytest.raises(ValueError, match="right arm"):
        orientation_contract(stats, agent="left")
    with pytest.raises(ValueError, match="unrecognized"):
        validate_normalization({**stats, "orientation_contract": "unknown"})


def test_native_preparation_applies_declared_yaw_chart_once():
    b = backend()
    states = np.zeros((3, 8), np.float32)
    states[:, 5] = [np.pi - 0.1, -np.pi + 0.1, np.pi]
    stats = training_normalization(states, states[:, :7], orientation=RIGHT_YAW_CONTRACT)
    native_predict(b, np.zeros((16, 16, 3), np.uint8), states[0], "Right arm: grasp.", stats)
    expected = (
        referenced_orientation(states[0], RIGHT_YAW_CONTRACT)
        - np.asarray(stats["state"]["mean"], np.float32)
    ) / np.asarray(stats["state"]["std"], np.float32)
    np.testing.assert_allclose(b._model.last_batch["observation.state"][0].numpy(), expected)


def test_right_yaw_adapter_serialization_is_arm_specific_and_strict(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "right.pt"
    stats = training_normalization(
        np.zeros((3, 8)), np.zeros((3, 7)), orientation=RIGHT_YAW_CONTRACT
    )
    fitted = {
        "weight": np.zeros((32, 480), np.float32),
        "bias": np.zeros(32, np.float32),
        "fit": {"horizon": 1},
    }
    save_native_adapter(path, "right", fitted, stats, {"train_seed": 10004})
    loaded = load_native_adapter(path, "right")
    assert loaded["orientation_contract"] == RIGHT_YAW_CONTRACT
    assert "yaw relative to pi" in loaded["action_contract"]
    with pytest.raises(ValueError, match="right arm"):
        save_native_adapter(tmp_path / "bad-left.pt", "left", fitted, stats, {})
    original = torch.load(path, map_location="cpu", weights_only=True)
    for name, value, message in (
        ("orientation_contract", "world_xyz_euler_v1", "differs from normalization"),
        ("weight", torch.zeros(31, 480), "output projection"),
        ("bias", torch.full((32,), float("nan")), "output projection"),
    ):
        torch.save({**original, name: value}, path)
        with pytest.raises(ValueError, match=message):
            load_native_adapter(path, "right")


def test_cross_arm_initializer_preserves_native_weights_lineage_and_cannot_deploy(
    monkeypatch, tmp_path
):
    import hashlib
    import json

    from relay_lab.cloth import action_dataset, smol_adaptation

    torch = pytest.importorskip("torch")
    native = {
        name: torch.zeros(shape) for name, shape in smol_adaptation.FINAL_PARAMETER_SHAPES.items()
    }
    source = tmp_path / "left.pt"
    fitted = {
        "weight": np.ones((32, 480), np.float32),
        "bias": np.zeros(32, np.float32),
        "fit": {"horizon": 1},
    }
    source_sha = save_native_adapter(
        source,
        "left",
        fitted,
        normalization(),
        {"train_seeds": [0, 10004, 10005, 10007], "validation_seeds": [10008]},
        native_parameters=native,
    )
    episode = tmp_path / "episode"
    episode.mkdir()
    manifest = {"seed": 10004, "split": "train", "success": True}
    (episode / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(action_dataset, "load_episode", lambda _: (manifest, {}))
    stats = training_normalization(
        np.zeros((3, 8)), np.zeros((3, 7)), orientation=RIGHT_YAW_CONTRACT
    )
    monkeypatch.setattr(smol_adaptation, "episode_normalization", lambda *a, **k: stats)
    path = tmp_path / "right-init.pt"
    report = prepare_right_initializer(source, episode, path, expected_sha256=source_sha)
    right = load_native_adapter(path, "right", expected_sha256=report["adapter_sha256"])
    assert right["fit"]["initialization_only"] is True
    assert right["training"]["cross_arm_source_adapter_sha256"] == source_sha
    assert right["training"]["adaptation_train_seeds"] == [0, 10004, 10005, 10007]
    assert right["training"]["initial_function_preserved"] is False
    assert torch.equal(right["weight"], torch.from_numpy(fitted["weight"]))
    assert all(
        torch.equal(value, right["native_parameters"][name]) for name, value in native.items()
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    with pytest.raises(ValueError, match="not a fitted deployment policy"):
        SmolNativePolicy("right", path, action_horizon=1)
    manifest["seed"] = 10008
    with pytest.raises(ValueError, match="cannot consume the source validation"):
        prepare_right_initializer(source, episode, tmp_path / "bad.pt")


def test_native_flow_label_and_euler_step_preserve_world_actions():
    rng = np.random.default_rng(7)
    actions = rng.normal(size=(50, 7)).astype(np.float32)
    noise = rng.normal(size=(50, 32)).astype(np.float32)
    stats = training_normalization(np.zeros((3, 8)), actions)
    velocity = velocity_labels(actions, stats, noise)
    generated = noise - velocity
    assert unnormalize(generated[:, :7], stats["action"]) == pytest.approx(actions, abs=1e-6)
    assert generated[:, 7:] == pytest.approx(np.zeros((50, 25)))


def test_native_predict_calls_full_chunk_api_and_observes_existing_head():
    model = backend()
    state = np.arange(8, dtype=np.float32)
    action, features = native_predict(
        model,
        np.zeros((16, 16, 3), dtype=np.uint8),
        state,
        "Fold the left sleeve",
        normalization(),
        capture_features=True,
    )
    assert model._model.calls == 1
    assert features.shape == (50, 480)
    assert action.shape == (50, 7)
    assert model._model.last_batch["observation.state"].numpy() == pytest.approx(state[None])
    assert len(model._model.model.action_out_proj._forward_pre_hooks) == 0
    first = action.copy()
    install_native_head(model, np.zeros((32, 480)), np.zeros(32))
    second, _ = native_predict(
        model, np.zeros((16, 16, 3), dtype=np.uint8), state, "Fold the left sleeve", normalization()
    )
    assert model._model.calls == 2
    assert not np.allclose(first, second)


def test_native_feature_hook_is_removed_on_model_failure():
    model = backend()

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    model._model.predict_action_chunk = fail
    with pytest.raises(RuntimeError, match="native failure"):
        native_predict(
            model,
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros(8),
            "Fold",
            normalization(),
            capture_features=True,
        )
    assert len(model._model.model.action_out_proj._forward_pre_hooks) == 0


def test_ridge_updates_existing_projection_shape_and_recovers_affine_flow():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(550, 480)).astype(np.float32)
    weight = rng.normal(size=(32, 480)).astype(np.float32) * 0.01
    labels = features @ weight.T + np.arange(32) * 0.01
    fit = fit_native_head(features, labels, ridge=1e-4)
    assert fit["weight"].shape == (32, 480)
    assert fit["bias"].shape == (32,)
    assert fit["fit"]["updated_parameter_count"] == 15392
    assert fit["fit"]["velocity_rmse"] < 1e-5


def test_native_adapter_checks_exact_bytes_arm_base_and_stats(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "adapter.pt"
    fitted = {
        "weight": np.zeros((32, 480), dtype=np.float32),
        "bias": np.zeros(32, dtype=np.float32),
        "fit": {"updated_parameter_count": 15392},
    }
    digest = save_native_adapter(
        path, "left", fitted, normalization(), {"train_seed": 10, "validation_seed": 20}
    )
    result = load_native_adapter(path, "left", expected_sha256=digest)
    assert result["native_module"] == "model.action_out_proj"
    with pytest.raises(ValueError, match="hash"):
        load_native_adapter(path, "left", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="incompatible"):
        load_native_adapter(path, "right")
    with pytest.raises(FileExistsError):
        save_native_adapter(path, "left", fitted, normalization(), {})


def test_native_adapter_rejects_invalid_shapes_or_denoising_contract():
    model = backend()
    model._config.num_steps = 10
    with pytest.raises(ValueError, match="exactly one"):
        native_predict(
            model, np.zeros((16, 16, 3), dtype=np.uint8), np.zeros(8), "Fold", normalization()
        )
    with pytest.raises(ValueError, match="nonfinite"):
        fit_native_head(np.zeros((2, 479)), np.zeros((2, 32)))
    with pytest.raises(ValueError):
        training_normalization(np.zeros((1, 8)), np.zeros((1, 7)))


def test_live_adapter_uses_native_actions_without_teacher_landmarks_and_clears_on_hold():
    policy = SmolNativePolicy.__new__(SmolNativePolicy)
    policy.agent, policy.action_horizon = "left", 1
    policy.backend = backend()
    policy.adapter = {"normalization": normalization(), "noise_seed": 2718}
    policy.calls, policy.seconds, policy._queue, policy._instruction = 0, 0, deque(), None
    observation = {
        "agent": "left",
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "ee_position": [0, 0, 0.2],
        "ee_rotation": np.eye(3),
        "gripper_closed": False,
        "grasped": False,
    }
    first = policy.act(observation, "approach_sleeve_above")
    second = policy.act(
        {**observation, "landmarks": {"sleeve": [99, 99, 99]}, "teacher_target": [99, 99, 99]},
        "approach_sleeve_above",
    )
    assert first == second
    assert policy.backend._model.calls == 2
    assert policy.last_prediction["source"] == "native_smolvla_flow"
    assert not policy.last_prediction["coordinator_hold"]
    held = policy.act(observation, "approach_sleeve_above", hold=True)
    assert held == {"target": pytest.approx([0, 0, 0.2]), "grip": False}
    assert policy.backend._model.calls == 2
    assert policy.last_prediction["coordinator_hold"]
    assert not policy._queue


def test_rejected_feature_cache_requires_identical_successful_prefix_before_rebinding(
    tmp_path, monkeypatch
):
    from relay_lab.cloth import action_dataset

    original, settled = tmp_path / "original", tmp_path / "settled"
    original.mkdir()
    settled.mkdir()
    (settled / "manifest.json").write_text('{"success":true}\n')
    old = {
        "tick": np.arange(2),
        "hold": np.array([False, False]),
        "rgb": np.zeros((2, 2, 2, 3)),
        "target": np.zeros((2, 3)),
    }
    new = {
        "tick": np.arange(3),
        "hold": np.array([False, False, True]),
        "rgb": np.zeros((3, 2, 2, 3)),
        "target": np.zeros((3, 3)),
    }

    def load(path, require_success=True):
        if path == original:
            assert not require_success
            return {"seed": 10004}, {"left": old}
        assert require_success
        return {"seed": 10004}, {"left": new}

    monkeypatch.setattr(action_dataset, "load_episode", load)
    assert len(verify_settled_rebind(original, settled, "left")) == 64
    new["target"][0, 0] = 0.01
    with pytest.raises(ValueError, match="changed recorded"):
        verify_settled_rebind(original, settled, "left")
    new["target"][0, 0] = 0
    new["hold"][-1] = False
    with pytest.raises(ValueError, match="uncached"):
        verify_settled_rebind(original, settled, "left")


def test_gradient_adapter_allows_only_pinned_native_projection_tensors():
    torch = pytest.importorskip("torch")
    payload = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in NATIVE_PARAMETER_SHAPES.items()
    }
    assert validate_native_parameters(payload) is payload
    with pytest.raises(ValueError, match="unapproved"):
        validate_native_parameters({**payload, "teacher.network.weight": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="invalid"):
        validate_native_parameters(
            {**payload, "state_proj.bias": torch.zeros(960, dtype=torch.float16)}
        )


def test_native_runtime_rejects_executing_unsupervised_chunk_tokens(monkeypatch):
    from relay_lab.cloth import smol_adaptation

    monkeypatch.setattr(
        smol_adaptation,
        "load_native_adapter",
        lambda *_args, **_kwargs: {
            "training": {"output_refit_horizon": 1},
            "fit": {},
        },
    )
    with pytest.raises(ValueError, match="supervised horizon"):
        SmolNativePolicy("left", "unused.pt", action_horizon=8)


def test_native_token0_repeat_uses_no_future_tokens_and_invalidates_on_phase_and_hold():
    policy = SmolNativePolicy.__new__(SmolNativePolicy)
    policy.agent, policy.action_horizon, policy.action_repeat = "left", 1, 4
    policy.backend = backend()
    policy.adapter = {"normalization": normalization(), "noise_seed": 2718}
    policy.calls, policy.seconds, policy._queue, policy._instruction = 0, 0, deque(), None
    observation = {
        "agent": "left",
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "ee_position": [0, 0, 0.2],
        "ee_rotation": np.eye(3),
        "gripper_closed": False,
        "grasped": False,
    }
    outputs = []
    for age in range(4):
        outputs.append(policy.act(observation, "approach_sleeve_above"))
        assert policy.last_prediction["used_index"] == 0
        assert policy.last_prediction["inference_age_ticks"] == age
    assert all(output == outputs[0] for output in outputs)
    assert policy.calls == 1
    policy.act(observation, "approach_sleeve_above")
    assert policy.calls == 2
    policy.act(observation, "approach_sleeve")
    assert policy.calls == 3
    assert policy.last_prediction["inference_age_ticks"] == 0
    policy.act(observation, "approach_sleeve", hold=True)
    assert not policy._queue
    policy.act(observation, "approach_sleeve")
    assert policy.calls == 4
