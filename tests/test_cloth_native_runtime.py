"""Native adapter artifact boundaries without allocating large models."""

import hashlib
import json

import pytest

from relay_lab.cloth import native_runtime
from relay_lab.cloth.runtime import ClothConfig


@pytest.fixture
def package(tmp_path):
    weights = b"verified-native-head"
    (tmp_path / "native_decoder.safetensors").write_bytes(weights)
    right = tmp_path / "adapter.json"
    right.write_text(json.dumps({"decoder_sha256": hashlib.sha256(weights).hexdigest()}))
    left = tmp_path / "left.pt"
    left.write_bytes(b"smol-native-head")
    paths = {"left": left, "right": right}
    hashes = {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in paths.items()}
    return paths, hashes


def test_right_decoder_tampering_fails_before_common_world_or_models(package, tmp_path, monkeypatch):
    paths, hashes = package
    (tmp_path / "native_decoder.safetensors").write_bytes(b"unapproved-head")
    called = []
    monkeypatch.setattr(native_runtime, "run_cloth", lambda *a, **kw: called.append(kw))
    with pytest.raises(ValueError, match="decoder does not match"):
        native_runtime.run_native_cloth(
            ClothConfig(), tmp_path / "run", checkpoint_paths=paths,
            expected_checkpoint_hashes=hashes,
        )
    assert not called


def test_manifest_change_is_not_accepted_using_same_head(package, tmp_path):
    paths, hashes = package
    metadata = json.loads(paths["right"].read_text())
    paths["right"].write_text(json.dumps({**metadata, "training": "different provenance"}))
    with pytest.raises(ValueError, match="manifest SHA-256"):
        native_runtime.run_native_cloth(
            ClothConfig(), tmp_path / "run", checkpoint_paths=paths,
            expected_checkpoint_hashes=hashes,
        )


def test_initial_cancellation_is_latched_and_never_uses_default_controller(tmp_path):
    checks = []

    def stop_once():
        checks.append(True)
        return len(checks) == 1

    result = native_runtime.run_native_cloth(
        ClothConfig(video=False), tmp_path / "run",
        checkpoint_paths={"left": "missing", "right": "missing"}, should_stop=stop_once,
    )
    assert result["status"] == "cancelled" and result["ticks"] == 0
    assert result["models"] == {} and len(checks) == 1


def test_right_snapshot_retains_captured_weights_after_source_replacement(package, tmp_path, monkeypatch):
    from relay_lab.cloth import xvla_adaptation

    paths, hashes = package
    calls = []

    def native(**kwargs):
        calls.append((kwargs, (kwargs["adapter"] / "native_decoder.safetensors").read_bytes()))
        return "actual-native-constructor"

    def common(config, output, **kwargs):
        output.mkdir()
        snapshot = output / "right.pt"
        snapshot.write_bytes(paths["right"].read_bytes())
        (tmp_path / "native_decoder.safetensors").write_bytes(b"later-replacement")
        return kwargs["policy_factory"]("right", snapshot)

    monkeypatch.setattr(xvla_adaptation, "NativeXVLA", native)
    monkeypatch.setattr(native_runtime, "run_cloth", common)
    result = native_runtime.run_native_cloth(
        ClothConfig(), tmp_path / "run", checkpoint_paths=paths,
        expected_checkpoint_hashes=hashes,
    )
    assert result == "actual-native-constructor"
    assert calls[0][1] == b"verified-native-head"
    assert calls[0][0]["expected_sha256"] == hashlib.sha256(calls[0][1]).hexdigest()


def test_left_snapshot_replacement_cannot_change_the_loaded_model(package, tmp_path, monkeypatch):
    paths, hashes = package

    def common(config, output, **kwargs):
        output.mkdir()
        snapshot = output / "left.pt"
        snapshot.write_bytes(b"changed-after-the-common-snapshot-check")
        return kwargs["policy_factory"]("left", snapshot)

    monkeypatch.setattr(native_runtime, "run_cloth", common)
    with pytest.raises(ValueError, match="hash does not match"):
        native_runtime.run_native_cloth(
            ClothConfig(), tmp_path / "run", checkpoint_paths=paths,
            expected_checkpoint_hashes=hashes,
        )


@pytest.mark.parametrize("horizon,repeat", [(8, 4), (1, True), (1, 0), (1, 51)])
def test_invalid_native_repetition_rejected_before_reading_weights(tmp_path, horizon, repeat):
    with pytest.raises(ValueError, match="repetition"):
        native_runtime.run_native_cloth(
            ClothConfig(), tmp_path / "run",
            checkpoint_paths={"left": "missing", "right": "missing"},
            action_horizon=horizon, action_repeat=repeat,
        )


@pytest.mark.parametrize("overrides", [{}, {"action_horizon": 1, "action_repeat": 4}])
def test_token_zero_repetition_is_explicit_for_both_models(package, tmp_path, monkeypatch, overrides):
    from relay_lab.cloth import smol_adaptation, xvla_adaptation

    paths, hashes = package
    calls = {}
    monkeypatch.setattr(smol_adaptation, "SmolNativePolicy", lambda *args, **kwargs: calls.update(left=kwargs))
    monkeypatch.setattr(xvla_adaptation, "NativeXVLA", lambda **kwargs: calls.update(right=kwargs))

    def common(config, output, **kwargs):
        output.mkdir()
        for arm in ("left", "right"):
            snapshot = output / f"{arm}.pt"
            snapshot.write_bytes(paths[arm].read_bytes())
            kwargs["policy_factory"](arm, snapshot)

    monkeypatch.setattr(native_runtime, "run_cloth", common)
    native_runtime.run_native_cloth(
        ClothConfig(), tmp_path / "run", checkpoint_paths=paths,
        expected_checkpoint_hashes=hashes, **overrides,
    )
    assert calls["left"]["action_horizon"] == 1
    assert calls["left"]["action_repeat"] == 4
    assert calls["right"]["action_horizon"] == 4


def test_smol_pair_loads_distinct_arm_adapters_with_their_approved_hashes(package, tmp_path, monkeypatch):
    from relay_lab.cloth import smol_adaptation

    paths, hashes = package
    calls = []

    def policy(arm, snapshot, **kwargs):
        calls.append((arm, snapshot.read_bytes(), kwargs))
        return object()

    def common(config, output, **kwargs):
        assert kwargs["expected_checkpoint_hashes"] == hashes
        output.mkdir()
        for arm in ("left", "right"):
            snapshot = output / f"{arm}.pt"
            snapshot.write_bytes(paths[arm].read_bytes())
            kwargs["policy_factory"](arm, snapshot)

    monkeypatch.setattr(smol_adaptation, "SmolNativePolicy", policy)
    monkeypatch.setattr(native_runtime, "run_cloth", common)
    native_runtime.run_smol_pair(
        ClothConfig(), tmp_path / "run", checkpoint_paths=paths,
        expected_checkpoint_hashes=hashes,
    )
    assert [call[0] for call in calls] == ["left", "right"]
    for arm, content, options in calls:
        assert content == paths[arm].read_bytes()
        assert options == {
            "device": "mps", "action_horizon": 1, "action_repeat": 4,
            "expected_sha256": hashes[arm],
        }


def test_smol_pair_cancelled_before_models_or_world(tmp_path):
    result = native_runtime.run_smol_pair(
        ClothConfig(video=False), tmp_path / "run",
        checkpoint_paths={"left": "missing", "right": "missing"},
        should_stop=lambda: True,
    )
    assert result["status"] == "cancelled" and result["models"] == {}
    assert result["ticks"] == 0


def test_smol_pair_rejects_unapproved_bytes_before_models_or_world(package, tmp_path):
    paths, hashes = package
    paths["right"].write_bytes(b"unapproved replacement")
    result = native_runtime.run_smol_pair(
        ClothConfig(video=False), tmp_path / "run", checkpoint_paths=paths,
        expected_checkpoint_hashes=hashes,
    )
    assert result["status"] == "failed" and result["models"] == {}
    assert result["ticks"] == 0 and "right checkpoint SHA-256" in result["error"]


def test_smol_pair_cli_binds_snapshots_without_explicit_approvals(package, tmp_path, monkeypatch):
    paths, hashes = package

    def common(config, output, **kwargs):
        assert kwargs["expected_checkpoint_hashes"] == hashes
        output.mkdir()
        snapshot = output / "left.pt"
        snapshot.write_bytes(b"replaced snapshot after capture")
        return kwargs["policy_factory"]("left", snapshot)

    monkeypatch.setattr(native_runtime, "run_cloth", common)
    with pytest.raises(ValueError, match="hash does not match"):
        native_runtime.run_smol_pair(
            ClothConfig(), tmp_path / "run", checkpoint_paths=paths,
        )
