"""Mocked collection contracts; these tests never load a VLA or create a world."""

import gzip
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth import mixed_dataset as mixed


def _write(path, value):
    path.write_text(json.dumps(value))


def _rebind(path, name):
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"][name] = mixed._sha(path / name)
    _write(path / "manifest.json", manifest)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from relay_lab.cloth import policies, smol_adaptation

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "right.pt").write_bytes(b"original-right-teacher")
    left = checkpoints / "smol-v3.pt"
    left.write_bytes(b"approved-smol-v3")
    state = SimpleNamespace(
        left=left,
        left_hash=mixed._sha(left),
        calls=[],
        observed=[],
        replay_calls=0,
        success=True,
        pending=False,
        replay_verified=True,
        mismatched_replay=False,
        corrupt_label=False,
        missing_source=False,
        corrupt_final=False,
        corrupt_replay_metrics=False,
    )

    class Smol:
        def __init__(self, arm, snapshot, **kwargs):
            assert arm == "left"
            assert kwargs == {
                "device": "mps",
                "expected_sha256": state.left_hash,
                "action_horizon": 1,
                "action_repeat": 4,
            }
            assert snapshot.read_bytes() == b"approved-smol-v3"
            state.calls.append("left_constructor")
            self.model_info = {
                "foundation_vla": True,
                "adapter_format": mixed.SMOL_FORMAT,
                "checkpoint_sha256": state.left_hash,
                "action_horizon": 1,
                "action_repeat_steps": 4,
            }

    class Teacher:
        def __init__(self, arm, snapshot):
            assert arm == "right"
            self.model_info = {
                "foundation_vla": False,
                "checkpoint_sha256": mixed._sha(snapshot),
            }
            self.torch = SimpleNamespace(
                set_num_threads=lambda value: state.calls.append(("cpu_threads", value))
            )
            self.last_prediction = {}
            state.calls.append("teacher_constructor")

        def act(self, observation, instruction, *, clearance, hold):
            assert observation["landmarks"]["private"] == [99, 99, 99]
            state.observed.append(observation)
            self.last_prediction = {"coordinator_hold": hold}
            return {"target": [0.15, 0.1, 0.02], "grip": False}

    def runner(
        config,
        output,
        *,
        checkpoint_paths,
        expected_checkpoint_hashes,
        policy_factory,
        should_stop,
        on_progress,
    ):
        output.mkdir()
        (output / "models").mkdir()
        for arm in ("left", "right"):
            contents = checkpoint_paths[arm].read_bytes()
            assert hashlib.sha256(contents).hexdigest() == expected_checkpoint_hashes[arm]
            (output / "models" / f"{arm}.pt").write_bytes(contents)
        (output / "models/communication.json").write_text("{}")
        left_policy = policy_factory("left", output / "models/left.pt")
        right_policy = policy_factory("right", output / "models/right.pt")
        rows = []
        for tick in range(6):
            phase_index = 0 if tick < 2 else 2 if state.pending and tick == 5 else 1
            observation = {
                "tick": tick,
                "agent": "right",
                "rgb": np.full((128, 128, 3), tick, np.uint8),
                "ee_position": [0.2, 0.1, 0.02],
                "ee_rotation": np.eye(3).tolist(),
                "grasped": False,
                "gripper_closed": False,
                "landmarks": {"private": [99, 99, 99]},
            }
            action = right_policy.act(
                observation, mixed.PHASES[phase_index], clearance=0.2, hold=False
            )
            if state.pending and tick == 5:
                break
            if state.corrupt_label and tick == 0:
                action = {**action, "target": [0.3, 0.1, 0.02]}
            after_phase = 1 if tick >= 1 else 0
            if state.pending and tick == 4:
                after_phase = 2
            rows.append(
                {
                    "decision_tick": tick,
                    "tick": tick + 1,
                    "phase": {"right": mixed.PHASES[after_phase]},
                    "observations": {
                        "right": {key: value for key, value in observation.items() if key != "rgb"}
                    },
                    "actions": {"right": action},
                    "predictions": {"right": dict(right_policy.last_prediction)},
                    "decisions": {"right": {"hold": False}},
                }
            )
        summary = {
            "config": asdict(config),
            "ticks": len(rows),
            "models": {"left": left_policy.model_info, "right": right_policy.model_info},
            "status": "cancelled"
            if state.pending
            else "succeeded"
            if state.success
            else "exhausted",
            "success": state.success and not state.pending,
            "metrics": {"success": state.success and not state.pending, "physics_valid": True},
            "error": None,
        }
        _write(output / "summary.json", summary)
        _write(output / "config.json", asdict(config))
        final_metrics = dict(summary["metrics"])
        if state.corrupt_final:
            final_metrics["success"] = False
        _write(output / "final.json", {"tick": len(rows), "metrics": final_metrics})
        with gzip.open(output / "events.jsonl.gz", "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        if state.missing_source:
            (output / "models/communication.json").unlink()
        return summary

    def replay(source, output, should_stop):
        state.replay_calls += 1
        output.mkdir()
        result = {
            "kind": "independent_mujoco_action_reexecution",
            "verified": state.replay_verified,
            "source_artifact_sha256": {
                name: mixed._sha(source / name)
                for name in ("summary.json", "events.jsonl.gz", "final.json")
            },
            "final_vertices_bitwise_equal": True,
            "same_final_metrics": True,
            "same_fixture_state": True,
            "same_tick_count": True,
            "steps": 5 if state.pending else 6,
            "reexecutes_neural_models": False,
            "final_metrics": json.loads((source / "final.json").read_text())["metrics"],
        }
        if state.corrupt_replay_metrics:
            result["final_metrics"]["success"] = False
        if state.mismatched_replay:
            result["source_artifact_sha256"]["events.jsonl.gz"] = "0" * 64
        _write(output / "verification.json", result)
        return result

    monkeypatch.setattr(mixed, "DEFAULT_DIRECTORY", checkpoints)
    monkeypatch.setattr(smol_adaptation, "SmolNativePolicy", Smol)
    monkeypatch.setattr(policies, "ClothPolicy", Teacher)
    monkeypatch.setattr(mixed, "run_cloth", runner)
    monkeypatch.setattr(mixed, "_verify_physics", replay)
    monkeypatch.setattr(mixed, "_require_offline_assets", lambda: None)
    state.collect = lambda output, **kwargs: mixed.collect_mixed_episode(
        output, left_adapter=state.left, expected_left_sha256=state.left_hash, **kwargs
    )
    return state


def test_mixed_collection_is_right_only_imitation_with_actual_labels(rig, tmp_path):
    output = tmp_path / "mixed"
    result = rig.collect(output)
    manifest, agents = mixed.load_mixed_episode(output)
    assert manifest == result
    assert manifest["success"] and manifest["source_success"]
    assert manifest["two_vla_success"] is False
    assert manifest["future_action_chunks_available"] is False
    assert manifest["seed"] == 10007
    assert manifest["source_models"]["left"]["foundation_vla"] is True
    assert manifest["source_models"]["right"]["foundation_vla"] is False
    assert rig.replay_calls == 1
    assert rig.calls == ["left_constructor", "teacher_constructor", ("cpu_threads", 1)]
    assert set(agents) == {"right"}
    assert set(agents["right"]) == mixed.LEARNER_KEYS
    assert agents["right"]["tick"].tolist() == [0, 2, 4]
    assert agents["right"]["target"].tolist() == [[0.15, 0.1, 0.02]] * 3
    assert agents["right"]["rgb"][:, 0, 0, 0].tolist() == [0, 2, 4]


def test_unsuccessful_raw_source_is_preserved_and_rejected_by_default(rig, tmp_path):
    rig.success = False
    output = tmp_path / "failure"
    result = rig.collect(output)
    assert result["source_status"] == "exhausted"
    assert result["source_success"] is False and result["success"] is False
    assert rig.replay_calls == 0
    assert json.loads((output / "source/summary.json").read_text())["status"] == "exhausted"
    with pytest.raises(ValueError, match="verified physical folding"):
        mixed.load_mixed_episode(output)
    _, agents = mixed.load_mixed_episode(output, require_success=False)
    assert len(agents["right"]["tick"]) == 3


@pytest.mark.parametrize("failure", ["replay_verified", "mismatched_replay"])
def test_independent_proof_is_required_for_success(rig, tmp_path, failure):
    setattr(rig, failure, failure != "replay_verified")
    output = tmp_path / failure
    result = rig.collect(output)
    assert result["source_status"] == "succeeded" and result["source_success"] is True
    assert result["success"] is False
    assert "independent physics proof" in result["error"]
    with pytest.raises(ValueError, match="verified physical folding"):
        mixed.load_mixed_episode(output)


def test_success_summary_cannot_override_failed_measured_physics(rig, tmp_path):
    rig.corrupt_final = True
    output = tmp_path / "fabricated-summary"
    result = rig.collect(output)
    assert result["source_status"] == "succeeded"  # Retain the source's claim.
    assert result["success"] is False
    assert "summary differs from measured final physics" in result["error"]
    assert rig.replay_calls == 0
    with pytest.raises(ValueError, match="verified physical folding"):
        mixed.load_mixed_episode(output)


def test_replay_must_report_the_same_successful_physical_metrics(rig, tmp_path):
    rig.corrupt_replay_metrics = True
    result = rig.collect(tmp_path / "wrong-replay-metrics")
    assert result["success"] is False
    assert "independent physics proof" in result["error"]


def test_loader_rejects_rebound_summary_and_final_physics_mismatch(rig, tmp_path):
    output = tmp_path / "rebound-final"
    rig.collect(output)
    final = json.loads((output / "source/final.json").read_text())
    final["metrics"]["success"] = False
    _write(output / "source/final.json", final)
    _rebind(output, "source/final.json")
    with pytest.raises(ValueError, match="summary differs from measured final physics"):
        mixed.load_mixed_episode(output)


def test_unexecuted_prediction_after_cancellation_is_never_a_label(rig, tmp_path):
    rig.pending = True
    output = tmp_path / "pending"
    result = rig.collect(output)
    assert result["source_status"] == "cancelled" and result["success"] is False
    _, arrays = mixed.load_mixed_episode(output, require_success=False)
    assert arrays["right"]["tick"].tolist() == [0, 2, 4]
    proof = json.loads((output / "verification.json").read_text())
    assert proof["discarded_unexecuted_candidates"] == 1


def test_label_must_match_actual_executed_command(rig, tmp_path):
    rig.corrupt_label = True
    result = rig.collect(tmp_path / "wrong-command")
    assert result["success"] is False
    assert "actual executed teacher command" in result["error"]
    assert rig.replay_calls == 0


def test_incomplete_source_preserves_raw_outcome_without_claiming_verification(rig, tmp_path):
    rig.missing_source = True
    output = tmp_path / "incomplete"
    result = rig.collect(output)
    assert result["source_status"] == "succeeded"
    assert result["success"] is False and result["error"]
    assert (output / "source/summary.json").exists()


@pytest.mark.parametrize("max_ticks", [0, 1801, True])
def test_budget_is_bounded_before_any_model_load(rig, tmp_path, max_ticks):
    output = tmp_path / "over-budget"
    with pytest.raises(ValueError, match="1800"):
        rig.collect(output, max_ticks=max_ticks)
    assert not output.exists() and rig.calls == []


def test_cancel_before_creation_does_not_load_models(rig, tmp_path):
    output = tmp_path / "cancelled"
    with pytest.raises(RuntimeError, match="cancelled"):
        rig.collect(output, should_stop=lambda: True)
    assert not output.exists() and rig.calls == []


def test_changed_candidate_fails_before_model_loading(rig, tmp_path):
    rig.left.write_bytes(b"unapproved")
    with pytest.raises(ValueError, match="immutable candidate"):
        rig.collect(tmp_path / "bad-pin")
    assert rig.calls == []


def test_online_model_loading_is_rejected_before_creating_output(rig, tmp_path, monkeypatch):
    def online():
        raise RuntimeError("Start mixed collection with HF_HUB_OFFLINE=1")

    monkeypatch.setattr(mixed, "_require_offline_assets", online)
    output = tmp_path / "online"
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"):
        rig.collect(output)
    assert not output.exists() and rig.calls == []


def test_loader_rejects_changed_pixels_or_privileged_input_columns(rig, tmp_path):
    output = tmp_path / "tampered"
    rig.collect(output)
    with np.load(output / "right.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["rgb"][0, 0, 0, 0] = 255
    np.savez_compressed(output / "right.npz", **arrays)
    _rebind(output, "right.npz")
    with pytest.raises(ValueError, match="actual executed teacher command/input"):
        mixed.load_mixed_episode(output)
    arrays["landmarks"] = np.zeros((3, 3))
    np.savez_compressed(output / "right.npz", **arrays)
    _rebind(output, "right.npz")
    with pytest.raises(ValueError, match="privileged state"):
        mixed.load_mixed_episode(output)


def test_manifest_cannot_relabel_a_mixed_run_as_two_vla_success(rig, tmp_path):
    output = tmp_path / "wrong-claim"
    rig.collect(output)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["two_vla_success"] = True
    _write(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="right-only"):
        mixed.load_mixed_episode(output)


def test_loader_requires_actual_source_identity_not_only_recorded_hashes(rig, tmp_path):
    output = tmp_path / "wrong-controller"
    rig.collect(output)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["approved_checkpoint_sha256"]["left"] = "0" * 64
    _write(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="controller identities"):
        mixed.load_mixed_episode(output)
