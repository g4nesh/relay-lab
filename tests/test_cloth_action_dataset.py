"""Input isolation, physics/action alignment and native action conventions."""

import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth import action_dataset as dataset


def episode_arrays():
    count = 5
    return {
        "rgb": np.zeros((count, 16, 16, 3), dtype=np.uint8),
        "ee_position": np.zeros((count, 3), dtype=np.float32),
        "ee_rotation": np.repeat(np.eye(3)[None], count, axis=0),
        "grasped": np.zeros(count, dtype=bool),
        "gripper_closed": np.zeros(count, dtype=bool),
        "phase": np.asarray([0, 0, 1, 1, 1]),
        "hold": np.asarray([False, False, False, True, True]),
        "target": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "grip": np.asarray([False, False, True, True, True]),
        "tick": np.arange(count),
        "post_ee_position": np.ones((count, 3), dtype=np.float32),
        "executed_target": np.ones((count, 3), dtype=np.float32),
    }


def write_episode(path, arrays=None, **overrides):
    path.mkdir()
    arrays = episode_arrays() if arrays is None else arrays
    files = {}
    for agent in dataset.AGENTS:
        file = path / f"{agent}.npz"
        np.savez_compressed(file, **arrays)
        files[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {
        "schema": dataset.SCHEMA,
        "split": "train",
        "seed": 10004,
        "success": True,
        "ticks": len(arrays["tick"]),
        "image_size": 16,
        "files": files,
        **overrides,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_smol_actions_are_absolute_metres_and_signed_gripper():
    target = [0.21, -0.19, 0.05]
    rotation = np.diag([1.0, -1.0, -1.0])
    closed = dataset.smol_action(target, rotation, True)
    opened = dataset.smol_action(target, rotation, False)
    np.testing.assert_allclose(closed, [*target, np.pi, 0, 0, 1])
    assert opened[-1] == -1
    np.testing.assert_allclose(dataset.smol_state(target, rotation, True)[-2:], [0.012, -0.012])
    np.testing.assert_allclose(dataset.smol_state(target, rotation, False)[-2:], [0.022, -0.022])


def test_rotation_uses_measured_frame_and_rejects_invalid_matrices():
    quarter_turn = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.testing.assert_allclose(dataset.rotation_xyz(quarter_turn), [0, 0, np.pi / 2])
    with pytest.raises(ValueError, match="proper rotation"):
        dataset.rotation_xyz(np.diag([1, 1, -1]))
    with pytest.raises(ValueError, match="rotation"):
        dataset.rotation_xyz(np.full((3, 3), np.nan))


def test_xvla_never_supplies_peer_pose_or_peer_training_labels():
    state = dataset.xvla_state([0.2, 0.1, 0.3], np.eye(3), True, "right")
    assert np.array_equal(state[:10], np.zeros(10))
    np.testing.assert_allclose(state[10:], [0.2, 0.1, 0.3, 1, 0, 0, 1, 0, 0, 1])
    action, mask = dataset.xvla_action([0.3, 0.2, 0.1], np.eye(3), False, "right")
    assert set(np.flatnonzero(mask)) == {10, 11, 12, 19}
    assert action[19] == 0
    assert not mask[:10].any()


def test_future_chunks_are_pre_step_and_do_not_cross_instructions_or_holds():
    arrays = episode_arrays()
    chunk = dataset.future_chunk(arrays, 0, 4, family="smolvla", agent="left")
    assert chunk["indices"].tolist() == [0, 1, 1, 1]
    assert chunk["action_is_pad"].tolist() == [False, False, True, True]
    np.testing.assert_array_equal(chunk["actions"][:2, :3], arrays["target"][:2])
    hold_boundary = dataset.future_chunk(arrays, 2, 3, family="xvla", agent="right")
    assert hold_boundary["indices"].tolist() == [2, 2, 2]
    assert hold_boundary["action_is_pad"].tolist() == [False, True, True]
    np.testing.assert_array_equal(hold_boundary["actions"][0, 10:13], arrays["target"][2])
    tail = dataset.future_chunk(arrays, 4, 3, family="smolvla", agent="left")
    assert tail["indices"].tolist() == [4, 4, 4]
    assert tail["action_is_pad"].tolist() == [False, True, True]


def test_chunk_will_not_hide_a_missing_physics_step():
    arrays = episode_arrays()
    arrays["tick"][1] = 4
    chunk = dataset.future_chunk(arrays, 0, 4, family="smolvla", agent="left")
    assert chunk["action_is_pad"].tolist() == [False, True, True, True]


@pytest.mark.parametrize("seed", [103, 9999, -1, True, 2**32])
def test_collector_rejects_evaluation_seed_before_creating_output(tmp_path, seed):
    output = tmp_path / "never_created"
    with pytest.raises(ValueError, match="training seeds"):
        dataset.collect_episode(output, seed=seed)
    assert not output.exists()


def test_loader_verifies_array_digest_and_preserves_temporal_order(tmp_path):
    path = write_episode(tmp_path / "episode")
    manifest, agents = dataset.load_episode(path)
    assert manifest["seed"] == 10004
    assert agents["left"]["tick"].tolist() == list(range(5))
    with (path / "right.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest changed"):
        dataset.load_episode(path)


@pytest.mark.parametrize("kind", ["landmarks", "skipped_tick", "nonfinite"])
def test_loader_rejects_privileged_or_misaligned_inputs(tmp_path, kind):
    arrays = episode_arrays()
    if kind == "landmarks":
        arrays["landmarks"] = np.zeros((5, 3))
    elif kind == "skipped_tick":
        arrays["tick"][3] += 1
    else:
        arrays["target"][0, 0] = np.nan
    path = write_episode(tmp_path / "episode", arrays)
    with pytest.raises(ValueError):
        dataset.load_episode(path)


def test_failed_teacher_rollout_is_not_accepted_as_successful_training(tmp_path):
    path = write_episode(tmp_path / "episode", success=False)
    with pytest.raises(ValueError, match="did not complete"):
        dataset.load_episode(path)
    manifest, _ = dataset.load_episode(path, require_success=False)
    assert manifest["success"] is False


def test_instruction_matches_arm_and_phase_without_hidden_state():
    assert dataset.phase_instruction("right", "grasp_hem") == (
        "Fold the T-shirt. Right arm: close the gripper on the hem tab."
    )
    with pytest.raises(ValueError, match="unknown"):
        dataset.phase_instruction("peer", "grasp_hem")


def test_local_proprio_does_not_confuse_commanded_closure_with_attachment():
    local_world = SimpleNamespace(
        _sites={"left": 0, "right": 1},
        data=SimpleNamespace(site_xmat=np.repeat(np.eye(3).reshape(1, 9), 2, axis=0)),
        _grip={"left": True, "right": False},
        holding={"left": None, "right": 17},
        ee=lambda agent: np.asarray([-0.2 if agent == "left" else 0.2, 0, 0.1]),
    )
    local = dataset.local_proprio(local_world, "left")
    assert set(local) == {"ee_position", "ee_rotation", "gripper_closed", "grasped"}
    assert local["gripper_closed"] is True
    assert local["grasped"] is False
    assert local["ee_position"][0] == -0.2


def test_immutable_collection_never_overwrites_an_existing_episode(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        dataset.collect_episode(output, seed=10004)
    assert marker.read_text() == "original"


@pytest.mark.parametrize("success_tick", [2, 5])
def test_collector_aligns_camera_pose_and_command_before_the_actual_step(
    tmp_path, monkeypatch, success_tick
):
    from relay_lab.cloth import communication, policies, runtime, world

    executed = []

    class World:
        def __init__(self, **kwargs):
            self.tick = 0
            self.fixture_active = True
            self._sites = {"left": 0, "right": 1}
            self.data = SimpleNamespace(site_xmat=np.repeat(np.eye(3).reshape(1, 9), 2, axis=0))
            self._targets = {a: np.zeros(3) for a in dataset.AGENTS}
            self._grip = dict.fromkeys(dataset.AGENTS, False)
            self.holding = dict.fromkeys(dataset.AGENTS, None)

        def snapshot(self):
            return {"metrics": self.metrics(), "physics_config": {}, "vertices": []}

        def metrics(self):
            return {"success": self.tick == success_tick}

        def ee(self, agent):
            return np.asarray([self.tick, 0, 0], dtype=float)

        def observe(self, agent):
            return {
                "agent": agent,
                "tick": self.tick,
                "ee_position": self.ee(agent),
                "grasped": bool(self._grip[agent]),
            }

        def render(self, agent, width, height):
            return np.full((height, width, 3), self.tick, dtype=np.uint8)

        def step(self, actions):
            executed.append(actions)
            for agent, action in actions.items():
                self._targets[agent] = np.asarray(action["target"])
                self._grip[agent] = action["grip"]
            self.tick += 1

        def release_fixture(self):
            self.fixture_active = False

        def close(self):
            pass

    class Policy:
        def __init__(self, agent):
            self.last_prediction = {"coordinator_hold": False}
            self.model_info = {"foundation_vla": False, "agent": agent}
            self.torch = SimpleNamespace(set_num_threads=lambda _: None)

        def act(self, observation, phase, **kwargs):
            return {"target": [observation["tick"] + 10, 0, 0], "grip": True}

    class Controller:
        phase, done = "settle", False

        def __init__(self, *args):
            pass

        def decision(self, *args, **kwargs):
            return {"clearance": 0.23, "hold": False}

        def update(self, before, after, *args):
            self.done = after["tick"] >= 2

    class Channel:
        def __init__(self, *args, **kwargs):
            pass

        def publish(self, *args):
            pass

        def advance(self, *args):
            pass

        def receive(self, *args):
            return None

    monkeypatch.setattr(world, "ClothWorld", World)
    monkeypatch.setattr(policies, "ClothPolicy", Policy)
    monkeypatch.setattr(runtime, "LocalPhaseController", Controller)
    monkeypatch.setattr(runtime, "PhaseJudge", lambda _: None)
    monkeypatch.setattr(communication, "Channel", Channel)
    monkeypatch.setattr(communication, "observation_features", lambda _: {})
    monkeypatch.setattr(communication.LearnedFeatureCodec, "load", lambda _: None)
    path = tmp_path / "real_alignment"
    result = dataset.collect_episode(path, seed=10004, max_ticks=success_tick, image_size=16)
    assert result["success"]
    assert result["terminal_settle_ticks"] == success_tick - 2
    _, agents = dataset.load_episode(path)
    for agent, arrays in agents.items():
        assert arrays["tick"].tolist() == list(range(success_tick))
        assert arrays["rgb"][:, 0, 0, 0].tolist() == list(range(success_tick))
        assert arrays["ee_position"][:, 0].tolist() == list(range(success_tick))
        assert arrays["post_ee_position"][:, 0].tolist() == list(range(1, success_tick + 1))
        assert arrays["gripper_closed"].tolist() == [False] + [True] * (success_tick - 1)
        assert arrays["grip"].all()
        assert arrays["target"].tolist() == [row[agent]["target"] for row in executed]
        assert arrays["target"][:, 0].tolist() == list(range(10, 10 + success_tick))
    assert "landmarks" not in agents["left"]


def test_terminal_extension_requires_exact_original_physics_before_reusing_rgb(
    tmp_path, monkeypatch
):
    from relay_lab.cloth import world

    arrays = episode_arrays()
    arrays["phase"][:] = len(dataset.PHASES) - 1
    source = write_episode(
        tmp_path / "source",
        arrays,
        success=False,
        scenario="delayed",
        metrics={"physics_valid": True},
    )
    for name in ("initial.json", "final.json"):
        (source / name).write_text(json.dumps({"vertices": [[0.0, 0.0, 0.0]]}))
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update(
        {
            "initial_sha256": dataset._sha(source / "initial.json"),
            "final_sha256": dataset._sha(source / "final.json"),
        }
    )
    (source / "manifest.json").write_text(json.dumps(manifest))
    original_bytes = (source / "manifest.json").read_bytes()

    class DivergentWorld:
        vertices = np.asarray([[0.0, 0.0, 0.001]])

        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(world, "ClothWorld", DivergentWorld)
    output = tmp_path / "never_success"
    with pytest.raises(ValueError, match="initial cloth mesh changed"):
        dataset.extend_settling(source, output)
    assert (source / "manifest.json").read_bytes() == original_bytes
    assert not (output / "manifest.json").exists()
    assert json.loads((output / "prefix-verification.json").read_text())["verified"] is False


@pytest.fixture
def correction_source(tmp_path, monkeypatch):
    from relay_lab.cloth import policies, world
    from relay_lab.smolvla import DEFAULT_MODEL_ID

    executed = []

    class World:
        def __init__(self, **kwargs):
            self.tick = 0
            self.fixture_active = True
            self._sites = {"left": 0, "right": 1}
            self.data = SimpleNamespace(site_xmat=np.repeat(np.eye(3).reshape(1, 9), 2, axis=0))
            self._grip = dict.fromkeys(dataset.AGENTS, False)
            self.holding = dict.fromkeys(dataset.AGENTS, None)

        @property
        def vertices(self):
            return np.asarray([[self.tick, 0, 0]], dtype=float)

        def ee(self, agent):
            return np.asarray([self.tick, 0, 0], dtype=float)

        def snapshot(self):
            return {
                "tick": self.tick,
                "vertices": self.vertices.tolist(),
                "robots": {agent: {"ee": self.ee(agent).tolist()} for agent in dataset.AGENTS},
                "metrics": {"success": False, "physics_valid": True},
            }

        def observe(self, agent, rgb=False):
            result = {
                "agent": agent,
                "tick": self.tick,
                "ee_position": self.ee(agent).tolist(),
                "grasped": False,
                "landmarks": {"private": [99, 99, 99]},
            }
            if rgb:
                result["rgb"] = np.full((128, 128, 3), self.tick, dtype=np.uint8)
            return result

        def step(self, actions):
            assert actions["left"]["target"] == [self.tick + 1, 0, 0]
            executed.append(actions)
            self.tick += 1
            return self.snapshot()

        def close(self):
            pass

    class Teacher:
        def __init__(self, agent):
            self.torch = SimpleNamespace(set_num_threads=lambda _: None)
            self.model_info = {"foundation_vla": False, "agent": agent}
            self.last_prediction = {"coordinator_hold": False}

        def act(self, observation, instruction, **kwargs):
            assert observation["landmarks"]["private"] == [99, 99, 99]
            return {"target": [9.0, 9.0, 9.0], "grip": True}

    source = tmp_path / "native_failure"
    source.mkdir()
    source_world = World()
    summary = {
        "config": {"seed": 0, "scenario": "delayed", "fixture": True},
        "models": {
            "left": {"foundation_vla": True, "inference_calls": 2, "model_id": DEFAULT_MODEL_ID}
        },
        "status": "exhausted",
        "success": False,
        "ticks": 2,
        "initial_physics_sha256": hashlib.sha256(source_world.vertices.tobytes()).hexdigest(),
    }
    records = []
    for tick in range(2):
        actions = {agent: {"target": [tick + 1, 0, 0], "grip": False} for agent in dataset.AGENTS}
        pixels = source_world.observe("left", rgb=True)["rgb"]
        snapshot = source_world.step(actions)
        records.append(
            {
                "decision_tick": tick,
                "tick": tick + 1,
                "actions": actions,
                "predictions": {
                    "left": {
                        "native_action_executed": True,
                        "coordinator_hold": False,
                        "inference_age_ticks": 0,
                        "inference_id": tick + 1,
                        "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                        "model_language_prompt": dataset.phase_instruction(
                            "left", dataset.PHASES[0]
                        )
                        + "\n",
                    }
                },
                "decisions": {"left": {"clearance": 0.23}},
                "events": [],
                "phase": {"left": dataset.PHASES[0]},
                "robots": snapshot["robots"],
                "metrics": snapshot["metrics"],
            }
        )
    (source / "summary.json").write_text(json.dumps(summary))
    (source / "config.json").write_text(json.dumps(summary["config"]))
    (source / "final.json").write_text(json.dumps(source_world.snapshot()))
    with gzip.open(source / "events.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(record) + "\n" for record in records))
    executed.clear()
    monkeypatch.setattr(world, "ClothWorld", World)
    monkeypatch.setattr(policies, "ClothPolicy", Teacher)
    return SimpleNamespace(source=source, executed=executed, records=records, summary=summary)


def test_dagger_labels_never_control_the_world_or_imply_future_actions(correction_source, tmp_path):
    output = tmp_path / "corrections"
    manifest = dataset.collect_corrections(correction_source.source, output)
    assert manifest["replay_verified"] is True
    assert manifest["source_success"] is False
    assert manifest["teacher"]["actions_executed"] == 0
    assert manifest["future_action_chunks_available"] is False
    assert len(correction_source.executed) == 2
    _, agents = dataset.load_corrections(output)
    assert agents["left"]["tick"].tolist() == [0, 1]
    assert agents["left"]["target"].tolist() == [[9, 9, 9], [9, 9, 9]]
    assert agents["left"]["ee_position"][:, 0].tolist() == [0, 1]
    assert "landmarks" not in agents["left"]
    assert "right" not in agents
    with pytest.raises(ValueError, match="not a native action training episode"):
        dataset.load_episode(output)


def test_dagger_rejects_camera_mismatch_before_native_replay(correction_source, tmp_path):
    correction_source.records[0]["predictions"]["left"]["rgb_sha256"] = "0" * 64
    with gzip.open(correction_source.source / "events.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in correction_source.records))
    output = tmp_path / "rejected_corrections"
    with pytest.raises(ValueError, match="camera differs"):
        dataset.collect_corrections(correction_source.source, output)
    assert correction_source.executed == []
    assert not (output / "manifest.json").exists()


def test_dagger_rejects_benchmark_seeds_before_creating_output(correction_source, tmp_path):
    correction_source.summary["config"]["seed"] = 103
    (correction_source.source / "summary.json").write_text(json.dumps(correction_source.summary))
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="benchmark evaluation"):
        dataset.collect_corrections(correction_source.source, output)
    assert not output.exists()


@pytest.fixture(params=("xvla", "smol"))
def right_correction_source(correction_source, monkeypatch, request):
    from relay_lab.cloth import native_audit
    from relay_lab.cloth.pretrained import XVLA_ID
    from relay_lab.smolvla import DEFAULT_MODEL_ID

    source = correction_source
    source.family = request.param
    source.summary["models"]["right"] = {
        "foundation_vla": True,
        "inference_calls": 2,
        "model_id": XVLA_ID if source.family == "xvla" else DEFAULT_MODEL_ID,
    }
    for record in source.records:
        prediction = record["predictions"].pop("left")
        if source.family == "xvla":
            prediction["native_action_head_executed"] = prediction.pop("native_action_executed")
            prediction["action_age_ticks"] = prediction.pop("inference_age_ticks")
        prediction["model_language_prompt"] = dataset.phase_instruction("right", dataset.PHASES[0])
        record["predictions"]["right"] = prediction
        record["decisions"]["right"] = record["decisions"].pop("left")
        record["phase"]["right"] = record["phase"].pop("left")
    (source.source / "summary.json").write_text(json.dumps(source.summary))
    with gzip.open(source.source / "events.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in source.records))
    source.audit = {
        "verified": True,
        "model_families": {"right": source.family},
        "source_sha256": {
            name: hashlib.sha256((source.source / name).read_bytes()).hexdigest()
            for name in ("summary.json", "events.jsonl.gz", "final.json")
        },
    }
    monkeypatch.setattr(native_audit, "audit_native_run", lambda _: source.audit)
    return source


def _rewrite_correction_source(source):
    (source.source / "summary.json").write_text(json.dumps(source.summary))
    with gzip.open(source.source / "events.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in source.records))
    source.audit["source_sha256"] = {
        name: dataset._sha(source.source / name)
        for name in ("summary.json", "events.jsonl.gz", "final.json")
    }


def test_right_dagger_preserves_local_inputs_and_never_executes_labels(
    right_correction_source, tmp_path
):
    source, output = right_correction_source, tmp_path / "right-corrections"
    result = dataset.collect_corrections(source.source, output, agent="right")
    manifest, arrays = dataset.load_corrections(output)
    assert result == manifest
    assert manifest["native_audit_verified"] is True
    assert "native-audit.json" in manifest["files"]
    assert manifest["future_action_chunks_available"] is False
    assert manifest["teacher"]["actions_executed"] == 0
    assert manifest["source_success"] is False
    assert set(arrays) == {"right"}
    assert arrays["right"]["tick"].tolist() == [0, 1]
    assert arrays["right"]["target"].tolist() == [[9, 9, 9], [9, 9, 9]]
    assert arrays["right"]["ee_position"][:, 0].tolist() == [0, 1]
    assert "landmarks" not in arrays["right"]
    assert len(source.executed) == 2


def test_right_dagger_samples_fresh_inferences_not_cached_targets(
    right_correction_source, tmp_path
):
    source = right_correction_source
    age_key = "action_age_ticks" if source.family == "xvla" else "inference_age_ticks"
    source.records[1]["predictions"]["right"][age_key] = 1
    source.records[1]["predictions"]["right"]["inference_id"] = 1
    source.summary["models"]["right"]["inference_calls"] = 1
    _rewrite_correction_source(source)
    output = tmp_path / "fresh-only"
    dataset.collect_corrections(source.source, output, agent="right")
    _, arrays = dataset.load_corrections(output)
    assert arrays["right"]["tick"].tolist() == [0]
    assert len(source.executed) == 2


def test_right_dagger_cannot_infer_model_family_from_prediction_flags(
    right_correction_source, tmp_path
):
    source = right_correction_source
    prediction = source.records[0]["predictions"]["right"]
    if source.family == "smol":
        prediction["native_action_head_executed"] = prediction.pop("native_action_executed")
        prediction["action_age_ticks"] = prediction.pop("inference_age_ticks")
    else:
        prediction["native_action_executed"] = prediction.pop("native_action_head_executed")
        prediction["inference_age_ticks"] = prediction.pop("action_age_ticks")
    _rewrite_correction_source(source)
    output = tmp_path / "wrong-family-fields"
    with pytest.raises(ValueError, match="prediction fields differ"):
        dataset.collect_corrections(source.source, output, agent="right")
    assert source.executed == []
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("changed", ["identity", "audit_family"])
def test_right_dagger_requires_supported_audited_model_identity(
    right_correction_source, tmp_path, changed
):
    source = right_correction_source
    if changed == "identity":
        source.summary["models"]["right"]["model_id"] = "unapproved/native-like"
        _rewrite_correction_source(source)
    else:
        source.audit["model_families"]["right"] = "smol" if source.family == "xvla" else "xvla"
    output = tmp_path / "wrong-family-pin"
    with pytest.raises(ValueError, match="model identity"):
        dataset.collect_corrections(source.source, output, agent="right")
    assert source.executed == [] and not output.exists()


@pytest.mark.parametrize("replacement", ["unapproved/native-like", "other_family"])
def test_right_correction_loader_rechecks_manifest_model_identity(
    right_correction_source, tmp_path, replacement
):
    from relay_lab.cloth.pretrained import XVLA_ID
    from relay_lab.smolvla import DEFAULT_MODEL_ID

    source, output = right_correction_source, tmp_path / "changed-controller"
    dataset.collect_corrections(source.source, output, agent="right")
    manifest = json.loads((output / "manifest.json").read_text())
    if replacement == "other_family":
        replacement = DEFAULT_MODEL_ID if source.family == "xvla" else XVLA_ID
    manifest["source_controller"]["model_id"] = replacement
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model identity"):
        dataset.load_corrections(output)


def test_historical_familyless_audit_can_only_establish_right_xvla(
    right_correction_source, tmp_path
):
    source, output = right_correction_source, tmp_path / "legacy-audit"
    dataset.collect_corrections(source.source, output, agent="right")
    audit = json.loads((output / "native-audit.json").read_text())
    audit.pop("model_families")
    (output / "native-audit.json").write_text(json.dumps(audit))
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["native-audit.json"] = dataset._sha(output / "native-audit.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    if source.family == "xvla":
        _, arrays = dataset.load_corrections(output)
        assert arrays["right"]["tick"].tolist() == [0, 1]
    else:
        with pytest.raises(ValueError, match="model identity differs"):
            dataset.load_corrections(output)


@pytest.mark.parametrize("changed", ["verified", "digest"])
def test_right_dagger_rejects_unverified_or_unbound_native_source(
    right_correction_source, tmp_path, changed
):
    source = right_correction_source
    if changed == "verified":
        source.audit["verified"] = False
    else:
        source.audit["source_sha256"]["events.jsonl.gz"] = "0" * 64
    output = tmp_path / "unverified-right"
    with pytest.raises(ValueError, match="source-bound native audit"):
        dataset.collect_corrections(source.source, output, agent="right")
    assert not output.exists()
    assert source.executed == []


def test_right_dagger_rejects_another_arms_instruction(right_correction_source, tmp_path):
    output = tmp_path / "wrong-arm"
    dataset.collect_corrections(right_correction_source.source, output, agent="right")
    rows = json.loads((output / "corrections.json").read_text())
    rows[0]["instruction"] = dataset.phase_instruction("left", dataset.PHASES[0])
    (output / "corrections.json").write_text(json.dumps(rows))
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["corrections.json"] = hashlib.sha256(
        (output / "corrections.json").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity or order"):
        dataset.load_corrections(output)


def test_dagger_cancellation_after_label_cannot_execute_physics(correction_source, tmp_path):
    output = tmp_path / "cancelled-corrections"
    checks = iter([False, False, True])
    with pytest.raises(RuntimeError, match="cancelled before physics step"):
        dataset.collect_corrections(
            correction_source.source, output, should_cancel=lambda: next(checks)
        )
    assert correction_source.executed == []
    assert not (output / "manifest.json").exists()
    assert json.loads((output / "verification.json").read_text())["verified"] is False


@pytest.mark.parametrize("blocked", ["grasped", "hold", "grip", "phase"])
def test_recovery_exploration_never_moves_attached_or_held_tools(blocked):
    label = {"target": [0.1, 0.2, 0.05], "grip": blocked == "grip"}
    action, offset = dataset.recovery_behavior(
        label,
        agent="left",
        phase="carry_hem" if blocked == "phase" else "approach_hem",
        active_ticks=8,
        grasped=blocked == "grasped",
        hold=blocked == "hold",
        seed=10009,
    )
    assert not offset.any()
    np.testing.assert_allclose(action["target"], label["target"])


@pytest.fixture
def recovery_world(monkeypatch):
    from relay_lab.cloth import communication, policies, runtime, world

    seen = SimpleNamespace(actions=[], renders=0)

    class World:
        def __init__(self, **kwargs):
            self.tick, self.fixture_active = 0, True
            self._sites = {"left": 0, "right": 1}
            self.data = SimpleNamespace(site_xmat=np.repeat(np.eye(3).reshape(1, 9), 2, axis=0))
            self._grip = dict.fromkeys(dataset.AGENTS, False)
            self.holding = dict.fromkeys(dataset.AGENTS, None)

        def ee(self, agent):
            return np.asarray([self.tick, 0, 0], dtype=float)

        def metrics(self):
            return {"success": self.tick >= 12, "physics_valid": True}

        def snapshot(self):
            return {
                "tick": self.tick,
                "metrics": self.metrics(),
                "physics_config": {},
                "robots": {a: {"ee": self.ee(a).tolist()} for a in dataset.AGENTS},
            }

        def observe(self, agent):
            return {
                "agent": agent,
                "tick": self.tick,
                "ee_position": self.ee(agent).tolist(),
                "grasped": False,
                "landmarks": {"private": [99, 99, 99]},
            }

        def render(self, agent, width, height):
            seen.renders += 1
            return np.full((height, width, 3), self.tick, dtype=np.uint8)

        def step(self, actions):
            seen.actions.append(actions)
            self.tick += 1
            return self.snapshot()

        def close(self):
            pass

    class Policy:
        def __init__(self, agent):
            self.model_info = {"agent": agent, "foundation_vla": False}
            self.torch = SimpleNamespace(set_num_threads=lambda _: None)
            self.last_prediction = {"coordinator_hold": False}

        def act(self, observation, phase, **kwargs):
            return {"target": [observation["tick"] + 0.125, 0.25, 0.0625], "grip": False}

    class Controller:
        phase, active_ticks, done = "approach_sleeve_above", 0, False

        def __init__(self, *args):
            pass

        def decision(self, *args, **kwargs):
            return {"clearance": 0.23, "hold": False}

        def update(self, before, after, *args):
            self.active_ticks += 1
            self.done = after["tick"] >= 12

    class Channel:
        def __init__(self, *args, **kwargs):
            pass

        def publish(self, *args):
            pass

        def advance(self, *args):
            pass

        def receive(self, *args):
            return None

    monkeypatch.setattr(world, "ClothWorld", World)
    monkeypatch.setattr(policies, "ClothPolicy", Policy)
    monkeypatch.setattr(runtime, "LocalPhaseController", Controller)
    monkeypatch.setattr(runtime, "PhaseJudge", lambda _: None)
    monkeypatch.setattr(communication, "Channel", Channel)
    monkeypatch.setattr(communication, "observation_features", lambda _: {})
    monkeypatch.setattr(communication.LearnedFeatureCodec, "load", lambda _: None)
    return seen


def test_recovery_labels_remain_separate_from_actual_exploration(recovery_world, tmp_path):
    output = tmp_path / "recovery"
    manifest = dataset.collect_recovery(output, seed=10009, max_ticks=12, image_size=32)
    assert manifest["success"] and manifest["future_action_chunks_available"] is False
    assert manifest["perturbation"]["steps"] == {"left": 4, "right": 4}
    _, agents = dataset.load_recovery(output)
    for agent, arrays in agents.items():
        assert arrays["tick"].tolist() == [0, 4, 8]
        assert arrays["ee_position"][:, 0].tolist() == [0, 4, 8]
        assert arrays["rgb"][:, 0, 0, 0].tolist() == [0, 4, 8]
        assert arrays["target"][-1].tolist() == [8.125, 0.25, 0.0625]
        assert arrays["target"][-1].tolist() != recovery_world.actions[8][agent]["target"]
        assert "landmarks" not in arrays and "behavior_action" not in arrays
    with pytest.raises(ValueError, match="not a native action"):
        dataset.load_episode(output)
    rows = json.loads((output / "left-samples.json").read_text())
    rows[-1]["target"] = rows[-1]["behavior_action"]["target"]
    (output / "left-samples.json").write_text(json.dumps(rows))
    manifest["files"]["left-samples.json"] = dataset._sha(output / "left-samples.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="confused with actual behavior"):
        dataset.load_recovery(output)


def test_recovery_cancellation_during_camera_capture_executes_no_action(recovery_world, tmp_path):
    output = tmp_path / "cancelled"
    manifest = dataset.collect_recovery(
        output,
        seed=10009,
        max_ticks=12,
        image_size=32,
        should_cancel=lambda: recovery_world.renders >= 2,
    )
    assert manifest["status"] == "cancelled"
    assert manifest["ticks"] == 0
    assert manifest["samples"] == {"left": 0, "right": 0}
    assert recovery_world.actions == []
    with pytest.raises(ValueError, match="not a complete"):
        dataset.load_recovery(output)


def test_recovery_keeps_failed_physical_outcome_distinct_from_one_step_labels(
    recovery_world, tmp_path
):
    output = tmp_path / "incomplete_fold"
    manifest = dataset.collect_recovery(output, seed=10010, max_ticks=11, image_size=32)
    assert manifest["status"] == "exhausted"
    assert manifest["success"] is False
    assert manifest["metrics"]["success"] is False
    loaded, agents = dataset.load_recovery(output)
    assert loaded["future_action_chunks_available"] is False
    assert len(agents["left"]["target"]) == 3
    assert len(recovery_world.actions) == 11


def test_recovery_cannot_use_evaluation_seed(tmp_path):
    with pytest.raises(ValueError, match="training seeds"):
        dataset.collect_recovery(tmp_path / "forbidden", seed=103)
    assert not (tmp_path / "forbidden").exists()
