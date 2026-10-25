"""Experiment orchestration tests without loading foundation models or OpenGL."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_cloth_latent_adapter import _checkpoint, _observation
from test_cloth_latent_adapter import paired_cache as paired_cache  # noqa: PLC0414 - pytest fixture re-export

from relay_lab.cloth import features as features_module
from relay_lab.cloth import latent_dataset, latent_experiment
from relay_lab.cloth import world as world_module
from relay_lab.cloth.communication import AGENTS
from relay_lab.cloth.features import feature_contract
from relay_lab.cloth.latent_adapter import MODES, WIDTHS, LearnedLatentPolicy
from relay_lab.cloth.latent_training import load_pairs, predictions, train


def _forbidden(*args, **kwargs):
    pytest.fail("Neither teacher targets nor Qwen may supply actions/peer information here")


def test_two_epoch_training_integration_serializes_absent_class_rates_as_null(paired_cache, tmp_path):
    output = tmp_path / "trained"
    report = train(*paired_cache, output, epochs=2)
    assert set(report["modes"]) == set(MODES)
    saved = json.loads((output / "evaluation.json").read_text())
    assert saved["training"]["full_task_success_evaluation"] is False
    assert saved["training"]["delay_training"] is False
    data, _, _ = load_pairs(*paired_cache)
    batch = data["test"]
    for mode in MODES:
        assert 0 <= report["modes"][mode]["selected_epoch"] < 2
        policy = LearnedLatentPolicy(output / f"{mode}.pt")
        assert policy.mode == mode
        fresh = predictions(policy.network, batch)
        aged = policy.network(
            batch["features"], batch["states"], batch["facts"],
            torch.ones(batch["size"]), torch.full((batch["size"],), 0.8),
        )
        for arm in AGENTS:
            # Zero-age training, including AdamW weight decay and checkpoint
            # roundtripping, must preserve the unused age column exactly.
            assert torch.count_nonzero(policy.network.heads[arm][0].weight[:, -1]) == 0
            torch.testing.assert_close(fresh[arm], aged[arm], rtol=0, atol=0)
            metrics = saved["modes"][mode]["test"]["arms"][arm]
            assert metrics["hazard_count"] == 0
            assert metrics["hazard_hold_rate"] is None
            assert metrics["clear_count"] == 1
    assert "NaN" not in (output / "evaluation.json").read_text()


@pytest.fixture
def frozen_sequence(tmp_path, monkeypatch):
    """Tiny synthetic cache; loader integrity has separate exhaustive tests."""
    dataset, cache, checkpoints = (tmp_path / name for name in ("scenes", "cache", "models"))
    for path in (dataset, cache, checkpoints):
        path.mkdir()
    rows = []
    for condition in ("clear", "left_obstacle", "right_obstacle"):
        for arm in AGENTS:
            observation = _observation(arm)
            if condition == f"{arm}_obstacle":
                observation["obstacle"] = {"position": [0.1, 0.0, 0.1], "radius": .04, "active": True}
            rows.append({
                "id": f"test-104-s0-{condition}-{arm}", "agent": arm, "split": "test",
                "snapshot_id": "test-104-s0", "counterfactual": condition,
                "phase": "lift_hem", "observation": observation,
                "rgb_sha256": hashlib.sha256(f"{arm}-{condition}".encode()).hexdigest(),
                "labels": {"coordinated_hold": condition != "clear"},
            })
    (dataset / "rows.json").write_text(json.dumps(rows))
    sources = {}
    for arm in AGENTS:
        ids = np.array([i for i, row in enumerate(rows) if row["agent"] == arm])
        np.savez_compressed(
            cache / f"{arm}.npz", row_indices=ids,
            features=np.stack([np.linspace(-1, 1, WIDTHS[arm], dtype=np.float32) + i for i in ids]),
        )
        sources[arm] = {"model": feature_contract("smolvla" if arm == "left" else "xvla")}
    for mode in MODES:
        _checkpoint(checkpoints, mode)
    monkeypatch.setattr(latent_experiment, "load_pairs", lambda *args: ({}, sources, {}))
    monkeypatch.setattr(world_module, "ClothWorld", _forbidden)
    monkeypatch.setattr(features_module, "FrozenVLAFeatureExtractor", _forbidden)
    from relay_lab import llm, mediator
    monkeypatch.setattr(llm, "LocalLLM", _forbidden)
    monkeypatch.setattr(mediator, "LLMMediator", _forbidden)
    return dataset, cache, checkpoints


def test_frozen_scene_compare_is_causal_matched_and_explicitly_not_closed_loop(frozen_sequence, tmp_path):
    destination = tmp_path / "comparison"
    report = latent_experiment.compare(*frozen_sequence, destination, backend="memory")
    assert report["kind"] == "held-out-frozen-scene-observation-sequence"
    assert report["actions_integrated"] is False
    assert report["full_task_success_evaluation"] is False
    assert len(report["attempts"]) == 8
    sequences, schedules = {}, {}
    for attempt in report["attempts"]:
        assert attempt["status"] == "completed"
        mode, scenario = attempt["mode"], attempt["scenario"]
        assert attempt["communication"]["backend"]["durable"] is False
        folder = destination / f"{scenario}-{mode}"
        records = json.loads((folder / "decisions.json").read_text())
        assert len(records) == 12
        sequence = []
        for record in records:
            for arm in AGENTS:
                decision = record["decisions"][arm]
                sequence.append((record["tick"], arm, decision["row_id"]))
                belief = decision["belief"]
                if belief:
                    assert belief["recipient"] == arm
                    assert belief["observation_tick"] < record["tick"]
                    assert belief["age_ticks"] == record["tick"] - belief["observation_tick"]
                    assert belief["expires_tick"] > record["tick"]
                else:
                    assert decision["prediction"]["missing_message_hold"] == (mode != "none")
        sequences[(scenario, mode)] = sequence
        journal = json.loads((folder / "communication.json").read_text())
        packets = [event["packet"] for event in journal["events"] if event["kind"] == "published"]
        schedules[(scenario, mode)] = [
            tuple(packet[key] for key in ("sender", "observation_tick", "deliver_tick", "dropped"))
            for packet in packets
        ]
    assert all(sequence == next(iter(sequences.values())) for sequence in sequences.values())
    for scenario in ("one_tick", "delayed_dropped"):
        assert schedules[(scenario, "none")] == []
        assert schedules[(scenario, "embeddings")] == schedules[(scenario, "structured_text")]
        assert schedules[(scenario, "hybrid")] == schedules[(scenario, "structured_text")]


def test_compare_retains_every_predeclared_attempt_when_one_model_cannot_load(frozen_sequence, tmp_path):
    (frozen_sequence[2] / "embeddings.pt").unlink()
    output = tmp_path / "failed-comparison"
    report = latent_experiment.compare(*frozen_sequence, output, backend="memory")
    attempts = report["attempts"]
    assert len(attempts) == 8
    assert sum(a["status"] == "failed" for a in attempts) == 2
    assert sum(a["status"] == "completed" for a in attempts) == 6
    for attempt in attempts:
        if attempt["mode"] == "embeddings":
            assert attempt["error_type"] == "FileNotFoundError"
            path = output / f"{attempt['scenario']}-embeddings" / "decisions.json"
            assert json.loads(path.read_text()) == []
    assert json.loads((output / "manifest.json").read_text()) == report


@pytest.fixture
def mocked_live(tmp_path, monkeypatch):
    """Real learned checkpoint and causal memory channel, mocked sensing/physics."""
    import imageio.v2 as imageio

    from relay_lab import llm, mediator

    _, policy, ids = _checkpoint(tmp_path, "embeddings")
    events = {"extract": [], "worlds": [], "frames": [], "writers": []}

    class World:
        def __init__(self, **kwargs):
            self.tick, self.closed, self.released = 0, False, False
            self.config, self.executed = kwargs, []
            self.positions = {a: _observation(a)["ee_position"] for a in AGENTS}
            self.grips = {a: False for a in AGENTS}
            self.fail_tick = None
            events["worlds"].append(self)

        def observe(self, arm):
            return {**_observation(arm), "tick": self.tick,
                    "ee_position": self.positions[arm], "grasped": self.grips[arm],
                    "grasp_target": "sleeve" if self.grips[arm] else "none"}

        def snapshot(self):
            return {"tick": self.tick, "metrics": self.metrics(), "robots": {
                a: {"position": list(self.positions[a]), "grasped": self.grips[a]} for a in AGENTS
            }}

        def metrics(self):
            return {"success": False, "tick": self.tick}

        def step(self, actions):
            if self.fail_tick == self.tick + 1:
                raise RuntimeError("injected physics failure")
            assert set(actions) == set(AGENTS)
            self.executed.append(copy.deepcopy(actions))
            self.tick += 1
            self.positions = {a: list(actions[a]["target"]) for a in AGENTS}
            self.grips = {a: bool(actions[a]["grip"]) for a in AGENTS}
            return self.snapshot()

        def render(self, camera, **kwargs):
            return np.full((16, 16, 3), self.tick, np.uint8)

        def release_fixture(self):
            self.released = True

        def close(self):
            self.closed = True

        reference_target = staticmethod(_forbidden)

    class Extractor:
        def __init__(self, family, arm):
            self.family, self.arm, self.space_id = family, arm, ids[arm]

        def extract(self, observation, instruction):
            assert observation["agent"] == self.arm
            assert not {"labels", "peer", "action", "teacher", "counterfactual"} & observation.keys()
            assert instruction == "Fold the T-shirt. approach sleeve above."
            events["extract"].append((self.arm, observation["tick"], instruction))
            vector = np.linspace(-1, 1, WIDTHS[self.arm], dtype=np.float32) + observation["tick"]
            return SimpleNamespace(vector=vector, metadata={
                "feature_space_id": self.space_id,
                "rgb_sha256": hashlib.sha256(observation["rgb"].tobytes()).hexdigest(),
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                "vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                "test_mock": True,
            })

    class Writer:
        def __init__(self, *args, **kwargs):
            self.closed = False
            events["writers"].append(self)

        def append_data(self, frame):
            events["frames"].append(frame.copy())

        def close(self):
            self.closed = True

    monkeypatch.setattr(world_module, "ClothWorld", World)
    monkeypatch.setattr(features_module, "FrozenVLAFeatureExtractor", Extractor)
    monkeypatch.setattr(latent_dataset, "private_rgb", lambda w, a, s: np.full((s, s, 3), w.tick, np.uint8))
    monkeypatch.setattr(imageio, "get_writer", Writer)
    monkeypatch.setattr(llm, "LocalLLM", _forbidden)
    monkeypatch.setattr(mediator, "LLMMediator", _forbidden)
    return policy.path, events, World


def test_live_infers_both_arms_before_physics_and_records_applied_learned_actions(mocked_live, tmp_path):
    checkpoint, events, _ = mocked_live
    output = tmp_path / "live"
    report = latent_experiment.live(checkpoint, output, backend="memory", calls=2, seed=105)
    assert report["status"] == "completed"
    assert report["full_task_success_evaluation"] is False
    assert report["native_action_heads_used"] is False
    assert report["teacher_targets_used"] is False
    assert [(a, tick) for a, tick, _ in events["extract"]] == [("left", 0), ("right", 0), ("left", 4), ("right", 4)]
    world = events["worlds"][0]
    assert world.config == {"seed": 105, "disturbances": True, "obstacle_schedule": ((8, 24, "left"), (32, 44, "right"))}
    assert len(events["frames"]) == 8
    assert len(world.executed) == 8
    for index, record in enumerate(report["records"]):
        assert record["observation_tick"] == index * 4
        assert record["after_tick"] == (index + 1) * 4
        assert record["metrics"]["tick"] == record["after_tick"]
        for arm in AGENTS:
            decision = record["decisions"][arm]
            assert decision["action"] == world.executed[index * 4][arm]
            assert decision["local_observation"]["tick"] == index * 4
            assert decision["prediction"]["replacement_head"] is True
            if index == 0:
                assert decision["belief"] is None
                assert decision["prediction"]["missing_message_hold"]
            else:
                assert decision["belief"]["observation_tick"] == 0
                assert decision["belief"]["age_ticks"] == 4
    assert world.closed and world.released and events["writers"][0].closed
    assert json.loads((output / "report.json").read_text()) == report


def test_live_preserves_decisions_when_physics_fails_mid_action_block(mocked_live, tmp_path, monkeypatch):
    checkpoint, events, world_type = mocked_live
    original_init = world_type.__init__

    def failing_init(self, **kwargs):
        original_init(self, **kwargs)
        self.fail_tick = 3

    monkeypatch.setattr(world_type, "__init__", failing_init)
    output = tmp_path / "partial-live"
    with pytest.raises(RuntimeError, match="physics failure"):
        latent_experiment.live(checkpoint, output, backend="memory", calls=2)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert len(report["records"]) == 1
    record = report["records"][0]
    assert record["observation_tick"] == 0
    assert record["after_tick"] == 2
    assert set(record["decisions"]) == set(AGENTS)
    assert events["worlds"][0].closed and events["writers"][0].closed


def test_live_retains_checkpoint_loading_failure(tmp_path):
    output = tmp_path / "missing-model"
    with pytest.raises(FileNotFoundError):
        latent_experiment.live(tmp_path / "missing.pt", output, backend="memory", calls=2)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "FileNotFoundError"
    assert report["records"] == []
