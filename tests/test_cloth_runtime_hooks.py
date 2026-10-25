"""Authorization boundaries and evidence retention without model/physics loading."""

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth import policies, runtime, world


@pytest.fixture
def rig(monkeypatch, tmp_path):
    state = SimpleNamespace(worlds=[], loads=[], stop=False, on_judge=None, steps=[])
    paths = {}
    for arm in runtime.AGENTS:
        paths[arm] = tmp_path / f"{arm}.pt"
        paths[arm].write_bytes(("trusted-" + arm).encode())
    state.paths = paths
    state.hashes = {a: hashlib.sha256(p.read_bytes()).hexdigest() for a, p in paths.items()}

    class World:
        PHASES = ("approach", "settle")
        CONTROL_DT = 0.04

        def __init__(self, **kwargs):
            self.tick, self.events, self.closed = 0, [], False
            self.fixture_active = True
            state.worlds.append(self)

        def snapshot(self):
            return {
                "tick": self.tick,
                "vertices": [[0, 0, 0]],
                "robots": {a: {"ee": [0, 0, 0]} for a in runtime.AGENTS},
                "metrics": self.metrics(),
                "obstacle": None,
                "fixture_active": self.fixture_active,
                "physics_config": {},
            }

        def observe(self, arm, rgb=False):
            result = {"agent": arm, "tick": self.tick}
            if rgb:
                result["rgb"] = np.zeros((16, 16, 3), dtype=np.uint8)
            return result

        def step(self, actions):
            state.steps.append(actions)
            self.tick += 1
            return self.snapshot()

        def release_fixture(self):
            self.fixture_active = False
            self.events.append({"event": "vacuum_fixture_released", "tick": self.tick})

        def metrics(self):
            return {"success": self.tick >= 2}

        def close(self):
            self.closed = True

    class Policy:
        def __init__(self, arm, checkpoint):
            self.path = Path(checkpoint)
            self.agent = arm
            state.loads.append((arm, self.path, self.path.read_bytes()))
            self.last_prediction = {"coordinator_hold": False}

        @property
        def model_info(self):
            return {"checkpoint_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}

        def act(self, *args, **kwargs):
            return {"target": [0, 0, 0], "grip": True}

    class Codec:
        @staticmethod
        def load(path):
            return Codec()

        def save(self, path):
            Path(path).write_text("{}")

    class Channel:
        def __init__(self, *args, **kwargs):
            self.counter = dict.fromkeys(
                ("published", "delivered", "dropped", "expired", "out_of_order", "packet_bytes"),
                0,
            )
            self.metrics, self.config, self.events = {}, {}, []

        def publish(self, *args):
            pass

        def advance(self, *args, **kwargs):
            pass

        def receive(self, *args):
            return None

    class Judge:
        def __init__(self, mode):
            self.calls, self.traces = 0, []

        def advance(self, *args):
            self.calls += 1
            if state.on_judge:
                state.on_judge()
            return True

        def close(self):
            pass

    class Controller:
        def __init__(self, arm, phases):
            self.agent, self.phases = arm, phases
            self.index, self.done, self.waiting_ticks = 0, False, 0
            self.last_decision = {}

        @property
        def phase(self):
            return self.phases[self.index]

        def decision(self, *args, **kwargs):
            self.last_decision = {"hold": False, "clearance": 0.23, "reason": "local subgoal"}
            return self.last_decision

        def update(self, before, after, action, belief, judge):
            judge.advance()
            self.done = self.index == len(self.phases) - 1
            self.index = min(self.index + 1, len(self.phases) - 1)
            return {"event": "phase_completed", "agent": self.agent, "tick": after["tick"]}

    monkeypatch.setattr(world, "ClothWorld", World)
    monkeypatch.setattr(policies, "ClothPolicy", Policy)
    monkeypatch.setattr(policies, "DEFAULT_DIRECTORY", tmp_path)
    monkeypatch.setattr(runtime, "LearnedFeatureCodec", Codec)
    monkeypatch.setattr(runtime, "Channel", Channel)
    monkeypatch.setattr(runtime, "PhaseJudge", Judge)
    monkeypatch.setattr(runtime, "LocalPhaseController", Controller)
    monkeypatch.setattr(runtime, "observation_features", lambda obs: {})
    state.config = runtime.ClothConfig(video=False, max_ticks=10)
    state.output = tmp_path / "run"
    return state


def run(rig, **kwargs):
    return runtime.run_cloth(
        rig.config,
        rig.output,
        checkpoint_paths=rig.paths,
        expected_checkpoint_hashes=rig.hashes,
        **kwargs,
    )


def test_wrong_approval_stops_before_world_or_checkpoint_loader(rig):
    rig.hashes["right"] = "0" * 64
    result = run(rig)
    assert result["status"] == "failed"
    assert "right checkpoint SHA-256" in result["error"]
    assert not result["success"]
    assert rig.loads == rig.worlds == []
    assert json.loads((rig.output / "summary.json").read_text())["status"] == "failed"


def test_loads_verified_snapshot_even_when_source_changes_after_world_creation(rig, monkeypatch):
    original = world.ClothWorld

    def replacing_world(**kwargs):
        rig.paths["left"].write_bytes(b"replacement after verification")
        return original(**kwargs)

    monkeypatch.setattr(world, "ClothWorld", replacing_world)
    result = run(rig)
    assert result["status"] == "succeeded"
    assert rig.loads[0] == ("left", rig.output / "models/left.pt", b"trusted-left")
    assert result["models"]["left"]["checkpoint_sha256"] == rig.hashes["left"]


def test_cancel_before_initialization_never_loads_or_moves(rig):
    result = run(rig, should_stop=lambda: True)
    assert result["status"] == "cancelled" and not result["success"]
    assert rig.loads == rig.worlds == rig.steps == []
    assert result["ticks"] == 0


def test_cancellation_after_qwen_preserves_executed_tick_without_releasing_fixture(rig):
    rig.on_judge = lambda: setattr(rig, "stop", True)
    reports = []
    result = run(rig, should_stop=lambda: rig.stop, on_progress=reports.append)
    assert result["status"] == "cancelled" and not result["success"]
    assert result["ticks"] == len(rig.steps) == 1
    assert result["planner"]["calls"] == 1
    assert rig.worlds[0].closed and rig.worlds[0].fixture_active
    assert result["fixture_active"] is True
    with gzip.open(rig.output / "events.jsonl.gz", "rt") as stream:
        rows = [json.loads(row) for row in stream]
    assert len(rows) == 1 and rows[0]["tick"] == 1
    assert any(event["event"] == "cancelled" for event in rows[0]["events"])
    assert reports[-1]["status"] == "cancelled" and reports[-1]["tick"] == 1


def test_cancellation_during_inference_is_checked_before_step(rig, monkeypatch):
    original = policies.ClothPolicy.act
    calls = []

    def predict_then_cancel(self, *args, **kwargs):
        calls.append(self.agent)
        action = original(self, *args, **kwargs)
        rig.stop = True
        return action

    monkeypatch.setattr(policies.ClothPolicy, "act", predict_then_cancel)
    result = run(rig, should_stop=lambda: rig.stop)
    assert result["status"] == "cancelled"
    assert result["ticks"] == 0 and not rig.steps
    assert calls == ["left"]
    assert rig.worlds[0].fixture_active and rig.worlds[0].closed


def test_cancellation_during_model_load_never_loads_second_arm(rig):
    def factory(arm, snapshot):
        result = policies.ClothPolicy(arm, snapshot)
        rig.stop = True
        return result

    result = run(rig, should_stop=lambda: rig.stop, policy_factory=factory)
    assert result["status"] == "cancelled" and result["ticks"] == 0
    assert [arm for arm, _, _ in rig.loads] == ["left"]
    assert not rig.steps and rig.worlds[0].closed


def test_progress_is_bounded_and_includes_terminal_measured_result(rig):
    reports = []
    result = run(rig, on_progress=reports.append)
    assert result["status"] == "succeeded"
    assert [report["tick"] for report in reports] == [0, 1, 2, 2]
    assert reports[-1]["status"] == "succeeded"
    assert reports[-1]["metrics"] == {"success": True}
    assert reports[-1]["phase_indices"] == {"left": 1, "right": 1}
    assert reports[-1]["done"] == {"left": True, "right": True}
    assert reports[-1]["total_phases"] == 2
    assert len(reports) <= 2 * reports[-1]["total_phases"] + 2
    config = json.loads((rig.output / "config.json").read_text())
    assert "on_progress" not in config and "should_stop" not in config


def test_progress_failure_stops_before_more_movement_and_is_not_retried(rig):
    reports = []

    def report(value):
        reports.append(value)
        if value["tick"] == 1:
            raise RuntimeError("report delivery unavailable")

    result = run(rig, on_progress=report)
    assert result["status"] == "failed" and not result["success"]
    assert result["ticks"] == len(rig.steps) == 1
    assert "report delivery unavailable" in result["error"]
    assert len(reports) == 2
    assert rig.worlds[0].closed and rig.worlds[0].fixture_active


def test_terminal_report_failure_cannot_return_success(rig):
    def report(value):
        if value["status"] == "succeeded":
            raise RuntimeError("terminal report failed")

    result = run(rig, on_progress=report)
    assert result["status"] == "failed" and not result["success"]
    assert result["metrics"]["success"] is True
    assert "progress callback" in result["error"]


def test_default_checkpoints_remain_supported(rig):
    result = runtime.run_cloth(rig.config, rig.output)
    assert result["status"] == "succeeded"
    assert {a: m["checkpoint_sha256"] for a, m in result["models"].items()} == rig.hashes


def test_sequence_completion_waits_for_actual_measured_settling(rig, monkeypatch):
    holds = []
    original_act = policies.ClothPolicy.act

    def act(self, observation, phase, **options):
        holds.append((observation["tick"], options["hold"]))
        return original_act(self, observation, phase, **options)

    monkeypatch.setattr(policies.ClothPolicy, "act", act)
    monkeypatch.setattr(world.ClothWorld, "metrics", lambda self: {"success": self.tick >= 4})
    result = run(rig)
    assert result["success"] and result["ticks"] == 4
    assert result["terminal_settle_ticks"] == 2
    assert all(hold for tick, hold in holds if tick >= 2)


def test_waiting_cannot_award_success_without_measured_geometry(rig, monkeypatch):
    monkeypatch.setattr(world.ClothWorld, "metrics", lambda self: {"success": False})
    result = run(rig)
    assert not result["success"]
    assert result["ticks"] == rig.config.max_ticks
    assert result["status"] == "exhausted"


@pytest.mark.parametrize(
    "options",
    [
        {"checkpoint_paths": {"left": "left.pt"}},
        {"expected_checkpoint_hashes": {"left": "0" * 64, "right": "invalid"}},
        {"on_progress": "not callable"},
        {"should_stop": True},
        {"policy_factory": "untrusted remote name"},
    ],
)
def test_invalid_hook_contract_rejected_before_creating_output(rig, options):
    with pytest.raises((TypeError, ValueError)):
        runtime.run_cloth(rig.config, rig.output, **options)
    assert not rig.output.exists()


def test_native_factory_receives_approved_snapshot_and_only_local_rgb_proprio(rig, monkeypatch):
    original_observe = world.ClothWorld.observe
    seen = []

    def observe(self, arm, rgb=False):
        return {
            **original_observe(self, arm, rgb=rgb),
            "ee_position": [0, 0, .2], "ee_rotation": np.eye(3).tolist(),
            "gripper_closed": False, "grasped": False,
            "landmarks": {"hidden_teacher_target": [1, 2, 3]},
            "obstacle": {"active": True},
        }

    class Native:
        def __init__(self, arm, snapshot):
            assert snapshot.read_bytes() == ("trusted-" + arm).encode()
            self.last_prediction = {"coordinator_hold": False}
            self.model_info = {"foundation_vla": True, "native_action_head_executed": True}

        def act(self, observation, *args, **kwargs):
            seen.append(set(observation))
            return {"target": [0, 0, .2], "grip": False}

    monkeypatch.setattr(world.ClothWorld, "observe", observe)
    result = run(rig, policy_factory=Native)
    assert result["success"]
    assert not rig.loads  # The small task controller was never constructed.
    assert len(seen) == 4
    assert all(keys == {
        "agent", "tick", "rgb", "ee_position", "ee_rotation", "gripper_closed", "grasped"
    } for keys in seen)
    assert any("native action heads" in item for item in result["limitations"])
