"""Communication/prompt contract checks; these do not claim real-model inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.cloth import pretrained_team
from relay_lab.cloth.communication import FEATURES, LearnedFeatureCodec
from relay_lab.cloth.pretrained_team import (
    PromptMediator,
    compact_context,
    instruction_from_context,
)


def local(agent="left"):
    return {
        "agent": agent,
        "grasp_target": "hem",
        "obstacle": None,
        "landmarks": {"hem": [0.2, -0.3, 0.0]},
        "unavailable_peer_ground_truth": {"secret": "never pass this"},
    }


def delivered():
    return {
        "recipient": "left",
        "sender": "right",
        "features": {
            "ee_x": 0.245,
            "ee_y": -0.124,
            "ee_z": 0.258,
            "grasp_hem": 1.0,
            "grasp_sleeve": 0.0,
            "obstacle_active": 1.0,
        },
        "age_ticks": 4,
        "packet_id": "right-0",
        "precision": "lossy learned reconstruction; exact safety facts",
    }


def test_only_delivered_peer_measurements_enter_the_language_context():
    context = compact_context(local(), delivered())
    assert set(context) == {"agent", "own_grasp", "own_obstacle", "peer"}
    assert context["peer"]["packet_id"] == "right-0"
    assert context["peer"]["tool_xyz_m"] == [0.24, -0.12, 0.26]
    prompt = instruction_from_context(context)
    assert "Peer grip hem; obstacle present" in prompt
    assert "report age 4 ticks" in prompt
    assert "secret" not in prompt
    assert "unknown" in instruction_from_context(compact_context(local(), None))
    with pytest.raises(ValueError, match="recipient"):
        compact_context(local("right"), delivered())


def test_qwen_contexts_are_separate_and_fresh_report_facts_are_not_cached(monkeypatch):
    inputs = []

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, messages):
            inputs.append(messages)
            return SimpleNamespace(text="Wait for a clear shared workspace.", to_dict=dict)

        def close(self):
            pass

    monkeypatch.setattr("relay_lab.llm.LocalLLM", FakeLLM)
    mediator = PromptMediator("qwen")
    prompt, _ = mediator.instruction(local(), delivered())
    assert len(inputs) == 1
    new = delivered()
    new["age_ticks"] = 8
    new["features"]["ee_z"] = 0.35
    prompt, context = mediator.instruction(local(), new)
    assert len(inputs) == 1  # same categorical event, fresh numeric facts appended
    assert "0.35 m" in prompt and "report age 8 ticks" in prompt
    assert context["peer"]["age_ticks"] == 8
    mediator.instruction(local("right"), None)
    assert len(inputs) == 2
    assert "right-0" not in inputs[-1][1]["content"]
    assert len(mediator.traces) == 2


def test_failed_qwen_generation_aborts_without_instruction_fallback(monkeypatch):
    class BrokenLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, messages):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("relay_lab.llm.LocalLLM", BrokenLLM)
    with pytest.raises(RuntimeError, match="model unavailable"):
        PromptMediator("qwen").instruction(local(), delivered())


def test_joint_loop_shares_tick_and_reads_only_delivered_channel_reports(tmp_path, monkeypatch):
    """Test doubles exercise wiring, never substitute for measured model probes."""
    model_calls = []
    rows = [
        dict(zip(FEATURES, row.tolist()))
        for row in np.random.default_rng(4).normal(size=(60, len(FEATURES)))
    ]
    codec = LearnedFeatureCodec.fit(rows)
    monkeypatch.setattr(pretrained_team.LearnedFeatureCodec, "load", lambda path: codec)

    class FakeWorld:
        CONTROL_DT = 0.04

        def __init__(self, **kwargs):
            self.tick = 0

        def observe(self, agent, rgb=False):
            return {
                "agent": agent,
                "tick": self.tick,
                "ee_position": [-0.2 if agent == "left" else 0.2, 0.0, 0.3],
                "grasped": False,
                "grasp_target": "none",
                "landmarks": {},
                "obstacle": None,
                "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
            }

        def render(self, *args, **kwargs):
            return np.zeros((16, 16, 3), dtype=np.uint8)

        def snapshot(self):
            return {
                "robots": {agent: {"holding": None} for agent in ("left", "right")},
                "metrics": self.metrics(),
            }

        def step(self, actions):
            assert set(actions) == {"left", "right"}
            self.tick += 1
            return self.snapshot()

        def metrics(self):
            return {"success": False}

        def release_fixture(self):
            pass

        def close(self):
            pass

    class FakePolicy:
        def __init__(self, agent, device, seed):
            self.agent, self.last_prediction = agent, {}
            self.model_info = {"test_double": True}

        def act(self, observation, instruction):
            assert observation["agent"] == self.agent
            model_calls.append((self.agent, observation["tick"], instruction))
            self.last_prediction = {"raw_chunk": [[0, 0, 0]]}
            return {"target": observation["ee_position"], "grip": False}

    monkeypatch.setattr(pretrained_team, "ClothWorld", FakeWorld)
    monkeypatch.setattr(pretrained_team, "SmolVLAClothProbe", FakePolicy)
    monkeypatch.setattr(pretrained_team, "XVLAClothProbe", FakePolicy)
    monkeypatch.setattr("imageio.v3.imwrite", lambda *args: None)
    monkeypatch.setattr(
        "imageio.v2.get_writer",
        lambda *args, **kwargs: SimpleNamespace(append_data=lambda frame: None, close=lambda: None),
    )
    report = pretrained_team.run(tmp_path / "contract-only", calls=2, coordination="memory")
    assert [row[:2] for row in model_calls] == [
        ("left", 0),
        ("right", 0),
        ("left", 4),
        ("right", 4),
    ]
    first, second = report["records"]
    assert first["decisions"]["left"]["belief"] is None
    assert second["decisions"]["left"]["belief"]["packet_id"] == "right-0"
    assert second["decisions"]["right"]["belief"]["packet_id"] == "left-0"
    assert "report age 4 ticks" in second["decisions"]["left"]["instruction"]
    assert report["communication"]["published"] == 4
    assert report["communication"]["delivered"] == 4
    assert report["physical_success_at_budget"] is False
    assert report["scripted_target_fallback"] is False
