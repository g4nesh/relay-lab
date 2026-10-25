"""An action replay must reproduce physical state and reject a changed action."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from relay_lab.cloth.world import ClothWorld

SPEC = importlib.util.spec_from_file_location(
    "cloth_reexecution", Path(__file__).parents[1] / "scripts/replay_cloth_actions.py"
)
reexecution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reexecution)


@pytest.fixture
def original(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    world = ClothWorld(seed=17, disturbances=False)
    try:
        with gzip.open(source / "events.jsonl.gz", "wt") as stream:
            for tick in range(3):
                actions = {
                    arm: {"target": world.observe(arm)["ee_position"], "grip": False}
                    for arm in ("left", "right")
                }
                snapshot = world.step(actions)
                stream.write(json.dumps({
                    "decision_tick": tick, "actions": actions,
                    "robots": snapshot["robots"], "events": [],
                }) + "\n")
        (source / "final.json").write_text(json.dumps(world.snapshot()))
    finally:
        world.close()
    (source / "summary.json").write_text(json.dumps({
        "run_id": "test", "ticks": 3,
        "config": {"seed": 17, "scenario": "clear", "fixture": True},
    }))
    return source


def test_reexecutes_every_physics_step_and_matches_final_mesh(original, tmp_path):
    result = reexecution.replay(original, tmp_path / "replay", video=False)
    assert result["verified"] and result["steps"] == 3
    assert result["final_vertices_bitwise_equal"]
    assert result["maximum_robot_state_error"] == 0
    assert result["reexecutes_neural_models"] is False
    assert result["final_metrics"] == json.loads((original / "final.json").read_text())["metrics"]
    assert result["final_metrics"]["success"] is False


def test_changed_motor_action_fails_physical_verification(original, tmp_path):
    path = original / "events.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    rows[0]["actions"]["left"]["target"][0] += 0.05
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "tampered"
    with pytest.raises(ValueError, match="Physics differs"):
        reexecution.replay(original, output, video=False)
    report = json.loads((output / "verification.json").read_text())
    assert not report["verified"] and "Physics differs" in report["error"]


def test_replay_uses_the_bytes_it_hashed_even_if_a_source_path_changes(original, tmp_path, monkeypatch):
    path = original / "events.jsonl.gz"
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    class ReplacedSourceWorld(ClothWorld):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            path.write_bytes(b"replaced after capture")

    monkeypatch.setattr(reexecution, "ClothWorld", ReplacedSourceWorld)
    result = reexecution.replay(original, tmp_path / "replay", video=False)
    assert result["verified"] and result["steps"] == 3
    assert result["source_artifact_sha256"]["events.jsonl.gz"] == original_hash
    assert hashlib.sha256(path.read_bytes()).hexdigest() != original_hash
