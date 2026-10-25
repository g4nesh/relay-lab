import json

import pytest

from relay_lab.factory.replay import export_factory_replay


def state(tick, stage="queued"):
    return {
        "tick": tick,
        "time": tick * 0.1,
        "success": stage == "completed",
        "robots": [],
        "cargo": [],
        "jobs": {"job-1": {"job_id": "job-1", "stage": stage}},
        "rgb": [1, 2, 3],
    }


def test_replay_keeps_off_grid_completion_and_causal_labels(tmp_path):
    trace = tmp_path / "events.jsonl"
    rows = [
        {
            "type": "created",
            "scene": {"stations": []},
            "state": state(0),
            "backend": {"backend": "spacetimedb"},
            "policy": {"policy": "factory-vla", "checkpoint_sha256": "a" * 64},
        },
        {"type": "decision", "tick": 0, "mode": "llm", "reason": "Recorded choice"},
    ]
    for tick in range(1, 8):
        rows.append(
            {
                "type": "tick",
                "tick": tick,
                "state": state(tick, "completed" if tick == 7 else "queued"),
                "outcomes": [{"status": "completed"}] if tick == 7 else [],
                "motor_actions": [
                    {"robot_id": "arm-01", "command_id": "cmd-1", "peer_record_ids": ["amr-01-5"]}
                ],
            }
        )
    # A later pre-commit observation must not appear as committed playback.
    rows.append({"type": "execution_observed", "tick": 8, "state": state(8), "committed": False})
    trace.write_text("\n".join(json.dumps({"run_id": "recorded-run", **row}) for row in rows))
    result = export_factory_replay(trace, tmp_path / "replay.json")
    assert [frame["tick"] for frame in result["frames"]] == [0, 5, 7]
    final = result["frames"][-1]
    assert final["mission"]["completed"] == 1
    assert final["recorded_status"] == "succeeded" and final["status"] == "replay"
    assert final["causality"]["motor_inputs"][0]["peer_record_ids"] == ["amr-01-5"]
    assert final["planner"]["reason"] == "Recorded choice"
    assert final["policy"]["checkpoint_sha256"] == "a" * 64
    assert final["coordination"]["connected"] is False
    assert all(
        frame["recorded"] and not frame["live"] and "rgb" not in frame for frame in result["frames"]
    )
    assert trace.exists()


def test_replay_rejects_cross_run_trace_and_summary(tmp_path):
    trace = tmp_path / "events.jsonl"
    trace.write_text(
        json.dumps({"type": "created", "run_id": "one", "scene": {}, "state": state(0)})
        + "\n"
        + json.dumps({"type": "tick", "run_id": "two", "tick": 1, "state": state(1)})
    )
    with pytest.raises(ValueError, match="different run"):
        export_factory_replay(trace, tmp_path / "replay.json")


def test_replay_retains_stage_change_without_terminal_event(tmp_path):
    trace = tmp_path / "events.jsonl"
    rows = [
        {"type": "created", "scene": {}, "state": state(0)},
        {"type": "tick", "tick": 1, "state": state(1, "loaded")},
        {"type": "tick", "tick": 2, "state": state(2, "loaded")},
    ]
    trace.write_text("\n".join(json.dumps({"run_id": "one", **row}) for row in rows))
    result = export_factory_replay(trace, tmp_path / "replay.json")
    assert [frame["tick"] for frame in result["frames"]] == [0, 1, 2]


@pytest.mark.parametrize("status", ["failed", "cancelled", "exhausted"])
def test_replay_final_status_uses_summary_without_promoting_uncommitted_frame(tmp_path, status):
    trace = tmp_path / "events.jsonl"
    rows = [
        {"type": "created", "scene": {}, "state": state(0), "policy": {"calls": 0}},
        {
            "type": "tick",
            "tick": 1,
            "state": {**state(1), "motor_policy": {"model": {"calls": 5, "peer_fusions": 3}}},
        },
        {"type": "execution_observed", "tick": 2, "state": state(2), "committed": False},
    ]
    trace.write_text("\n".join(json.dumps({"run_id": "one", **row}) for row in rows))
    result = export_factory_replay(
        trace,
        tmp_path / "replay.json",
        {"run_id": "one", "status": status, "tick": 2, "error": "recorded reason"},
    )
    final = result["frames"][-1]
    assert final["tick"] == 1
    assert final["policy"] == {"calls": 5, "peer_fusions": 3}
    assert final["recorded_status"] == status and final["error"] == "recorded reason"
    assert final["terminal_annotation"]["last_committed_tick"] == 1


def test_replay_rejects_summary_success_without_committed_evidence(tmp_path):
    trace = tmp_path / "events.jsonl"
    trace.write_text(
        json.dumps({"type": "created", "run_id": "one", "scene": {}, "state": state(0)})
    )
    with pytest.raises(ValueError, match="committed success"):
        export_factory_replay(
            trace, tmp_path / "replay.json", {"run_id": "one", "status": "succeeded", "tick": 1}
        )
