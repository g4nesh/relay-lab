"""A complete rollout exercises the real physics/controller/transport boundary."""

import json

import numpy as np
import pytest

from relay_lab.runner import RunConfig, run_episode


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_scripted_rollout_completes_and_records_local_inputs(tmp_path, seed):
    result = run_episode(RunConfig(seed=seed, video=False), tmp_path / "episode")
    assert result["success"]
    assert result["packed"] == result["order"]
    assert result["steps"] < 600
    assert result["messages"]["delivered"] > 0
    lines = (tmp_path / "episode" / "steps.jsonl").read_text().splitlines()
    assert len(lines) == result["steps"]
    for line in lines:
        record = json.loads(line)
        picker = record["agents"]["picker"]
        assert picker["observation"]["order"] is None
        assert picker["observation"]["packed"] == []
        assert record["state"]["tick"] == record["tick"] + 1
        for agent in record["agents"].values():
            assert all(mail["sent_tick"] < record["tick"] for mail in agent["inbox"])
    assert (tmp_path / "episode" / "replay.html").exists()
    assert not (tmp_path / "episode" / "rollout.mp4").exists()


def test_paired_no_communication_uses_same_scene_and_drops_messages(tmp_path):
    on = run_episode(RunConfig(seed=7, video=False), tmp_path / "on")
    off = run_episode(RunConfig(seed=7, communication=False, video=False), tmp_path / "off")
    assert on["initial_state"] == off["initial_state"]
    assert on["success"] and not off["success"]
    assert off["messages"]["sent"] == off["messages"]["delivered"] == 0
    assert off["messages"]["dropped"] > 0
    assert off["steps"] == off["max_steps"]


def test_output_is_not_overwritten(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("existing work")
    with pytest.raises(ValueError, match="not empty"):
        run_episode(RunConfig(video=False), tmp_path)
    assert marker.read_text() == "existing work"


@pytest.mark.parametrize("environment_close_fails", [False, True])
def test_video_close_failure_still_closes_environment_and_transport(
    tmp_path, monkeypatch, environment_close_fails
):
    from relay_lab import environment, runner

    closed = []

    class TrackedEnvironment(environment.RelayEnv):
        def render(self, **kwargs):
            return np.zeros((32, 32, 3), dtype=np.uint8)

        def close(self):
            closed.append("environment")
            super().close()
            if environment_close_fails:
                raise OSError("environment close failed")

    class TrackedBus(runner.LocalMessageBus):
        def close(self):
            closed.append("transport")

    class BrokenVideoWriter:
        def append_data(self, frame):
            pass

        def close(self):
            closed.append("video")
            raise OSError("video close failed")

    monkeypatch.setattr(environment, "RelayEnv", TrackedEnvironment)
    monkeypatch.setattr(runner, "LocalMessageBus", TrackedBus)
    monkeypatch.setattr(runner.imageio, "get_writer", lambda *args, **kwargs: BrokenVideoWriter())
    with pytest.raises(OSError, match="close failed"):
        run_episode(RunConfig(max_steps=1, video=True), tmp_path / "failed-video")
    assert closed[-2:] == ["environment", "transport"]


def test_requested_task_order_and_recursive_feedback_are_recorded(tmp_path):
    order = ("mint", "gold", "coral")
    directory = tmp_path / "requested-order"
    summary = run_episode(RunConfig(order=order, video=False), directory)
    assert summary["success"]
    assert tuple(summary["order"]) == tuple(summary["packed"]) == order
    records = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    first = records[0]["agents"]["picker"]
    second = records[1]["agents"]["picker"]
    assert first["instruction"]["feedback"] == ""
    assert first["action_feedback"]["report"]["feedback"] in second["instruction"]["feedback"]
    assert (
        len((directory / "action-feedback.jsonl").read_text().splitlines()) == 2 * summary["steps"]
    )


def test_feedback_ablation_preserves_scene_and_disables_prompt_injection(tmp_path):
    on = run_episode(RunConfig(max_steps=3, video=False), tmp_path / "on")
    off = run_episode(RunConfig(max_steps=3, video=False, feedback=False), tmp_path / "off")
    assert on["initial_state"] == off["initial_state"]
    rows = [
        json.loads(line) for line in (tmp_path / "off" / "steps.jsonl").read_text().splitlines()
    ]
    assert all(
        not agent["instruction"]["feedback"] and agent["action_feedback"] is None
        for row in rows
        for agent in row["agents"].values()
    )


def test_order_override_does_not_change_source_scene():
    from relay_lab.environment import RelayEnv

    with RelayEnv(seed=7) as original, RelayEnv(seed=7, order=("mint", "coral", "gold")) as changed:
        assert original.state()["objects"] == changed.state()["objects"]
        assert changed.observe("picker").order is None
        assert changed.observe("packer").order == ("mint", "coral", "gold")
