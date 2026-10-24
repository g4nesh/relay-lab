"""Runner ordering checks around real MuJoCo steps and a controllable DB boundary."""

import json
from dataclasses import asdict

import pytest

from relay_lab import environment, runner, spacetime
from relay_lab.types import AGENTS, Action


class ControlledDatabase:
    def __init__(self, events):
        self.backend_info = {"backend": "spacetimedb", "database": "test"}
        self.events = events
        self.task = None
        self.fail_on = None
        self.use_alternate_readback = False
        self.claims = []
        self.contexts = {agent: {"text": "", "critical": False, "tick": -1} for agent in AGENTS}

    def create_task(self, task_id, intent, spec, max_steps):
        self.events.append("create")
        if self.task is None:
            self.task = {
                "task_id": task_id,
                "worker_id": "",
                "status": "queued",
                "phase": "ready",
                "next_tick": 0,
                "max_steps": max_steps,
            }
        return dict(self.task)

    def claim_task(self, task_id, worker_id):
        self.events.append("claim")
        self.claims.append(worker_id)
        if self.task["worker_id"] and self.task["worker_id"] != worker_id:
            raise spacetime.SpacetimeRejected("Another worker owns this task")
        self.task.update(worker_id=worker_id, status="running")
        return dict(self.task)

    def get_task(self, task_id):
        return dict(self.task)

    def task_context(self, task_id):
        return self.contexts

    def _barrier(self, name):
        self.events.append(name)
        if self.fail_on == name:
            raise spacetime.SpacetimeUnavailable(f"No confirmed {name} response")

    def plan_tick(self, task_id, worker_id, tick, picker, packer):
        self._barrier("plan")
        self.task["phase"] = "planned"
        plans = json.loads(json.dumps({"picker": picker, "packer": packer}))
        if self.use_alternate_readback:
            for agent in AGENTS:
                plans[agent]["instruction"]["text"] = f"Committed instruction for {agent}."
        return {f"{agent}_json": json.dumps(plans[agent]) for agent in AGENTS}

    def prepare_tick(self, task_id, worker_id, tick, picker, packer):
        self._barrier("prepare")
        self.task["phase"] = "executing"
        actions = {"picker": picker, "packer": packer}
        if self.use_alternate_readback:
            actions = {
                agent: {"action": {"delta": [0.012, 0.0, 0.0], "gripper": 1}} for agent in AGENTS
            }
        return {f"{agent}_json": json.dumps(actions[agent]) for agent in AGENTS}

    def commit_tick(self, task_id, worker_id, tick, picker, packer, outcome):
        self._barrier("commit")
        self.contexts = {
            agent: {"text": report["text"], "critical": report["critical"], "tick": tick}
            for agent, report in (("picker", picker), ("packer", packer))
        }
        status = "exhausted" if tick + 1 >= self.task["max_steps"] else "running"
        self.task.update(
            next_tick=tick + 1, status=status, phase="ready" if status == "running" else "terminal"
        )
        return {}

    def abort_task(self, task_id, worker_id, reason):
        self.events.append("abort")
        self.task.update(status="failed", phase="terminal")
        return dict(self.task)


@pytest.fixture
def boundary(monkeypatch):
    events, predictions, executed = [], [], []
    database = ControlledDatabase(events)

    class TrackedEnvironment(environment.RelayEnv):
        def step(self, actions):
            events.append("step")
            executed.append({agent: asdict(action) for agent, action in actions.items()})
            return super().step(actions)

        def close(self):
            events.append("close")
            return super().close()

    class TrackedPolicy:
        needs_rgb = False

        def __init__(self, agent):
            self.agent = agent

        def act(self, observation, instruction):
            events.append(f"predict:{self.agent}")
            predictions.append((self.agent, instruction))
            return Action()

    monkeypatch.setattr(environment, "RelayEnv", TrackedEnvironment)
    monkeypatch.setattr(runner, "ScriptedPolicy", TrackedPolicy)
    monkeypatch.setattr(spacetime.SpacetimeConfig, "from_env", lambda: object())
    monkeypatch.setattr(spacetime, "SpacetimeClient", lambda config: database)
    return database, events, predictions, executed


def config(**changes):
    return runner.RunConfig(
        coordination="spacetime", video=False, max_steps=changes.pop("max_steps", 2), **changes
    )


@pytest.mark.parametrize("failed_barrier", ["plan", "prepare"])
def test_unconfirmed_plan_or_actions_never_execute_physics(boundary, tmp_path, failed_barrier):
    database, events, predictions, executed = boundary
    database.fail_on = failed_barrier
    with pytest.raises(spacetime.SpacetimeUnavailable, match="No confirmed"):
        runner.run_episode(config(), tmp_path / failed_barrier)
    assert executed == []
    assert events[-2:] == ["abort", "close"]
    assert len(predictions) == (0 if failed_barrier == "plan" else 2)
    assert not (tmp_path / failed_barrier / "summary.json").exists()


def test_failed_result_commit_stops_before_a_second_physical_step(boundary, tmp_path):
    database, events, predictions, executed = boundary
    database.fail_on = "commit"
    output = tmp_path / "commit-failure"
    with pytest.raises(spacetime.SpacetimeUnavailable, match="No confirmed"):
        runner.run_episode(config(), output)
    assert len(executed) == 1
    assert len(predictions) == 2
    assert events[-4:] == ["step", "commit", "abort", "close"]
    assert not (output / "summary.json").exists()
    assert (output / "steps.jsonl").read_text() == ""
    # Uncommitted observations must not be advertised as durable execution.
    assert (output / "execution.jsonl").read_text() == ""


def test_models_and_physics_use_the_committed_readback(boundary, tmp_path):
    database, events, predictions, executed = boundary
    database.use_alternate_readback = True
    runner.run_episode(config(max_steps=1), tmp_path / "readback")
    assert events == [
        "create",
        "claim",
        "plan",
        "predict:picker",
        "predict:packer",
        "prepare",
        "step",
        "commit",
        "close",
    ]
    assert all(
        instruction.text == f"Committed instruction for {agent}."
        for agent, instruction in predictions
    )
    assert all(
        action == {"delta": [0.012, 0.0, 0.0], "gripper": 1} for action in executed[0].values()
    )


def test_committed_feedback_precedes_next_prediction(boundary, tmp_path):
    database, events, predictions, executed = boundary
    runner.run_episode(config(), tmp_path / "two-ticks")
    assert len(executed) == 2
    assert events == [
        "create",
        "claim",
        "plan",
        "predict:picker",
        "predict:packer",
        "prepare",
        "step",
        "commit",
        "plan",
        "predict:picker",
        "predict:packer",
        "prepare",
        "step",
        "commit",
        "close",
    ]
    assert all(not instruction.feedback for _, instruction in predictions[:2])
    assert all("observed move" in instruction.feedback for _, instruction in predictions[2:])
    assert database.task["next_tick"] == 2


def test_restart_gets_a_new_worker_and_cannot_reclaim_task(boundary, tmp_path):
    database, events, predictions, executed = boundary
    episode = config(max_steps=1, task_id="same-task")
    runner.run_episode(episode, tmp_path / "first")
    with pytest.raises(spacetime.SpacetimeRejected, match="Another worker"):
        runner.run_episode(episode, tmp_path / "second")
    assert len(database.claims) == 2
    assert database.claims[0] != database.claims[1]
    assert len(executed) == 1
    assert len(predictions) == 2
    assert events[-3:] == ["create", "claim", "close"]
