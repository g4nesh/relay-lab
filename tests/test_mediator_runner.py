"""Exercise real mediator validation and physics with controlled LLM/DB boundaries.

The deterministic language-model double is deliberately not evidence of learned
planning quality. It checks that the runner actually uses model output, commits
observations before replanning, and stops on unavailable or invalid results.
"""

import json
from copy import deepcopy

import pytest

from relay_lab import environment, llm, mediator, runner, spacetime
from relay_lab.mediator import MediatorError
from relay_lab.types import AGENTS, Action


class AuditedDatabase:
    def __init__(self):
        self.backend_info = {"backend": "spacetimedb", "database": "test-mediator"}
        self.task = None
        self.plans = []
        self.commits = []
        self.abort_calls = 0
        self.fail_commit = False
        self.corrupt_context = False
        self.contexts = {agent: {"text": "", "critical": False, "tick": -1} for agent in AGENTS}

    def create_task(self, task_id, intent, spec, max_steps):
        self.task = {
            "task_id": task_id,
            "worker_id": "",
            "status": "queued",
            "phase": "ready",
            "next_tick": 0,
            "max_steps": max_steps,
        }
        return self.task.copy()

    def claim_task(self, task_id, worker_id):
        self.task.update(worker_id=worker_id, status="running")
        return self.task.copy()

    def get_task(self, task_id):
        return self.task.copy()

    def task_context(self, task_id):
        return deepcopy(self.contexts)

    def plan_tick(self, task_id, worker_id, tick, picker, packer):
        assert self.task["phase"] == "ready" and self.task["next_tick"] == tick
        self.task["phase"] = "planned"
        plans = {"picker": picker, "packer": packer}
        self.plans.append(deepcopy(plans))
        return {f"{agent}_json": json.dumps(plans[agent]) for agent in AGENTS}

    def prepare_tick(self, task_id, worker_id, tick, picker, packer):
        assert self.task["phase"] == "planned" and self.task["next_tick"] == tick
        self.task["phase"] = "executing"
        return {"picker_json": json.dumps(picker), "packer_json": json.dumps(packer)}

    def commit_tick(self, task_id, worker_id, tick, picker, packer, outcome):
        assert self.task["phase"] == "executing" and self.task["next_tick"] == tick
        if self.fail_commit:
            raise spacetime.SpacetimeUnavailable("Commit confirmation unavailable")
        self.commits.append(deepcopy({"picker": picker, "packer": packer, "outcome": outcome}))
        self.contexts = {
            agent: {"text": report["text"], "critical": report["critical"], "tick": tick}
            for agent, report in (("picker", picker), ("packer", packer))
        }
        if self.corrupt_context:
            self.contexts["picker"]["text"] = "Unverified replacement of committed evidence"
        status = (
            "succeeded"
            if outcome["success"]
            else "exhausted"
            if tick + 1 >= self.task["max_steps"]
            else "running"
        )
        self.task.update(
            next_tick=tick + 1, status=status, phase="ready" if status == "running" else "terminal"
        )
        return {}

    def abort_task(self, task_id, worker_id, reason):
        self.abort_calls += 1
        self.task.update(status="failed", phase="terminal")
        return self.task.copy()


class LegalPlanningDouble:
    """Return explicit legal JSON, so the real mediator still parses every plan."""

    def __init__(self, database):
        self.database = database
        self.states = []
        self.closed = False
        self.invalid = False
        self.mediator = None
        self.inputs = None

    def generate(self, messages):
        observations, _, contexts = self.inputs
        state = {
            "tick": observations["packer"].tick,
            "packing_order": list(observations["packer"].order),
            "robots": {
                agent: {
                    "holding": observations[agent].holding,
                    "observed_feedback": contexts[agent]["text"],
                }
                for agent in AGENTS
            },
            "verified_events": list(self.mediator._recent_events),
            "legal_actions": self.mediator._legal_actions(observations),
        }
        for robot in state["robots"].values():
            assert robot["observed_feedback"] in messages[1]["content"]
        assert self.database.task["phase"] == "ready"
        assert self.database.task["next_tick"] == state["tick"]
        assert len(self.database.commits) == state["tick"]
        self.states.append(deepcopy(state))
        if self.invalid:
            return llm.Generation("not valid JSON", 0.001, 100, 4)
        chosen = {}
        for agent in AGENTS:
            legal = state["legal_actions"][agent]
            holding = state["robots"][agent]["holding"]
            if holding:
                target = (
                    "relay"
                    if agent == "picker"
                    else f"slot_{state['packing_order'].index(holding)}"
                )
                preferred = f"place:{holding}:{target}"
                chosen[agent] = preferred if preferred in legal else "wait"
            else:
                chosen[agent] = next(
                    (
                        f"pick:{color}"
                        for color in state["packing_order"]
                        if f"pick:{color}" in legal
                    ),
                    "wait",
                )
        return llm.Generation(
            json.dumps({**chosen, "reason": "Test double chooses an explicit legal subgoal."}),
            0.001,
            100,
            40,
        )

    def close(self):
        self.closed = True


@pytest.fixture
def boundaries(monkeypatch):
    database = AuditedDatabase()
    language_model = LegalPlanningDouble(database)
    worlds = []

    class AuditedMediator(mediator.LLMMediator):
        def __init__(self, model):
            super().__init__(model)
            model.mediator = self

        def decide(self, observations, execution_events=None, contexts=None):
            self.llm.inputs = (observations, execution_events, contexts)
            return super().decide(observations, execution_events, contexts)

    class TrackedEnvironment(environment.RelayEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0
            self.did_close = False
            worlds.append(self)

        def step(self, actions):
            self.step_calls += 1
            return super().step(actions)

        def close(self):
            self.did_close = True
            return super().close()

    monkeypatch.setattr(environment, "RelayEnv", TrackedEnvironment)
    monkeypatch.setattr(spacetime.SpacetimeConfig, "from_env", lambda: object())
    monkeypatch.setattr(spacetime, "SpacetimeClient", lambda config: database)
    monkeypatch.setattr(llm, "LocalLLM", lambda **kwargs: language_model)
    monkeypatch.setattr(mediator, "LLMMediator", AuditedMediator)
    return database, language_model, worlds


def config(**kwargs):
    return runner.RunConfig(
        planner="llm", coordination="spacetime", policy="scripted", video=False, **kwargs
    )


def test_full_mediated_task_uses_committed_evidence_and_records_model_decisions(
    boundaries, tmp_path
):
    database, language_model, worlds = boundaries
    output = tmp_path / "mediated-task"
    summary = runner.run_episode(config(order=("mint", "gold", "coral")), output)
    assert summary["success"] and summary["packed"] == ["mint", "gold", "coral"]
    assert database.task["status"] == "succeeded"
    assert len(database.plans) == len(database.commits) == worlds[0].step_calls == summary["steps"]
    assert 1 < len(language_model.states) < summary["steps"] / 5
    for state in language_model.states:
        for agent in AGENTS:
            expected = (
                "" if state["tick"] == 0 else database.commits[state["tick"] - 1][agent]["text"]
            )
            assert state["robots"][agent]["observed_feedback"] == expected
        assert all(event["tick"] <= state["tick"] for event in state["verified_events"])
    for plan in database.plans:
        for agent in AGENTS:
            assert plan[agent]["mediator"]["fallback"] is False
            assert plan[agent]["mediator"]["tick"] <= plan[agent]["observation"]["tick"]
    records = [json.loads(line) for line in (output / "mediator.jsonl").read_text().splitlines()]
    assert len(records) == len(language_model.states) == summary["mediator"]["decisions"]
    assert all(record["attempts"][0]["text"] for record in records)
    assert worlds[0].did_close and language_model.closed


def test_invalid_model_output_aborts_before_any_action_without_rule_fallback(boundaries, tmp_path):
    database, language_model, worlds = boundaries
    language_model.invalid = True
    output = tmp_path / "invalid-model"
    with pytest.raises(MediatorError, match="Two LLM plans failed"):
        runner.run_episode(config(max_steps=5), output)
    assert len(language_model.states) == 2
    assert not database.plans and not database.commits and worlds[0].step_calls == 0
    assert database.abort_calls == 1 and database.task["status"] == "failed"
    assert not (output / "summary.json").exists()
    failed_record = json.loads((output / "mediator-last-record.json").read_text())
    assert len(failed_record["attempts"]) == 2
    assert all(attempt["validation_error"] for attempt in failed_record["attempts"])
    assert worlds[0].did_close and language_model.closed


def test_commit_failure_stops_before_next_mediator_or_policy_decision(boundaries, tmp_path):
    database, language_model, worlds = boundaries
    database.fail_commit = True
    output = tmp_path / "commit-failure"
    with pytest.raises(spacetime.SpacetimeUnavailable, match="Commit confirmation"):
        runner.run_episode(config(max_steps=5), output)
    assert len(language_model.states) == len(database.plans) == worlds[0].step_calls == 1
    assert not database.commits and database.abort_calls == 1
    assert (output / "execution.jsonl").read_text() == ""
    assert (output / "action-feedback.jsonl").read_text() == ""
    assert not (output / "summary.json").exists()
    assert worlds[0].did_close and language_model.closed


@pytest.mark.parametrize("feedback", [True, False])
def test_inconsistent_database_context_cannot_reach_mediator_or_another_step(
    boundaries, tmp_path, feedback
):
    database, language_model, worlds = boundaries
    database.corrupt_context = True
    with pytest.raises(spacetime.SpacetimeConflict, match="memory differs"):
        runner.run_episode(config(max_steps=5, feedback=feedback), tmp_path / "corrupt-context")
    assert len(language_model.states) == len(database.plans) == len(database.commits) == 1
    assert worlds[0].step_calls == database.abort_calls == 1
    assert worlds[0].did_close and language_model.closed


def test_feedback_text_ablation_preserves_evidence_but_excludes_judge_input(boundaries, tmp_path):
    database, language_model, _ = boundaries
    output = tmp_path / "text-ablation"
    summary = runner.run_episode(config(feedback=False), output)
    assert summary["success"]
    assert len(language_model.states) > 1
    # Structured events still govern safe handoffs and semantic state changes;
    # this flag ablates extra textual reports and judge-triggered interrupts.
    assert any(state["verified_events"] for state in language_model.states)
    assert all(
        robot["observed_feedback"] == ""
        for state in language_model.states
        for robot in state["robots"].values()
    )
    assert all(
        plan[agent]["instruction"]["feedback"] == ""
        and plan[agent]["instruction"]["replan"] is False
        for plan in database.plans
        for agent in AGENTS
    )
    # Durable observations and critical evidence still exist, even though they
    # were not fed into either neural model through the feedback context.
    assert all(commit[agent]["text"] for commit in database.commits for agent in AGENTS)
    assert any(commit[agent]["critical"] for commit in database.commits for agent in AGENTS)
    records = [json.loads(line) for line in (output / "mediator.jsonl").read_text().splitlines()]
    assert all("observed outcome report:" not in record["model_input"] for record in records)


def test_wall_clipping_does_not_call_mediator_or_replan_each_tick(
    boundaries, tmp_path, monkeypatch
):
    database, language_model, _ = boundaries

    class WallSeekingPolicy:
        needs_rgb = False

        def __init__(self, agent):
            self.agent = agent

        def act(self, observation, instruction):
            return Action((-0.03, 0, 0)) if self.agent == "picker" else Action()

    monkeypatch.setattr(runner, "ScriptedPolicy", WallSeekingPolicy)
    summary = runner.run_episode(config(max_steps=100), tmp_path / "clipped")
    assert not summary["success"] and summary["steps"] == 100
    assert 1 < len(language_model.states) <= 4
    interruptions = [plan for plan in database.plans if plan["picker"]["instruction"]["replan"]]
    assert len(interruptions) == 1
    assert language_model.closed
