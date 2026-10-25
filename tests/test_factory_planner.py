import json

import pytest

from relay_lab.factory.planner import FactoryPlanner, parse_decision
from relay_lab.llm import Generation, LLMError


def candidate(robot, resource, job=None):
    return {
        "candidate_id": robot,
        "robot_id": robot,
        "resources": [robot, resource],
        "job_id": job,
        "description": "move workpiece",
        "next_stage": "loaded",
    }


def test_rejects_conflicting_or_invented_actions():
    choices = [candidate("a", "lane"), candidate("b", "lane")]
    for actions in (["a0", "a1"], ["a8"], ["a0", "a0"], []):
        with pytest.raises(ValueError):
            parse_decision(json.dumps({"actions": actions, "reason": "go"}), choices)
    chosen, reason = parse_decision('{"actions":["a1"],"reason":"free the lane"}', choices)
    assert chosen == [choices[1]] and reason == "free the lane"


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(list(messages))
        return Generation(next(self.outputs), 0.2, 100, 20)

    def close(self):
        pass


def test_real_planner_bounded_repair_and_verified_feedback():
    model = Model(
        ['{"actions":["x"],"reason":"go"}', '{"actions":["a0"],"reason":"carrier ready"}']
    )
    planner = FactoryPlanner(llm=model)
    result = planner.choose(
        {"jobs": {"j1": {"stage": "queued"}}}, [candidate("a", "lane")], [{"text": "docked"}]
    )
    assert result.reason == "carrier ready"
    assert planner.calls == 2
    assert "docked" in model.prompts[0][1]["content"]
    assert len(model.prompts[1]) == 4


def test_llm_never_falls_back_to_rules():
    planner = FactoryPlanner(llm=Model(["invalid", "invalid"]))
    with pytest.raises(LLMError, match="twice"):
        planner.choose({"jobs": {}}, [candidate("a", "lane")], [])
