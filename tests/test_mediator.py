import json
from dataclasses import replace

import pytest

from relay_lab.llm import Generation, LLMError
from relay_lab.mediator import LLMMediator, MediatorError
from relay_lab.types import Observation


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def generate(self, messages):
        self.messages.append(list(messages))
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return Generation(result, 0.5, 500, 35)


def output(picker="pick:mint", packer="wait"):
    return json.dumps({"picker": picker, "packer": packer, "reason": "Advance the required order."})


def observations(tick=0):
    return {
        "picker": Observation(
            tick,
            "picker",
            (-0.54, 0, 0.30),
            None,
            {"mint": (-0.5, 0, 0.03), "gold": (-0.5, 0.2, 0.03)},
            {"relay": (0, 0, 0.035), "home": (-0.54, 0, 0.3)},
        ),
        "packer": Observation(
            tick,
            "packer",
            (0.48, 0, 0.30),
            None,
            {},
            {"relay": (0, 0, 0.035), "slot_0": (0.48, -0.22, 0.035)},
            packed=(None, None, None),
            order=("mint", "gold", "coral"),
        ),
    }


def event(kind="blocked", tick=1, **kwargs):
    return {
        "event_id": f"test:picker:{kind}:{tick}",
        "agent_id": "picker",
        "tick": tick,
        "kind": kind,
        "object_id": "mint",
        "target_id": "relay",
        "evidence": {"picker_clear": True},
        **kwargs,
    }


def test_model_selects_color_not_the_first_legal_action():
    model = FakeLLM([output("pick:gold")])
    mediator = LLMMediator(model)
    selected = mediator.decide(observations())
    assert selected["picker"].instruction.object_id == "gold"
    assert mediator.stats["fallback"] is False
    assert mediator.last_record["attempts"][0]["text"] == output("pick:gold")


def test_observed_motion_and_updated_text_do_not_reinvoke_model():
    model = FakeLLM([output()])
    mediator = LLMMediator(model)
    first = mediator.decide(observations())
    for tick in range(1, 25):
        state = observations(tick)
        state["picker"] = replace(state["picker"], ee_position=(-0.54, 0, 0.3 - tick * 0.008))
        assert mediator.decide(state, contexts={"picker": {"text": f"Moved at {tick}"}}) == first
    assert len(model.messages) == 1


def test_holding_transition_replans_immediately():
    model = FakeLLM([output(), output("place:mint:relay")])
    mediator = LLMMediator(model)
    mediator.decide(observations())
    state = observations(1)
    state["picker"] = replace(state["picker"], holding="mint")
    result = mediator.decide(state)
    assert result["picker"].instruction.kind == "place"
    assert mediator.last_record["trigger"] == "observed_transition"
    assert mediator.calls == 2


def test_packer_cannot_collect_until_verified_release_and_clearance():
    state = observations()
    for role in state:
        state[role] = replace(state[role], objects={"mint": (0, 0, 0.03)})
    mediator = LLMMediator(FakeLLM([output("wait", "pick:mint"), output("wait", "wait")]))
    result = mediator.decide(state)
    assert result["packer"].instruction.kind == "wait"
    assert "unavailable" in mediator.last_record["attempts"][0]["validation_error"]
    mediator.llm = FakeLLM([output("wait", "pick:mint")])
    state = {a: replace(obs, tick=2) for a, obs in state.items()}
    result = mediator.decide(state, {"picker": [event("transfer_ready", tick=2)]})
    assert result["packer"].instruction.kind == "pick"


def test_wrong_packing_slot_rejected_then_model_corrects_it():
    state = observations()
    state["packer"] = replace(state["packer"], holding="mint")
    model = FakeLLM(
        [output("pick:gold", "place:mint:slot_1"), output("pick:gold", "place:mint:slot_0")]
    )
    mediator = LLMMediator(model)
    result = mediator.decide(state)
    assert result["packer"].instruction.target_id == "slot_0"
    assert mediator.generation_calls == 2


def test_repeated_failures_do_not_replan_every_tick():
    model = FakeLLM([output(), output()])
    mediator = LLMMediator(model, failure_cooldown=15)
    mediator.decide(observations())
    for tick in range(1, 20):
        mediator.decide(observations(tick), {"picker": [event(tick=tick)]})
    assert mediator.calls == 2
    assert mediator.last_record["tick"] == 15


def test_one_failure_during_cooldown_is_deferred_until_eligible_tick():
    mediator = LLMMediator(FakeLLM([output(), output()]), failure_cooldown=5)
    mediator.decide(observations())
    mediator.decide(observations(1), {"picker": [event(tick=1)]})
    for tick in range(2, 5):
        mediator.decide(observations(tick))
        assert mediator.calls == 1
    mediator.decide(observations(5))
    assert mediator.calls == 2
    assert mediator.last_record["trigger"] == "verified_failure"
    for tick in range(6, 12):
        mediator.decide(observations(tick))
    assert mediator.calls == 2


def test_observed_transition_reconsideration_consumes_deferred_failure():
    mediator = LLMMediator(FakeLLM([output(), output("place:mint:relay")]), failure_cooldown=5)
    mediator.decide(observations())
    mediator.decide(observations(1), {"picker": [event(tick=1)]})
    state = observations(2)
    state["picker"] = replace(state["picker"], holding="mint")
    mediator.decide(state)
    assert mediator.calls == 2
    for tick in range(3, 10):
        mediator.decide({agent: replace(obs, tick=tick) for agent, obs in state.items()})
    assert mediator.calls == 2


def test_duplicate_evidence_cannot_trigger_a_second_call():
    model = FakeLLM([output(), output()])
    mediator = LLMMediator(model, failure_cooldown=2)
    mediator.decide(observations())
    failure = event(tick=2)
    mediator.decide(observations(2), {"picker": [failure]})
    mediator.decide(observations(4), {"picker": [failure]})
    assert mediator.calls == 2


def test_sustained_stall_requests_bounded_reconsideration():
    model = FakeLLM([output(), output()])
    mediator = LLMMediator(model, stall_ticks=5)
    mediator.decide(observations())
    for tick in range(1, 7):
        mediator.decide(observations(tick))
    assert mediator.calls == 2
    assert mediator.last_record["trigger"] == "sustained_stall"


@pytest.mark.parametrize(
    "bad", ["nope", "{}", "[]", output("delete:task"), output("place:mint:relay")]
)
def test_invalid_model_output_fails_closed_after_one_repair(bad):
    mediator = LLMMediator(FakeLLM([bad, bad]))
    with pytest.raises(MediatorError, match="Two LLM plans"):
        mediator.decide(observations())
    assert mediator.decisions == {}
    assert mediator.generation_calls == 2
    assert mediator.last_record["schema"] == "relay-lab.mediator.v1"
    assert mediator.last_record["call_id"] == 1
    assert mediator.last_record["status"] == "failed"
    assert mediator.last_record["instructions"] == {}
    assert "PICKER currently holds" in mediator.last_record["model_input"]


def test_model_failure_never_uses_rules_as_fallback():
    mediator = LLMMediator(FakeLLM([LLMError("GPU unavailable")]))
    with pytest.raises(MediatorError, match="no fallback"):
        mediator.decide(observations())
    assert mediator.decisions == {}
    assert mediator.last_record["call_id"] == 1
    assert mediator.last_record["status"] == "failed"
    assert mediator.last_record["attempts"] == [{"error": "LLMError: GPU unavailable"}]
    assert "PICKER currently holds" in mediator.last_record["model_input"]


def test_duplicate_json_keys_fail_even_when_last_value_is_legal():
    ambiguous = '{"picker":"place:mint:relay","picker":"pick:mint","packer":"wait","reason":"go"}'
    mediator = LLMMediator(FakeLLM([ambiguous, ambiguous]))
    with pytest.raises(MediatorError, match="Two LLM plans"):
        mediator.decide(observations())
    assert "unique keys" in mediator.last_record["attempts"][0]["validation_error"]
    assert mediator.decisions == {}


def test_failure_after_success_records_current_state_not_stale_plan():
    mediator = LLMMediator(FakeLLM([output(), OSError("worker pipe failed")]))
    mediator.decide(observations())
    state = observations(1)
    state["picker"] = replace(state["picker"], holding="mint")
    with pytest.raises(MediatorError, match="no fallback"):
        mediator.decide(state)
    record = mediator.last_record
    assert record["call_id"] == 2 and record["tick"] == 1
    assert record["status"] == "failed" and record["instructions"] == {}
    assert "PICKER currently holds: mint" in record["model_input"]
    assert mediator.calls == 1


def test_inference_failure_after_invalid_output_keeps_both_attempts():
    mediator = LLMMediator(FakeLLM(["not JSON", LLMError("timeout")]))
    with pytest.raises(MediatorError, match="no fallback"):
        mediator.decide(observations())
    assert len(mediator.last_record["attempts"]) == 2
    assert mediator.last_record["attempts"][0]["text"] == "not JSON"
    assert mediator.last_record["attempts"][1]["error"] == "LLMError: timeout"


def test_mismatched_tick_and_future_evidence_are_rejected():
    mediator = LLMMediator(FakeLLM([]))
    state = observations()
    state["packer"] = replace(state["packer"], tick=1)
    with pytest.raises(ValueError, match="share a simulation tick"):
        mediator.decide(state)
    with pytest.raises(ValueError, match="future"):
        mediator.decide(observations(), {"picker": [event(tick=10)]})
