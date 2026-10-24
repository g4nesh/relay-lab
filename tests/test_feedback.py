import json
from dataclasses import replace

import pytest

from relay_lab.feedback import ExecutionFeedback
from relay_lab.types import Action, Instruction, Observation

PICK = Instruction("pick", "Pick coral.", "coral")
PLACE = Instruction("place", "Place coral on relay.", "coral", "relay")
WAIT = Instruction("wait", "Clear the relay.")


def observation(agent="picker", **changes):
    return replace(
        Observation(
            tick=0,
            agent_id=agent,
            ee_position=(-0.4, 0.0, 0.3),
            holding=None,
            objects={"coral": (-0.4, 0.0, 0.03)},
            targets={"relay": (0.0, 0.0, 0.035), "slot_0": (0.48, -0.22, 0.035)},
        ),
        **changes,
    )


def types(events):
    return [event.kind for event in events]


def test_wrong_grasp_reports_actual_holding_and_blocked_not_requested_success():
    tracker = ExecutionFeedback("picker", "episode-7")
    before = observation()
    after = replace(before, tick=1, holding="mint")
    events = tracker.update(
        before, after, PICK, [{"type": "grasp", "agent": "picker", "object": "mint"}]
    )
    assert types(events) == ["started", "holding", "blocked"]
    assert events[1].object_id == "mint"
    assert events[1].evidence["matches_instruction"] is False
    assert events[2].object_id == "coral"
    assert events[2].reason == "wrong_object_held"
    assert tracker.update(after, replace(after, tick=2), PICK) == []


def test_pick_completion_needs_matching_attachment_and_simulator_event():
    before = observation()
    action = Action((0, 0, 0), -1)
    tracker = ExecutionFeedback("picker", "episode")
    events = tracker.update(before, replace(before, tick=1), PICK, action=action)
    assert types(events) == ["started"]
    after = replace(before, tick=2, holding="coral")
    events = tracker.update(
        replace(before, tick=1),
        after,
        PICK,
        [{"type": "grasp", "agent": "picker", "object": "coral"}],
    )
    assert types(events) == ["holding", "completed"]
    assert events[-1].scope == "instruction"
    assert events[-1].reason == "requested_object_grasped"
    assert tracker.update(after, replace(after, tick=3), PICK) == []


def release_at_relay(tracker, *, position=(0.0, 0.0, 0.03)):
    before = observation(holding="coral", ee_position=(0, 0, 0.035))
    after = replace(before, tick=1, holding=None, objects={"coral": position})
    events = tracker.update(
        before, after, PLACE, [{"type": "release", "agent": "picker", "object": "coral"}]
    )
    assert "completed" not in types(events)
    assert "transfer_ready" not in types(events)
    return after


def test_relay_readiness_requires_release_stationarity_and_picker_clear_across_wait():
    tracker = ExecutionFeedback("picker", "episode")
    after = release_at_relay(tracker)
    for tick in (2, 3):
        next_observation = replace(after, tick=tick)
        assert tracker.update(after, next_observation, WAIT) == []
        after = next_observation
    clear = replace(after, tick=4, ee_position=(0, 0, 0.20))
    events = tracker.update(after, clear, WAIT)
    assert types(events) == ["transfer_ready", "completed"]
    assert events[0].instruction_id == "picker:1"
    assert events[0].object_id == "coral"
    assert events[0].evidence["stationary_ticks"] == 3
    assert events[0].evidence["picker_clear"] is True
    assert tracker.update(clear, replace(clear, tick=5), WAIT) == []


@pytest.mark.parametrize("position", [(0, 0, 0.15), (0.07, 0, 0.03)])
def test_airborne_or_off_pad_object_never_announces_transfer(position):
    tracker = ExecutionFeedback("picker", "episode")
    after = release_at_relay(tracker, position=position)
    for tick in range(2, 8):
        clear = replace(after, tick=tick, ee_position=(-0.3, 0, 0.3))
        assert tracker.update(after, clear, WAIT) == []
        after = clear


def test_moving_cube_and_missing_or_peer_held_cube_do_not_claim_transfer():
    tracker = ExecutionFeedback("picker", "episode")
    after = release_at_relay(tracker)
    moving = replace(after, tick=2, ee_position=(-0.3, 0, 0.3), objects={"coral": (0.02, 0, 0.03)})
    assert tracker.update(after, moving, WAIT) == []
    invisible = replace(moving, tick=3, objects={})
    assert tracker.update(moving, invisible, WAIT) == []
    returned = replace(moving, tick=4)
    assert tracker.update(invisible, returned, WAIT) == []
    assert tracker.update(returned, replace(returned, tick=5), WAIT) == []


def test_already_present_cube_without_own_release_cannot_claim_place():
    tracker = ExecutionFeedback("picker", "episode")
    before = observation(objects={"coral": (0, 0, 0.03)})
    for tick in range(1, 4):
        after = replace(before, tick=tick)
        events = tracker.update(before, after, PLACE, action=Action(gripper=1))
        assert "transfer_ready" not in types(events)
        assert "completed" not in types(events)
        before = after


def test_slot_completion_uses_local_settled_identity_not_release_or_global_success():
    tracker = ExecutionFeedback("packer", "episode")
    instruction = Instruction("place", "Place coral in slot 1.", "coral", "slot_0")
    before = observation("packer", holding="coral", packed=(None, None, None))
    after = replace(before, tick=1, holding=None, packed=("mint", None, None))
    events = tracker.update(
        before,
        after,
        instruction,
        [
            {"type": "release", "agent": "packer", "object": "coral"},
            {"type": "success"},
            {"type": "packed", "slot": 0, "object": "coral"},
        ],
    )
    assert types(events) == ["started"]
    settled = replace(after, tick=2, packed=("coral", None, None))
    events = tracker.update(after, settled, WAIT)
    assert types(events) == ["completed"]
    assert events[0].evidence["packed_object"] == "coral"


def test_clipped_commands_report_executed_motion_and_do_not_spam():
    tracker = ExecutionFeedback("picker", "episode", stalled_ticks=2)
    before = observation(ee_position=(-0.72, 0, 0.035))
    action = Action((-0.03, 0, 0))
    for tick in range(1, 5):
        after = replace(before, tick=tick)
        events = tracker.update(before, after, PICK, action=action)
        if tick == 2:
            assert types(events) == ["blocked"]
            assert events[0].reason == "motion_limited"
            assert events[0].evidence["executed_delta"] == [0.0, 0.0, 0.0]
            assert events[0].evidence["requested_delta"] == [-0.03, 0.0, 0.0]
        elif tick > 2:
            assert events == []
        before = after


def test_missed_grasp_is_reported_once_and_waiting_is_not_a_stall():
    tracker = ExecutionFeedback("picker", "episode", stalled_ticks=2)
    before = observation()
    miss = [{"type": "grasp_missed", "agent": "picker"}]
    events = tracker.update(before, replace(before, tick=1), PICK, miss)
    assert [event.reason for event in events] == ["local_instruction_active", "grasp_missed"]
    before = replace(before, tick=1)
    for tick in range(2, 6):
        after = replace(before, tick=tick)
        assert tracker.update(before, after, WAIT) == []
        before = after


def test_events_are_json_safe_private_and_duplicate_transitions_are_idempotent():
    tracker = ExecutionFeedback("packer", "episode-7")
    before = observation("packer", order=("mint", "gold", "coral"))
    after = replace(before, tick=1)
    events = tracker.update(before, after, PICK, [{"type": "success"}])
    payload = json.dumps([event.to_dict() for event in events], allow_nan=False)
    assert "order" not in payload
    assert "gold" not in payload
    assert "success" not in payload
    assert events[0].task_id == "episode-7"
    assert tracker.update(before, after, PICK) == []
    with pytest.raises(ValueError, match="another agent"):
        tracker.update(observation(), replace(observation(), tick=2), PICK)
    tracker.reset()
    assert types(tracker.update(before, after, PICK)) == ["started"]


def test_real_scripted_episode_verifies_three_transfers_and_all_local_subtasks():
    from relay_lab.communication import LocalMessageBus
    from relay_lab.environment import RelayEnv
    from relay_lab.policies import RuleOrchestrator, ScriptedPolicy
    from relay_lab.types import AGENTS

    trackers = {agent: ExecutionFeedback(agent, "episode-7") for agent in AGENTS}
    planners = {agent: RuleOrchestrator(agent) for agent in AGENTS}
    policies = {agent: ScriptedPolicy(agent) for agent in AGENTS}
    bus = LocalMessageBus()
    reported = []
    with RelayEnv(seed=7) as env:
        for _ in range(400):
            before = {agent: env.observe(agent) for agent in AGENTS}
            decisions = {
                agent: planners[agent].decide(before[agent], bus.receive(agent, env.tick))
                for agent in AGENTS
            }
            for agent in AGENTS:
                if decisions[agent].message:
                    bus.send(agent, decisions[agent].message, env.tick)
            actions = {
                agent: policies[agent].act(before[agent], decisions[agent].instruction)
                for agent in AGENTS
            }
            result = env.step(actions)
            for agent in AGENTS:
                reported.extend(
                    trackers[agent].update(
                        before[agent],
                        env.observe(agent),
                        decisions[agent].instruction,
                        result["events"],
                        actions[agent],
                    )
                )
            if env.success:
                break
        assert env.success
        ready = [event for event in reported if event.kind == "transfer_ready"]
        assert [event.object_id for event in ready] == list(env.order)
        assert len([event for event in reported if event.kind == "completed"]) == 12
        assert not any(event.kind == "blocked" for event in reported)
