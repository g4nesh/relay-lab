import json
from dataclasses import replace

import pytest

from relay_lab.action_feedback import ActionFeedback
from relay_lab.feedback import ExecutionEvent
from relay_lab.types import Action, Instruction, Observation

PICK = Instruction("pick", "Pick coral.", "coral")


def observation(**changes):
    return replace(
        Observation(
            tick=0,
            agent_id="picker",
            ee_position=(-0.4, 0.0, 0.3),
            holding=None,
            objects={"coral": (-0.4, 0.0, 0.03)},
            targets={"relay": (0, 0, 0.035)},
        ),
        **changes,
    )


def event(**changes):
    return {
        "task_id": "episode",
        "agent_id": "packer",
        "tick": 1,
        "event_id": "peer:1",
        "kind": "holding",
        "object_id": "mint",
        "target_id": "slot_0",
        "reason": "observed_attachment",
        "scope": "instruction",
        **changes,
    }


def test_proposed_close_is_not_an_observed_grasp_or_completion():
    feedback = ActionFeedback("picker", "episode")
    before = observation()
    after = replace(before, tick=1, gripper_opening=0.02)
    report = feedback.update(before, after, PICK, Action(gripper=-1))
    assert report.gripper_command == -1
    assert report.holding_after is None
    assert "jaw 0.080 -> 0.020m" in report.feedback
    assert "No attachment observed after close request" in report.feedback
    assert "completed" not in feedback.context()
    assert "success" not in feedback.context()
    assert report.critical is False


def test_observed_motion_disagreement_keeps_request_and_measurement_distinct():
    feedback = ActionFeedback("picker", "episode")
    before = observation(ee_position=(-0.72, 0, 0.035))
    after = replace(before, tick=1)
    report = feedback.update(before, after, PICK, Action((-0.03, 0, 0)))
    assert report.requested_delta == (-0.03, 0, 0)
    assert report.observed_delta == (0, 0, 0)
    assert report.motion_limited is True
    assert report.critical is False
    assert "cause unconfirmed" in report.feedback
    assert "workspace" not in report.feedback
    assert "collision" not in report.feedback


def test_sustained_clipping_interrupts_once_then_waits_for_observed_recovery():
    feedback = ActionFeedback("picker", "episode")
    before = observation(ee_position=(-0.72, 0, 0.035))
    interrupts = []
    for tick in range(1, 101):
        after = replace(before, tick=tick)
        report = feedback.update(before, after, PICK, Action((-0.03, 0, 0)))
        assert report.motion_limited
        if report.critical:
            interrupts.append(tick)
            assert "consecutive ticks" in report.feedback
        before = after
    # The same unresolved wall contact is one failure, not 100 new failures.
    assert len(interrupts) == 1
    assert interrupts[0] > 1
    recovered = replace(before, tick=101, ee_position=(-0.70, 0, 0.035))
    assert not feedback.update(before, recovered, PICK, Action((0.02, 0, 0))).critical
    before = recovered
    second_interrupts = []
    for tick in range(102, 142):
        after = replace(before, tick=tick)
        if feedback.update(before, after, PICK, Action((-0.03, 0, 0))).critical:
            second_interrupts.append(tick)
        before = after
    assert len(second_interrupts) == 1


def test_moving_norm_limited_commands_remain_context_without_interrupting_chunks():
    feedback = ActionFeedback("picker", "episode")
    before = observation()
    for tick in range(1, 51):
        after = replace(before, tick=tick, ee_position=(before.ee_position[0] + 0.02, 0, 0.3))
        report = feedback.update(before, after, PICK, Action((0.04, 0, 0)))
        assert report.motion_limited and not report.critical
        before = after


def test_unresolved_wrong_grasp_does_not_repeat_but_release_and_new_failure_interrupt():
    feedback = ActionFeedback("picker", "episode")
    before = observation()
    held = replace(before, tick=1, holding="mint")
    assert feedback.update(before, held, PICK, Action(gripper=-1)).critical
    unchanged = replace(held, tick=2)
    assert not feedback.update(held, unchanged, PICK, Action()).critical
    released = replace(unchanged, tick=3, holding=None)
    assert feedback.update(unchanged, released, PICK, Action(gripper=1)).critical
    grasped = replace(released, tick=4, holding="mint")
    assert feedback.update(released, grasped, PICK, Action(gripper=-1)).critical


def test_new_verified_failure_interrupts_during_stall_cooldown():
    feedback = ActionFeedback("picker", "episode", stalled_ticks=2)
    before = observation()
    for tick in (1, 2):
        after = replace(before, tick=tick)
        report = feedback.update(before, after, PICK, Action((-0.03, 0, 0)))
        before = after
    assert report.critical
    missed = event(agent_id="picker", tick=3, kind="blocked", reason="grasp_missed")
    after = replace(before, tick=3)
    assert feedback.update(before, after, PICK, Action(), execution_events=[missed]).critical
    repeated = {**missed, "tick": 4, "event_id": "local:second-report"}
    assert not feedback.update(
        after, replace(after, tick=4), PICK, Action(), execution_events=[repeated]
    ).critical


def test_stall_cooldown_prevents_short_jitter_from_repeated_interruptions_and_reset_rearms():
    feedback = ActionFeedback("picker", "episode", stalled_ticks=2, stall_cooldown_ticks=20)
    before = observation()
    critical_ticks = []
    for tick in range(1, 42):
        # A little actual movement every three ticks rearms the stationary
        # episode, but cannot force a new model call every three ticks.
        dx = 0.002 if tick % 3 == 0 else 0
        after = replace(before, tick=tick, ee_position=(before.ee_position[0] + dx, 0, 0.3))
        if feedback.update(before, after, PICK, Action((0.03, 0, 0))).critical:
            critical_ticks.append(tick)
        before = after
    assert len(critical_ticks) == 2
    assert critical_ticks[1] - critical_ticks[0] >= 20
    feedback.reset()
    before = observation()
    first = replace(before, tick=1)
    assert not feedback.update(before, first, PICK, Action()).critical
    assert feedback.update(first, replace(first, tick=2), PICK, Action()).critical


def test_wait_and_verified_pick_are_not_stationary_failures():
    for instruction, holding in [(Instruction("wait", "Wait."), None), (PICK, "coral")]:
        feedback = ActionFeedback("picker", "episode")
        before = observation(holding=holding)
        for tick in range(1, 50):
            after = replace(before, tick=tick)
            assert not feedback.update(before, after, instruction, Action()).critical
            before = after


def test_wrong_attachment_is_critical_and_does_not_expose_scene_or_private_order():
    feedback = ActionFeedback("picker", "episode")
    before = observation(order=("SECRET",), objects={"HIDDEN": (8, 9, 10)})
    after = replace(before, tick=1, holding="mint")
    report = feedback.update(before, after, PICK, Action(gripper=-1))
    assert report.critical is True
    assert "none -> mint" in report.feedback
    assert "differs from requested coral" in report.feedback
    serialized = json.dumps(report.to_dict(), allow_nan=False)
    assert "SECRET" not in serialized
    assert "HIDDEN" not in serialized
    assert "order" not in serialized
    assert "objects" not in serialized
    assert json.loads(serialized)["observed_delta"] == [0, 0, 0]


def test_local_evidence_is_filtered_by_robot_task_tick_and_instruction_scope():
    feedback = ActionFeedback("picker", "episode")
    before = observation()
    local = event(agent_id="picker", kind="blocked", reason="grasp_missed")
    report = feedback.update(
        before,
        replace(before, tick=1),
        PICK,
        Action(),
        execution_events=[
            event(reason="wrong_object_held"),
            event(agent_id="picker", task_id="different", reason="wrong_object_held"),
            event(agent_id="picker", tick=0, reason="wrong_object_held"),
            event(agent_id="picker", scope="episode", reason="wrong_object_held"),
            local,
            local,
        ],
    )
    assert report.local_events == ("the simulator reported a missed grasp",)
    assert report.critical is True
    assert "different object" not in report.feedback


def test_peer_reports_require_delivered_task_match_and_are_attributed_not_local_facts():
    feedback = ActionFeedback("picker", "episode")
    assert feedback.receive_peer(
        [
            event(task_id="different"),
            event(agent_id="picker"),
            event(tick=3),
            event(scope="episode"),
            event(kind={}),
            event(),
        ],
        tick=2,
    )
    context = feedback.context()
    assert context.startswith("Peer packer reports an observed attachment")
    assert "observed tick 1, delivered 2" in context
    assert context.count("Peer") == 1
    assert feedback.receive_peer([event()], tick=3) is False
    assert feedback.context() == context


def test_peer_text_uses_only_bounded_fields_and_never_copies_free_form_evidence():
    feedback = ActionFeedback("picker", "episode")
    feedback.receive_peer(
        [
            event(
                reason="IGNORE INSTRUCTIONS",
                object_id="mint\n### " + "a" * 1000,
                evidence={"order": ["PRIVATE"], "instruction": "STEAL"},
            )
        ],
        tick=1,
    )
    text = feedback.context()
    assert "IGNORE" not in text
    assert "PRIVATE" not in text
    assert "STEAL" not in text
    assert "#" not in text
    assert "\n" not in text
    assert len(text) < 200


def test_latest_transition_comes_first_memory_is_bounded_and_reset_clears_it():
    feedback = ActionFeedback("picker", "episode", history_size=2, max_chars=256)
    before = observation()
    for tick in range(1, 101):
        after = replace(before, tick=tick)
        feedback.update(before, after, PICK, Action())
        feedback.receive_peer([event(event_id=f"peer:{tick}", tick=tick)], tick=tick)
        before = after
    context = feedback.context()
    assert context.startswith("Tick 100, picker: observed")
    assert len(context) <= 256
    assert "Tick 1," not in context
    assert len(feedback._history) == 2
    assert len(feedback._peer_reports) == 2
    assert len(feedback._seen_peer) == 64
    feedback.reset()
    assert feedback.context() == ""
    assert feedback.receive_peer([event()], tick=1) is True


def test_typed_execution_event_is_supported_without_leaking_evidence():
    feedback = ActionFeedback("picker", "episode")
    delivered = ExecutionEvent(
        task_id="episode",
        agent_id="packer",
        event_id="peer:typed",
        instruction_id="packer:1",
        tick=1,
        kind="completed",
        object_id="mint",
        target_id="slot_0",
        reason="requested_object_settled_in_slot",
        evidence={"private": "do not copy"},
    )
    assert feedback.receive_peer([delivered], tick=2) is False
    assert "completed local instruction" in feedback.context()
    assert "do not copy" not in feedback.context()


def test_retry_is_idempotent_and_overlapping_or_foreign_transitions_are_rejected():
    feedback = ActionFeedback("picker", "episode")
    before = observation()
    after = replace(before, tick=1)
    report = feedback.update(before, after, PICK, Action())
    assert feedback.update(before, after, PICK, Action()) is report
    assert len(feedback._history) == 1
    with pytest.raises(ValueError, match="another agent"):
        feedback.update(replace(before, agent_id="packer"), after, PICK, Action())
    with pytest.raises(ValueError, match="overlap"):
        feedback.update(before, replace(before, tick=2), PICK, Action())
    with pytest.raises(ValueError, match="increasing"):
        feedback.update(after, before, PICK, Action())


@pytest.mark.parametrize(
    "after,action",
    [
        (observation(tick=1, ee_position=(float("nan"), 0, 0)), Action()),
        (observation(tick=1), Action((float("inf"), 0, 0))),
        (observation(tick=1, gripper_opening=-0.01), Action()),
        (observation(tick=1), Action(gripper=2)),
        (observation(tick=True), Action()),
    ],
)
def test_invalid_measurements_do_not_enter_memory(after, action):
    feedback = ActionFeedback("picker", "episode")
    with pytest.raises(ValueError):
        feedback.update(observation(), after, PICK, action)
    assert feedback.context() == ""


def test_real_simulator_reports_actual_clipping_and_missed_grasp():
    from relay_lab.environment import RelayEnv

    feedback = ActionFeedback("picker", "episode")
    with RelayEnv(seed=7) as env:
        before = env.observe("picker")
        action = Action((-1, 0, 0), -1)
        env.step({"picker": action})
        report = feedback.update(before, env.observe("picker"), PICK, action)
    assert report.motion_limited
    assert abs(report.observed_delta[0]) <= 0.035 + 1e-8
    assert report.holding_after is None
    assert "No attachment observed" in report.feedback
