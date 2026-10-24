"""Closed-loop plumbing tests; fake model responsiveness is not learned skill."""

from dataclasses import replace

import numpy as np
import pytest

from relay_lab.orchestration import FeedbackOrchestrator, LocalFeedbackCoordinator
from relay_lab.smolvla import SmolVLAPolicy
from relay_lab.types import Action, Decision, Instruction, Observation


class ConstantPlanner:
    def __init__(self):
        self.calls = []

    def decide(self, observation, inbox, execution_events):
        self.calls.append((observation, inbox, execution_events))
        return Decision(Instruction("pick", "Pick coral.", "coral"), "task coordination")


class ResponsiveBackend:
    """A deliberate test double that changes an action only on feedback text."""

    seed = 7
    last_inference_seconds = 0.001

    def __init__(self):
        self.prompts = []

    def predict(self, front, wrist, state, prompt, *, noise_seed):
        self.prompts.append(prompt)
        if "No attachment observed" in prompt or "none -> mint" in prompt:
            action = [0.2, 0, 0, 0, 0, 0, -1]  # Move and open after failure.
        else:
            action = [0, 0, 0, 0, 0, 0, 1]  # Initial close request.
        return np.tile(action, (8, 1)).astype(float)


def observation(agent="picker", **changes):
    return replace(
        Observation(
            tick=0,
            agent_id=agent,
            ee_position=(-0.4, 0, 0.3) if agent == "picker" else (0.4, 0, 0.3),
            holding=None,
            objects={},
            targets={},
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
        **changes,
    )


def peer_event(**changes):
    return {
        "task_id": "episode",
        "agent_id": "packer",
        "event_id": "episode:packer:1",
        "tick": 1,
        "kind": "holding",
        "object_id": "mint",
        "target_id": "slot_0",
        "reason": "observed_attachment",
        "scope": "instruction",
        **changes,
    }


def test_real_observed_missed_grasp_changes_the_next_model_input_and_action():
    from relay_lab.environment import RelayEnv

    backend = ResponsiveBackend()
    policy = SmolVLAPolicy("picker", backend, action_horizon=1)
    planner = ConstantPlanner()
    coordinator = LocalFeedbackCoordinator("episode")
    orchestrator = FeedbackOrchestrator("picker", planner, coordinator)
    with RelayEnv(seed=7) as env:
        # The adapter only needs image-shaped values for this fake model. Actual
        # motion/aperture/holding observations still come from MuJoCo.
        before = replace(
            env.observe("picker"), rgb=observation().rgb, wrist_rgb=observation().wrist_rgb
        )
        first = orchestrator.decide(before, [], [])
        action = policy.act(before, first.instruction)
        assert action.gripper == -1 and action.delta == (0, 0, 0)
        env.step({"picker": action})
        after = replace(env.observe("picker"), rgb=before.rgb, wrist_rgb=before.wrist_rgb)
        evidence = orchestrator.observe(before, after, first.instruction, action, [])
        assert evidence["report"]["holding_after"] is None
        assert "No attachment observed" in evidence["text"]
        assert "completed" not in evidence["text"]
        assert env.success is False
        next_decision = orchestrator.decide(after, [], [])
        correction = policy.act(after, next_decision.instruction)
    assert planner.calls[0][1:] == planner.calls[1][1:] == ([], [])
    assert first.instruction.text == next_decision.instruction.text
    assert backend.prompts[0] == "Pick coral."
    assert "Execution feedback: Tick 1, picker: observed" in backend.prompts[1]
    assert "No attachment observed" in backend.prompts[1]
    assert correction.gripper == 1 and correction.delta[0] > 0
    assert policy.last_prediction["model_prompt"] == backend.prompts[1]


def test_feedback_and_private_observations_stay_in_their_robot_context():
    coordinator = LocalFeedbackCoordinator("episode")
    picker = FeedbackOrchestrator("picker", ConstantPlanner(), coordinator)
    packer = FeedbackOrchestrator("packer", ConstantPlanner(), coordinator)
    backend = ResponsiveBackend()
    policies = {
        agent: SmolVLAPolicy(agent, backend, action_horizon=1) for agent in ("picker", "packer")
    }
    before = observation(order=("PRIVATE_ORDER",), objects={"HIDDEN": (7, 8, 9)})
    first = picker.decide(before, [], [])
    picker.observe(before, replace(before, tick=1, holding="mint"), first.instruction, Action(), [])
    picker_decision = picker.decide(replace(before, tick=1, holding="mint"), [], [])
    packer_decision = packer.decide(observation("packer", tick=1), [], [])
    policies["picker"].act(replace(before, tick=1, holding="mint"), picker_decision.instruction)
    policies["packer"].act(observation("packer", tick=1), packer_decision.instruction)
    assert "none -> mint" in backend.prompts[0]
    assert backend.prompts[1] == "Pick coral."
    assert "PRIVATE_ORDER" not in " ".join(backend.prompts)
    assert "HIDDEN" not in " ".join(backend.prompts)
    assert packer_decision.instruction.feedback == ""


def test_critical_evidence_invalidates_stale_chunk_once_and_then_consumes_normally():
    backend = ResponsiveBackend()
    policy = SmolVLAPolicy("picker", backend, action_horizon=5)
    orchestrator = FeedbackOrchestrator(
        "picker", ConstantPlanner(), LocalFeedbackCoordinator("episode")
    )
    before = observation()
    decision = orchestrator.decide(before, [], [])
    first_action = policy.act(before, decision.instruction)
    after = replace(before, tick=1, holding="mint")
    orchestrator.observe(before, after, decision.instruction, first_action, [])
    critical = orchestrator.decide(after, [], [])
    assert critical.instruction.replan is True
    assert policy.act(after, critical.instruction).gripper == 1
    assert len(backend.prompts) == 2
    ordinary = orchestrator.decide(after, [], [])
    assert ordinary.instruction.replan is False
    assert ordinary.instruction.feedback == critical.instruction.feedback
    assert policy.act(after, ordinary.instruction).gripper == 1
    assert len(backend.prompts) == 2
    assert policy.last_prediction["chunk_index"] == 1


def test_stationary_clipping_preserves_chunk_latency_and_interrupts_only_on_sustained_failure():
    class ClippedBackend(ResponsiveBackend):
        def predict(self, front, wrist, state, prompt, *, noise_seed):
            self.prompts.append(prompt)
            return np.tile([1, 0, 0, 0, 0, 0, 0], (8, 1)).astype(float)

    backend = ClippedBackend()
    policy = SmolVLAPolicy("picker", backend, action_horizon=5)
    orchestrator = FeedbackOrchestrator(
        "picker", ConstantPlanner(), LocalFeedbackCoordinator("episode")
    )
    before = observation()
    interruptions = []
    for tick in range(80):
        decision = orchestrator.decide(before, [], [])
        if decision.instruction.replan:
            interruptions.append(tick)
        action = policy.act(before, decision.instruction)
        after = replace(before, tick=tick + 1)
        evidence = orchestrator.observe(before, after, decision.instruction, action, [])
        assert evidence["report"]["motion_limited"]
        before = after
    assert len(interruptions) == 1
    assert interruptions[0] > 1
    # Queued actions continue between normal five-tick predictions. The old
    # judge made 80 predictions by treating every clipped delta as new evidence.
    assert len(backend.prompts) <= 17
    assert "cause unconfirmed" in backend.prompts[-1]


def test_planner_requested_replan_is_preserved_without_a_new_judge_interrupt():
    class RetryPlanner(ConstantPlanner):
        def decide(self, observation, inbox, execution_events):
            decision = super().decide(observation, inbox, execution_events)
            return replace(decision, instruction=replace(decision.instruction, replan=True))

    orchestrator = FeedbackOrchestrator(
        "picker", RetryPlanner(), LocalFeedbackCoordinator("episode")
    )
    assert orchestrator.decide(observation(), [], []).instruction.replan is True


def test_new_peer_evidence_enters_context_before_planning_and_is_consumed_once():
    calls = []

    class OrderedCoordinator(LocalFeedbackCoordinator):
        def context(self, agent_id, peer_events, tick):
            calls.append(("context", list(peer_events)))
            return super().context(agent_id, peer_events, tick)

    class OrderedPlanner(ConstantPlanner):
        def decide(self, observation, inbox, execution_events):
            calls.append(("planner", list(execution_events)))
            return super().decide(observation, inbox, execution_events)

    orchestrator = FeedbackOrchestrator("picker", OrderedPlanner(), OrderedCoordinator("episode"))
    event = peer_event()
    delivered = orchestrator.decide(observation(tick=2), [], [event])
    assert calls == [("context", [event]), ("planner", [event])]
    assert "Peer packer reports an observed attachment" in delivered.instruction.feedback
    assert "observed tick 1, delivered 2" in delivered.instruction.feedback
    assert delivered.instruction.replan is True
    retry = orchestrator.decide(observation(tick=2), [], [event])
    assert retry.instruction.replan is False
    assert retry.instruction.feedback == delivered.instruction.feedback
    assert delivered.message == "task coordination"


def test_future_peer_event_is_not_actionable_before_its_observation_tick():
    coordinator = LocalFeedbackCoordinator("episode")
    event = peer_event(tick=4)
    premature = coordinator.context("picker", [event], 3)
    assert premature == {"text": "", "critical": False}
    delivered = coordinator.context("picker", [event], 4)
    assert delivered["critical"] is True
    assert "observed tick 4" in delivered["text"]


def test_orchestrator_rejects_other_robot_observations_for_decision_and_digest():
    orchestrator = FeedbackOrchestrator(
        "picker", ConstantPlanner(), LocalFeedbackCoordinator("episode")
    )
    with pytest.raises(ValueError, match="another agent"):
        orchestrator.decide(observation("packer"), [], [])
    with pytest.raises(ValueError, match="another agent"):
        orchestrator.observe(
            observation("packer"),
            observation("packer", tick=1),
            Instruction("wait", "Wait."),
            Action(),
            [],
        )
