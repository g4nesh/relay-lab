"""Reusable task-planner / feedback-coordinator boundary for action models.

Task logic stays in the task adapter; measured feedback stays per robot and can
be committed behind the Spacetime execution barrier. Neither model actions nor
their textual descriptions prove success.
"""

from dataclasses import replace
from typing import Protocol

from .action_feedback import ActionFeedback
from .types import AGENTS, Action, AgentId, Decision, Instruction, Message, Observation


class TaskPlanner(Protocol):
    def decide(
        self, observation: Observation, inbox: list[Message], execution_events: list[dict]
    ) -> Decision: ...


class FeedbackCoordinator(Protocol):
    def context(self, agent_id: AgentId, peer_events: list[dict], tick: int) -> dict: ...

    def digest(
        self,
        agent_id: AgentId,
        before: Observation,
        after: Observation,
        instruction: Instruction,
        action: Action,
        execution_events: list[dict],
    ) -> dict: ...


class LocalFeedbackCoordinator:
    """Low-latency deterministic judge of observed simulator transitions."""

    def __init__(self, task_id: str):
        self._feedback = {agent: ActionFeedback(agent, task_id) for agent in AGENTS}
        self._critical = dict.fromkeys(AGENTS, False)

    def context(self, agent_id: AgentId, peer_events: list[dict], tick: int) -> dict:
        feedback = self._feedback[agent_id]
        critical = bool(feedback.receive_peer(peer_events, tick))
        result = {"text": feedback.context(), "critical": self._critical[agent_id] or critical}
        self._critical[agent_id] = False
        return result

    def digest(
        self,
        agent_id: AgentId,
        before: Observation,
        after: Observation,
        instruction: Instruction,
        action: Action,
        execution_events: list[dict],
    ) -> dict:
        feedback = self._feedback[agent_id]
        report = feedback.update(
            before, after, instruction, action, execution_events=execution_events
        )
        self._critical[agent_id] |= report.critical
        return {
            "report": report.to_dict(),
            "text": feedback.context(),
            "critical": self._critical[agent_id],
        }


class FeedbackOrchestrator:
    """Compose any compatible task planner with a grounded feedback service.

    `decide` supplies prior observations as language. `observe` must be called
    after executing the returned action, and never with predicted world state.
    Routine context changes enter the next model chunk; important changes can
    request early replanning through Instruction.replan.
    """

    def __init__(self, agent_id: AgentId, planner: TaskPlanner, coordinator: FeedbackCoordinator):
        if agent_id not in AGENTS:
            raise ValueError("Unknown robot")
        self.agent_id, self.planner, self.coordinator = agent_id, planner, coordinator
        self.last_context: dict = {"text": "", "critical": False}

    def decide(
        self, observation: Observation, inbox: list[Message], execution_events: list[dict]
    ) -> Decision:
        if observation.agent_id != self.agent_id:
            raise ValueError("Orchestrator received another agent's observation")
        self.last_context = self.coordinator.context(
            self.agent_id, execution_events, observation.tick
        )
        decision = self.planner.decide(observation, inbox, execution_events)
        return replace(
            decision,
            instruction=replace(
                decision.instruction,
                feedback=self.last_context["text"],
                replan=decision.instruction.replan or self.last_context["critical"],
            ),
        )

    def observe(
        self,
        before: Observation,
        after: Observation,
        instruction: Instruction,
        action: Action,
        execution_events: list[dict],
    ) -> dict:
        return self.coordinator.digest(
            self.agent_id, before, after, instruction, action, execution_events
        )
