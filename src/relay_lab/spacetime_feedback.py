"""Read authoritative robot memory before accepting the next planner context."""

from .orchestration import FeedbackCoordinator
from .spacetime import SpacetimeClient, SpacetimeConflict
from .types import AGENTS, Action, AgentId, Instruction, Observation


class SpacetimeFeedbackCoordinator:
    """Keep the grounded interpreter behind Spacetime's execution barrier.

    There is deliberately no automatic recovery of an interpreter or simulator.
    The active process must still agree with its last committed text. Newly
    delivered peer events pass through the interpreter, and the runner stores
    the resulting plan before inference.
    """

    def __init__(
        self,
        task_id: str,
        worker_id: str,
        client: SpacetimeClient,
        delegate: FeedbackCoordinator,
    ):
        self.task_id = task_id
        self.worker_id = worker_id
        self.client = client
        self.delegate = delegate

    @staticmethod
    def _context(value: dict) -> dict:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("text"), str)
            or len(value["text"]) > 850
            or type(value.get("critical")) is not bool
        ):
            raise SpacetimeConflict("Interpreter returned invalid robot feedback context")
        return {"text": value["text"], "critical": value["critical"]}

    def context(self, agent_id: AgentId, peer_events: list[dict], tick: int) -> dict:
        if agent_id not in AGENTS or type(tick) is not int or tick < 0:
            raise ValueError("Unknown robot or invalid decision tick")
        task = self.client.get_task(self.task_id)
        if task is None or (
            task.get("worker_id") != self.worker_id
            or task.get("status") != "running"
            or task.get("phase") != "ready"
            or task.get("next_tick") != tick
        ):
            raise SpacetimeConflict("Spacetime has not authorized planning this worker's next tick")
        stored = self.client.task_context(self.task_id)[agent_id]
        if stored["tick"] != tick - 1:
            raise SpacetimeConflict("Spacetime robot feedback does not match the preceding tick")
        authoritative = self._context(stored)
        local = self._context(self.delegate.context(agent_id, [], tick))
        if local["text"] != authoritative["text"]:
            raise SpacetimeConflict(
                "Interpreter memory differs from Spacetime's committed feedback"
            )
        critical = authoritative["critical"] or local["critical"]
        if peer_events:
            # This second call only adds validated, attributed peer observations
            # to the already checked memory; own text has not been re-created.
            local = self._context(self.delegate.context(agent_id, peer_events, tick))
            return {"text": local["text"], "critical": critical or local["critical"]}
        return {"text": authoritative["text"], "critical": critical}

    def digest(
        self,
        agent_id: AgentId,
        before: Observation,
        after: Observation,
        instruction: Instruction,
        action: Action,
        execution_events: list[dict],
    ) -> dict:
        # This may update interpreter memory only after the world moved. The
        # runner commits both reports atomically; a failure halts the episode.
        return self.delegate.digest(agent_id, before, after, instruction, action, execution_events)
