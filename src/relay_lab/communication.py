"""A small, deterministic local transport with delayed, private inboxes."""

from collections import deque

from .types import AGENTS, AgentId, Message


class LocalMessageBus:
    """Deliver to the other agent no earlier than the next simulation tick.

    This delay makes a rollout independent of which orchestrator is called first.
    Queue and transcript bounds keep an unattended rollout from growing forever.
    """

    def __init__(
        self,
        enabled: bool = True,
        latency_ticks: int = 1,
        max_pending: int = 256,
        max_history: int = 2048,
        max_message_length: int = 280,
    ) -> None:
        for name, value in (
            ("latency_ticks", latency_ticks),
            ("max_pending", max_pending),
            ("max_history", max_history),
            ("max_message_length", max_message_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.enabled = enabled
        self.latency_ticks = latency_ticks
        self.max_pending = max_pending
        self.max_message_length = max_message_length
        self._pending: deque[Message] = deque()
        self._history: deque[Message] = deque(maxlen=max_history)
        self.sent = 0
        self.delivered = 0
        self.dropped = 0

    @property
    def transcript(self) -> list[Message]:
        """A bounded snapshot of accepted messages, including in-flight messages."""
        return list(self._history)

    def reset(self) -> None:
        self._pending.clear()
        self._history.clear()
        self.sent = self.delivered = self.dropped = 0

    @staticmethod
    def _validate_tick(tick: int) -> None:
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a non-negative integer")

    def send(self, sender: AgentId, text: str, tick: int) -> Message | None:
        if sender not in AGENTS:
            raise ValueError(f"Unknown sender: {sender!r}")
        self._validate_tick(tick)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message text must be a non-empty string")
        if len(text) > self.max_message_length:
            raise ValueError(f"Message exceeds {self.max_message_length} characters")
        if not self.enabled or len(self._pending) >= self.max_pending:
            self.dropped += 1
            return None
        recipient: AgentId = "packer" if sender == "picker" else "picker"
        message = Message(sender, recipient, text, tick, tick + self.latency_ticks)
        self._pending.append(message)
        self._history.append(message)
        self.sent += 1
        return message

    def receive(self, agent_id: AgentId, tick: int) -> list[Message]:
        if agent_id not in AGENTS:
            raise ValueError(f"Unknown recipient: {agent_id!r}")
        self._validate_tick(tick)
        received = [
            message
            for message in self._pending
            if message.recipient == agent_id and message.deliver_tick <= tick
        ]
        self._pending = deque(
            message
            for message in self._pending
            if not (message.recipient == agent_id and message.deliver_tick <= tick)
        )
        self.delivered += len(received)
        return received
