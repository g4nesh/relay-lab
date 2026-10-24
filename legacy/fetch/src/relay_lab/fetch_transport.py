"""Fetch uAgents peer messaging and grounded action orchestration.

This local transport exercises the SDK's serialization, typed handlers, contexts,
and acknowledgement round trips. An optional third agent turns measured robot
transitions into private feedback for the next model decision. These agents do
not register or send traffic to Agentverse; the mailbox gateway exposes the team.
"""

import asyncio
import concurrent.futures
import json
import logging
import math
import secrets
import threading
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import fields
from functools import lru_cache
from importlib.metadata import version
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from .communication import LocalMessageBus
from .types import AGENTS, Action, AgentId, Instruction, Message, Observation

_ORCHESTRATION_BYTES = 32768


class FetchTransportError(RuntimeError):
    """A Fetch message did not complete its verified acknowledgement round trip."""


@lru_cache(maxsize=1)
def _sdk():
    """Keep Fetch optional for ordinary simulator imports."""
    try:
        from uagents import Agent, Model, Protocol
        from uagents.dispatch import dispatcher
        from uagents.resolver import Resolver
        from uagents_core.registration import AgentRegistrationPolicy
        from uagents_core.types import DeliveryStatus
    except ImportError as exc:
        raise FetchTransportError(
            "Fetch transport requires the fetch extra: uv sync --extra fetch"
        ) from exc

    class NoRegistration(AgentRegistrationPolicy):
        async def register(self, *args, **kwargs):
            return None

    class LocalOnlyResolver(Resolver):
        async def resolve(self, destination):
            # A lost local recipient must fail, never resolve to a public address.
            return None, []

    class PeerMessage(Model):
        task_id: str
        message_id: str
        sender: str
        recipient: str
        kind: str
        text: str
        sent_tick: int
        deliver_tick: int
        event_json: str = ""

    class PeerAcknowledgement(Model):
        task_id: str
        message_id: str

    class OrchestrationRequest(Model):
        task_id: str
        request_id: str
        agent_id: str
        sequence: int
        phase: str
        tick: int
        payload_json: str

    class OrchestrationReply(Model):
        task_id: str
        request_id: str
        agent_id: str
        sequence: int
        phase: str
        tick: int
        result_json: str

    return (
        Agent,
        Protocol,
        dispatcher,
        NoRegistration,
        LocalOnlyResolver,
        PeerMessage,
        PeerAcknowledgement,
        DeliveryStatus,
        OrchestrationRequest,
        OrchestrationReply,
    )


class FetchMessageBus(LocalMessageBus):
    """Drop-in bus backed by real, private Fetch SDK agents.

    ``send`` waits for the receiving SDK handler's acknowledgement. Only then
    can ``receive`` expose the decoded message, at its requested simulation tick.
    A timeout aborts the transport; there is no direct-message fallback.
    With ``orchestration=True``, a third SDK agent handles action interpretation
    in its own typed request/reply protocol, including when peer mail is disabled.
    """

    def __init__(
        self,
        enabled: bool = True,
        latency_ticks: int = 1,
        max_pending: int = 256,
        max_history: int = 2048,
        max_message_length: int = 280,
        *,
        task_id: str | None = None,
        seed: str | None = None,
        timeout_seconds: float = 5.0,
        orchestration: bool = False,
    ):
        super().__init__(enabled, latency_ticks, max_pending, max_history, max_message_length)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < timeout_seconds <= 60
        ):
            raise ValueError("timeout_seconds must be between 0 and 60")
        self.task_id = task_id or str(uuid4())
        if not isinstance(self.task_id, str) or not 1 <= len(self.task_id) <= 128:
            raise ValueError("task_id must contain 1 to 128 characters")
        self._seed = seed or secrets.token_hex(32)
        self._session = uuid5(NAMESPACE_URL, "relay-lab:" + self.task_id)
        self.timeout_seconds = float(timeout_seconds)
        self.orchestration = bool(orchestration)
        self._max_history = max_history
        self._loop = None
        self._thread = None
        self._agents = {}
        self._contexts = {}
        self._acks = {}
        self._seen = OrderedDict()
        self._event_pending = deque()
        self._event_history = deque(maxlen=max_history)
        self._wire_history = deque(maxlen=max_history)
        self._started = threading.Event()
        self._closed = False
        self._failure = None
        self._lock = threading.RLock()
        self._handler_deliveries = self._acknowledged = self._duplicates = 0
        self._rejected = self._event_sent = self._event_delivered = 0
        self._current_tick = 0
        self._orchestration_waiting = {}
        self._orchestration_seen = OrderedDict()
        self._orchestration_history = deque(maxlen=max_history)
        self._action_feedback = {}
        self._orchestration_sequence = {agent_id: 0 for agent_id in AGENTS}
        self._orchestration_received_sequence = {agent_id: 0 for agent_id in AGENTS}
        self._orchestration_ticks = {agent_id: 0 for agent_id in AGENTS}
        self._orchestration_critical = {agent_id: False for agent_id in AGENTS}
        self._orchestration_requests = self._orchestration_receipts = 0
        self._orchestration_duplicates = self._orchestration_rejected = 0

    def start(self):
        with self._lock:
            self._check_healthy()
            if self._thread is not None or not (self.enabled or self.orchestration):
                return self
            self._thread = threading.Thread(
                target=self._thread_main, name="relay-fetch-peers", daemon=True
            )
            self._thread.start()
            if not self._started.wait(self.timeout_seconds):
                self._failure = "Fetch SDK agents did not start before the timeout"
            self._check_healthy()
        return self

    def _check_healthy(self):
        if self._closed:
            raise FetchTransportError("Fetch transport is closed")
        if self._failure:
            raise FetchTransportError(self._failure)

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._initialize())
            self._started.set()
            if not self._closed:
                self._loop.run_forever()
        except Exception as exc:  # noqa: BLE001 -- propagate thread failures to the caller
            self._failure = f"Fetch SDK startup failed: {exc}"
            self._started.set()
        finally:
            tasks = asyncio.all_tasks(self._loop)
            for task in tasks:
                task.cancel()
            if tasks:
                self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            if self._agents:
                dispatcher = _sdk()[2]
                for agent in self._agents.values():
                    dispatcher.unregister(agent.address, agent)
            self._loop.close()

    async def _initialize(self):
        Agent, Protocol, dispatcher, NoRegistration, LocalOnlyResolver, PeerMessage, PeerAck = (
            _sdk()[:7]
        )
        OrchestrationRequest, OrchestrationReply = _sdk()[8:10]
        if self.orchestration:
            from .action_feedback import ActionFeedback

            self._action_feedback = {
                agent_id: ActionFeedback(agent_id, self.task_id) for agent_id in AGENTS
            }
        actor_ids = (*AGENTS, "coordinator") if self.orchestration else AGENTS
        for agent_id in actor_ids:
            agent = Agent(
                name=f"Relay Lab {agent_id}",
                seed=f"{self._seed}:{agent_id}",
                loop=self._loop,
                registration_policy=NoRegistration(),
                resolve=LocalOnlyResolver(),
                enable_agent_inspector=False,
                publish_agent_details=False,
                mark_inactive_on_shutdown=False,
                report_events=False,
                log_level=logging.ERROR,
            )
            self._agents[agent_id] = agent
            if len(dispatcher.sinks.get(agent.address, set())) != 1:
                raise FetchTransportError("A Fetch peer identity is already active in this process")

            def bind_peer_handlers(recipient, protocol):
                @protocol.on_message(PeerMessage, replies=PeerAck)
                async def message_handler(ctx, sender, message):
                    await self._handle_peer_message(recipient, ctx, sender, message)

                @protocol.on_message(PeerAck)
                async def acknowledgement_handler(ctx, sender, acknowledgement):
                    self._handle_acknowledgement(recipient, ctx, sender, acknowledgement)

            if agent_id in AGENTS:
                protocol = Protocol(name="RelayLabExecutionFeedback", version="1.0.0")
                bind_peer_handlers(agent_id, protocol)
                agent.include(protocol, publish_manifest=False)
            if self.orchestration:
                protocol = Protocol(name="RelayLabActionOrchestration", version="1.0.0")

                def bind_orchestration_handlers(recipient, protocol):
                    if recipient == "coordinator":

                        @protocol.on_message(OrchestrationRequest, replies=OrchestrationReply)
                        async def request_handler(ctx, sender, request):
                            await self._handle_orchestration_request(ctx, sender, request)
                    else:

                        @protocol.on_message(OrchestrationReply)
                        async def reply_handler(ctx, sender, reply):
                            self._handle_orchestration_reply(recipient, ctx, sender, reply)

                bind_orchestration_handlers(agent_id, protocol)
                agent.include(protocol, publish_manifest=False)

            # The pinned SDK's setup()/run_async() unconditionally contact the
            # Almanac on startup. Start only its real local message receivers.
            # _build_context and its session field are pinned SDK-private hooks;
            # keep them covered by integration tests when upgrading Fetch.
            context = agent._build_context()
            context._session = self._session
            self._contexts[agent_id] = context
            agent.start_message_receivers()

    async def _handle_peer_message(self, recipient, ctx, sender, message):
        PeerAck = _sdk()[6]
        expected = "packer" if recipient == "picker" else "picker"
        valid = (
            sender == self._agents[expected].address
            and message.sender == expected
            and message.recipient == recipient
            and message.task_id == self.task_id
            and ctx.session == self._session
            and message.kind in {"coordination", "execution-feedback"}
            and message.deliver_tick == message.sent_tick + self.latency_ticks
            and message.sent_tick >= 0
        )
        try:
            UUID(message.message_id)
        except (ValueError, TypeError):
            valid = False
        if message.kind == "execution-feedback":
            try:
                event = json.loads(message.event_json)
                valid = valid and (
                    isinstance(event, dict)
                    and event.get("agent_id") == expected
                    and event.get("task_id") == self.task_id
                    and event.get("tick") == message.sent_tick
                )
            except (ValueError, TypeError):
                valid = False
        if not valid:
            self._rejected += 1
            return
        previous = self._seen.get(message.message_id)
        if previous is not None and previous != message.model_dump():
            self._rejected += 1
            return
        if previous is None:
            self._seen[message.message_id] = message.model_dump()
            self._handler_deliveries += 1
            while len(self._seen) > self._max_history * 2:
                self._seen.popitem(last=False)
        else:
            self._duplicates += 1
        await ctx.send(sender, PeerAck(task_id=self.task_id, message_id=message.message_id))

    def _handle_acknowledgement(self, recipient, ctx, sender, acknowledgement):
        waiting = self._acks.get(acknowledgement.message_id)
        if (
            waiting is None
            or recipient != waiting[0]
            or sender != self._agents[waiting[1]].address
            or acknowledgement.task_id != self.task_id
            or ctx.session != self._session
            or acknowledgement.message_id not in self._seen
        ):
            self._rejected += 1
            return
        future = waiting[2]
        if not future.done():
            self._acknowledged += 1
            future.set_result(dict(self._seen[acknowledgement.message_id]))

    async def _transmit(self, payload):
        PeerMessage, DeliveryStatus = _sdk()[5], _sdk()[7]
        future = self._loop.create_future()
        self._acks[payload["message_id"]] = (payload["sender"], payload["recipient"], future)
        try:
            status = await self._contexts[payload["sender"]].send(
                self._agents[payload["recipient"]].address, PeerMessage(**payload)
            )
            if status.status != DeliveryStatus.DELIVERED:
                raise FetchTransportError(f"Fetch SDK could not deliver message: {status.detail}")
            return await asyncio.wait_for(future, self.timeout_seconds)
        finally:
            self._acks.pop(payload["message_id"], None)

    def _exchange(self, payload):
        self.start()
        future = asyncio.run_coroutine_threadsafe(self._transmit(payload), self._loop)
        try:
            received = future.result(timeout=self.timeout_seconds + 0.5)
        except (TimeoutError, concurrent.futures.TimeoutError) as exc:
            future.cancel()
            self._failure = "Fetch peer acknowledgement timed out; rollout stopped"
            raise FetchTransportError(self._failure) from exc
        except Exception as exc:
            self._failure = f"Fetch peer exchange failed; rollout stopped: {exc}"
            raise FetchTransportError(self._failure) from exc
        self._wire_history.append({**received, "session_id": str(self._session)})
        return received

    def _payload(self, sender, text, tick, *, event_json=""):
        return {
            "task_id": self.task_id,
            "message_id": str(uuid4()),
            "sender": sender,
            "recipient": "packer" if sender == "picker" else "picker",
            "kind": "execution-feedback" if event_json else "coordination",
            "text": text,
            "sent_tick": tick,
            "deliver_tick": tick + self.latency_ticks,
            "event_json": event_json,
        }

    @staticmethod
    def _json_payload(value):
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > _ORCHESTRATION_BYTES:
            raise ValueError("Orchestration payload exceeds 32768 bytes")
        return encoded

    @staticmethod
    def _vector(value):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                for item in value
            )
        ):
            raise ValueError("Expected a finite three-dimensional vector")
        return tuple(value)

    @classmethod
    def _decode_observation(cls, payload, agent_id):
        names = {item.name for item in fields(Observation)} - {"rgb", "wrist_rgb"}
        if not isinstance(payload, dict) or set(payload) != names:
            raise ValueError("Observation must contain only local state, without camera pixels")
        cls._validate_tick(payload["tick"])
        if payload["agent_id"] != agent_id:
            raise ValueError("Observation belongs to another robot")
        values = dict(payload)
        for key in ("ee_position", "ee_orientation"):
            values[key] = cls._vector(values[key])
        for key in ("objects", "targets"):
            positions = values[key]
            if not isinstance(positions, dict) or len(positions) > 32:
                raise ValueError("Observation object mapping is invalid")
            if any(not isinstance(name, str) or len(name) > 128 for name in positions):
                raise ValueError("Observation object identifier is invalid")
            values[key] = {name: cls._vector(position) for name, position in positions.items()}
        for key in ("packed", "order"):
            value = values[key]
            if key == "order" and value is None:
                continue
            if not isinstance(value, (list, tuple)) or len(value) > 32:
                raise ValueError("Observation task state is invalid")
            if any(
                item is not None and (not isinstance(item, str) or len(item) > 128)
                for item in value
            ):
                raise ValueError("Observation task identifier is invalid")
            values[key] = tuple(value)
        holding = values["holding"]
        if holding is not None and (not isinstance(holding, str) or len(holding) > 128):
            raise ValueError("Observation held-object identifier is invalid")
        opening = values["gripper_opening"]
        if (
            isinstance(opening, bool)
            or not isinstance(opening, (int, float))
            or not math.isfinite(opening)
        ):
            raise ValueError("Observation gripper opening is invalid")
        return Observation(**values)

    @staticmethod
    def _validate_events(events):
        if (
            not isinstance(events, list)
            or len(events) > 32
            or any(not isinstance(event, dict) for event in events)
        ):
            raise ValueError("Expected at most 32 observed event dictionaries")

    def _process_orchestration(self, request):
        """Only the coordinator SDK handler calls this stateful interpreter."""
        if len(request.payload_json.encode("utf-8")) > _ORCHESTRATION_BYTES:
            raise ValueError("Orchestration payload exceeds 32768 bytes")
        payload = json.loads(request.payload_json)
        tracker = self._action_feedback[request.agent_id]
        if request.phase == "context":
            if not isinstance(payload, dict) or set(payload) != {"peer_events"}:
                raise ValueError("Invalid context request")
            events = payload["peer_events"]
            self._validate_events(events)
            peer = "packer" if request.agent_id == "picker" else "picker"
            if any(
                event.get("task_id") != self.task_id or event.get("agent_id") != peer
                for event in events
            ):
                raise ValueError("Peer feedback source does not match the task")
            peer_critical = tracker.receive_peer(events, request.tick)
            critical = self._orchestration_critical[request.agent_id] or bool(peer_critical)
            result = {"text": tracker.context(), "critical": critical}
            self._orchestration_critical[request.agent_id] = False
            return result
        if (
            request.phase != "digest"
            or not isinstance(payload, dict)
            or set(payload) != {"before", "after", "instruction", "action", "execution_events"}
        ):
            raise ValueError("Invalid action digestion request")
        before = self._decode_observation(payload["before"], request.agent_id)
        after = self._decode_observation(payload["after"], request.agent_id)
        if (
            before.tick < self._orchestration_ticks[request.agent_id]
            or after.tick <= before.tick
            or request.tick != after.tick
        ):
            raise ValueError("Action observations are stale or out of order")
        instruction = payload["instruction"]
        if (
            not isinstance(instruction, dict)
            or set(instruction) != {"kind", "text", "object_id", "target_id"}
            or instruction["kind"] not in {"pick", "place", "wait"}
            or not isinstance(instruction["text"], str)
            or len(instruction["text"]) > 1024
            or any(
                instruction[key] is not None
                and (not isinstance(instruction[key], str) or len(instruction[key]) > 128)
                for key in ("object_id", "target_id")
            )
        ):
            raise ValueError("Invalid local instruction")
        action = payload["action"]
        if (
            not isinstance(action, dict)
            or set(action) != {"delta", "gripper"}
            or isinstance(action["gripper"], bool)
            or action["gripper"] not in {-1, 0, 1}
        ):
            raise ValueError("Invalid action")
        delta = self._vector(action["delta"])
        self._validate_events(payload["execution_events"])
        report = tracker.update(
            before,
            after,
            Instruction(**instruction),
            Action(delta=delta, gripper=action["gripper"]),
            execution_events=payload["execution_events"],
        )
        self._orchestration_critical[request.agent_id] |= report.critical
        return {
            "report": report.to_dict(),
            "text": tracker.context(),
            "critical": self._orchestration_critical[request.agent_id],
        }

    async def _handle_orchestration_request(self, ctx, sender, request):
        Reply = _sdk()[9]
        try:
            UUID(request.request_id)
            self._validate_tick(request.tick)
            valid = (
                request.agent_id in AGENTS
                and sender == self._agents[request.agent_id].address
                and request.task_id == self.task_id
                and ctx.session == self._session
                and request.phase in {"digest", "context"}
                and request.sequence > 0
            )
            if not valid:
                raise ValueError("Uncorrelated orchestration request")
            previous = self._orchestration_seen.get(request.request_id)
            if previous is not None:
                if previous[0] != request.model_dump():
                    raise ValueError("Request identifier reused with different contents")
                reply = previous[1]
                self._orchestration_duplicates += 1
            else:
                if (
                    request.sequence != self._orchestration_received_sequence[request.agent_id] + 1
                    or request.tick < self._orchestration_ticks[request.agent_id]
                ):
                    raise ValueError("Orchestration requests arrived out of order")
                result = self._process_orchestration(request)
                reply = {
                    key: getattr(request, key)
                    for key in ("task_id", "request_id", "agent_id", "sequence", "phase", "tick")
                }
                reply["result_json"] = self._json_payload(result)
                self._orchestration_received_sequence[request.agent_id] = request.sequence
                self._orchestration_ticks[request.agent_id] = request.tick
                self._orchestration_seen[request.request_id] = (request.model_dump(), reply)
                while len(self._orchestration_seen) > self._max_history * 2:
                    self._orchestration_seen.popitem(last=False)
                self._orchestration_requests += 1
        except (ValueError, TypeError, KeyError, OverflowError):
            self._orchestration_rejected += 1
            return
        await ctx.send(sender, Reply(**reply))

    def _handle_orchestration_reply(self, recipient, ctx, sender, reply):
        waiting = self._orchestration_waiting.get(reply.request_id)
        if waiting is None:
            self._orchestration_rejected += 1
            return
        request, future = waiting
        valid = (
            recipient == request["agent_id"]
            and sender == self._agents["coordinator"].address
            and ctx.session == self._session
            and all(
                getattr(reply, key) == request[key]
                for key in ("task_id", "agent_id", "sequence", "phase", "tick")
            )
        )
        try:
            if len(reply.result_json.encode("utf-8")) > _ORCHESTRATION_BYTES:
                raise ValueError("Oversized orchestration reply")
            result = json.loads(reply.result_json)
            expected = (
                {"text", "critical", "report"}
                if request["phase"] == "digest"
                else {"text", "critical"}
            )
            if (
                not valid
                or not isinstance(result, dict)
                or set(result) != expected
                or not isinstance(result["text"], str)
                or len(result["text"]) > 4096
                or not isinstance(result["critical"], bool)
                or (request["phase"] == "digest" and not isinstance(result["report"], dict))
            ):
                raise ValueError("Invalid orchestration reply")
            if request["phase"] == "digest" and any(
                result["report"].get(key) != request[key] for key in ("task_id", "agent_id", "tick")
            ):
                raise ValueError("Action report does not match its request")
        except (ValueError, TypeError):
            self._orchestration_rejected += 1
            return
        if not future.done():
            self._orchestration_receipts += 1
            future.set_result(result)

    async def _transmit_orchestration(self, payload):
        Request, DeliveryStatus = _sdk()[8], _sdk()[7]
        future = self._loop.create_future()
        self._orchestration_waiting[payload["request_id"]] = (payload, future)
        try:
            status = await self._contexts[payload["agent_id"]].send(
                self._agents["coordinator"].address, Request(**payload)
            )
            if status.status != DeliveryStatus.DELIVERED:
                raise FetchTransportError(
                    f"Fetch SDK could not deliver orchestration: {status.detail}"
                )
            return await asyncio.wait_for(future, self.timeout_seconds)
        finally:
            self._orchestration_waiting.pop(payload["request_id"], None)

    def _exchange_orchestration(self, payload):
        self.start()
        future = asyncio.run_coroutine_threadsafe(self._transmit_orchestration(payload), self._loop)
        try:
            result = future.result(timeout=self.timeout_seconds + 0.5)
        except (TimeoutError, concurrent.futures.TimeoutError) as exc:
            future.cancel()
            self._failure = "Fetch orchestration reply timed out; rollout stopped"
            raise FetchTransportError(self._failure) from exc
        except Exception as exc:
            self._failure = f"Fetch orchestration exchange failed; rollout stopped: {exc}"
            raise FetchTransportError(self._failure) from exc
        self._orchestration_history.append(
            {
                **{key: value for key, value in payload.items() if key != "payload_json"},
                "session_id": str(self._session),
                "result": deepcopy(result),
            }
        )
        return result

    def _orchestrate(self, agent_id, phase, tick, body):
        self._check_healthy()
        if not self.orchestration:
            raise FetchTransportError("Fetch action orchestration is not enabled")
        if agent_id not in AGENTS:
            raise ValueError(f"Unknown robot: {agent_id!r}")
        self._validate_tick(tick)
        encoded = self._json_payload(body)
        self._orchestration_sequence[agent_id] += 1
        return self._exchange_orchestration(
            {
                "task_id": self.task_id,
                "request_id": str(uuid4()),
                "agent_id": agent_id,
                "sequence": self._orchestration_sequence[agent_id],
                "phase": phase,
                "tick": tick,
                "payload_json": encoded,
            }
        )

    def digest(self, agent_id, before, after, instruction, action, execution_events):
        """Send requested action and measured local transition to the Fetch coordinator.

        Camera pixels stay at the VLA. The coordinator returns grounded language,
        and only its validated SDK reply can enter the next model context.
        """
        with self._lock:

            def observation_payload(observation):
                return {
                    item.name: getattr(observation, item.name)
                    for item in fields(Observation)
                    if item.name not in {"rgb", "wrist_rgb"}
                }

            return self._orchestrate(
                agent_id,
                "digest",
                after.tick,
                {
                    "before": observation_payload(before),
                    "after": observation_payload(after),
                    "instruction": {
                        name: getattr(instruction, name)
                        for name in ("kind", "text", "object_id", "target_id")
                    },
                    "action": {item.name: getattr(action, item.name) for item in fields(Action)},
                    "execution_events": list(execution_events),
                },
            )

    def context(self, agent_id, peer_events, tick):
        """Return bounded robot-private feedback; consume its pending critical flag."""
        with self._lock:
            return self._orchestrate(agent_id, "context", tick, {"peer_events": list(peer_events)})

    def send(self, sender: AgentId, text: str, tick: int) -> Message | None:
        with self._lock:
            self._check_healthy()
            if sender not in AGENTS:
                raise ValueError(f"Unknown sender: {sender!r}")
            self._validate_tick(tick)
            self._current_tick = tick
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Message text must be a non-empty string")
            if len(text) > self.max_message_length:
                raise ValueError(f"Message exceeds {self.max_message_length} characters")
            if not self.enabled or len(self._pending) >= self.max_pending:
                self.dropped += 1
                return None
            decoded = self._exchange(self._payload(sender, text, tick))
            message = Message(
                *(
                    decoded[key]
                    for key in ("sender", "recipient", "text", "sent_tick", "deliver_tick")
                )
            )
            self._pending.append(message)
            self._history.append(message)
            self.sent += 1
            return message

    def receive(self, agent_id: AgentId, tick: int) -> list[Message]:
        with self._lock:
            self._check_healthy()
            self._validate_tick(tick)
            self._current_tick = tick
            return super().receive(agent_id, tick)

    def publish_event(self, agent_id: AgentId, event: dict) -> dict | None:
        """Send simulator-observed feedback; this does not assert VLA success."""
        with self._lock:
            self._check_healthy()
            if agent_id not in AGENTS or not isinstance(event, dict):
                raise ValueError("A known agent and an event dictionary are required")
            if event.get("agent_id") != agent_id or event.get("task_id") != self.task_id:
                raise ValueError("Feedback agent_id and task_id must match the sender and run")
            tick = event.get("tick")
            self._validate_tick(tick)
            event_json = json.dumps(event, allow_nan=False, separators=(",", ":"))
            if len(event_json.encode("utf-8")) > 8192:
                raise ValueError("Feedback event exceeds 8192 bytes")
            if not self.enabled or len(self._event_pending) >= self.max_pending:
                self.dropped += 1
                return None
            decoded = self._exchange(self._payload(agent_id, "", tick, event_json=event_json))
            result = {key: value for key, value in decoded.items() if key != "event_json"}
            result["event"] = json.loads(decoded["event_json"])
            result["session_id"] = str(self._session)
            self._event_pending.append(result)
            self._event_history.append(result)
            self._event_sent += 1
            return deepcopy(result)

    def receive_events(self, agent_id: AgentId, tick: int | None = None) -> list[dict]:
        """Read peer event payloads; omitted tick uses the latest send/receive tick."""
        with self._lock:
            self._check_healthy()
            if agent_id not in AGENTS:
                raise ValueError(f"Unknown recipient: {agent_id!r}")
            tick = self._current_tick if tick is None else tick
            self._validate_tick(tick)
            self._current_tick = tick
            ready = [
                item
                for item in self._event_pending
                if (item["recipient"] == agent_id and item["deliver_tick"] <= tick)
            ]
            ready_ids = {item["message_id"] for item in ready}
            self._event_pending = deque(
                item for item in self._event_pending if item["message_id"] not in ready_ids
            )
            self._event_delivered += len(ready)
            return [deepcopy(item["event"]) for item in ready]

    @property
    def event_transcript(self):
        return deepcopy(list(self._event_history))

    @property
    def wire_transcript(self):
        return deepcopy(list(self._wire_history))

    @property
    def orchestration_transcript(self):
        return deepcopy(list(self._orchestration_history))

    @property
    def transport_info(self):
        return {
            "kind": "fetch-uagents-local",
            "sdk_version": version("uagents") if self._agents else None,
            "task_id": self.task_id,
            "session_id": str(self._session),
            "addresses": {name: agent.address for name, agent in self._agents.items()},
            "network": "SDK local dispatcher",
            "public_registration": False,
            "sent": self.sent,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "sdk_handler_deliveries": self._handler_deliveries,
            "acknowledgements": self._acknowledged,
            "duplicates": self._duplicates,
            "rejected": self._rejected,
            "feedback_sent": self._event_sent,
            "feedback_delivered": self._event_delivered,
            "orchestration": self.orchestration,
            "orchestration_requests": self._orchestration_requests,
            "orchestration_receipts": self._orchestration_receipts,
            "orchestration_duplicates": self._orchestration_duplicates,
            "orchestration_rejected": self._orchestration_rejected,
            "failure": self._failure,
        }

    def reset(self):
        """Reset simulation counters and use a fresh session to reject stale mail."""
        with self._lock:
            self._check_healthy()
            super().reset()
            self._session = uuid4()
            if self._loop and self._loop.is_running():

                async def clear_session():
                    self._seen.clear()
                    self._orchestration_seen.clear()
                    for tracker in self._action_feedback.values():
                        tracker.reset()
                    for context in self._contexts.values():
                        context._session = self._session

                asyncio.run_coroutine_threadsafe(clear_session(), self._loop).result(
                    timeout=self.timeout_seconds
                )
            self._event_pending.clear()
            self._event_history.clear()
            self._wire_history.clear()
            self._handler_deliveries = self._acknowledged = self._duplicates = 0
            self._rejected = self._event_sent = self._event_delivered = 0
            self._current_tick = 0
            self._orchestration_history.clear()
            for agent_id in AGENTS:
                self._orchestration_sequence[agent_id] = 0
                self._orchestration_received_sequence[agent_id] = 0
                self._orchestration_ticks[agent_id] = 0
                self._orchestration_critical[agent_id] = False
            self._orchestration_requests = self._orchestration_receipts = 0
            self._orchestration_duplicates = self._orchestration_rejected = 0

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=self.timeout_seconds + 0.5)
                if self._thread.is_alive():
                    raise FetchTransportError("Fetch transport thread did not stop before timeout")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
