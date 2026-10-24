"""Fetch Chat Protocol entry point for bounded, real Relay Lab experiments.

The chat gateway launches the simulator; it never pretends a model's command is
an observed accomplishment. Robot-to-robot messages travel through the separate
Fetch transport in the child process. No LLM or ASI API key is involved here.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from uagents import Agent, Protocol
from uagents.registration import AlmanacApiRegistrationPolicy
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.registration import AgentRegistrationPolicy
from uagents_core.types import DeliveryStatus

from .tasks import RELAY_COMMAND, TaskSpec, parse_task, task_capabilities

HELP = (
    "Relay Lab coordinates two simulated robots with Fetch execution feedback. Commands: "
    "`pack coral, mint, gold`, `run relay`, `run relay seed 7`, `status`, "
    "`capabilities`, or `help`. "
    "The operator selects the action policy and step budget. SmolVLA is a pretrained "
    "LIBERO checkpoint, not adapted to this custom scene; task success is not guaranteed. "
    "Progress describes simulator observations, not physical robots."
)
MAX_TEXT = 256
MAX_SEEN_MESSAGES = 1024
MAX_EVENT_BYTES = 65536
MAX_EVENT_MESSAGES = 60


class SimulationTimeout(TimeoutError):
    """The experiment's wall-clock budget expired, independently of chat delivery."""


@dataclass(frozen=True)
class GatewayConfig:
    seed: str = field(repr=False)
    output_root: Path = Path("runs/fetch")
    policy: str = "smolvla"
    device: str = "auto"
    max_steps: int = 600
    timeout_seconds: float = 1800
    default_seed: int = 7
    port: int = 8010
    local: bool = False
    video: bool = True
    poll_seconds: float = 1.0
    progress_seconds: float = 2.0
    coordination: str = "memory"

    def __post_init__(self):
        if not isinstance(self.seed, str) or not self.seed.strip():
            raise ValueError("A stable, private gateway identity seed is required")
        if self.policy not in ("smolvla", "scripted"):
            raise ValueError("Gateway policy must be smolvla or scripted")
        if self.coordination not in ("memory", "spacetime"):
            raise ValueError("Unsupported coordination backend")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("Unsupported inference device")
        if not 1 <= self.max_steps <= 600:
            raise ValueError("Gateway max_steps must be between 1 and 600")
        if not 0 < self.timeout_seconds <= 3600:
            raise ValueError("Gateway timeout must be positive and at most 3600 seconds")
        if not 0 <= self.default_seed <= 2**31 - 1:
            raise ValueError("Gateway seed must be between 0 and 2147483647")
        if not 1 <= self.port <= 65535:
            raise ValueError("Gateway port must be between 1 and 65535")
        if not 0 < self.poll_seconds <= 10 or not 0 <= self.progress_seconds <= 60:
            raise ValueError("Invalid gateway polling or progress interval")


@dataclass
class RelayJob:
    task_id: str
    owner: str
    session: str
    context: Any
    seed: int
    output: Path
    task_spec: TaskSpec = field(default_factory=TaskSpec)
    started: float = field(default_factory=time.monotonic)
    task: asyncio.Task | None = None
    process: Any = None
    state: str = "starting"
    event_offset: int = 0
    events: OrderedDict = field(default_factory=OrderedDict)
    last_progress: float = float("-inf")
    progress_count: int = 0
    latest: str = "Waiting for simulator observations."


class RelayChatGateway:
    """Session-safe ACP handlers, with an injectable process factory for testing."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        agent_address: str | None = None,
        spawn_job: Callable | None = None,
    ):
        self.config = config
        self.agent_address = agent_address
        self.spawn_job = spawn_job or asyncio.create_subprocess_exec
        self.active: RelayJob | None = None
        self.last: tuple[str, str, str] | None = None
        self.seen: OrderedDict[tuple[str, str, str], float] = OrderedDict()

    async def handle_ack(self, ctx, sender: str, message: ChatAcknowledgement) -> None:
        # Acknowledgements are terminal protocol receipts, never new chat requests.
        return None

    def _same_session(self, ctx, sender: str) -> bool:
        return bool(
            self.active and self.active.owner == sender and self.active.session == str(ctx.session)
        )

    async def _send(self, ctx, sender: str, message) -> None:
        result = await asyncio.wait_for(ctx.send(sender, message), timeout=30)
        # The SDK reports delivery failures as values, not necessarily exceptions.
        # None is supported only for lightweight embedding/test Context adapters.
        if result is not None and result.status != DeliveryStatus.DELIVERED:
            raise ConnectionError("Fetch message delivery failed")

    async def _reply(self, ctx, sender: str, text: str, *, terminal: bool) -> None:
        content = [TextContent(text=text)]
        if terminal:
            content.append(EndSessionContent())
        await self._send(ctx, sender, ChatMessage(content=content))

    async def handle_message(self, ctx, sender: str, message: ChatMessage) -> None:
        # Ack first, even on duplicate, unsupported, oversized, or busy requests.
        await self._send(ctx, sender, ChatAcknowledgement(acknowledged_msg_id=message.msg_id))
        now = time.monotonic()
        key = (sender, str(ctx.session), str(message.msg_id))
        while self.seen and (now - next(iter(self.seen.values())) > 3600):
            self.seen.popitem(last=False)
        if key in self.seen:
            return
        self.seen[key] = now
        while len(self.seen) > MAX_SEEN_MESSAGES:
            self.seen.popitem(last=False)

        terminal = not self._same_session(ctx, sender)
        # Don't concatenate an arbitrarily large payload or treat resources as commands.
        if len(message.content) > 8:
            await self._reply(ctx, sender, "Command too long. " + HELP, terminal=terminal)
            return
        pieces = [part.text for part in message.content if isinstance(part, TextContent)]
        if sum(map(len, pieces)) > MAX_TEXT:
            await self._reply(ctx, sender, "Command too long. " + HELP, terminal=terminal)
            return
        if not pieces and any(isinstance(part, EndSessionContent) for part in message.content):
            return  # A session-close receipt is not a request to start a new conversation.
        text = " ".join(pieces).strip().lower()
        if self.agent_address:
            # ASI:One includes its selected agent's mention in the text payload.
            # Remove only our own single routing prefix, after the input limits.
            text = re.sub(rf"^@{re.escape(self.agent_address)}\s+", "", text, count=1)
        if not text or text == "help":
            await self._reply(ctx, sender, HELP, terminal=terminal)
            return
        if text == "capabilities":
            await self._reply(ctx, sender, task_capabilities(), terminal=terminal)
            return
        if text == "status":
            if self._same_session(ctx, sender):
                job = self.active
                status = f"Task {job.task_id}: {job.state}. {job.latest}"
            elif self.active:
                status = "The simulator is busy with another request. Try again later."
            elif self.last and self.last[:2] == (sender, str(ctx.session)):
                status = self.last[2]
            else:
                status = "The simulator is idle. " + HELP
            await self._reply(ctx, sender, status, terminal=terminal)
            return

        match = RELAY_COMMAND.fullmatch(text)
        seed = int(match[1]) if match and match[1] else self.config.default_seed
        if seed > 2**31 - 1:
            await self._reply(
                ctx, sender, "Seed must be between 0 and 2147483647.", terminal=terminal
            )
            return
        task_spec = parse_task(text)
        if task_spec is None:
            await self._reply(
                ctx,
                sender,
                "Unknown command or unsupported task. Choose all three parcel colors exactly "
                "once; arbitrary task instructions are not supported. " + HELP,
                terminal=terminal,
            )
            return
        if self.active:
            await self._reply(
                ctx, sender, "The simulator is already running. Use `status`.", terminal=terminal
            )
            return

        task_id = str(uuid4())
        job = RelayJob(
            task_id=task_id,
            owner=sender,
            session=str(ctx.session),
            context=ctx,  # Keep the initiating Context: its session travels with every reply.
            seed=seed,
            output=self.config.output_root.resolve() / task_id,
            task_spec=task_spec,
        )
        self.active = job  # Claim before the next await, so concurrent requests cannot race.
        try:
            await self._reply(
                ctx,
                sender,
                f"Task {task_id} accepted. {task_spec.goal} "
                f"Running seed {seed} with {self.config.policy}, "
                f"Fetch robot messages, and at most {self.config.max_steps} simulation steps. "
                "I will report observed progress and the actual task outcome.",
                terminal=False,
            )
        except Exception:
            self.active = None
            raise
        job.task = asyncio.create_task(self._run(job), name=f"relay-{task_id}")

    def command(self, job: RelayJob) -> list[str]:
        argv = [
            sys.executable,
            "-m",
            "relay_lab",
            "run",
            "--transport",
            "fetch",
            "--coordination",
            self.config.coordination,
            "--policy",
            self.config.policy,
            "--device",
            self.config.device,
            "--seed",
            str(job.seed),
            "--max-steps",
            str(self.config.max_steps),
            "--task-id",
            job.task_id,
            "--output",
            str(job.output),
        ]
        if job.task_spec.order is not None:
            argv.extend(["--order", *job.task_spec.order])
        if not self.config.video:
            argv.append("--no-video")
        return argv

    async def _progress(self, job: RelayJob) -> None:
        path = job.output / "execution.jsonl"
        if not path.exists():
            return
        # Consume only complete, bounded records. A writer may still be flushing a line.
        with path.open("rb") as log:
            log.seek(job.event_offset)
            for _ in range(256):
                line = log.readline(MAX_EVENT_BYTES + 1)
                if not line or not line.endswith(b"\n"):
                    if len(line) > MAX_EVENT_BYTES:
                        raise ValueError("Execution log contains an oversized event")
                    break
                job.event_offset = log.tell()
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(event, dict) or event.get("task_id") != job.task_id:
                    continue
                if event.get("schema") != "relay-lab.execution.v1":
                    continue
                kind, agent = event.get("kind"), event.get("agent_id")
                if kind not in {"started", "holding", "transfer_ready", "completed", "blocked"}:
                    continue
                if agent not in {"picker", "packer"} or event.get("scope") != "instruction":
                    continue
                signature = (agent, event.get("instruction_id"), kind, event.get("reason"))
                if signature in job.events:
                    continue
                job.events[signature] = True
                while len(job.events) > 512:
                    job.events.popitem(last=False)
                noun = str(event.get("object_id") or "local instruction")[:40]
                detail = {
                    "started": f"started an instruction for {noun}",
                    "holding": f"is observed holding {noun} in the simulator",
                    "transfer_ready": f"observed {noun} released and stationary at the relay",
                    "completed": f"completed a local instruction for {noun} (not the whole task)",
                    "blocked": f"reported a blocked instruction for {noun}",
                }[kind]
                reason = str(event.get("reason", ""))[:100].replace("_", " ")
                job.latest = f"Step {event.get('tick', '?')}: {agent} {detail}. {reason}."
                now = time.monotonic()
                if (
                    job.progress_count < MAX_EVENT_MESSAGES
                    and now - job.last_progress >= self.config.progress_seconds
                ):
                    await self._reply(job.context, job.owner, job.latest, terminal=False)
                    job.progress_count += 1
                    job.last_progress = now

    async def _stop_process(self, process) -> None:
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()

    def _summary(self, job: RelayJob, returncode: int) -> str:
        prefix = f"Task {job.task_id}: "
        if returncode:
            return prefix + (
                f"simulator failed (exit {returncode}). No successful outcome is claimed. "
                "The operator can inspect the local error.json and console log."
            )
        path = job.output / "summary.json"
        if not path.exists() or path.stat().st_size > 100000:
            return prefix + "simulator exited without a valid summary; outcome is unverified."
        try:
            summary = json.loads(path.read_text())
            success, packed, order, steps = (
                summary["success"],
                summary["packed"],
                summary["order"],
                summary["steps"],
            )
            if (
                type(success) is not bool
                or not isinstance(packed, list)
                or not isinstance(order, list)
                or not order
                or type(steps) is not int
                or not 0 <= steps <= self.config.max_steps
                or len(packed) != len(order)
                or sorted(order) != ["coral", "gold", "mint"]
                or (job.task_spec.order is not None and tuple(order) != job.task_spec.order)
                or any(item is not None and item not in order for item in packed)
                or (success and packed != order)
                or summary.get("task_id") != job.task_id
                or summary.get("policy") != self.config.policy
                or summary.get("transport") != "fetch"
            ):
                raise ValueError("Inconsistent simulator summary")
        except (ValueError, TypeError, KeyError):
            return prefix + "simulator summary was invalid; outcome is unverified."
        if self.config.coordination == "spacetime":
            from .spacetime import SpacetimeClient, SpacetimeConfig, SpacetimeError

            try:
                task = SpacetimeClient(SpacetimeConfig.from_env()).get_task(job.task_id)
                if (
                    not task
                    or task["status"] != ("succeeded" if success else "exhausted")
                    or task["phase"] != "terminal"
                    or task["next_tick"] != steps
                    or json.loads(task["outcome_json"])
                    != {"tick": steps, "packed": packed, "order": order, "success": success}
                ):
                    raise ValueError("Database outcome does not match the rollout")
            except (SpacetimeError, ValueError, KeyError, TypeError):
                return (
                    prefix
                    + "SpacetimeDB could not verify the terminal result; outcome is unverified."
                )
        outcome = "SUCCESS" if success else "TASK NOT COMPLETED"
        correctly_packed = sum(
            actual is not None and actual == expected for actual, expected in zip(packed, order)
        )
        suffix = (
            " This pretrained SmolVLA checkpoint is not adapted to Relay Lab."
            if self.config.policy == "smolvla"
            else " This was the scripted reference policy."
        )
        return (
            f"{prefix}{outcome}. Packed {correctly_packed}/{len(order)} parcels in their "
            f"assigned slots in {steps} steps. "
            f"Order: {', '.join(order)}. "
            f"Action policy: {self.config.policy}; robot transport: Fetch uAgents."
            + (" Coordination: SpacetimeDB." if self.config.coordination == "spacetime" else "")
            + f"{suffix} Replay and detailed evidence were saved locally under this task ID."
        )

    async def _run(self, job: RelayJob) -> None:
        final = f"Task {job.task_id}: stopped before a verified result was available."
        waiter = None
        try:
            job.output.parent.mkdir(parents=True, exist_ok=True)
            # Keep launch logs outside the empty output directory required by the runner.
            request = {
                "task_id": job.task_id,
                "session": job.session,
                "requester": job.owner,
                "seed": job.seed,
                "policy": self.config.policy,
                "max_steps": self.config.max_steps,
                "task": {
                    "kind": job.task_spec.kind,
                    "order": job.task_spec.order,
                    "goal": job.task_spec.goal,
                },
            }
            (job.output.parent / f"{job.task_id}.request.json").write_text(
                json.dumps(request, indent=2) + "\n"
            )
            with (job.output.parent / f"{job.task_id}.console.log").open("wb") as console:
                job.process = await self.spawn_job(
                    *self.command(job), stdout=console, stderr=asyncio.subprocess.STDOUT
                )
                job.state = "running"
                waiter = asyncio.create_task(job.process.wait())
                while not waiter.done():
                    if time.monotonic() - job.started >= self.config.timeout_seconds:
                        raise SimulationTimeout("Simulation exceeded the configured time budget")
                    await self._progress(job)
                    await asyncio.wait({waiter}, timeout=self.config.poll_seconds)
                await self._progress(job)
                final = self._summary(job, waiter.result())
        except SimulationTimeout:
            await self._stop_process(job.process)
            final = (
                f"Task {job.task_id}: TIME LIMIT. The simulator was stopped after the configured "
                f"{self.config.timeout_seconds:g}-second budget; task completion is unverified."
            )
        except asyncio.CancelledError:
            await self._stop_process(job.process)
            final = f"Task {job.task_id}: gateway stopped; task completion is unverified."
        except Exception as exc:
            await self._stop_process(job.process)
            final = (
                f"Task {job.task_id}: execution failed ({type(exc).__name__}); "
                "task completion is unverified. The operator can inspect the local logs."
            )
            job.context.logger.exception("Relay task %s failed", job.task_id)
        finally:
            if waiter and not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            job.state = "finished"
            self.last = (job.owner, job.session, final)
            try:
                await self._reply(job.context, job.owner, final, terminal=True)
            except Exception:
                job.context.logger.exception("Could not deliver task %s result", job.task_id)
            finally:
                if self.active is job:
                    self.active = None

    async def close(self) -> None:
        if self.active and self.active.task:
            self.active.task.cancel()
            await asyncio.gather(self.active.task, return_exceptions=True)


class LocalRegistration(AgentRegistrationPolicy):
    """Explicitly disable public discovery and ledger registration in local tests."""

    async def register(self, *args, **kwargs):
        return None


class LocalGatewayAgent(Agent):
    """SDK 0.26.1 otherwise sends an Almanac status update even with a no-op policy.

    This narrow override disables that unconditional network call for local
    smoke tests. Public mailbox agents retain the SDK's standard lifecycle.
    """

    async def _update_agent_status(self, active: bool):
        return None


def build_gateway(config: GatewayConfig) -> Agent:
    """Build a standard ACP agent; mailbox mode uses free off-chain registration."""
    agent_type = LocalGatewayAgent if config.local else Agent
    agent = agent_type(
        name="Relay Lab",
        seed=config.seed,
        port=config.port,
        mailbox=not config.local,
        endpoint=f"http://127.0.0.1:{config.port}/submit" if config.local else None,
        registration_policy=(
            LocalRegistration() if config.local else AlmanacApiRegistrationPolicy()
        ),
        publish_agent_details=not config.local,
        readme_path=str(Path(__file__).parent / "assets" / "fetch-profile.md"),
        report_events=False,
        mark_inactive_on_shutdown=not config.local,
        description=(
            "Fetch coordinates simulated robots and turns observed action results into text "
            "for the next VLA prediction. Request parcel packing tasks through ASI:One."
        ),
    )
    gateway = RelayChatGateway(config, agent_address=agent.address)
    protocol = Protocol(spec=chat_protocol_spec)
    protocol.on_message(ChatMessage)(gateway.handle_message)
    protocol.on_message(ChatAcknowledgement)(gateway.handle_ack)
    agent.include(protocol, publish_manifest=not config.local)

    @agent.on_event("shutdown")
    async def close(ctx):
        await gateway.close()

    # Useful for embedding and local smoke tests; no SDK internals are modified.
    agent.relay_gateway = gateway
    return agent


def run_gateway(config: GatewayConfig) -> None:
    # Mailbox traffic is outbound. The inspector and optional local client need
    # only loopback; the pinned SDK otherwise binds every network interface.
    import uagents.asgi

    if config.coordination == "spacetime":
        from .spacetime import SpacetimeClient, SpacetimeConfig

        # Fail at startup instead of advertising an unusable task service.
        SpacetimeClient(SpacetimeConfig.from_env()).get_task("gateway-healthcheck")

    uagents.asgi.HOST = "127.0.0.1"
    build_gateway(config).run()
