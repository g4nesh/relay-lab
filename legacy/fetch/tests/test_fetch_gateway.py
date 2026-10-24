"""ACP and subprocess boundaries; tests never load an action model."""

import asyncio
import json
import logging
from uuid import uuid4

import pytest

pytest.importorskip("uagents")

from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.types import DeliveryStatus, MsgStatus

from relay_lab.fetch_gateway import (
    MAX_SEEN_MESSAGES,
    GatewayConfig,
    LocalRegistration,
    RelayChatGateway,
    build_gateway,
)


class Context:
    def __init__(self, session=None):
        self.session = session or uuid4()
        self.sent = []
        self.logger = logging.getLogger("gateway-test")

    async def send(self, destination, message):
        self.sent.append((destination, self.session, message))
        return MsgStatus(
            status=DeliveryStatus.DELIVERED,
            detail="test delivery",
            destination=destination,
            endpoint="local",
            session=self.session,
        )

    @property
    def text(self):
        return [item.text() for _, _, item in self.sent if isinstance(item, ChatMessage)]


class Process:
    def __init__(self, exit_code=0, *, done=True):
        self.returncode = None
        self.exit_code = exit_code
        self.finished = asyncio.Event()
        self.terminated = False
        if done:
            self.finished.set()

    async def wait(self):
        await self.finished.wait()
        self.returncode = self.exit_code
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.exit_code = -15
        self.finished.set()

    def kill(self):
        self.terminate()


def chat(text, *, msg_id=None):
    return ChatMessage(content=[TextContent(text)], msg_id=msg_id)


def configuration(tmp_path, **kwargs):
    return GatewayConfig(
        seed="test-only-identity",
        output_root=tmp_path,
        local=True,
        video=False,
        poll_seconds=0.001,
        progress_seconds=0,
        **kwargs,
    )


def terminal(message):
    return isinstance(message, ChatMessage) and any(
        isinstance(part, EndSessionContent) for part in message.content
    )


def summary_for(job, **updates):
    return {
        "task_id": job.task_id,
        "policy": "smolvla",
        "transport": "fetch",
        "success": False,
        "packed": [None, None, None],
        "order": ["coral", "gold", "mint"],
        "steps": 600,
        **updates,
    }


def event_for(job, kind="completed", **updates):
    return {
        "schema": "relay-lab.execution.v1",
        "task_id": job.task_id,
        "scope": "instruction",
        "agent_id": "picker",
        "instruction_id": "picker:1",
        "event_id": "picker:1",
        "kind": kind,
        "tick": 23,
        "object_id": "coral",
        "target_id": "relay",
        "reason": "observed_attachment",
        "evidence": {},
        **updates,
    }


@pytest.mark.parametrize("failed_message", ["acknowledgement", "acceptance"])
def test_failed_sdk_delivery_does_not_launch_simulation(tmp_path, failed_message):
    async def scenario():
        launches = []

        async def launch(*args, **kwargs):
            launches.append(args)
            return Process()

        class FailingContext(Context):
            async def send(self, destination, message):
                result = await super().send(destination, message)
                if (
                    failed_message == "acknowledgement" and isinstance(message, ChatAcknowledgement)
                ) or (failed_message == "acceptance" and isinstance(message, ChatMessage)):
                    result.status = DeliveryStatus.FAILED
                return result

        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        with pytest.raises(ConnectionError, match="Fetch message delivery failed"):
            await gateway.handle_message(FailingContext(), "alice", chat("run relay"))
        await asyncio.sleep(0)
        assert launches == []
        assert gateway.active is None

    asyncio.run(scenario())


def test_failed_terminal_sdk_delivery_is_logged_and_releases_simulator(tmp_path, caplog):
    async def scenario():
        class FinalFailureContext(Context):
            async def send(self, destination, message):
                result = await super().send(destination, message)
                if terminal(message):
                    result.status = DeliveryStatus.FAILED
                return result

        async def launch(*args, **kwargs):
            job = gateway.active
            job.output.mkdir()
            (job.output / "summary.json").write_text(json.dumps(summary_for(job)))
            return Process()

        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        ctx = FinalFailureContext()
        await gateway.handle_message(ctx, "alice", chat("run relay"))
        await gateway.active.task
        assert gateway.active is None
        assert "Could not deliver task" in caplog.text
        assert "Fetch message delivery failed" in caplog.text
        assert gateway.last[:2] == ("alice", str(ctx.session))

    asyncio.run(scenario())


def test_ack_precedes_launch_and_final_result_is_truthful(tmp_path):
    async def scenario():
        ctx = Context()
        calls = []

        async def launch(*argv, **kwargs):
            assert isinstance(ctx.sent[0][2], ChatAcknowledgement)
            assert "accepted" in ctx.text[0]
            assert not terminal(ctx.sent[-1][2])
            calls.append(argv)
            job = gateway.active
            job.output.mkdir()
            (job.output / "summary.json").write_text(json.dumps(summary_for(job)))
            (job.output / "execution.jsonl").write_text(json.dumps(event_for(job)) + "\n")
            return Process()

        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        message = chat("run relay seed 42")
        await gateway.handle_message(ctx, "alice", message)
        job = gateway.active
        await job.task
        assert ctx.sent[0][2].acknowledged_msg_id == message.msg_id
        assert all(
            destination == "alice" and session == ctx.session
            for destination, session, _ in ctx.sent
        )
        assert calls[0][calls[0].index("--seed") + 1] == "42"
        assert calls[0][calls[0].index("--task-id") + 1] == job.task_id
        assert "--transport" in calls[0] and "fetch" in calls[0]
        assert "--no-video" in calls[0]
        assert "not the whole task" in ctx.text[1]
        assert "TASK NOT COMPLETED" in ctx.text[-1]
        assert "Packed 0/3" in ctx.text[-1]
        assert terminal(ctx.sent[-1][2])
        assert sum(terminal(msg) for _, _, msg in ctx.sent) == 1
        assert gateway.active is None
        assert (tmp_path / f"{job.task_id}.request.json").exists()

    asyncio.run(scenario())


def test_duplicate_uuid_is_acked_but_never_relaunches(tmp_path):
    async def scenario():
        launches = []
        process = Process(done=False)

        async def launch(*args, **kwargs):
            launches.append(args)
            return process

        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        ctx = Context()
        message = chat("run relay")
        await gateway.handle_message(ctx, "alice", message)
        await asyncio.sleep(0)
        await gateway.handle_message(ctx, "alice", message)
        assert len(launches) == 1
        assert sum(isinstance(msg, ChatAcknowledgement) for _, _, msg in ctx.sent) == 2
        assert len(ctx.text) == 1
        await gateway.close()
        assert process.terminated
        assert "unverified" in ctx.text[-1]

    asyncio.run(scenario())


def test_owner_session_isolation_and_busy_request(tmp_path):
    async def scenario():
        process = Process(done=False)

        async def launch(*args, **kwargs):
            return process

        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        owner, stranger = Context(), Context()
        same_owner_new_session = Context()
        await gateway.handle_message(owner, "alice", chat("run relay"))
        job = gateway.active
        await asyncio.sleep(0)
        await gateway.handle_message(stranger, "mallory", chat("status"))
        await gateway.handle_message(same_owner_new_session, "alice", chat("status"))
        await gateway.handle_message(stranger, "mallory", chat("run relay"))
        assert job.task_id not in " ".join(stranger.text + same_owner_new_session.text)
        assert all(terminal(msg) for _, _, msg in stranger.sent if isinstance(msg, ChatMessage))
        await gateway.handle_message(owner, "alice", chat("status"))
        assert job.task_id in owner.text[-1]
        assert not terminal(owner.sent[-1][2])
        assert job.context is owner
        await gateway.close()
        assert "gateway stopped" in owner.text[-1]
        assert "gateway stopped" not in " ".join(stranger.text)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command",
    [
        "run relay; rm -rf /",
        "run relay seed -1",
        "run relay seed 2147483648",
        "run relay --policy scripted",
        "x" * 300,
    ],
)
def test_invalid_commands_never_spawn(tmp_path, command):
    async def scenario():
        async def launch(*args, **kwargs):
            pytest.fail("Invalid command launched a process")

        ctx = Context()
        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        await gateway.handle_message(ctx, "alice", chat(command))
        assert isinstance(ctx.sent[0][2], ChatAcknowledgement)
        assert terminal(ctx.sent[-1][2])
        assert gateway.active is None

    asyncio.run(scenario())


@pytest.mark.parametrize("command", ["run relay", "pack gold, coral, mint", "help", "capabilities"])
def test_own_asi_routing_mention_is_accepted(tmp_path, command):
    async def scenario():
        process = Process(done=False)
        launches = []

        async def launch(*args, **kwargs):
            launches.append(args)
            return process

        address = "agent1ownaddress"
        gateway = RelayChatGateway(configuration(tmp_path), agent_address=address, spawn_job=launch)
        ctx = Context()
        await gateway.handle_message(ctx, "alice", chat(f"@{address} {command}"))
        await asyncio.sleep(0)
        assert "Unknown command" not in ctx.text[0]
        if command in {"run relay", "pack gold, coral, mint"}:
            assert len(launches) == 1
            assert "accepted" in ctx.text[0]
        else:
            assert not launches
            assert "Commands:" in ctx.text[0] or "Available task adapters:" in ctx.text[0]
        await gateway.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command",
    [
        "@agent1otheraddress run relay",
        "@agent1ownaddress @agent1ownaddress run relay",
        "@agent1ownaddress @agent1otheraddress run relay",
        "@agent1ownaddressrun relay",
        "@agent1ownaddress please run relay",
        "@agent1ownaddress `run relay`",
        "@agent1otheraddress pack coral, mint, gold",
        "@agent1ownaddress @agent1ownaddress pack coral, mint, gold",
        "@agent1ownaddresspack coral, mint, gold",
        "@agent1ownaddress pack coral, mint, gold; run relay",
        "@agent1ownaddress " + " " * 256 + "run relay",
    ],
)
def test_mentions_do_not_weaken_command_validation_or_input_limits(tmp_path, command):
    async def scenario():
        async def launch(*args, **kwargs):
            pytest.fail("Unsupported addressed command launched a process")

        gateway = RelayChatGateway(
            configuration(tmp_path), agent_address="agent1ownaddress", spawn_job=launch
        )
        ctx = Context()
        await gateway.handle_message(ctx, "alice", chat(command))
        assert terminal(ctx.sent[-1][2])
        assert "Unknown command" in ctx.text[0] or "Command too long" in ctx.text[0]
        assert gateway.active is None

    asyncio.run(scenario())


@pytest.mark.parametrize("matching_summary", [True, False])
def test_explicit_order_reaches_child_and_must_match_verified_result(tmp_path, matching_summary):
    async def scenario():
        calls = []

        async def launch(*argv, **kwargs):
            calls.append(argv)
            job = gateway.active
            job.output.mkdir()
            order = ["mint", "coral", "gold"] if matching_summary else ["coral", "gold", "mint"]
            (job.output / "summary.json").write_text(
                json.dumps(summary_for(job, order=order, packed=order, success=True))
            )
            return Process()

        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        ctx = Context()
        await gateway.handle_message(ctx, "alice", chat("Please pack mint then coral then gold"))
        job = gateway.active
        assert job.task_spec.order == ("mint", "coral", "gold")
        await job.task
        index = calls[0].index("--order")
        assert calls[0][index + 1 : index + 4] == ("mint", "coral", "gold")
        request = json.loads((tmp_path / f"{job.task_id}.request.json").read_text())
        assert request["task"]["order"] == ["mint", "coral", "gold"]
        assert request["task"]["goal"] == job.task_spec.goal
        assert "mint, then coral, then gold" in ctx.text[0]
        if matching_summary:
            assert "SUCCESS" in ctx.text[-1]
            assert "Order: mint, coral, gold" in ctx.text[-1]
        else:
            assert "summary was invalid" in ctx.text[-1]
            assert "SUCCESS" not in ctx.text[-1]

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["error", "missing", "invalid", "success", "partial", "timeout"])
def test_job_terminal_outcomes(tmp_path, kind):
    async def scenario():
        process = Process(exit_code=2 if kind == "error" else 0, done=kind != "timeout")

        async def launch(*argv, **kwargs):
            job = gateway.active
            job.output.mkdir()
            if kind not in {"missing", "timeout"}:
                summary = summary_for(job)
                if kind == "success":
                    summary.update(success=True, packed=summary["order"])
                if kind == "partial":
                    summary.update(packed=[summary["order"][0], None, None])
                if kind == "invalid":
                    summary.update(success=True)  # Success with an empty tray must be rejected.
                (job.output / "summary.json").write_text(json.dumps(summary))
            return process

        gateway = RelayChatGateway(configuration(tmp_path, timeout_seconds=0.01), spawn_job=launch)
        ctx = Context()
        await gateway.handle_message(ctx, "alice", chat("run relay"))
        await gateway.active.task
        expected = {
            "error": "exit 2",
            "missing": "without a valid summary",
            "invalid": "summary was invalid",
            "success": "SUCCESS",
            "partial": "Packed 1/3",
            "timeout": "TIME LIMIT",
        }[kind]
        assert expected in ctx.text[-1]
        assert terminal(ctx.sent[-1][2])
        assert gateway.active is None
        if kind == "timeout":
            assert process.terminated

    asyncio.run(scenario())


def test_progress_checks_task_and_scope_deduplicates_and_waits_for_complete_lines(tmp_path):
    async def scenario():
        process = Process(done=False)

        async def launch(*args, **kwargs):
            return process

        gateway = RelayChatGateway(configuration(tmp_path), spawn_job=launch)
        ctx = Context()
        await gateway.handle_message(ctx, "alice", chat("run relay"))
        job = gateway.active
        job.output.mkdir()
        path = job.output / "execution.jsonl"
        event = event_for(job, kind="blocked")
        path.write_text(
            "\n".join(
                json.dumps(item)
                for item in [
                    event_for(job, task_id="wrong"),
                    event_for(job, scope="episode"),
                    event,
                    event,
                ]
            )
            + "\n"
        )
        await gateway._progress(job)
        assert len(ctx.text) == 2
        assert "blocked instruction" in ctx.text[-1]
        partial = json.dumps(event_for(job, kind="transfer_ready"))
        with path.open("a") as stream:
            stream.write(partial)
        await gateway._progress(job)
        assert len(ctx.text) == 2
        with path.open("a") as stream:
            stream.write("\n")
        await gateway._progress(job)
        assert "released and stationary" in ctx.text[-1]
        await asyncio.sleep(0)
        await gateway.close()

    asyncio.run(scenario())


def test_seen_messages_bounded_and_duplicate_id_scoped_to_session(tmp_path):
    async def scenario():
        gateway = RelayChatGateway(configuration(tmp_path))
        ctx = Context()
        repeated = uuid4()
        await gateway.handle_message(ctx, "alice", chat("help", msg_id=repeated))
        other = Context()
        await gateway.handle_message(other, "alice", chat("help", msg_id=repeated))
        assert len(other.text) == 1
        for _ in range(MAX_SEEN_MESSAGES + 2):
            await gateway.handle_message(ctx, "alice", chat("help"))
        assert len(gateway.seen) == MAX_SEEN_MESSAGES

    asyncio.run(scenario())


def test_local_agent_uses_standard_acp_and_no_public_registration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def scenario():
        agent = build_gateway(configuration(tmp_path))
        assert isinstance(agent._registration_policy, LocalRegistration)
        assert agent._publish_agent_details is False
        assert agent._use_mailbox is False
        assert agent._report_events is False
        assert chat_protocol_spec.digest in agent.protocols
        assert agent.relay_gateway.config.policy == "smolvla"
        assert agent.relay_gateway.agent_address == agent.address
        # The SDK's unconditionally invoked startup status hook is a local no-op.
        import uagents.agent

        async def fail_network(*args, **kwargs):
            pytest.fail("Local startup contacted Almanac")

        monkeypatch.setattr(uagents.agent, "update_agent_status", fail_network)
        await agent.run_startup_tasks()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "updates",
    [
        {"max_steps": 601},
        {"policy": "http"},
        {"timeout_seconds": 3601},
        {"default_seed": -1},
        {"seed": ""},
    ],
)
def test_operator_configuration_is_bounded(updates):
    with pytest.raises(ValueError):
        GatewayConfig(**{"seed": "test-secret", **updates})


@pytest.mark.parametrize("command", ["run relay seed 7", "pack gold then mint then coral"])
def test_real_acp_chat_launches_fetch_simulator_and_returns_observed_success(
    tmp_path, monkeypatch, command
):
    """Real SDK dispatch, ACP serialization, subprocess, MuJoCo and Fetch peer bus.

    Both ACP endpoints are registered in the SDK's local dispatcher, so no HTTP
    server, account, public resolver, or action-model download is needed.
    """
    from uagents import Agent, Protocol
    from uagents.dispatch import dispatcher
    from uagents_core.types import DeliveryStatus

    monkeypatch.chdir(tmp_path)

    async def scenario():
        config = GatewayConfig(
            seed=str(uuid4()),
            output_root=tmp_path / "episodes",
            local=True,
            policy="scripted",
            video=False,
            max_steps=600,
            timeout_seconds=30,
            poll_seconds=0.01,
            progress_seconds=0,
        )
        gateway = build_gateway(config)
        client = Agent(
            name=f"relay-test-client-{uuid4().hex[:8]}",
            seed=str(uuid4()),
            registration_policy=LocalRegistration(),
            publish_agent_details=False,
            report_events=False,
            mark_inactive_on_shutdown=False,
            enable_agent_inspector=False,
        )
        received = []
        completed = asyncio.Event()
        protocol = Protocol(spec=chat_protocol_spec)

        @protocol.on_message(ChatAcknowledgement)
        async def acknowledged(ctx, sender, message):
            received.append((sender, ctx.session, message))

        @protocol.on_message(ChatMessage)
        async def reported(ctx, sender, message):
            received.append((sender, ctx.session, message))
            await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=message.msg_id))
            if terminal(message):
                completed.set()

        client.include(protocol)
        context = client._build_context()
        request = chat(command)
        try:
            gateway.start_message_receivers()
            client.start_message_receivers()
            result = await context.send(gateway.address, request)
            assert result.status == DeliveryStatus.DELIVERED
            await asyncio.wait_for(completed.wait(), timeout=35)
            if gateway.relay_gateway.active:
                await gateway.relay_gateway.active.task

            assert isinstance(received[0][2], ChatAcknowledgement)
            assert received[0][2].acknowledged_msg_id == request.msg_id
            assert all(
                sender == gateway.address and session == context.session
                for sender, session, _ in received
            )
            reports = [message for _, _, message in received if isinstance(message, ChatMessage)]
            assert "accepted" in reports[0].text()
            assert any("observed" in message.text() for message in reports[1:-1])
            assert "SUCCESS" in reports[-1].text(), reports[-1].text()
            assert "Packed 3/3" in reports[-1].text()
            assert "scripted reference" in reports[-1].text()
            assert sum(terminal(message) for message in reports) == 1

            summaries = list(config.output_root.glob("*/summary.json"))
            assert len(summaries) == 1
            summary = json.loads(summaries[0].read_text())
            assert summary["success"] is True
            assert summary["packed"] == summary["order"]
            if command.startswith("pack"):
                assert summary["order"] == ["gold", "mint", "coral"]
            assert summary["transport"] == "fetch"
            assert summary["messages"]["delivered"] > 0
            assert 0 < summary["steps"] <= 600
            assert (summaries[0].parent / "execution.jsonl").stat().st_size > 0
        finally:
            await gateway.relay_gateway.close()
            receivers = []
            for agent in (gateway, client):
                if agent._message_queue_task:
                    agent._message_queue_task.cancel()
                    receivers.append(agent._message_queue_task)
            await asyncio.gather(*receivers, return_exceptions=True)
            for agent in (gateway, client):
                dispatcher.unregister(agent.address, agent)

    asyncio.run(scenario())
