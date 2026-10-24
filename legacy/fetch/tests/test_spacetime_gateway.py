"""The chat gateway verifies the database result, then delivers it through ACP."""

import asyncio
import json
import os
from pathlib import Path
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

from relay_lab.fetch_gateway import (
    GatewayConfig,
    LocalRegistration,
    RelayChatGateway,
    RelayJob,
    build_gateway,
)
from relay_lab.spacetime import SpacetimeClient, SpacetimeConfig, SpacetimeUnavailable


def _terminal(message):
    return isinstance(message, ChatMessage) and any(
        isinstance(part, EndSessionContent) for part in message.content
    )


@pytest.mark.parametrize(
    "database_result",
    [
        "matching_success",
        "matching_exhaustion",
        "missing",
        "status",
        "phase",
        "tick",
        "outcome",
        "unavailable",
    ],
)
def test_gateway_requires_matching_database_terminal_evidence(
    tmp_path, monkeypatch, database_result
):
    config = GatewayConfig(seed="unit-test-seed", output_root=tmp_path, coordination="spacetime")
    gateway = RelayChatGateway(config)
    job = RelayJob(
        task_id=str(uuid4()),
        owner="chat-owner",
        session=str(uuid4()),
        context=None,
        seed=7,
        output=tmp_path / "episode",
    )
    success = database_result != "matching_exhaustion"
    order = ["coral", "mint", "gold"]
    packed, steps = (order, 329) if success else ([None, None, None], 600)
    summary = {
        "task_id": job.task_id,
        "policy": "smolvla",
        "transport": "fetch",
        "coordination": "spacetime",
        "success": success,
        "packed": packed,
        "order": order,
        "steps": steps,
    }
    job.output.mkdir()
    (job.output / "summary.json").write_text(json.dumps(summary))
    state = {
        "status": "succeeded" if success else "exhausted",
        "phase": "terminal",
        "next_tick": steps,
        "outcome_json": json.dumps(
            {"tick": steps, "packed": packed, "order": order, "success": success}
        ),
    }
    if database_result == "status":
        state["status"] = "running"
    elif database_result == "phase":
        state["phase"] = "executing"
    elif database_result == "tick":
        state["next_tick"] += 1
    elif database_result == "outcome":
        state["outcome_json"] = json.dumps(
            {"tick": steps, "packed": [None] * 3, "order": order, "success": False}
        )
    looked_up = []

    def get_task(self, task_id):
        looked_up.append(task_id)
        if database_result == "unavailable":
            raise SpacetimeUnavailable("Database unavailable")
        return None if database_result == "missing" else state

    monkeypatch.setattr(
        SpacetimeConfig,
        "from_env",
        classmethod(lambda cls: SpacetimeConfig(token="unit-test-token")),
    )
    monkeypatch.setattr(SpacetimeClient, "get_task", get_task)
    result = gateway._summary(job, 0)
    assert looked_up == [job.task_id]
    if database_result == "matching_success":
        assert "SUCCESS" in result and "Packed 3/3" in result
        assert "Coordination: SpacetimeDB" in result
    elif database_result == "matching_exhaustion":
        assert "TASK NOT COMPLETED" in result and "Packed 0/3" in result
        assert "Coordination: SpacetimeDB" in result
    else:
        assert "outcome is unverified" in result
        assert "SUCCESS" not in result and "TASK NOT COMPLETED" not in result


@pytest.mark.skipif(
    os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1",
    reason="Requires real local SpacetimeDB and published relay-lab module",
)
def test_real_acp_task_runs_child_simulator_and_verifies_spacetime_terminal_result(
    tmp_path, monkeypatch
):
    """Actual ACP SDK dispatch → Python child → Fetch/MuJoCo → Spacetime → ACP.

    The ACP endpoints use the SDK's local dispatcher. Database calls use the real
    published module, and the child runs the complete scripted reference task.
    This deliberately makes no claim to exercise public ASI:One routing.
    """
    from uagents import Agent, Protocol
    from uagents.dispatch import dispatcher
    from uagents_core.types import DeliveryStatus

    token_file = Path(
        os.environ.get(
            "RELAY_SPACETIME_TOKEN_FILE",
            str(Path(__file__).resolve().parents[1] / ".relay-lab" / "spacetime-token"),
        )
    ).resolve()
    monkeypatch.setenv("RELAY_SPACETIME_TOKEN_FILE", str(token_file))
    database = SpacetimeClient(SpacetimeConfig.from_env())
    monkeypatch.chdir(tmp_path)

    async def scenario():
        config = GatewayConfig(
            seed=str(uuid4()),
            output_root=tmp_path / "episodes",
            local=True,
            policy="scripted",
            coordination="spacetime",
            video=False,
            max_steps=600,
            timeout_seconds=120,
            poll_seconds=0.05,
            progress_seconds=0,
        )
        gateway = build_gateway(config)
        client = Agent(
            name=f"spacetime-test-client-{uuid4().hex[:8]}",
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
            if _terminal(message):
                completed.set()

        client.include(protocol)
        context = client._build_context()
        request = ChatMessage(content=[TextContent("pack gold then mint then coral")])
        try:
            gateway.start_message_receivers()
            client.start_message_receivers()
            sent = await context.send(gateway.address, request)
            assert sent.status == DeliveryStatus.DELIVERED
            await asyncio.wait_for(completed.wait(), timeout=125)
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
            assert "Coordination: SpacetimeDB" in reports[-1].text()
            assert "scripted reference" in reports[-1].text()
            assert sum(_terminal(message) for message in reports) == 1

            summaries = list(config.output_root.glob("*/summary.json"))
            assert len(summaries) == 1
            summary = json.loads(summaries[0].read_text())
            task_id = summary["task_id"]
            assert summary["success"] is True
            assert summary["packed"] == summary["order"] == ["gold", "mint", "coral"]
            assert summary["transport"] == "fetch" and summary["coordination"] == "spacetime"
            assert summary["messages"]["delivered"] > 0
            assert 0 < summary["steps"] <= 600
            state = database.get_task(task_id)
            assert state["status"] == "succeeded" and state["phase"] == "terminal"
            assert state["next_tick"] == summary["steps"]
            assert json.loads(state["outcome_json"])["packed"] == summary["packed"]
            assert database.get_plan(task_id, 0) is not None
            assert database.get_action(task_id, summary["steps"] - 1) is not None
            assert database.get_feedback(task_id, summary["steps"] - 1) is not None
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
