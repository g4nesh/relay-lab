import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from relay_lab.fetch_transport import FetchMessageBus, FetchTransportError
from relay_lab.types import Action, Instruction, Observation


@pytest.fixture
def coordinator():
    pytest.importorskip("uagents")
    with FetchMessageBus(task_id="orchestration-test", orchestration=True) as bus:
        yield bus


def observation(agent_id="picker", tick=0, **changes):
    return replace(
        Observation(
            tick=tick,
            agent_id=agent_id,
            ee_position=(0.0, 0.0, 0.3),
            holding=None,
            objects={"mint": (0.1, 0.1, 0.03)},
            targets={"relay": (0.0, 0.0, 0.03)},
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            wrist_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        ),
        **changes,
    )


def digest(bus, *, agent_id="picker", before=None, after=None, action=None):
    before = before or observation(agent_id)
    after = after or replace(before, tick=before.tick + 1)
    return bus.digest(
        agent_id,
        before,
        after,
        Instruction("pick", "Pick the mint parcel.", object_id="mint"),
        action or Action(),
        [],
    )


def test_real_sdk_coordinator_returns_grounded_feedback_without_camera_pixels(coordinator):
    result = digest(coordinator, action=Action(delta=(0.03, 0.0, 0.0), gripper=-1))
    assert result["report"]["requested_delta"] == [0.03, 0.0, 0.0]
    assert result["report"]["observed_delta"] == [0.0, 0.0, 0.0]
    assert "No attachment observed" in result["text"]
    assert "cause unconfirmed" in result["text"]
    assert result["critical"] is True
    request = next(iter(coordinator._orchestration_seen.values()))[0]
    decoded = json.loads(request["payload_json"])
    assert "rgb" not in decoded["before"]
    assert "wrist_rgb" not in decoded["after"]
    info = coordinator.transport_info
    assert set(info["addresses"]) == {"picker", "packer", "coordinator"}
    assert info["orchestration_requests"] == info["orchestration_receipts"] == 1
    assert info["acknowledgements"] == info["sdk_handler_deliveries"] == 0
    assert coordinator.orchestration_transcript[0]["result"] == result


def test_context_is_robot_private_and_critical_is_consumed_once(coordinator):
    digest(coordinator, after=observation(tick=1, holding="mint"))
    assert coordinator.context("packer", [], 1) == {"text": "", "critical": False}
    result = coordinator.context("picker", [], 1)
    assert "holding none -> mint" in result["text"]
    assert result["critical"] is True
    assert coordinator.context("picker", [], 1) == {**result, "critical": False}


def test_critical_transition_is_latched_across_later_noncritical_actions(coordinator):
    holding = observation(tick=1, holding="mint")
    digest(coordinator, after=holding)
    latest = digest(coordinator, before=holding, after=replace(holding, tick=2))
    assert latest["report"]["critical"] is False
    assert latest["critical"] is True
    assert coordinator.context("picker", [], 2)["critical"] is True
    assert coordinator.context("picker", [], 2)["critical"] is False


def test_peer_feedback_is_attributed_and_critical_only_on_first_receipt(coordinator):
    event = {
        "task_id": coordinator.task_id,
        "agent_id": "picker",
        "event_id": "handoff-1",
        "tick": 1,
        "kind": "transfer_ready",
        "object_id": "mint",
        "target_id": "relay",
    }
    result = coordinator.context("packer", [event], 2)
    assert "Peer picker reports" in result["text"]
    assert result["critical"] is True
    assert coordinator.context("packer", [event], 2)["critical"] is False
    assert coordinator.context("picker", [], 2)["text"] == ""


def test_disabling_peer_messages_preserves_own_fetch_feedback(monkeypatch):
    pytest.importorskip("uagents")
    import aiohttp
    import requests
    from uagents import Agent

    def forbidden(*args, **kwargs):
        pytest.fail("Local action orchestration must not contact external services")

    monkeypatch.setattr(Agent, "_update_agent_status", forbidden)
    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    with FetchMessageBus(enabled=False, orchestration=True) as bus:
        assert bus.send("picker", "relay-ready:mint", 0) is None
        result = digest(bus)
        assert result["report"]["agent_id"] == "picker"
        assert bus.context("picker", [], 1)["text"]
        assert bus.transport_info["sent"] == 0
        assert bus.transport_info["orchestration_receipts"] == 2


def test_duplicate_request_does_not_apply_action_twice(coordinator):
    first = digest(coordinator)
    original = next(iter(coordinator._orchestration_seen.values()))[0]
    assert coordinator._exchange_orchestration(original) == first
    assert coordinator.transport_info["orchestration_requests"] == 1
    assert coordinator.transport_info["orchestration_receipts"] == 2
    assert coordinator.transport_info["orchestration_duplicates"] == 1


def test_reusing_request_id_with_changed_contents_is_rejected(coordinator):
    digest(coordinator)
    changed = dict(next(iter(coordinator._orchestration_seen.values()))[0])
    changed["phase"] = "context"
    changed["payload_json"] = json.dumps({"peer_events": []})
    coordinator.timeout_seconds = 0.05
    with pytest.raises(FetchTransportError, match="reply timed out"):
        coordinator._exchange_orchestration(changed)
    assert coordinator.transport_info["orchestration_rejected"] == 1
    assert coordinator.transport_info["orchestration_requests"] == 1


def test_reply_from_wrong_sender_is_not_accepted(coordinator, monkeypatch):
    original = coordinator._handle_orchestration_reply

    def forge_sender(recipient, ctx, sender, reply):
        original(recipient, ctx, coordinator._agents["packer"].address, reply)

    monkeypatch.setattr(coordinator, "_handle_orchestration_reply", forge_sender)
    coordinator.timeout_seconds = 0.05
    with pytest.raises(FetchTransportError, match="reply timed out"):
        digest(coordinator)
    assert coordinator.transport_info["orchestration_receipts"] == 0
    assert coordinator.transport_info["orchestration_rejected"] == 1


@pytest.mark.parametrize("invalid", ["task", "sequence", "tick", "robot", "pixels", "oversized"])
def test_invalid_requests_are_rejected_without_direct_fallback(coordinator, invalid):
    digest(coordinator)
    request = dict(next(iter(coordinator._orchestration_seen.values()))[0])
    request.update(request_id=str(uuid4()), sequence=2)
    body = json.loads(request["payload_json"])
    body["before"]["tick"] = 1
    body["after"]["tick"] = 2
    request["tick"] = 2
    if invalid == "task":
        request["task_id"] = "different-task"
    elif invalid == "sequence":
        request["sequence"] = 4
    elif invalid == "tick":
        body["before"]["tick"] = 0
    elif invalid == "robot":
        body["before"]["agent_id"] = "packer"
    elif invalid == "pixels":
        body["after"]["rgb"] = [[[0, 0, 0]]]
    elif invalid == "oversized":
        body["instruction"]["text"] = "x" * 33000
    request["payload_json"] = json.dumps(body)
    coordinator.timeout_seconds = 0.05
    with pytest.raises(FetchTransportError, match="reply timed out"):
        coordinator._exchange_orchestration(request)
    assert coordinator.transport_info["orchestration_rejected"] == 1
    assert coordinator.transport_info["orchestration_requests"] == 1
    with pytest.raises(FetchTransportError, match="rollout stopped"):
        coordinator.context("picker", [], 2)


def test_lost_reply_stops_model_context_instead_of_using_local_fallback(coordinator, monkeypatch):
    monkeypatch.setattr(coordinator, "_handle_orchestration_reply", lambda *_: None)
    coordinator.timeout_seconds = 0.05
    with pytest.raises(FetchTransportError, match="reply timed out"):
        digest(coordinator)
    assert coordinator.orchestration_transcript == []
    assert coordinator.transport_info["orchestration_requests"] == 1
    assert coordinator.transport_info["orchestration_receipts"] == 0


def test_reset_clears_coordinator_memory_and_rejects_old_session(coordinator):
    digest(coordinator, after=observation(tick=1, holding="mint"))
    old_session = coordinator._session
    coordinator.reset()
    assert coordinator.context("picker", [], 0) == {"text": "", "critical": False}
    assert coordinator.transport_info["orchestration_requests"] == 1

    async def stale_sender_context():
        coordinator._contexts["picker"]._session = old_session

    asyncio.run_coroutine_threadsafe(stale_sender_context(), coordinator._loop).result(1)
    coordinator.timeout_seconds = 0.05
    with pytest.raises(FetchTransportError, match="reply timed out"):
        digest(coordinator)
    assert coordinator.transport_info["orchestration_rejected"] == 1


def test_oversized_public_payload_fails_before_any_sdk_request(coordinator):
    with pytest.raises(ValueError, match="32768"):
        coordinator.context("picker", [{"data": "x" * 33000}], 0)
    assert coordinator.transport_info["orchestration_requests"] == 0
    assert coordinator.context("picker", [], 0) == {"text": "", "critical": False}


def test_orchestration_must_be_explicitly_enabled():
    with (
        FetchMessageBus(enabled=False) as bus,
        pytest.raises(FetchTransportError, match="not enabled"),
    ):
        bus.context("picker", [], 0)
