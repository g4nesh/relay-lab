import asyncio
import json

import pytest

from relay_lab.fetch_transport import FetchMessageBus, FetchTransportError


@pytest.fixture
def bus():
    pytest.importorskip("uagents")
    with FetchMessageBus(task_id="test-task", latency_ticks=2) as transport:
        yield transport


def test_real_sdk_acknowledgement_and_simulation_tick_delivery(bus):
    message = bus.send("packer", "request:mint", 4)
    assert bus.receive("picker", 5) == []
    assert bus.receive("packer", 6) == []
    assert bus.receive("picker", 6) == [message]
    assert bus.receive("picker", 7) == []
    info = bus.transport_info
    assert info["sdk_handler_deliveries"] == info["acknowledgements"] == 1
    assert info["sent"] == info["delivered"] == 1
    assert len(set(info["addresses"].values())) == 2
    assert all(address.startswith("agent1") for address in info["addresses"].values())
    wire = bus.wire_transcript[0]
    assert wire["task_id"] == "test-task"
    assert wire["session_id"] == info["session_id"]
    assert "seed" not in json.dumps(info)


def test_local_transport_never_calls_external_registration_or_http(monkeypatch):
    pytest.importorskip("uagents")
    import aiohttp
    import requests
    from uagents import Agent

    def forbidden(*args, **kwargs):
        raise AssertionError("Local Fetch peers must not contact external services")

    monkeypatch.setattr(Agent, "_update_agent_status", forbidden)
    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    with FetchMessageBus() as transport:
        message = transport.send("picker", "relay-ready:mint", 0)
        assert transport.receive("packer", 1) == [message]
        assert transport.transport_info["public_registration"] is False


def test_feedback_is_typed_authenticated_and_delayed_separately(bus):
    event = {
        "task_id": bus.task_id,
        "agent_id": "picker",
        "tick": 10,
        "kind": "transfer_ready",
        "object_id": "mint",
        "evidence": {"released": True, "arm_clear": True},
    }
    receipt = bus.publish_event("picker", event)
    assert receipt["event"] == event
    assert receipt["recipient"] == "packer"
    assert bus.receive("packer", 12) == []
    assert bus.receive_events("picker", 12) == []
    assert bus.receive_events("packer", 11) == []
    assert bus.receive_events("packer", 12) == [event]
    assert bus.receive_events("packer", 13) == []
    assert bus.transport_info["feedback_sent"] == 1
    assert bus.transport_info["feedback_delivered"] == 1
    assert bus.transport_info["acknowledgements"] == 1
    with pytest.raises(ValueError, match="agent_id"):
        bus.publish_event("packer", event)
    with pytest.raises(ValueError, match="task_id"):
        bus.publish_event("picker", {**event, "task_id": "another-task"})


def test_missing_acknowledgement_stops_without_direct_fallback(bus, monkeypatch):
    bus.timeout_seconds = 0.05
    monkeypatch.setattr(bus, "_handle_acknowledgement", lambda *_: None)
    with pytest.raises(FetchTransportError, match="acknowledgement timed out"):
        bus.send("picker", "relay-ready:mint", 0)
    assert bus.transport_info["sdk_handler_deliveries"] == 1
    assert bus.sent == 0
    assert bus.transcript == []
    with pytest.raises(FetchTransportError, match="rollout stopped"):
        bus.receive("packer", 999)


def test_duplicate_wire_message_is_acknowledged_without_second_delivery(bus):
    first = bus.send("picker", "relay-ready:mint", 0)
    original = {key: value for key, value in bus.wire_transcript[0].items() if key != "session_id"}
    bus._exchange(original)
    assert bus.receive("packer", 2) == [first]
    assert bus.transport_info["sdk_handler_deliveries"] == 1
    assert bus.transport_info["duplicates"] == 1
    assert bus.transport_info["acknowledgements"] == 2


def test_receiver_rejects_other_task_without_acknowledging(bus):
    bus.timeout_seconds = 0.05
    payload = bus._payload("picker", "relay-ready:mint", 0)
    payload["task_id"] = "different-run"
    with pytest.raises(FetchTransportError, match="timed out"):
        bus._exchange(payload)
    assert bus.transport_info["rejected"] == 1
    assert bus.transport_info["acknowledgements"] == 0


def test_receiver_rejects_feedback_claiming_another_robot(bus):
    bus.timeout_seconds = 0.05
    payload = bus._payload(
        "picker",
        "",
        0,
        event_json=json.dumps({"agent_id": "packer", "task_id": bus.task_id, "tick": 0}),
    )
    with pytest.raises(FetchTransportError, match="timed out"):
        bus._exchange(payload)
    assert bus.transport_info["rejected"] == 1


def test_disabled_transport_does_not_import_or_start_sdk(monkeypatch):
    import relay_lab.fetch_transport as module

    monkeypatch.setattr(module, "_sdk", lambda: pytest.fail("SDK must not be loaded"))
    with FetchMessageBus(enabled=False) as transport:
        assert transport.send("packer", "request:gold", 0) is None
        assert transport.receive("picker", 999) == []
        assert transport.transport_info["addresses"] == {}
        assert transport.dropped == 1


def test_queue_bounds_and_reset_make_a_new_session():
    pytest.importorskip("uagents")
    with FetchMessageBus(max_pending=1, max_history=1) as transport:
        transport.send("packer", "request:mint", 0)
        assert transport.send("packer", "request:gold", 0) is None
        assert transport.transport_info["sdk_handler_deliveries"] == 1
        previous_session = transport.transport_info["session_id"]
        transport.reset()
        assert transport.transport_info["session_id"] != previous_session
        assert transport.receive("picker", 1) == []
        assert (transport.sent, transport.delivered, transport.dropped) == (0, 0, 0)
        second = transport.send("packer", "request:gold", 1)
        assert transport.receive("picker", 2) == [second]


def test_close_stops_thread_and_unregisters_sdk_recipients(bus):
    from uagents.dispatch import dispatcher

    addresses = list(bus.transport_info["addresses"].values())
    bus.close()
    assert not bus._thread.is_alive()
    assert all(not dispatcher.contains(address) for address in addresses)
    with pytest.raises(FetchTransportError, match="closed"):
        bus.send("picker", "relay-ready:mint", 0)
    bus.close()


def test_stale_envelope_session_is_rejected(bus):
    old_session = bus._session
    bus.reset()
    bus.timeout_seconds = 0.05

    async def restore_old_context():
        bus._contexts["picker"]._session = old_session

    asyncio.run_coroutine_threadsafe(restore_old_context(), bus._loop).result(1)
    with pytest.raises(FetchTransportError, match="timed out"):
        bus.send("picker", "relay-ready:mint", 0)
    assert bus.transport_info["rejected"] == 1


@pytest.mark.parametrize("timeout", [0, -1, 61, float("nan")])
def test_invalid_timeouts_fail_before_starting_sdk(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        FetchMessageBus(timeout_seconds=timeout)
