import pytest

from relay_lab.communication import LocalMessageBus


def test_delivery_is_delayed_private_and_consumed_once():
    bus = LocalMessageBus(latency_ticks=2)
    message = bus.send("packer", "request:mint", tick=4)
    assert message.deliver_tick == 6
    assert bus.receive("picker", 4) == []
    assert bus.receive("picker", 5) == []
    assert bus.receive("packer", 6) == []
    assert bus.receive("picker", 6) == [message]
    assert bus.receive("picker", 7) == []
    assert (bus.sent, bus.delivered, bus.dropped) == (1, 1, 0)


def test_disabled_bus_and_bounded_queue():
    disabled = LocalMessageBus(enabled=False)
    assert disabled.send("packer", "request:mint", 0) is None
    assert disabled.receive("picker", 100) == []
    assert disabled.transcript == []
    bus = LocalMessageBus(max_pending=1, max_history=1)
    first = bus.send("packer", "request:mint", 0)
    assert bus.send("packer", "request:gold", 0) is None
    assert bus.dropped == 1
    assert bus.receive("picker", 1) == [first]
    second = bus.send("packer", "request:gold", 1)
    assert bus.transcript == [second]
    bus.reset()
    assert bus.transcript == []
    assert bus.receive("picker", 100) == []
    assert (bus.sent, bus.delivered, bus.dropped) == (0, 0, 0)


def test_invalid_transport_parameters_fail_early():
    with pytest.raises(ValueError, match="latency_ticks"):
        LocalMessageBus(latency_ticks=0)
    with pytest.raises(ValueError, match="characters"):
        LocalMessageBus(max_message_length=4).send("picker", "hello", 0)
