"""Real local SpacetimeDB checks; opt in with RELAY_SPACETIME_INTEGRATION=1."""

import os
from uuid import uuid4

import pytest

from relay_lab.factory.spacetime import SpacetimeCoordinator
from relay_lab.spacetime import SpacetimeConfig, SpacetimeError, SpacetimeUnavailable


@pytest.fixture
def backend():
    if os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1":
        pytest.skip("Set RELAY_SPACETIME_INTEGRATION=1 for real factory database tests")
    backend = SpacetimeCoordinator(SpacetimeConfig.from_env())
    backend.create(
        "factory-check-" + uuid4().hex,
        "worker-test",
        [
            {"robot_id": "arm-01", "capabilities": ["transfer", "wait"]},
            {"robot_id": "amr-01", "capabilities": ["navigate", "wait"]},
        ],
        [{"job_id": "job-1", "stage": "queued", "custodian": None}],
        ["crossing", "fixture", "receive"],
    )
    return backend


def command(cid="cmd-1", robot="arm-01", **overrides):
    return {
        "command_id": cid,
        "robot_id": robot,
        "capability": "transfer",
        "job_id": "job-1",
        "resources": ["fixture"],
        "lease_ticks": 2,
        "payload": {"expected_stage": "queued", "next_stage": "loaded"},
        **overrides,
    }


def message(mid="request-1", seq=0, **overrides):
    return {
        "message_id": mid,
        "sender": "arm-01",
        "recipient": "amr-01",
        "sequence": seq,
        "kind": "request",
        "correlation_id": None,
        "expires_tick": 30,
        "priority": 5,
        "confidence": 1,
        "payload": {"job_id": "job-1", "command_id": "cmd-1"},
        **overrides,
    }


def observed(backend, cmd, stage="queued", status="running"):
    state = backend.snapshot()["state"]
    state["tick"] += 1
    state["jobs"]["job-1"]["stage"] = stage
    return state, [
        {
            "command_id": cmd["command_id"],
            "robot_id": cmd["robot_id"],
            "status": status,
            "observed": {"stage": stage, "custodian": None},
        }
    ]


def test_database_cas_resources_and_atomic_messages(backend):
    a = command(resources=["crossing"])
    b = command("cmd-2", "amr-01", capability="navigate", job_id=None, resources=["crossing"])
    with pytest.raises(SpacetimeError):
        backend.prepare(1, [a])
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [a, b], [message()])
    assert backend.snapshot()["phase"] == "ready"
    assert not backend.pending("amr-01")
    assert not backend.snapshot()["resources"]["crossing"]["holder"]
    assert backend.prepare(0, [a], [message()]) == [a]
    assert backend.prepare(0, [a], [message()]) == [a]
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [{**a, "lease_ticks": 4}], [message()])


def test_database_active_command_cannot_replay_after_terminal(backend):
    cmd = command()
    backend.prepare(0, [cmd])
    state, outcomes = observed(backend, cmd)
    backend.commit(0, state, outcomes)
    backend.commit(0, state, outcomes)
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [cmd])
    with pytest.raises(SpacetimeError):
        backend.prepare(1, [])
    backend.prepare(1, [cmd])
    assert backend.snapshot()["resources"]["fixture"]["expires_tick"] == 3
    state, outcomes = observed(backend, cmd, stage="loaded", status="completed")
    backend.commit(1, state, outcomes)
    assert not backend.snapshot()["resources"]["fixture"]["holder"]
    with pytest.raises(SpacetimeError):
        backend.prepare(2, [cmd])


def test_database_message_order_expiry_and_wrong_sender_ack(backend):
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [], [message(seq=2)])
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [], [message(expires_tick=0)])
    bad_ack = message("ack-1", 1, kind="ack", correlation_id="request-1")
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [], [message(), bad_ack])
    good_ack = message(
        "ack-1", 0, sender="amr-01", recipient="arm-01", kind="ack", correlation_id="request-1"
    )
    backend.prepare(0, [], [message(), good_ack])
    assert not backend.pending("amr-01")


def test_database_rejects_fabricated_success_and_capability(backend):
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [command(robot="amr-01")])
    cmd = command()
    backend.prepare(0, [cmd])
    state, outcomes = observed(backend, cmd, stage="completed", status="completed")
    state["success"] = True
    with pytest.raises(SpacetimeError):
        backend.commit(0, state, outcomes)
    assert backend.snapshot()["phase"] == "executing"
    assert backend.snapshot()["revision"] == 0


def test_unavailable_backend_never_returns_an_authorization(monkeypatch):
    backend = SpacetimeCoordinator(SpacetimeConfig(token="test-only-token"))
    backend.run_id, backend.worker_id = "offline-test", "worker-test"
    calls = []

    def offline(*args, **kwargs):
        calls.append(args)
        raise SpacetimeUnavailable("offline")

    monkeypatch.setattr(backend.client, "_request", offline)
    with pytest.raises(SpacetimeUnavailable):
        backend.prepare(0, [command()])
    assert calls


def test_database_acknowledged_handoff_and_station_custody(backend):
    handoff = SpacetimeCoordinator(backend.client.config)
    handoff.create(
        "factory-handoff-" + uuid4().hex,
        "worker-test",
        [
            {"robot_id": "arm-01", "capabilities": ["transfer"]},
            {"robot_id": "amr-01", "capabilities": ["navigate"]},
        ],
        [{"job_id": "job-1", "stage": "at_cell", "custodian": "amr-01"}],
        ["fixture", "amr-01", "arm-01"],
    )
    cmd = command(
        payload={
            "expected_stage": "at_cell",
            "next_stage": "in_fixture",
            "handoff_message_id": "request-1",
            "target": "fixture",
        }
    )
    with pytest.raises(SpacetimeError):
        handoff.prepare(0, [cmd])
    ack = message(
        "ack-1", 0, sender="amr-01", recipient="arm-01", kind="ack", correlation_id="request-1"
    )
    handoff.prepare(0, [cmd], [message(), ack])
    state, outcomes = observed(handoff, cmd, stage="at_cell")
    state["jobs"]["job-1"]["custodian"] = "arm-01"
    outcomes[0]["observed"]["custodian"] = "arm-01"
    handoff.commit(0, state, outcomes)
    handoff.prepare(1, [cmd])
    state, outcomes = observed(handoff, cmd, stage="in_fixture", status="completed")
    state["jobs"]["job-1"]["custodian"] = "fixture"
    outcomes[0]["observed"]["custodian"] = "fixture"
    assert handoff.commit(1, state, outcomes)["state"]["jobs"]["job-1"]["custodian"] == "fixture"


def test_database_abort_revokes_old_authorizations(backend):
    cmd = command()
    backend.prepare(0, [cmd])
    assert backend.abort("simulator stopped")["status"] == "failed"
    assert backend.abort("simulator stopped")["status"] == "failed"
    assert not backend.snapshot()["resources"]["fixture"]["holder"]
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [cmd])


def test_database_implicit_actor_lease_and_required_resources(backend):
    transfer = command(resources=["amr-01"])
    move = command("cmd-2", "amr-01", capability="navigate", job_id=None, resources=["crossing"])
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [transfer, move])
    missing = command(resources=["crossing"], payload={"source": "amr-01", "target": "fixture"})
    with pytest.raises(SpacetimeError):
        backend.prepare(0, [missing])
    backend.prepare(0, [transfer])
    state, outcomes = observed(backend, transfer)
    backend.commit(0, state, outcomes)
    with pytest.raises(SpacetimeError):
        backend.prepare(1, [transfer, move])
    assert backend.snapshot()["resources"]["amr-01"]["holder"] == "arm-01"


def test_database_nontransfer_cannot_reassign_custody(backend):
    move = command(robot="amr-01", capability="navigate")
    backend.prepare(0, [move])
    state, outcomes = observed(backend, move)
    state["jobs"]["job-1"]["custodian"] = "amr-01"
    outcomes[0]["observed"]["custodian"] = "amr-01"
    with pytest.raises(SpacetimeError):
        backend.commit(0, state, outcomes)
    assert backend.snapshot()["revision"] == 0
