import copy

import pytest

from relay_lab.factory.coordination import CoordinationError, MemoryCoordinator, MessageQueue

ROBOTS = [
    {"robot_id": "arm-01", "capabilities": ["transfer", "wait"]},
    {"robot_id": "amr-01", "capabilities": ["navigate", "wait"]},
]


def coordinator():
    backend = MemoryCoordinator()
    backend.create(
        "factory-test",
        "worker-1",
        ROBOTS,
        [{"job_id": "job-1", "stage": "queued", "custodian": None}],
        ["crossing", "fixture"],
    )
    return backend


def command(cid="cmd-1", robot="arm-01", resources=None, lease=2):
    return {
        "command_id": cid,
        "robot_id": robot,
        "capability": "transfer",
        "job_id": "job-1",
        "resources": resources or ["fixture"],
        "lease_ticks": lease,
        "payload": {"expected_stage": "queued", "next_stage": "loaded"},
    }


def result(backend, cmd, status="running", stage="queued", tick=1):
    state = copy.deepcopy(backend.snapshot()["state"])
    state["tick"] = tick
    state["jobs"]["job-1"]["stage"] = stage
    outcomes = [
        {
            "command_id": cmd["command_id"],
            "robot_id": cmd["robot_id"],
            "status": status,
            "observed": {"stage": stage},
        }
    ]
    return state, outcomes


def message(mid="msg-1", sequence=0, **overrides):
    return {
        "message_id": mid,
        "sender": "arm-01",
        "recipient": "amr-01",
        "sequence": sequence,
        "kind": "request",
        "correlation_id": None,
        "expires_tick": 100,
        "priority": 1,
        "confidence": 0.9,
        "payload": {"subgoal": "clear crossing", "job_id": "job-1", "command_id": "cmd-1"},
        **overrides,
    }


def test_prepare_is_atomic_when_shared_resource_conflicts():
    backend = coordinator()
    a = command(resources=["crossing"])
    b = {**command("cmd-2", "amr-01", ["crossing"]), "job_id": None, "capability": "navigate"}
    with pytest.raises(CoordinationError, match="Resource leased"):
        backend.prepare(0, [a, b])
    assert backend.snapshot()["phase"] == "ready"
    assert not backend.commands
    assert all(not lock["holder"] for lock in backend.resources.values())


def test_wrong_capability_and_stale_revision_stop_before_physics():
    backend = coordinator()
    with pytest.raises(CoordinationError, match="Unsupported capability"):
        backend.prepare(0, [command(robot="amr-01")])
    with pytest.raises(CoordinationError, match="revision"):
        backend.prepare(1, [command()])


def test_stable_running_command_renews_then_finishes_and_cannot_replay():
    backend = coordinator()
    cmd = command()
    assert backend.prepare(0, [cmd]) == [cmd]
    assert backend.prepare(0, [cmd]) == [cmd]
    state, outcomes = result(backend, cmd)
    backend.commit(0, state, outcomes)
    backend.commit(0, state, outcomes)
    with pytest.raises(CoordinationError):
        backend.prepare(0, [cmd])
    with pytest.raises(CoordinationError, match="cannot be omitted"):
        backend.prepare(1, [])
    backend.prepare(1, [cmd])
    assert backend.resources["fixture"]["expires_tick"] == 3
    state, outcomes = result(backend, cmd, "completed", "loaded", 2)
    backend.commit(1, state, outcomes)
    assert backend.resources["fixture"]["holder"] == ""
    with pytest.raises(CoordinationError, match="Terminal or changed"):
        backend.prepare(2, [cmd])


def test_changed_duplicate_and_unobserved_stage_change_rejected():
    backend = coordinator()
    cmd = command()
    backend.prepare(0, [cmd])
    with pytest.raises(CoordinationError, match="Conflicting duplicate"):
        backend.prepare(0, [{**cmd, "lease_ticks": 4}])
    state, outcomes = result(backend, cmd, "completed", "completed")
    state["success"] = True
    with pytest.raises(CoordinationError, match="Invalid factory process"):
        backend.commit(0, state, outcomes)
    assert backend.snapshot()["revision"] == 0
    assert backend.snapshot()["phase"] == "executing"


def test_expired_lease_can_be_reacquired_and_live_lease_blocks():
    backend = coordinator()
    backend.resources["fixture"] = {"holder": "amr-01", "command_id": "old", "expires_tick": 1}
    with pytest.raises(CoordinationError, match="Resource leased"):
        backend.prepare(0, [command()])
    backend.resources["fixture"]["expires_tick"] = 0
    backend.prepare(0, [command()])
    assert backend.resources["fixture"]["holder"] == "arm-01"


def test_queue_dedup_causal_ack_expiry_sequence_and_bounds():
    queue = MessageQueue({"arm-01", "amr-01"}, limit=1)
    first = message()
    queue.publish(first, 0)
    queue.publish(first, 0)
    with pytest.raises(CoordinationError, match="sequence"):
        queue.publish(message("future", 3), 0)
    with pytest.raises(CoordinationError, match="queue is full"):
        queue.publish(message("next", 1), 0)
    ack = message(
        "ack-1", 0, sender="amr-01", recipient="arm-01", kind="ack", correlation_id="msg-1"
    )
    queue.publish(ack, 1)
    assert queue.pending("amr-01", 1) == []
    queue.publish(message("next", 1), 2)
    assert queue.pending("amr-01", 101) == []
    with pytest.raises(CoordinationError, match="expired"):
        queue.publish(message("expired", 2, expires_tick=2), 2)


def test_aging_prevents_perpetual_low_priority_starvation():
    queue = MessageQueue({"arm-01", "amr-01"})
    queue.publish(message("old", 0, priority=0, expires_tick=200), 0)
    queue.publish(message("new", 1, priority=10, expires_tick=200), 50)
    assert [m["message_id"] for m in queue.pending("amr-01", 50)] == ["old", "new"]


def test_failed_batch_does_not_consume_messages():
    backend = coordinator()
    invalid = command(robot="amr-01")
    with pytest.raises(CoordinationError):
        backend.prepare(0, [invalid], [message()])
    assert backend.queue.events == {}


def test_acknowledged_handoff_authorizes_mid_skill_custody_change():
    backend = MemoryCoordinator()
    backend.create(
        "handoff-test",
        "worker-1",
        ROBOTS,
        [{"job_id": "job-1", "stage": "at_cell", "custodian": "amr-01"}],
        ["fixture", "amr-01", "arm-01"],
    )
    cmd = command()
    cmd["payload"] = {
        "expected_stage": "at_cell",
        "next_stage": "in_fixture",
        "handoff_message_id": "msg-1",
        "target": "fixture",
    }
    with pytest.raises(CoordinationError, match="acknowledged handoff"):
        backend.prepare(0, [cmd])
    ack = message(
        "ack-1", 0, sender="amr-01", recipient="arm-01", kind="ack", correlation_id="msg-1"
    )
    backend.prepare(0, [cmd], [message(), ack])
    state, outcomes = result(backend, cmd, stage="at_cell")
    state["jobs"]["job-1"]["custodian"] = "arm-01"
    outcomes[0]["observed"]["custodian"] = "arm-01"
    backend.commit(0, state, outcomes)
    backend.prepare(1, [cmd])
    state, outcomes = result(backend, cmd, "completed", "in_fixture", 2)
    state["jobs"]["job-1"]["custodian"] = "fixture"
    outcomes[0]["observed"]["custodian"] = "fixture"
    backend.commit(1, state, outcomes)
    assert backend.snapshot()["state"]["jobs"]["job-1"]["custodian"] == "fixture"


def test_abort_releases_leases_and_revokes_old_batch_authority():
    backend = coordinator()
    cmd = command()
    backend.prepare(0, [cmd])
    backend.abort("worker stopped")
    assert backend.abort("worker stopped")["status"] == "failed"
    assert not backend.resources["fixture"]["holder"]
    with pytest.raises(CoordinationError):
        backend.prepare(0, [cmd])
    with pytest.raises(CoordinationError):
        backend.prepare(1, [command("cmd-2")])


def test_actor_lease_cannot_be_omitted_to_evade_peer_reservation():
    backend = coordinator()
    a = command(resources=["amr-01"])
    b = {**command("cmd-2", "amr-01", ["crossing"]), "job_id": None, "capability": "navigate"}
    with pytest.raises(CoordinationError, match="Resource leased"):
        backend.prepare(0, [a, b])
    backend.prepare(0, [a])
    state, outcomes = result(backend, a)
    backend.commit(0, state, outcomes)
    with pytest.raises(CoordinationError, match="Resource leased"):
        backend.prepare(1, [a, b])
    assert backend.resources["amr-01"]["holder"] == "arm-01"


def test_omitted_transfer_source_or_target_is_rejected():
    backend = coordinator()
    cmd = command(resources=["crossing"])
    cmd["payload"].update(source="amr-01", target="fixture")
    with pytest.raises(CoordinationError, match="omitted required"):
        backend.prepare(0, [cmd])


def test_non_transfer_cannot_change_custody_with_same_stage():
    backend = coordinator()
    cmd = {**command(robot="amr-01"), "capability": "navigate"}
    backend.prepare(0, [cmd])
    state, outcomes = result(backend, cmd)
    state["jobs"]["job-1"]["custodian"] = "amr-01"
    outcomes[0]["observed"]["custodian"] = "amr-01"
    with pytest.raises(CoordinationError, match="Only a transfer"):
        backend.commit(0, state, outcomes)
    assert backend.snapshot()["revision"] == 0


def test_handoff_ack_for_another_job_cannot_authorize_transfer():
    backend = MemoryCoordinator()
    backend.create(
        "handoff-scope",
        "worker-1",
        ROBOTS,
        [{"job_id": "job-1", "stage": "at_cell", "custodian": "amr-01"}],
        ["fixture"],
    )
    cmd = command()
    cmd["payload"] = {
        "expected_stage": "at_cell",
        "next_stage": "in_fixture",
        "handoff_message_id": "msg-1",
    }
    request = message(payload={"job_id": "another-job"})
    ack = message(
        "ack-1", 0, sender="amr-01", recipient="arm-01", kind="ack", correlation_id="msg-1"
    )
    with pytest.raises(CoordinationError, match="acknowledged handoff"):
        backend.prepare(0, [cmd], [request, ack])


def test_handoff_ack_cannot_be_reused_for_a_different_command():
    backend = MemoryCoordinator()
    backend.create(
        "handoff-command",
        "worker-1",
        ROBOTS,
        [{"job_id": "job-1", "stage": "at_cell", "custodian": "amr-01"}],
        ["fixture"],
    )
    cmd = command("cmd-2")
    cmd["payload"] = {
        "expected_stage": "at_cell",
        "next_stage": "in_fixture",
        "handoff_message_id": "msg-1",
    }
    ack = message(
        "ack-1", 0, sender="amr-01", recipient="arm-01", kind="ack", correlation_id="msg-1"
    )
    with pytest.raises(CoordinationError, match="acknowledged handoff"):
        backend.prepare(0, [cmd], [message(), ack])
