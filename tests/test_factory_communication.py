"""Equal semantic transport rules for job-scoped text and learned vectors."""

import copy
import os
from uuid import uuid4

import pytest

from relay_lab.factory.communication import SCHEMA, SemanticIndex, channel_config
from relay_lab.factory.coordination import STAGE_CAPABILITIES, STAGES, MemoryCoordinator
from relay_lab.factory.spacetime import SpacetimeCoordinator
from relay_lab.spacetime import SpacetimeConfig, SpacetimeError, SpacetimeUnavailable

SHA = "a" * 64
ROBOTS = [
    {"robot_id": robot, "capabilities": ["transfer", "navigate", "process", "wait"]}
    for robot in ("humanoid-01", "gantry-01")
]


@pytest.fixture(params=["memory", "spacetime"])
def backend(request):
    if request.param == "spacetime":
        if os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1":
            pytest.skip("Real semantic database tests are opt-in")
        result = SpacetimeCoordinator(SpacetimeConfig.from_env())
    else:
        result = MemoryCoordinator()
    jobs = {
        f"job-{i}": {"job_id": f"job-{i}", "stage": "queued", "custodian": None}
        for i in range(1, 4)
    }
    state = {
        "tick": 0,
        "task": "dispatch-routing",
        "success": False,
        "jobs": jobs,
        "robots": [{"id": r["robot_id"], "active_command": None} for r in ROBOTS],
    }
    result.create(
        "semantic-test-" + uuid4().hex,
        "worker-test",
        ROBOTS,
        list(jobs.values()),
        ["receive"],
        200,
        state,
    )
    return result


def packet(backend, tick=1, jid="job-1", mode="embeddings", **change):
    return {
        "schema": SCHEMA,
        "space_id": "factory-route-v1",
        "checkpoint_sha256": SHA,
        "mode": mode,
        "run_id": backend.snapshot()["run_id"],
        "job_id": jid,
        "sender": "humanoid-01",
        "recipient": "gantry-01",
        "topic": "dispatch_route",
        "packet_id": f"humanoid-01-{jid}-{tick}",
        "emitted_tick": tick,
        "expires_tick": tick + 15,
        "observation_tick": tick - 1,
        "source_command_id": f"{jid}-queued",
        "source_frame_sha256": "b" * 64,
        **({"vector": [1.0] + [0.0] * 31} if mode == "embeddings" else {"text": '{"route":"A"}'}),
        **change,
    }


def advance(backend, packets=None, jid=None, *, correct=True, mutate=None):
    before = backend.snapshot()
    state = copy.deepcopy(before["state"])
    state["tick"] += 1
    commands, outcomes = [], []
    if jid:
        job = state["jobs"][jid]
        stage = job["stage"]
        next_stage = STAGES[STAGES.index(stage) + 1]
        cid = f"{jid}-{stage}"
        commands = [
            {
                "command_id": cid,
                "robot_id": "humanoid-01",
                "job_id": jid,
                "capability": STAGE_CAPABILITIES[STAGES.index(stage)],
                "resources": ["receive"],
                "lease_ticks": 2,
                "payload": {"expected_stage": stage, "next_stage": next_stage},
            }
        ]
        job["stage"] = next_stage
        observed = {"stage": next_stage, "custodian": None}
        if next_stage == "completed":
            job["routing_correct"] = observed["routing_correct"] = correct
        outcomes = [
            {
                "command_id": cid,
                "robot_id": "humanoid-01",
                "status": "completed",
                "observed": observed,
            }
        ]
    state["success"] = all(
        j["stage"] == "completed" and j["routing_correct"] for j in state["jobs"].values()
    )
    if mutate:
        mutate(state, outcomes)
    backend.prepare(before["revision"], commands)
    return backend.commit(before["revision"], state, outcomes, semantic_packets=packets or [])


@pytest.mark.parametrize("mode", ["embeddings", "text"])
def test_two_job_beliefs_survive_latest_peer_updates_without_payload_mixing(backend, mode):
    backend.configure_semantic_channel(mode, SHA)
    a, b = packet(backend, mode=mode), packet(backend, 2, "job-2", mode)
    advance(backend, [a], "job-1")
    advance(backend, [b], "job-2")
    backend.refresh_semantic_packets(2)
    for jid, source in (("job-1", a), ("job-2", b)):
        result = backend.retrieve_semantic_packet("gantry-01", jid, 2)
        assert result["packet_id"] == source["packet_id"]
        assert ("vector" in result) == (mode == "embeddings")
        assert ("text" in result) == (mode == "text")
    assert backend.semantic_metrics["cached_records"] == 2
    assert backend.semantic_metrics["record_limit"] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "text"},
        {"space_id": "other"},
        {"checkpoint_sha256": "c" * 64},
        {"run_id": "other-run"},
        {"job_id": "job-2"},
        {"sender": "gantry-01"},
        {"recipient": "humanoid-01"},
        {"observation_tick": 1},
        {"observation_tick": -1},
        {"emitted_tick": 2},
        {"expires_tick": 18},
        {"source_command_id": "missing"},
        {"source_frame_sha256": "bad"},
        {"vector": [0.0] * 32},
        {"vector": [float("nan")] + [0.0] * 31},
        {"text": '{"route":"A"}'},
        {"private_route": "A"},
    ],
)
def test_invalid_packet_rolls_back_state_and_channel_atomically(backend, change):
    backend.configure_semantic_channel("embeddings", SHA)
    with pytest.raises((SpacetimeError, ValueError)):
        advance(backend, [packet(backend, **change)], "job-1")
    assert backend.snapshot()["tick"] == 0
    backend.refresh_semantic_packets(0)
    assert backend.semantic_metrics["cached_records"] == 0


def test_retransmission_preserves_old_observation_and_expires_at_same_delivery_age(backend):
    backend.configure_semantic_channel("embeddings", SHA)
    first = packet(backend)
    advance(backend, [first], "job-1")
    for _ in range(3):
        advance(backend)
    refreshed = {**first, "packet_id": "humanoid-01-job-1-5", "emitted_tick": 5, "expires_tick": 20}
    advance(backend, [refreshed])
    backend.refresh_semantic_packets(5)
    row = backend.retrieve_semantic_packet("gantry-01", "job-1", 5)
    assert row["observation_age_ticks"] == 5 and row["packet_age_ticks"] == 0
    for _ in range(15):
        advance(backend)
    backend.refresh_semantic_packets(20)
    assert backend.retrieve_semantic_packet("gantry-01", "job-1", 20) is None
    assert (
        backend.semantic_metrics["cached_records"] == 1
    )  # origin retained, never eligible when expired
    renewed = {**first, "packet_id": "humanoid-01-job-1-25", "emitted_tick": 25, "expires_tick": 40}
    for _ in range(4):
        advance(backend)
    advance(backend, [renewed])
    backend.refresh_semantic_packets(25)
    assert backend.retrieve_semantic_packet("gantry-01", "job-1", 25)["observation_age_ticks"] == 25


@pytest.mark.parametrize("mutation", ["payload", "observation", "frame", "cadence"])
def test_retransmission_cannot_claim_new_evidence_or_change_learned_payload(backend, mutation):
    backend.configure_semantic_channel("embeddings", SHA)
    first = packet(backend)
    advance(backend, [first], "job-1")
    target = 2 if mutation == "cadence" else 5
    for _ in range(target - 2):
        advance(backend)
    row = {
        **first,
        "packet_id": f"humanoid-01-job-1-{target}",
        "emitted_tick": target,
        "expires_tick": target + 15,
    }
    if mutation == "payload":
        row["vector"] = [-1.0] + [0.0] * 31
    if mutation == "observation":
        row["observation_tick"] = target - 1
    if mutation == "frame":
        row["source_frame_sha256"] = "c" * 64
    with pytest.raises(SpacetimeError):
        advance(backend, [row])
    assert backend.snapshot()["tick"] == target - 1


def test_none_mode_has_no_packets_and_mode_and_access_are_immutable(backend):
    backend.configure_semantic_channel("none", SHA)
    with pytest.raises(SpacetimeError):
        backend.configure_semantic_channel("text", SHA)
    with pytest.raises(SpacetimeError):
        backend.retrieve_semantic_packet("humanoid-01", "job-1", 0)
    with pytest.raises(SpacetimeError):
        backend.retrieve_semantic_packet("gantry-01", "job-9", 0)
    with pytest.raises(SpacetimeError):
        backend.retrieve_semantic_packet("gantry-01", "job-1", 1)
    assert backend.retrieve_semantic_packet("gantry-01", "job-1", 0) is None
    with pytest.raises(SpacetimeError):
        advance(backend, [packet(backend)], "job-1")


def test_completed_bad_route_remains_physical_completion_and_rejects_only_final_job(backend):
    backend.configure_semantic_channel("text", SHA)
    advance(backend, [packet(backend, mode="text")], "job-1")
    for jid in ("job-1", "job-2", "job-3"):
        while backend.snapshot()["state"]["jobs"][jid]["stage"] != "completed":
            result = advance(backend, jid=jid, correct=jid != "job-1")
        if jid != "job-3":
            assert result["status"] == "running"
    assert result["status"] == "rejected" and result["phase"] == "terminal"
    assert all(job["stage"] == "completed" for job in result["state"]["jobs"].values())
    assert result["state"]["success"] is False
    backend.refresh_semantic_packets(result["tick"])
    assert backend.semantic_metrics["cached_records"] == 0


def test_correct_routes_succeed_and_task_cannot_be_swapped(backend):
    backend.configure_semantic_channel("none", SHA)
    with pytest.raises(SpacetimeError):
        advance(backend, mutate=lambda state, outcomes: state.update(task="standard"))
    # Retry the still-authorized empty batch with its original task.
    advance(backend)
    for jid in ("job-1", "job-2", "job-3"):
        while backend.snapshot()["state"]["jobs"][jid]["stage"] != "completed":
            result = advance(backend, jid=jid)
    assert result["status"] == "succeeded"


def test_readback_failure_does_not_create_unconfirmed_local_delivery(monkeypatch):
    backend = SpacetimeCoordinator(SpacetimeConfig(token="test-token"))
    backend.run_id, backend._committed_tick = "run-test", 1
    backend.semantic_index = SemanticIndex("run-test", ["job-1"], channel_config("text", SHA))

    def unavailable(*args, **kwargs):
        raise SpacetimeUnavailable("offline")

    monkeypatch.setattr(backend.client, "_sql", unavailable)
    with pytest.raises(SpacetimeUnavailable):
        backend.refresh_semantic_packets(1)
    assert backend.semantic_metrics["refreshes"] == 0


@pytest.mark.parametrize("failure", ["prepare", "commit"])
def test_runtime_does_not_observe_before_authorization_or_deliver_uncommitted_cue(
    backend, tmp_path, monkeypatch, failure
):
    """Use a minimal observed-step fixture; no physical task success is claimed."""
    import numpy as np

    from relay_lab.factory.runtime import FactoryRun
    from relay_lab.factory.world import FactoryWorld

    live = isinstance(backend, SpacetimeCoordinator)
    coordinator = SpacetimeCoordinator(backend.client.config) if live else MemoryCoordinator()

    class CueCodec:
        @property
        def model_info(self):
            return {"space_id": "factory-route-v1", "checkpoint_sha256": SHA}

        def encode(self, rgb):
            return {"vector": [1.0] + [0.0] * 31, **self.model_info}

    class ObservedStep(FactoryWorld):
        def __init__(self):
            super().__init__(task="dispatch-routing", batch_size=1)
            self.observations = self.steps = 0

        def candidates(self):
            return [
                {
                    "candidate_id": "cue",
                    "robot_id": "humanoid-01",
                    "job_id": "job-001",
                    "capability": "transfer",
                    "kind": "transfer",
                    "cargo_id": "part-001",
                    "source": "receive",
                    "target": "amr-01",
                    "expected_stage": "queued",
                    "next_stage": "loaded",
                    "resources": ["receive", "amr-01", "part-001"],
                    "description": "Synthetic authorized observation fixture",
                }
            ]

        def inspect_private_cue(self, robot, job):
            assert coordinator.snapshot()["phase"] == "executing"
            self.observations += 1
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def step(self, commands=None):
            self.steps += 1
            self.tick += 1
            return {
                "snapshot": self.snapshot(),
                "events": [],
                "motor_actions": [],
                "outcomes": [
                    {
                        "command_id": commands[0]["command_id"],
                        "robot_id": "humanoid-01",
                        "status": "running",
                        "detail": "fixture observation",
                        "observed": {"stage": "queued", "custodian": None},
                    }
                ],
            }

    world = ObservedStep()
    run = FactoryRun(
        task="dispatch-routing",
        orders=1,
        world=world,
        coordinator=coordinator,
        routing_codec=CueCodec(),
        planner="rules",
        output=tmp_path / "run",
    )

    def rejected(*args, **kwargs):
        raise SpacetimeUnavailable("Injected unconfirmed operation")

    monkeypatch.setattr(coordinator, failure, rejected)
    try:
        with pytest.raises(SpacetimeUnavailable):
            run.tick()
        assert world.observations == world.steps == int(failure == "commit")
        assert run.status == "failed" and coordinator.snapshot()["tick"] == 0
        assert run.routing_exchange.published == 0
        assert coordinator.semantic_metrics["refreshes"] == 0
        coordinator.refresh_semantic_packets(0)
        assert coordinator.retrieve_semantic_packet("gantry-01", "job-001", 0) is None
        with pytest.raises(RuntimeError, match="stopped"):
            run.tick()
        assert world.steps == int(failure == "commit")
    finally:
        run.close()
