"""Protocol tests use synthetic unit vectors; they do not claim model competence."""

import copy
import os
from uuid import uuid4

import pytest

from relay_lab.factory.coordination import MemoryCoordinator
from relay_lab.factory.embeddings import EmbeddingError, EmbeddingIndex, space_config
from relay_lab.factory.spacetime import SpacetimeCoordinator
from relay_lab.spacetime import SpacetimeConfig, SpacetimeError

SHA = "a" * 64
SPACE = "factory-vla-v1"
ROBOTS = [{"robot_id": r, "capabilities": ["wait", "transfer"]} for r in ("arm-01", "amr-01")]


@pytest.fixture(params=["memory", "spacetime"])
def backend(request):
    if request.param == "spacetime":
        if os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1":
            pytest.skip("Real embedding database checks are opt-in")
        result = SpacetimeCoordinator(SpacetimeConfig.from_env())
    else:
        result = MemoryCoordinator()
    run_id = "embedding-test-" + uuid4().hex
    state = {
        "tick": 0,
        "success": False,
        "jobs": {"job-1": {"job_id": "job-1", "stage": "queued", "custodian": None}},
        "robots": [{"id": r["robot_id"], "active_command": None} for r in ROBOTS],
    }
    result.create(
        run_id, "worker-test", ROBOTS, list(state["jobs"].values()), ["fixture"], 100, state
    )
    result.configure_embedding_space(SPACE, SHA)
    return result


def record(backend, tick, robot="amr-01", **overrides):
    return {
        "schema": "relay.factory.embedding.v1",
        "embedding_id": f"{robot}-{tick}",
        "run_id": backend.snapshot()["run_id"],
        "robot_id": robot,
        "tick": tick,
        "command_id": None,
        "job_id": None,
        "modality": "vla-hidden",
        "space_id": SPACE,
        "checkpoint_sha256": SHA,
        "dimension": 32,
        "vector": [1.0] + [0.0] * 31,
        "expires_tick": tick + 64,
        "decoder": {"text": "model predicts idle", "confidence": 0.7, "version": "head-v1"},
        **overrides,
    }


def advance(backend, records, *, state=None, commands=None, outcomes=None):
    old = backend.snapshot()
    state = copy.deepcopy(state or old["state"])
    state["tick"] = old["tick"] + 1
    backend.prepare(old["revision"], commands or [])
    return backend.commit(old["revision"], state, outcomes or [], embeddings=records)


@pytest.mark.parametrize(
    "change",
    [
        {"dimension": 31},
        {"vector": [1.0] * 31},
        {"vector": [0.0] * 32},
        {"vector": [float("nan")] + [0.0] * 31},
        {"space_id": "another-space"},
        {"checkpoint_sha256": "b" * 64},
        {"schema": "unknown"},
        {"run_id": "another-run"},
        {"tick": 0},
        {"tick": 2},
        {"expires_tick": 1},
        {"embedding_id": "reused-id"},
        {"decoder": {"text": "x" * 5000, "confidence": 0.7, "version": "v1"}},
        {"command_id": "not-authorized", "job_id": "job-1"},
    ],
)
def test_bad_embedding_aborts_observed_commit_atomically(backend, change):
    item = {**record(backend, 1), **change}
    with pytest.raises((SpacetimeError, ValueError)):
        advance(backend, [item])
    assert backend.snapshot()["revision"] == 0
    assert backend.snapshot()["phase"] == "executing"


def test_latest_per_peer_precedes_cosine_and_space_is_immutable(backend):
    advance(backend, [record(backend, 1)])
    advance(backend, [record(backend, 2, vector=[-1.0] + [0.0] * 31)])
    backend.refresh_embeddings(2)
    matches = backend.retrieve_embeddings("arm-01", [1.0] + [0.0] * 31, 2)
    assert len(matches) == 1 and matches[0]["tick"] == 2
    assert matches[0]["cosine_similarity"] == -1
    with pytest.raises(SpacetimeError):
        backend.configure_embedding_space(SPACE, "b" * 64)
    assert backend.embedding_metrics["client_cosine"]["p50_ms"] >= 0
    assert backend.retrieve_embeddings("arm-01", [1.0] + [0.0] * 31, 2, relevant_peers=[]) == []


def test_bounded_pruning_and_expired_cache_is_never_consumed(backend):
    for tick in range(1, 21):
        advance(backend, [record(backend, tick, robot=r["robot_id"]) for r in ROBOTS])
    backend.refresh_embeddings(20)
    assert backend.embedding_metrics["cached_records"] == 32
    if isinstance(backend, SpacetimeCoordinator):
        before = backend.client.body_bytes
        backend.refresh_embeddings(20)
        assert backend.client.body_bytes == before  # Same revision reuses exact cache.
        assert backend.embedding_metrics["transport_bytes"] > 0
    assert len(backend.retrieve_embeddings("arm-01", [1.0] + [0.0] * 31, 20)) == 1
    # A stale cached frame must expire even when no network refresh occurs.
    index = backend.embedding_index
    assert index.retrieve("arm-01", [1.0] + [0.0] * 31, 85) == []
    assert index.metrics["stale_drops"] > 0


def test_duplicate_batch_is_idempotent_but_changed_vector_rejected(backend):
    item = record(backend, 1)
    advance(backend, [item])
    state = backend.snapshot()["state"]
    backend.commit(0, state, [], embeddings=[item])
    changed = {**item, "vector": [-1.0] + [0.0] * 31}
    with pytest.raises(SpacetimeError):
        backend.commit(0, state, [], embeddings=[changed])


def test_idle_embedding_cannot_hide_an_active_command(backend):
    state = backend.snapshot()["state"]
    state["robots"][1]["active_command"] = "active-command"
    with pytest.raises(SpacetimeError):
        advance(backend, [record(backend, 1)], state=state)


def test_queries_reject_invalid_vectors_and_future_time(backend):
    with pytest.raises(SpacetimeError):
        backend.retrieve_embeddings("arm-01", [0.0] * 32, 0)
    with pytest.raises(SpacetimeError):
        backend.retrieve_embeddings("arm-01", [1.0] + [0.0] * 31, 1)


def test_index_rejects_mismatched_records_on_readback():
    index = EmbeddingIndex("run-test", {"arm-01"}, space_config(SPACE, SHA))
    with pytest.raises(EmbeddingError):
        index.refresh([{"schema": "wrong"}], 1)


def test_expired_database_records_are_pruned_on_commit(backend):
    advance(backend, [record(backend, 1, expires_tick=2)])
    advance(backend, [])
    backend.refresh_embeddings(2)
    assert backend.embedding_metrics["cached_records"] == 0
    assert backend.retrieve_embeddings("arm-01", [1.0] + [0.0] * 31, 2) == []


def test_rejected_second_packet_cannot_leak_first_packet(backend):
    good = record(backend, 1, robot="arm-01")
    bad = record(backend, 1, robot="amr-01", vector=[0.0] * 32)
    with pytest.raises(SpacetimeError):
        advance(backend, [good, bad])
    backend.refresh_embeddings(0)
    assert backend.embedding_metrics["cached_records"] == 0
    assert backend.snapshot()["revision"] == 0


@pytest.mark.parametrize("packet_robot,packet_job", [("amr-01", "job-1"), ("arm-01", "wrong-job")])
def test_existing_command_does_not_authorize_other_robot_or_job_packet(
    backend, packet_robot, packet_job
):
    cmd = {
        "command_id": "cmd-1",
        "robot_id": "arm-01",
        "capability": "wait",
        "job_id": "job-1",
        "resources": ["fixture"],
        "lease_ticks": 2,
        "payload": {},
    }
    state = backend.snapshot()["state"]
    state["robots"][0]["active_command"] = "cmd-1"
    packet = record(backend, 1, robot=packet_robot, command_id="cmd-1", job_id=packet_job)
    outcomes = [
        {
            "command_id": "cmd-1",
            "robot_id": "arm-01",
            "status": "running",
            "observed": {"stage": "queued", "custodian": None},
        }
    ]
    with pytest.raises(SpacetimeError):
        advance(backend, [packet], state=state, commands=[cmd], outcomes=outcomes)
    assert backend.snapshot()["revision"] == 0


def test_network_refresh_failure_is_propagated_without_cache_fallback(monkeypatch):
    from relay_lab.spacetime import SpacetimeUnavailable

    backend = SpacetimeCoordinator(SpacetimeConfig(token="test-token"))
    backend.run_id, backend.worker_id = "run-test", "worker-test"
    backend._committed_tick = 1
    backend.embedding_index = EmbeddingIndex(
        "run-test", {"arm-01", "amr-01"}, space_config(SPACE, SHA)
    )

    def offline(*args, **kwargs):
        raise SpacetimeUnavailable("database unavailable")

    monkeypatch.setattr(backend.client, "_sql", offline)
    with pytest.raises(SpacetimeUnavailable):
        backend.refresh_embeddings(1)
    assert backend.embedding_metrics["refreshes"] == 0
