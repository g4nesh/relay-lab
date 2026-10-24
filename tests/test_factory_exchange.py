"""Latents only become peer inputs after a confirmed database round trip."""

from copy import deepcopy

import pytest

from relay_lab.factory.exchange import LatentExchange

HASH = "a" * 64
VECTOR = [1.0] + [0.0] * 31


class World:
    tick = 1
    deliveries = None

    def embedding_observations(self):
        return [
            {
                "robot_id": "amr-01",
                "tick": self.tick,
                "vector": VECTOR,
                "space_id": "factory-vla-v1",
                "checkpoint_sha256": HASH,
                "decoder": {"text": "carrier ready", "confidence": 0.9, "version": "v1"},
            }
        ]

    def set_peer_embeddings(self, records):
        self.deliveries = deepcopy(records)


class Backend:
    def __init__(self):
        self.refreshed = []
        self.retrieved = []
        self.config = None
        self.embedding_metrics = {}

    def configure_embedding_space(self, *config):
        self.config = config

    def refresh_embeddings(self, tick):
        self.refreshed.append(tick)

    def retrieve_embeddings(self, robot, vector, tick, limit):
        self.retrieved.append((robot, vector, tick, limit))
        return []


def test_latents_are_not_delivered_before_confirmation():
    world, backend = World(), Backend()
    exchange = LatentExchange(world, backend, "run", HASH)
    exchange.initialize()
    records = exchange.collect({"tick": 1})
    exchange.deliver(1)
    assert not backend.retrieved
    assert not exchange.latest
    exchange.confirmed(records, 1)
    exchange.deliver(2)
    assert backend.refreshed == [1]
    assert len(backend.retrieved) == 1
    exchange.deliver(16)
    assert len(backend.retrieved) == 1  # expired query and peer inputs cannot persist
    assert world.deliveries == {}


def test_stale_observations_cannot_be_relabeled_as_fresh():
    world, backend = World(), Backend()
    exchange = LatentExchange(world, backend, "run", HASH)
    world.tick = 0
    with pytest.raises(ValueError, match="physical tick"):
        exchange.collect({"tick": 1})


def test_database_refresh_failure_never_uses_local_vectors():
    class Offline(Backend):
        def refresh_embeddings(self, tick):
            raise RuntimeError("database disconnected")

    world, backend = World(), Offline()
    exchange = LatentExchange(world, backend, "run", HASH)
    records = exchange.collect({"tick": 1})
    with pytest.raises(RuntimeError, match="disconnected"):
        exchange.confirmed(records, 1)
    assert not exchange.latest
    assert not exchange.published


def test_explicit_disabled_mode_emits_and_consumes_nothing():
    world, backend = World(), Backend()
    exchange = LatentExchange(world, backend, "run", HASH, enabled=False)
    exchange.initialize()
    assert exchange.collect({"tick": 1}) == []
    assert exchange.deliver(1) == {}
    assert backend.config is None and backend.refreshed == []
    assert not exchange.view()["enabled"]
