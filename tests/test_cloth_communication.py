"""Matched transport, causal delivery, immutable provenance and true codec checks."""

import copy
import hashlib
import json
import os
from uuid import uuid4

import numpy as np
import pytest

from relay_lab.cloth.communication import (
    FEATURES,
    SAFETY_FIELDS,
    Channel,
    LearnedFeatureCodec,
    MemoryJournal,
    NetworkConfig,
    canonical,
    migrate_training_features,
    observation_features,
)
from relay_lab.cloth.spacetime import SpacetimeJournal
from relay_lab.spacetime import SpacetimeConfig, SpacetimeError, SpacetimeUnavailable


@pytest.fixture
def codec():
    rng = np.random.default_rng(8123)
    data = rng.normal(size=(128, len(FEATURES))) * 0.1
    return LearnedFeatureCodec.fit(
        [dict(zip(FEATURES, row.tolist())) for row in data],
        dimensions=8,
        training={"split": "test-fixture"},
    )


def features(value=0.1):
    return {key: value + index * 0.01 for index, key in enumerate(FEATURES)}


def channel(codec, mode="structured_text", **network):
    return Channel(
        "cloth-test",
        mode=mode,
        codec=codec,
        network=NetworkConfig(min_delay_ticks=1, max_delay_ticks=1, drop_probability=0, **network),
    )


def test_local_features_do_not_include_remote_world_truth():
    local = {
        "ee_position": [0.1, 0.2, 0.3],
        "grasped": True,
        "grasp_target": "hem",
        "landmarks": {key: [0.4, 0.5, 0.6] for key in ("hem", "sleeve", "shoulder")},
        "obstacle": None,
        "private_evaluator": {"future_obstacle": [100, 100, 100]},
    }
    extracted = observation_features(local)
    assert set(extracted) == set(FEATURES)
    assert extracted["grasped"] == 1
    assert extracted["grasp_hem"] == 1 and extracted["grasp_sleeve"] == 0
    assert extracted["obstacle_active"] == 0
    assert 100 not in extracted.values()
    del local["landmarks"]["hem"]
    masked = observation_features(local)
    assert masked["hem_visible"] == 0 and masked["hem_x"] == 0
    assert masked["sleeve_visible"] == 1


def test_attachment_identity_cannot_be_guessed_from_a_boolean():
    observation = {"ee_position": [0, 0, 0.2], "grasped": True, "landmarks": {}, "obstacle": None}
    with pytest.raises(ValueError, match="Measured grasp target"):
        observation_features(observation)
    sleeve = observation_features({**observation, "grasp_target": "sleeve"})
    hem = observation_features({**observation, "grasp_target": "hem"})
    assert sleeve["grasp_sleeve"] == 1 and sleeve["grasp_hem"] == 0
    assert hem["grasp_hem"] == 1 and hem["grasp_sleeve"] == 0
    with pytest.raises(ValueError):
        observation_features({**observation, "grasp_target": "none"})


def test_training_migration_drops_ambiguous_positive_grasps():
    old = {key: 0.0 for key in FEATURES if key not in {"grasp_sleeve", "grasp_hem"}}
    records, report = migrate_training_features([old, {**old, "grasped": 1.0}])
    assert len(records) == 1 and records[0]["grasp_hem"] == records[0]["grasp_sleeve"] == 0
    assert report["discarded_unknown_attachment"] == 1
    assert report["inferred_targets_from_phase"] is False
    assert "grasp_hem" not in old


def test_codec_is_trained_reduced_and_checkpoint_identified(codec, tmp_path):
    original = features()
    latent = codec.encode(original)
    assert len(latent) == 8 < len(original)
    assert codec.decode(latent) != original
    path = tmp_path / "codec.json"
    codec.save(path)
    restored = LearnedFeatureCodec.load(path)
    assert restored.sha256 == codec.sha256
    assert restored.encode(original) == latent
    assert restored.model_info["kind"] == "standardized-pca-v1"
    changed = copy.deepcopy(codec.checkpoint)
    changed["mean"][0] += 0.01
    assert LearnedFeatureCodec(changed).sha256 != codec.sha256


def test_codec_rejects_untrained_and_corrupt_inputs(codec):
    with pytest.raises(ValueError):
        LearnedFeatureCodec.fit([features()] * 3)
    with pytest.raises(ValueError):
        codec.encode({**features(), "ee_x": float("nan")})
    with pytest.raises(ValueError):
        codec.decode([float("nan")] * 8)
    with pytest.raises(ValueError):
        codec.decode([0] * 9)
    corrupt = copy.deepcopy(codec.checkpoint)
    corrupt["components"][0][0] = 10
    with pytest.raises(ValueError):
        LearnedFeatureCodec(corrupt)


@pytest.mark.parametrize("mode", ["embeddings", "structured_text", "hybrid"])
def test_delayed_causal_receive_requires_committed_batch(codec, mode):
    c = channel(codec, mode)
    packet = c.publish("left", {"features": features(), "source_sha256": "b" * 64}, 0)
    c.advance(0)
    assert c.receive("right", 0) is None
    assert c.receive("left", 0) is None
    c.advance(1)
    belief = c.receive("right", 1)
    assert belief["packet_id"] == packet["packet_id"]
    assert belief["age_ticks"] == 1 and belief["observation_tick"] == 0
    assert belief["source_sha256"] == "b" * 64
    assert c.counter["delivered"] == 1
    assert c.metrics["network_delay_ticks"]["mean"] == 1


def test_structured_text_is_full_precision_and_hybrid_preserves_exact_safety(codec):
    original = features()
    results = {}
    for mode in ("embeddings", "structured_text", "hybrid"):
        c = channel(codec, mode)
        c.publish("left", {"features": original}, 0)
        c.advance(0)
        c.advance(1)
        results[mode] = c.receive("right", 1)["features"]
    assert results["structured_text"] == original
    assert results["embeddings"] != original
    assert {key: results["hybrid"][key] for key in SAFETY_FIELDS} == {
        key: original[key] for key in SAFETY_FIELDS
    }
    geometry = set(FEATURES) - set(SAFETY_FIELDS)
    assert all(results["hybrid"][key] == results["embeddings"][key] for key in geometry)


def test_modes_have_identical_seeded_transport_conditions(codec):
    schedule = {}
    for mode in ("embeddings", "structured_text", "hybrid"):
        c = Channel(
            "run-" + mode,
            mode=mode,
            codec=codec,
            network=NetworkConfig(seed=71, max_delay_ticks=17, drop_probability=0.3),
        )
        records = []
        for tick in range(30):
            for sender in ("left", "right"):
                packet = c.publish(sender, features(tick / 100), tick)
                records.append((sender, tick, packet["deliver_tick"], packet["dropped"]))
            c.advance(tick)
        schedule[mode] = records
    assert schedule["embeddings"] == schedule["structured_text"] == schedule["hybrid"]
    assert any(row[-1] for row in schedule["hybrid"])
    assert len({row[2] - row[1] for row in schedule["hybrid"]}) > 1


def test_later_arriving_old_fact_never_overwrites_new_fact(codec):
    network = NetworkConfig(seed=2, max_delay_ticks=20, drop_probability=0, ttl_ticks=50)
    pair = next(
        (a, b)
        for a in range(20)
        for b in range(a + 1, 21)
        if network.schedule("left", a)[0] > network.schedule("left", b)[0]
    )
    c = Channel("reordered", mode="structured_text", codec=codec, network=network)
    a, b = pair
    for tick in range(network.schedule("left", a)[0] + 1):
        if tick in pair:
            c.publish("left", features(0.2 if tick == b else 0.1), tick)
        c.advance(tick)
    assert c.receive("right", c.tick)["observation_tick"] == b
    assert c.counter["out_of_order"] == 1


def test_expiry_measured_from_observation_not_reception(codec):
    c = channel(codec, ttl_ticks=2)
    c.publish("left", features(), 0)
    c.advance(0)
    c.advance(1)
    assert c.receive("right", 1)
    c.advance(2)
    assert c.receive("right", 2) is None
    assert c.counter["stale_reads"] == 1


def test_packet_already_expired_in_transit_is_never_delivered(codec):
    c = Channel(
        "expired",
        mode="structured_text",
        codec=codec,
        network=NetworkConfig(
            min_delay_ticks=3, max_delay_ticks=3, ttl_ticks=2, drop_probability=0
        ),
    )
    c.publish("left", features(), 0)
    c.advance(0)
    c.advance(3)
    assert c.receive("right", 3) is None
    assert c.counter["expired"] == 1


def test_none_keeps_local_perception_but_shares_no_evidence(codec):
    c = channel(codec, "none")
    assert c.publish("left", features(), 0) is None
    c.advance(0, state={"tick": 0, "public_phase": "lift"})
    c.advance(1)
    assert c.receive("right", 1) is None
    assert c.counter["local_observations"] == 1 and len(c.encode_ms) == 1
    assert c.counter["packet_bytes"] == 0
    assert json.loads(c.backend.frames[0])["packets"] == []


def test_unconfirmed_database_does_not_deliver_or_advance_clock(codec):
    class Unavailable(MemoryJournal):
        def commit(self, *args, **kwargs):
            raise SpacetimeUnavailable("unconfirmed")

    c = Channel("unconfirmed", mode="structured_text", codec=codec, backend=Unavailable())
    c.publish("left", features(), 0)
    with pytest.raises(SpacetimeUnavailable):
        c.advance(0)
    assert c.tick == -1 and not c.queue and not c.latest and len(c.pending) == 1


def test_database_cannot_inject_different_readback(codec):
    class Corrupt(MemoryJournal):
        def commit(self, tick, packets, state=None):
            returned = super().commit(tick, packets, state)
            returned[0]["source_sha256"] = "c" * 64
            return returned

    c = Channel("corrupt", mode="structured_text", codec=codec, backend=Corrupt())
    c.publish("left", features(), 0)
    with pytest.raises(ValueError, match="readback"):
        c.advance(0)
    assert not c.latest


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "other"},
        {"recipient": "left"},
        {"sender": "third"},
        {"observation_tick": -1},
        {"emitted_tick": 9},
        {"deliver_tick": 0},
        {"expires_tick": 1000},
        {"dropped": True},
        {"checkpoint_sha256": "e" * 64},
        {"source_sha256": "invalid"},
        {"mode": "embeddings"},
        {"hidden_ground_truth": 1},
        {"content": {"vector": [0] * 8, "text": "{}"}},
    ],
)
def test_packets_reject_scope_causal_and_representation_tampering(codec, change):
    c = channel(codec)
    packet = c.publish("left", features(), 0)
    with pytest.raises(ValueError):
        c.validate({**packet, **change})


def test_source_mutation_does_not_change_pending_or_delivered_packet(codec):
    c = channel(codec)
    original = features()
    packet = c.publish("left", original, 0)
    packet["content"]["text"] = "{}"
    original["ee_x"] = 90
    c.advance(0)
    c.advance(1)
    belief = c.receive("right", 1)
    assert belief["features"] == features()
    belief["features"]["ee_x"] = 80
    assert c.receive("right", 1)["features"] == features()


def test_duplicate_and_future_reads_rejected(codec):
    c = channel(codec)
    c.publish("left", features(), 0)
    with pytest.raises(ValueError):
        c.publish("left", features(), 0)
    with pytest.raises(ValueError):
        c.advance(1)
    c.advance(0)
    with pytest.raises(ValueError):
        c.receive("right", 1)
    with pytest.raises(ValueError):
        c.receive("third", 0)


def test_json_byte_metrics_include_actual_transmitted_content(codec):
    c = channel(codec)
    packet = c.publish("left", features(), 0)
    assert c.metrics["payload_bytes"] == len(canonical(packet["content"]).encode())
    assert c.metrics["packet_bytes"] == len(canonical(packet).encode())
    assert packet["source_sha256"] == hashlib.sha256(canonical(features()).encode()).hexdigest()


def test_spacetime_adapter_uses_confirmed_authoritative_batch(monkeypatch, codec):
    backend = SpacetimeJournal(SpacetimeConfig(token="test-only-token"))
    calls = []

    def mutate(reducer, args, readback, expected):
        calls.append((reducer, args))
        return dict(expected)

    monkeypatch.setattr(backend.client, "_mutate", mutate)
    c = Channel("durable-test", mode="structured_text", codec=codec, backend=backend)
    c.publish("left", features(), 0)
    c.advance(0)
    assert [call[0] for call in calls] == ["create_cloth_channel", "commit_cloth_frame"]
    assert backend.backend_info["durable"]


def test_memory_journal_idempotence_and_conflict():
    backend = MemoryJournal()
    backend.create("test", {})
    assert backend.commit(0, []) == []
    assert backend.commit(0, []) == []
    with pytest.raises(ValueError):
        backend.commit(0, [], {"changed": True})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_delay_ticks": 0},
        {"drop_probability": float("nan")},
        {"max_delay_ticks": 0},
        {"ttl_ticks": True},
        {"seed": -1},
    ],
)
def test_invalid_network_configuration(kwargs):
    with pytest.raises(ValueError):
        NetworkConfig(**kwargs)


def test_codec_evaluation_cannot_mutate_trained_checkpoint(codec):
    before = codec.sha256
    report = codec.evaluate([features(0.1), features(0.2)])
    assert report["samples"] == 2
    assert report["max_absolute_error"] > 0
    assert set(report["rmse_by_feature"]) == set(FEATURES)
    assert codec.sha256 == before


@pytest.mark.parametrize("mode", ["embeddings", "structured_text", "hybrid", "none"])
def test_live_spacetime_private_packets_and_transaction_conflicts(codec, mode):
    if os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1":
        pytest.skip("Set RELAY_SPACETIME_INTEGRATION=1 for live cloth database tests")
    backend = SpacetimeJournal()
    c = Channel(
        "cloth-check-" + uuid4().hex,
        mode=mode,
        codec=codec,
        backend=backend,
        network=NetworkConfig(min_delay_ticks=1, max_delay_ticks=1, drop_probability=0),
    )
    c.publish("left", features(), 0)
    c.advance(0, state={"tick": 0, "phase": "gather"})
    assert c.receive("right", 0) is None
    c.advance(1)
    assert (c.receive("right", 1) is not None) == (mode != "none")
    assert backend.history()[0]["state"]["phase"] == "gather"
    with pytest.raises(SpacetimeError):
        backend.commit(0, [], {"tick": 0, "phase": "different"})
    with pytest.raises(SpacetimeError):
        backend.commit(2, [], {"tick": 3})
    assert len(backend.history()) == 1
    rows = backend.client._sql(f"SELECT * FROM cloth_packet WHERE run_id = '{backend.run_id}'")
    assert len(rows) == int(mode != "none")
    if rows:
        assert (
            json.loads(rows[0]["packet_json"])["source_sha256"]
            == hashlib.sha256(canonical(features()).encode()).hexdigest()
        )
