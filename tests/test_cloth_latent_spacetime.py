"""Raw reducer checks for the learned-latent protocol; no Python channel validation.

Opt in after publishing the updated local module with
RELAY_SPACETIME_INTEGRATION=1. Unique run IDs preserve all previous evidence.
"""

import copy
import json
import os
from uuid import uuid4

import pytest

from relay_lab.cloth.communication import FEATURES, SAFETY_FIELDS, canonical
from relay_lab.cloth.spacetime import SpacetimeJournal
from relay_lab.spacetime import SpacetimeError


def configuration(mode="embeddings"):
    return {
        "schema": "relay.cloth.learned-latent.v1",
        "mode": mode,
        "model": {
            "kind": "cross-vla-adapter-v1",
            "space_id": "cloth-cross-vla-test-v1",
            "checkpoint_sha256": "a" * 64,
            "dimension": 64,
            "source_dimensions": {"left": 960, "right": 1024},
            "source_extractor_ids": {"left": "b" * 64, "right": "c" * 64},
        },
        "features": list(FEATURES),
        "hybrid_fields": list(SAFETY_FIELDS),
        "network": {
            "seed": 7,
            "min_delay_ticks": 1,
            "max_delay_ticks": 4,
            "drop_probability": 0.1,
            "ttl_ticks": 8,
        },
    }


def packet(run_id, config, sender="left", tick=0):
    mode = config["mode"]
    content = {}
    if mode in ("embeddings", "hybrid"):
        content["vector"] = [index / 100 for index in range(64)]
    if mode in ("structured_text", "hybrid"):
        fields = FEATURES if mode == "structured_text" else SAFETY_FIELDS
        content["text"] = canonical(dict.fromkeys(fields, 0.0))
    return {
        "schema": config["schema"],
        "run_id": run_id,
        "mode": mode,
        "packet_id": f"{sender}-{tick}",
        "sender": sender,
        "recipient": "right" if sender == "left" else "left",
        "observation_tick": tick,
        "emitted_tick": tick,
        "deliver_tick": tick + 2,
        "expires_tick": tick + 8,
        "dropped": False,
        "source_sha256": "d" * 64,
        "checkpoint_sha256": config["model"]["checkpoint_sha256"],
        "space_id": config["model"]["space_id"],
        "content": content,
    }


@pytest.fixture
def backend():
    if os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1":
        pytest.skip("Set RELAY_SPACETIME_INTEGRATION=1 for live learned-latent reducer tests")
    return SpacetimeJournal()


@pytest.mark.parametrize("mode", ["embeddings", "structured_text", "hybrid", "none"])
def test_live_learned_channel_preserves_exact_valid_payload(backend, mode):
    config = configuration(mode)
    run_id = "latent-check-" + uuid4().hex
    backend.create(run_id, config)
    batch = [] if mode == "none" else [packet(run_id, config)]
    assert backend.commit(0, batch, {"tick": 0}) == batch
    assert backend.commit(0, batch, {"tick": 0}) == batch
    assert backend.history() == [{"tick": 0, "packets": batch, "state": {"tick": 0}}]
    rows = backend.client._sql(f"SELECT * FROM cloth_packet WHERE run_id = '{run_id}'")
    assert len(rows) == len(batch)
    if batch:
        assert json.loads(rows[0]["packet_json"]) == batch[0]
    changed = copy.deepcopy(config)
    changed["model"]["source_extractor_ids"]["left"] = "e" * 64
    with pytest.raises(SpacetimeError):
        backend.create(run_id, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "standardized-pca-v1"),
        ("dimension", 8),
        ("dimension", True),
        ("source_dimensions", {"left": 960, "right": 768}),
        ("source_dimensions", {"left": 960}),
        ("source_dimensions", {"left": 960, "right": 1024, "extra": 1}),
        ("source_extractor_ids", {"left": "b" * 64}),
        ("source_extractor_ids", {"left": "b" * 64, "right": "invalid"}),
        ("source_extractor_ids", {"left": "b" * 64, "right": 123}),
        ("source_extractor_ids", {"left": "b" * 64, "right": "c" * 64, "extra": "d" * 64}),
        ("checkpoint_sha256", "not-a-checkpoint"),
    ],
)
def test_live_learned_configuration_rejects_incompatible_sources(backend, field, value):
    config = configuration()
    config["model"][field] = value
    run_id = "latent-invalid-config-" + uuid4().hex
    with pytest.raises(SpacetimeError):
        backend.create(run_id, config)
    assert backend.client._sql(f"SELECT * FROM cloth_channel WHERE run_id = '{run_id}'") == []


@pytest.mark.parametrize(
    "mode,mutation",
    [
        ("embeddings", {"content": {"vector": [0.0] * 63}}),
        ("embeddings", {"content": {"vector": [1.01] * 64}}),
        ("embeddings", {"content": {"vector": [0.0] * 64, "text": "{}"}}),
        ("structured_text", {"content": {"vector": [0.0] * 64}}),
        ("structured_text", {"content": {"text": canonical(dict.fromkeys(SAFETY_FIELDS, 0.0))}}),
        ("hybrid", {"content": {"vector": [0.0] * 64}}),
        (
            "hybrid",
            {"content": {"vector": [0.0] * 64, "text": canonical(dict.fromkeys(FEATURES, 0.0))}},
        ),
        ("embeddings", {"schema": "relay.cloth.communication.v2"}),
        ("embeddings", {"checkpoint_sha256": "e" * 64}),
        ("embeddings", {"space_id": "other-space"}),
        ("embeddings", {"observation_tick": 1}),
        ("embeddings", {"recipient": "left"}),
        ("embeddings", {"deliver_tick": 0}),
        ("embeddings", {"expires_tick": 99}),
        ("none", {}),
    ],
)
def test_live_invalid_latent_packet_rolls_back_entire_frame(backend, mode, mutation):
    config = configuration(mode)
    run_id = "latent-invalid-packet-" + uuid4().hex
    backend.create(run_id, config)
    invalid = packet(run_id, config)
    invalid.update(mutation)
    # The valid first sender must also roll back if the second packet is invalid.
    packets = [packet(run_id, config, sender="right"), invalid]
    with pytest.raises(SpacetimeError):
        backend.commit(0, packets, {"tick": 0})
    assert backend.history() == []
    assert backend.client._sql(f"SELECT * FROM cloth_packet WHERE run_id = '{run_id}'") == []
    assert backend.commit(0, [], {"tick": 0}) == []
