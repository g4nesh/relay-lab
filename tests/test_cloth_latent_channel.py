"""Learned-message transport preserves causal scope and native feature identity.

The tiny trained fixtures use synthetic features exclusively to test protocol
mechanics. They are not foundation inference or cloth performance evidence.
"""

import copy
import os
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch

from relay_lab.cloth.communication import FEATURES, SAFETY_FIELDS, MemoryJournal, NetworkConfig
from relay_lab.cloth.latent_adapter import (
    FORMAT,
    LOCAL_DIM,
    WIDTHS,
    DualLatentAdapter,
    LearnedLatentPolicy,
)
from relay_lab.cloth.latent_channel import LatentChannel, feature_source_hash
from relay_lab.cloth.spacetime import SpacetimeJournal
from relay_lab.spacetime import SpacetimeUnavailable

EXTRACTORS = {"left": "a" * 64, "right": "b" * 64}
NATIVE = {arm: np.linspace(-0.5, 0.5, width, dtype=np.float32) for arm, width in WIDTHS.items()}
FACTS = {key: float(index) / 100 for index, key in enumerate(FEATURES)}


@pytest.fixture
def policy_factory(tmp_path):
    torch.set_num_threads(3)

    def make(mode="embeddings"):
        torch.manual_seed(33)
        model = DualLatentAdapter(mode)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        features = {arm: torch.from_numpy(value)[None] for arm, value in NATIVE.items()}
        outputs = model(
            features,
            {arm: torch.zeros(1, LOCAL_DIM) for arm in WIDTHS},
            {arm: torch.zeros(1, len(FEATURES)) for arm in WIDTHS},
            torch.ones(1),
            torch.zeros(1),
        )
        sum(value.square().mean() for value in outputs.values()).backward()
        optimizer.step()
        path = tmp_path / f"{mode}.pt"
        torch.save(
            {
                "format": FORMAT,
                "mode": mode,
                "widths": WIDTHS,
                "state_dict": model.state_dict(),
                "training": {"kind": "synthetic protocol-test fixture", "steps": 1},
                "extractor_ids": EXTRACTORS,
            },
            path,
        )
        return LearnedLatentPolicy(path)

    return make


def channel(policy, *, network=None, backend=None):
    return LatentChannel(
        "latent-" + uuid4().hex,
        policy=policy,
        network=network or NetworkConfig(min_delay_ticks=1, max_delay_ticks=1, drop_probability=0),
        backend=backend,
    )


def publish(c, sender="left", tick=0):
    return c.publish(
        sender,
        NATIVE[sender],
        FACTS,
        tick,
        extractor_id=EXTRACTORS[sender],
        source_sha256="c" * 64,
    )


@pytest.mark.parametrize("mode", ["embeddings", "structured_text", "hybrid"])
def test_delivery_passes_learned_vector_without_pca_reconstruction(policy_factory, mode):
    policy = policy_factory(mode)
    c = channel(policy)
    sent = publish(c)
    c.advance(0)
    assert c.receive("right", 0) is None
    c.advance(1)
    belief = c.receive("right", 1)
    assert belief["checkpoint_sha256"] == policy.sha256
    assert belief["age_ticks"] == 1 and belief["observation_tick"] == 0
    if mode == "embeddings":
        assert belief["features"] == {}
    elif mode == "structured_text":
        assert belief["features"] == FACTS
    else:
        assert belief["features"] == {key: FACTS[key] for key in SAFETY_FIELDS}
    if mode == "structured_text":
        assert belief["vector"] == [0.0] * 64
    else:
        assert belief["vector"] == sent["content"]["vector"]
        assert len(belief["vector"]) == 64
        assert np.isfinite(belief["vector"]).all()
    belief["vector"][0] = 99
    assert c.receive("right", 1)["vector"][0] != 99


def test_unconfirmed_commit_cannot_deliver_or_advance(policy_factory):
    class Unconfirmed(MemoryJournal):
        def commit(self, *args, **kwargs):
            raise SpacetimeUnavailable("no durability acknowledgement")

    c = channel(policy_factory(), backend=Unconfirmed())
    publish(c)
    with pytest.raises(SpacetimeUnavailable):
        c.advance(0)
    assert c.tick == -1 and not c.queue and not c.latest
    assert len(c.pending) == 1 and c.counter["delivered"] == 0


def test_ack_readback_cannot_inject_a_different_vector(policy_factory):
    class ChangedReadback(MemoryJournal):
        def commit(self, tick, packets, state=None):
            returned = super().commit(tick, packets, state)
            returned[0]["content"]["vector"][0] += 0.01
            return returned

    c = channel(policy_factory(), backend=ChangedReadback())
    publish(c)
    with pytest.raises(ValueError, match="readback differs"):
        c.advance(0)
    assert c.tick == -1 and not c.latest and not c.queue


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": "relay.cloth.communication.v2"},
        {"checkpoint_sha256": "d" * 64},
        {"space_id": "unrelated-model-space"},
        {"run_id": "other-run"},
        {"recipient": "left"},
        {"deliver_tick": 0},
        {"expires_tick": 100},
        {"content": {"vector": [0.0] * 63}},
        {"content": {"vector": [1.1] * 64}},
        {"content": {"vector": [True] * 64}},
        {"content": {"vector": [0.0] * 64, "text": "{}"}},
    ],
)
def test_incompatible_packets_are_rejected(policy_factory, mutation):
    c = channel(policy_factory())
    altered = copy.deepcopy(publish(c))
    altered.update(mutation)
    with pytest.raises(ValueError):
        c.validate(altered)


def test_cross_native_extractors_cannot_be_silently_interchanged(policy_factory):
    c = channel(policy_factory())
    with pytest.raises(ValueError, match="extractor"):
        c.publish(
            "left",
            NATIVE["left"],
            FACTS,
            0,
            extractor_id=EXTRACTORS["right"],
            source_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="features"):
        c.publish(
            "left",
            NATIVE["right"],
            FACTS,
            0,
            extractor_id=EXTRACTORS["left"],
            source_sha256="c" * 64,
        )
    assert c.sent_ticks == set() and c.pending == []


def test_older_arrival_cannot_overwrite_newer_native_observation(policy_factory):
    network = NetworkConfig(seed=2, max_delay_ticks=20, drop_probability=0, ttl_ticks=50)
    earlier, later = next(
        (a, b)
        for a in range(20)
        for b in range(a + 1, 21)
        if network.schedule("left", a)[0] > network.schedule("left", b)[0]
    )
    c = channel(policy_factory(), network=network)
    for tick in range(network.schedule("left", earlier)[0] + 1):
        if tick in (earlier, later):
            publish(c, tick=tick)
        c.advance(tick)
    assert c.receive("right", c.tick)["observation_tick"] == later
    assert c.counter["out_of_order"] == 1


def test_expiry_is_from_observation_and_missing_stays_unknown(policy_factory):
    network = NetworkConfig(min_delay_ticks=1, max_delay_ticks=1, drop_probability=0, ttl_ticks=2)
    c = channel(policy_factory(), network=network)
    publish(c)
    c.advance(0)
    assert c.receive("left", 0) is None
    c.advance(1)
    assert c.receive("right", 1)["age_ticks"] == 1
    c.advance(2)
    assert c.receive("right", 2) is None
    assert c.counter["stale_reads"] == 1


def test_expired_in_transit_is_not_accepted(policy_factory):
    network = NetworkConfig(min_delay_ticks=3, max_delay_ticks=3, drop_probability=0, ttl_ticks=2)
    c = channel(policy_factory(), network=network)
    publish(c)
    c.advance(0)
    c.advance(3)
    assert c.receive("right", 3) is None
    assert c.counter["expired"] == 1 and c.counter["delivered"] == 0


def test_no_communication_has_no_peer_payload(policy_factory):
    c = channel(policy_factory("none"))
    assert publish(c) is None
    c.advance(0, state={"tick": 0})
    c.advance(1)
    assert c.receive("right", 1) is None
    assert c.counter["packet_bytes"] == 0 and c.counter["local_observations"] == 1


def test_source_hash_binds_feature_contract_and_inputs_not_wall_time():
    metadata = {
        "rgb_sha256": "a" * 64,
        "instruction_sha256": "b" * 64,
        "vector_sha256": "c" * 64,
        "feature_space_id": "d" * 64,
        "inference_ms": 10,
    }
    original = feature_source_hash(SimpleNamespace(metadata=metadata))
    assert original == feature_source_hash(
        SimpleNamespace(metadata={**metadata, "inference_ms": 20})
    )
    for key in ("rgb_sha256", "instruction_sha256", "vector_sha256", "feature_space_id"):
        changed = SimpleNamespace(metadata={**metadata, key: "e" * 64})
        assert feature_source_hash(changed) != original


def test_live_learned_policy_transport_through_actual_spacetime(policy_factory):
    if os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1":
        pytest.skip("Set RELAY_SPACETIME_INTEGRATION=1 for actual learned-channel journal test")
    backend = SpacetimeJournal()
    c = channel(policy_factory("hybrid"), backend=backend)
    sent = publish(c)
    c.advance(0, state={"tick": 0, "test_fixture": "synthetic-feature-trained-adapter"})
    c.advance(1)
    belief = c.receive("right", 1)
    assert belief["vector"] == sent["content"]["vector"]
    assert belief["features"] == {key: FACTS[key] for key in SAFETY_FIELDS}
    assert backend.history()[0]["packets"] == [sent]
