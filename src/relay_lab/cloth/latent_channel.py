"""Causal transport of learned 64D messages across incompatible VLA backbones.

Uses the existing exact-readback Spacetime journal. The inherited delivery loop
rejects expired and out-of-order observations. A receiver gets a learned vector,
never a PCA reconstruction masquerading as a VLA hidden state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np

from .communication import (
    _HASH,
    _ID,
    AGENTS,
    FEATURES,
    SAFETY_FIELDS,
    Channel,
    MemoryJournal,
    NetworkConfig,
    canonical,
    check_features,
)
from .latent_adapter import DIMENSION, MODES

SCHEMA = "relay.cloth.learned-latent.v1"


class LatentChannel(Channel):
    def __init__(self, run_id, *, policy, network=None, backend=None):
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id) or policy.mode not in MODES:
            raise ValueError("Invalid learned communication run")
        self.run_id, self.mode, self.codec = run_id, policy.mode, policy
        self.network, self.backend = network or NetworkConfig(), backend or MemoryJournal()
        self.config = {
            "schema": SCHEMA,
            "mode": self.mode,
            "model": policy.model_info,
            "network": dict(self.network.__dict__),
            "features": list(FEATURES),
            "hybrid_fields": list(SAFETY_FIELDS),
        }
        self.backend.create(run_id, self.config)
        self.tick = -1
        self.pending, self.queue, self.latest, self.events = [], [], {}, []
        self.arrival_ticks = {}
        self.sent_ticks, self.received_ids = set(), set()
        self.counter = {
            name: 0
            for name in (
                "published",
                "delivered",
                "dropped",
                "expired",
                "out_of_order",
                "payload_bytes",
                "packet_bytes",
                "retrievals",
                "missing_reads",
                "stale_reads",
                "local_observations",
            )
        }
        self.delays, self.ages, self.encode_ms, self.decode_ms, self.durable_ms = [], [], [], [], []

    def publish(self, sender, native_features, features, tick, *, extractor_id, source_sha256):
        if sender not in AGENTS or type(tick) is not int or tick < max(0, self.tick):
            raise ValueError("Invalid local observation sender/tick")
        if (sender, tick) in self.sent_ticks:
            raise ValueError("Duplicate local observation")
        if self.pending and self.pending[0]["emitted_tick"] != tick:
            raise ValueError("Commit the previous learned message batch first")
        if not isinstance(source_sha256, str) or not _HASH.fullmatch(source_sha256):
            raise ValueError("Invalid source observation hash")
        facts = check_features(features)
        started = time.perf_counter()
        # Every condition executes its local sender; only the content is ablated.
        vector = self.codec.encode(sender, native_features, extractor_id)
        self.encode_ms.append((time.perf_counter() - started) * 1000)
        self.sent_ticks.add((sender, tick))
        self.counter["local_observations"] += 1
        if self.mode == "none":
            return None
        content = {}
        if self.mode in ("embeddings", "hybrid"):
            content["vector"] = vector
        if self.mode == "structured_text":
            content["text"] = canonical(facts)
        elif self.mode == "hybrid":
            content["text"] = canonical({k: facts[k] for k in SAFETY_FIELDS})
        due, dropped = self.network.schedule(sender, tick)
        packet = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "mode": self.mode,
            "packet_id": f"{sender}-{tick}",
            "sender": sender,
            "recipient": AGENTS[1 - AGENTS.index(sender)],
            "observation_tick": tick,
            "emitted_tick": tick,
            "deliver_tick": due,
            "expires_tick": tick + self.network.ttl_ticks,
            "dropped": dropped,
            "source_sha256": source_sha256,
            "checkpoint_sha256": self.codec.sha256,
            "space_id": self.codec.model_info["space_id"],
            "content": content,
        }
        self.validate(packet)
        self.pending.append(copy.deepcopy(packet))
        self.counter["published"] += 1
        self.counter["payload_bytes"] += len(canonical(content).encode())
        self.counter["packet_bytes"] += len(canonical(packet).encode())
        return copy.deepcopy(packet)

    def validate(self, packet):
        expected = {
            "schema",
            "run_id",
            "mode",
            "packet_id",
            "sender",
            "recipient",
            "observation_tick",
            "emitted_tick",
            "deliver_tick",
            "expires_tick",
            "dropped",
            "source_sha256",
            "checkpoint_sha256",
            "space_id",
            "content",
        }
        if (
            not isinstance(packet, dict)
            or set(packet) != expected
            or len(canonical(packet)) > 12000
        ):
            raise ValueError("Invalid learned packet shape")
        sender, tick = packet["sender"], packet["observation_tick"]
        if sender not in AGENTS or type(tick) is not int or tick < 0 or self.mode == "none":
            raise ValueError("Invalid learned packet sender/tick")
        if any(
            type(packet[k]) is not int for k in ("emitted_tick", "deliver_tick", "expires_tick")
        ):
            raise ValueError("Invalid learned packet clock types")
        due, dropped = self.network.schedule(sender, tick)
        required = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "mode": self.mode,
            "packet_id": f"{sender}-{tick}",
            "recipient": AGENTS[1 - AGENTS.index(sender)],
            "emitted_tick": tick,
            "deliver_tick": due,
            "expires_tick": tick + self.network.ttl_ticks,
            "dropped": dropped,
            "checkpoint_sha256": self.codec.sha256,
            "space_id": self.codec.model_info["space_id"],
        }
        if type(packet["dropped"]) is not bool or any(packet[k] != v for k, v in required.items()):
            raise ValueError("Learned packet scope/schedule/checkpoint mismatch")
        if not isinstance(packet["source_sha256"], str) or not _HASH.fullmatch(
            packet["source_sha256"]
        ):
            raise ValueError("Invalid learned packet provenance")
        content = packet["content"]
        keys = (
            {"vector"}
            if self.mode == "embeddings"
            else {"text"}
            if self.mode == "structured_text"
            else {"vector", "text"}
        )
        if not isinstance(content, dict) or set(content) != keys:
            raise ValueError("Unexpected learned packet content")
        if "vector" in content:
            vector = content["vector"]
            if (
                not isinstance(vector, list)
                or len(vector) != DIMENSION
                or any(
                    type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 1.00001
                    for v in vector
                )
            ):
                raise ValueError("Invalid tanh-bounded learned message")
        if "text" in content:
            if not isinstance(content["text"], str):
                raise ValueError("Learned structured facts must be text")
            facts = json.loads(content["text"])
            check_features(facts, FEATURES if self.mode == "structured_text" else SAFETY_FIELDS)
            if canonical(facts) != content["text"]:
                raise ValueError("Noncanonical learned structured facts")

    def receive(self, recipient, tick):
        if recipient not in AGENTS or type(tick) is not int or tick != self.tick:
            raise ValueError("Read learned messages only at the current simulation tick")
        self.counter["retrievals"] += 1
        packet = self.latest.get(recipient)
        if packet is None:
            self.counter["missing_reads"] += 1
            return None
        if tick >= packet["expires_tick"]:
            self.counter["stale_reads"] += 1
            return None
        self.validate(packet)
        started = time.perf_counter()
        content = packet["content"]
        result = {
            "vector": copy.deepcopy(content.get("vector", [0.0] * DIMENSION)),
            "features": json.loads(content["text"]) if "text" in content else {},
            "sender": packet["sender"],
            "recipient": recipient,
            "packet_id": packet["packet_id"],
            "age_ticks": tick - packet["observation_tick"],
            "observation_tick": packet["observation_tick"],
            "expires_tick": packet["expires_tick"],
            "source_sha256": packet["source_sha256"],
            "checkpoint_sha256": packet["checkpoint_sha256"],
            "space_id": packet["space_id"],
            "representation": self.mode,
        }
        self.decode_ms.append((time.perf_counter() - started) * 1000)
        self.ages.append(result["age_ticks"])
        return result


def feature_source_hash(feature_record):
    """Bind pixels, language, feature bytes and the pinned extraction contract."""
    metadata = feature_record.metadata
    return hashlib.sha256(
        canonical(
            {
                k: metadata[k]
                for k in ("rgb_sha256", "instruction_sha256", "vector_sha256", "feature_space_id")
            }
        ).encode()
    ).hexdigest()
