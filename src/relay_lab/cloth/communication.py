"""Causal, delayed sharing of two arms' *local* observations.

The learned representation is a fitted linear feature codec (PCA), not a
foundation VLA hidden state. Structured text preserves every measured feature;
hybrid additionally preserves the safety fields exactly. All modes experience
the same seeded simulation-time network schedule. Database acknowledgements
are a separate durability barrier, never a source of simulated network delay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "relay.cloth.communication.v2"
AGENTS = ("left", "right")
MODES = ("embeddings", "structured_text", "hybrid", "none")
FEATURES = tuple(
    f"{point}_{axis}" for point in ("ee", "hem", "sleeve", "shoulder") for axis in "xyz"
) + (
    "grasped",
    "obstacle_x",
    "obstacle_y",
    "obstacle_z",
    "obstacle_radius",
    "obstacle_active",
    "hem_visible",
    "sleeve_visible",
    "shoulder_visible",
    "grasp_sleeve",
    "grasp_hem",
)
SAFETY_FIELDS = (
    "grasped",
    "obstacle_x",
    "obstacle_y",
    "obstacle_z",
    "obstacle_radius",
    "obstacle_active",
    "hem_visible",
    "sleeve_visible",
    "shoulder_visible",
    "grasp_sleeve",
    "grasp_hem",
)
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("Communication values must be finite JSON") from error


def check_features(features, names=FEATURES):
    if not isinstance(features, dict) or set(features) != set(names):
        raise ValueError("Local feature schema mismatch")
    if any(
        type(features[name]) not in (int, float)
        or not math.isfinite(features[name])
        or abs(features[name]) > 100
        for name in names
    ):
        raise ValueError("Local features must be bounded finite numbers")
    return {name: float(features[name]) for name in names}


def observation_features(observation):
    """Extract only fields returned by ClothWorld.observe; never consult the world."""
    result = {}
    for point, coordinates in (
        ("ee", observation["ee_position"]),
        *(
            (name, observation["landmarks"].get(name, (0, 0, 0)))
            for name in ("hem", "sleeve", "shoulder")
        ),
    ):
        if len(coordinates) != 3:
            raise ValueError("Local landmark must have three coordinates")
        result.update({f"{point}_{axis}": float(value) for axis, value in zip("xyz", coordinates)})
    result["grasped"] = float(bool(observation["grasped"]))
    target = observation.get("grasp_target", "none" if not observation["grasped"] else None)
    if target not in ("none", "sleeve", "hem") or bool(observation["grasped"]) != (
        target != "none"
    ):
        raise ValueError("Measured grasp target must identify the actual own attachment")
    result["grasp_sleeve"] = float(target == "sleeve")
    result["grasp_hem"] = float(target == "hem")
    obstacle = observation.get("obstacle")
    position = obstacle["position"] if obstacle else (0, 0, 0)
    result.update({f"obstacle_{axis}": float(value) for axis, value in zip("xyz", position)})
    result["obstacle_radius"] = float(obstacle["radius"]) if obstacle else 0.0
    result["obstacle_active"] = float(bool(obstacle and obstacle["active"]))
    for name in ("hem", "sleeve", "shoulder"):
        result[f"{name}_visible"] = float(name in observation["landmarks"])
    return check_features(result)


def migrate_training_features(samples):
    """Keep known legacy no-grasp measurements; never infer attachment from phase.

    Old positive-grasp records lack the actual clamp identity and must be
    recollected. This helper leaves the input untouched and reports exclusions.
    """
    migrated, rejected, unchanged = [], 0, 0
    legacy = set(FEATURES) - {"grasp_sleeve", "grasp_hem"}
    for original in samples:
        row = dict(original)
        if set(row) == set(FEATURES):
            migrated.append(check_features(row))
            unchanged += 1
        elif set(row) == legacy and row.get("grasped") == 0:
            migrated.append(check_features({**row, "grasp_sleeve": 0.0, "grasp_hem": 0.0}))
        elif set(row) == legacy and row.get("grasped") == 1:
            rejected += 1
        else:
            raise ValueError("Unknown training feature schema; cannot migrate measurements")
    return migrated, {
        "input_rows": len(samples),
        "retained_rows": len(migrated),
        "already_current": unchanged,
        "discarded_unknown_attachment": rejected,
        "inferred_targets_from_phase": False,
    }


class LearnedFeatureCodec:
    """A trained, standardized PCA bottleneck with a reproducible JSON checkpoint."""

    def __init__(self, checkpoint):
        self.checkpoint = copy.deepcopy(checkpoint)
        if set(checkpoint) != {
            "kind",
            "space_id",
            "features",
            "mean",
            "scale",
            "components",
            "training",
        }:
            raise ValueError("Invalid feature codec checkpoint")
        if checkpoint["kind"] != "standardized-pca-v1" or checkpoint["features"] != list(FEATURES):
            raise ValueError("Unsupported feature codec")
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float64)
        self.scale = np.asarray(checkpoint["scale"], dtype=np.float64)
        self.components = np.asarray(checkpoint["components"], dtype=np.float64)
        if (
            self.mean.shape != (len(FEATURES),)
            or self.scale.shape != self.mean.shape
            or self.components.ndim != 2
            or self.components.shape[1] != len(FEATURES)
            or not 1 <= len(self.components) < len(FEATURES)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or not np.isfinite(self.components).all()
            or (self.scale <= 0).any()
            or not np.allclose(
                self.components @ self.components.T, np.eye(len(self.components)), atol=1e-5
            )
        ):
            raise ValueError("Invalid PCA parameters")
        training = checkpoint["training"]
        if (
            checkpoint["space_id"] != f"cloth-local-pca-{len(self.components)}-v2"
            or not isinstance(training, dict)
            or type(training.get("samples")) is not int
            or training["samples"] < 2 * len(FEATURES)
            or not isinstance(training.get("feature_sha256"), str)
            or not _HASH.fullmatch(training["feature_sha256"])
        ):
            raise ValueError("Invalid trained codec provenance")
        self.sha256 = hashlib.sha256(canonical(checkpoint).encode()).hexdigest()

    @classmethod
    def fit(cls, samples, *, dimensions=8, training=None):
        rows = [check_features(row) for row in samples]
        if len(rows) < len(FEATURES) * 2 or not 1 <= dimensions < len(FEATURES):
            raise ValueError(
                f"PCA needs at least {2 * len(FEATURES)} training observations and a reduced dimension"
            )
        matrix = np.asarray([[row[key] for key in FEATURES] for row in rows], dtype=np.float64)
        mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
        scale[scale < 1e-6] = 1.0
        _, _, vectors = np.linalg.svd((matrix - mean) / scale, full_matrices=False)
        components = vectors[:dimensions]
        # Fix the arbitrary SVD signs so identical training data has one identity.
        for row in components:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        checkpoint = {
            "kind": "standardized-pca-v1",
            "space_id": f"cloth-local-pca-{dimensions}-v2",
            "features": list(FEATURES),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "components": components.tolist(),
            "training": {
                **(training or {}),
                "samples": len(rows),
                "feature_sha256": hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest(),
            },
        }
        return cls(checkpoint)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical(self.checkpoint) + "\n")

    @property
    def model_info(self):
        return {
            "kind": self.checkpoint["kind"],
            "space_id": self.checkpoint["space_id"],
            "checkpoint_sha256": self.sha256,
            "dimension": len(self.components),
            "source_dimension": len(FEATURES),
            "training": copy.deepcopy(self.checkpoint["training"]),
        }

    def encode(self, features):
        row = check_features(features)
        values = np.asarray([row[key] for key in FEATURES])
        return (((values - self.mean) / self.scale) @ self.components.T).astype(np.float32).tolist()

    def decode(self, vector):
        value = np.asarray(vector, dtype=np.float64)
        if (
            value.shape != (len(self.components),)
            or not np.isfinite(value).all()
            or (np.abs(value) > 1000).any()
        ):
            raise ValueError("Invalid feature latent")
        features = (value @ self.components) * self.scale + self.mean
        return check_features(dict(zip(FEATURES, features.tolist())))

    def evaluate(self, samples):
        """Report reconstruction errors on supplied held-out observations; no fitting."""
        rows = [check_features(row) for row in samples]
        if not rows:
            raise ValueError("Codec evaluation requires observations")
        truth = np.asarray([[row[name] for name in FEATURES] for row in rows])
        decoded = [self.decode(self.encode(row)) for row in rows]
        reconstructed = np.asarray([[row[name] for name in FEATURES] for row in decoded])
        errors = reconstructed - truth
        return {
            "samples": len(rows),
            "checkpoint_sha256": self.sha256,
            "rmse_by_feature": dict(zip(FEATURES, np.sqrt(np.mean(errors**2, axis=0)).tolist())),
            "max_absolute_error": float(np.max(np.abs(errors))),
            "obstacle_presence_accuracy": float(
                np.mean(
                    (reconstructed[:, FEATURES.index("obstacle_active")] >= 0.5)
                    == (truth[:, FEATURES.index("obstacle_active")] >= 0.5)
                )
            ),
        }


@dataclass(frozen=True)
class NetworkConfig:
    seed: int = 0
    min_delay_ticks: int = 1
    max_delay_ticks: int = 12
    drop_probability: float = 0.1
    ttl_ticks: int = 24

    def __post_init__(self):
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.min_delay_ticks) is not int
            or type(self.max_delay_ticks) is not int
            or not 1 <= self.min_delay_ticks <= self.max_delay_ticks <= 10000
            or type(self.ttl_ticks) is not int
            or not 1 <= self.ttl_ticks <= 10000
            or type(self.drop_probability) not in (int, float)
            or not 0 <= self.drop_probability <= 1
        ):
            raise ValueError("Invalid communication network configuration")

    def schedule(self, sender, tick):
        # Neither representation, run ID, payload, outcomes nor global RNG state enters this schedule.
        digest = hashlib.sha256(f"cloth-network-v1:{self.seed}:{sender}:{tick}".encode()).digest()
        delay = self.min_delay_ticks + int.from_bytes(digest[:8], "big") % (
            self.max_delay_ticks - self.min_delay_ticks + 1
        )
        dropped = int.from_bytes(digest[8:16], "big") / 2**64 < self.drop_probability
        return tick + delay, dropped


class MemoryJournal:
    """Reference transaction store, with immutable exact readback like the DB adapter."""

    def __init__(self):
        self.frames, self.config, self.run_id = {}, None, None

    def create(self, run_id, config):
        if self.config is not None and (self.run_id != run_id or self.config != config):
            raise ValueError("Journal already belongs to another channel")
        self.run_id, self.config = run_id, copy.deepcopy(config)

    def commit(self, tick, packets, state=None):
        value = {"packets": packets, "state": state or {}}
        encoded = canonical(value)
        if tick in self.frames:
            if self.frames[tick] != encoded:
                raise ValueError("Conflicting communication commit")
            return json.loads(self.frames[tick])["packets"]
        if self.frames and tick < max(self.frames):
            raise ValueError("Communication commit moved backwards")
        self.frames[tick] = encoded
        return json.loads(self.frames[tick])["packets"]

    @property
    def backend_info(self):
        return {"kind": "memory", "durable": False, "index": "exact run/packet identity"}


class Channel:
    def __init__(self, run_id, *, mode, codec, network=None, backend=None):
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id) or mode not in MODES:
            raise ValueError("Invalid channel run or representation")
        self.run_id, self.mode, self.codec = run_id, mode, codec
        self.network, self.backend = network or NetworkConfig(), backend or MemoryJournal()
        self.config = {
            "schema": SCHEMA,
            "mode": mode,
            "model": codec.model_info,
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

    def publish(self, sender, observation, tick):
        if sender not in AGENTS or type(tick) is not int or tick < max(0, self.tick):
            raise ValueError("Invalid local observation sender/tick")
        if (sender, tick) in self.sent_ticks:
            raise ValueError("Duplicate local observation")
        if self.pending and self.pending[0]["emitted_tick"] != tick:
            raise ValueError("Flush the previous communication batch first")
        features = check_features(observation.get("features", observation))
        source = observation.get(
            "source_sha256", hashlib.sha256(canonical(features).encode()).hexdigest()
        )
        if not isinstance(source, str) or not _HASH.fullmatch(source):
            raise ValueError("Invalid observation provenance hash")
        self.sent_ticks.add((sender, tick))
        self.counter["local_observations"] += 1
        start = time.perf_counter()
        # All conditions run the same local encoder; only transmitted content differs.
        latent = self.codec.encode(features)
        self.encode_ms.append((time.perf_counter() - start) * 1000)
        if self.mode == "none":
            return None
        content = {}
        if self.mode in ("embeddings", "hybrid"):
            content["vector"] = latent
        if self.mode == "structured_text":
            content["text"] = canonical(features)
        elif self.mode == "hybrid":
            content["text"] = canonical({key: features[key] for key in SAFETY_FIELDS})
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
            "source_sha256": source,
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
        fields = {
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
        if not isinstance(packet, dict) or set(packet) != fields or len(canonical(packet)) > 12000:
            raise ValueError("Invalid communication packet shape")
        sender, tick = packet["sender"], packet["observation_tick"]
        if sender not in AGENTS or type(tick) is not int or tick < 0:
            raise ValueError("Invalid communication sender/time")
        if (
            any(
                type(packet[name]) is not int
                for name in ("emitted_tick", "deliver_tick", "expires_tick")
            )
            or type(packet["dropped"]) is not bool
        ):
            raise ValueError("Invalid communication schedule types")
        due, dropped = self.network.schedule(sender, tick)
        required = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "mode": self.mode,
            "packet_id": f"{sender}-{tick}",
            "recipient": AGENTS[1 - AGENTS.index(sender)],
            "emitted_tick": tick,
            "deliver_tick": due,
            "dropped": dropped,
            "expires_tick": tick + self.network.ttl_ticks,
            "checkpoint_sha256": self.codec.sha256,
            "space_id": self.codec.model_info["space_id"],
        }
        if self.mode == "none" or any(packet.get(key) != value for key, value in required.items()):
            raise ValueError("Communication scope, schedule, or codec mismatch")
        if not isinstance(packet["source_sha256"], str) or not _HASH.fullmatch(
            packet["source_sha256"]
        ):
            raise ValueError("Invalid communication provenance")
        content = packet["content"]
        expected = (
            {"vector"}
            if self.mode == "embeddings"
            else {"text"}
            if self.mode == "structured_text"
            else {"vector", "text"}
        )
        if not isinstance(content, dict) or set(content) != expected:
            raise ValueError("Mixed or unsupported representation")
        if "vector" in content:
            self.codec.decode(content["vector"])
        if "text" in content:
            if not isinstance(content["text"], str):
                raise ValueError("Structured text must be JSON text")
            try:
                facts = json.loads(content["text"])
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid structured text JSON") from error
            check_features(facts, FEATURES if self.mode == "structured_text" else SAFETY_FIELDS)
            if canonical(facts) != content["text"]:
                raise ValueError("Structured text must have an unambiguous canonical encoding")

    def advance(self, tick, state=None):
        if type(tick) is not int or tick < max(0, self.tick):
            raise ValueError("Communication clock cannot move backwards")
        if self.pending or state is not None:
            if any(packet["emitted_tick"] != tick for packet in self.pending):
                raise ValueError("Commit observations at their actual simulation tick")
            start = time.perf_counter()
            confirmed = self.backend.commit(tick, copy.deepcopy(self.pending), state)
            self.durable_ms.append((time.perf_counter() - start) * 1000)
            # Readback must match exactly; an ACK alone cannot inject/relabel messages.
            if canonical(confirmed) != canonical(self.pending):
                raise ValueError("Communication readback differs from the committed batch")
            for packet in confirmed:
                self.validate(packet)
                self.events.append(
                    {"kind": "published", "tick": tick, "packet": copy.deepcopy(packet)}
                )
                if packet["dropped"]:
                    self.counter["dropped"] += 1
                else:
                    self.queue.append(copy.deepcopy(packet))
            self.pending.clear()
        self.tick = tick
        waiting = []
        for packet in sorted(self.queue, key=lambda p: (p["deliver_tick"], p["packet_id"])):
            if packet["deliver_tick"] > tick:
                waiting.append(packet)
                continue
            if tick >= packet["expires_tick"]:
                self.counter["expired"] += 1
                reason = "expired"
            elif (
                packet["recipient"] in self.latest
                and self.latest[packet["recipient"]]["observation_tick"]
                >= packet["observation_tick"]
            ):
                self.counter["out_of_order"] += 1
                reason = "out_of_order"
            else:
                self.latest[packet["recipient"]] = copy.deepcopy(packet)
                self.arrival_ticks[packet["recipient"]] = tick
                self.received_ids.add(packet["packet_id"])
                self.counter["delivered"] += 1
                self.delays.append(tick - packet["emitted_tick"])
                reason = "delivered"
            self.events.append(
                {
                    "kind": reason,
                    "tick": tick,
                    "packet_id": packet["packet_id"],
                    "observation_tick": packet["observation_tick"],
                }
            )
        self.queue = waiting

    def receive(self, recipient, tick):
        if recipient not in AGENTS or type(tick) is not int or tick != self.tick:
            raise ValueError("Recipient must read the current advanced simulation tick")
        self.counter["retrievals"] += 1
        packet = self.latest.get(recipient)
        if packet is None:
            self.counter["missing_reads"] += 1
            return None
        if tick >= packet["expires_tick"]:
            self.counter["stale_reads"] += 1
            return None
        start = time.perf_counter()
        features = (
            self.codec.decode(packet["content"]["vector"]) if "vector" in packet["content"] else {}
        )
        if "text" in packet["content"]:
            features.update(json.loads(packet["content"]["text"]))
        self.decode_ms.append((time.perf_counter() - start) * 1000)
        age = tick - packet["observation_tick"]
        self.ages.append(age)
        return {
            "features": features,
            "sender": packet["sender"],
            "recipient": recipient,
            "packet_id": packet["packet_id"],
            "observation_tick": packet["observation_tick"],
            "emitted_tick": packet["emitted_tick"],
            "scheduled_delivery_tick": packet["deliver_tick"],
            "received_tick": self.arrival_ticks[recipient],
            "age_ticks": age,
            "expires_tick": packet["expires_tick"],
            "source_sha256": packet["source_sha256"],
            "checkpoint_sha256": packet["checkpoint_sha256"],
            "space_id": packet["space_id"],
            "representation": self.mode,
            "precision": "exact observed numeric facts"
            if self.mode == "structured_text"
            else "lossy learned reconstruction; exact safety facts"
            if self.mode == "hybrid"
            else "lossy learned reconstruction",
            "uncertainty": {
                "age_ticks": age,
                "lossy_geometry": self.mode != "structured_text",
                "safety_fields_exact": self.mode in ("structured_text", "hybrid"),
            },
        }

    @property
    def metrics(self):
        def summarize(values):
            return {
                "count": len(values),
                "mean": float(np.mean(values)) if values else 0.0,
                "p95": float(np.percentile(values, 95)) if values else 0.0,
            }

        return {
            **self.counter,
            "network_delay_ticks": summarize(self.delays),
            "observation_age_ticks": summarize(self.ages),
            "encode_ms": summarize(self.encode_ms),
            "decode_ms": summarize(self.decode_ms),
            "durable_commit_ms": summarize(self.durable_ms),
            "pending_delivery": len(self.queue),
            "backend": self.backend.backend_info,
            "schedule": "seed/sender/observation tick; independent of representation and payload",
            "byte_accounting": "UTF-8 JSON content/envelope; excludes HTTP and SQL readback",
            "model": self.codec.model_info,
        }
