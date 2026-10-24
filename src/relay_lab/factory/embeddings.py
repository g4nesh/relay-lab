"""Versioned learned VLA latents with bounded exact, latest-peer retrieval.

SpacetimeDB stores and indexes records by run and robot. Cosine scoring happens
in this client over at most one eligible record per peer; this is not a native
vector/ANN index. Decoder text is a model prediction, never physical evidence.
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
from collections import deque

from relay_lab.spacetime import SpacetimeConflict

DIMENSION = 32
SCHEMA = "relay.factory.embedding.v1"
MAX_PER_ROBOT = 16
MAX_RECORD_BYTES = 4096


class EmbeddingError(SpacetimeConflict):
    pass


def check(condition, message):
    if not condition:
        raise EmbeddingError(message)


def encode(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise EmbeddingError("Embedding must contain finite JSON values") from None


def unit_vector(value):
    check(isinstance(value, list) and len(value) == DIMENSION, "Embedding dimension must be 32")
    check(
        all(type(x) in (int, float) and math.isfinite(x) for x in value), "Invalid embedding vector"
    )
    check(abs(sum(x * x for x in value) - 1) <= 0.002, "Embedding must be unit normalized")
    return value


def space_config(space_id, checkpoint_sha256):
    check(
        isinstance(space_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", space_id),
        "Invalid embedding space",
    )
    check(
        isinstance(checkpoint_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256),
        "Invalid checkpoint SHA256",
    )
    return {
        "schema": SCHEMA,
        "space_id": space_id,
        "checkpoint_sha256": checkpoint_sha256,
        "dimension": DIMENSION,
        "modality": "vla-hidden",
        "max_per_robot": MAX_PER_ROBOT,
    }


def validate_record(record, config, run_id, tick, robots, commands, state):
    check(
        isinstance(record, dict) and len(encode(record).encode()) <= MAX_RECORD_BYTES,
        "Oversized embedding record",
    )
    for key in ("schema", "space_id", "checkpoint_sha256", "dimension", "modality"):
        check(record.get(key) == config[key], "Embedding schema, model or space mismatch")
    check(record.get("run_id") == run_id, "Cross-run embedding")
    robot = record.get("robot_id")
    check(robot in robots, "Unknown embedding robot")
    check(
        type(record.get("tick")) is int and record["tick"] == tick, "Stale or future embedding tick"
    )
    check(
        record.get("embedding_id") == f"{robot}-{tick}", "Embedding identity must be robot_id-tick"
    )
    check(
        type(record.get("expires_tick")) is int and tick < record["expires_tick"] <= tick + 64,
        "Invalid embedding expiry",
    )
    unit_vector(record.get("vector"))
    decoder = record.get("decoder")
    check(
        isinstance(decoder, dict)
        and isinstance(decoder.get("text"), str)
        and len(decoder["text"]) <= 512
        and isinstance(decoder.get("version"), str)
        and 0 < len(decoder["version"]) <= 128,
        "Invalid embedding semantic decoder trace",
    )
    confidence = decoder.get("confidence")
    check(
        type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1,
        "Invalid decoder confidence",
    )
    observed = next(
        (r for r in state.get("robots", []) if r.get("id", r.get("robot_id")) == robot), None
    )
    check(observed is not None, "Embedding has no observed robot state")
    cid = record.get("command_id")
    if cid is None:
        check(
            record.get("job_id") is None and not observed.get("active_command"),
            "Idle embedding claims active robot",
        )
    else:
        command = commands.get(cid)
        check(
            command is not None
            and command["robot_id"] == robot
            and command.get("job_id") == record.get("job_id"),
            "Embedding command or job scope mismatch",
        )
        check(
            observed.get("active_command") in (None, cid),
            "Embedding command differs from observed active command",
        )
    return record


class EmbeddingIndex:
    def __init__(self, run_id, robot_ids, config):
        self.run_id, self.robot_ids, self.config = run_id, set(robot_ids), config
        self.records = {}
        self.cache_tick = 0
        self.db_ms, self.cosine_ms = deque(maxlen=1000), deque(maxlen=1000)
        self.transport_bytes = self.write_bytes = self.refreshes = self.retrievals = 0
        self.stale_drops = self.cache_age_ticks = 0

    def fork(self):
        result = copy.copy(self)
        result.records = self.records.copy()
        return result

    def commit(self, records, tick, commands, state):
        check(
            isinstance(records, list) and len(records) <= len(self.robot_ids),
            "Too many embedding records",
        )
        seen = set()
        for record in records:
            validate_record(record, self.config, self.run_id, tick, self.robot_ids, commands, state)
            ident = record["embedding_id"]
            check(record["robot_id"] not in seen, "Duplicate robot embedding in batch")
            seen.add(record["robot_id"])
            prior = self.records.get(ident)
            check(prior is None or prior == record, "Conflicting duplicate embedding")
            self.records[ident] = copy.deepcopy(record)
            self.write_bytes += len(encode(record).encode())
        self.records = {key: row for key, row in self.records.items() if row["expires_tick"] > tick}
        for robot in self.robot_ids:
            rows = sorted(
                (r for r in self.records.values() if r["robot_id"] == robot),
                key=lambda r: r["tick"],
                reverse=True,
            )
            for old in rows[MAX_PER_ROBOT:]:
                self.records.pop(old["embedding_id"])
        self.cache_tick = tick

    def refresh(self, records, tick, *, db_ms=0.0, transport_bytes=0):
        check(
            len(records) <= len(self.robot_ids) * MAX_PER_ROBOT,
            "Embedding read exceeds bounded index",
        )
        for record in records:
            for key in ("schema", "space_id", "checkpoint_sha256", "dimension", "modality"):
                check(record.get(key) == self.config[key], "Retrieved embedding space mismatch")
            check(
                record["run_id"] == self.run_id
                and record["robot_id"] in self.robot_ids
                and record["tick"] <= tick,
                "Retrieved embedding scope or tick mismatch",
            )
            unit_vector(record["vector"])
        self.records = {r["embedding_id"]: copy.deepcopy(r) for r in records}
        self.cache_tick = tick
        self.refreshes += 1
        self.db_ms.append(db_ms)
        self.transport_bytes += transport_bytes

    def retrieve(self, robot_id, query_vector, tick, relevant_peers=None, limit=4):
        check(robot_id in self.robot_ids, "Unknown retrieving robot")
        check(
            type(tick) is int and tick >= self.cache_tick, "Retrieval tick precedes cache snapshot"
        )
        check(type(limit) is int and 1 <= limit <= 32, "Invalid retrieval limit")
        unit_vector(query_vector)
        peers = self.robot_ids - {robot_id} if relevant_peers is None else set(relevant_peers)
        check(peers <= self.robot_ids - {robot_id}, "Invalid peer retrieval scope")
        started = time.perf_counter()
        latest = {}
        expired = 0
        for row in self.records.values():
            if row["robot_id"] not in peers:
                continue
            if row["tick"] > tick or row["expires_tick"] <= tick:
                expired += 1
                continue
            old = latest.get(row["robot_id"])
            if old is None or row["tick"] > old["tick"]:
                latest[row["robot_id"]] = row
        ranked = [
            {
                **copy.deepcopy(row),
                "cosine_similarity": sum(
                    a * b for a, b in zip(query_vector, row["vector"], strict=True)
                ),
                "age_ticks": tick - row["tick"],
            }
            for row in latest.values()
        ]
        ranked.sort(key=lambda row: (-row["cosine_similarity"], row["robot_id"]))
        self.cosine_ms.append((time.perf_counter() - started) * 1000)
        self.retrievals += 1
        self.stale_drops += expired
        self.cache_age_ticks = tick - self.cache_tick
        return ranked[:limit]

    @property
    def metrics(self):
        def percentiles(values):
            ordered = sorted(values)
            return {
                f"p{p}_ms": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * p / 100) - 1)]
                if ordered
                else 0.0
                for p in (50, 95)
            }

        return {
            "index": "Spacetime B-tree run/robot + bounded client exact cosine; latest per peer",
            "db_refresh": percentiles(self.db_ms),
            "client_cosine": percentiles(self.cosine_ms),
            "transport_bytes": self.transport_bytes,
            "embedding_payload_write_bytes": self.write_bytes,
            "refreshes": self.refreshes,
            "retrievals": self.retrievals,
            "cache_tick": self.cache_tick,
            "cache_age_ticks": self.cache_age_ticks,
            "stale_drops": self.stale_drops,
            "cached_records": len(self.records),
            "record_limit": len(self.robot_ids) * MAX_PER_ROBOT,
        }
