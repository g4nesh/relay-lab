"""Job-scoped learned beliefs; identical delivery rules for text and vectors.

A refreshed packet is a retransmission of an old observation, not a new visual
measurement. Original evidence/payload are immutable until job completion.
The trusted worker enforces recipient views; this is not multi-tenant security.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections import deque

from relay_lab.spacetime import SpacetimeConflict

from .embeddings import unit_vector

SCHEMA = "relay.factory.semantic.v1"
SPACE = "factory-route-v1"
SENDER, RECIPIENT = "humanoid-01", "gantry-01"
FIELDS = {
    "schema",
    "space_id",
    "checkpoint_sha256",
    "mode",
    "run_id",
    "job_id",
    "sender",
    "recipient",
    "topic",
    "packet_id",
    "emitted_tick",
    "expires_tick",
    "observation_tick",
    "source_command_id",
    "source_frame_sha256",
}


def require(condition, message):
    if not condition:
        raise SpacetimeConflict(message)


def encode(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise SpacetimeConflict("Semantic packet must contain finite JSON values") from None


def channel_config(mode, checkpoint_sha256, space_id=SPACE, period_ticks=5, ttl_ticks=15):
    require(mode in ("embeddings", "text", "none"), "Unsupported semantic mode")
    require(space_id == SPACE, "Unsupported semantic space")
    require(
        isinstance(checkpoint_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256),
        "Invalid semantic checkpoint",
    )
    require(type(period_ticks) is int and 1 <= period_ticks <= 64, "Invalid semantic cadence")
    require(
        type(ttl_ticks) is int and 15 <= ttl_ticks <= 64 and period_ticks <= ttl_ticks,
        "Invalid semantic TTL",
    )
    return {
        "schema": SCHEMA,
        "mode": mode,
        "checkpoint_sha256": checkpoint_sha256,
        "space_id": space_id,
        "period_ticks": period_ticks,
        "ttl_ticks": ttl_ticks,
    }


def origin(packet):
    return {
        key: value
        for key, value in packet.items()
        if key not in ("packet_id", "emitted_tick", "expires_tick")
    }


def validate_packet(packet, config, run_id, jobs):
    require(isinstance(packet, dict), "Semantic packet must be an object")
    require(config["mode"] != "none", "Semantic channel is disabled")
    payload = "vector" if config["mode"] == "embeddings" else "text"
    require(
        set(packet) == FIELDS | {payload},
        "Semantic packet fields disclose mixed or unsupported data",
    )
    require(len(encode(packet).encode()) <= 4096, "Semantic packet exceeds limit")
    for key in ("schema", "space_id", "checkpoint_sha256", "mode"):
        require(packet[key] == config[key], "Semantic packet schema/model/mode mismatch")
    require(
        packet["run_id"] == run_id
        and isinstance(packet["job_id"], str)
        and packet["job_id"] in jobs,
        "Semantic run/job scope mismatch",
    )
    require(
        packet["sender"] == SENDER
        and packet["recipient"] == RECIPIENT
        and packet["topic"] == "dispatch_route",
        "Semantic peer/topic mismatch",
    )
    require(
        isinstance(packet["source_frame_sha256"], str)
        and re.fullmatch(r"[a-f0-9]{64}", packet["source_frame_sha256"]),
        "Invalid semantic source frame",
    )
    require(
        isinstance(packet["source_command_id"], str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", packet["source_command_id"]),
        "Invalid semantic source command",
    )
    tick = packet["emitted_tick"]
    require(type(tick) is int and tick >= 1, "Invalid semantic emission tick")
    require(
        type(packet["observation_tick"]) is int and 0 <= packet["observation_tick"] < tick,
        "Invalid semantic observation tick",
    )
    require(
        type(packet["expires_tick"]) is int
        and packet["expires_tick"] == tick + config["ttl_ticks"],
        "Semantic TTL differs from configured delivery",
    )
    require(
        packet["packet_id"] == f"{SENDER}-{packet['job_id']}-{tick}",
        "Invalid semantic packet identity",
    )
    if payload == "vector":
        unit_vector(packet[payload])
    else:
        require(
            isinstance(packet[payload], str) and 0 < len(packet[payload]) <= 256,
            "Invalid semantic text payload",
        )


class SemanticIndex:
    def __init__(self, run_id, jobs, config):
        require(1 <= len(jobs) <= 3, "Routing supports one to three jobs")
        self.run_id, self.jobs, self.config = run_id, set(jobs), config
        self.records = {}
        self.cache_tick = 0
        self.db_ms, self.retrieval_ms = deque(maxlen=1000), deque(maxlen=1000)
        self.refreshes = self.retrievals = self.transport_bytes = self.write_bytes = (
            self.stale_drops
        ) = 0

    def fork(self):
        clone = copy.copy(self)
        clone.records = self.records.copy()
        return clone

    def commit(self, packets, tick, commands, state, previous):
        require(
            isinstance(packets, list) and len(packets) <= len(self.jobs),
            "Too many semantic packets",
        )
        seen = set()
        for packet in packets:
            validate_packet(packet, self.config, self.run_id, self.jobs)
            jid = packet["job_id"]
            require(
                jid not in seen and packet["emitted_tick"] == tick,
                "Duplicate job or stale/future semantic packet",
            )
            seen.add(jid)
            require(
                state["jobs"][jid]["stage"] != "completed", "Cannot publish completed-job beliefs"
            )
            old = self.records.get(jid)
            if old:
                require(
                    tick % self.config["period_ticks"] == 0,
                    "Semantic retransmission outside cadence",
                )
                require(
                    origin(old) == origin(packet), "Semantic evidence and payload are immutable"
                )
            else:
                command = commands.get(packet["source_command_id"], {})
                require(
                    packet["observation_tick"] == tick - 1,
                    "Initial semantic observation must precede this physical tick",
                )
                require(
                    previous["jobs"][jid]["stage"] == "queued"
                    and command.get("robot_id") == SENDER
                    and command.get("job_id") == jid
                    and command.get("capability") == "transfer",
                    "Initial semantic evidence requires this job's authorized humanoid transfer",
                )
            self.records[jid] = copy.deepcopy(packet)
            self.write_bytes += len(encode(packet).encode())
        self.records = {
            jid: row
            for jid, row in self.records.items()
            if state["jobs"][jid]["stage"] != "completed"
        }

    def refresh(self, packets, tick, *, db_ms=0.0, transport_bytes=0):
        require(
            isinstance(packets, list) and len(packets) <= len(self.jobs),
            "Semantic read exceeds job bound",
        )
        checked = {}
        for packet in packets:
            validate_packet(packet, self.config, self.run_id, self.jobs)
            require(
                packet["job_id"] not in checked and packet["emitted_tick"] <= tick,
                "Duplicate or future semantic read",
            )
            checked[packet["job_id"]] = copy.deepcopy(packet)
        self.records, self.cache_tick = checked, tick
        self.refreshes += 1
        self.db_ms.append(db_ms)
        self.transport_bytes += transport_bytes

    def retrieve(self, recipient, job_id, tick):
        require(
            recipient == RECIPIENT and job_id in self.jobs, "Semantic recipient/job access mismatch"
        )
        require(type(tick) is int and tick >= self.cache_tick, "Semantic retrieval predates cache")
        start = time.perf_counter()
        row = self.records.get(job_id)
        result = None
        if row and row["emitted_tick"] <= tick < row["expires_tick"]:
            result = {
                **copy.deepcopy(row),
                "observation_age_ticks": tick - row["observation_tick"],
                "packet_age_ticks": tick - row["emitted_tick"],
            }
        elif row:
            self.stale_drops += 1
        self.retrievals += 1
        self.retrieval_ms.append((time.perf_counter() - start) * 1000)
        return result

    @property
    def metrics(self):
        def pct(values):
            values = sorted(values)
            return {
                f"p{p}_ms": values[min(len(values) - 1, int(len(values) * p / 100))]
                if values
                else 0.0
                for p in (50, 95)
            }

        return {
            "index": "Spacetime B-tree run/job; exact job-scoped retrieval",
            "db_refresh": pct(self.db_ms),
            "local_retrieval": pct(self.retrieval_ms),
            "transport_bytes": self.transport_bytes,
            "packet_payload_write_bytes": self.write_bytes,
            "refreshes": self.refreshes,
            "retrievals": self.retrievals,
            "cache_tick": self.cache_tick,
            "cached_records": len(self.records),
            "record_limit": len(self.jobs),
            "stale_drops": self.stale_drops,
        }
