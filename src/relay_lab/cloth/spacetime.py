"""Confirmed durable cloth communication batches through the existing owner API.

This is a trusted-worker message journal and state checkpoint, not a distributed
physics scheduler. Its synchronous HTTP barrier cannot authorize a second robot
process, and synthetic packet delays advance only with the caller's sim clock.
"""

from __future__ import annotations

import json

from relay_lab.spacetime import SpacetimeClient, SpacetimeConfig, SpacetimeConflict, _identifier

from .communication import canonical


class SpacetimeJournal:
    def __init__(self, config=None, *, worker_id="cloth-worker"):
        self.client = SpacetimeClient(config or SpacetimeConfig.from_env())
        self.worker_id = _identifier(worker_id, "cloth worker")
        self.run_id = None

    def _row(self, table, field, key):
        # Table and field are internal constants; externally supplied IDs are validated.
        rows = self.client._sql(f"SELECT * FROM {table} WHERE {field} = '{key}'")
        if len(rows) > 1:
            raise SpacetimeConflict("Duplicate cloth journal record")
        return rows[0] if rows else None

    def create(self, run_id, config):
        self.run_id = _identifier(run_id, "cloth run")
        encoded = canonical(config)
        self.client._mutate(
            "create_cloth_channel",
            [self.run_id, self.worker_id, encoded],
            lambda: self._row("cloth_channel", "run_id", self.run_id),
            {"config_json": encoded, "worker_id": self.worker_id},
        )

    def commit(self, tick, packets, state=None):
        if self.run_id is None or type(tick) is not int or not 0 <= tick < 2**32:
            raise ValueError("Cloth journal must be initialized at a valid tick")
        encoded = canonical({"packets": packets, "state": state or {}})
        row = self.client._mutate(
            "commit_cloth_frame",
            [self.run_id, self.worker_id, tick, encoded],
            lambda: self._row("cloth_frame", "key", f"{self.run_id}:{tick}"),
            {"payload_json": encoded},
        )
        # The exact authoritative batch is returned. No local sender fallback.
        return json.loads(row["payload_json"])["packets"]

    def history(self):
        if self.run_id is None:
            raise ValueError("Cloth journal is not initialized")
        rows = self.client._sql(f"SELECT * FROM cloth_frame WHERE run_id = '{self.run_id}'")
        return [
            json.loads(row["payload_json"]) | {"tick": row["tick"]}
            for row in sorted(rows, key=lambda row: row["tick"])
        ]

    @property
    def backend_info(self):
        return {
            "kind": "spacetimedb",
            "durable": True,
            "run_id": self.run_id,
            "index": "private B-tree run/frame and packet identity; no ANN",
            "protocol": "owner-authenticated idempotent commit + durability ACK + exact readback",
            "trusted_worker": True,
            "physics": "single-process simulator",
        }
