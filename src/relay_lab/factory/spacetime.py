"""Durable factory coordination over the existing private SpacetimeDB module.

No local simulation is authorized until confirmed reducer ACK and exact SQL
readback both succeed. Retrying a database request never retries physical work.
"""

from __future__ import annotations

import json
from typing import Any

from relay_lab.spacetime import (
    SpacetimeClient,
    SpacetimeConfig,
    SpacetimeConflict,
    SpacetimeError,
    _identifier,
)

from .coordination import canonical


class SpacetimeCoordinator:
    def __init__(self, config: SpacetimeConfig | None = None):
        self.client = SpacetimeClient(config or SpacetimeConfig.from_env())
        self.run_id: str | None = None
        self.worker_id: str | None = None

    @property
    def backend_info(self) -> dict[str, Any]:
        return {
            **self.client.backend_info,
            "protocol": "factory.v1 CAS command barriers + leases + causal events",
            "trusted_worker": True,
            "durable": True,
        }

    def _run(self) -> dict | None:
        if self.run_id is None:
            raise SpacetimeError("Factory run is not bound")
        rows = self.client._sql(f"SELECT * FROM factory_run WHERE run_id = '{self.run_id}'")
        if len(rows) > 1:
            raise SpacetimeError("Duplicate factory run")
        return rows[0] if rows else None

    def _batch(self, revision: int) -> dict | None:
        self.client._tick(revision)
        if self.run_id is None:
            raise SpacetimeError("Factory run is not bound")
        rows = self.client._sql(
            f"SELECT * FROM factory_batch WHERE key = '{self.run_id}:{revision}'"
        )
        if len(rows) > 1:
            raise SpacetimeError("Duplicate factory batch")
        return rows[0] if rows else None

    def create(
        self,
        run_id: str,
        worker_id: str,
        robots: list[dict],
        jobs: list[dict],
        resources: list[str],
        max_ticks: int = 1000,
        state: dict | None = None,
    ) -> dict:
        self.run_id = _identifier(run_id, "factory run")
        self.worker_id = _identifier(worker_id, "factory worker")
        spec_json = canonical(
            {"robots": robots, "jobs": jobs, "resources": resources, "max_ticks": max_ticks}
        )
        initial = state or {
            "tick": 0,
            "jobs": {j["job_id"]: dict(j) for j in jobs},
            "success": False,
        }
        self.client._mutate(
            "create_factory_run",
            [run_id, worker_id, spec_json, canonical(initial)],
            self._run,
            {"run_id": run_id, "worker_id": worker_id, "spec_json": spec_json},
        )
        return self.snapshot()

    def prepare(
        self, revision: int, commands: list[dict], messages: list[dict] | None = None
    ) -> list[dict]:
        self.client._tick(revision)
        encoded = canonical({"commands": commands, "messages": messages or []})
        row = self.client._mutate(
            "prepare_factory_batch",
            [self.run_id, self.worker_id, revision, encoded],
            lambda: self._batch(revision),
            {"prepare_json": encoded},
        )
        if row["result_json"]:
            raise SpacetimeConflict("Committed factory batch cannot authorize physics twice")
        # Execute the authoritative readback, not a mutable caller-owned list.
        return json.loads(row["prepare_json"])["commands"]

    def commit(
        self, revision: int, state: dict, outcomes: list[dict], messages: list[dict] | None = None
    ) -> dict:
        self.client._tick(revision)
        encoded = canonical({"state": state, "outcomes": outcomes, "messages": messages or []})
        self.client._mutate(
            "commit_factory_batch",
            [self.run_id, self.worker_id, revision, encoded],
            lambda: self._batch(revision),
            {"result_json": encoded},
        )
        return self.snapshot()

    def snapshot(self) -> dict:
        row = self._run()
        if row is None:
            raise SpacetimeError("Factory run not found")
        locks = self.client._sql(f"SELECT * FROM factory_resource WHERE run_id = '{self.run_id}'")
        return {
            "run_id": row["run_id"],
            "worker_id": row["worker_id"],
            "revision": row["revision"],
            "tick": row["tick"],
            "phase": row["phase"],
            "status": row["status"],
            "spec": json.loads(row["spec_json"]),
            "state": json.loads(row["state_json"]),
            "resources": {
                lock["resource_id"]: {
                    "holder": lock["holder"],
                    "command_id": lock["command_id"],
                    "expires_tick": lock["expires_tick"],
                }
                for lock in locks
            },
        }

    def abort(self, reason: str) -> dict:
        if not isinstance(reason, str) or not 0 < len(reason.strip()) <= 2048:
            raise ValueError("Invalid abort reason")
        self.client._mutate(
            "abort_factory_run",
            [self.run_id, self.worker_id, reason],
            self._run,
            {"status": "failed", "phase": "terminal"},
        )
        return self.snapshot()

    def pending(self, recipient: str) -> list[dict]:
        _identifier(recipient, "recipient")
        run = self._run()
        if run is None:
            raise SpacetimeError("Factory run not found")
        rows = self.client._sql(f"SELECT * FROM factory_event WHERE run_id = '{self.run_id}'")
        rows = [
            r
            for r in rows
            if r["recipient"] in (recipient, "broadcast")
            and r["sender"] != recipient
            and not r["acknowledged"]
            and r["expires_tick"] > run["tick"]
            and json.loads(r["payload_json"])["kind"] != "ack"
        ]
        rows.sort(
            key=lambda r: (
                -(r["priority"] + (run["tick"] - r["created_tick"]) // 4),
                r["created_tick"],
                r["message_id"],
            )
        )
        return [json.loads(r["payload_json"]) for r in rows]
