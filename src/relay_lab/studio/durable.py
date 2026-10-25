"""Spacetime-owned control state with disposable, private SQLite execution caches.

API requests restore the committed checkpoint, acquire a database lease, and only
return a response after a confirmed checkpoint commit. Task reducer calls carry
the same lease, fencing a stalled or overlapping instance out of task writes.
No external blob store, local-volume durability, or background writeback is used.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import zlib
from contextlib import contextmanager
from pathlib import Path

from relay_lab.factory.spacetime import SpacetimeCoordinator, _MeasuredClient
from relay_lab.spacetime import (
    SpacetimeClient,
    SpacetimeConflict,
    SpacetimeError,
    SpacetimeUnavailable,
)

FILES = frozenset(
    {
        "access.sqlite3",
        "model-governance.sqlite3",
        "workers.sqlite3",
        "telemetry.sqlite3",
        "coordination.sqlite3",
        "workspace.json",
        "setup-code",
    }
)
MAX_RAW = 8 * 1024 * 1024
MAX_ENCODED = 1024 * 1024
FENCED = frozenset(
    {
        "create_factory_run",
        "prepare_factory_batch",
        "commit_factory_batch",
        "abort_factory_run",
    }
)

COORDINATION_FENCED = frozenset({
    "studio_heartbeat", "studio_revoke_worker", "studio_sync_task",
    "studio_record_metric", "studio_configure_job", "studio_run_job",
})


class _FencedClient(_MeasuredClient):
    def __init__(self, config, state):
        super().__init__(config)
        self.state = state

    def _request(self, endpoint, body, *, content_type):
        if endpoint.startswith("/call/"):
            name = endpoint.split("/call/", 1)[1].split("?", 1)[0]
            if name not in FENCED | COORDINATION_FENCED or not self.state.holder:
                raise SpacetimeConflict(
                    "A current Studio request is required for task writes"
                )
            target = ("studio_fenced_" + name.removeprefix("studio_")) if name in COORDINATION_FENCED else "studio_" + name
            endpoint = f"/call/{target}?confirmed=true"
            body = json.dumps([self.state.holder, *json.loads(body)]).encode()
        return super()._request(endpoint, body, content_type=content_type)


class DurableState:
    def __init__(self, storage, config, *, client=None):
        self.storage = Path(storage)
        self.storage.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.client = client or SpacetimeClient(config)
        self.config = config
        self.lock = threading.RLock()
        self.local = threading.local()
        self.revision = None

    @property
    def holder(self):
        return getattr(self.local, "holder", None)

    def coordinator(self):
        coordinator = SpacetimeCoordinator(self.config)
        coordinator.client = _FencedClient(self.config, self)
        return coordinator

    def row(self):
        rows = self.client._sql("SELECT * FROM studio_state WHERE id = 0")
        if len(rows) > 1:
            raise SpacetimeConflict("Duplicate Studio state")
        return rows[0] if rows else None

    def _call(self, reducer, args):
        # Retry the identical operation only, to obtain an actual durability ACK.
        body = json.dumps(args, separators=(",", ":"), allow_nan=False).encode()
        for attempt in range(2):
            try:
                self.client._request(
                    f"/call/{reducer}?confirmed=true",
                    body,
                    content_type="application/json",
                )
                return
            except SpacetimeUnavailable:
                if attempt:
                    raise

    def _capture(self):
        data = {}
        total = 0
        for name in sorted(FILES):
            path = self.storage / name
            if path.is_symlink():
                raise SpacetimeError("Refusing a symlink in Studio state")
            if not path.exists():
                continue
            # All store transactions/connections are closed at this request
            # boundary; SQLite uses its default DELETE journal mode.
            if path.stat().st_size > MAX_RAW:
                raise SpacetimeError("Studio state exceeded the deployment limit")
            raw = path.read_bytes()
            total += len(raw)
            if total > MAX_RAW:
                raise SpacetimeError("Studio state exceeded the deployment limit")
            data[name] = base64.b64encode(raw).decode("ascii")
        raw = json.dumps({"version": 1, "files": data}, sort_keys=True).encode()
        if len(raw) > MAX_RAW * 2:
            raise SpacetimeError("Studio state exceeded the deployment limit")
        encoded = base64.b64encode(zlib.compress(raw, 6)).decode("ascii")
        if len(encoded) > MAX_ENCODED:
            raise SpacetimeError("Studio checkpoint exceeded the deployment limit")
        return encoded

    def _restore(self, payload):
        data = {}
        if payload:
            try:
                if not isinstance(payload, str) or len(payload) > MAX_ENCODED:
                    raise ValueError
                compressed = base64.b64decode(payload, validate=True)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed, MAX_RAW * 2 + 1)
                if len(raw) > MAX_RAW * 2 or not decoder.eof or decoder.unused_data:
                    raise ValueError
                value = json.loads(raw)
                if value.get("version") != 1 or not isinstance(
                    value.get("files"), dict
                ):
                    raise ValueError
                if set(value["files"]) - FILES:
                    raise ValueError
                data = {
                    name: base64.b64decode(content, validate=True)
                    for name, content in value["files"].items()
                }
                if sum(map(len, data.values())) > MAX_RAW:
                    raise ValueError
                for name, content in data.items():
                    if name.endswith(".sqlite3") and not content.startswith(
                        b"SQLite format 3\x00"
                    ):
                        raise ValueError
            except (ValueError, TypeError, KeyError, zlib.error):
                raise SpacetimeError(
                    "Invalid durable Studio checkpoint; refusing a blank reset"
                ) from None
        # Validate the entire payload before replacing any disposable local cache.
        for name in FILES:
            path = self.storage / name
            if path.is_symlink():
                raise SpacetimeError("Refusing a symlink in Studio state")
            for suffix in ("-wal", "-shm", "-journal"):
                (self.storage / (name + suffix)).unlink(missing_ok=True)
            if name not in data:
                path.unlink(missing_ok=True)
                continue
            tmp = self.storage / (name + ".restore")
            fd = os.open(
                tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(data[name])
            tmp.replace(path)

    @contextmanager
    def request(self):
        # One request in this process; the database lease also serializes a
        # second Render instance during a rolling deployment or cold restart.
        with self.lock:
            if self.holder:
                raise SpacetimeConflict("Nested Studio state request")
            holder = "req-" + secrets.token_hex(20)
            self.local.holder = holder
            try:
                self._call("studio_acquire", [holder])
                row = self.row()
                if not row:
                    raise SpacetimeConflict("Studio lease could not be confirmed")
                self._restore(row["payload"])
                self.revision = row["revision"]
                yield
                payload = self._capture()
                if payload == row["payload"]:
                    # Even a read verifies its lease before returning. This
                    # prevents a superseded instance returning stale auth data.
                    self._call("studio_finish_read", [holder, self.revision])
                else:
                    operation = "commit-" + secrets.token_hex(20)
                    self._call(
                        "studio_commit_state",
                        [holder, self.revision, payload, operation],
                    )
                    # The confirmed reducer transaction is authoritative. A new
                    # request can start immediately after it releases the lease.
                    self.revision += 1
            except BaseException:
                try:
                    self._call("studio_release", [holder])
                except SpacetimeError:
                    pass  # The bounded lease expires; never pretend persistence succeeded.
                raise
            finally:
                self.local.holder = None

    def probe(self):
        rows = self.client._sql("SELECT revision FROM studio_state WHERE id = 0")
        if len(rows) != 1 or rows[0]["revision"] < 1:
            raise SpacetimeUnavailable("Studio has no initialized durable state")
