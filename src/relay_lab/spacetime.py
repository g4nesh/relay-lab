"""Synchronous, durable coordination through SpacetimeDB's authenticated HTTP API.

This client never executes a simulator action. Mutations are idempotent module
reducers, followed by authoritative readback. A failed or uncertain write stops
the caller instead of falling back to uncoordinated local execution.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_DATABASE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_MAX_RESPONSE = 2 * 1024 * 1024
_MAX_JSON = 64 * 1024


class SpacetimeError(RuntimeError):
    """The coordination operation did not establish its required postcondition."""


class SpacetimeRejected(SpacetimeError):
    """The server rejected an authenticated request."""


class SpacetimeUnavailable(SpacetimeError):
    """The server is unavailable or a mutation's commit status remains uncertain."""


class SpacetimeConflict(SpacetimeError):
    """Stored state disagrees with the exact operation being attempted."""


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must contain 1–128 ASCII letters, digits, '_' or '-'")
    return value


def _json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("Coordination payload must be a JSON object")
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result.encode("utf-8")) > _MAX_JSON:
        raise ValueError("Coordination payload exceeds 64 KiB")
    return result


def _read_token(path: Path) -> str:
    """Read a user-owned private regular file without following its final symlink."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise ValueError("Cannot read private Spacetime token file; run Spacetime setup") from None
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError("Spacetime token must be a user-owned regular file with mode 0600")
        token = stream.read(16385).strip()
    if not token or len(token) > 16384 or any(char.isspace() for char in token):
        raise ValueError("Invalid Spacetime bearer token")
    return token


@dataclass(frozen=True)
class SpacetimeConfig:
    server: str = "http://127.0.0.1:3000"
    database: str = "relay-lab"
    token: str = field(default="", repr=False, compare=False)
    timeout: float = 5.0
    attempts: int = 2

    def __post_init__(self) -> None:
        parts = urlsplit(self.server)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Spacetime server must be an HTTP(S) origin without credentials")
        try:
            _ = parts.port
        except ValueError:
            raise ValueError("Spacetime server has an invalid port") from None
        try:
            loopback = ipaddress.ip_address(parts.hostname).is_loopback
        except ValueError:
            loopback = parts.hostname == "localhost"
        if parts.scheme == "http" and not loopback:
            raise ValueError("Remote Spacetime servers require HTTPS")
        if not isinstance(self.database, str) or not _DATABASE.fullmatch(self.database):
            raise ValueError("Spacetime database must be a lowercase name or hex identity")
        if (
            not isinstance(self.token, str)
            or not self.token
            or len(self.token) > 16384
            or any(c.isspace() for c in self.token)
        ):
            raise ValueError("A Spacetime owner bearer token is required")
        if (
            not isinstance(self.timeout, (int, float))
            or not 0 < self.timeout <= 30
            or type(self.attempts) is not int
            or not 1 <= self.attempts <= 3
        ):
            raise ValueError("Spacetime timeout must be 0–30 seconds and attempts 1–3")

    @classmethod
    def from_env(cls, *, token_file: Path | None = None) -> SpacetimeConfig:
        token = os.environ.get("RELAY_SPACETIME_TOKEN", "")
        if not token:
            token = _read_token(
                token_file
                or Path(os.environ.get("RELAY_SPACETIME_TOKEN_FILE", ".relay-lab/spacetime-token"))
            )
        return cls(
            server=os.environ.get("RELAY_SPACETIME_SERVER", "http://127.0.0.1:3000"),
            database=os.environ.get("RELAY_SPACETIME_DATABASE", "relay-lab"),
            token=token,
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A redirect must never forward this database owner's credential.
        return None


class SpacetimeClient:
    def __init__(self, config: SpacetimeConfig):
        self.config = config
        self._base = f"{config.server.rstrip('/')}/v1/database/{quote(config.database, safe='')}"
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, endpoint: str, body: bytes, *, content_type: str) -> bytes:
        request = Request(
            self._base + endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": content_type,
            },
        )
        try:
            with self._opener.open(request, timeout=self.config.timeout) as response:
                result = response.read(_MAX_RESPONSE + 1)
        except HTTPError as exc:
            # Do not echo response bodies, URLs or request headers: these can
            # contain credentials or arbitrary server-controlled content.
            if exc.code < 500 and exc.code not in (408, 429):
                raise SpacetimeRejected(f"Spacetime request rejected (HTTP {exc.code})") from None
            raise SpacetimeUnavailable(f"Spacetime request failed (HTTP {exc.code})") from None
        except (URLError, TimeoutError, ConnectionError, OSError):
            raise SpacetimeUnavailable(
                "Spacetime request did not receive a confirmed response"
            ) from None
        if len(result) > _MAX_RESPONSE:
            raise SpacetimeError("Spacetime response exceeds 2 MiB")
        return result

    def _sql(self, query: str) -> list[dict[str, Any]]:
        for attempt in range(self.config.attempts):
            try:
                raw = self._request("/sql", query.encode(), content_type="text/plain")
                break
            except SpacetimeUnavailable:
                if attempt + 1 == self.config.attempts:
                    raise
                time.sleep(0.05)
        try:
            results = json.loads(raw)
            if not isinstance(results, list) or len(results) != 1:
                raise ValueError
            result = results[0]
            names = []
            for element in result["schema"]["elements"]:
                name = element["name"]
                if isinstance(name, dict):
                    name = name.get("some")
                if not isinstance(name, str) or not name or name in names:
                    raise ValueError
                names.append(name)
            rows = result["rows"]
            if not isinstance(rows, list):
                raise TypeError
            if any(not isinstance(row, list) or len(row) != len(names) for row in rows):
                raise ValueError
            return [dict(zip(names, row, strict=True)) for row in rows]
        except (ValueError, KeyError, TypeError):
            raise SpacetimeError("Invalid Spacetime SQL response") from None

    def _mutate(
        self,
        reducer: str,
        args: list[Any],
        readback: Callable[[], dict[str, Any] | None],
        expected: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Retry only a byte-identical reducer; never retry physical execution.

        Even if readback finds a write after a lost response, re-call the
        idempotent reducer with ``confirmed=true`` to obtain durability ACK.
        SQL visibility by itself is not treated as that acknowledgement.
        """
        body = json.dumps(args, separators=(",", ":"), allow_nan=False).encode()
        for attempt in range(self.config.attempts):
            try:
                self._request(
                    f"/call/{reducer}?confirmed=true", body, content_type="application/json"
                )
            except SpacetimeUnavailable:
                # Inspect before retrying to catch a conflicting concurrent
                # write, but a failed read also cannot grant permission to act.
                try:
                    row = readback()
                except SpacetimeUnavailable:
                    row = None
                if row is not None:
                    self._verify(row, expected)
                if attempt + 1 == self.config.attempts:
                    raise SpacetimeUnavailable(
                        f"Spacetime {reducer} commit status is uncertain; stop execution"
                    ) from None
                time.sleep(0.05)
                continue
            row = readback()
            if row is None:
                raise SpacetimeConflict(f"Spacetime {reducer} has no committed record")
            self._verify(row, expected)
            return row
        raise AssertionError("Unreachable retry loop")

    @staticmethod
    def _verify(row: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
        if any(row.get(key) != value for key, value in expected.items()):
            raise SpacetimeConflict("Spacetime stored state conflicts with the requested operation")

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task_id = _identifier(task_id, "task_id")
        rows = self._sql(f"SELECT * FROM task WHERE task_id = '{task_id}'")
        if len(rows) > 1:
            raise SpacetimeError("Spacetime returned duplicate task records")
        return rows[0] if rows else None

    def _frame(self, table: str, task_id: str, tick: int) -> dict[str, Any] | None:
        task_id = _identifier(task_id, "task_id")
        self._tick(tick)
        if table not in ("plan_frame", "action_frame", "feedback_frame"):
            raise ValueError("Unknown coordination table")
        rows = self._sql(f"SELECT * FROM {table} WHERE frame_id = '{task_id}:{tick}'")
        if len(rows) > 1:
            raise SpacetimeError("Spacetime returned duplicate frame records")
        return rows[0] if rows else None

    @staticmethod
    def _tick(tick: int) -> None:
        if type(tick) is not int or not 0 <= tick < 2**32:
            raise ValueError("tick must be an unsigned 32-bit integer")

    def get_plan(self, task_id: str, tick: int) -> dict[str, Any] | None:
        return self._frame("plan_frame", task_id, tick)

    def get_action(self, task_id: str, tick: int) -> dict[str, Any] | None:
        return self._frame("action_frame", task_id, tick)

    def get_feedback(self, task_id: str, tick: int) -> dict[str, Any] | None:
        return self._frame("feedback_frame", task_id, tick)

    def create_task(
        self, task_id: str, intent: str, spec: Mapping[str, Any], max_steps: int
    ) -> dict[str, Any]:
        task_id = _identifier(task_id, "task_id")
        if not isinstance(intent, str) or not intent.strip() or len(intent) > 4096:
            raise ValueError("Task intent must contain 1–4096 characters")
        if type(max_steps) is not int or not 1 <= max_steps <= 100_000:
            raise ValueError("max_steps must be between 1 and 100000")
        spec_json = _json(spec)
        return self._mutate(
            "create_task",
            [task_id, intent, spec_json, max_steps],
            lambda: self.get_task(task_id),
            {"task_id": task_id, "intent": intent, "spec_json": spec_json, "max_steps": max_steps},
        )

    def claim_task(self, task_id: str, worker_id: str) -> dict[str, Any]:
        task_id = _identifier(task_id, "task_id")
        worker_id = _identifier(worker_id, "worker_id")
        return self._mutate(
            "claim_task",
            [task_id, worker_id],
            lambda: self.get_task(task_id),
            {"task_id": task_id, "worker_id": worker_id},
        )

    def _write_frame(
        self,
        reducer: str,
        table: str,
        task_id: str,
        worker_id: str,
        tick: int,
        picker: Mapping[str, Any],
        packer: Mapping[str, Any],
        outcome: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_id = _identifier(task_id, "task_id")
        worker_id = _identifier(worker_id, "worker_id")
        self._tick(tick)
        picker_json, packer_json = _json(picker), _json(packer)
        is_feedback = table == "feedback_frame"
        expected = {
            "frame_id": f"{task_id}:{tick}",
            "task_id": task_id,
            "worker_id": worker_id,
            "tick": tick,
            "picker_report_json" if is_feedback else "picker_json": picker_json,
            "packer_report_json" if is_feedback else "packer_json": packer_json,
        }
        args = [task_id, worker_id, tick, picker_json, packer_json]
        if outcome is not None:
            outcome_json = _json(outcome)
            args.append(outcome_json)
            expected["outcome_json"] = outcome_json
        return self._mutate(reducer, args, lambda: self._frame(table, task_id, tick), expected)

    def plan_tick(
        self,
        task_id: str,
        worker_id: str,
        tick: int,
        picker: Mapping[str, Any],
        packer: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Store and read back both robot plans before model inference."""
        return self._write_frame(
            "plan_tick", "plan_frame", task_id, worker_id, tick, picker, packer
        )

    def prepare_tick(
        self,
        task_id: str,
        worker_id: str,
        tick: int,
        picker: Mapping[str, Any],
        packer: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist both proposed actions before the caller's one simulator step."""
        return self._write_frame(
            "prepare_tick", "action_frame", task_id, worker_id, tick, picker, packer
        )

    def commit_tick(
        self,
        task_id: str,
        worker_id: str,
        tick: int,
        picker: Mapping[str, Any],
        packer: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically commit observed reports and simulator outcome for both robots."""
        return self._write_frame(
            "commit_tick", "feedback_frame", task_id, worker_id, tick, picker, packer, outcome
        )

    def task_context(self, task_id: str) -> dict[str, dict[str, Any]]:
        """Read each robot's own last committed feedback, never its peer's private state."""
        task_id = _identifier(task_id, "task_id")
        rows = self._sql(f"SELECT * FROM assignment WHERE task_id = '{task_id}'")
        result = {}
        try:
            for row in rows:
                agent = row["agent_id"]
                if agent not in ("picker", "packer") or agent in result:
                    raise ValueError
                text, critical = "", False
                if row["last_feedback_json"]:
                    feedback = json.loads(row["last_feedback_json"])
                    text, critical = feedback["text"], feedback["critical"]
                    if not isinstance(text, str) or type(critical) is not bool:
                        raise ValueError
                result[agent] = {
                    "text": text,
                    "critical": critical,
                    "tick": row["last_feedback_tick"],
                }
            if set(result) != {"picker", "packer"}:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise SpacetimeError("Spacetime task assignments or feedback are incomplete") from None
        return result

    def abort_task(self, task_id: str, worker_id: str, reason: str) -> dict[str, Any]:
        task_id = _identifier(task_id, "task_id")
        worker_id = _identifier(worker_id, "worker_id")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2048:
            raise ValueError("Abort reason must contain 1–2048 characters")
        return self._mutate(
            "abort_task",
            [task_id, worker_id, reason],
            lambda: self.get_task(task_id),
            {"task_id": task_id, "worker_id": worker_id, "status": "failed", "reason": reason},
        )

    @property
    def backend_info(self) -> dict[str, str]:
        """Non-secret provenance suitable for a rollout's coordination.json."""
        return {
            "backend": "spacetimedb",
            "server": self.config.server,
            "database": self.config.database,
            "protocol": "authenticated HTTP reducers and SQL readback",
            "durability": "confirmed=true",
        }
