"""Durable, explicit approvals for exact Studio model configurations.

Roles originate in the server's authenticated session. The registry never
infers an approval from an existing workspace, model family, or file name.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path


class RegistryConflict(ValueError):
    """The caller is approving a superseded model record."""


def _locked(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return run


def _actor(actor, *, manager=False):
    if (
        not isinstance(actor, dict)
        or actor.get("active") is not True
        or actor.get("role") not in ("manager", "employee")
        or not isinstance(actor.get("id"), str)
        or not actor["id"]
    ):
        raise PermissionError("An active authenticated account is required")
    if manager and actor["role"] != "manager":
        raise PermissionError("Only managers can register, release, or retire models")
    return actor


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _text(value, label, limit, *, empty=False):
    if (
        not isinstance(value, str)
        or len(value) > limit
        or (not empty and not value.strip())
        or any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"{label} must be {'0' if empty else '1'}–{limit} plain text characters")
    return value.strip()


def _identity(spec):
    return "mdl_" + hashlib.sha256(_json(spec).encode()).hexdigest()


def _workspace(value, *, complete=False):
    # Local import keeps service -> governance integration free of import cycles.
    from .service import validate_workspace

    return validate_workspace(value, complete=complete)


def _spec(value):
    return _workspace(
        {
            "version": 1,
            "name": "Model validation",
            "objective": "Validate an exact released model configuration.",
            "mode": "mediated",
            "direction": "both",
            "agents": [value],
        }
    )["agents"][0]


def _record(row):
    return {
        "id": row["id"],
        "spec": json.loads(row["spec_json"]),
        "status": row["status"],
        "revision": row["revision"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_by": row["updated_by"],
        "updated_at": row["updated_at"],
        "reason": row["reason"],
    }


class ModelRegistry:
    def __init__(self, storage):
        storage = Path(storage)
        storage.mkdir(parents=True, exist_ok=True)
        self.path = storage / "model-governance.sqlite3"
        # The server holds this across authorization + inference, so retirement
        # cannot overtake an exchange that has already been authorized.
        self.lock = threading.RLock()
        with self._db(write=True) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS governance_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                )"""
            )
            db.execute(
                "INSERT OR IGNORE INTO governance_metadata VALUES ('registry_id', ?)",
                ("registry_" + uuid.uuid4().hex,),
            )
            self.registry_id = db.execute(
                "SELECT value FROM governance_metadata WHERE key='registry_id'"
            ).fetchone()["value"]
            db.execute(
                """CREATE TABLE IF NOT EXISTS model_release (
                    id TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('draft', 'released', 'retired')),
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    created_by TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL, updated_at TEXT NOT NULL,
                    reason TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS model_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_id TEXT NOT NULL, username TEXT NOT NULL, role TEXT NOT NULL,
                    action TEXT NOT NULL, model_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, previous_status TEXT,
                    status TEXT NOT NULL, reason TEXT NOT NULL,
                    created_at TEXT NOT NULL, spec_json TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES model_release(id)
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS model_change_request (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK(kind IN ('add', 'edit', 'release', 'retire')),
                    message TEXT NOT NULL, model_id TEXT REFERENCES model_release(id),
                    status TEXT NOT NULL CHECK(status IN ('pending', 'resolved', 'declined')),
                    created_by TEXT NOT NULL, username TEXT NOT NULL, created_at TEXT NOT NULL,
                    resolved_by TEXT, resolved_at TEXT, reason TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS request_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL REFERENCES model_change_request(id),
                    actor_id TEXT NOT NULL, username TEXT NOT NULL, role TEXT NOT NULL,
                    action TEXT NOT NULL, status TEXT NOT NULL,
                    reason TEXT NOT NULL, created_at TEXT NOT NULL
                )"""
            )

    @contextmanager
    def _db(self, *, write=False):
        # Each operation owns its connection, including concurrent HTTP requests.
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _audit(db, actor, record, action, previous_status):
        db.execute(
            """INSERT INTO model_audit
               (actor_id, username, role, action, model_id, revision, previous_status,
                status, reason, created_at, spec_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                actor["id"],
                actor.get("username", ""),
                actor["role"],
                action,
                record["id"],
                record["revision"],
                previous_status,
                record["status"],
                record["reason"],
                record["updated_at"],
                _json(record["spec"]),
            ),
        )

    @_locked
    def sync_workspace(self, actor, workspace):
        """Register unseen exact specs as drafts; retain previous versions."""
        _actor(actor, manager=True)
        workspace = _workspace(workspace)
        records = []
        with self._db(write=True) as db:
            for spec in workspace["agents"]:
                model_id = _identity(spec)
                row = db.execute("SELECT * FROM model_release WHERE id = ?", (model_id,)).fetchone()
                if row is None:
                    now = datetime.now(timezone.utc).isoformat()
                    db.execute(
                        """INSERT INTO model_release VALUES (?, ?, 'draft', 1, ?, ?, ?, ?, '')""",
                        (model_id, _json(spec), actor["id"], now, actor["id"], now),
                    )
                    row = db.execute(
                        "SELECT * FROM model_release WHERE id = ?", (model_id,)
                    ).fetchone()
                    self._audit(db, actor, _record(row), "register", None)
                records.append(_record(row))
        return {"models": records}

    @_locked
    def catalog(self, actor):
        _actor(actor)
        with self._db() as db:
            rows = db.execute("SELECT * FROM model_release ORDER BY updated_at DESC, id").fetchall()
            return {"models": [_record(row) for row in rows]}

    @_locked
    def transition(self, actor, model_id, status, expected_revision, reason=""):
        _actor(actor, manager=True)
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("A model record ID is required")
        if status not in ("released", "retired"):
            raise ValueError("A model can only be released or retired")
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("A positive expected model revision is required")
        if not isinstance(reason, str) or len(reason) > 1000:
            raise ValueError("Release reason must be at most 1000 characters")
        with self._db(write=True) as db:
            row = db.execute("SELECT * FROM model_release WHERE id = ?", (model_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown model record")
            record = _record(row)
            if record["revision"] != expected_revision:
                raise RegistryConflict(
                    "Model revision changed; refresh the catalog before approval"
                )
            normalized = _spec(record["spec"])
            if normalized != record["spec"] or _identity(normalized) != model_id:
                raise ValueError("Stored model configuration failed identity validation")
            if (
                status == "released"
                and normalized["source"]["kind"] == "huggingface"
                and not normalized["source"]["revision"]
            ):
                raise ValueError("Pin a Hugging Face model revision before releasing it")
            if record["status"] == status:
                return record
            previous_status = record["status"]
            record.update(
                status=status,
                revision=expected_revision + 1,
                updated_by=actor["id"],
                updated_at=datetime.now(timezone.utc).isoformat(),
                reason=reason.strip(),
            )
            db.execute(
                """UPDATE model_release SET status=?, revision=?, updated_by=?,
                   updated_at=?, reason=? WHERE id=? AND revision=?""",
                (
                    status,
                    record["revision"],
                    record["updated_by"],
                    record["updated_at"],
                    record["reason"],
                    model_id,
                    expected_revision,
                ),
            )
            self._audit(
                db, actor, record, "release" if status == "released" else "retire", previous_status
            )
            return record

    @staticmethod
    def _released(db, spec, *, revision=None):
        model_id = _identity(spec)
        row = db.execute("SELECT * FROM model_release WHERE id = ?", (model_id,)).fetchone()
        if (
            row is None
            or row["status"] != "released"
            or json.loads(row["spec_json"]) != spec
            or (revision is not None and row["revision"] != revision)
        ):
            raise PermissionError(
                "This exact model configuration has no current release; ask a manager to approve it"
            )
        return {"id": model_id, "revision": row["revision"], "spec": spec}

    def _check_identity(self, db):
        current = db.execute(
            "SELECT value FROM governance_metadata WHERE key='registry_id'"
        ).fetchone()
        if current is None or current["value"] != self.registry_id:
            raise PermissionError("Model registry was replaced; restart and create a new session")

    @_locked
    def authorize_workspace(self, actor, workspace):
        """Bind a session to current releases. Managers have no draft bypass."""
        _actor(actor)
        workspace = _workspace(workspace, complete=True)
        with self._db() as db:
            self._check_identity(db)
            models = [self._released(db, spec) for spec in workspace["agents"]]
        return {"version": 1, "registry_id": self.registry_id, "models": models}

    @_locked
    def authorize_snapshot(self, actor, snapshot):
        """Recheck every exchange/resume; retirement and re-release revoke old sessions."""
        _actor(actor)
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("version") != 1
            or snapshot.get("registry_id") != self.registry_id
            or not isinstance(snapshot.get("models"), list)
            or len(snapshot["models"]) != 2
        ):
            raise PermissionError(
                "Session has no valid model release snapshot; create a new session"
            )
        models, seen = [], set()
        with self._db() as db:
            self._check_identity(db)
            for item in snapshot["models"]:
                if not isinstance(item, dict) or type(item.get("revision")) is not int:
                    raise PermissionError("Session model release snapshot is invalid")
                try:
                    spec = _spec(item.get("spec"))
                except (ValueError, TypeError, KeyError) as exc:
                    raise PermissionError("Session model release snapshot is invalid") from exc
                if (
                    spec != item.get("spec")
                    or item.get("id") != _identity(spec)
                    or spec["id"] in seen
                    or item["revision"] < 1
                ):
                    raise PermissionError("Session model release snapshot is invalid")
                seen.add(spec["id"])
                models.append(self._released(db, spec, revision=item["revision"]))
        return {"version": 1, "registry_id": self.registry_id, "models": models}

    @_locked
    def has_release(self, workspace):
        """Readiness hint only; execution must use authenticated authorization."""
        try:
            workspace = _workspace(workspace, complete=True)
            with self._db() as db:
                self._check_identity(db)
                for spec in workspace["agents"]:
                    self._released(db, spec)
        except (ValueError, PermissionError):
            return False
        return True

    @staticmethod
    def _request_audit(db, actor, record, action):
        db.execute(
            """INSERT INTO request_audit
               (request_id, actor_id, username, role, action, status, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["id"],
                actor["id"],
                actor.get("username", ""),
                actor["role"],
                action,
                record["status"],
                record["reason"],
                record["resolved_at"] or record["created_at"],
            ),
        )

    @_locked
    def submit_request(self, actor, kind, message, model_id=None):
        """Create a plain-text proposal; it never changes a model's approval."""
        _actor(actor)
        if not isinstance(kind, str) or kind not in ("add", "edit", "release", "retire"):
            raise ValueError("Choose an add, edit, release, or retire request")
        message = _text(message, "Request", 2000)
        if model_id is not None and (not isinstance(model_id, str) or not model_id):
            raise ValueError("Choose an existing model record or omit the model ID")
        with self._db(write=True) as db:
            if (
                model_id is not None
                and not db.execute("SELECT 1 FROM model_release WHERE id=?", (model_id,)).fetchone()
            ):
                raise ValueError("Unknown model record")
            record = {
                "id": "req_" + uuid.uuid4().hex,
                "kind": kind,
                "message": message,
                "model_id": model_id,
                "status": "pending",
                "created_by": actor["id"],
                "username": actor.get("username", ""),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "resolved_by": None,
                "resolved_at": None,
                "reason": "",
            }
            db.execute(
                """INSERT INTO model_change_request
                   (id, kind, message, model_id, status, created_by, username, created_at,
                    resolved_by, resolved_at, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(record.values()),
            )
            self._request_audit(db, actor, record, "request.submitted")
            return record

    @_locked
    def requests(self, actor):
        _actor(actor)
        with self._db() as db:
            condition, params = (
                ("", ())
                if actor["role"] == "manager"
                else (
                    "WHERE created_by=?",
                    (actor["id"],),
                )
            )
            rows = db.execute(
                f"""SELECT * FROM model_change_request {condition}
                    ORDER BY (status='pending') DESC, created_at DESC, id DESC LIMIT 500""",
                params,
            ).fetchall()
            return {"requests": [dict(row) for row in rows]}

    @_locked
    def resolve_request(self, actor, request_id, status, reason=""):
        """Close the proposal; releasing/retiring a model stays an explicit operation."""
        _actor(actor, manager=True)
        if not isinstance(status, str) or status not in ("resolved", "declined"):
            raise ValueError("A request can only be resolved or declined")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A request ID is required")
        reason = _text(reason, "Resolution reason", 1000, empty=True)
        with self._db(write=True) as db:
            row = db.execute(
                "SELECT * FROM model_change_request WHERE id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Unknown model change request")
            record = dict(row)
            if record["status"] != "pending":
                raise RegistryConflict("Request was already resolved; refresh the request queue")
            record.update(
                status=status,
                resolved_by=actor["id"],
                resolved_at=datetime.now(timezone.utc).isoformat(),
                reason=reason,
            )
            db.execute(
                """UPDATE model_change_request SET status=?, resolved_by=?, resolved_at=?,
                   reason=? WHERE id=? AND status='pending'""",
                (status, record["resolved_by"], record["resolved_at"], reason, request_id),
            )
            self._request_audit(db, actor, record, "request." + status)
            return record

    @_locked
    def audit_log(self, actor):
        _actor(actor, manager=True)
        with self._db() as db:
            rows = db.execute("SELECT * FROM model_audit ORDER BY id DESC LIMIT 500").fetchall()
            events = []
            for row in rows:
                event = dict(row)
                event["spec"] = json.loads(event.pop("spec_json"))
                events.append(event)
            for row in db.execute("SELECT * FROM request_audit ORDER BY id DESC LIMIT 500"):
                event = dict(row)
                event["id"] = "request-event-" + str(event["id"])
                event["target"] = event["request_id"]
                events.append(event)
            events.sort(key=lambda item: item["created_at"], reverse=True)
            return {"events": events[:500]}
