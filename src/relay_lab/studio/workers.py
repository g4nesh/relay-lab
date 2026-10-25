"""Revocable outbound agent connections bound to one approved model revision."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .access import AuthenticationError
from .service import ARMS, bounded

TOKEN_TTL = 30 * 24 * 60 * 60
LEASE_SECONDS = 45


class WorkerStore:
    def __init__(self, storage, access, registry, service):
        self.path = Path(storage) / "workers.sqlite3"
        self.access, self.registry, self.service = access, registry, service
        self.coordination = None
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runtime_worker (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, agent_id TEXT NOT NULL,
                model_id TEXT NOT NULL, release_revision INTEGER NOT NULL,
                registry_id TEXT NOT NULL, issuer_id TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE, created_at REAL NOT NULL,
                expires_at REAL NOT NULL, last_seen REAL, active INTEGER NOT NULL,
                controller_label TEXT NOT NULL, execution_kind TEXT NOT NULL
            )""")
        self.path.chmod(0o600)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _issuer(self, worker):
        return self.access.authorize_actor({"id": worker["issuer_id"]}, "models.manage")

    def _approved(self, worker, actor):
        if worker["registry_id"] != self.registry.registry_id:
            raise PermissionError("This agent connection belongs to another workspace")
        models = self.registry.catalog(actor)["models"]
        if not any(
            m["id"] == worker["model_id"]
            and m["revision"] == worker["release_revision"]
            and m["status"] == "released"
            and m["spec"]["id"] == worker["agent_id"]
            for m in models
        ):
            raise PermissionError("This agent's model release is no longer approved")

    def _public(self, worker):
        value = {
            key: worker[key]
            for key in (
                "id",
                "name",
                "agent_id",
                "model_id",
                "release_revision",
                "created_at",
                "expires_at",
                "last_seen",
                "controller_label",
                "execution_kind",
            )
        }
        value["active"] = bool(worker["active"])
        value["lease_seconds"] = LEASE_SECONDS
        if not worker["active"]:
            status = "revoked"
        elif worker["expires_at"] <= time.time():
            status = "expired"
        else:
            try:
                self._approved(worker, self._issuer(worker))
                status = (
                    "never_connected"
                    if worker["last_seen"] is None
                    else "online"
                    if time.time() - worker["last_seen"] < LEASE_SECONDS
                    else "offline"
                )
            except PermissionError:
                status = "release_blocked"
        value["status"] = status
        return value

    def list(self, actor):
        actor = self.access.authorize_actor(actor)
        catalog = self.registry.catalog(actor)["models"]
        agents = []
        for spec in self.service.workspace()["agents"]:
            approved = next(
                (m for m in catalog if m["spec"] == spec and m["status"] == "released"), None
            )
            agents.append(
                {
                    "id": spec["id"],
                    "name": spec["name"],
                    "release_ready": approved is not None,
                    "model_id": approved["id"] if approved else None,
                    "release_revision": approved["revision"] if approved else None,
                }
            )
        with self._db() as db:
            rows = db.execute(
                "SELECT * FROM runtime_worker ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return {
            "workers": [self._public(dict(r)) for r in rows],
            "agents": agents,
            "poll_interval_seconds": 3,
            "limit": 100,
        }

    def create(self, actor, value):
        actor = self.access.authorize_actor(actor, "models.manage")
        if not isinstance(value, dict):
            raise TypeError("Expected connection details")
        agent_id = value.get("agent_id")
        if agent_id not in ARMS:
            raise ValueError("Select an existing agent")
        kind = value.get("execution_kind", "simulation")
        if kind not in ("simulation", "observation"):
            raise ValueError("Choose simulation or observation runtime")
        name = bounded(value.get("name"), "Connection name", 80)
        with self.registry.lock:
            release = self.registry.authorize_workspace(actor, self.service.workspace())
            model = next(m for m in release["models"] if m["spec"]["id"] == agent_id)
            token = "relay_worker_" + secrets.token_urlsafe(40)
            worker = {
                "id": "worker-" + uuid.uuid4().hex[:16],
                "name": name,
                "agent_id": agent_id,
                "model_id": model["id"],
                "release_revision": model["revision"],
                "registry_id": self.registry.registry_id,
                "issuer_id": actor["id"],
                "token_hash": hashlib.sha256(token.encode()).hexdigest(),
                "created_at": time.time(),
                "expires_at": time.time() + TOKEN_TTL,
                "last_seen": None,
                "active": 1,
                "controller_label": "",
                "execution_kind": kind,
            }
            with self._db() as db:
                count = db.execute(
                    "SELECT count(*) FROM runtime_worker WHERE active=1 AND expires_at>?",
                    (time.time(),),
                ).fetchone()[0]
                if count >= 32:
                    raise ValueError("Revoke an unused connection before adding another (limit 32)")
                db.execute(
                    "INSERT INTO runtime_worker VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    tuple(worker.values()),
                )
            try:
                self.access.audit(
                    actor,
                    "worker.created",
                    worker["id"],
                    {"agent_id": agent_id, "model_id": model["id"]},
                )
            except Exception:
                with self._db() as db:
                    db.execute("DELETE FROM runtime_worker WHERE id=?", (worker["id"],))
                raise
        return {
            "worker": self._public(worker),
            "token": token,
            "token_shown_once": True,
            "expires_at": worker["expires_at"],
            "command": "python -m relay_lab.studio.worker --url <workspace-origin> --token-file ./worker-token --handler your_module:handle",
        }

    def revoke(self, actor, worker_id):
        actor = self.access.authorize_actor(actor, "models.manage")
        with self.registry.lock, self._db() as db:
            row = db.execute("SELECT * FROM runtime_worker WHERE id=?", (worker_id,)).fetchone()
            if row is None:
                raise ValueError("Agent connection not found")
            db.execute("UPDATE runtime_worker SET active=0 WHERE id=?", (worker_id,))
            self.access.audit(actor, "worker.revoked", worker_id)
        if self.coordination is not None:
            self.coordination.revoke(worker_id)
        return {"worker": self._public({**dict(row), "active": 0})}

    def revoked_ids(self, *, after="", limit=100):
        """Private, bounded reconciliation view containing no credential fields."""
        if not isinstance(after, str) or len(after) > 128:
            raise ValueError("Invalid worker reconciliation cursor")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Worker reconciliation pages must contain 1–100 IDs")
        with self._db() as db:
            rows = db.execute(
                "SELECT id FROM runtime_worker WHERE active=0 AND registry_id=? AND id>? "
                "ORDER BY id LIMIT ?",
                (self.registry.registry_id, after, limit),
            ).fetchall()
        return [row["id"] for row in rows]

    def authenticate(self, token):
        if not isinstance(token, str) or not token.startswith("relay_worker_") or len(token) > 128:
            raise AuthenticationError("A valid agent connection token is required")
        with self._db() as db:
            row = db.execute(
                "SELECT * FROM runtime_worker WHERE token_hash=?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            ).fetchone()
        if row is None or not row["active"] or row["expires_at"] <= time.time():
            raise AuthenticationError("Agent connection expired or was revoked")
        worker = dict(row)
        self._approved(worker, self._issuer(worker))
        return {
            key: worker[key]
            for key in (
                "id",
                "agent_id",
                "model_id",
                "release_revision",
                "registry_id",
                "execution_kind",
            )
        }

    def heartbeat(self, worker, value):
        label = bounded(
            value.get("controller_label", "External agent runtime"), "Controller label", 120
        )
        with self._db() as db:
            row = db.execute("SELECT * FROM runtime_worker WHERE id=?", (worker["id"],)).fetchone()
            if row is None or not row["active"] or row["expires_at"] <= time.time():
                raise AuthenticationError("Agent connection expired or was revoked")
            self._approved(dict(row), self._issuer(dict(row)))
            if self.coordination is not None:
                self.coordination.heartbeat(worker, LEASE_SECONDS)
            now = time.time()
            db.execute(
                "UPDATE runtime_worker SET last_seen=?,controller_label=? WHERE id=?",
                (now, label, worker["id"]),
            )
        return {"worker": self._public({**dict(row), "last_seen": now, "controller_label": label})}
