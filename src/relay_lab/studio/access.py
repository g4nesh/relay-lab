"""Local dashboard identities and permissions, separate from robot transport.

Only salted password hashes and hashes of bearer credentials are persisted.
The first manager must prove access to the private, one-use setup-code file.
Every privileged operation reloads its actor from the database; roles supplied
by a caller are never an authorization source.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SESSION_TTL = 8 * 60 * 60
INVITE_TTL = 24 * 60 * 60
LOGIN_WINDOW = 15 * 60
LOGIN_LIMIT = 10
GLOBAL_LOGIN_LIMIT = 100
PERMISSIONS = {
    "manager": (
        "workspace.edit",
        "models.manage",
        "models.release",
        "sessions.run",
        "team.manage",
        "audit.read",
    ),
    "employee": ("sessions.run",),
}
_SECRET_KEYS = {"password", "token", "csrf", "code", "secret", "authorization", "cookie"}


class AuthenticationError(PermissionError):
    """Credentials are missing, expired, revoked, or incorrect (HTTP 401)."""


class AuthorizationError(PermissionError):
    """The current account does not have the requested permission (HTTP 403)."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _username(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Username must be a handle or email address")  # noqa: TRY004
    value = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._@+\-]{2,79}", value):
        raise ValueError("Username must contain 3–80 letters, numbers, or . _ @ + -")
    return value


def _name(value) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 100:
        raise ValueError("Display name must contain 1–100 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Display name cannot contain control characters")
    return value.strip()


def _password(value) -> str:
    if not isinstance(value, str) or not 12 <= len(value) <= 128:
        raise ValueError("Password must contain 12–128 characters")
    return value


def _hash_password(value: str, salt: bytes | None = None) -> str:
    salt = secrets.token_bytes(16) if salt is None else salt
    hashed = hashlib.scrypt(
        value.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=32,
        maxmem=64 * 1024 * 1024,
    )
    return "scrypt$" + salt.hex() + "$" + hashed.hex()


def _check_password(value: str, stored: str) -> bool:
    try:
        algorithm, salt, _ = stored.split("$")
        if algorithm != "scrypt":
            return False
        return hmac.compare_digest(_hash_password(value, bytes.fromhex(salt)), stored)
    except (ValueError, TypeError):
        return False


def _public_user(row) -> dict:
    return {key: row[key] for key in ("id", "username", "name", "role")} | {
        "active": bool(row["active"]),
    }


def _safe_detail(value):
    if isinstance(value, dict):
        return {
            str(key): "[redacted]"
            if any(part in str(key).lower() for part in _SECRET_KEYS)
            else _safe_detail(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_detail(item) for item in value]
    return value


class AccessStore:
    def __init__(self, storage):
        self.storage = Path(storage)
        self.storage.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.storage / "access.sqlite3"
        self.setup_path = self.storage / "setup-code"
        # O_NOFOLLOW refuses an existing symlink instead of chmod/opening its target.
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("""CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('manager', 'employee')),
                active INTEGER NOT NULL CHECK(active IN (0, 1)),
                password_hash TEXT NOT NULL, created_at REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                csrf TEXT NOT NULL, expires_at REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS invites (
                code_hash TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('manager', 'employee')),
                expires_at REAL NOT NULL, created_by TEXT NOT NULL REFERENCES users(id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS login_attempts (
                key TEXT PRIMARY KEY, failures INTEGER NOT NULL, expires_at REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY, at REAL NOT NULL, actor_id TEXT NOT NULL,
                actor_username TEXT NOT NULL, action TEXT NOT NULL,
                target TEXT NOT NULL, detail TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS audit_at ON audit_events(at)")
            if db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                self.setup_path.unlink(missing_ok=True)
            else:
                self._prepare_setup(db)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _prepare_setup(self, db):
        existing = db.execute("SELECT value FROM metadata WHERE key='setup_hash'").fetchone()
        if existing and self.setup_path.is_file() and not self.setup_path.is_symlink():
            self.setup_path.chmod(0o600)
            if hmac.compare_digest(_digest(self.setup_path.read_text().strip()), existing[0]):
                return
        if self.setup_path.is_symlink():
            raise ValueError("Setup code path must not be a symlink")
        # The transaction serializes startup and first-account creation across processes.
        code = secrets.token_urlsafe(32)
        descriptor = os.open(
            self.setup_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(code + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        db.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('setup_hash',?)",
            (_digest(code),),
        )

    def status(self) -> dict:
        with self._transaction() as db:
            return {"setup_required": db.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None}

    @staticmethod
    def _actor(db, actor, permission=None):
        identity = actor.get("id") if isinstance(actor, dict) else None
        if not isinstance(identity, str):
            identity = None
        row = db.execute("SELECT * FROM users WHERE id=?", (identity,)).fetchone()
        if row is None or not row["active"]:
            raise AuthenticationError("Sign in to continue")
        if permission and permission not in PERMISSIONS[row["role"]]:
            raise AuthorizationError("A manager account is required for this action")
        return _public_user(row)

    def authorize_actor(self, actor, permission=None):
        """Reload an account before using a delegated runtime credential."""
        with self._transaction() as db:
            return self._actor(db, actor, permission)

    @staticmethod
    def _audit(db, actor, action, target, detail=None):
        if not isinstance(action, str) or not 1 <= len(action) <= 100:
            raise ValueError("Invalid audit action")
        if not isinstance(target, str) or len(target) > 200:
            raise ValueError("Invalid audit target")
        encoded = json.dumps(_safe_detail(detail or {}), allow_nan=False, ensure_ascii=False)
        if len(encoded) > 8000:
            raise ValueError("Audit detail is too large")
        db.execute(
            "INSERT INTO audit_events VALUES (?,?,?,?,?,?,?)",
            (
                uuid.uuid4().hex,
                time.time(),
                actor["id"],
                actor["username"],
                action,
                target,
                encoded,
            ),
        )

    @staticmethod
    def _credentials(db, user):
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires_at = time.time() + SESSION_TTL
        db.execute("DELETE FROM sessions WHERE expires_at<=?", (time.time(),))
        db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?)",
            (
                _digest(token),
                user["id"],
                csrf,
                expires_at,
            ),
        )
        return {
            "token": token,
            "csrf": csrf,
            "user": user,
            "permissions": list(PERMISSIONS[user["role"]]),
            "expires_at": expires_at,
        }

    def setup(self, code, username, name, password) -> dict:
        username, name = _username(username), _name(name)
        password = _password(password)
        with self._transaction() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='setup_hash'").fetchone()
            if db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                raise AuthenticationError("Workspace setup is already complete")
            if (
                not isinstance(code, str)
                or not 20 <= len(code) <= 128
                or not row
                or not hmac.compare_digest(
                    _digest(code),
                    row[0],
                )
            ):
                raise AuthenticationError("Invalid setup code")
            password_hash = _hash_password(password)
            user = {
                "id": uuid.uuid4().hex,
                "username": username,
                "name": name,
                "role": "manager",
                "active": True,
            }
            db.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                (
                    user["id"],
                    username,
                    name,
                    "manager",
                    1,
                    password_hash,
                    time.time(),
                ),
            )
            db.execute("DELETE FROM metadata WHERE key='setup_hash'")
            self._audit(db, user, "account.setup", user["id"])
            result = self._credentials(db, user)
        self.setup_path.unlink(missing_ok=True)
        return result

    def login(self, username, password) -> dict:
        try:
            username = _username(username)
        except ValueError:
            username = ""  # Same failure and expensive password check as an unknown user.
        if not isinstance(password, str) or len(password) > 128:
            password = ""
        key, now = _digest(username), time.time()
        result, error = None, None
        with self._transaction() as db:
            db.execute("DELETE FROM login_attempts WHERE expires_at<=?", (now,))
            attempts = {
                row["key"]: row["failures"]
                for row in db.execute(
                    "SELECT * FROM login_attempts WHERE key IN (?, 'global')",
                    (key,),
                )
            }
            if (
                attempts.get(key, 0) >= LOGIN_LIMIT
                or attempts.get("global", 0) >= GLOBAL_LOGIN_LIMIT
            ):
                error = "Too many sign-in attempts. Try again later."
            else:
                row = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
                if row:
                    matches = _check_password(password, row["password_hash"])
                else:
                    _hash_password(password, bytes(16))
                    matches = False
                if row and row["active"] and matches:
                    user = _public_user(row)
                    db.execute("DELETE FROM login_attempts WHERE key=?", (key,))
                    self._audit(db, user, "account.login", user["id"])
                    result = self._credentials(db, user)
                else:
                    for attempt_key in (key, "global"):
                        db.execute(
                            """INSERT INTO login_attempts VALUES (?,1,?)
                            ON CONFLICT(key) DO UPDATE SET failures=failures+1""",
                            (attempt_key, now + LOGIN_WINDOW),
                        )
                    # The global limiter caps new keys to 101/window; this also bounds old DBs.
                    db.execute("""DELETE FROM login_attempts WHERE key IN (
                        SELECT key FROM login_attempts WHERE key != 'global'
                        ORDER BY expires_at DESC LIMIT -1 OFFSET 1000)""")
                    error = "Invalid username or password"
        if error:
            raise AuthenticationError(error)
        return result

    def accept_invite(self, code, password) -> dict:
        if not isinstance(code, str) or not 20 <= len(code) <= 128:
            raise AuthenticationError("Invalid or expired invitation")
        password = _password(password)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM invites WHERE code_hash=?", (_digest(code),)).fetchone()
            if row is None or row["expires_at"] <= time.time():
                raise AuthenticationError("Invalid or expired invitation")
            inviter = db.execute("SELECT * FROM users WHERE id=?", (row["created_by"],)).fetchone()
            if inviter is None or not inviter["active"] or inviter["role"] != "manager":
                raise AuthenticationError(
                    "Invitation is no longer valid; ask a manager to reissue it"
                )
            password_hash = _hash_password(password)
            user = {
                "id": uuid.uuid4().hex,
                "username": row["username"],
                "name": row["name"],
                "role": row["role"],
                "active": True,
            }
            db.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                (
                    user["id"],
                    user["username"],
                    user["name"],
                    user["role"],
                    1,
                    password_hash,
                    time.time(),
                ),
            )
            db.execute("DELETE FROM invites WHERE code_hash=?", (_digest(code),))
            self._audit(db, user, "account.joined", user["id"], {"invited_by": row["created_by"]})
            return self._credentials(db, user)

    def authenticate(self, token) -> dict:
        if not isinstance(token, str) or not 20 <= len(token) <= 128:
            raise AuthenticationError("Sign in to continue")
        result = None
        with self._transaction() as db:
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (time.time(),))
            row = db.execute(
                """SELECT users.*, sessions.csrf, sessions.expires_at
                FROM sessions JOIN users ON users.id=sessions.user_id
                WHERE token_hash=? AND users.active=1""",
                (_digest(token),),
            ).fetchone()
            if row:
                result = {
                    "user": _public_user(row),
                    "csrf": row["csrf"],
                    "permissions": list(PERMISSIONS[row["role"]]),
                    "expires_at": row["expires_at"],
                }
        if result is None:
            raise AuthenticationError("Session expired or revoked; sign in again")
        return result

    def logout(self, token):
        if not isinstance(token, str):
            return
        with self._transaction() as db:
            row = db.execute("SELECT user_id FROM sessions WHERE token_hash=?", (_digest(token),))
            session = row.fetchone()
            if session:
                user = db.execute("SELECT * FROM users WHERE id=?", (session[0],)).fetchone()
                self._audit(db, _public_user(user), "account.logout", user["id"])
            db.execute("DELETE FROM sessions WHERE token_hash=?", (_digest(token),))

    def list_users(self, actor) -> dict:
        with self._transaction() as db:
            self._actor(db, actor, "team.manage")
            return {
                "users": [
                    _public_user(row)
                    for row in db.execute(
                        "SELECT * FROM users ORDER BY created_at, id",
                    )
                ]
            }

    def invite(self, actor, username, name, role) -> dict:
        username, name = _username(username), _name(name)
        if not isinstance(role, str) or role not in PERMISSIONS:
            raise ValueError("Role must be manager or employee")
        with self._transaction() as db:
            actor = self._actor(db, actor, "team.manage")
            if db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                raise ValueError("An account already uses that username")
            code, expires_at = secrets.token_urlsafe(32), time.time() + INVITE_TTL
            db.execute("DELETE FROM invites WHERE expires_at<=?", (time.time(),))
            db.execute(
                """INSERT INTO invites VALUES (?,?,?,?,?,?)
                ON CONFLICT(username) DO UPDATE SET code_hash=excluded.code_hash,
                name=excluded.name, role=excluded.role, expires_at=excluded.expires_at,
                created_by=excluded.created_by""",
                (
                    _digest(code),
                    username,
                    name,
                    role,
                    expires_at,
                    actor["id"],
                ),
            )
            self._audit(db, actor, "team.invited", username, {"role": role})
            return {
                "code": code,
                "expires_at": expires_at,
                "username": username,
                "name": name,
                "role": role,
            }

    def update_user(self, actor, user_id, role, active) -> dict:
        if not isinstance(role, str) or role not in PERMISSIONS or not isinstance(active, bool):
            raise ValueError("Choose manager or employee and a boolean active state")
        if not isinstance(user_id, str):
            raise ValueError("Account not found")  # noqa: TRY004
        with self._transaction() as db:
            actor = self._actor(db, actor, "team.manage")
            row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if row is None:
                raise ValueError("Account not found")
            if row["active"] and row["role"] == "manager" and (not active or role != "manager"):
                count = db.execute("SELECT count(*) FROM users WHERE active=1 AND role='manager'")
                if count.fetchone()[0] <= 1:
                    raise ValueError("Keep at least one active manager")
            if row["role"] != role or bool(row["active"]) != active:
                db.execute(
                    "UPDATE users SET role=?,active=? WHERE id=?", (role, int(active), user_id)
                )
                db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
                # An old invite must never survive its issuing manager's loss of privileges.
                if role != "manager" or not active:
                    db.execute("DELETE FROM invites WHERE created_by=?", (user_id,))
                self._audit(
                    db,
                    actor,
                    "team.updated",
                    user_id,
                    {
                        "previous_role": row["role"],
                        "previous_active": bool(row["active"]),
                        "role": role,
                        "active": active,
                    },
                )
            return _public_user(db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())

    def audit(self, actor, action, target, detail=None):
        with self._transaction() as db:
            actor = self._actor(db, actor)
            self._audit(db, actor, action, target, detail)

    def audit_log(self, actor) -> dict:
        with self._transaction() as db:
            self._actor(db, actor, "audit.read")
            events = []
            for row in db.execute("SELECT * FROM audit_events ORDER BY at DESC, id DESC LIMIT 200"):
                events.append(dict(row) | {"detail": json.loads(row["detail"])})
            return {"events": events}
