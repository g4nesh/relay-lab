"""Password-free profiles for a trusted, loopback-only dashboard.

Profiles select an operator context; they are not an identity security boundary.
Hosted deployments continue using the authenticated AccessStore.
"""

from __future__ import annotations

import secrets
import time
import uuid

from .access import AccessStore, AuthenticationError, _digest, _public_user


class ProfileStore(AccessStore):
    def __init__(self, storage):
        super().__init__(storage)
        self.profile_nonce = secrets.token_urlsafe(32)
        with self._transaction() as db:
            if not db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                for username, name, role in (
                    ("manager", "Manager", "manager"),
                    ("operator", "Operator", "employee"),
                ):
                    db.execute(
                        "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                        (uuid.uuid4().hex, username, name, role, 1, "local-profile", time.time()),
                    )
                db.execute("DELETE FROM metadata WHERE key='setup_hash'")
        self.setup_path.unlink(missing_ok=True)

    def profiles(self):
        with self._transaction() as db:
            rows = db.execute(
                "SELECT * FROM users WHERE active=1 ORDER BY role DESC, created_at, id LIMIT 100"
            ).fetchall()
            return {"profiles": [_public_user(row) for row in rows], "nonce": self.profile_nonce}

    def select_profile(self, user_id, nonce, previous_token=""):
        if not isinstance(nonce, str) or not secrets.compare_digest(nonce, self.profile_nonce):
            raise PermissionError("Refresh the account switcher before choosing a profile")
        if not isinstance(user_id, str) or len(user_id) > 80:
            raise ValueError("Choose an existing workspace profile")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM users WHERE id=? AND active=1", (user_id,)).fetchone()
            if row is None:
                raise AuthenticationError("This profile is no longer available")
            user = _public_user(row)
            if previous_token:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (_digest(previous_token),))
            self._audit(db, user, "account.selected", user["id"])
            return self._credentials(db, user)
