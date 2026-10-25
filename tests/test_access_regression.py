"""Authorization, credential lifecycle, and concurrent first-manager boundaries."""

import json
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from relay_lab.studio import access
from relay_lab.studio.access import AccessStore, AuthenticationError, AuthorizationError

PASSWORD = "a long test passphrase"


@pytest.fixture
def store(tmp_path):
    return AccessStore(tmp_path / "studio")


@pytest.fixture
def manager(store):
    return store.setup(store.setup_path.read_text().strip(), "manager", "Lab Manager", PASSWORD)


def join(store, manager, username="operator", role="employee"):
    invite = store.invite(manager["user"], username, username.title(), role)
    return store.accept_invite(invite["code"], PASSWORD)


def test_bootstrap_is_private_and_never_exposed_by_status(store):
    assert store.status() == {"setup_required": True}
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.setup_path.stat().st_mode) == 0o600
    code = store.setup_path.read_text().strip()
    assert len(code) >= 32
    reopened = AccessStore(store.storage)
    assert reopened.setup_path.read_text().strip() == code
    with pytest.raises(AuthenticationError, match="Invalid setup code"):
        store.setup("incorrect-code-of-sufficient-length", "manager", "Lab", PASSWORD)
    assert store.status() == {"setup_required": True}


def test_setup_creates_one_manager_and_consumes_code(store, manager):
    assert store.status() == {"setup_required": False}
    assert not store.setup_path.exists()
    assert manager["user"]["role"] == "manager"
    assert "models.release" in manager["permissions"]
    assert "team.manage" in manager["permissions"]
    with pytest.raises(AuthenticationError, match="already complete"):
        store.setup("x" * 43, "second", "Second", PASSWORD)
    reopened = AccessStore(store.storage)
    assert reopened.authenticate(manager["token"])["user"] == manager["user"]
    assert not reopened.setup_path.exists()


def test_setup_race_creates_exactly_one_account(store):
    second = AccessStore(store.storage)
    code = store.setup_path.read_text().strip()

    def attempt(pair):
        instance, username = pair
        try:
            return instance.setup(code, username, username, PASSWORD)
        except AuthenticationError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, [(store, "first"), (second, "second")]))
    winners = [item for item in outcomes if item]
    assert len(winners) == 1
    assert len(store.list_users(winners[0]["user"])["users"]) == 1


def test_bearer_tokens_invites_and_passwords_are_not_stored_raw(store, manager):
    invite = store.invite(manager["user"], "operator", "Operator", "employee")
    with sqlite3.connect(store.path) as db:
        persisted = "\n".join(db.iterdump())
        password_hash = db.execute("SELECT password_hash FROM users").fetchone()[0]
    assert PASSWORD not in persisted
    assert manager["token"] not in persisted
    assert invite["code"] not in persisted
    assert password_hash.startswith("scrypt$")
    assert all(key not in manager["user"] for key in ("password_hash", "password", "token"))
    assert "token" not in store.authenticate(manager["token"])


@pytest.mark.parametrize("password", ["short", "x" * 129, None, 12])
def test_setup_requires_strong_length_without_default_credentials(store, password):
    with pytest.raises(ValueError, match="12–128"):
        store.setup(store.setup_path.read_text().strip(), "manager", "Lab", password)
    assert store.status()["setup_required"]


def test_login_normalizes_username_and_issues_independent_sessions(store, manager):
    other = store.login("  MANAGER  ", PASSWORD)
    assert other["user"] == manager["user"]
    assert other["token"] != manager["token"]
    assert other["csrf"] != manager["csrf"]
    store.logout(other["token"])
    with pytest.raises(AuthenticationError):
        store.authenticate(other["token"])
    assert store.authenticate(manager["token"])["user"] == manager["user"]


def test_login_unknown_and_wrong_password_are_indistinguishable(store, manager):
    messages = []
    for username, password in [("manager", "wrong"), ("unknown", PASSWORD), ([], PASSWORD)]:
        with pytest.raises(AuthenticationError) as failure:
            store.login(username, password)
        messages.append(str(failure.value))
    assert messages == ["Invalid username or password"] * 3


def test_login_lockout_persists_and_recovers_after_window(store, manager, monkeypatch):
    now = access.time.time()
    monkeypatch.setattr(access.time, "time", lambda: now)
    for _ in range(access.LOGIN_LIMIT):
        with pytest.raises(AuthenticationError, match="Invalid username"):
            store.login("manager", "wrong")
    reopened = AccessStore(store.storage)
    with pytest.raises(AuthenticationError, match="Too many"):
        reopened.login("manager", PASSWORD)
    monkeypatch.setattr(access.time, "time", lambda: now + access.LOGIN_WINDOW + 1)
    assert reopened.login("manager", PASSWORD)["user"] == manager["user"]


def test_global_login_limit_prevents_unbounded_username_attempts(store, manager, monkeypatch):
    monkeypatch.setattr(access, "GLOBAL_LOGIN_LIMIT", 3)
    for index in range(3):
        with pytest.raises(AuthenticationError, match="Invalid username"):
            store.login(f"unknown-{index}", "wrong")
    with pytest.raises(AuthenticationError, match="Too many"):
        store.login("yet-another-name", "wrong")
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM login_attempts").fetchone()[0] == 4


def test_session_expires_after_eight_hours(store, manager, monkeypatch):
    assert manager["expires_at"] - access.time.time() == pytest.approx(access.SESSION_TTL, abs=1)
    monkeypatch.setattr(access.time, "time", lambda: manager["expires_at"])
    with pytest.raises(AuthenticationError, match="expired"):
        store.authenticate(manager["token"])
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


def test_invite_is_one_use_role_bound_and_expires(store, manager, monkeypatch):
    invitation = store.invite(manager["user"], "operator", "Operator", "employee")
    employee = store.accept_invite(invitation["code"], PASSWORD)
    assert employee["user"]["role"] == "employee"
    assert employee["permissions"] == ["sessions.run"]
    with pytest.raises(AuthenticationError):
        store.accept_invite(invitation["code"], PASSWORD)
    expired = store.invite(manager["user"], "another", "Another", "manager")
    monkeypatch.setattr(access.time, "time", lambda: expired["expires_at"])
    with pytest.raises(AuthenticationError, match="expired"):
        store.accept_invite(expired["code"], PASSWORD)


def test_invite_reissue_invalidates_previous_code(store, manager):
    old = store.invite(manager["user"], "operator", "Operator", "manager")
    current = store.invite(manager["user"], "operator", "Operator", "employee")
    with pytest.raises(AuthenticationError):
        store.accept_invite(old["code"], PASSWORD)
    assert store.accept_invite(current["code"], PASSWORD)["user"]["role"] == "employee"


def test_invalid_bootstrap_and_invitation_do_not_trigger_expensive_password_hashing(
    store, monkeypatch
):
    def unexpected_hash(*args):
        pytest.fail("Do not hash passwords before validating a one-use credential")

    monkeypatch.setattr(access, "_hash_password", unexpected_hash)
    with pytest.raises(AuthenticationError, match="Invalid setup code"):
        store.setup("invalid-code-of-sufficient-length", "manager", "Manager", PASSWORD)
    with pytest.raises(AuthenticationError, match="Invalid or expired invitation"):
        store.accept_invite("invalid-code-of-sufficient-length", PASSWORD)


def test_concurrent_invitation_acceptance_creates_only_one_identity(store, manager):
    invitation = store.invite(manager["user"], "operator", "Operator", "employee")
    second = AccessStore(store.storage)

    def attempt(instance):
        try:
            return instance.accept_invite(invitation["code"], PASSWORD)
        except AuthenticationError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, [store, second]))
    assert len([item for item in outcomes if item]) == 1
    assert len(store.list_users(manager["user"])["users"]) == 2


def test_employee_cannot_promote_self_or_spoof_manager_role(store, manager):
    employee = join(store, manager)
    forged = employee["user"] | {"role": "manager"}
    operations = [
        lambda: store.invite(forged, "intruder", "Intruder", "manager"),
        lambda: store.update_user(forged, employee["user"]["id"], "manager", True),
        lambda: store.list_users(forged),
        lambda: store.audit_log(forged),
    ]
    for operation in operations:
        with pytest.raises(AuthorizationError, match="manager"):
            operation()
    assert store.authenticate(employee["token"])["permissions"] == ["sessions.run"]


def test_role_change_revokes_all_sessions_and_uses_current_permissions(store, manager):
    other = join(store, manager, "second-manager", "manager")
    another_session = store.login("second-manager", PASSWORD)
    old_actor = other["user"]
    changed = store.update_user(manager["user"], old_actor["id"], "employee", True)
    assert changed["role"] == "employee"
    for credential in (other, another_session):
        with pytest.raises(AuthenticationError):
            store.authenticate(credential["token"])
    with pytest.raises(AuthorizationError):
        store.invite(old_actor, "intruder", "Intruder", "manager")
    assert store.login("second-manager", PASSWORD)["permissions"] == ["sessions.run"]


def test_deactivation_revokes_session_login_and_outstanding_invites(store, manager):
    other = join(store, manager, "second-manager", "manager")
    pending = store.invite(other["user"], "pending", "Pending", "employee")
    store.update_user(manager["user"], other["user"]["id"], "manager", False)
    with pytest.raises(AuthenticationError):
        store.authenticate(other["token"])
    with pytest.raises(AuthenticationError):
        store.login("second-manager", PASSWORD)
    with pytest.raises(AuthenticationError):
        store.list_users(other["user"])
    with pytest.raises(AuthenticationError):
        store.accept_invite(pending["code"], PASSWORD)
    store.update_user(manager["user"], other["user"]["id"], "manager", True)
    with pytest.raises(AuthenticationError):
        store.accept_invite(pending["code"], PASSWORD)


@pytest.mark.parametrize(
    "role,active", [("employee", True), ("manager", False), ("employee", False)]
)
def test_last_active_manager_cannot_be_removed(store, manager, role, active):
    with pytest.raises(ValueError, match="at least one active manager"):
        store.update_user(manager["user"], manager["user"]["id"], role, active)
    assert store.authenticate(manager["token"])["user"]["role"] == "manager"


def test_concurrent_manager_demotions_leave_an_active_manager(store, manager):
    other = join(store, manager, "second-manager", "manager")
    second_store = AccessStore(store.storage)

    def demote(pair):
        instance, credential = pair
        try:
            instance.update_user(credential["user"], credential["user"]["id"], "employee", True)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(demote, [(store, manager), (second_store, other)]))
    assert sorted(results) == [False, True]
    with sqlite3.connect(store.path) as db:
        assert (
            db.execute("SELECT count(*) FROM users WHERE active=1 AND role='manager'").fetchone()[0]
            == 1
        )


def test_existing_account_cannot_be_reinvited(store, manager):
    with pytest.raises(ValueError, match="already uses"):
        store.invite(manager["user"], "MANAGER", "Other", "employee")


@pytest.mark.parametrize("role", ["admin", "owner", "", [], None])
def test_unknown_roles_are_rejected(store, manager, role):
    with pytest.raises(ValueError, match="Role must"):
        store.invite(manager["user"], "operator", "Operator", role)
    with pytest.raises(ValueError, match="manager or employee"):
        store.update_user(manager["user"], manager["user"]["id"], role, True)


def test_audit_tracks_real_actor_and_redacts_secret_fields(store, manager):
    employee = join(store, manager)
    store.audit(
        employee["user"] | {"username": "spoofed"},
        "session.created",
        "session-1",
        {
            "token": "secret-bearer",
            "nested": {"password": PASSWORD},
            "mode": "latent",
        },
    )
    events = store.audit_log(manager["user"])["events"]
    event = next(item for item in events if item["action"] == "session.created")
    assert event["actor_username"] == "operator"
    assert event["actor_id"] == employee["user"]["id"]
    assert event["detail"] == {
        "token": "[redacted]",
        "nested": {"password": "[redacted]"},
        "mode": "latent",
    }
    assert "secret-bearer" not in json.dumps(events)
    assert PASSWORD not in json.dumps(events)


def test_symlink_database_and_setup_file_are_refused(tmp_path):
    target = tmp_path / "external"
    target.write_text("unchanged")
    database_dir = tmp_path / "db-link"
    database_dir.mkdir()
    (database_dir / "access.sqlite3").symlink_to(target)
    with pytest.raises(OSError):
        AccessStore(database_dir)
    setup_dir = tmp_path / "setup-link"
    setup_dir.mkdir()
    (setup_dir / "setup-code").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        AccessStore(setup_dir)
    assert target.read_text() == "unchanged"
