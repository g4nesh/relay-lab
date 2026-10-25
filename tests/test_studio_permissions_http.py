"""Exercise real dashboard identities and model approvals through HTTP dispatch."""

import copy
import json
import re
import threading
from http.client import HTTPResponse
from http.cookies import SimpleCookie
from io import BytesIO
from types import SimpleNamespace

import pytest

from relay_lab.llm import Generation
from relay_lab.studio.access import AccessStore
from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.server import StudioHandler
from relay_lab.studio.service import StudioService, example_workspace

PASSWORD = "correct horse robot battery"


class MemorySocket:
    def __init__(self, data):
        self.data, self.sent = data, bytearray()

    def makefile(self, *args):
        return BytesIO(self.data)

    def sendall(self, data):
        self.sent.extend(data)


class Journal:
    """In-memory transport double; identity, registry, service, and HTTP are real."""

    def __init__(self):
        self.runs, self.batches, self.messages = {}, {}, {}
        self.commits, self.acks = 0, 0

    def coordinator(self):
        journal = self

        class Coordinator:
            def __init__(self):
                self.client = self

            def create(self, run_id, worker_id, *args, **kwargs):
                self.run_id, self.worker_id = run_id, worker_id
                journal.runs[run_id] = {
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "revision": 0,
                    "tick": 0,
                    "phase": "ready",
                    "status": "running",
                    "state": copy.deepcopy(kwargs["state"]),
                }
                journal.messages[run_id] = []

            def snapshot(self):
                return copy.deepcopy(journal.runs[self.run_id])

            def prepare(self, revision, commands):
                assert revision == journal.runs[self.run_id]["revision"] and commands == []
                journal.batches[(self.run_id, revision)] = {
                    "prepare_json": json.dumps({"commands": [], "messages": []}),
                    "result_json": "",
                }
                journal.runs[self.run_id]["phase"] = "executing"

            def commit(self, revision, state, outcomes, *, messages):
                run = journal.runs[self.run_id]
                assert run["revision"] == revision
                journal.batches[(self.run_id, revision)]["result_json"] = json.dumps(
                    {"state": state, "outcomes": outcomes, "messages": messages}
                )
                journal.messages[self.run_id].extend(copy.deepcopy(messages))
                run.update(
                    state=copy.deepcopy(state),
                    revision=revision + 1,
                    tick=revision + 1,
                    phase="ready",
                )
                journal.commits += 1

            def _batch(self, revision):
                return copy.deepcopy(journal.batches.get((self.run_id, revision)))

            def _mutate(self, reducer, args, reader, expected):
                assert reducer == "commit_factory_batch"
                assert reader()["result_json"] == expected["result_json"]
                journal.acks += 1
                return reader()

            def _sql(self, query):
                match = re.search(r"run_id = '([^']+)'", query)
                if "factory_event" in query:
                    return [{"payload_json": json.dumps(m)} for m in journal.messages[match[1]]]
                if "SELECT state_json" in query:
                    return [{"state_json": json.dumps(journal.runs[match[1]]["state"])}]
                return [
                    {key: run[key] for key in ("run_id", "tick", "status", "phase")}
                    for run in journal.runs.values()
                ]

        return Coordinator()


class LLM:
    def __init__(self):
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return Generation("Hold position; the peer reports an obstacle.", 0.01, 20, 10)

    def close(self):
        pass


class Bridge:
    def __init__(self, root, *, accounts=True):
        self.journal, self.llm = Journal(), LLM()
        self.service = StudioService(
            root, coordinator_factory=self.journal.coordinator, llm=self.llm
        )
        self.access = AccessStore(self.service.storage)
        self.registry = ModelRegistry(self.service.storage)
        if accounts:
            self.manager = self.access.setup(
                self.access.setup_path.read_text().strip(),
                "manager",
                "Manager",
                PASSWORD,
            )
            invitation = self.access.invite(
                self.manager["user"], "employee", "Employee", "employee"
            )
            self.employee = self.access.accept_invite(invitation["code"], PASSWORD)

    def request(self, path, *, method="GET", value=None, actor=None, headers=None, raw=None):
        payload = raw if raw is not None else (b"" if value is None else json.dumps(value).encode())
        fields = {
            "Host": "127.0.0.1:8770",
            "Connection": "close",
            "Content-Length": str(len(payload)),
            "Content-Type": "application/json",
        }
        if actor:
            fields.update(
                Cookie=f"relay_studio_8770={actor['token']}", **{"X-Relay-Token": actor["csrf"]}
            )
        fields.update(headers or {})
        request = "\r\n".join(
            [f"{method} {path} HTTP/1.1", *(f"{key}: {value}" for key, value in fields.items())]
        )
        sock = MemorySocket((request + "\r\n\r\n").encode() + payload)
        StudioHandler(
            sock,
            ("127.0.0.1", 1),
            SimpleNamespace(server_port=8770),
            service=self.service,
            access=self.access,
            registry=self.registry,
        )
        response = HTTPResponse(MemorySocket(bytes(sock.sent)), method=method)
        response.begin()
        body = response.read()
        data = (
            json.loads(body)
            if response.getheader("Content-Type").startswith("application/json")
            else body
        )
        return response.status, dict(response.getheaders()), data

    def save_and_release(self):
        workspace = {**example_workspace(), "mode": "mediated"}
        assert (
            self.request("/api/workspace", method="POST", value=workspace, actor=self.manager)[0]
            == 200
        )
        records = self.request("/api/models", actor=self.manager)[2]["models"]
        for record in records:
            status, _, data = self.request(
                f"/api/models/{record['id']}/release",
                method="POST",
                actor=self.manager,
                value={
                    "expected_revision": record["revision"],
                    "reason": "Reviewed exact configuration",
                },
            )
            assert status == 200, data
        return workspace


@pytest.fixture
def bridge(tmp_path):
    return Bridge(tmp_path)


def test_queued_write_rechecks_account_after_role_revocation(bridge, monkeypatch):
    queued = threading.Event()
    original_guard = StudioHandler.guard
    results = []

    def notify_guard(handler):
        original_guard(handler)
        queued.set()

    monkeypatch.setattr(StudioHandler, "guard", notify_guard)
    invite = bridge.access.invite(bridge.manager["user"], "secondmanager", "Second", "manager")
    second = bridge.access.accept_invite(invite["code"], PASSWORD)
    worker = threading.Thread(
        target=lambda: results.append(
            bridge.request("/api/workspace", method="POST", actor=second, value=example_workspace())
        )
    )
    with bridge.registry.lock:
        worker.start()
        assert queued.wait(2)
        # The identity store is authoritative even if credentials were valid
        # when the HTTP request arrived and waited behind another exchange.
        bridge.access.update_user(bridge.manager["user"], second["user"]["id"], "employee", True)
    worker.join(2)
    assert not worker.is_alive()
    assert results[0][0] == 401
    assert bridge.service.workspace()["agents"] == []
    assert bridge.registry.catalog(bridge.manager["user"])["models"] == []


def test_public_setup_requires_private_code_and_returns_httponly_cookie(tmp_path):
    bridge = Bridge(tmp_path, accounts=False)
    status, _, state = bridge.request("/api/auth/status")
    assert status == 200 and state == {"setup_required": True, "authenticated": False}
    assert bridge.access.setup_path.read_text().strip() not in json.dumps(state)
    status, _, _ = bridge.request(
        "/api/auth/setup",
        method="POST",
        value={
            "code": "wrong-code-" * 4,
            "username": "owner",
            "name": "Owner",
            "password": PASSWORD,
        },
    )
    assert status == 401 and bridge.access.status()["setup_required"]
    code = bridge.access.setup_path.read_text().strip()
    status, headers, result = bridge.request(
        "/api/auth/setup",
        method="POST",
        value={
            "code": code,
            "username": "owner",
            "name": "Owner",
            "password": PASSWORD,
            "role": "employee",  # Initial role comes from server policy.
        },
    )
    assert status == 200 and result["user"]["role"] == "manager"
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
    assert "Path=/" in headers["Set-Cookie"]
    cookie = SimpleCookie(headers["Set-Cookie"])
    credential = cookie["relay_studio_8770"].value
    assert credential != result["token"] and credential not in json.dumps(result)
    assert PASSWORD not in json.dumps(result) and code not in json.dumps(result)
    assert not bridge.access.setup_path.exists()


def test_employee_request_is_visible_to_manager_but_cannot_self_resolve(bridge):
    status, _, request = bridge.request(
        "/api/requests",
        method="POST",
        actor=bridge.employee,
        value={"kind": "release", "message": "Please review the paired model revisions."},
    )
    assert status == 201 and request["status"] == "pending"
    assert bridge.request("/api/requests", actor=bridge.employee)[2]["requests"][0] == request
    assert bridge.request("/api/requests", actor=bridge.manager)[2]["requests"][0] == request
    route = f"/api/requests/{request['id']}/resolve"
    assert (
        bridge.request(route, method="POST", actor=bridge.employee, value={"status": "resolved"})[0]
        == 403
    )
    status, _, resolved = bridge.request(
        route,
        method="POST",
        actor=bridge.manager,
        value={"status": "resolved", "reason": "Reviewed; release approval is a separate step."},
    )
    assert status == 200 and resolved["status"] == "resolved"
    assert bridge.registry.catalog(bridge.manager["user"])["models"] == []
    assert (
        bridge.request(route, method="POST", actor=bridge.manager, value={"status": "declined"})[0]
        == 409
    )


def test_stored_release_cannot_authorize_different_session_configuration(bridge):
    workspace = bridge.save_and_release()
    status, _, session = bridge.request(
        "/api/sessions", method="POST", actor=bridge.employee, value=workspace
    )
    assert status == 201
    sid = session["session_id"]
    bridge.service.sessions[sid]["workspace"]["agents"][0]["rig"] = "http://localhost:9999/rogue"
    assert (
        bridge.request(
            f"/api/sessions/{sid}/exchange",
            method="POST",
            actor=bridge.employee,
            value={
                "request_id": "mismatch",
                "source": "manual",
                "reports": {"left": "A", "right": "B"},
            },
        )[0]
        == 403
    )
    assert bridge.llm.calls == 0 and bridge.journal.commits == 0


@pytest.mark.parametrize(
    "route",
    ["/api/bootstrap", "/api/models", "/api/team", "/api/audit", "/api/sessions", "/api/health"],
)
def test_anonymous_private_api_is_401(bridge, route):
    assert bridge.request(route)[0] == 401
    assert bridge.journal.runs == {} and bridge.llm.calls == 0


def test_login_and_bootstrap_only_expose_session_csrf_not_credentials(bridge):
    status, headers, result = bridge.request(
        "/api/auth/login",
        method="POST",
        value={
            "username": "employee",
            "password": PASSWORD,
            "role": "manager",
        },
    )
    assert status == 200 and result["user"]["role"] == "employee"
    cookie = SimpleCookie(headers["Set-Cookie"])
    token = cookie["relay_studio_8770"].value
    actor = {"token": token, "csrf": result["token"]}
    status, _, bootstrap = bridge.request("/api/bootstrap", actor=actor)
    assert status == 200 and bootstrap["token"] == result["token"]
    assert bootstrap["permissions"] == ["sessions.run"]
    assert token not in json.dumps(bootstrap) and PASSWORD not in json.dumps(bootstrap)
    assert "password_hash" not in json.dumps(bootstrap)
    assert (
        bridge.request(
            "/api/auth/login",
            method="POST",
            value={
                "username": "employee",
                "password": "wrong password",
            },
        )[0]
        == 401
    )


@pytest.mark.parametrize("token_kind", ["missing", "other_identity", "forged"])
def test_csrf_is_bound_to_authenticated_identity(bridge, token_kind):
    token = {"missing": "", "other_identity": bridge.employee["csrf"], "forged": "manager"}[
        token_kind
    ]
    status, _, _ = bridge.request(
        "/api/workspace",
        method="POST",
        value=example_workspace(),
        actor=bridge.manager,
        headers={"X-Relay-Token": token},
    )
    assert status == 403
    assert bridge.service.workspace()["agents"] == []
    assert bridge.registry.catalog(bridge.manager["user"])["models"] == []


@pytest.mark.parametrize(
    "operation",
    ["workspace", "upload", "team_read", "invite", "team_edit", "release", "retire", "audit"],
)
def test_employee_cannot_forge_manager_permissions(bridge, operation):
    model = bridge.registry.sync_workspace(bridge.manager["user"], example_workspace())["models"][0]
    routes = {
        "workspace": ("/api/workspace", "POST", {**example_workspace(), "role": "manager"}),
        "upload": ("/api/upload", "POST", {"role": "manager"}),
        "team_read": ("/api/team", "GET", None),
        "invite": (
            "/api/team/invitations",
            "POST",
            {"username": "intruder", "name": "Intruder", "role": "manager"},
        ),
        "team_edit": (
            f"/api/team/{bridge.employee['user']['id']}",
            "POST",
            {"role": "manager", "active": True},
        ),
        "release": (
            f"/api/models/{model['id']}/release",
            "POST",
            {"expected_revision": 1, "role": "manager"},
        ),
        "retire": (
            f"/api/models/{model['id']}/retire",
            "POST",
            {"expected_revision": 1, "role": "manager"},
        ),
        "audit": ("/api/audit", "GET", None),
    }
    route, method, value = routes[operation]
    status, _, data = bridge.request(
        route,
        method=method,
        value=value,
        actor=bridge.employee,
        headers={"X-Relay-Role": "manager", "X-Relay-User": bridge.manager["user"]["id"]},
    )
    assert status == 403, data
    assert bridge.service.workspace()["agents"] == []
    assert list(bridge.service.uploads.iterdir()) == []
    assert bridge.journal.runs == {} and bridge.llm.calls == 0
    assert len(bridge.access.list_users(bridge.manager["user"])["users"]) == 2
    assert all(
        model["status"] == "draft"
        for model in bridge.registry.catalog(bridge.manager["user"])["models"]
    )


def test_draft_cannot_run_then_employee_executes_release_and_retirement_blocks(bridge):
    workspace = {**example_workspace(), "mode": "mediated"}
    assert (
        bridge.request("/api/workspace", method="POST", value=workspace, actor=bridge.manager)[0]
        == 200
    )
    for actor in (bridge.manager, bridge.employee):
        assert (
            bridge.request("/api/sessions", method="POST", value=workspace, actor=actor)[0] == 403
        )
    assert bridge.journal.runs == {}
    bridge.save_and_release()
    status, _, result = bridge.request(
        "/api/sessions", method="POST", value=workspace, actor=bridge.employee
    )
    assert status == 201, result
    sid = result["session_id"]
    snapshot = bridge.journal.runs[sid]["state"]["model_release"]
    assert len(snapshot["models"]) == 2
    exchange = {
        "request_id": "first",
        "source": "manual",
        "reports": {"left": "Obstacle", "right": "Locally clear"},
    }
    status, _, result = bridge.request(
        f"/api/sessions/{sid}/exchange", method="POST", value=exchange, actor=bridge.employee
    )
    assert status == 200, result
    assert bridge.llm.calls > 0 and bridge.journal.commits == 1
    calls = bridge.llm.calls
    model = snapshot["models"][0]
    status, _, retired = bridge.request(
        f"/api/models/{model['id']}/retire",
        method="POST",
        actor=bridge.manager,
        value={"expected_revision": model["revision"], "reason": "Withdrawn"},
    )
    assert status == 200, retired
    for role in (bridge.employee, bridge.manager):
        for operation in ("exchange", "resume"):
            status, _, _ = bridge.request(
                f"/api/sessions/{sid}/{operation}",
                method="POST",
                actor=role,
                value={**exchange, "request_id": "second"},
            )
            assert status == 403
    assert bridge.llm.calls == calls and bridge.journal.commits == 1
    assert bridge.journal.acks == 0
    status, _, history = bridge.request(f"/api/sessions/{sid}", actor=bridge.employee)
    assert status == 200 and not history["release_ready"] and not history["resumable"]
    assert len(history["messages"]) == 2


def test_stale_release_is_409_and_rerelease_cannot_revive_old_session(bridge):
    workspace = bridge.save_and_release()
    status, _, created = bridge.request(
        "/api/sessions", method="POST", value=workspace, actor=bridge.employee
    )
    assert status == 201
    model = bridge.registry.catalog(bridge.manager["user"])["models"][0]
    assert (
        bridge.request(
            f"/api/models/{model['id']}/release",
            method="POST",
            actor=bridge.manager,
            value={"expected_revision": 1},
        )[0]
        == 409
    )
    status, _, retired = bridge.request(
        f"/api/models/{model['id']}/retire",
        method="POST",
        actor=bridge.manager,
        value={"expected_revision": model["revision"]},
    )
    assert status == 200
    assert (
        bridge.request(
            f"/api/models/{model['id']}/release",
            method="POST",
            actor=bridge.manager,
            value={"expected_revision": retired["revision"]},
        )[0]
        == 200
    )
    assert (
        bridge.request(
            f"/api/sessions/{created['session_id']}/resume",
            method="POST",
            actor=bridge.employee,
            value={},
        )[0]
        == 403
    )
    assert (
        bridge.request("/api/sessions", method="POST", value=workspace, actor=bridge.employee)[0]
        == 201
    )


@pytest.mark.parametrize("field", ["source", "rig", "objective"])
def test_employee_cannot_connect_alternative_unsaved_workspace(bridge, field):
    workspace = bridge.save_and_release()
    if field == "source":
        workspace["agents"][0]["source"]["ref"] = "other/model"
    elif field == "rig":
        workspace["agents"][0]["rig"] = "http://127.0.0.1:9999/robot"
    else:
        workspace["objective"] = "A different task"
    status, _, result = bridge.request(
        "/api/sessions", method="POST", actor=bridge.employee, value=workspace
    )
    assert status == 403 and "saved workspace" in result["error"]
    assert bridge.journal.runs == {}


def test_old_session_without_release_snapshot_remains_inspect_only(bridge):
    workspace = {**example_workspace(), "mode": "mediated"}
    sid = bridge.service.connect(workspace)["session_id"]
    bridge.service.sessions.clear()  # Restart: history must come from durable state.
    status, _, history = bridge.request(f"/api/sessions/{sid}", actor=bridge.employee)
    assert status == 200 and history["status"] == "inspect_only"
    assert not history["resumable"] and not history["release_ready"]
    status, _, catalog = bridge.request("/api/sessions", actor=bridge.employee)
    assert status == 200 and not catalog["sessions"][0]["resumable"]
    for operation in ("resume", "exchange"):
        assert (
            bridge.request(
                f"/api/sessions/{sid}/{operation}", method="POST", actor=bridge.employee, value={}
            )[0]
            == 403
        )
    assert bridge.service.sessions == {} and bridge.llm.calls == 0


def test_manager_can_invite_employee_and_revoke_http_credentials(bridge):
    status, _, invite = bridge.request(
        "/api/team/invitations",
        method="POST",
        actor=bridge.manager,
        value={"username": "new-operator", "name": "New operator", "role": "employee"},
    )
    assert status == 201
    status, headers, joined = bridge.request(
        "/api/auth/join",
        method="POST",
        value={
            "code": invite["code"],
            "password": PASSWORD,
            "role": "manager",
        },
    )
    assert status == 200 and joined["user"]["role"] == "employee"
    cookie = SimpleCookie(headers["Set-Cookie"])
    actor = {"token": cookie["relay_studio_8770"].value, "csrf": joined["token"]}
    assert bridge.request("/api/bootstrap", actor=actor)[0] == 200
    assert (
        bridge.request(
            f"/api/team/{joined['user']['id']}",
            method="POST",
            actor=bridge.manager,
            value={"role": "employee", "active": False},
        )[0]
        == 200
    )
    assert bridge.request("/api/bootstrap", actor=actor)[0] == 401
    assert (
        bridge.request(
            "/api/auth/join", method="POST", value={"code": invite["code"], "password": PASSWORD}
        )[0]
        == 401
    )


def test_audit_combines_account_and_model_events_without_secrets(bridge):
    bridge.save_and_release()
    status, _, audit = bridge.request("/api/audit", actor=bridge.manager)
    assert status == 200, audit
    actions = {event["action"] for event in audit["events"]}
    assert {"account.setup", "workspace.saved", "release", "register"} <= actions
    serialized = json.dumps(audit)
    assert bridge.manager["token"] not in serialized
    assert bridge.manager["csrf"] not in serialized and PASSWORD not in serialized


def test_logout_cookie_is_cleared_and_session_cannot_be_replayed(bridge):
    status, headers, result = bridge.request(
        "/api/auth/logout", method="POST", actor=bridge.employee, value={}
    )
    assert status == 200 and result["signed_out"]
    assert "Max-Age=0" in headers["Set-Cookie"] and "HttpOnly" in headers["Set-Cookie"]
    assert bridge.request("/api/bootstrap", actor=bridge.employee)[0] == 401


def test_public_auth_and_private_mutation_reject_foreign_origin(bridge):
    headers = {"Origin": "https://malicious.example"}
    assert (
        bridge.request(
            "/api/auth/login",
            method="POST",
            value={"username": "manager", "password": PASSWORD},
            headers=headers,
        )[0]
        == 403
    )
    assert (
        bridge.request(
            "/api/workspace",
            method="POST",
            value=example_workspace(),
            actor=bridge.manager,
            headers=headers,
        )[0]
        == 403
    )
    assert bridge.service.workspace()["agents"] == []
