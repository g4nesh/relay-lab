"""HTTP boundaries for readiness, durable sessions and private camera previews."""

import json
from http.client import HTTPResponse
from io import BytesIO
from types import SimpleNamespace

import pytest

from relay_lab.studio.access import AccessStore
from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.server import StudioHandler
from relay_lab.studio.service import example_workspace

SID = "studio-" + "a" * 16


class MemorySocket:
    def __init__(self, data):
        self.data, self.sent = data, bytearray()

    def makefile(self, *args):
        return BytesIO(self.data)

    def sendall(self, data):
        self.sent.extend(data)


class Service:
    def __init__(self):
        self.calls = []
        self.workspace = example_workspace()
        self.snapshot = None

    def health(self):
        return {"database": {"available": True}, "hardware_execution": False}

    def diagnostics(self, value):
        self.calls.append(("diagnostics", value))
        return {"ready": False, "checks": [{"id": "pair", "status": "blocked"}]}

    def list_sessions(self):
        return {"sessions": [{"session_id": SID}], "backend": "SpacetimeDB"}

    def history(self, sid):
        self.calls.append(("history", sid))
        return {
            "session_id": sid,
            "messages": [],
            "workspace": self.workspace,
            "model_release": self.snapshot,
            "resumable": True,
        }

    def recover(self, sid):
        self.calls.append(("recover", sid))
        return {
            "session_id": sid,
            "recovered": True,
            "model_release": self.snapshot,
            "resumable": True,
        }

    def sample_image(self, condition, arm):
        self.calls.append(("image", condition, arm))
        return b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def bridge(tmp_path):
    service = Service()
    access = AccessStore(tmp_path)
    credentials = access.setup(
        access.setup_path.read_text().strip(), "test-manager", "Test Manager", "test passphrase 123"
    )
    registry = ModelRegistry(tmp_path)
    catalog = registry.sync_workspace(credentials["user"], service.workspace)
    for model in catalog["models"]:
        registry.transition(credentials["user"], model["id"], "released", model["revision"])
    service.snapshot = registry.authorize_workspace(credentials["user"], service.workspace)

    def request(path, *, method="GET", value=None, headers=None):
        payload = b"" if value is None else json.dumps(value).encode()
        fields = {
            "Host": "127.0.0.1:8770",
            "Connection": "close",
            "Content-Length": str(len(payload)),
            "Content-Type": "application/json",
            "Cookie": "relay_studio_8770=" + credentials["token"],
            **(headers or {}),
        }
        if fields.get("X-Relay-Token") == "token":
            fields["X-Relay-Token"] = credentials["csrf"]
        raw = "\r\n".join([f"{method} {path} HTTP/1.1", *(f"{k}: {v}" for k, v in fields.items())])
        sock = MemorySocket((raw + "\r\n\r\n").encode() + payload)
        StudioHandler(
            sock,
            ("127.0.0.1", 1),
            SimpleNamespace(server_port=8770),
            service=service,
            access=access,
            registry=registry,
        )
        response = HTTPResponse(MemorySocket(bytes(sock.sent)), method=method)
        response.begin()
        return response.status, dict(response.getheaders()), response.read()

    return service, request


def test_health_and_durable_catalog_are_actual_service_reads(bridge):
    _, request = bridge
    status, _, body = request("/api/health")
    assert status == 200 and json.loads(body)["database"]["available"]
    status, _, body = request("/api/sessions")
    assert status == 200 and json.loads(body)["sessions"][0]["session_id"] == SID
    status, _, body = request(f"/api/sessions/{SID}")
    assert status == 200 and json.loads(body)["session_id"] == SID


@pytest.mark.parametrize("route", ["/api/diagnostics", f"/api/sessions/{SID}/resume"])
def test_diagnostics_and_recovery_require_session_csrf_token(bridge, route):
    service, request = bridge
    assert request(route, method="POST", value={})[0] == 403
    assert service.calls == []
    status, _, body = request(route, method="POST", value={}, headers={"X-Relay-Token": "token"})
    assert status == 200
    data = json.loads(body)
    assert data.get("recovered") is True or data.get("ready") is False


def test_legacy_session_is_readable_but_cannot_resume_without_releases(bridge):
    service, request = bridge
    service.snapshot = None
    status, _, body = request(f"/api/sessions/{SID}")
    data = json.loads(body)
    assert status == 200 and not data["release_ready"] and not data["resumable"]
    status, _, _ = request(
        f"/api/sessions/{SID}/resume", method="POST", value={}, headers={"X-Relay-Token": "token"}
    )
    assert status == 403
    assert all(call[0] != "recover" for call in service.calls)


def test_captured_image_route_returns_binary_without_text_charset(bridge):
    service, request = bridge
    status, headers, body = request("/api/samples/left_obstacle/right.png")
    assert status == 200 and body.startswith(b"\x89PNG")
    assert headers["Content-Type"] == "image/png"
    assert headers["Cache-Control"] == "no-store"
    assert service.calls == [("image", "left_obstacle", "right")]


@pytest.mark.parametrize(
    "route",
    [
        "/api/samples/../workspace.json",
        "/api/samples/clear/both.png",
        "/api/samples/unseen/right.png",
        "/api/sessions/studio-invalid",
        f"/api/sessions/{SID}/exchange/extra",
        f"/api/sessions/{SID}/resume",
    ],
)
def test_unknown_or_malformed_read_route_does_not_reach_service(bridge, route):
    service, request = bridge
    assert request(route)[0] == 404
    assert service.calls == []


def test_unavailable_database_is_visible_without_manufactured_health(bridge):
    service, request = bridge

    def unavailable():
        raise RuntimeError("Database offline")

    service.health = unavailable
    status, _, body = request("/api/health")
    assert status == 503 and "Database offline" in json.loads(body)["error"]


def test_camera_and_catalog_reject_foreign_origins(bridge):
    service, request = bridge
    for route in ("/api/samples/clear/left.png", "/api/sessions", "/api/health"):
        assert request(route, headers={"Origin": "https://unrelated.example"})[0] == 403
    assert service.calls == []
