"""Loopback-only Studio UI and a small bridge to existing coordination services."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import struct
import uuid
from datetime import datetime
from functools import partial
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .access import AccessStore, AuthenticationError
from .analytics import AnalyticsService, estimate_effort
from .governance import ModelRegistry, RegistryConflict
from .service import StudioService, example_workspace
from .tasks import TaskConflict, TaskService
from .workers import WorkerStore

WEB = Path(__file__).parent / "web"
ASSETS = {
    "/": ("index.html", "text/html"),
    "/studio.css": ("studio.css", "text/css"),
    "/studio.js": ("studio.js", "text/javascript"),
    "/analytics.js": ("analytics.js", "text/javascript"),
    "/analytics.css": ("analytics.css", "text/css"),
    "/tasks.js": ("tasks.js", "text/javascript"),
    "/tasks.css": ("tasks.css", "text/css"),
}
MAX_UPLOAD = 4 * 1024**3
SESSION_ROUTE = r"/api/sessions/(studio-[a-f0-9]{16})"


def validate_safetensors(path):
    """Check the inert file envelope without importing or executing model code."""
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("Not a safetensors file")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(16 * 1024**2, size - 8):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(f.read(length))
    if not isinstance(header, dict):
        raise TypeError("Invalid safetensors header")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("The weights file contains no tensors")
    for tensor in tensors.values():
        if not isinstance(tensor, dict) or not {"dtype", "shape", "data_offsets"} <= tensor.keys():
            raise ValueError("Invalid tensor metadata")
        offsets = tensor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
        ):
            raise ValueError("Invalid tensor offsets")
        if not 0 <= offsets[0] <= offsets[1] <= size - 8 - length:
            raise ValueError("Tensor data is incomplete")
    return len(tensors)


class StudioHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        *args,
        service,
        access,
        registry,
        analytics=None,
        deployment=None,
        tasks=None,
        workers=None,
        **kwargs,
    ):
        self.service, self.access, self.registry = service, access, registry
        self.analytics = analytics
        self.deployment = deployment
        self.tasks = tasks
        self.workers = workers
        self.identity = None
        super().__init__(*args, **kwargs)

    def respond(self, value, status=200, mime="application/json", *, cookie=None):
        body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        charset = (
            "; charset=utf-8" if mime.startswith("text/") or mime == "application/json" else ""
        )
        self.send_header("Content-Type", mime + charset)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def guard(self):
        if self.deployment is not None:
            self.deployment.guard(self.headers, self.server.server_port)
            return
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            raise PermissionError("Studio accepts only its local origin")
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + h for h in hosts}:
            raise PermissionError("Cross-origin requests are not allowed")

    @property
    def cookie_name(self):
        if self.deployment is not None and self.deployment.hosted:
            return "__Host-relay_studio"
        return f"relay_studio_{self.server.server_port}"

    @property
    def cookie_security(self):
        return "; Secure" if self.deployment is not None and self.deployment.hosted else ""

    def credential(self):
        authorization = self.headers.get("Authorization", "")
        if authorization:
            if not authorization.startswith("Bearer "):
                raise AuthenticationError("Sign in to Relay Studio")
            return authorization.removeprefix("Bearer ")
        cookie = SimpleCookie()
        cookie.load(self.headers.get("Cookie", ""))
        value = cookie.get(self.cookie_name)
        return value.value if value else ""

    def authenticate(self, mutation=False):
        self.identity = self.access.authenticate(self.credential())
        if mutation and not secrets.compare_digest(
            self.headers.get("X-Relay-Token", ""), self.identity["csrf"]
        ):
            raise PermissionError("Refresh the page to obtain your session's request token")
        return self.identity["user"]

    def require(self, permission):
        if permission not in self.identity["permissions"]:
            raise PermissionError("A manager must approve this operation")

    def auth_response(self, credentials):
        cookie = (
            f"{self.cookie_name}={credentials['token']}; Path=/; HttpOnly; "
            f"SameSite=Strict; Max-Age=28800{self.cookie_security}"
        )
        return self.respond(
            {
                "user": credentials["user"],
                "permissions": credentials["permissions"],
                "token": credentials["csrf"],
            },
            cookie=cookie,
        )

    def release_status(self, actor, value, *, snapshot=False):
        try:
            if snapshot:
                self.registry.authorize_snapshot(actor, value)
            else:
                self.registry.authorize_workspace(actor, value)
            return {"release_ready": True, "release_error": None}
        except (PermissionError, ValueError) as error:
            return {"release_ready": False, "release_error": str(error)}

    def session_status(self, actor, history):
        history.update(self.release_status(actor, history.get("model_release"), snapshot=True))
        history["resumable"] = bool(history.get("resumable") and history["release_ready"])
        return history

    def authorize_history(self, actor, history):
        snapshot = self.registry.authorize_snapshot(actor, history.get("model_release"))
        if [model["spec"] for model in snapshot["models"]] != history["workspace"]["agents"]:
            raise PermissionError("Session configuration differs from its approved model releases")

    def body(self):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("Expected application/json")
        length = int(self.headers.get("Content-Length", "0"))
        if not 1 <= length <= 128 * 1024:
            raise ValueError("Request must be 1–131072 bytes")
        value = json.loads(
            self.rfile.read(length),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
        )
        if not isinstance(value, dict):
            raise TypeError("Expected a JSON object")
        return value

    def dispatch(self, mutation=False):
        path = urlsplit(self.path).path
        # Check identity *after* entering the governance boundary. A request
        # queued behind inference must not retain a role revoked while waiting.
        # Team updates share this boundary so a committed demotion cannot race
        # an approved write. Logout remains immediate; uploads recheck on audit.
        serialized = mutation and path not in (
            "/api/auth/login",
            "/api/auth/setup",
            "/api/auth/join",
            "/api/auth/logout",
            "/api/upload",
            "/api/diagnostics",
            "/api/analytics/estimate",
        )
        if serialized:
            self.guard()
            with self.registry.lock:
                return self.dispatch_authorized(mutation)
        return self.dispatch_authorized(mutation)

    def dispatch_authorized(self, mutation=False):
        self.guard()
        path = urlsplit(self.path).path
        if not mutation and path in ASSETS:
            filename, mime = ASSETS[path]
            return self.respond((WEB / filename).read_bytes(), mime=mime)
        if not mutation and path == "/api/auth/status":
            result = {**self.access.status(), "authenticated": False}
            try:
                actor = self.authenticate()
                result.update(
                    authenticated=True, user=actor, permissions=self.identity["permissions"]
                )
            except AuthenticationError:
                pass
            return self.respond(result)
        if mutation and path in ("/api/auth/login", "/api/auth/setup", "/api/auth/join"):
            value = self.body()
            if path.endswith("/login"):
                credentials = self.access.login(value.get("username"), value.get("password"))
            elif path.endswith("/setup"):
                credentials = self.access.setup(
                    value.get("code"),
                    value.get("username"),
                    value.get("name"),
                    value.get("password"),
                )
            else:
                credentials = self.access.accept_invite(value.get("code"), value.get("password"))
            if path.endswith("/setup"):
                self.registry.sync_workspace(credentials["user"], self.service.workspace())
            return self.auth_response(credentials)
        if path.startswith("/api/agent/"):
            return self.agent_request(path, mutation)
        actor = self.authenticate(mutation)
        if not mutation:
            if path == "/api/tasks":
                return self.respond(self.task_service.list(actor))
            if match := re.fullmatch(r"/api/tasks/(goal-[a-f0-9]{16})", path):
                return self.respond(self.task_service.get(actor, match[1]))
            if path == "/api/workers":
                return self.respond(self.worker_store.list(actor))
            if path == "/api/analytics":
                analytics = self.analytics or AnalyticsService(
                    self.service.root, self.service, self.registry, self.access
                )
                return self.respond(analytics.snapshot(actor))
            if path == "/api/bootstrap":
                workspace = self.service.workspace()
                return self.respond(
                    {
                        "token": self.identity["csrf"],
                        "user": actor,
                        "permissions": self.identity["permissions"],
                        "workspace": workspace,
                        "example": example_workspace(),
                        "capabilities": self.service.capabilities(),
                        **self.registry.catalog(actor),
                        **self.release_status(actor, workspace),
                    }
                )
            if path == "/api/models":
                return self.respond(
                    {
                        **self.registry.catalog(actor),
                        **self.release_status(actor, self.service.workspace()),
                    }
                )
            if path == "/api/requests":
                return self.respond(self.registry.requests(actor))
            if path == "/api/team":
                return self.respond(self.access.list_users(actor))
            if path == "/api/audit":
                self.require("audit.read")
                events = (
                    self.access.audit_log(actor)["events"]
                    + self.registry.audit_log(actor)["events"]
                )
                for event in events:
                    timestamp = event.get("created_at", event.get("at", 0))
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp).timestamp()
                    event["created_at"] = timestamp
                events.sort(key=lambda event: event["created_at"], reverse=True)
                return self.respond({"events": events[:100]})
            if path == "/api/health":
                return self.respond(self.service.health())
            if path == "/api/sessions":
                result = self.service.list_sessions()
                for session in result["sessions"]:
                    self.session_status(actor, session)
                return self.respond(result)
            if match := re.fullmatch(SESSION_ROUTE, path):
                return self.respond(self.session_status(actor, self.service.history(match[1])))
            if match := re.fullmatch(
                r"/api/samples/(clear|left_obstacle|right_obstacle)/(left|right)\.png", path
            ):
                return self.respond(self.service.sample_image(match[1], match[2]), mime="image/png")
        else:
            if path == "/api/upload":
                self.require("models.manage")
                return self.upload()
            value = self.body()
            if path == "/api/tasks":
                self.require("sessions.run")
                return self.respond(self.task_service.create(actor, value), 201)
            if match := re.fullmatch(r"/api/tasks/(goal-[a-f0-9]{16})/cancel", path):
                self.require("sessions.run")
                return self.respond(self.task_service.cancel(actor, match[1], value))
            if path == "/api/workers":
                self.require("models.manage")
                return self.respond(self.worker_store.create(actor, value), 201)
            if match := re.fullmatch(r"/api/workers/(worker-[a-f0-9]{16})/revoke", path):
                self.require("models.manage")
                return self.respond(self.worker_store.revoke(actor, match[1]))
            if path == "/api/analytics/estimate":
                return self.respond(estimate_effort(value))
            if path == "/api/requests":
                return self.respond(
                    self.registry.submit_request(
                        actor, value.get("kind"), value.get("message"), value.get("model_id")
                    ),
                    201,
                )
            if match := re.fullmatch(r"/api/requests/(req_[a-f0-9]{32})/resolve", path):
                return self.respond(
                    self.registry.resolve_request(
                        actor, match[1], value.get("status"), value.get("reason", "")
                    )
                )
            if path == "/api/auth/logout":
                self.access.logout(self.credential())
                return self.respond(
                    {"signed_out": True},
                    cookie=(
                        f"{self.cookie_name}=; Path=/; HttpOnly; SameSite=Strict; "
                        f"Max-Age=0{self.cookie_security}"
                    ),
                )
            if path == "/api/team/invitations":
                return self.respond(
                    self.access.invite(
                        actor, value.get("username"), value.get("name"), value.get("role")
                    ),
                    201,
                )
            if match := re.fullmatch(r"/api/team/([a-f0-9-]{16,36})", path):
                return self.respond(
                    self.access.update_user(actor, match[1], value.get("role"), value.get("active"))
                )
            if match := re.fullmatch(r"/api/models/(mdl_[a-f0-9]{64})/(release|retire)", path):
                self.require("models.release")
                return self.respond(
                    self.registry.transition(
                        actor,
                        match[1],
                        "released" if match[2] == "release" else "retired",
                        value.get("expected_revision"),
                        value.get("reason", ""),
                    )
                )
            if path == "/api/workspace":
                self.require("workspace.edit")
                workspace = self.service.validate(value)
                with self.registry.lock:
                    self.registry.sync_workspace(actor, workspace)
                    saved = self.service.save(workspace)
                    self.access.audit(
                        actor, "workspace.saved", "workspace", {"name": saved["name"]}
                    )
                return self.respond(saved)
            if path == "/api/diagnostics":
                result = self.service.diagnostics(value)
                release = self.release_status(actor, value)
                result.update(release)
                result["checks"].append(
                    {
                        "id": "release",
                        "label": "Manager release approval",
                        "status": "ready" if release["release_ready"] else "blocked",
                        "detail": release["release_error"]
                        or "Both model revisions are approved for Studio use.",
                    }
                )
                result["ready"] = bool(result["ready"] and release["release_ready"])
                return self.respond(result)
            if path == "/api/sessions":
                self.require("sessions.run")
                with self.registry.lock:
                    workspace = self.service.validate(value, complete=True)
                    if workspace != self.service.workspace():
                        raise PermissionError(
                            "Use the saved workspace; a manager must save configuration changes"
                        )
                    release = self.registry.authorize_workspace(actor, workspace)
                    result = self.service.connect(workspace, model_release=release)
                    self.access.audit(actor, "session.created", result["session_id"])
                    return self.respond({**result, "release_ready": True}, 201)
            if match := re.fullmatch(SESSION_ROUTE + r"/exchange", path):
                self.require("sessions.run")
                with self.registry.lock:
                    history = self.service.history(match[1])
                    self.authorize_history(actor, history)
                    return self.respond(self.service.exchange(match[1], value))
            if match := re.fullmatch(SESSION_ROUTE + r"/resume", path):
                self.require("sessions.run")
                with self.registry.lock:
                    history = self.service.history(match[1])
                    self.authorize_history(actor, history)
                    return self.respond(self.session_status(actor, self.service.recover(match[1])))
        self.respond({"error": "Not found"}, 404)

    @property
    def task_service(self):
        if self.tasks is None:
            self.tasks = TaskService(self.service, self.registry)
        return self.tasks

    @property
    def worker_store(self):
        if self.workers is None:
            self.workers = WorkerStore(
                self.service.storage, self.access, self.registry, self.service
            )
        return self.workers

    def agent_request(self, path, mutation):
        # Worker tokens never inherit browser authority and cannot create tasks,
        # approve releases, change accounts, or read another runtime's inbox.
        authorization = self.headers.get("Authorization", "")
        if not authorization.startswith("Bearer relay_worker_"):
            raise AuthenticationError("Use an agent connection token")
        worker = self.worker_store.authenticate(authorization.removeprefix("Bearer "))
        if not mutation:
            if path == "/api/agent/me":
                return self.respond({"worker": worker})
            if path == "/api/agent/tasks":
                return self.respond(self.task_service.inbox(worker))
            if match := re.fullmatch(r"/api/agent/tasks/(goal-[a-f0-9]{16})", path):
                return self.respond(self.task_service.worker_get(worker, match[1]))
        else:
            value = self.body()
            if path == "/api/agent/heartbeat":
                return self.respond(self.worker_store.heartbeat(worker, value))
            if match := re.fullmatch(r"/api/agent/tasks/(goal-[a-f0-9]{16})/(claim|report)", path):
                method = (
                    self.task_service.claim if match[2] == "claim" else self.task_service.report
                )
                return self.respond(method(worker, match[1], value))
        return self.respond({"error": "Not found"}, 404)

    def upload(self):
        if self.deployment is not None and self.deployment.hosted:
            raise PermissionError(
                "Upload model weights on the agent host; hosted uploads are disabled"
            )
        name = unquote(self.headers.get("X-Filename", ""))
        if (
            not name
            or len(name) > 200
            or Path(name).name != name
            or "\\" in name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Upload a .safetensors weights file")
        length = int(self.headers.get("Content-Length", "0"))
        if not 10 <= length <= MAX_UPLOAD:
            raise ValueError(
                "Weights must be between 10 bytes and 4 GiB; use a local path for larger files"
            )
        asset = uuid.uuid4().hex + ".safetensors"
        path = self.service.uploads / asset
        digest, remaining = hashlib.sha256(), length
        try:
            with path.open("xb") as output:
                while remaining:
                    block = self.rfile.read(min(1024**2, remaining))
                    if not block:
                        raise ValueError("Upload was interrupted")
                    output.write(block)
                    digest.update(block)
                    remaining -= len(block)
            tensors = validate_safetensors(path)
            metadata = {
                "asset": asset,
                "name": name,
                "bytes": length,
                "sha256": digest.hexdigest(),
                "tensors": tensors,
                "loaded": False,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata))
            self.access.audit(
                self.identity["user"],
                "model.uploaded",
                asset,
                {"sha256": metadata["sha256"], "bytes": length},
            )
            self.respond(metadata, 201)
        except Exception:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            raise

    def handle_request(self, mutation=False):
        try:
            self.dispatch(mutation)
        except AuthenticationError as error:
            self.respond({"error": str(error)}, 401)
        except (RegistryConflict, TaskConflict) as error:
            self.respond({"error": str(error)}, 409)
        except PermissionError as error:
            self.respond({"error": str(error)}, 403)
        except (ValueError, TypeError, KeyError) as error:
            self.respond({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:  # noqa: BLE001 - report failed services without a fallback
            # Surface failures; never manufacture successful inference/DB results.
            detail = (
                "The service could not complete this request; inspect the operation before retrying"
                if self.deployment is not None and self.deployment.hosted
                else f"{type(error).__name__}: {error}"
            )
            self.respond({"error": detail}, 503)

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request(True)

    def log_message(self, *args):
        pass


def serve_studio(root=None, port=8770):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid Studio port")
    service = StudioService(root or Path.cwd())
    access = AccessStore(service.storage)
    registry = ModelRegistry(service.storage)
    analytics = AnalyticsService(service.root, service, registry, access)
    tasks = TaskService(service, registry)
    workers = WorkerStore(service.storage, access, registry, service)
    handler = partial(
        StudioHandler,
        service=service,
        access=access,
        registry=registry,
        analytics=analytics,
        tasks=tasks,
        workers=workers,
    )
    try:
        with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
            print(f"Relay Studio: http://127.0.0.1:{server.server_port}/", flush=True)
            if access.status()["setup_required"]:
                print(f"First manager setup code: {service.storage / 'setup-code'}", flush=True)
            server.serve_forever()
    finally:
        service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    options = parser.parse_args()
    serve_studio(options.root, options.port)
