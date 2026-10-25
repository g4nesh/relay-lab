"""Real authentication and origin checks through the production ASGI transport."""

import asyncio
import json
import threading
from http.cookies import SimpleCookie
from types import SimpleNamespace

import pytest

from relay_lab.studio.access import AccessStore
from relay_lab.studio.asgi import StudioApplication, create_app
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.service import StudioService


def request(app, path="/api/auth/status", *, method="GET", body=b"", headers=None, chunks=None):
    async def run():
        output = []
        events = chunks or [{"type": "http.request", "body": body}]

        async def receive():
            return events.pop(0)

        async def send(event):
            output.append(event)

        values = [(b"host", b"robots.example.com")]
        values.extend(headers or [])
        await app(
            {
                "type": "http",
                "method": method,
                "path": path,
                "headers": values,
                "query_string": b"",
            },
            receive,
            send,
        )
        return output

    output = asyncio.run(run())
    return output[0]["status"], dict(output[0]["headers"]), output[1]["body"]


@pytest.fixture
def runtime(tmp_path):
    service = StudioService(tmp_path, experiments=False)
    access = AccessStore(service.storage)
    registry = ModelRegistry(service.storage)
    return {
        "service": service,
        "access": access,
        "registry": registry,
        "analytics": None,
        "tasks": None,
        "workers": None,
    }


@pytest.fixture
def app(runtime, tmp_path):
    config = DeploymentConfig(
        public_origin="https://robots.example.com", bind="0.0.0.0", root=tmp_path
    )
    application = create_app(config=config, runtime=runtime)
    yield application
    asyncio.run(application.close())


@pytest.mark.parametrize(
    "origin",
    [
        "http://robots.example.com",
        "https://robots.example.com/",
        "https://u:p@example.com",
        "https://robots.example.com?q=x",
        "https://robots.example.com#x",
        "https://EXAMPLE.COM",
        "https://robots.example.com:bad",
        "https://robots.example.com:70000",
        "https://exa mple.com",
        "https://bad\\host.com",
        "https://empty..label.com",
    ],
)
def test_bad_public_origins_rejected(origin):
    with pytest.raises(ValueError):
        DeploymentConfig(public_origin=origin)


def test_non_loopback_requires_https():
    with pytest.raises(ValueError, match="non-loopback"):
        DeploymentConfig(bind="0.0.0.0")
    assert not DeploymentConfig().hosted
    assert DeploymentConfig(public_origin="https://robots.example.com", bind="0.0.0.0").hosted


def test_second_process_cannot_share_customer_volume(app, runtime):
    with pytest.raises(RuntimeError, match="one Studio process"):
        create_app(config=app.config, runtime=runtime)


def test_public_assets_and_authenticated_private_routes(app):
    assert request(app, "/")[0] == 200
    assert request(app, "/api/auth/status")[0] == 200
    assert request(app, "/api/analytics")[0] == 401
    assert request(app, "/api/team")[0] == 401
    assert request(app, "/api/coordination")[0] == 401


def test_scheduled_jobs_require_manager_and_csrf(app, runtime):
    access = runtime["access"]
    owner = access.setup(
        (runtime["service"].storage / "setup-code").read_text().strip(),
        "owner",
        "Owner",
        "correct horse robot battery",
    )
    invite = access.invite(owner["user"], "operator", "Operator", "employee")
    operator = access.accept_invite(invite["code"], "another robot battery password")
    calls = []

    class Coordination:
        def snapshot(self, realtime):
            return {"enabled": True, "jobs": []}

        def configure_job(self, value):
            calls.append(("configure", value))
            return value

        def run_job(self, value):
            calls.append(("run", value))
            return value

        def close(self):
            pass

    runtime["coordination"] = Coordination()
    for credentials in (operator, owner):
        auth = [(b"authorization", ("Bearer " + credentials["token"]).encode())]
        assert request(app, "/api/coordination", headers=auth)[0] == 200
        for route in ("/api/coordination/jobs", "/api/coordination/jobs/run"):
            payload = json.dumps({"job_id": "health", "enabled": False}).encode()
            assert request(app, route, method="POST", body=payload, headers=auth)[0] == 403
            assert (
                request(
                    app,
                    route,
                    method="POST",
                    body=payload,
                    headers=[*auth, (b"x-relay-token", b"invalid")],
                )[0]
                == 403
            )
            status, _, _ = request(
                app,
                route,
                method="POST",
                body=payload,
                headers=[
                    *auth,
                    (b"content-type", b"application/json"),
                    (b"x-relay-token", credentials["csrf"].encode()),
                ],
            )
            assert status == (200 if credentials is owner else 403)
    assert [kind for kind, _ in calls] == ["configure", "run"]
    assert (
        len(
            [
                event
                for event in access.audit_log(owner["user"])["events"]
                if event["action"].startswith("coordination.")
            ]
        )
        == 2
    )
    access.logout(owner["token"])
    for route in ("/api/coordination/jobs", "/api/coordination/jobs/run"):
        assert (
            request(
                app,
                route,
                method="POST",
                body=b'{"job_id":"health"}',
                headers=[
                    (b"authorization", ("Bearer " + owner["token"]).encode()),
                    (b"x-relay-token", owner["csrf"].encode()),
                ],
            )[0]
            == 401
        )
    assert len(calls) == 2


def test_wrong_host_and_forwarded_spoof_rejected(app):
    # Duplicate Host is rejected before any header can mask another one.
    assert request(app, headers=[(b"host", b"evil.test")])[0] == 400
    assert (
        request(
            app,
            headers=[
                (b"origin", b"https://evil.test"),
                (b"x-forwarded-host", b"robots.example.com"),
                (b"x-forwarded-proto", b"https"),
            ],
        )[0]
        == 403
    )
    with pytest.raises(PermissionError):
        app.config.guard({"Host": "evil.test", "X-Forwarded-Host": "robots.example.com"})


def test_setup_cookie_csrf_logout_preserved(app, runtime):
    code = (runtime["service"].storage / "setup-code").read_text().strip()
    payload = {
        "code": code,
        "username": "owner",
        "name": "Owner",
        "password": "correct horse robot battery",
    }
    status, headers, body = request(
        app,
        "/api/auth/setup",
        method="POST",
        body=json.dumps(payload).encode(),
        headers=[(b"content-type", b"application/json")],
    )
    assert status == 200
    assert b"__Host-relay_studio=" in headers[b"set-cookie"]
    assert b"Secure" in headers[b"set-cookie"] and b"HttpOnly" in headers[b"set-cookie"]
    cookie = SimpleCookie(headers[b"set-cookie"].decode())
    secret = cookie["__Host-relay_studio"].value
    csrf = json.loads(body)["token"]
    auth = [
        (b"cookie", f"__Host-relay_studio={secret}".encode()),
        (b"content-type", b"application/json"),
    ]
    assert request(app, "/api/auth/logout", method="POST", body=b"{}", headers=auth)[0] == 403
    status, headers, _ = request(
        app,
        "/api/auth/logout",
        method="POST",
        body=b"{}",
        headers=[*auth, (b"x-relay-token", csrf.encode())],
    )
    assert status == 200 and b"Secure" in headers[b"set-cookie"]
    assert request(app, "/api/bootstrap", headers=auth)[0] == 401


def test_streamed_and_declared_body_size_limits(app):
    assert request(app, method="POST", headers=[(b"content-length", b"131073")])[0] == 413
    assert (
        request(
            app,
            method="POST",
            chunks=[
                {"type": "http.request", "body": b"x" * 70000, "more_body": True},
                {"type": "http.request", "body": b"x" * 70000},
            ],
        )[0]
        == 413
    )
    assert request(app, method="POST", body=b"{}", headers=[(b"content-length", b"3")])[0] == 400
    assert app.inflight == 0


def test_hosted_uploads_disabled_and_duplicate_authorization_rejected(app):
    assert request(app, "/api/upload", method="POST")[0] == 403
    assert (
        request(app, headers=[(b"authorization", b"Bearer a"), (b"authorization", b"Bearer b")])[0]
        == 400
    )
    assert request(app, method="DELETE")[0] == 405


def test_readiness_reports_only_boolean(app, runtime):
    def fail():
        raise RuntimeError("secret token and private topology")

    runtime["service"].factory = fail
    status, _, body = request(app, "/readyz")
    assert status == 503 and json.loads(body) == {"ready": False}
    assert request(app, "/livez")[0] == 200
    runtime["service"].factory = lambda: SimpleNamespace(client=SimpleNamespace(_sql=lambda _: []))
    status, _, body = request(app, "/readyz")
    assert status == 200 and json.loads(body) == {"ready": True}


def test_busy_and_deadline_do_not_release_running_mutation(runtime, tmp_path):
    started, release = threading.Event(), threading.Event()
    config = DeploymentConfig(
        public_origin="https://robots.example.com",
        root=tmp_path,
        max_concurrent=1,
        request_timeout_seconds=0.02,
    )
    app = StudioApplication(runtime, config)

    def blocking(*args):
        started.set()
        release.wait(2)
        return 200, [], b"{}"

    app._dispatch = blocking

    async def exercise():
        output = []

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(event):
            output.append(event)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/auth/status",
            "headers": [(b"host", b"robots.example.com")],
        }
        await app(scope, receive, send)
        assert started.is_set() and output[0]["status"] == 504 and app.inflight == 1
        output.clear()
        await app(scope, receive, send)
        assert output[0]["status"] == 503 and app.inflight == 1
        release.set()
        await app.close()
        await asyncio.sleep(0)
        assert app.inflight == 0

    asyncio.run(exercise())


def test_body_deadline_releases_capacity(runtime, tmp_path):
    config = DeploymentConfig(
        public_origin="https://robots.example.com", root=tmp_path, body_timeout_seconds=0.01
    )
    app = create_app(config=config, runtime=runtime)

    async def exercise():
        output = []

        async def receive():
            await asyncio.sleep(1)

        async def send(event):
            output.append(event)

        await app(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/auth/login",
                "headers": [(b"host", b"robots.example.com")],
            },
            receive,
            send,
        )
        assert output[0]["status"] == 408 and app.inflight == 0
        await app.close()

    asyncio.run(exercise())


def test_hosted_failure_redacts_secrets(app, runtime):
    credentials = runtime["access"].setup(
        (runtime["service"].storage / "setup-code").read_text().strip(),
        "owner",
        "Owner",
        "correct horse robot battery",
    )

    def fail():
        raise RuntimeError("secret-deployment-token")

    runtime["service"].health = fail
    status, _, body = request(
        app,
        "/api/health",
        headers=[(b"authorization", ("Bearer " + credentials["token"]).encode())],
    )
    assert status == 503 and b"secret-deployment-token" not in body
