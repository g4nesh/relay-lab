"""Opt-in integration against a NEW isolated database containing this module.

RELAY_DURABLE_TEST_SERVER, RELAY_DURABLE_TEST_DATABASE and
RELAY_DURABLE_TEST_TOKEN_FILE must name a disposable test deployment.
"""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from http.cookies import SimpleCookie
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from relay_lab.spacetime import SpacetimeUnavailable
from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.asgi import create_app
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.service import example_workspace


def request(
    app, path, value=None, credentials=None, worker=None, host="robots.example.com"
):
    async def run():
        headers = [(b"host", host.encode()), (b"content-type", b"application/json")]
        if credentials:
            headers.extend(
                [
                    (b"cookie", credentials["cookie"].encode()),
                    (b"x-relay-token", credentials["csrf"].encode()),
                ]
            )
        if worker:
            headers.append((b"authorization", ("Bearer " + worker).encode()))
        body = b"" if value is None else json.dumps(value).encode()
        events = [{"type": "http.request", "body": body}]
        out = []

        async def receive():
            return events.pop(0)

        async def send(event):
            out.append(event)

        await app(
            {
                "type": "http",
                "method": "GET" if value is None else "POST",
                "path": path,
                "headers": headers,
                "query_string": b"",
            },
            receive,
            send,
        )
        content = out[1]["body"]
        return out[0]["status"], dict(out[0]["headers"]), json.loads(content)

    return asyncio.run(run())


def test_real_spacetime_restart_auth_governance_workers_tasks_and_failures(
    tmp_path, monkeypatch
):
    server = os.environ.get("RELAY_DURABLE_TEST_SERVER")
    if not server:
        pytest.skip("Requires explicitly configured isolated Spacetime database")
    monkeypatch.setenv("RELAY_SPACETIME_SERVER", server)
    database = os.environ["RELAY_DURABLE_TEST_DATABASE"] + "-" + uuid.uuid4().hex[:8]
    subprocess.run(
        [
            sys.executable,
            "deploy/publish-spacetime.py",
            "--server",
            server,
            "--database",
            database,
            "--token-file",
            os.environ["RELAY_DURABLE_TEST_TOKEN_FILE"],
            "--bundle",
            "spacetimedb/dist/bundle.js",
        ],
        check=True,
    )
    monkeypatch.setenv("RELAY_SPACETIME_DATABASE", database)
    monkeypatch.setenv(
        "RELAY_SPACETIME_TOKEN_FILE", os.environ["RELAY_DURABLE_TEST_TOKEN_FILE"]
    )
    monkeypatch.delenv("RELAY_SPACETIME_TOKEN", raising=False)
    monkeypatch.setenv("RELAY_STUDIO_STORAGE", "spacetime")
    apps = []

    def start(name):
        app = create_app(
            config=DeploymentConfig(
                root=tmp_path / name,
                public_origin="https://robots.example.com",
                bind="0.0.0.0",
            )
        )
        apps.append(app)
        return app

    def checked(app, path, value=None, *, credentials=None, worker=None, status=200):
        actual, headers, data = request(app, path, value, credentials, worker)
        assert actual == status, (path, actual, data)
        return headers, data

    try:
        app = start("first-disk")
        assert checked(app, "/readyz")[1]["ready"]
        assert checked(app, "/api/auth/status")[1]["setup_required"]
        code = app.runtime["access"].setup_path.read_text().strip()
        headers, data = checked(
            app,
            "/api/auth/setup",
            {
                "code": code,
                "username": "testmanager",
                "name": "Test Manager",
                "password": "test-only-strong-passphrase",
            },
        )
        cookie = SimpleCookie(headers[b"set-cookie"].decode())
        credentials = {
            "cookie": "__Host-relay_studio=" + cookie["__Host-relay_studio"].value,
            "csrf": data["token"],
        }
        assert b"Secure" in headers[b"set-cookie"]
        workspace = example_workspace()
        workspace.update(name="Durable restart proof", mode="mediated")
        checked(app, "/api/workspace", workspace, credentials=credentials)
        records = checked(app, "/api/models", credentials=credentials)[1]["models"]
        for model in records:
            checked(
                app,
                f"/api/models/{model['id']}/release",
                {"expected_revision": model["revision"], "reason": "Integration test"},
                credentials=credentials,
            )
        worker = checked(
            app,
            "/api/workers",
            {
                "name": "Test observer",
                "agent_id": "left",
                "execution_kind": "observation",
            },
            credentials=credentials,
            status=201,
        )[1]
        goal = checked(
            app,
            "/api/tasks",
            {
                "request_id": "create-live",
                "objective": "Inspect the fixture",
                "success_criteria": ["Observation recorded"],
                "target": "left",
            },
            credentials=credentials,
            status=201,
        )[1]
        cancelled = checked(
            app,
            "/api/tasks",
            {
                "request_id": "cancel-create-live",
                "objective": "Task that will be cancelled",
                "success_criteria": ["Stopped safely"],
                "target": "left",
            },
            credentials=credentials,
            status=201,
        )[1]
        cancelled = checked(
            app,
            f"/api/tasks/{cancelled['id']}/cancel",
            {
                "request_id": "cancel-live",
                "expected_revision": 0,
                "reason": "Test cancellation",
            },
            credentials=credentials,
        )[1]
        assert cancelled["status"] == "cancelled"
        checked(
            app,
            f"/api/agent/tasks/{cancelled['id']}/claim",
            {"request_id": "late-claim", "expected_revision": cancelled["revision"]},
            worker=worker["token"],
            status=409,
        )
        pending_before = checked(app, "/api/coordination", credentials=credentials)[1]["pending_updates"]
        assert pending_before > 0
        registry_id = app.runtime["registry"].registry_id
        asyncio.run(app.close())

        # Completely different local directory: no SQLite files or setup code copied.
        app = start("empty-replacement-disk")
        assert app.runtime["registry"].registry_id == registry_id
        assert checked(app, "/api/coordination", credentials=credentials)[1]["pending_updates"] == pending_before
        app.runtime["coordination"].flush()
        coordination = checked(app, "/api/coordination", credentials=credentials)[1]
        assert coordination["available"], coordination
        assert coordination["pending_updates"] == 0
        assert {j["job_id"] for j in coordination["jobs"]} == {"health", "rollup", "cleanup"}
        checked(app, "/api/coordination/jobs", {
            "job_id": "health", "schedule": "daily", "enabled": False,
        }, credentials=credentials)
        assert checked(app, "/api/auth/status", credentials=credentials)[1][
            "authenticated"
        ]
        assert not app.runtime["access"].setup_path.exists()
        assert all(
            m["status"] == "released"
            for m in checked(app, "/api/models", credentials=credentials)[1]["models"]
        )
        assert (
            checked(app, f"/api/tasks/{goal['id']}", credentials=credentials)[1][
                "objective"
            ]
            == goal["objective"]
        )
        checked(
            app,
            "/api/agent/heartbeat",
            {"controller_label": "Test observer"},
            worker=worker["token"],
        )
        claimed = checked(
            app,
            f"/api/agent/tasks/{goal['id']}/claim",
            {"request_id": "claim-live", "expected_revision": 0},
            worker=worker["token"],
        )[1]
        report = {
            "request_id": "report-live",
            "expected_revision": claimed["revision"],
            "status": "succeeded",
            "progress": 100,
            "summary": "Fixture observed",
            "evidence": ["test-frame"],
        }
        finished = checked(
            app, f"/api/agent/tasks/{goal['id']}/report", report, worker=worker["token"]
        )[1]
        assert finished["status"] == "succeeded"
        assert finished["success_basis"] == "agent_reported"
        duplicate = checked(
            app, f"/api/agent/tasks/{goal['id']}/report", report, worker=worker["token"]
        )[1]
        assert duplicate["revision"] == finished["revision"]

        # A failed checkpoint must suppress the successful HTTP response, and
        # the next request restores the previously committed workspace.
        original = app.durable.client._request

        def fail(endpoint, body, **kwargs):
            if "/call/studio_commit_state?" in endpoint:
                raise SpacetimeUnavailable("injected outage")
            return original(endpoint, body, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(app.durable.client, "_request", fail)
            checked(
                app,
                "/api/workspace",
                {**workspace, "name": "must-not-stick"},
                credentials=credentials,
                status=503,
            )
        assert (
            checked(app, "/api/bootstrap", credentials=credentials)[1]["workspace"][
                "name"
            ]
            == workspace["name"]
        )
        assert request(app, "/api/auth/status", host="attacker.example")[0] == 403
        # A second instance has an old local credential cache. Each stream
        # authorization must restore current Spacetime authority first.
        stale_app = start("stale-stream-cache")
        stream_scope = {"method": "GET", "path": "/api/agent/events", "headers": [
            (b"host", b"robots.example.com"),
            (b"authorization", ("Bearer " + worker["token"]).encode()),
        ]}
        stale_app._authorize_events(stream_scope)
        checked(
            app,
            f"/api/workers/{worker['worker']['id']}/revoke",
            {},
            credentials=credentials,
        )
        asyncio.run(app.close())
        with pytest.raises((AuthenticationError, PermissionError)):
            stale_app._authorize_events(stream_scope)
        app = start("third-empty-disk")
        app.runtime["coordination"].flush()
        coordination = checked(app, "/api/coordination", credentials=credentials)[1]
        health = next(j for j in coordination["jobs"] if j["job_id"] == "health")
        assert health["enabled"] is False and health["schedule"] == "daily"
        assert next(r for r in coordination["presence"] if r["worker_id"] == worker["worker"]["id"])["status"] == "revoked"
        assert request(app, "/api/agent/tasks", worker=worker["token"])[0] in (401, 403)
        assert (
            checked(app, f"/api/tasks/{goal['id']}", credentials=credentials)[1][
                "status"
            ]
            == "succeeded"
        )

        # Database enforcement rejects a stale lease and legacy unfenced writer.
        client = app.durable.client
        app.durable._call("studio_acquire", ["live-fence-owner"])

        def rejected(name, args, message):
            before = app.durable.row()
            req = Request(
                client._base + "/call/" + name + "?confirmed=true",
                data=json.dumps(args).encode(),
                method="POST",
                headers={
                    "Authorization": "Bearer " + client.config.token,
                    "Content-Type": "application/json",
                },
            )
            with pytest.raises(HTTPError) as error:
                client._opener.open(req, timeout=5)
            assert error.value.code == 530
            assert message in error.value.read().decode()
            assert app.durable.row() == before

        rejected(
            "studio_create_factory_run",
            ["stale-holder", "x", "x", "{}", "{}"],
            "Studio request lease expired or superseded",
        )
        rejected(
            "create_factory_run",
            ["x", "x", "{}", "{}"],
            "Studio storage requires a fenced request",
        )
        rejected(
            "studio_commit_state",
            ["stale-holder", 0, "eA==", "stale-operation"],
            "Studio request lease expired or superseded",
        )
        rejected(
            "studio_fenced_configure_job",
            ["stale-holder", registry_id, "health", True, "hourly", 7],
            "Studio request lease expired or superseded",
        )
        rejected(
            "studio_configure_job", [registry_id, "health", True, "hourly", 7],
            "Studio storage requires a fenced request",
        )
        app.durable._call("studio_release", ["live-fence-owner"])
        checked(app, "/readyz")
    finally:
        for app in apps:
            if not app.closed:
                asyncio.run(app.close())
