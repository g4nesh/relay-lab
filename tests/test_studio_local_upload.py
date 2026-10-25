"""Local research uploads stay bounded and authorize before receiving model bytes."""

import asyncio
import json
import struct
import threading

import pytest

from relay_lab.studio import asgi, server
from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.profiles import ProfileStore
from relay_lab.studio.service import StudioService


def weights(size=200_000):
    header = json.dumps({"weight": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * size


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # Exercise transport and safetensors handling without loading research
    # packages. The same suite must run in the isolated dashboard export.
    service = StudioService(tmp_path, experiments=False)
    service.experiments_enabled = True
    access = ProfileStore(service.storage)
    profiles = access.profiles()
    credentials = {
        p["role"]: access.select_profile(p["id"], profiles["nonce"])
        for p in profiles["profiles"]
    }
    runtime = {"service": service, "access": access, "registry": ModelRegistry(service.storage),
               "analytics": None, "tasks": None, "workers": None, "coordination": None}
    app = asgi.create_app(config=DeploymentConfig(root=tmp_path), runtime=runtime)
    streams = []
    original = asgi.tempfile.TemporaryFile

    def tracked(**kwargs):
        stream = original(**kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(asgi.tempfile, "TemporaryFile", tracked)
    yield app, credentials, streams
    asyncio.run(app.close())


async def upload(app, credentials, *, body=None, headers=None, events=None, receiver=None):
    body = weights() if body is None else body
    values = {"host": app.config.authority, "content-length": str(len(body)),
              "x-filename": "weights.safetensors"}
    if credentials:
        values.update(authorization="Bearer " + credentials["token"],
                      **{"x-relay-token": credentials["csrf"]})
    values.update(headers or {})
    chunks = events if events is not None else [
        {"type": "http.request", "body": body[i:i + 32768], "more_body": i + 32768 < len(body)}
        for i in range(0, len(body), 32768)
    ]
    output = []
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        return await receiver() if receiver else chunks.pop(0)

    async def send(event):
        output.append(event)

    await app({"type": "http", "method": "POST", "path": "/api/upload",
               "headers": [(k.encode(), v.encode()) for k, v in values.items() if v is not None],
               "query_string": b""}, receive, send)
    status = output[0]["status"] if output else None
    result = json.loads(output[1]["body"]) if len(output) > 1 else None
    return status, result, reads


def test_valid_large_local_safetensors_are_streamed_and_validated(setup):
    app, people, streams = setup
    body = weights()
    status, result, reads = asyncio.run(upload(app, people["manager"], body=body))
    assert status == 201 and reads > 1
    assert result["bytes"] == len(body) and result["tensors"] == 1
    assert (app.runtime["service"].uploads / result["asset"]).read_bytes() == body
    assert len(streams) == 1 and streams[0].closed
    assert app.inflight == app.upload_inflight == 0


@pytest.mark.parametrize("case, expected", [
    ("anonymous", 401), ("employee", 403), ("csrf", 403), ("origin", 403),
    ("hosted", 403), ("control", 403), ("oversize", 413), ("length", 400),
    ("missing", 411),
])
def test_denials_happen_before_reading_or_spooling(setup, case, expected):
    app, people, streams = setup
    credentials = people["manager"]
    headers = {}
    if case == "anonymous":
        credentials = None
    elif case == "employee":
        credentials = people["employee"]
    elif case == "csrf":
        headers["x-relay-token"] = "wrong"
    elif case == "origin":
        headers["origin"] = "https://untrusted.example.com"
    elif case == "hosted":
        app.config = DeploymentConfig(public_origin="https://robots.example.com", bind="0.0.0.0")
    elif case == "control":
        app.runtime["service"].disable_experiments()
    elif case == "oversize":
        headers["content-length"] = str(asgi.MAX_UPLOAD + 1)
    elif case == "length":
        headers["content-length"] = "invalid"
    elif case == "missing":
        headers["content-length"] = None
    status, _, reads = asyncio.run(upload(app, credentials, headers=headers))
    assert status == expected and reads == 0 and streams == []
    assert app.inflight == app.upload_inflight == 0


@pytest.mark.parametrize("case, expected", [("malformed", 400), ("overrun", 413),
                                           ("short", 400), ("disconnect", None)])
def test_invalid_and_interrupted_uploads_clean_up(setup, case, expected):
    app, people, streams = setup
    body, headers, events = weights(), {}, None
    if case == "malformed":
        body = b"not safetensors" * 20_000
    elif case == "overrun":
        headers["content-length"] = str(len(body) - 1)
    elif case == "short":
        headers["content-length"] = str(len(body) + 1)
    else:
        events = [{"type": "http.request", "body": body[:100], "more_body": True},
                  {"type": "http.disconnect"}]
    status, _, _ = asyncio.run(upload(app, people["manager"], body=body, headers=headers, events=events))
    assert status == expected
    assert len(streams) == 1 and streams[0].closed
    assert list(app.runtime["service"].uploads.iterdir()) == []
    assert app.inflight == app.upload_inflight == 0


def test_body_timeout_cleans_up(setup):
    app, people, streams = setup
    app.upload_body_timeout_seconds = 0.01

    async def never():
        await asyncio.Event().wait()

    status, _, _ = asyncio.run(upload(app, people["manager"], receiver=never))
    assert status == 408 and streams[0].closed
    assert app.inflight == app.upload_inflight == 0


def test_authority_is_rechecked_before_existing_upload_handler(setup, monkeypatch):
    app, people, streams = setup
    original = app.runtime["access"].authenticate
    calls = 0

    def authenticate(token):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AuthenticationError("Session revoked")
        return original(token)

    monkeypatch.setattr(app.runtime["access"], "authenticate", authenticate)
    status, _, reads = asyncio.run(upload(app, people["manager"]))
    assert status == 401 and reads > 0 and streams[0].closed
    assert list(app.runtime["service"].uploads.iterdir()) == []


def test_executor_timeout_retains_spool_and_slot_until_handler_finishes(setup, monkeypatch):
    app, people, streams = setup
    app.config = DeploymentConfig(root=app.config.root, request_timeout_seconds=0.02)
    entered, release = threading.Event(), threading.Event()
    original = app._dispatch

    def delayed(scope, body):
        entered.set()
        assert release.wait(2)
        assert not body.stream.closed
        return original(scope, body)

    monkeypatch.setattr(app, "_dispatch", delayed)

    async def run():
        try:
            status, _, _ = await upload(app, people["manager"])
            assert status == 504 and entered.is_set()
            assert not streams[0].closed and app.upload_inflight == app.inflight == 1
            status, _, reads = await upload(app, people["manager"])
            assert status == 503 and reads == 0 and len(streams) == 1
        finally:
            release.set()
        for _ in range(100):
            if app.upload_inflight == 0:
                break
            await asyncio.sleep(0.01)
        assert streams[0].closed and app.inflight == app.upload_inflight == 0

    asyncio.run(run())


def test_ordinary_json_limit_remains_128_kib(setup):
    app, _, streams = setup

    async def run():
        output = []

        async def receive():
            raise AssertionError("Oversized body must not be read")

        async def send(event):
            output.append(event)

        await app({"type": "http", "method": "POST", "path": "/api/workspace",
                   "headers": [(b"host", b"127.0.0.1:8770"), (b"content-length", b"200000")]},
                  receive, send)
        assert output[0]["status"] == 413

    asyncio.run(run())
    assert streams == []


def test_cancelled_receive_releases_private_spool(setup):
    app, people, streams = setup
    reading = asyncio.Event()

    async def never():
        reading.set()
        await asyncio.Event().wait()

    async def run():
        task = asyncio.create_task(upload(app, people["manager"], receiver=never))
        await reading.wait()
        assert len(streams) == 1 and not streams[0].closed
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert streams[0].closed and app.inflight == app.upload_inflight == 0

    asyncio.run(run())


@pytest.mark.parametrize("change", ["session_revoked", "manager_demoted"])
def test_authority_lost_during_validation_prevents_publication(setup, monkeypatch, change):
    app, people, streams = setup
    access = app.runtime["access"]
    manager = people["manager"]
    registry = app.runtime["registry"]
    if change == "manager_demoted":
        # Keep another active manager, matching the account lifecycle rules.
        other_manager = access.update_user(
            manager["user"], people["employee"]["user"]["id"], "manager", True
        )
    original = server.validate_safetensors

    def revoke_after_validation(path):
        count = original(path)
        if change == "session_revoked":
            access.logout(manager["token"])
        else:
            with registry.lock:
                access.update_user(other_manager, manager["user"]["id"], "employee", True)
        return count

    monkeypatch.setattr(server, "validate_safetensors", revoke_after_validation)
    status, _, reads = asyncio.run(upload(app, manager))
    assert status in (401, 403) and reads > 0
    assert len(streams) == 1 and streams[0].closed
    assert list(app.runtime["service"].uploads.iterdir()) == []
    assert app.inflight == app.upload_inflight == 0
    with access._transaction() as db:
        assert db.execute(
            "SELECT count(*) FROM audit_events WHERE action='model.uploaded'"
        ).fetchone()[0] == 0
