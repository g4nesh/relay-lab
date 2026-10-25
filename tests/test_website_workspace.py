"""Local website routes must retain Studio's authenticated ASGI lifecycle."""

import asyncio
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.asgi import StudioApplication
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.realtime import RealtimeHub

spec = importlib.util.spec_from_file_location(
    "serve_workspace", Path(__file__).resolve().parents[1] / "scripts/serve_workspace.py"
)
website = importlib.util.module_from_spec(spec)
spec.loader.exec_module(website)


class Studio:
    def __init__(self, root):
        self.config = DeploymentConfig(root=root, port=8780)
        self.closed = False
        self.calls = []

    async def __call__(self, scope, receive, send):
        self.calls.append(scope)
        if scope["type"] == "lifespan":
            for kind in ("startup", "shutdown"):
                assert (await receive())["type"] == f"lifespan.{kind}"
                await send({"type": f"lifespan.{kind}.complete"})
        else:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"event: sync\n\n", "more_body": True})
            await send({"type": "http.response.body", "body": b"event: update\n\n"})


@pytest.fixture
def app(tmp_path):
    site = tmp_path / "website"
    site.mkdir()
    (site / "index.html").write_text("Website homepage")
    (site / "projects/cloth").mkdir(parents=True)
    (site / "projects/cloth/index.html").write_text("Cloth workcell")
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text(
        '<head><link href="/workspace-design.css"></head><body>'
        '<a href="http://127.0.0.1:8769/latent">Adapters</a></body>'
    )
    return website.WebsiteApplication(Studio(tmp_path), site=site, web=web)


async def request(app, path, *, method="GET", headers=None):
    output = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(event):
        output.append(event)

    await app({
        "type": "http", "path": path, "method": method,
        "headers": headers if headers is not None else [(b"host", b"127.0.0.1:8780")],
    }, receive, send)
    return output


def body(events):
    return b"".join(e.get("body", b"") for e in events)


@pytest.mark.parametrize("path", ["/api/tasks", "/api/events", "/api/agent/events", "/readyz", "/studio.js"])
def test_backend_and_sse_are_forwarded_without_buffering(app, path):
    result = asyncio.run(request(app, path))
    assert app.studio.calls[0]["path"] == path
    assert result[1]["more_body"] is True
    assert body(result) == b"event: sync\n\nevent: update\n\n"


@pytest.mark.parametrize("headers,status", [
    ([(b"host", b"attacker.example")], 403),
    ([(b"host", b"127.0.0.1:8780"), (b"origin", b"https://attacker.example")], 403),
    ([(b"host", b"127.0.0.1:8780"), (b"host", b"attacker.example")], 400),
])
def test_static_routes_enforce_host_and_origin(app, headers, status):
    result = asyncio.run(request(app, "/", headers=headers))
    assert result[0]["status"] == status
    assert b"Website homepage" not in body(result)


@pytest.mark.parametrize("path", ["/../secret.txt", "/%2e%2e/secret.txt", "/linked.txt", "/escape/"])
def test_static_routes_reject_traversal_and_symlinks(app, path):
    secret = app.site.parent / "secret.txt"
    secret.write_text("private")
    (app.site / "linked.txt").symlink_to(secret)
    outside = app.site.parent / "outside"
    outside.mkdir()
    (outside / "index.html").write_text("private")
    (app.site / "escape").symlink_to(outside, target_is_directory=True)
    result = asyncio.run(request(app, path))
    assert result[0]["status"] == 404
    assert b"private" not in body(result)


def test_live_studio_page_uses_shared_styles_once_and_local_experiment_links(app):
    result = asyncio.run(request(app, "/projects/studio/"))
    assert result[0]["status"] == 200
    assert body(result).count(b"/workspace-design.css") == 1
    assert b"/projects/cloth/latent.html" in body(result)
    assert b"/projects/studio/local-tabs.js" in body(result)
    assert dict(result[0]["headers"])[b"cache-control"] == b"no-store"
    assert not app.studio.calls


def test_cloth_routes_and_etag_cache_remain_available(app):
    first = asyncio.run(request(app, "/projects/cloth/"))
    assert body(first) == b"Cloth workcell"
    headers = dict(first[0]["headers"])
    assert headers[b"cache-control"] == b"no-cache"
    cached = asyncio.run(request(app, "/projects/cloth/", headers=[
        (b"host", b"127.0.0.1:8780"), (b"if-none-match", headers[b"etag"]),
    ]))
    assert cached[0]["status"] == 304 and body(cached) == b""
    head = asyncio.run(request(app, "/projects/cloth/", method="HEAD"))
    assert head[0]["status"] == 200 and body(head) == b""


def test_large_recordings_are_sent_in_bounded_chunks(app):
    (app.site / "recording.mp4").write_bytes(b"x" * (600 * 1024))
    result = asyncio.run(request(app, "/recording.mp4"))
    assert dict(result[0]["headers"])[b"cache-control"] == b"public, max-age=604800"
    assert len(body(result)) == 600 * 1024
    assert all(len(event.get("body", b"")) <= 256 * 1024 for event in result)


def test_real_studio_lifespan_starts_and_stops_coordination(tmp_path):
    calls = []

    class Hub(RealtimeHub):
        def start(self):
            calls.append("subscriptions started")

        async def close(self):
            calls.append("subscriptions stopped")
            await super().close()

    class Access:
        def authenticate(self, _token):
            raise AuthenticationError("Not signed in")

    service = SimpleNamespace(
        storage=tmp_path, experiments_enabled=True, close=lambda: calls.append("service closed")
    )
    studio = StudioApplication({
        "service": service,
        "access": Access(),
        "registry": SimpleNamespace(lock=threading.RLock()),
        "realtime": Hub(service),
        "coordination": SimpleNamespace(
            start=lambda: calls.append("coordination started"),
            close=lambda: calls.append("coordination stopped"),
        ),
    }, DeploymentConfig(root=tmp_path, port=8780))
    app = website.WebsiteApplication(studio)

    async def run():
        unauth = await request(app, "/api/events")
        assert unauth[0]["status"] == 401
        events = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
        result = []

        async def receive():
            return next(events)

        async def send(event):
            result.append(event["type"])

        await app({"type": "lifespan"}, receive, send)
        assert result == ["lifespan.startup.complete", "lifespan.shutdown.complete"]

    asyncio.run(run())
    assert calls == ["subscriptions started", "coordination started", "subscriptions stopped", "coordination stopped", "service closed"]
    assert studio.closed
