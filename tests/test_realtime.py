"""Bounded live invalidations, auth loss, transport isolation, and recovery."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.asgi import StudioApplication
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.realtime import TOPICS, RealtimeHub


class Access:
    def __init__(self):
        self.valid = True

    def authenticate(self, token):
        if token != "browser-secret" or not self.valid:
            raise AuthenticationError("revoked secret must never reach SSE")
        return {"user": {"id": "owner"}, "permissions": [], "csrf": "secret"}


class Workers:
    def __init__(self):
        self.valid = True

    def authenticate(self, token):
        if token != "relay_worker_secret" or not self.valid:
            raise AuthenticationError("revoked worker")
        return {"id": "worker-private"}


class Hub(RealtimeHub):
    def start(self):
        pass


def application(tmp_path):
    service = SimpleNamespace(storage=tmp_path, experiments_enabled=False, close=lambda: None)
    hub = Hub(service, max_listeners=2)
    runtime = {
        "service": service,
        "access": Access(),
        "workers": Workers(),
        "registry": SimpleNamespace(lock=threading.RLock()),
        "realtime": hub,
    }
    app = StudioApplication(runtime, DeploymentConfig(root=tmp_path))
    app.stream_heartbeat_seconds = 0.03
    app.stream_send_timeout = 0.1
    return app, hub


async def connection(app, path="/api/events", *, headers=None, disconnect=None, observed=None):
    incoming = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": b""})
    output = []

    async def receive():
        if disconnect is not None and incoming.empty():
            await disconnect.wait()
            return {"type": "http.disconnect"}
        return await incoming.get()

    async def send(event):
        output.append(event)
        if observed is not None:
            observed(event)

    scope = {
        "type": "http",
        "path": path,
        "method": "GET",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8770"), *(headers or [])],
    }
    await app(scope, receive, send)
    return output


COOKIE = [(b"cookie", b"relay_studio_8770=browser-secret")]
WORKER = [(b"authorization", b"Bearer relay_worker_secret")]


def test_coalesces_without_payloads_and_resyncs():
    async def run():
        hub = Hub(None)
        queue = hub.subscribe()
        for _ in range(1000):
            hub.publish(["tasks"])
        hub.publish(["workers"])
        assert queue.qsize() == 1
        item = queue.get_nowait()
        assert item == {"event": "sync", "topics": sorted(TOPICS), "sequence": 1001}
        hub.accept({"type": "status", "status": "ready", "token": "never forwarded"})
        assert queue.get_nowait()["event"] == "sync"
        hub.accept({"type": "status", "status": "degraded"})
        assert hub.snapshot()["polling_fallback"]
        hub.accept({"type": "status", "status": "ready"})
        assert queue.get_nowait()["event"] == "sync"
        with pytest.raises(ValueError):
            hub.accept({"type": "update", "topics": ["private-id"]})
        await hub.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "path,headers", [("/api/events", []), ("/api/events", WORKER), ("/api/agent/events", COOKIE)]
)
def test_stream_authentication_cannot_cross_boundaries(tmp_path, path, headers):
    async def run():
        app, hub = application(tmp_path)
        try:
            out = await connection(app, path, headers=headers)
            assert out[0]["status"] == 401
            assert not hub.listeners
        finally:
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "path,headers,store",
    [("/api/events", COOKIE, "access"), ("/api/agent/events", WORKER, "workers")],
)
def test_revoked_stream_closes_without_more_events(tmp_path, path, headers, store):
    async def run():
        app, hub = application(tmp_path)

        def observe(event):
            if event.get("body", b"").startswith(b"event: sync"):
                app.runtime[store].valid = False
                hub.publish(["tasks"])

        try:
            out = await asyncio.wait_for(
                connection(app, path, headers=headers, observed=observe), 1
            )
            body = b"".join(e.get("body", b"") for e in out)
            assert body.count(b"event: sync") == 1
            assert b"event: update" not in body
            assert b"private" not in body and b"secret" not in body
            assert not hub.listeners and app.inflight == 0
        finally:
            await app.close()

    asyncio.run(run())


def test_stream_capacity_does_not_occupy_request_executor(tmp_path):
    async def run():
        app, hub = application(tmp_path)
        disconnect = asyncio.Event()
        ready = 0

        def observe(event):
            nonlocal ready
            if event.get("body", b"").startswith(b"event: sync"):
                ready += 1

        sessions = [
            asyncio.create_task(
                connection(app, headers=COOKIE, disconnect=disconnect, observed=observe)
            )
            for _ in range(2)
        ]
        try:
            for _ in range(100):
                if ready == 2:
                    break
                await asyncio.sleep(0.01)
            assert ready == 2 and app.inflight == 0
            refused = await connection(app, headers=COOKIE)
            assert refused[0]["status"] == 503
            health = await connection(app, "/livez")
            assert health[0]["status"] == 200
            assert json.loads(health[1]["body"]) == {"ready": True}
            disconnect.set()
            await asyncio.wait_for(asyncio.gather(*sessions), 1)
            assert not hub.listeners
        finally:
            disconnect.set()
            await asyncio.gather(*sessions, return_exceptions=True)
            await app.close()

    asyncio.run(run())


def test_duplicate_authorization_is_rejected_before_stream(tmp_path):
    async def run():
        app, hub = application(tmp_path)
        try:
            result = await connection(app, "/api/agent/events", headers=WORKER + WORKER)
            assert result[0]["status"] == 400
            assert not hub.listeners
        finally:
            await app.close()

    asyncio.run(run())


def test_missing_bridge_is_degraded_and_stops(tmp_path):
    async def run():
        hub = RealtimeHub(None, bridge=tmp_path / "missing.cjs")
        hub.start()
        await asyncio.sleep(0.02)
        assert hub.snapshot()["status"] == "degraded"
        assert hub.snapshot()["reason"] == "bridge_unavailable"
        await hub.close()
        assert hub.task.done()

    asyncio.run(run())


def test_real_session_logout_revokes_open_connection(tmp_path):
    from relay_lab.studio.access import AccessStore

    async def run():
        app, hub = application(tmp_path)
        access = AccessStore(tmp_path)
        credentials = access.setup(
            access.setup_path.read_text().strip(), "owner", "Owner", "correct horse robot battery"
        )
        app.runtime["access"] = access
        headers = [(b"cookie", ("relay_studio_8770=" + credentials["token"]).encode())]

        def observe(event):
            if event.get("body", b"").startswith(b"event: sync"):
                access.logout(credentials["token"])

        try:
            output = await asyncio.wait_for(connection(app, headers=headers, observed=observe), 1)
            assert output[0]["status"] == 200
            assert b"keepalive" not in b"".join(item.get("body", b"") for item in output)
            refused = await connection(app, headers=headers)
            assert refused[0]["status"] == 401
            assert not hub.listeners
        finally:
            await app.close()

    asyncio.run(run())


def test_slow_consumer_is_disconnected_and_releases_capacity(tmp_path):
    async def run():
        app, hub = application(tmp_path)
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": b""})

        async def send(event):
            if event["type"] == "http.response.body":
                await asyncio.Future()

        scope = {
            "type": "http",
            "path": "/api/events",
            "method": "GET",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1:8770"), *COOKIE],
        }
        try:
            await asyncio.wait_for(app(scope, incoming.get, send), 1)
            assert not hub.listeners and app.inflight == 0
        finally:
            await app.close()

    asyncio.run(run())


def test_bridge_restarts_and_credentials_only_enter_private_stdin(tmp_path, monkeypatch):
    import shutil

    if not shutil.which("node"):
        pytest.skip("Node22+ is required for the subscription transport")
    script = tmp_path / "test-bridge.cjs"
    script.write_text("""let input='';process.stdin.on('data',v=>input+=v);process.stdin.on('end',()=>{
      const config=JSON.parse(input);
      if(config.token!=='owner-secret' || process.argv.includes('owner-secret') || process.env.RELAY_SPACETIME_TOKEN) process.exit(9);
      process.stdout.write(JSON.stringify({type:'status',status:'ready'})+'\\n');
      process.stdout.write(JSON.stringify({type:'update',topics:['tasks'],secret:config.token})+'\\n');
      setTimeout(()=>process.exit(1),75);
    });""")
    monkeypatch.setenv("RELAY_SPACETIME_TOKEN", "must-not-be-in-child-environment")

    async def run():
        config = SimpleNamespace(
            server="http://127.0.0.1:3000", database="test", token="owner-secret"
        )
        service = SimpleNamespace(
            factory=lambda: SimpleNamespace(client=SimpleNamespace(config=config))
        )
        hub = RealtimeHub(service, bridge=script)
        queue = hub.subscribe()
        seen = []
        try:
            for _ in range(200):
                if not queue.empty():
                    seen.append(queue.get_nowait())
                if hub.sequence >= 5:
                    break
                await asyncio.sleep(0.01)
            assert hub.sequence >= 5  # initial ready, disconnect, second ready and updates
            assert "owner-secret" not in json.dumps(seen)
        finally:
            await hub.close()
        assert hub.task.done() and hub.process is None

    asyncio.run(run())


def test_update_during_idle_reauthorization_is_not_lost(tmp_path):
    async def run():
        app, hub = application(tmp_path)
        disconnect = asyncio.Event()
        original = app._stream_authority
        checks = 0
        updates = []

        async def authorize(scope):
            nonlocal checks
            await original(scope)
            checks += 1
            if checks == 3:  # After initial sync, at the first idle heartbeat.
                hub.publish(["tasks"])
                await asyncio.sleep(0.005)

        app._stream_authority = authorize

        def observe(event):
            if event.get("body", b"").startswith(b"event: update"):
                updates.append(event)
                disconnect.set()

        try:
            await asyncio.wait_for(
                connection(app, headers=COOKIE, disconnect=disconnect, observed=observe), 1
            )
            assert len(updates) == 1
        finally:
            disconnect.set()
            await app.close()

    asyncio.run(run())


def test_readiness_requires_coordination_schema_when_enabled(tmp_path):
    async def run():
        app, _ = application(tmp_path)
        seen = []

        def query(sql):
            seen.append(sql)
            if "studio_presence" in sql:
                raise RuntimeError("table absent, private diagnostic")
            return []

        app.runtime["service"].factory = lambda: SimpleNamespace(client=SimpleNamespace(_sql=query))
        app.runtime["coordination"] = SimpleNamespace(close=lambda: None)
        try:
            out = await connection(app, "/readyz")
            assert out[0]["status"] == 503
            assert json.loads(out[1]["body"]) == {"ready": False}
            assert any("studio_job" in query for query in seen)
            assert b"diagnostic" not in out[1]["body"]
        finally:
            await app.close()

    asyncio.run(run())


def test_shutdown_drains_executors_and_retains_state_lock_on_writer_failure(tmp_path):
    from io import BytesIO

    async def run():
        app, _ = application(tmp_path)
        app.state_lock = BytesIO()
        stopped, closed = [], []
        app.runtime["service"].close = lambda: closed.append(True)

        def finish():
            assert app.auth_executor._shutdown and app.executor._shutdown
            assert not app.state_lock.closed
            raise RuntimeError("Writer has not stopped")

        app.runtime["coordination"] = SimpleNamespace(
            request_stop=lambda: stopped.append(True), close=finish
        )
        with pytest.raises(RuntimeError, match="workspace lock retained"):
            await app.close()
        assert stopped and not closed and not app.state_lock.closed
        assert app.closed and not app._cleanup_complete
        app.runtime["coordination"].close = lambda: None
        await app.close()
        assert closed == [True] and app.state_lock.closed and app._cleanup_complete
        await app.close()
        assert closed == [True]

    asyncio.run(run())
