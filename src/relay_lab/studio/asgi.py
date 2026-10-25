"""Bounded ASGI transport for the same authenticated Studio application.

Run one process per customer workspace. Model inference belongs on agent hosts;
this transport does not turn a dashboard request into hardware execution.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from .access import AccessStore
from .analytics import AnalyticsService
from .deployment import DeploymentConfig
from .governance import ModelRegistry
from .profiles import ProfileStore
from .server import StudioHandler
from .service import StudioService


class _Handler(StudioHandler):
    """Reuse route and authorization logic without a second HTTP listener."""

    def __init__(self, scope, body, runtime, config):
        for name, value in runtime.items():
            setattr(self, name, value)
        self.deployment, self.identity = config, None
        self.server = SimpleNamespace(server_port=config.port)
        self.command = scope["method"]
        self.path = scope.get("raw_path", scope["path"].encode()).decode("ascii")
        if scope.get("query_string"):
            self.path += "?" + scope["query_string"].decode("ascii")
        self.headers = Message()
        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                self.headers[name.decode("ascii")] = value.decode("latin1")
        self.headers["Content-Length"] = str(len(body))
        self.rfile, self.wfile = BytesIO(body), BytesIO()
        self.status, self.response_headers = 500, []

    def send_response(self, code, message=None):
        self.status = code

    def send_header(self, name, value):
        self.response_headers.append((name.lower().encode("ascii"), value.encode("latin1")))

    def end_headers(self):
        pass


class StudioApplication:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.state_lock = None
        if config.hosted:
            import fcntl

            self.state_lock = (runtime["service"].storage / "deployment.lock").open("a")
            try:
                fcntl.flock(self.state_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.state_lock.close()
                raise RuntimeError(
                    "Only one Studio process may open this customer volume"
                ) from None
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent, thread_name_prefix="studio"
        )
        self.inflight = 0
        self.closed = False

    def _dispatch(self, scope, body):
        handler = _Handler(scope, body, self.runtime, self.config)
        if scope["path"] in ("/livez", "/readyz"):
            handler.guard()
            ready = scope["path"] == "/livez"
            if not ready:
                try:
                    coordinator = self.runtime["service"].factory()
                    coordinator.client._sql("SELECT run_id FROM factory_run LIMIT 1")
                    ready = True
                except Exception:  # noqa: BLE001 - no credentials or topology in public health
                    ready = False
            handler.respond({"ready": ready}, 200 if ready else 503)
        else:
            handler.handle_request(scope["method"] == "POST")
        return handler.status, handler.response_headers, handler.wfile.getvalue()

    @staticmethod
    async def _respond(send, status, message):
        body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"cache-control", b"no-store"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-content-type-options", b"nosniff"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await self.close()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["method"] not in ("GET", "POST"):
            await self._respond(send, 405, "Method not allowed")
            return
        headers = Message()
        counts = {}
        for key, value in scope.get("headers", []):
            key = key.lower()
            counts[key] = counts.get(key, 0) + 1
            headers[key.decode("ascii")] = value.decode("latin1")
        if (
            any(
                counts.get(key, 0) > 1
                for key in (
                    b"host",
                    b"origin",
                    b"authorization",
                    b"cookie",
                    b"content-length",
                    b"x-relay-token",
                )
            )
            or sum(len(k) + len(v) for k, v in scope.get("headers", [])) > 32768
        ):
            await self._respond(send, 400, "Invalid request headers")
            return
        try:
            self.config.guard(headers)
        except PermissionError as error:
            await self._respond(send, 403, str(error))
            return
        if self.closed or self.inflight >= self.config.max_concurrent:
            await self._respond(send, 503, "Studio is busy; try again shortly")
            return
        if self.config.hosted and scope["path"] == "/api/upload":
            await self._respond(
                send, 403, "Hosted weights uploads are disabled; use the agent host"
            )
            return
        try:
            content_length = headers.get("Content-Length")
            if content_length is not None and (
                not content_length.isdecimal() or int(content_length) < 0
            ):
                raise ValueError
            if content_length is not None and int(content_length) > self.config.max_body_bytes:
                await self._respond(send, 413, "Request exceeds 128 KiB")
                return
        except (ValueError, OverflowError):
            await self._respond(send, 400, "Invalid Content-Length")
            return
        self.inflight += 1
        dispatched = False
        try:
            body = bytearray()
            async with asyncio.timeout(self.config.body_timeout_seconds):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    if event["type"] != "http.request":
                        continue
                    body.extend(event.get("body", b""))
                    if len(body) > self.config.max_body_bytes:
                        await self._respond(send, 413, "Request exceeds 128 KiB")
                        return
                    if not event.get("more_body", False):
                        break
            if content_length is not None and len(body) != int(content_length):
                await self._respond(send, 400, "Content-Length does not match request")
                return
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self.executor, self._dispatch, scope, bytes(body))
            dispatched = True

            def finished(result):
                self.inflight -= 1
                if not result.cancelled():
                    result.exception()  # Consume errors after a disconnected or timed-out caller.

            future.add_done_callback(finished)
            # Cancellation cannot stop a database commit. Keep its slot occupied until
            # the thread finishes, and tell the caller to inspect the operation.
            try:
                status, response_headers, response_body = await asyncio.wait_for(
                    asyncio.shield(future), self.config.request_timeout_seconds
                )
            except TimeoutError:
                await self._respond(
                    send, 504, "Request timed out; inspect operation state before retrying"
                )
                return
            await send(
                {"type": "http.response.start", "status": status, "headers": response_headers}
            )
            await send({"type": "http.response.body", "body": response_body})
        except TimeoutError:
            await self._respond(send, 408, "Request body timed out")
        except Exception:  # noqa: BLE001 - don't leak deployment paths or credentials
            await self._respond(send, 503, "The request could not be completed")
        finally:
            if not dispatched:
                self.inflight -= 1

    async def close(self):
        self.closed = True
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
        if pump := self.runtime.get("schedule_pump"):
            pump.stop()
        self.runtime["service"].close()
        if self.state_lock is not None:
            self.state_lock.close()


def create_app(*, config=None, runtime=None):
    config = config or DeploymentConfig.from_env()
    if runtime is None:
        os.umask(0o077)
        root = Path(config.root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if config.hosted:
            from relay_lab.factory.spacetime import SpacetimeCoordinator
            from relay_lab.spacetime import SpacetimeConfig

            token_file = Path(
                os.environ.get(
                    "RELAY_SPACETIME_TOKEN_FILE", str(root / ".relay-lab/spacetime-token")
                )
            )
            backend = SpacetimeConfig.from_env(token_file=token_file)
            service = StudioService(root, coordinator_factory=lambda: SpacetimeCoordinator(backend))
        else:
            service = StudioService(root)
        access_type = (
            AccessStore
            if config.hosted or os.environ.get("RELAY_STUDIO_PASSWORD_LOGIN") == "1"
            else ProfileStore
        )
        access = access_type(service.storage)
        registry = ModelRegistry(service.storage)
        if isinstance(access, ProfileStore):
            manager = next(
                (p for p in access.profiles()["profiles"] if p["role"] == "manager"), None
            )
            if manager:
                registry.sync_workspace(manager, service.workspace())
        from .schedules import SchedulePump, ScheduleService
        from .tasks import TaskService
        from .workers import WorkerStore

        tasks = TaskService(service, registry)
        schedules = ScheduleService(tasks, access)
        runtime = {
            "service": service,
            "access": access,
            "registry": registry,
            "analytics": AnalyticsService(root, service, registry, access),
            "tasks": tasks,
            "workers": WorkerStore(service.storage, access, registry, service),
            "schedules": schedules,
            "schedule_pump": SchedulePump(schedules),
        }
        runtime["schedule_pump"].start()
    return StudioApplication(runtime, config)


def main():
    import uvicorn

    config = DeploymentConfig.from_env()
    uvicorn.run(
        create_app(config=config),
        host=config.bind,
        port=config.port,
        workers=1,
        proxy_headers=False,
        limit_concurrency=32,
        backlog=64,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=180,
        h11_max_incomplete_event_size=16384,
        access_log=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
