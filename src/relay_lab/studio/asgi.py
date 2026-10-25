"""Bounded ASGI transport for the same authenticated Studio application.

Run one process per customer workspace. Model inference belongs on agent hosts;
this transport does not turn a dashboard request into hardware execution.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from .access import AccessStore, AuthenticationError
from .analytics import AnalyticsService
from .deployment import DeploymentConfig, experiments_requested
from .governance import ModelRegistry
from .profiles import ProfileStore
from .realtime import RealtimeHub
from .server import MAX_UPLOAD, StudioHandler
from .service import StudioService


class _UploadBody:
    """A seekable, disk-backed request retained until its executor finishes."""

    def __init__(self, stream, length):
        self.stream, self.length = stream, length


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
        if isinstance(body, _UploadBody):
            self.headers["Content-Length"] = str(body.length)
            self.rfile = body.stream
        else:
            self.headers["Content-Length"] = str(len(body))
            self.rfile = BytesIO(body)
        self.wfile = BytesIO()
        self.status, self.response_headers = 500, []

    def send_response(self, code, message=None):
        self.status = code

    def send_header(self, name, value):
        self.response_headers.append((name.lower().encode("ascii"), value.encode("latin1")))

    def end_headers(self):
        pass


class StudioApplication:
    def __init__(self, runtime, config, *, durable=None):
        self.runtime, self.config = runtime, config
        self.durable = durable
        if config.hosted and runtime["service"].experiments_enabled:
            runtime["service"].disable_experiments()
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
        self._cleanup_complete = False
        self.auth_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="studio-stream-auth"
        )
        self.auth_inflight = 0
        self.auth_slots = asyncio.Semaphore(2)
        self.stream_heartbeat_seconds = 10.0
        self.stream_send_timeout = 5.0
        self.upload_inflight = 0
        self.upload_body_timeout_seconds = 180.0
        self.runtime.setdefault("realtime", RealtimeHub(runtime["service"]))

    def _dispatch(self, scope, body):
        if self.durable is not None and scope["path"].startswith("/api/"):
            with self.durable.request():
                return self._dispatch_local(scope, body)
        return self._dispatch_local(scope, body)

    def _dispatch_local(self, scope, body):
        handler = _Handler(scope, body, self.runtime, self.config)
        if scope["path"] in ("/livez", "/readyz"):
            handler.guard()
            ready = scope["path"] == "/livez"
            if not ready:
                try:
                    if self.durable is not None:
                        self.durable.probe()
                    coordinator = self.runtime["service"].factory()
                    coordinator.client._sql("SELECT run_id FROM factory_run LIMIT 1")
                    if self.runtime.get("coordination") is not None:
                        for table in ("studio_job", "studio_presence", "studio_task_watch"):
                            coordinator.client._sql(f"SELECT key FROM {table} LIMIT 1")
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

    def _authorize_events(self, scope):
        # Restore current authority before touching caches; do not retain a
        # database lease for the lifetime of a streaming connection.
        if self.durable is not None:
            with self.durable.request():
                return self._authorize_events_local(scope)
        return self._authorize_events_local(scope)

    def _authorize_events_local(self, scope):
        handler = _Handler(scope, b"", self.runtime, self.config)
        # Re-read local identity, issuer authority, and approved model revision.
        # A stream never upgrades a worker token into browser authority.
        with self.runtime["registry"].lock:
            if scope["path"] == "/api/agent/events":
                token = handler.headers.get("Authorization", "")
                if not token.startswith("Bearer relay_worker_"):
                    raise AuthenticationError("Use an agent connection token")
                handler.worker_store.authenticate(token.removeprefix("Bearer "))
            else:
                if handler.headers.get("Authorization"):
                    raise AuthenticationError("Use your dashboard session cookie")
                handler.authenticate()

    async def _stream_authority(self, scope, *, authorize=None):
        # Stream authentication has its own small executor. A quiet connection
        # holds neither a normal request slot nor a worker thread.
        if self.auth_inflight >= self.runtime["realtime"].max_listeners + 2:
            raise RuntimeError("Authentication is busy")
        self.auth_inflight += 1
        acquired = dispatched = False
        try:
            await asyncio.wait_for(self.auth_slots.acquire(), self.config.request_timeout_seconds)
            acquired = True
            future = asyncio.get_running_loop().run_in_executor(
                self.auth_executor, authorize or self._authorize_events, scope
            )
            dispatched = True

            def finished(result):
                self.auth_inflight -= 1
                self.auth_slots.release()
                if not result.cancelled():
                    result.exception()

            future.add_done_callback(finished)
            await asyncio.wait_for(asyncio.shield(future), self.config.request_timeout_seconds)
        finally:
            if not dispatched:
                self.auth_inflight -= 1
                if acquired:
                    self.auth_slots.release()

    def _authorize_upload(self, scope):
        handler = _Handler(scope, b"", self.runtime, self.config)
        with self.runtime["registry"].lock:
            handler.guard()
            handler.authenticate(True)
            handler.require("models.manage")

    async def _upload(self, scope, headers, receive, send):
        """Authorize before disk I/O; keep one bounded local upload on disk."""
        if self.config.hosted or not self.runtime["service"].experiments_enabled:
            await self._respond(send, 403, "Weights uploads require local research mode")
            return
        if self.upload_inflight:
            await self._respond(send, 503, "Another weights upload is in progress")
            return
        declared = headers.get("Content-Length")
        if declared is None:
            await self._respond(send, 411, "Weights uploads require Content-Length")
            return
        if not declared.isdecimal():
            await self._respond(send, 400, "Invalid Content-Length")
            return
        try:
            length = int(declared)
        except ValueError:
            await self._respond(send, 400, "Invalid Content-Length")
            return
        if not 10 <= length <= MAX_UPLOAD:
            await self._respond(send, 413, "Weights must be between 10 bytes and 4 GiB")
            return
        self.inflight += 1
        self.upload_inflight += 1
        spool = None
        dispatched = False
        try:
            try:
                await self._stream_authority(scope, authorize=self._authorize_upload)
            except AuthenticationError:
                await self._respond(send, 401, "Sign in before uploading weights")
                return
            except PermissionError as error:
                await self._respond(send, 403, str(error))
                return
            # TemporaryFile is private and unlinked on Unix; closing removes it
            # on every platform. No model bytes accumulate in a request buffer.
            # Ownership transfers to the executor callback after body receipt.
            spool = tempfile.TemporaryFile(mode="w+b", prefix="relay-upload-")  # noqa: SIM115
            received = 0
            async with asyncio.timeout(self.upload_body_timeout_seconds):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    if event["type"] != "http.request":
                        continue
                    chunk = event.get("body", b"")
                    received += len(chunk)
                    if received > length or received > MAX_UPLOAD:
                        await self._respond(send, 413, "Upload exceeds its declared size or limit")
                        return
                    spool.write(chunk)
                    if not event.get("more_body", False):
                        break
            if received != length:
                await self._respond(send, 400, "Content-Length does not match request")
                return
            spool.seek(0)
            future = asyncio.get_running_loop().run_in_executor(
                self.executor, self._dispatch, scope, _UploadBody(spool, length)
            )
            dispatched = True

            def finished(result):
                # A response timeout/cancel cannot interrupt a running handler.
                # Its stream and concurrency slot belong to that future until done.
                spool.close()
                self.inflight -= 1
                self.upload_inflight -= 1
                if not result.cancelled():
                    result.exception()

            future.add_done_callback(finished)
            try:
                status, response_headers, response_body = await asyncio.wait_for(
                    asyncio.shield(future), self.config.request_timeout_seconds
                )
            except TimeoutError:
                await self._respond(send, 504, "Upload timed out; inspect model uploads before retrying")
                return
            await send({"type": "http.response.start", "status": status, "headers": response_headers})
            await send({"type": "http.response.body", "body": response_body})
        except TimeoutError:
            await self._respond(send, 408, "Upload body timed out")
        except Exception:  # noqa: BLE001 - filesystem/auth internals stay private
            await self._respond(send, 503, "The upload could not be completed")
        finally:
            if not dispatched:
                if spool is not None:
                    spool.close()
                self.inflight -= 1
                self.upload_inflight -= 1

    async def _events(self, scope, receive, send):
        hub = self.runtime["realtime"]
        try:
            await self._stream_authority(scope)
        except AuthenticationError:
            await self._respond(send, 401, "Connection authorization expired")
            return
        except PermissionError:
            await self._respond(send, 403, "Connection is no longer authorized")
            return
        except Exception:  # noqa: BLE001 - credentials and SDK diagnostics are private
            await self._respond(send, 503, "Live connection is temporarily unavailable")
            return
        try:
            queue = hub.subscribe()
        except RuntimeError:
            await self._respond(send, 503, "Live connection limit reached")
            return
        disconnected = asyncio.create_task(receive())
        pending = None
        started = False
        last_state = None

        async def write(event):
            await asyncio.wait_for(send(event), self.stream_send_timeout)

        try:
            await write(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream; charset=utf-8"),
                        (b"cache-control", b"no-store"),
                        (b"x-accel-buffering", b"no"),
                        (b"x-content-type-options", b"nosniff"),
                    ],
                }
            )
            started = True
            while not self.closed:
                pending = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    (pending, disconnected),
                    timeout=self.stream_heartbeat_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnected in done:
                    if disconnected.result().get("type") == "http.disconnect":
                        break
                    disconnected = asyncio.create_task(receive())
                # Check before every event and every idle heartbeat. Revocation
                # closes the stream; subsequent reconnect must authenticate.
                await self._stream_authority(scope)
                if self.closed or (
                    disconnected.done() and disconnected.result().get("type") == "http.disconnect"
                ):
                    break
                if hub.state != last_state:
                    last_state = hub.state
                    status_body = (
                        "event: status\ndata: " + json.dumps({"state": last_state}) + "\n\n"
                    ).encode()
                    await write(
                        {"type": "http.response.body", "body": status_body, "more_body": True}
                    )
                # Auth may await while an idle heartbeat times out. Inspect the
                # current future, not asyncio.wait's earlier snapshot, to retain
                # an invalidation that arrived during reauthorization.
                if pending.done() and not pending.cancelled():
                    item = pending.result()
                    payload = json.dumps(
                        {"topics": item["topics"], "sequence": item["sequence"]},
                        separators=(",", ":"),
                    )
                    body = f"event: {item['event']}\ndata: {payload}\n\n".encode()
                else:
                    body = b": keepalive\n\n"
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                pending = None
                await write({"type": "http.response.body", "body": body, "more_body": True})
        except (Exception, asyncio.CancelledError):  # noqa: BLE001, S110 - client disconnect/auth loss
            # No exception text or credentials enter the public stream.
            pass
        finally:
            hub.unsubscribe(queue)
            disconnected.cancel()
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await asyncio.gather(disconnected, return_exceptions=True)
            if started:
                try:
                    await write({"type": "http.response.body", "body": b"", "more_body": False})
                except (Exception, asyncio.CancelledError):  # noqa: BLE001, S110 - already disconnected
                    pass

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    self.runtime["realtime"].start()
                    coordination = self.runtime.get("coordination")
                    if coordination is not None:
                        try:
                            await asyncio.to_thread(coordination.start)
                        except Exception:  # noqa: BLE001, S110 - degraded status is reported separately
                            pass  # Coordination exposes degraded status; ordinary requests still work.
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
                    b"x-filename",
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
        if scope["path"] == "/api/upload" and scope["method"] == "POST":
            await self._upload(scope, headers, receive, send)
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
        streaming = scope["path"] in ("/api/events", "/api/agent/events")
        if streaming and scope["method"] != "GET":
            await self._respond(send, 405, "Live connections require GET")
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
            if streaming:
                if body:
                    await self._respond(send, 400, "Live connections do not accept a request body")
                    return
                self.inflight -= 1
                dispatched = True
                await self._events(scope, receive, send)
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
        if self._cleanup_complete:
            return
        self.closed = True
        coordination = self.runtime.get("coordination")
        if coordination is not None and hasattr(coordination, "request_stop"):
            coordination.request_stop()
        errors = []
        # Drain request holders before joining the background writer. One failed
        # cleanup must not strand either executor; retain the workspace lock if
        # any writer/resource cannot be confirmed stopped, and allow a later retry.
        for cleanup in (
            self.runtime["realtime"].close,
            lambda: asyncio.to_thread(self.auth_executor.shutdown, wait=True, cancel_futures=True),
            lambda: asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True),
            *([lambda: asyncio.to_thread(coordination.close)] if coordination is not None else []),
        ):
            try:
                await cleanup()
            except Exception as error:  # noqa: BLE001 - finish independent cleanup before failing
                errors.append(error)
        if errors:
            raise RuntimeError("Studio shutdown incomplete; workspace lock retained") from errors[0]
        self.runtime["service"].close()
        if self.state_lock is not None:
            self.state_lock.close()
        self._cleanup_complete = True


def create_app(*, config=None, runtime=None):
    config = config or DeploymentConfig.from_env()
    durable = None
    if runtime is None:
        os.umask(0o077)
        root = Path(config.root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        coordinator_factory = None
        if config.hosted:
            from relay_lab.factory.spacetime import SpacetimeCoordinator
            from relay_lab.spacetime import SpacetimeConfig

            token_file = Path(os.environ.get(
                "RELAY_SPACETIME_TOKEN_FILE", str(root / ".relay-lab/spacetime-token")
            ))
            backend = SpacetimeConfig.from_env(token_file=token_file)
            coordinator_factory = lambda: SpacetimeCoordinator(backend)
            if os.environ.get("RELAY_STUDIO_STORAGE") == "spacetime":
                from .durable import DurableState

                durable = DurableState(root / ".relay-lab/studio", backend)
                coordinator_factory = durable.coordinator
        if os.environ.get("RENDER") and durable is None:
            raise RuntimeError("Render requires RELAY_STUDIO_STORAGE=spacetime; local state is ephemeral")
        from contextlib import nullcontext

        with durable.request() if durable else nullcontext():
            runtime = _create_runtime(root, config, coordinator_factory, durable)
    return StudioApplication(runtime, config, durable=durable)


def _create_runtime(root, config, coordinator_factory, durable=None):
    from .coordination import CoordinationService
    from .tasks import TaskService
    from .workers import WorkerStore

    service = (
        StudioService(root, coordinator_factory=coordinator_factory, experiments=False)
        if config.hosted else StudioService(root, experiments=experiments_requested())
    )
    access_type = (
        AccessStore if config.hosted or os.environ.get("RELAY_STUDIO_PASSWORD_LOGIN") == "1"
        else ProfileStore
    )
    access = access_type(service.storage)
    registry = ModelRegistry(service.storage)
    if isinstance(access, ProfileStore):
        manager = next((p for p in access.profiles()["profiles"] if p["role"] == "manager"), None)
        if manager:
            registry.sync_workspace(manager, service.workspace())
    coordination = CoordinationService(service, registry, durable=durable)
    service.coordination = coordination
    tasks = TaskService(service, registry)
    workers = WorkerStore(service.storage, access, registry, service)
    coordination.workers = workers
    tasks.coordination = workers.coordination = coordination
    return {
        "coordination": coordination,
        "service": service, "access": access, "registry": registry,
        "analytics": AnalyticsService(root, service, registry, access),
        "tasks": tasks,
        "workers": workers,
    }



def serve_app(app, *, config):
    """Use the same lifecycle, streaming and limits for every Studio launcher."""
    import uvicorn

    uvicorn.run(
        app,
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


def main():
    config = DeploymentConfig.from_env()
    serve_app(create_app(config=config), config=config)


if __name__ == "__main__":
    main()
