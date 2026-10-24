"""Loopback-only live factory dashboard; one simulation thread owns its world."""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .runtime import FactoryRun
from .world import FactoryWorld

WEB = Path(__file__).parent / "web"


class FactorySession:
    def __init__(self, coordination="spacetime"):
        self.coordination = coordination
        self._run = None
        self._worker = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.RLock()
        self._closed = False
        self.speed = 4.0
        self._idle = self._preview()

    @staticmethod
    def _preview():
        world = FactoryWorld()
        try:
            state = world.snapshot()
            return {
                **state,
                "status": "ready",
                "sim_time": 0,
                "scene": world.scene(),
                "stations": world.scene()["stations"],
                "orders": list(state["jobs"].values()),
                "mission": {"completed": 0, "total": 3},
                "resources": [],
                "messages": [],
                "events": [],
                "metrics": {},
                "planner": {
                    "mode": "llm",
                    "model": "Qwen3-1.7B · local",
                    "calls": 0,
                    "control": "model-based embodiment adapters",
                    "reason": "Start a production order",
                },
                "coordination": {"backend": "spacetime", "connected": False},
                "error": None,
            }
        finally:
            world.close()

    def view(self):
        with self._lock:
            value = self._run.view() if self._run else deepcopy(self._idle)
            value["speed"] = self.speed
            value["pause_requested"] = self._pause.is_set()
            return value

    def start(self, options):
        unknown = set(options) - {"seed", "orders", "planner"}
        if unknown:
            raise ValueError("Unknown start options")
        seed, orders, planner = (
            options.get("seed", 7),
            options.get("orders", 3),
            options.get("planner", "llm"),
        )
        if (
            type(orders) is not int
            or not 1 <= orders <= 6
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise ValueError("Orders must be 1–6; seed must be a nonnegative 32-bit integer")
        if planner not in ("llm", "rules"):
            raise ValueError("Unknown planner")
        with self._lock:
            if self._closed:
                raise ValueError("Factory session is closed")
            if self._worker and self._worker.is_alive():
                raise ValueError("A run is already active; pause or reset it first")
            self._stop.clear()
            self._pause.clear()
            self._run = FactoryRun(
                seed=seed, orders=orders, planner=planner, coordination=self.coordination
            )
            self._run.status = "starting"
            self._run._refresh()
            self._worker = threading.Thread(target=self._loop, name="factory-worker", daemon=True)
            self._worker.start()
        return self.view()

    def _loop(self):
        run = self._run
        try:
            run.initialize()
            while not self._stop.is_set() and run.status not in (
                "succeeded",
                "failed",
                "exhausted",
            ):
                if self._pause.is_set():
                    run.status = "paused"
                    run._refresh()
                    self._wake.wait(0.2)
                    self._wake.clear()
                    continue
                start = time.monotonic()
                run.tick()
                with self._lock:
                    delay = max(0, run.world.dt / self.speed - (time.monotonic() - start))
                self._wake.wait(delay)
                self._wake.clear()
        except Exception as error:  # noqa: BLE001 - contain worker/HTTP boundary faults
            if run.status != "failed":
                run.fail(error)
        finally:
            run.close(cancelled=self._stop.is_set())
            if self._stop.is_set():
                with self._lock:
                    if self._run is run:
                        self._run = None
                        self._pause.clear()

    def pause(self):
        self._pause.set()
        self._wake.set()
        return self.view()

    def resume(self):
        self._pause.clear()
        self._wake.set()
        return self.view()

    def set_speed(self, value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.5 <= value <= 12
        ):
            raise ValueError("Speed must be between 0.5 and 12")
        with self._lock:
            self.speed = float(value)
        self._wake.set()
        return self.view()

    def reset(self):
        with self._lock:
            self._stop.set()
            self._wake.set()
            if self._worker and self._worker.is_alive():
                # Start cannot overlap reset while the owning worker is still
                # closing its world. That worker clears the view after cleanup.
                return {**self.view(), "reset_pending": True}
            self._run = self._worker = None
            self._pause.clear()
            return self.view()

    def close(self):
        with self._lock:
            self._closed = True
            self._stop.set()
            self._wake.set()
            worker = self._worker
        if worker:
            worker.join(timeout=50)


class FactoryHandler(BaseHTTPRequestHandler):
    server_version = "RelayFactory/1.0"

    def _json(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _local(self):
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return host in allowed and (not origin or origin in {f"http://{name}" for name in allowed})

    def do_GET(self):
        if not self._local():
            self._json({"error": "Loopback origin required"}, 403)
            return
        path = unquote(urlparse(self.path).path)
        if path == "/api/factory/state":
            self._json(self.server.session.view())
            return
        filename = (WEB / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        if (
            not filename.is_relative_to(WEB.resolve())
            or not filename.is_file()
            or filename.name not in {"index.html", "styles.css", "app.bundle.js", "favicon.svg"}
        ):
            self.send_error(404)
            return
        body = filename.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(str(filename))[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self._local():
            self._json({"error": "Loopback origin required"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 2048 or self.headers.get_content_type() != "application/json":
                raise ValueError("Small JSON request required")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise TypeError("Expected JSON object")
            action = self.path.removeprefix("/api/factory/")
            session = self.server.session
            if action == "start":
                result = session.start(data)
            elif action in ("pause", "resume", "reset"):
                result = getattr(session, action)()
            elif action == "speed":
                result = session.set_speed(data.get("speed"))
            else:
                self.send_error(404)
                return
            self._json(result)
        except (ValueError, TypeError) as error:
            self._json({"error": str(error)}, 400)
        except Exception as error:  # noqa: BLE001 - contain worker/HTTP boundary faults
            self._json({"error": str(error)}, 500)

    def log_message(self, format, *args):
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def serve_factory(port=8767, coordination="spacetime"):
    session = FactorySession(coordination)
    server = ThreadingHTTPServer(("127.0.0.1", port), FactoryHandler)
    server.session = session
    print(f"Factory dashboard: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        session.close()
