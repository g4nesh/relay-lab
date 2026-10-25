"""Private SDK subscription bridge; only bounded invalidations reach clients.

The database owner credential is passed through a private stdin pipe, never
arguments, logs, browser state, or SSE. Notifications are hints: every consumer
must reload and authorize the authoritative state before acting.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

TOPICS = frozenset({"tasks", "workers", "coordination"})


class RealtimeHub:
    def __init__(self, service, *, bridge=None, max_listeners=16):
        self.service = service
        self.bridge = Path(
            bridge
            or os.environ.get(
                "RELAY_STUDIO_SUBSCRIPTION_BRIDGE",
                str(Path(__file__).resolve().parents[3] / "subscriptions/src/bridge.cjs"),
            )
        )
        self.max_listeners = max_listeners
        self.listeners = set()
        self.sequence = 0
        self.state = "starting"
        self.reason = "connecting"
        self.closed = False
        self.task = None
        self.process = None

    def snapshot(self):
        return {
            "state": self.state,
            "status": self.state,
            "reason": self.reason,
            "transport": "spacetime-subscriptions",
            "polling_fallback": self.state != "ready",
            "connections": len(self.listeners),
            "connection_limit": self.max_listeners,
        }

    def status(self):
        return self.snapshot()

    def start(self):
        if self.task is None and not self.closed:
            self.task = asyncio.create_task(self._run(), name="studio-subscription-bridge")

    def subscribe(self):
        if self.closed or len(self.listeners) >= self.max_listeners:
            raise RuntimeError("Live connection limit reached")
        queue = asyncio.Queue(maxsize=1)
        self.listeners.add(queue)
        queue.put_nowait({"event": "sync", "topics": sorted(TOPICS), "sequence": self.sequence})
        self.start()
        return queue

    def unsubscribe(self, queue):
        self.listeners.discard(queue)

    def publish(self, topics, *, sync=False):
        topics = set(topics) & TOPICS
        if not topics:
            return
        self.sequence += 1
        for queue in tuple(self.listeners):
            combined, event = set(topics), "sync" if sync else "update"
            if queue.full():
                previous = queue.get_nowait()
                combined.update(previous["topics"])
                if previous["event"] == "sync":
                    event = "sync"
            queue.put_nowait(
                {"event": event, "topics": sorted(combined), "sequence": self.sequence}
            )

    def _state(self, state, reason):
        changed = (self.state, self.reason) != (state, reason)
        self.state, self.reason = state, reason
        if changed:
            self.publish(TOPICS, sync=True)

    def accept(self, value):
        """Strict protocol: ignore fields that could carry database payloads."""
        if not isinstance(value, dict):
            raise TypeError("Invalid bridge message")
        kind = value.get("type")
        if kind == "status" and value.get("status") in {"ready", "degraded"}:
            ready = value["status"] == "ready"
            self._state("ready" if ready else "degraded", "connected" if ready else "disconnected")
        elif kind == "update" and isinstance(value.get("topics"), list):
            topics = value["topics"]
            if len(topics) > len(TOPICS) or any(
                type(t) is not str or t not in TOPICS for t in topics
            ):
                raise ValueError("Invalid bridge topics")
            if self.state == "ready":
                self.publish(topics)
        elif kind != "heartbeat":
            raise ValueError("Invalid bridge message")

    async def _stop_process(self):
        process, self.process = self.process, None
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), 3)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _run(self):
        delay = 1
        while not self.closed:
            try:
                node = shutil.which("node")
                if not node or not self.bridge.is_file():
                    self._state("degraded", "bridge_unavailable")
                    await asyncio.sleep(30)
                    continue
                config = await asyncio.to_thread(lambda: self.service.factory().client.config)
                self.process = await asyncio.create_subprocess_exec(
                    node,
                    str(self.bridge),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=8192,
                    env={"PATH": os.environ.get("PATH", ""), "NODE_NO_WARNINGS": "1"},
                )
                self.process.stdin.write(
                    json.dumps(
                        {
                            "server": config.server,
                            "database": config.database,
                            "token": config.token,
                        }
                    ).encode()
                    + b"\n"
                )
                await self.process.stdin.drain()
                self.process.stdin.close()
                while not self.closed:
                    line = await asyncio.wait_for(self.process.stdout.readline(), 40)
                    if not line or len(line) > 4096:
                        raise ValueError("Bridge disconnected")
                    self.accept(json.loads(line))
                    if self.state == "ready":
                        delay = 1
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001 - never expose SDK/config errors or credentials
                self._state("degraded", "subscription_unavailable")
            finally:
                await self._stop_process()
            if not self.closed:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def close(self):
        self.closed = True
        self._state("closed", "shutdown")
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self._stop_process()
