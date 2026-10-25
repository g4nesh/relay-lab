"""Outbound agent bridge: python -m relay_lab.studio.worker --help.

A locally installed handler owns observation, inference, action limits and
execution. Studio supplies objectives and peer feedback; it never imports a
customer's model code on the dashboard server.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import re
import stat
import threading
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

TERMINAL = {"succeeded", "failed", "cancelled"}
# The bounded inbox can contain 100 tasks with up to 64 KiB each.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class AgentHTTPError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"Workspace request failed (HTTP {status}); retained local journal")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_token(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("Token file must be owned by you with chmod 600 permissions")
        with os.fdopen(fd, "r", closefd=False) as stream:
            token = stream.read(129).strip()
        if not re.fullmatch(r"relay_worker_[A-Za-z0-9_-]{20,100}", token):
            raise ValueError("Invalid agent token file")
        return token
    finally:
        os.close(fd)


class AgentClient:
    def __init__(self, url, token):
        parsed = urlsplit(url)
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if (
            parsed.scheme not in ("http", "https")
            or (parsed.scheme != "https" and not local)
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("Use the workspace HTTPS origin (HTTP is allowed only on localhost)")
        self.url, self.token = url.rstrip("/"), token

    def request(self, path, value=None):
        if not re.fullmatch(r"/api/agent/[a-z0-9/-]+", path):
            raise ValueError("Agent requests must use the scoped agent API")
        request = Request(
            self.url + path,
            data=None if value is None else json.dumps(value, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        try:
            with build_opener(NoRedirect()).open(request, timeout=30) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            exc.close()
            raise AgentHTTPError(exc.code) from None
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ValueError("Agent response exceeds the size limit")
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise TypeError("Invalid workspace response")
        return result


def report_value(value):
    if not isinstance(value, dict):
        raise TypeError("Handler must return an outcome object")
    status, progress, summary = value.get("status"), value.get("progress"), value.get("summary")
    if status not in ("running", "blocked", "succeeded", "failed"):
        raise ValueError("Invalid handler status")
    if type(progress) not in (int, float) or not 0 <= progress <= 100:
        raise ValueError("Handler progress must be a finite percentage")
    if status == "succeeded" and progress != 100:
        raise ValueError("Successful handler must report 100 percent progress")
    if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 1000:
        raise ValueError("Handler summary must contain 1–1000 characters")
    evidence = value.get("evidence", [])
    if (
        not isinstance(evidence, list)
        or len(evidence) > 8
        or any(not isinstance(item, str) or not 1 <= len(item.strip()) <= 500 for item in evidence)
    ):
        raise ValueError("Handler evidence must contain up to 8 text items of 1–500 characters")
    result = {"status": status, "progress": progress, "summary": summary, "evidence": evidence}
    json.dumps(result, allow_nan=False)
    return result


class AgentRuntime:
    def __init__(self, client, handler, state_dir, *, label="Customer agent adapter"):
        self.client, self.handler, self.label = client, handler, label
        self.worker = client.request("/api/agent/me")["worker"]
        if (
            not isinstance(self.worker, dict)
            or not isinstance(self.worker.get("id"), str)
            or not re.fullmatch(r"worker-[a-f0-9]{16}", self.worker["id"])
            or self.worker.get("agent_id") not in ("left", "right")
        ):
            raise ValueError("Workspace returned an invalid agent identity")
        # Separate journals even when two agents share one machine.
        self.directory = Path(state_dir) / self.worker["id"]
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.lock = self.directory.joinpath("runtime.lock").open("a")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("This connection already has a running local agent") from None
        self.stopped = threading.Event()
        self.heartbeat_thread = None

    def close(self):
        self.stopped.set()
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)
        if not self.lock.closed:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()

    def _path(self, task_id):
        if not re.fullmatch(r"goal-[a-f0-9]{16}", task_id):
            raise ValueError("Invalid task identifier")
        return self.directory / (task_id + ".json")

    def _save(self, task_id, state):
        path = self._path(task_id)
        temporary = path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _load(self, task_id):
        path = self._path(task_id)
        if not path.exists():
            return {"phase": "new", "step": 0}
        if path.is_symlink() or path.stat().st_size > 128 * 1024:
            raise ValueError("Invalid local task journal")
        return json.loads(path.read_text())

    def current(self, task_id):
        return self.client.request("/api/agent/tasks/" + task_id)

    def cancelled(self, task_id):
        if self.stopped.is_set():
            return True
        try:
            return self.current(task_id)["status"] in TERMINAL
        except Exception:  # noqa: BLE001 - loss of fresh authorization must stop execution
            # The controller must hold when authorization or fresh state is unavailable.
            return True

    def heartbeat(self):
        return self.client.request("/api/agent/heartbeat", {"controller_label": self.label})

    def start_heartbeat(self):
        self.heartbeat()

        def run():
            while not self.stopped.wait(10):
                try:
                    self.heartbeat()
                except Exception:  # noqa: BLE001 - loss of fresh authorization must stop execution
                    self.stopped.set()

        self.heartbeat_thread = threading.Thread(target=run, daemon=True)
        self.heartbeat_thread.start()

    def _pending(self, task, state, operation, value):
        state.update(
            phase="pending",
            operation=operation,
            payload={
                **value,
                "request_id": uuid.uuid4().hex,
                "expected_revision": task["revision"],
            },
        )
        self._save(task["id"], state)

    def _send(self, task_id, state):
        try:
            task = self.client.request(
                f"/api/agent/tasks/{task_id}/{state['operation']}", state["payload"]
            )
        except AgentHTTPError as exc:
            if exc.status != 409:
                raise
            task = self.current(task_id)
            receipt = next(
                (
                    r
                    for r in task.get("request_receipts", [])
                    if r["request_id"] == state["payload"]["request_id"]
                ),
                None,
            )
            if not receipt:
                if task["status"] in TERMINAL:
                    state["phase"] = "done"
                    self._save(task_id, state)
                    return task
                if not task["pending_update"]:
                    # The old proposal was not committed. Preserve the computed
                    # outcome and only rebase its envelope; never repeat the action.
                    state["payload"].update(
                        request_id=uuid.uuid4().hex, expected_revision=task["revision"]
                    )
                    self._save(task_id, state)
                return None
        if state["operation"] == "report":
            state["step"] += 1
            status = state["payload"]["status"]
            state["phase"] = (
                "done"
                if status in TERMINAL
                else "held"
                if state.get("hold")
                else "paused"
                if status == "blocked"
                else "ready"
            )
            state["revision"] = task["revision"]
        else:
            state["phase"] = "ready"
        state.pop("payload", None)
        state.pop("operation", None)
        self._save(task_id, state)
        return task

    def step(self, task):
        task_id = task["id"]
        state = self._load(task_id)
        if task["status"] in TERMINAL:
            if state["phase"] != "done":
                state["phase"] = "done"
                self._save(task_id, state)
            return
        if state["phase"] == "pending":
            task = self._send(task_id, state)
            if task is None:
                return
        # Read fresh feedback after any network retry, before authorizing a step.
        task = self.current(task_id)
        assignment = task["assignments"][self.worker["agent_id"]]
        if (
            task["status"] in TERMINAL
            or assignment["status"] in TERMINAL
            or state["phase"] in ("done", "held")
        ):
            return
        if state["phase"] == "paused" and task["revision"] <= state["revision"]:
            return
        if state["phase"] == "new" and assignment["worker_id"] is None:
            self._pending(task, state, "claim", {})
            task = self._send(task_id, state)
            if task is None:
                return
        elif state["phase"] in ("new", "executing"):
            state["hold"] = True
            self._pending(
                task,
                state,
                "report",
                {
                    "status": "blocked",
                    "progress": assignment["progress"],
                    "summary": "Local execution was interrupted or its journal is missing. Inspect the agent before issuing a new objective; no action was repeated.",
                    "evidence": [],
                },
            )
            self._send(task_id, state)
            return
        if self.stopped.is_set() or task["status"] in TERMINAL:
            return
        if state["step"] >= 50:
            state["hold"] = True
            result = {
                "status": "blocked",
                "progress": assignment["progress"],
                "summary": "Agent step budget reached; review progress and issue a new objective.",
                "evidence": [],
            }
        else:
            # Fsync before calling customer code. An interrupted action is never
            # retried blindly, even if it completed before the process crashed.
            state["phase"] = "executing"
            self._save(task_id, state)
            try:
                result = report_value(
                    self.handler(
                        {
                            "task": task,
                            "worker": dict(self.worker),
                            "step": state["step"],
                            "should_stop": lambda: self.cancelled(task_id),
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001 - customer adapter failures become blocked outcomes
                state["hold"] = True
                result = {
                    "status": "blocked",
                    "progress": assignment["progress"],
                    "summary": f"Agent handler stopped ({type(exc).__name__}); inspect local logs before issuing a new objective.",
                    "evidence": [],
                }
        self._pending(task, state, "report", result)
        self._send(task_id, state)

    def run_once(self):
        if self.stopped.is_set():
            raise RuntimeError("Agent connection lost its heartbeat; controller must hold")
        tasks = self.client.request("/api/agent/tasks")["tasks"]
        for task in tasks:
            self.step(task)
        return len(tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--handler", required=True, help="Locally installed module:function")
    parser.add_argument("--state-dir", type=Path, default=Path(".relay-agent"))
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--once", action="store_true", help="Process one inbox pass")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    module, separator, function = args.handler.partition(":")
    if not separator:
        parser.error("handler must use module:function syntax")
    handler = getattr(importlib.import_module(module), function)
    if not callable(handler):
        parser.error("handler must be callable")
    runtime = AgentRuntime(
        AgentClient(args.url, read_token(args.token_file)),
        handler,
        args.state_dir,
        label=args.handler[:120],
    )
    try:
        runtime.start_heartbeat()
        while True:
            try:
                count = runtime.run_once()
                print(f"Agent inbox checked: {count} assigned tasks", flush=True)
            except Exception as exc:
                print(f"Agent paused: {type(exc).__name__}; journal retained", flush=True)
                if args.once or isinstance(exc, AgentHTTPError) and exc.status in (401, 403):
                    raise
            if args.once:
                break
            if runtime.stopped.wait(args.interval):
                raise RuntimeError("Heartbeat unavailable; stopping agent polling")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
