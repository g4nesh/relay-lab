"""Worker push notifications are hints, never execution authority."""

import io
import json
import threading
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from relay_lab.studio import worker as sdk


def test_sse_bounded_frames_and_comments():
    body = (
        b': heartbeat\n\nevent: sync\ndata: {"topics":["tasks"],\n'
        b'data: "sequence":1}\n\nevent: update\ndata: malformed\n\n'
        b"event: update\ndata: [1]\n\nevent: unsupported\ndata: {}\n\n"
        b'event: status\ndata: {"state":"ready"}\n\n'
    )
    assert list(sdk.stream_events(io.BytesIO(body))) == [
        ("sync", {"topics": ["tasks"], "sequence": 1}),
        ("status", {"state": "ready"}),
    ]
    with pytest.raises(ValueError, match="size limit"):
        list(sdk.stream_events(io.BytesIO(b"data:" + b"x" * sdk.MAX_EVENT_BYTES)))


def test_event_request_uses_header_not_url_and_rejects_redirects(monkeypatch):
    seen = []
    response = io.BytesIO()
    response.headers = Message()
    response.headers["Content-Type"] = "text/event-stream; charset=utf-8"

    class Opener:
        def open(self, request, timeout):
            seen.append((request, timeout))
            return response

    def opener(handler):
        assert isinstance(handler, sdk.NoRedirect)
        assert handler.redirect_request(None) is None
        return Opener()

    monkeypatch.setattr(sdk, "build_opener", opener)
    client = sdk.AgentClient("https://workspace.example", "private-token")
    assert client.open_events() is response
    request, timeout = seen[0]
    assert request.full_url == "https://workspace.example/api/agent/events"
    assert request.get_header("Authorization") == "Bearer private-token"
    assert timeout == 25


@pytest.mark.parametrize("status", [401, 403])
def test_event_authorization_loss_stops_runtime(status):
    class Client:
        def open_events(self):
            raise sdk.AgentHTTPError(status)

    stopped, wakeup = threading.Event(), threading.Event()
    events = sdk.AgentEvents(Client(), stopped, wakeup)
    events.start()
    events.thread.join(timeout=1)
    assert stopped.is_set() and wakeup.is_set()
    assert not events.thread.is_alive()


def test_old_server_falls_back_without_stopping_runtime():
    class Client:
        def open_events(self):
            raise sdk.AgentHTTPError(404)

    stopped, wakeup = threading.Event(), threading.Event()
    events = sdk.AgentEvents(Client(), stopped, wakeup)
    events.start()
    events.thread.join(timeout=1)
    assert not stopped.is_set()
    assert not events.connected
    assert not events.thread.is_alive()


def test_duplicate_events_coalesce_and_do_not_execute_actions():
    body = b'event: update\ndata: {"sequence":1,"topics":["tasks"]}\n\n' * 10
    stopped, wakeup = threading.Event(), threading.Event()

    class Response(io.BytesIO):
        def __exit__(self, *args):
            stopped.set()
            return super().__exit__(*args)

    class Client:
        def open_events(self):
            return Response(body)

        def request(self, *args):
            raise AssertionError("Stream reader must not execute API operations")

    events = sdk.AgentEvents(Client(), stopped, wakeup)
    events.start()
    events.thread.join(timeout=1)
    assert wakeup.is_set()
    assert not events.thread.is_alive()


def test_degraded_sync_does_not_mark_stream_ready():
    stopped, wakeup, paused, release = (threading.Event() for _ in range(4))

    class Response(io.BytesIO):
        def readline(self, *args):
            value = super().readline(*args)
            if not value:
                paused.set()
                release.wait(2)
                stopped.set()
            return value

    class Client:
        def open_events(self):
            return Response(
                b'event: status\ndata: {"state":"degraded"}\n\n'
                b'event: sync\ndata: {"topics":["tasks"],"sequence":1}\n\n'
            )

    events = sdk.AgentEvents(Client(), stopped, wakeup)
    try:
        events.start()
        assert paused.wait(1)
        assert wakeup.is_set()
        assert not events.connected
    finally:
        release.set()
        events.close()


def test_close_unblocks_idle_live_connection():
    connected, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
            connected.set()
            release.wait(3)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stopped, wakeup = threading.Event(), threading.Event()
    client = sdk.AgentClient(f"http://127.0.0.1:{server.server_port}", "private-token")
    events = sdk.AgentEvents(client, stopped, wakeup)
    try:
        events.start()
        assert connected.wait(2)
        assert wakeup.wait(2)
        events.close()
        assert not events.thread.is_alive()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


def test_fallback_wakeup_keeps_runtime_cancellation_gate(tmp_path):
    class Client:
        def request(self, path):
            if path == "/api/agent/me":
                return {"worker": {"id": "worker-0123456789abcdef", "agent_id": "left"}}
            raise sdk.AgentHTTPError(403)

    runtime = sdk.AgentRuntime(Client(), lambda _: None, tmp_path)
    try:
        assert runtime.events is None  # One-shot runs never create a stream.
        runtime.wakeup.set()
        assert not runtime.wait_for_update(1)
        assert runtime.cancelled("goal-0123456789abcdef")
        runtime.stopped.set()
        runtime.wakeup.set()
        assert runtime.wait_for_update(1)
        with pytest.raises(RuntimeError, match="hold"):
            runtime.run_once()
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "event,payload,expected_wakes",
    [
        ("update", {"topics": ["workers"]}, 1),
        ("update", {"topics": ["coordination", "workers"]}, 1),
        ("update", {"topics": []}, 1),
        ("update", {"topics": "tasks"}, 1),
        ("update", {}, 1),
        ("update", {"topics": ["tasks"]}, 2),
        ("update", {"topics": ["coordination", "tasks"]}, 2),
        ("sync", {"topics": ["workers"]}, 2),
        ("sync", {}, 2),
        ("status", {"state": "ready"}, 1),
    ],
)
def test_only_task_updates_and_sync_wake_inbox(event, payload, expected_wakes):
    stopped = threading.Event()

    class Wakeup(threading.Event):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def set(self):
            self.calls += 1
            super().set()

    class Response(io.BytesIO):
        def __exit__(self, *args):
            stopped.set()
            return super().__exit__(*args)

    class Client:
        def open_events(self):
            return Response(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode())

    wakeup = Wakeup()
    events = sdk.AgentEvents(Client(), stopped, wakeup)
    events.start()
    events.thread.join(timeout=1)
    assert not events.thread.is_alive()
    # The first wake is mandatory reconciliation when the stream opens.
    assert wakeup.calls == expected_wakes


@pytest.mark.parametrize("state", ["revoked", "unauthorized"])
def test_authority_status_stops_even_without_task_topic(state):
    stopped, wakeup = threading.Event(), threading.Event()

    class Client:
        def open_events(self):
            return io.BytesIO(f'event: status\ndata: {{"state":"{state}"}}\n\n'.encode())

    events = sdk.AgentEvents(Client(), stopped, wakeup)
    events.start()
    events.thread.join(timeout=1)
    assert stopped.is_set() and wakeup.is_set()
    assert not events.thread.is_alive()
