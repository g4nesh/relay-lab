"""Real HTTP handler checks for loopback replay access boundaries and MP4 seeking."""

import gzip
import json
from http.client import HTTPResponse
from io import BytesIO
from types import SimpleNamespace

import pytest

from relay_lab.cloth.server import ClothReplayHandler, serve_cloth


class MemorySocket:
    def __init__(self, incoming):
        self.incoming, self.sent = incoming, bytearray()

    def makefile(self, *args):
        return BytesIO(self.incoming)

    def sendall(self, value):
        self.sent.extend(value)


@pytest.fixture
def exchange(tmp_path):
    root = tmp_path / "recordings"
    root.mkdir()
    run = root / "episode"
    run.mkdir()
    (run / "video.mp4").write_bytes(bytes(range(256)) * 100)
    (run / "summary.json").write_text('{"success":false}')
    (root / "manifest.json").write_text(json.dumps({"title": "Cloth test", "runs": []}))
    (root / ".private.json").write_text('{"private":true}')
    outside = tmp_path / "outside.json"
    outside.write_text('{"private":true}')
    (run / "linked.json").symlink_to(outside)
    (run / "trace.json.gz").write_bytes(gzip.compress(b'{"frames":[]}'))

    def request(path="/", method="GET", **headers):
        lines = [f"{method} {path} HTTP/1.1", "Host: localhost", "Connection: close"]
        lines.extend(f"{key.replace('_', '-')}: {value}" for key, value in headers.items())
        socket = MemorySocket(("\r\n".join(lines) + "\r\n\r\n").encode())
        ClothReplayHandler(socket, ("127.0.0.1", 12345), SimpleNamespace(), directory=root)
        response = HTTPResponse(MemorySocket(bytes(socket.sent)), method=method)
        response.begin()
        return response.status, dict(response.getheaders()), response.read()

    return root, request


def test_ui_and_manifest_are_available_without_starting_physics(exchange):
    root, request = exchange
    status, headers, body = request()
    assert status == 200 and b"Cloth coordination" in body
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert request("/styles.css")[0] == 200
    assert request("/app.js")[0] == 200
    status, headers, body = request("/api/manifest")
    assert status == 200 and json.loads(body) == {"title": "Cloth test", "runs": []}
    assert headers["Content-Type"].startswith("application/json")
    (root / "manifest.json").write_text('{"title":"Updated","runs":[]}')
    assert json.loads(request("/api/manifest")[2])["title"] == "Updated"


def test_manifest_head_and_invalid_manifest(exchange):
    root, request = exchange
    assert request("/api/manifest", method="HEAD")[2] == b""
    (root / "manifest.json").write_text('{"runs":null}')
    assert request("/api/manifest")[0] == 500


@pytest.mark.parametrize(
    "path",
    [
        "/runs/../outside.json",
        "/runs/%2e%2e/outside.json",
        "/runs/%2f../outside.json",
        "/runs/episode/linked.json",
        "/runs/.private.json",
        "/runs/episode/../../outside.json",
        "/runs/episode",
        "/runs/",
        "/runs/episode/",
        "/manifest.json",
        "/api/secrets",
        "/runs/episode/%00.json",
        "/runs/episode/..%5coutside.json",
        "/server.py",
    ],
)
def test_no_traversal_symlink_escape_private_files_or_directory_browsing(exchange, path):
    _, request = exchange
    assert request(path)[0] == 404


def test_video_byte_ranges_are_seekable_and_exact(exchange):
    _, request = exchange
    status, headers, body = request("/runs/episode/video.mp4", Range="bytes=100-199")
    assert status == 206
    assert headers["Content-Range"] == "bytes 100-199/25600"
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "video/mp4"
    assert body == bytes(range(100, 200))
    status, headers, body = request("/runs/episode/video.mp4", method="HEAD", Range="bytes=0-49")
    assert status == 206 and body == b"" and headers["Content-Length"] == "50"
    assert request("/runs/episode/video.mp4", Range="bytes=999999-")[0] == 416


def test_trace_and_summary_preserve_recorded_failure_and_compression(exchange):
    _, request = exchange
    assert json.loads(request("/runs/episode/summary.json")[2]) == {"success": False}
    status, headers, body = request("/runs/episode/trace.json.gz")
    assert status == 200 and headers["Content-Encoding"] == "gzip"
    assert {key.lower(): value for key, value in headers.items()}[
        "content-type"
    ] == "application/json"
    assert json.loads(gzip.decompress(body)) == {"frames": []}


def test_serve_requires_manifest_and_valid_loopback_port(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        serve_cloth(tmp_path)
    (tmp_path / "manifest.json").write_text('{"runs":[]}')
    for invalid in (-1, 65536, True):
        with pytest.raises(ValueError, match="port"):
            serve_cloth(tmp_path, port=invalid)


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
def test_browser_cancelled_media_requests_are_normal_disconnects(tmp_path, error):
    class CancelledSocket(MemorySocket):
        def sendall(self, value):
            raise error("Browser cancelled request")

    socket = CancelledSocket(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
    ClothReplayHandler(socket, ("127.0.0.1", 12345), SimpleNamespace(), directory=tmp_path)
