"""Exercise real HTTP parsing/response paths without requiring a listening socket."""

from http.client import HTTPResponse
from io import BytesIO
from types import SimpleNamespace

import pytest

from relay_lab.cli import parser
from relay_lab.server import ReplayRequestHandler, serve


class MemorySocket:
    def __init__(self, incoming):
        self.incoming = incoming
        self.sent = bytearray()

    def makefile(self, *args):
        return BytesIO(self.incoming)

    def sendall(self, value):
        self.sent.extend(value)


@pytest.fixture
def http_exchange(tmp_path):
    payload = bytes(range(256)) * 1024
    (tmp_path / "rollout.mp4").write_bytes(payload)
    (tmp_path / "empty.mp4").touch()

    def exchange(method="GET", path="/rollout.mp4", **headers):
        lines = [f"{method} {path} HTTP/1.1", "Host: localhost", "Connection: close"]
        lines.extend(f"{key.replace('_', '-')}: {value}" for key, value in headers.items())
        socket = MemorySocket(("\r\n".join(lines) + "\r\n\r\n").encode())
        ReplayRequestHandler(
            socket, ("127.0.0.1", 12345), SimpleNamespace(), directory=str(tmp_path)
        )
        response = HTTPResponse(MemorySocket(bytes(socket.sent)), method=method)
        response.begin()
        return response.status, dict(response.getheaders()), response.read()

    return payload, exchange


@pytest.mark.parametrize(
    ("requested", "start", "end"),
    [
        ("bytes=12345-12444", 12345, 12444),
        ("bytes=262044-", 262044, 262143),
        ("bytes=-77", 262067, 262143),
        ("bytes=262140-999999", 262140, 262143),
        ("bytes=0-131072", 0, 131072),
    ],
)
def test_byte_ranges_return_exact_payload_and_headers(http_exchange, requested, start, end):
    payload, exchange = http_exchange
    status, headers, body = exchange(Range=requested)
    assert status == 206
    assert headers["Content-Range"] == f"bytes {start}-{end}/{len(payload)}"
    assert headers["Content-Length"] == str(end - start + 1)
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "video/mp4"
    assert body == payload[start : end + 1]


def test_full_get_and_head_advertise_seekability_without_losing_content(http_exchange):
    payload, exchange = http_exchange
    status, headers, body = exchange()
    assert status == 200 and body == payload
    assert headers["Accept-Ranges"] == "bytes"
    status, headers, body = exchange("HEAD", Range="bytes=50-99")
    assert status == 206 and body == b""
    assert headers["Content-Length"] == "50"
    assert headers["Content-Range"] == f"bytes 50-99/{len(payload)}"
    status, headers, body = exchange("HEAD")
    assert status == 200 and body == b""
    assert headers["Content-Length"] == str(len(payload))


@pytest.mark.parametrize("requested", ["bytes=262144-", "bytes=50-49", "bytes=-0", "bytes=-"])
def test_unsatisfiable_range_has_416_and_current_size(http_exchange, requested):
    payload, exchange = http_exchange
    status, headers, body = exchange(Range=requested)
    assert status == 416 and body == b""
    assert headers["Content-Range"] == f"bytes */{len(payload)}"
    assert headers["Content-Length"] == "0"


def test_empty_missing_and_unhandled_range_requests(http_exchange):
    payload, exchange = http_exchange
    status, headers, body = exchange(path="/empty.mp4", Range="bytes=0-")
    assert status == 416 and headers["Content-Range"] == "bytes */0"
    assert exchange(path="/missing.mp4", Range="bytes=0-9")[0] == 404
    for requested in ("bytes=0-1,4-5", "items=0-2", "invalid"):
        status, _, body = exchange(Range=requested)
        assert status == 200 and body == payload
    status, _, body = exchange(Range="bytes=0-9", If_Range='"unknown-etag"')
    assert status == 200 and body == payload


def test_serve_cli_and_directory_validation(tmp_path):
    args = parser().parse_args(["serve", "--directory", str(tmp_path), "--port", "8766"])
    assert args.directory == tmp_path and args.port == 8766
    with pytest.raises(ValueError, match="does not exist"):
        serve(tmp_path / "missing")
    with pytest.raises(ValueError, match="port must"):
        serve(tmp_path, 99999)
