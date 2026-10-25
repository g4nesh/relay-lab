"""Loopback-only Studio UI and a small bridge to existing coordination services."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import struct
import uuid
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .service import StudioService, example_workspace

WEB = Path(__file__).parent / "web"
ASSETS = {
    "/": ("index.html", "text/html"),
    "/studio.css": ("studio.css", "text/css"),
    "/studio.js": ("studio.js", "text/javascript"),
}
MAX_UPLOAD = 4 * 1024**3
SESSION_ROUTE = r"/api/sessions/(studio-[a-f0-9]{16})"


def validate_safetensors(path):
    """Check the inert file envelope without importing or executing model code."""
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("Not a safetensors file")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(16 * 1024**2, size - 8):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(f.read(length))
    if not isinstance(header, dict):
        raise TypeError("Invalid safetensors header")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("The weights file contains no tensors")
    for tensor in tensors.values():
        if not isinstance(tensor, dict) or not {"dtype", "shape", "data_offsets"} <= tensor.keys():
            raise ValueError("Invalid tensor metadata")
        offsets = tensor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
        ):
            raise ValueError("Invalid tensor offsets")
        if not 0 <= offsets[0] <= offsets[1] <= size - 8 - length:
            raise ValueError("Tensor data is incomplete")
    return len(tensors)


class StudioHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, service, token, **kwargs):
        self.service, self.token = service, token
        super().__init__(*args, **kwargs)

    def respond(self, value, status=200, mime="application/json"):
        body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        charset = (
            "; charset=utf-8" if mime.startswith("text/") or mime == "application/json" else ""
        )
        self.send_header("Content-Type", mime + charset)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def guard(self, mutation=False):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            raise PermissionError("Studio accepts only its local origin")
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + h for h in hosts}:
            raise PermissionError("Cross-origin requests are not allowed")
        if mutation and not secrets.compare_digest(
            self.headers.get("X-Relay-Token", ""), self.token
        ):
            raise PermissionError("Refresh the page to obtain a local workspace token")

    def body(self):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("Expected application/json")
        length = int(self.headers.get("Content-Length", "0"))
        if not 1 <= length <= 128 * 1024:
            raise ValueError("Request must be 1–131072 bytes")
        value = json.loads(
            self.rfile.read(length),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
        )
        if not isinstance(value, dict):
            raise TypeError("Expected a JSON object")
        return value

    def dispatch(self, mutation=False):
        self.guard(mutation)
        path = urlsplit(self.path).path
        if not mutation:
            if path in ASSETS:
                filename, mime = ASSETS[path]
                return self.respond((WEB / filename).read_bytes(), mime=mime)
            if path == "/api/bootstrap":
                return self.respond(
                    {
                        "token": self.token,
                        "workspace": self.service.workspace(),
                        "example": example_workspace(),
                        "capabilities": self.service.capabilities(),
                    }
                )
            if path == "/api/health":
                return self.respond(self.service.health())
            if path == "/api/sessions":
                return self.respond(self.service.list_sessions())
            if match := re.fullmatch(SESSION_ROUTE, path):
                return self.respond(self.service.history(match[1]))
            if match := re.fullmatch(
                r"/api/samples/(clear|left_obstacle|right_obstacle)/(left|right)\.png", path
            ):
                return self.respond(self.service.sample_image(match[1], match[2]), mime="image/png")
        else:
            if path == "/api/upload":
                return self.upload()
            value = self.body()
            if path == "/api/workspace":
                return self.respond(self.service.save(value))
            if path == "/api/diagnostics":
                return self.respond(self.service.diagnostics(value))
            if path == "/api/sessions":
                return self.respond(self.service.connect(value), 201)
            if match := re.fullmatch(SESSION_ROUTE + r"/exchange", path):
                return self.respond(self.service.exchange(match[1], value))
            if match := re.fullmatch(SESSION_ROUTE + r"/resume", path):
                return self.respond(self.service.recover(match[1]))
        self.respond({"error": "Not found"}, 404)

    def upload(self):
        name = unquote(self.headers.get("X-Filename", ""))
        if (
            not name
            or len(name) > 200
            or Path(name).name != name
            or "\\" in name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Upload a .safetensors weights file")
        length = int(self.headers.get("Content-Length", "0"))
        if not 10 <= length <= MAX_UPLOAD:
            raise ValueError(
                "Weights must be between 10 bytes and 4 GiB; use a local path for larger files"
            )
        asset = uuid.uuid4().hex + ".safetensors"
        path = self.service.uploads / asset
        digest, remaining = hashlib.sha256(), length
        try:
            with path.open("xb") as output:
                while remaining:
                    block = self.rfile.read(min(1024**2, remaining))
                    if not block:
                        raise ValueError("Upload was interrupted")
                    output.write(block)
                    digest.update(block)
                    remaining -= len(block)
            tensors = validate_safetensors(path)
            metadata = {
                "asset": asset,
                "name": name,
                "bytes": length,
                "sha256": digest.hexdigest(),
                "tensors": tensors,
                "loaded": False,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata))
            self.respond(metadata, 201)
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def handle_request(self, mutation=False):
        try:
            self.dispatch(mutation)
        except PermissionError as error:
            self.respond({"error": str(error)}, 403)
        except (ValueError, TypeError, KeyError) as error:
            self.respond({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:  # noqa: BLE001 - report failed services without a fallback
            # Surface failures; never manufacture successful inference/DB results.
            self.respond({"error": f"{type(error).__name__}: {error}"}, 503)

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request(True)

    def log_message(self, *args):
        pass


def serve_studio(root=None, port=8770):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid Studio port")
    service = StudioService(root or Path.cwd())
    handler = partial(StudioHandler, service=service, token=secrets.token_urlsafe(32))
    try:
        with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
            print(f"Relay Studio: http://127.0.0.1:{server.server_port}/", flush=True)
            server.serve_forever()
    finally:
        service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    options = parser.parse_args()
    serve_studio(options.root, options.port)
