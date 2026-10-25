"""Read-only, loopback cloth replay UI with bounded local artifact access."""

from __future__ import annotations

import json
from functools import partial
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit

from relay_lab.server import ReplayRequestHandler

WEB = Path(__file__).parent / "web"
ASSETS = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
    "/latent": "latent.html",
    "/latent.js": "latent.js",
    "/latent.css": "latent.css",
}
EXTENSIONS = {".json", ".mp4", ".png", ".jpg", ".jpeg", ".webp", ".gz"}


class ClothReplayHandler(ReplayRequestHandler):
    def __init__(self, *args, directory, **kwargs):
        self.artifact_root = Path(directory).expanduser().resolve()
        super().__init__(*args, directory=str(self.artifact_root), **kwargs)

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Seeking or switching episodes cancels in-flight browser range requests.
            pass

    def artifact_path(self, url):
        path = unquote(urlsplit(url).path)
        if "\x00" in path or "\\" in path or any(part in (".", "..") for part in path.split("/")):
            return None
        if path in ASSETS:
            return WEB / ASSETS[path]
        if path == "/api/manifest":
            candidate = self.artifact_root / "manifest.json"
        elif path.startswith("/runs/"):
            relative = path.removeprefix("/runs/")
            if not relative or any(part.startswith(".") for part in relative.split("/")):
                return None
            candidate = self.artifact_root / relative
        else:
            return None
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.artifact_root) or resolved.suffix not in EXTENSIONS:
            return None
        return resolved

    def translate_path(self, path):
        candidate = self.artifact_path(path)
        return str(candidate if candidate is not None else self.artifact_root / ".not-served")

    def send_head(self):
        self._byte_range = None
        path = self.artifact_path(self.path)
        if path is None or not path.is_file():
            self.send_error(404, "Replay artifact not found")
            return None
        if unquote(urlsplit(self.path).path) == "/api/manifest":
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
                    raise TypeError("Manifest requires a runs list")
                body = json.dumps(value, allow_nan=False).encode()
            except (OSError, ValueError, TypeError):
                self.send_error(500, "Replay manifest is not valid")
                return None
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return BytesIO(body)
        return super().send_head()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        path = self.artifact_path(self.path)
        if path is not None and path.name.endswith(".json.gz"):
            self.send_header("Content-Encoding", "gzip")
        super().end_headers()

    def guess_type(self, path):
        if str(path).endswith(".json.gz"):
            return "application/json"
        return super().guess_type(path)

    def log_message(self, format, *args):
        # Recordings generate many range requests. Errors remain visible in the UI.
        pass


def serve_cloth(directory: Path | str, port: int = 8769):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir() or not (directory / "manifest.json").is_file():
        raise ValueError("Cloth replay directory must contain manifest.json")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    handler = partial(ClothReplayHandler, directory=directory)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Cloth replay: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Cloth replay stopped.", flush=True)
