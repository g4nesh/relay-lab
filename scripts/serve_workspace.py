"""Serve the research website and authenticated Studio together on loopback."""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import sys
from email.message import Message
from pathlib import Path
from urllib.parse import unquote

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))

from relay_lab.studio.asgi import StudioApplication, create_app, serve_app
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.server import ASSETS, WEB


class WebsiteApplication:
    """Public website files around the unmodified Studio ASGI application.

    All API authorization, subscriptions and lifecycle work stays with Studio.
    Static routes are read-only and enforce the same loopback Host/Origin policy.
    """

    def __init__(self, studio, *, site=BACKEND / "website", web=WEB):
        if studio.config.hosted or studio.config.bind not in ("127.0.0.1", "::1"):
            raise ValueError("The research website launcher is loopback-only")
        self.studio = studio
        self.config = studio.config
        self.site = Path(site).resolve()
        self.web = Path(web).resolve()
        self.static_inflight = 0

    def _target(self, path):
        target = (self.site / unquote(path).lstrip("/")).resolve()
        if target.is_dir():
            target = (target / "index.html").resolve()
        if not target.is_relative_to(self.site) or not target.is_file():
            return None
        return target

    def _studio_page(self):
        page = (self.web / "index.html").read_text()
        page = page.replace("http://127.0.0.1:8769/latent", "/projects/cloth/latent.html")
        page = page.replace("http://127.0.0.1:8769/", "/projects/cloth/")
        page = page.replace(
            "</body>", '<script src="/projects/studio/local-tabs.js?v=1" defer></script></body>'
        )
        return page.replace(
            'aria-label="Relay Studio home"', 'aria-label="Relay website homepage"'
        ).encode()

    @staticmethod
    async def _start(send, status, headers, body=b"", *, more=False):
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": more})

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or path.startswith("/api/")
            or path in ("/livez", "/readyz")
            or path in ASSETS and path != "/"
        ):
            return await self.studio(scope, receive, send)
        if scope["method"] not in ("GET", "HEAD"):
            return await StudioApplication._respond(send, 405, "Method not allowed")
        headers = Message()
        counts = {}
        for key, value in scope.get("headers", []):
            key = key.lower()
            counts[key] = counts.get(key, 0) + 1
            headers[key.decode("ascii")] = value.decode("latin1")
        if (
            counts.get(b"host", 0) != 1
            or counts.get(b"origin", 0) > 1
            or sum(len(k) + len(v) for k, v in scope.get("headers", [])) > 32768
        ):
            return await StudioApplication._respond(send, 400, "Invalid request headers")
        try:
            self.studio.config.guard(headers)
        except PermissionError as error:
            return await StudioApplication._respond(send, 403, str(error))
        if self.studio.closed or self.static_inflight >= self.studio.config.max_concurrent:
            return await StudioApplication._respond(send, 503, "Website is busy; try again shortly")
        self.static_inflight += 1
        try:
            if path in ("/projects/studio", "/projects/studio/"):
                page = await asyncio.to_thread(self._studio_page)
                return await self._start(send, 200, [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"content-length", str(len(page)).encode()),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ], b"" if scope["method"] == "HEAD" else page)
            try:
                target = await asyncio.to_thread(self._target, path)
            except (ValueError, OSError):
                target = None
            if target is None:
                return await StudioApplication._respond(send, 404, "Not found")
            stat = await asyncio.to_thread(target.stat)
            mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'.encode()
            binary = path.startswith("/assets/") or mime.startswith(("image/", "video/", "font/"))
            response_headers = [
                (b"content-type", mime.encode()),
                (b"etag", etag),
                (b"cache-control", b"public, max-age=604800" if binary else b"no-cache"),
                (b"x-content-type-options", b"nosniff"),
            ]
            if headers.get("If-None-Match", "").encode() == etag:
                return await self._start(send, 304, response_headers)
            response_headers.append((b"content-length", str(stat.st_size).encode()))
            if scope["method"] == "HEAD":
                return await self._start(send, 200, response_headers)
            # Stream recordings in bounded chunks; do not load videos into memory.
            with target.open("rb") as source:
                await send({"type": "http.response.start", "status": 200, "headers": response_headers})
                while chunk := await asyncio.to_thread(source.read, 256 * 1024):
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            self.static_inflight -= 1


def create_workspace_app(*, port=8780, root=BACKEND):
    # create_app honors RELAY_STUDIO_EXPERIMENTS=0. The full research package
    # supplies experiment support by default; the isolated export does not.
    config = DeploymentConfig(root=Path(root).resolve(), port=port)
    return WebsiteApplication(create_app(config=config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    app = create_workspace_app(port=args.port)
    print(f"Relay website: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Working Studio: http://127.0.0.1:{args.port}/projects/studio/", flush=True)
    serve_app(app, config=app.config)


if __name__ == "__main__":
    main()
