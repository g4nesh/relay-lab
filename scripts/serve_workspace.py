#!/usr/bin/env python3
"""Serve the Relay website and real Studio together on loopback only."""
import argparse
import mimetypes
import os
import sys
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote

BACKEND = Path(__file__).resolve().parents[1]
SITE = BACKEND / 'website'
sys.path.insert(0, str(BACKEND / 'src'))
from relay_lab.studio.server import StudioHandler, WEB, ASSETS
from relay_lab.studio.service import StudioService
from relay_lab.studio.profiles import ProfileStore
from relay_lab.studio.access import AccessStore
from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.analytics import AnalyticsService
from relay_lab.studio.tasks import TaskService
from relay_lab.studio.workers import WorkerStore

class LocalWorkspace(StudioHandler):
    def dispatch_authorized(self, mutation=False):
        path = urlsplit(self.path).path
        self.guard()
        if not mutation and path in ('/projects/studio/', '/projects/studio'):
            page = (WEB / 'index.html').read_text()
            page = page.replace('Connect robot policies. Make every exchange inspectable.', 'Configure the models and review how they share information.')
            page = page.replace('</head>', '<link rel="stylesheet" href="/projects/studio/hosted.css?v=local1"><link rel="preload" href="/assets/fonts/ibm-plex-sans-latin-400.woff2" as="font" type="font/woff2" crossorigin><link rel="preload" href="/assets/fonts/ibm-plex-sans-latin-600.woff2" as="font" type="font/woff2" crossorigin><link rel="stylesheet" href="/projects/studio/local.css?v=5"><link rel="stylesheet" href="/projects/shared/workspace-design.css?v=10"></head>')
            page = page.replace('http://127.0.0.1:8769/latent', '/projects/cloth/latent.html').replace('http://127.0.0.1:8769/', '/projects/cloth/')
            page = page.replace('</body>', '<script src="/projects/studio/local-tabs.js?v=1" defer></script></body>')
            page = page.replace('aria-label="Relay Studio home"', 'aria-label="Relay website homepage"')
            for before, after in [('Two minds.<span>One workcell.</span>', 'Workspace'), ('Approved to work together.', 'Models'), ('Team topology', 'Models and communication'), ('Channel configuration','Communication settings')]:
                page = page.replace(before, after)
            return self.respond(page.encode(), mime='text/html')
        if path.startswith('/api/') or path in ASSETS and path != '/':
            return super().dispatch_authorized(mutation)
        if mutation:
            return self.respond({'error':'Not found'}, 404)
        target = (SITE / unquote(path).lstrip('/')).resolve()
        base = (SITE).resolve()
        if not target.is_relative_to(base):
            return self.respond({'error':'Not found'},404)
        if target.is_dir(): target = target / 'index.html'
        if not target.is_file():return self.respond({'error':'Not found'},404)
        mime = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        stat = target.stat()
        etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        # Fonts, images and video change rarely: cache them for a week. Pages,
        # styles and scripts revalidate on every load and get a 304 when unchanged.
        static = path.startswith('/assets/') or mime.startswith(('image/','video/','font/'))
        cache = 'public, max-age=604800' if static else 'no-cache'
        if self.headers.get('If-None-Match') == etag:
            self.send_response(304)
            self.send_header('ETag',etag)
            self.send_header('Cache-Control',cache)
            self.end_headers();return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('ETag',etag)
        self.send_header('Cache-Control',cache)
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(data)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8780);args=parser.parse_args()
    os.chdir(BACKEND)
    service=StudioService(BACKEND)
    access=(AccessStore if os.environ.get('RELAY_STUDIO_PASSWORD_LOGIN')=='1' else ProfileStore)(service.storage)
    registry=ModelRegistry(service.storage)
    if isinstance(access,ProfileStore):
        manager=next((p for p in access.profiles()['profiles'] if p['role']=='manager'),None)
        if manager:registry.sync_workspace(manager,service.workspace())
    handler=partial(LocalWorkspace,service=service,access=access,registry=registry,analytics=AnalyticsService(BACKEND,service,registry,access),tasks=TaskService(service,registry),workers=WorkerStore(service.storage,access,registry,service))
    try:
        with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
            print(f'Relay website: http://127.0.0.1:{args.port}/',flush=True)
            print(f'Working Studio: http://127.0.0.1:{args.port}/projects/studio/',flush=True)
            server.serve_forever()
    finally:service.close()
