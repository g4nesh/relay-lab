# Website and local Studio

The complete Relay website lives in `website/`. The Python backend remains in
`src/relay_lab`, with the existing SpacetimeDB module in `spacetimedb/`.

## Run locally

Install backend dependencies and start/publish SpacetimeDB using the root README.
Then run from the repository root:

```sh
PYTHONPATH=src .venv/bin/python scripts/serve_workspace.py
```

Open http://127.0.0.1:8780/ for the website, or
http://127.0.0.1:8780/projects/studio/ for the working Studio. Use `--port` to
choose another port. The server binds to loopback only. Homepage links default
to port 8780.

The integrated server serves the real Studio frontend and authenticated API
at the same origin as the website. It retains local profiles, model approvals,
task services, workers, request guards, and analytics. Cloth workcell and
learned-adapter views open inside the Studio navigation shell. Recorded videos
and traces are bundled under `website/projects/cloth/`.

UI overrides are in `website/projects/studio/local.css`, `local-tabs.js`, and
`embedded.css`. Workcell and recorded analytics share
`website/projects/shared/workspace-white.css`. Source data and uncertainty
intervals remain available. Estimates are labeled separately from measurements.

Static hosting alone does not run the Python backend. The static Studio page
is an inspection view. Live Studio needs the local server or the deployment in
`docs/studio-deployment.md`.

## Runtime files

Virtual environments, model download caches, account databases, session state,
and SpacetimeDB credentials are not included. Set these up locally; never copy
another machine's `.relay-lab/` directory. The bundled `backend-source.zip` is a
historical downloadable snapshot; the current backend is the repository root.
