# Relay Studio

Standalone private robot coordination dashboard, separated from Relay Lab's
simulation and training experiments. This folder is the entire deployment project:
clone and deploy this repository without the original research checkout.

```bash
git clone https://github.com/g4nesh/relay-studio.git
cd relay-studio
```

## What's included

- Dashboard, login, manager/employee roles, model registration and release approvals.
- Goals, agent assignments, scoped outbound runtime connections, reports and analytics.
- SpacetimeDB client and pinned private coordination module.
- Python worker SDK, Docker image, persistent storage, HTTPS edge and health probes.
- Live Spacetime subscriptions with authenticated dashboard/worker updates.
- Durable heartbeat and task-deadline alerts; hourly/daily UTC maintenance jobs.
- Scheduled aggregates of observed task events and measured exchange timings.
- Optional Spacetime checkpoints for durable accounts and approvals on ephemeral hosts.

Inference runs on your agent hosts. This app does not include MuJoCo, Torch, MLX,
Qwen, VLA weights, demonstrations or experiment recordings. Model registration
does not load weights. Customer handlers provide inference and execution; the
generic goal loop supplies structured peer reports, not automatic latent adapters.

```text
Customer browser → Studio dashboard + API → private SpacetimeDB
                             ↑                       ↑
                   scoped outbound workers ← task/report state
                             ↓
                   your models and simulator
```

## Local preview

Use Python 3.12+ and Node.js 22+ on macOS or Linux. A compatible SpacetimeDB service with this
module is required for task operations; the UI can open before it is available.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
npm --prefix subscriptions ci
npm --prefix subscriptions run build
RELAY_STUDIO_EXPERIMENTS=0 RELAY_STUDIO_ROOT=./state \
  RELAY_STUDIO_SUBSCRIPTION_BRIDGE="$PWD/subscriptions/src/bridge.cjs" \
  RELAY_SPACETIME_TOKEN_FILE=/private/path/spacetime-owner-token \
  .venv/bin/relay-studio
```

The local preview binds only to `http://127.0.0.1:8770`. Its local profile picker
is for development. Hosted mode requires the HTTPS origin and password accounts.
Do not expose the local preview as a public service. Keep `state/` outside source
control; the private files below `.relay-lab/` are also ignored.

Publish the updated module before running the new app. `/readyz` checks the new
coordination tables. Docker includes the subscription runtime automatically;
plain Python installations need the Node build above. When subscriptions are
unavailable, clients explicitly fall back to polling. See [coordination and
scheduling](docs/coordination.md) for behavior, bounds and validation commands.

## Deploy

For Render Free with Spacetime as the durable store, follow
[the Render guide](docs/render-free.md). Its deployment profile restores local
caches from a private checkpoint and fences writes with a database lease.

For a persistent-volume Docker host, follow [the deployment guide](docs/studio-deployment.md).
The included Compose
stack runs Studio plus Caddy; SpacetimeDB runs separately. You need:

1. A Docker host with a domain and ports 80/443, or a container service with a TLS edge.
2. **One replica** with a persistent writable `/data` volume, owned by UID/GID 10001.
3. A dedicated compatible SpacetimeDB database and its private owner token file.

For a managed container service, build this folder's Dockerfile, route HTTPS to
port 8770, preserve the public Host header, set `RELAY_STUDIO_PUBLIC_ORIGIN` and
the Spacetime environment variables from `deploy/compose.yaml`, mount `/data`, and
install the owner token there using `deploy/init-state.py`. Set health probes to
`/livez` and `/readyz` with the configured Host. Use one process and one replica;
ephemeral hosts must use `RELAY_STUDIO_STORAGE=spacetime` as described in the
Render guide. One workspace and one intended replica remain supported in either mode.

The first manager setup code is created in `/data/.relay-lab/studio/setup-code`.
Give customers scoped runtime credentials, never the database owner token. See
[the agent integration guide](docs/agents.md) to connect your own model or simulator.

## Layout and updates

| Path | Purpose |
| --- | --- |
| `src/relay_lab/studio/web/` | Customer interface; no frontend build step |
| `src/relay_lab/studio/` | Auth, approvals, goals, analytics, HTTP server, worker SDK |
| `src/relay_lab/factory/` | Shared coordination protocol utilities; no simulator |
| `src/relay_lab/spacetime.py` | Private SpacetimeDB HTTP client |
| `spacetimedb/` | Pinned schema/reducers; build and publish separately |
| `subscriptions/` | Trusted Node SDK bridge and generated private-table bindings |
| `tests/` | Standalone auth, task, worker, streaming and scheduling integration checks |
| `deploy/` | Compose, TLS proxy and private credential setup |
| `render.yaml` | Optional Render deployment blueprint with Spacetime-backed storage |
| `export-manifest.json` | File inventory and hashes of this source snapshot |

The dashboard's shared design, logo and bundled IBM Plex fonts come from
Relay Lab's shared dashboard source.
They live inside `src/relay_lab/studio/web/`, so standalone builds retain the same
appearance without a separate website server. `coordination-design.css` styles
the new scheduling controls; `workspace-design.css`, `hosted.css` and `local.css`
preserve the upstream design. The fonts' license is included alongside them.

The module retains the existing coordination schema for compatibility. It does
not ship a physics runtime or replay assets. The current application supports a
two-model workspace and bounded task history; it is not a fleet-scale scheduler.

Relay Lab is the canonical home for shared dashboard and coordination changes;
this repository is its independently deployable control-plane export. The
explicit export boundary excludes simulation, training and research data.
Follow [the synchronization guide](docs/studio-sync.md) when changing shared
code or importing updates. The research checkout is not needed to build or run
this app. The export manifest records each source path and exact file hash.
Keep coordinated backups of `/data` and the private database for disk mode;
preserve the private checkpoint and task tables together in Spacetime storage mode.
Upgrades do not
automatically publish or reset the database. Roll back to the previous image with
the same compatible state and fenced protocol; never remove customer state as a
restart procedure. Read the storage-specific recovery guide before a rollback.
