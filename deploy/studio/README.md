# Relay Studio

A private workspace for coordinating robot agents: register models, approve
releases, assign shared goals, and follow what each agent reports. SpacetimeDB
keeps task state synchronized, delivers live updates, and runs scheduled checks
for disconnected workers and overdue tasks.

Relay Studio is the deployable dashboard from
[Relay Lab](https://github.com/g4nesh/relay-lab). It runs independently of the
research repository, with no simulation, training, model weights or datasets.

## From goal to execution

1. Register model references and have a manager approve their exact configurations.
2. Connect each agent host using a scoped worker credential.
3. Assign an objective and success criteria to the team.
4. Follow progress, peer reports, alerts and measured activity in the dashboard.

Your agent handler supplies observations, model inference and execution. Registering
weights does not load a model, and the generic worker loop exchanges structured
reports rather than automatically running Qwen or latent adapters. A completed
task means its assigned agents reported success; it is not an independent
verification of physical work. See [Connect your own agent](docs/agents.md).

## Demo and private workspace

Visit `/?demo=1` for the read-only sample dashboard. Its tasks, approvals and
analytics are labeled fixtures, not live robot results or measured performance.
The demo makes no private API calls or changes to customer state. The standalone
website deployment serves the product site at `/` and the dashboard at
`/dashboard`; existing query links redirect there. See the
[website deployment guide](https://github.com/g4nesh/relay-studio/blob/main/docs/website.md).

Choose **Open your workspace**, or visit `/?workspace=1`, to sign in or complete
first-manager setup. Local development keeps the profile picker. The private
workspace requires authentication when hosted.

## Run locally

Use Python 3.12+ and Node.js 22+ on macOS or Linux. Provision a private SpacetimeDB
database with this repository's module before using task operations. The SDK and
module are pinned to **2.13.0**; follow
[Prepare the database](docs/studio-deployment.md#prepare-the-database) to build and publish it.

```bash
git clone https://github.com/g4nesh/relay-studio.git
cd relay-studio
python3 -m venv .venv
.venv/bin/python -m pip install .
npm --prefix subscriptions ci
npm --prefix subscriptions run build
chmod 600 /private/path/spacetime-owner-token

RELAY_STUDIO_ROOT=./state RELAY_STUDIO_EXPERIMENTS=0 \
  RELAY_SPACETIME_SERVER=https://maincloud.spacetimedb.com \
  RELAY_SPACETIME_DATABASE=YOUR-DATABASE \
  RELAY_SPACETIME_TOKEN_FILE=/private/path/spacetime-owner-token \
  RELAY_STUDIO_SUBSCRIPTION_BRIDGE="$PWD/subscriptions/src/bridge.cjs" \
  .venv/bin/relay-studio
```

Open `http://127.0.0.1:8770/?workspace=1`. The local profile picker is for development; use a
hosted deployment for customer access. Keep `state/` and the database owner token
outside source control. Give agent hosts scoped worker tokens, never the owner credential.

## Deploy

Choose the storage profile that matches your host:

| Deployment | Durable state | Setup |
| --- | --- | --- |
| Render Free | Private Spacetime checkpoints and task tables; local caches are disposable | [Render guide](docs/render-free.md), including `render.yaml` |
| Docker / Compose | Persistent `/data` volume plus a private Spacetime database | [Deployment guide](docs/studio-deployment.md) |

Both profiles support **one private workspace per customer and one intended
replica**. Hosted access uses HTTPS and password accounts. Docker includes the
Node subscription bridge; direct Python installations need the Node build above.
Subscriptions fall back to polling when unavailable. See
[Coordination and scheduling](docs/coordination.md) for timing and capacity limits.

Use `/livez` and `/readyz` for health checks. Follow the selected guide to obtain
the first-manager setup code, verify persistence after restart, and back up the
workspace. Rollbacks must preserve compatible state and, in Spacetime storage
mode, the fenced write protocol. A code deployment does not publish the database module.

## Verify a change

```bash
.venv/bin/python -m pip install '.[dev]'
.venv/bin/python -m pytest tests -q
node --test tests/web/*.test.cjs
npm --prefix subscriptions run check
npm --prefix subscriptions test
npm --prefix spacetimedb ci
npm --prefix spacetimedb run check
npm --prefix spacetimedb test
```

Live database tests are opt-in and need a disposable test database; see the
coordination and Render guides. Unit checks alone do not establish deployment
health or robot task success.

## Keep the repositories aligned

Relay Lab owns the shared dashboard and coordination source; this repository is
its experiment-free deployment export. [The synchronization guide](docs/studio-sync.md)
describes the explicit allowlist, standalone packaging templates and read-only
drift check. `export-manifest.json` records the source paths and exact file hashes.
Customer accounts, tokens, task data and robot journals are never synchronized
between repositories.
