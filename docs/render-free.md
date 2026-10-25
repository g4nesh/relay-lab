# Render Free with Spacetime-only durable storage

Run the existing Docker image as one Render Free web service. Use a dedicated
Maincloud Free database. Qwen, VLA weights and robot/simulator runtimes remain on
agent hosts. No Render disk or second database provider is required.

## Publish and configure

The module and SDK are pinned to SpacetimeDB 2.13.0. Use the matching CLI:

```sh
npm --prefix spacetimedb ci
npm --prefix spacetimedb run check
spacetime build --module-path spacetimedb
spacetime login
spacetime publish YOUR-DATABASE --server maincloud --module-path spacetimedb --delete-data=never
```

In Render, connect this private repository, select Docker, the deployment branch,
and **Free ($0)**. Set health checks to `/readyz` and auto-deploy to Off until a
candidate has been verified. `render.yaml` describes the same settings.

Set these environment variables in Render's private service configuration:

| Variable | Value |
| --- | --- |
| `RELAY_STUDIO_STORAGE` | `spacetime` |
| `RELAY_SPACETIME_SERVER` | `https://maincloud.spacetimedb.com` |
| `RELAY_SPACETIME_DATABASE` | Your dedicated database name |
| `RELAY_SPACETIME_TOKEN` | The database owner's private credential |
| `RELAY_STUDIO_EXPERIMENTS` | `0` |

The app uses Render's `PORT` and `RENDER_EXTERNAL_URL` automatically. A custom
domain requires explicitly setting `RELAY_STUDIO_PUBLIC_ORIGIN` to its exact
HTTPS origin. The owner credential belongs only in backend secrets. It carries
the privileges of its Spacetime identity; it is not a worker or browser token.

## First manager

After `/readyz` succeeds, retrieve the one-time setup code from the private
Spacetime checkpoint on an administrator machine with the package installed:

```sh
python deploy/read-setup-code.py \
  --server https://maincloud.spacetimedb.com \
  --database YOUR-DATABASE \
  --token-file /private/path/spacetime-owner-token \
  --output /private/path/relay-first-manager-code
```

Open the hosted app, enter that code, and choose your own manager username and
password. After successful setup the code is removed from the next committed
checkpoint. Register/release models and create scoped worker credentials through
the dashboard. Existing local development accounts are not imported automatically.

## Persistence and failure behavior

`studio_state` is a private, versioned, compressed checkpoint of the accounts,
sessions, model registry, audit records, worker credentials/presence, workspace
configuration, telemetry and the coordination outbox. Existing task/report tables remain authoritative.
Local SQLite files are disposable execution caches, restored before each API
request. The HTTP response is withheld until a changed checkpoint receives a
confirmed Spacetime commit. Unchanged reads do not rewrite the checkpoint.

The separate `studio_lease` row serializes requests across overlapping instances.
Task and coordination mutations use fenced reducers that validate this same lease inside their
transaction. A stalled instance cannot commit state or task updates after losing
the lease. A crashed request can leave the workspace busy for up to three minutes;
no automatic physical command replay occurs. A timeout means an operation might
have committed: inspect the task and reuse its request identity.

One workspace and one intended replica remain the supported deployment. This is
a small-workspace compatibility adapter, not an unlimited or high-throughput
storage design. Checkpoints are capped at 8 MiB of decoded files and 1 MiB encoded;
exceeding a limit fails the request without acknowledging an unsaved change.
Polls and heartbeats consume Maincloud quota. Free Render instances sleep when
idle, so this deployment is not a real-time robot safety controller.

## Verify and recover

Check `/livez`, `/readyz`, first-manager setup, employee restrictions, exact model
release checks, worker task delivery and cancellation. Restart the Render service
and confirm account, model and worker identities and task state remain intact.
Never restore a blank checkpoint to repair a failed startup. Preserve the
Spacetime database and owner credential.

If a candidate fails readiness or persistence checks, suspend the new Render
service. Fix it before resuming. After enabling durable storage, old unfenced
application versions are intentionally rejected by task reducers: do not roll
back to the original disk-only app. Roll back only to a version supporting the
same checkpoint schema and fenced protocol, including the `coordination.sqlite3`
checkpoint entry. The unmodified `41ab077` image predates that entry and is not
a compatible rollback image. Keep the outbox in a rollback build even when its
background processor is disabled. Never use `--delete-data` as recovery.

Tests: `PYTHONPATH=src python -m pytest -q tests`. The live integration test requires
`RELAY_DURABLE_TEST_SERVER`, `RELAY_DURABLE_TEST_DATABASE` (a test name prefix), and
`RELAY_DURABLE_TEST_TOKEN_FILE`. It publishes a fresh isolated database under that
prefix and tests API/auth, empty-disk restarts, approvals, workers, task completion,
duplicate reports, cancellation, revocation, failed commits and stale writers.

### Public demo and private workspace

Hosted `/` opens a read-only sample dashboard immediately, without signup or
private API calls. All sample tasks, approvals, connections and analytics are
illustrative fixtures bundled in `demo.js`; they are not live robot results or
performance measurements. Demo exports retain this provenance. No demo writes,
worker credentials, SSE subscriptions or polling reach the real backend.

Use **Open your workspace** or `/?workspace=1` for the existing authenticated
workspace, including first-manager setup. Local profiles retain their normal
startup; `/?demo=1` previews the public experience locally. No authentication or
Spacetime persistence rules change. The public demo does not make the first
manager's setup code public or create an account.

Run frontend regression checks with `node --test tests/web/*.test.cjs`.
