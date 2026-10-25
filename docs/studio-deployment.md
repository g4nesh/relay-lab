# Deploy a customer workspace

Relay Studio can run as one private installation for one customer. The web app,
identity database, model approvals, scoped agent credentials and workspace files
belong to that installation. Its SpacetimeDB database holds task state and
coordination messages. Give each customer a separate application instance,
persistent volume, database and database owner credential.

This is a deployment package, not an already hosted service. A domain, TLS edge
and an accessible SpacetimeDB instance are required. Do not scale the app to
multiple processes or replicas: its approval locks and running session cache
currently require **one process per workspace**.
The hosted app holds an exclusive lock on its state volume and refuses a second
process using that same volume. Separate replicas with separate volumes are not a
supported way to share one database.

## What runs where

| Component | Responsibility | Persistent state |
| --- | --- | --- |
| Caddy | Public HTTPS, certificate renewal, request size limit | TLS certificates |
| Studio / Uvicorn | Login, manager approvals, objective routing, scoped agent API | `/data/.relay-lab/` |
| Private SpacetimeDB database | Authoritative tasks, commands, reports and peer messages | Your database provider |
| Customer agent host | Local model inference and robot-specific validation/execution | Model files and robot runtime |

The small container deliberately installs no Torch, MuJoCo or MLX. Customer
agents run their own inference runtime and connect using scoped worker credentials.
The captured-feature preview requires repository artifacts and a separate model
runtime; it will correctly report unavailable in this image. Registering a model
does not download or execute arbitrary weights. Hosted model uploads are disabled;
keep weights on the agent host and register their model reference in Studio.

## Prepare the database

Use a dedicated database with this repository's private module, published under
the owner identity whose credential Studio will use. Tables remain private;
workers and browsers never receive the database owner token. See the
[SpacetimeDB setup guide](spacetime.md) for the module and local development setup.

The repository pins the TypeScript SDK and local CLI to 2.10.0. Use that CLI to
build `spacetimedb/`, then publish the generated JavaScript bundle to a compatible
server. Verify compatibility in a staging database before upgrading a server or
module. The deployment does not automatically change or clear an existing database.

```bash
# Install/build using the pinned project CLI described in docs/spacetime.md.
.relay-lab/spacetime/cli/bin/2.10.0/spacetimedb-cli build --module-path spacetimedb

# Existing owner token: a private, user-owned regular file with mode 0600.
# The helper reads the token from that file; it never appears in arguments or logs.
.venv/bin/python deploy/publish-spacetime.py \
  --server https://YOUR-SPACETIME-SERVER \
  --database YOUR-PRIVATE-DATABASE \
  --token-file /private/path/spacetime-owner-token \
  --bundle spacetimedb/dist/bundle.js
```

The helper uses the authenticated publish API with `clear=false`. It does not
create a cloud account or bypass your provider's access controls. Obtain the owner
credential through your provider's normal administration flow. Remote Spacetime
connections require HTTPS; plain HTTP is accepted only on loopback for development.

## Start the installation

Point a domain at a Docker host and allow inbound ports 80 and 443. Do not expose
the Studio container's port 8770 directly. The Compose file publishes only Caddy.

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env: your domain, Spacetime HTTPS origin, private database name.
# Do not put the owner token in this file.

docker compose --env-file deploy/.env -f deploy/compose.yaml build

# First installation only. Token travels over stdin and is stored mode 0600
# in the named volume, owned by the app's UID 10001. No credential is echoed.
docker compose --env-file deploy/.env -f deploy/compose.yaml \
  run --rm -T --no-deps studio python /opt/relay/init-state.py \
  < /private/path/spacetime-owner-token

docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
```

The token installer refuses to overwrite an existing credential. The image build
uses an allowlist; `.relay-lab`, `.git`, examples, datasets, model checkpoints and
local environment files are not sent to Docker. The container runs as UID 10001,
with a read-only root filesystem and a writable state volume. Neither the Docker
image nor the source repository contains your workspace state.

Open `https://YOUR-DOMAIN`. Create the first manager with the one-time setup code
stored at `/data/.relay-lab/studio/setup-code`. The hosting operator can retrieve
it over an administrator terminal:

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml \
  exec studio cat /data/.relay-lab/studio/setup-code
```

Treat the setup code as a credential and use it only on the first-manager form.
It is consumed after setup. Managers invite employees and approve exact model
revisions. Create a scoped agent credential from the dashboard, keep it on the
corresponding customer agent host, and use the task/agent API to receive objectives
and report progress. The database owner credential stays only in the control plane.

## Boundaries and operations

`RELAY_STUDIO_PUBLIC_ORIGIN` is the exact lowercase HTTPS origin, without a trailing
slash. Host and Origin must match it. Studio ignores `Forwarded` and
`X-Forwarded-*`; it never trusts those headers to choose a cookie policy or customer.
HTTPS is terminated by the supplied Caddy edge, which preserves the original Host.
Session cookies are `__Host-` scoped, Secure, HttpOnly and SameSite=Strict. Browser
mutations retain per-session CSRF validation. The local development server remains
loopback-only with its existing local cookie behavior.

The ASGI entry point is `python -m relay_lab.studio.asgi` (or `relay-studio` after
installing the `deploy` extra). It runs one Uvicorn process with eight bounded
application slots, 32 connection slots, a 128 KiB body limit, a 15-second body
deadline, a 120-second operation-response deadline, and five-second keepalive.
No inference threads or work queues are created without bounds. An expired
response deadline does **not** undo a database operation: its slot remains occupied
until completion. Inspect task state and use its operation identity before retrying;
never assume that a timeout means no work happened.

`GET /livez` checks the app; `GET /readyz` checks authenticated access to the private
Spacetime schema. Both enforce the configured Host and return only a readiness
boolean. Public health responses expose no tokens, paths or backend addresses.
The authenticated diagnostic API remains available to workspace members. Container
health uses `/readyz`; no fabricated in-memory success replaces an unavailable DB.

Keep the entire `studio_state` volume and the database together when backing up or
restoring. Stop the app for a consistent filesystem snapshot; retain SQLite sidecar
files if any. Never use `docker compose down -v` for an ordinary restart. Restore
ownership to UID/GID 10001 and private files to mode 0600. Restore the same registry
identity and approvals alongside the database; creating a blank volume intentionally
invalidates old approval snapshots. Review pending tasks after a crash before
resuming agents. Do not assume exactly-once physical motion from network delivery.

For credential rotation, stop the app, replace the token inside the volume using a
private administrator channel with the same ownership/mode, then restart and check
readiness. Revoke old agent credentials through Studio. Update the pinned server
and dependency versions deliberately, rerun tests and stage a private database
before rolling out. No public hosting, billing, organization isolation within a
shared process, horizontal scaling, or hardware safety certification is implied.

Implementation references: [Uvicorn resource limits](https://uvicorn.dev/settings/),
[Caddy proxy headers and TLS](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy),
[Spacetime private tables](https://spacetimedb.com/docs/tables/), and
[authenticated publication](https://spacetimedb.com/docs/http/database/).
