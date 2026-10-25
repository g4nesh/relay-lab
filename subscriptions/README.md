# Trusted Spacetime subscription bridge

This Node 22+ process uses the SpacetimeDB 2.13 SDK and generated private-table
bindings. It maintains one backend WebSocket subscription. It is not an HTTP
server, a polling loop, or a browser SDK client.

```sh
npm ci
npm run check
npm test
npm run build
```

The Python ASGI application starts `src/bridge.cjs` and supplies its existing
validated database configuration through stdin. Do not invoke the script with a
token argument. Configure an installed Python package using:

```sh
RELAY_STUDIO_SUBSCRIPTION_BRIDGE=/app/subscriptions/src/bridge.cjs
```

Keep `src/*.cjs`, `dist/module_bindings/`, and production `node_modules/` together
in the runtime image. The Dockerfile builds and includes them. If Node, the
bindings, or the database subscription is unavailable, the dashboard reports a
degraded live connection and clients continue their authenticated polling.
Subscription failure is not disguised as healthy live delivery.

The generated bindings must match the deployed Spacetime module. After editing
its schema, regenerate using the matching CLI, then rebuild:

```sh
spacetime generate --module-path ../spacetimedb --lang typescript \
  --out-dir src/module_bindings --include-private -y
npm run build
```

`/api/events` authenticates a dashboard session cookie. `/api/agent/events`
authenticates a scoped worker bearer credential, including its current approved
model release and issuer authority. SSE clients receive:

- `status`: `{ "state": "starting|ready|degraded|closed" }`.
- `sync`: `{ "topics": ["tasks", "workers", "coordination"], "sequence": 1 }`.
- `update`: the same structure with the affected topic subset.

These are invalidations only. They contain no task text, row identifiers,
embeddings, robot commands, or credentials. Clients always re-fetch current
state through their existing authorized APIs. Each new connection receives a
sync; there is no event replay promise or exactly-once delivery guarantee.

One pending invalidation per connection coalesces bursts. Sixteen concurrent
streams are allowed per application process; two separate threads perform brief
identity checks. Connections reauthorize before each notification and every ten
seconds while idle. Slow consumers have a five-second write deadline. Streams
do not consume the normal HTTP executor while waiting. An SDK reconnect sends a
new sync, and the Python supervisor uses bounded reconnect backoff.

The SDK caches subscribed rows server-side, including current embeddings. The
single-customer database's retention policies must bound those tables. Do not
publish the owner credential or change private tables to public for this bridge.
