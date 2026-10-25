# Live coordination and scheduled maintenance

Relay Studio uses SpacetimeDB 2.13 native subscriptions and schedule tables.
No external cron service, Redis queue, or experimental
Spacetime submodule package is required. Calendar schedules currently support
**hourly** (top of the hour UTC) and **daily** (00:00 UTC). Arbitrary cron strings
and per-customer time zones are not supported in this version.

## Live delivery

One trusted Node process subscribes to private Spacetime task, message, presence,
attention, job and rollup tables. The ASGI app forwards bounded invalidations
through `/api/events` (browser session) and `/api/agent/events` (worker token).
Owner credentials travel only over a private process pipe. An event never carries
task text, embeddings, commands, row identifiers or credentials.

Clients reload their authorized snapshot on initial connection, reconnect, or
change. Notifications are hints, not permission to move a robot. Workers keep
their existing claim checks, release checks, action journal and cancellation
checks. Every connection reauthorizes before notifications and at least every
ten seconds while idle. Revoking a worker or its model release closes its stream.

The Tasks page shows whether delivery is live or using polling. Workers retain a
30-second fallback inbox check; `--interval` can override it. Their independent
10-second heartbeat remains active. Browser fallback refreshes every 30 seconds,
with a 60-second reconciliation refresh while live. Slow clients have bounded
queues and streams do not occupy ordinary request slots while waiting.

The bridge currently subscribes within one private customer database and sends
payload-free wakeups to its authorized clients. Per-worker message delivery and
goal contents remain scoped by the existing HTTP API. Large robot fleets and
cross-customer shared databases are outside this deployment profile.

## Deadlines and presence

Successful authenticated heartbeats renew a native 45-second presence lease.
Each renewal replaces the timer and advances its generation. A stale callback
cannot expire a newer lease. The scheduler marks an expired runtime `stale` and
creates one advisory attention record. Revocation leaves a tombstone, so a late
heartbeat cannot resurrect that runtime.

New tasks default to a 30-minute attention deadline; callers can choose 60–86400
seconds. The UI offers five presets. The deadline is measured from task creation
and does not reset on each report. Older tasks without a deadline remain
unchanged. New revisions supersede old timers; cancellation or terminal outcomes
resolve the notice. The scheduled callback also checks the authoritative task
journal, covering a terminal update whose metadata mirror has not caught up.

These schedules run in Spacetime while Studio is stopped, provided Spacetime is
running. They do not provide hard real-time timing or an emergency stop. A
deadline never retries, reassigns or resumes a physical action.

## Recurring jobs

On first connection, the app creates only missing jobs. A manager's paused or
modified configuration survives application restarts. Employees can inspect
jobs; only managers can change them or run one manually.

| Job | Default | Work |
| --- | --- | --- |
| Health checks | Hourly UTC | Reconcile expired leases and overdue task notices |
| Metrics rollup | Hourly UTC | Aggregate deduplicated numeric samples into hourly count/sum/min/max buckets |
| Retention cleanup | Daily UTC | Remove old aggregated samples, old rollups, resolved notices and terminal deadline projections |

Retention defaults to seven days and accepts 1–90 days through the manager API.
Changing a schedule without specifying retention preserves its existing value.
Cleanup retains authoritative task history and revocation tombstones. Jobs expose
their next/last execution, execution count, processed count and controlled errors.
Native reducer exceptions roll back and remain in Spacetime logs; this is not a
general-purpose retry engine for arbitrary failed code.

Rollup and cleanup mutate at most 256 records per transaction. A backlog schedules
a continuation one second later. A delayed calendar execution schedules the next
future boundary instead of replaying missed hours in a burst. Disabled jobs can
be run once manually without turning recurrence back on.

## Measurements and synchronization

Task state remains authoritative in its existing Spacetime journal. Deadline
projections and task-event measurements are queued in a local durable SQLite
outbox after confirmed task updates. Startup reconciliation repairs task metadata
and mirrors locally revoked worker IDs after a crash between commit and enqueue.
Worker history is read in bounded pages without credential fields; failed enqueues
remain eligible for reconciliation. Optional metadata failure never changes
the result of an already committed task or retries an action.

The outbox compares revisions atomically, reserves space for control metadata,
and only acknowledges the exact payload sent. Pending updates and omitted
measurements are visible in the coordination panel. Heartbeat presence is written
directly and read back before a heartbeat is acknowledged; database failure
therefore prevents the worker from continuing on an unconfirmed heartbeat.

Only observed task events and valid measured exchange values are recorded.
Recovered exchange responses do not generate invented inference latency.
Rollup buckets use **database ingestion time**, including delayed outbox delivery,
and are not a historical productivity benchmark. Samples deduplicate within the
configured database retention window. The dashboard shows sample counts alongside
averages; no factory-efficiency or backend speed claim is inferred from them.

Per-workspace module limits are 256 runtime identities (including revocations),
512 task watches, 1,024 attention records, and 8,192 samples/rollups each. The
existing dashboard task-history limit is still 100. Capacity errors are surfaced;
retention does not silently discard unaggregated samples. Local metric outbox
insertion stops at 10,000 total rows, reserving capacity up to 12,000 for control
metadata. Expired acknowledged local metric records are pruned after seven days.

## API

All browser routes require an authenticated workspace session. POST requests also
require the existing CSRF header. Worker credentials cannot administer jobs.

```text
GET  /api/coordination
POST /api/coordination/jobs
     {"job_id":"cleanup","schedule":"daily","enabled":true,"retention_days":7}
POST /api/coordination/jobs/run
     {"job_id":"rollup"}
```

Use the Tasks panel for schedule changes. A manual run with an uncertain HTTP
outcome must be inspected before retrying; the app does not automatically retry
manual runs. All privileged job changes are recorded in the workspace audit log.

## Build and validate

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[dev]'
npm --prefix spacetimedb ci
npm --prefix spacetimedb run check
npm --prefix spacetimedb test
npm --prefix subscriptions ci
npm --prefix subscriptions run build
npm --prefix subscriptions test
PYTHONPATH=src .venv/bin/python -m pytest tests -q
docker build -t relay-studio:coordination .
```

The live tests require an explicitly chosen disposable database with this module
already published. They create records and synthetic test samples there. They do
not publish, clear data, or invoke a model or robot. Do not target customer data.

```bash
RELAY_COORDINATION_TEST_DATABASE=your-disposable-database \
RELAY_COORDINATION_TEST_TOKEN_FILE=/private/path/owner-token \
PYTHONPATH=src .venv/bin/python -m pytest tests/test_coordination_live.py -q
```

Checks cover native expiry without Studio, renewal and revocation, overdue and
cancelled tasks, bounded rollup continuation, real SDK delivery and reconnect.
Unit and transport tests cover permissions, duplicate operations, metadata
restart/reconciliation, stream backpressure and shutdown. No test claims a robot
task completed physically. See the [bridge guide](../subscriptions/README.md)
for rebuilding generated bindings after schema changes.

## Upgrade and rollback

Back up the existing customer state and database together. Build the pinned module,
publish it with `clear=false`, then build/start the new app image. The additions
are private tables and reducers; existing task tables and receipts remain intact.
`/readyz` checks the added schema. Verify authenticated live delivery and job status
before rolling out beyond staging.

To roll back the app, first pause its recurring jobs if they should no longer run.
Restore a compatible prior app image with the same state volume, or the same
Spacetime checkpoint in Render mode (see the compatibility notes in
[Render recovery](render-free.md#verify-and-recover)). Leave additive database
tables in place; do not clear the database, delete customer state, or attempt to
drop new tables as an ordinary rollback. In Spacetime checkpoint mode, a rollback
app must support the checkpoint schema and fenced write protocol; older unfenced
writers are rejected. Native schedules remain separately manageable.

## Render Free persistence

In `RELAY_STUDIO_STORAGE=spacetime` mode, the coordination outbox is included in
the private Spacetime checkpoint. Local SQLite files are disposable caches.
Background flushes and stream reauthorization acquire the same bounded database
lease as normal requests; a live stream releases that lease between checks.
Coordination writes use fenced reducers, so an expired instance cannot change
jobs or presence. Native scheduled reducers run independently of this lease.
See [Render deployment](render-free.md). The Node bridge uses the matching 2.13 SDK.
