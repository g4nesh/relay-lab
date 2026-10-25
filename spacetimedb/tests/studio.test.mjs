import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

// Exercise the real reducer functions with a deterministic in-memory DB adapter.
// This checks state transitions; live Spacetime tests additionally verify native
// timer delivery, authorization and persistence (the adapter is not that evidence).
const moduleURL = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const stub = moduleURL(`
  const builder = new Proxy({}, { get: () => () => builder });
  export const t = new Proxy({}, { get: () => () => builder });
  export const table = (options, columns) => ({ options, columns, rowType: {} });
  export const schema = () => ({ reducer: (...args) => args.at(-1), init: fn => fn });
  export class SenderError extends Error {}
  export const ScheduleAt = { time: value => ({ tag:'Time', value }), interval: value => ({ tag:'Interval', value }) };
`);
async function compile(name, dependencies = {}) {
  let source = ts.transpileModule(await readFile(new URL(`../src/${name}.ts`, import.meta.url), 'utf8'),
    { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
  for (const [name, url] of Object.entries({ spacetimedb: stub, 'spacetimedb/server': stub, ...dependencies }))
    source = source.replaceAll(`from '${name}'`, `from '${url}'`).replaceAll(`from "${name}"`, `from '${url}'`);
  return moduleURL(source);
}
const policyURL = await compile('studio_policy');
const policy = await import(policyURL);
const tablesURL = await compile('studio_tables');
const mod = await import(await compile('index', { './studio_policy': policyURL, './studio_tables': tablesURL }));
const identity = name => ({ name, isEqual(other) { return this.name === other.name; } });
const owner = identity('owner'), database = identity('database');
function harness() {
  const tables = new Map();
  const table = name => {
    if (tables.has(name)) return tables.get(name);
    const rows = new Map(); let next = 1n;
    const primary = name === 'module_owner' ? 'id' : name === 'factory_run' ? 'run_id'
      : name === 'studio_workspace' ? 'workspace_id' : name.includes('watchdog') || name.includes('trigger') ? 'scheduledId' : 'key';
    const value = new Proxy({ rows, count: () => BigInt(rows.size),
      insert(row) { row = structuredClone(row); if (primary === 'scheduledId' && !row.scheduledId) row.scheduledId = next++;
        assert(!rows.has(row[primary]), `duplicate ${name}`); rows.set(row[primary], row); return structuredClone(row); } }, {
      get(target, field) {
        if (field in target) return target[field];
        return { find: key => structuredClone(rows.get(key) ?? null), delete: key => rows.delete(key),
          update: row => { assert(rows.has(row[primary]), `missing ${name}`); rows.set(row[primary], structuredClone(row)); },
          filter: key => [...rows.values()].filter(row => row[field] === key).map(row => structuredClone(row)) };
      }
    });
    tables.set(name, value); return value;
  };
  const db = new Proxy({}, { get: (_target, name) => table(name) });
  const ctx = { db, sender: owner, identity: database, timestamp: { microsSinceUnixEpoch: 10n * policy.DAY_US } };
  const originalGet = table;
  // Special owner view avoids attempting structuredClone on its method.
  ctx.db = new Proxy({}, { get: (_target, name) => name === 'module_owner'
    ? { id: { find: () => ({ id: 0, identity: owner }) } } : originalGet(name) });
  return { ctx, rows: name => [...table(name).rows.values()],
    advance: seconds => { ctx.timestamp = { microsSinceUnixEpoch: ctx.timestamp.microsSinceUnixEpoch + BigInt(seconds) * 1_000_000n }; },
    scheduled(fn, arg) {
      ctx.sender = database;
      try {
        mod[fn](ctx, { arg });
        // Native one-shot reducer schedules are removed after callback execution.
        table(fn === 'studio_execute_job' ? 'studio_job_trigger' : 'studio_watchdog').rows.delete(arg.scheduledId);
      } finally { ctx.sender = owner; }
    },
  };
}
const heartbeat = { workspace_id: 'acme', worker_id: 'worker1', agent_id: 'arm1', lease_seconds: 5 };
const task = { workspace_id: 'acme', task_id: 'task1', revision: 0, status: 'running', timeout_seconds: 10 };
const job = { workspace_id: 'acme', job_id: 'rollup', enabled: true, schedule: 'hourly', retention_days: 7 };

test('calendar boundaries are UTC and skip missed ticks; task deadlines do not drift', () => {
  assert.equal(policy.nextStudioBoundary(2n * policy.HOUR_US, 'hourly'), 3n * policy.HOUR_US);
  assert.equal(policy.nextStudioBoundary(2n * policy.DAY_US + 15n, 'daily'), 3n * policy.DAY_US);
  assert.throws(() => policy.nextStudioBoundary(0n, '* * * * *'));
  assert.equal(policy.studioTaskDeadline(500n, 5, { timeout_seconds: 5, deadline_us: 20n }), 20n);
  assert.equal(policy.studioTaskDeadline(500n, 0), 0n);
});

test('renewed heartbeat cancels the old timer and stale generations cannot expire it', () => {
  const h = harness(); mod.studio_heartbeat(h.ctx, heartbeat);
  const old = h.rows('studio_watchdog')[0]; h.advance(3); mod.studio_heartbeat(h.ctx, heartbeat);
  assert.equal(h.rows('studio_watchdog').length, 1);
  h.advance(3); h.scheduled('studio_expire_watchdog', old);
  assert.equal(h.rows('studio_presence')[0].status, 'online');
  h.advance(3); h.scheduled('studio_expire_watchdog', h.rows('studio_watchdog')[0]);
  assert.equal(h.rows('studio_presence')[0].status, 'stale');
  assert.equal(h.rows('studio_attention').length, 1);
  mod.studio_heartbeat(h.ctx, heartbeat);
  assert.equal(h.rows('studio_attention')[0].status, 'resolved');
});

test('revocation before initial heartbeat prevents delayed registration and direct scheduler calls fail', () => {
  const h = harness(); mod.studio_revoke_worker(h.ctx, heartbeat);
  assert.throws(() => mod.studio_heartbeat(h.ctx, heartbeat));
  assert.equal(h.rows('studio_presence')[0].status, 'revoked');
  assert.throws(() => mod.studio_expire_watchdog(h.ctx, { arg: {} }), /database scheduler/);
  h.ctx.sender = identity('outsider');
  assert.throws(() => mod.studio_configure_job(h.ctx, job), /module owner/);
});

test('task revisions preserve deadlines, ignore stale updates, and cancel outstanding timers', () => {
  const h = harness(); mod.studio_sync_task(h.ctx, task);
  const deadline = h.rows('studio_task_watch')[0].deadline_us;
  const old = h.rows('studio_watchdog')[0];
  h.advance(5); mod.studio_sync_task(h.ctx, { ...task, revision: 1 });
  assert.equal(h.rows('studio_task_watch')[0].deadline_us, deadline);
  mod.studio_sync_task(h.ctx, { ...task, status: 'cancelled' });
  assert.equal(h.rows('studio_task_watch')[0].status, 'running');
  h.advance(6); h.scheduled('studio_expire_watchdog', old);
  assert.equal(h.rows('studio_attention').length, 0);
  h.scheduled('studio_expire_watchdog', h.rows('studio_watchdog')[0]);
  assert.equal(h.rows('studio_attention')[0].status, 'open');
  mod.studio_sync_task(h.ctx, { ...task, revision: 2, status: 'cancelled' });
  assert.equal(h.rows('studio_watchdog').length, 0);
  assert.equal(h.rows('studio_attention')[0].status, 'resolved');
  assert.throws(() => mod.studio_sync_task(h.ctx, { ...task, revision: 3 }), /cannot reopen/);
});

test('native deadline reads authoritative terminal state even when projection is behind', () => {
  const h = harness();
  h.ctx.db.factory_run.insert({ run_id: task.task_id, state_json: JSON.stringify({ task: 'studio-objective-v1',
    registry_id: 'acme', goal: { status: 'running', created_at: Number(h.ctx.timestamp.microsSinceUnixEpoch) / 1e6 } }) });
  mod.studio_sync_task(h.ctx, task);
  h.ctx.db.factory_run.run_id.update({ run_id: task.task_id, state_json: JSON.stringify({ task: 'studio-objective-v1',
    registry_id: 'acme', goal: { status: 'cancelled' } }) });
  h.advance(11); h.scheduled('studio_expire_watchdog', h.rows('studio_watchdog')[0]);
  assert.equal(h.rows('studio_task_watch')[0].status, 'cancelled');
  assert.equal(h.rows('studio_attention').length, 0);
});

test('metric retries and repeated rollups cannot double count; workspace samples remain isolated', () => {
  const h = harness(); mod.studio_configure_job(h.ctx, job);
  const metric = { workspace_id: 'acme', event_id: 'event1', metric: 'exchange_latency_ms', value: 5 };
  mod.studio_record_metric(h.ctx, metric); mod.studio_record_metric(h.ctx, metric);
  mod.studio_record_metric(h.ctx, { ...metric, event_id: 'event2', value: 15 });
  mod.studio_record_metric(h.ctx, { ...metric, workspace_id: 'other', value: 99 });
  assert.throws(() => mod.studio_record_metric(h.ctx, { ...metric, value: 6 }), /Conflicting/);
  mod.studio_run_job(h.ctx, job); mod.studio_run_job(h.ctx, job);
  const row = h.rows('studio_rollup')[0];
  assert.equal(row.sample_count, 2n); assert.equal(row.total, 20);
  assert.equal(row.minimum, 5); assert.equal(row.maximum, 15);
  assert.equal(h.rows('studio_rollup').length, 1);
  assert.equal(h.rows('studio_metric').find(row => row.workspace_id === 'other').aggregated, false);
});

test('job configuration is idempotent, old trigger is inert, disabling remains disabled after manual run', () => {
  const h = harness(); mod.studio_configure_job(h.ctx, job);
  const old = h.rows('studio_job_trigger')[0];
  mod.studio_configure_job(h.ctx, job); assert.equal(h.rows('studio_job_trigger')[0].scheduledId, old.scheduledId);
  mod.studio_configure_job(h.ctx, { ...job, enabled: false });
  h.scheduled('studio_execute_job', old);
  assert.equal(h.rows('studio_job')[0].run_count, 0n);
  mod.studio_run_job(h.ctx, job);
  assert.equal(h.rows('studio_job')[0].next_run_us, 0n);
  assert.equal(h.rows('studio_job_trigger').length, 0);
});

test('cleanup removes expired aggregated samples but retains unaggregated data and revocations', () => {
  const h = harness(); mod.studio_configure_job(h.ctx, { ...job, enabled: false });
  mod.studio_record_metric(h.ctx, { workspace_id: 'acme', event_id: 'old', metric: 'exchange_bytes', value: 10 });
  mod.studio_run_job(h.ctx, job);
  mod.studio_record_metric(h.ctx, { workspace_id: 'acme', event_id: 'pending', metric: 'exchange_bytes', value: 20 });
  mod.studio_revoke_worker(h.ctx, heartbeat);
  h.advance(8 * 86400);
  mod.studio_configure_job(h.ctx, { ...job, job_id: 'cleanup', enabled: false });
  mod.studio_run_job(h.ctx, { ...job, job_id: 'cleanup' });
  assert.deepEqual(h.rows('studio_metric').map(row => row.event_id), ['pending']);
  assert.equal(h.rows('studio_rollup').length, 0);
  assert.equal(h.rows('studio_presence')[0].status, 'revoked');
});

test('workspace quotas reject more telemetry without overwriting dedupe evidence', () => {
  const h = harness();
  for (let index = 0; index < policy.STUDIO_LIMITS.metrics; index++)
    h.ctx.db.studio_metric.insert({ key: `acme:old${index}`, workspace_id: 'acme', event_id: `old${index}`,
      metric: 'exchange_bytes', value: 1, occurred_us: 0n, aggregated: true });
  assert.throws(() => mod.studio_record_metric(h.ctx, {
    workspace_id: 'acme', event_id: 'new', metric: 'exchange_bytes', value: 2 }), /capacity reached/);
  assert.equal(h.rows('studio_metric').length, policy.STUDIO_LIMITS.metrics);
});

test('large rollup batches continue durably without recounting samples', () => {
  const h = harness(); mod.studio_configure_job(h.ctx, job);
  for (let index = 0; index < policy.STUDIO_LIMITS.batch + 1; index++)
    h.ctx.db.studio_metric.insert({ key: `acme:event${index}`, workspace_id: 'acme', event_id: `event${index}`,
      metric: 'exchange_bytes', value: 2, occurred_us: h.ctx.timestamp.microsSinceUnixEpoch, aggregated: false });
  mod.studio_run_job(h.ctx, job);
  assert.equal(h.rows('studio_rollup')[0].sample_count, BigInt(policy.STUDIO_LIMITS.batch));
  assert.equal(h.rows('studio_job')[0].next_run_us, h.ctx.timestamp.microsSinceUnixEpoch + 1_000_000n);
  const trigger = h.rows('studio_job_trigger')[0]; h.advance(1);
  h.scheduled('studio_execute_job', trigger);
  assert.equal(h.rows('studio_rollup')[0].sample_count, BigInt(policy.STUDIO_LIMITS.batch + 1));
  assert.equal(h.rows('studio_job')[0].last_processed, 1);
  assert.equal(h.rows('studio_job_trigger').length, 1);
  assert.equal(h.rows('studio_job')[0].next_run_us, policy.nextStudioBoundary(h.ctx.timestamp.microsSinceUnixEpoch, 'hourly'));
});
