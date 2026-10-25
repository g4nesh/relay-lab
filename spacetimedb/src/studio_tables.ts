import { table, t } from 'spacetimedb/server';

// These are server-only projections. Neither owner credentials nor robot goals
// are published to browsers through these tables.
export const studio_workspace = table({ public: false }, {
  workspace_id: t.string().primaryKey(), created_at: t.timestamp(),
});
export const studio_presence = table({ public: false }, {
  key: t.string().primaryKey(), workspace_id: t.string().index('btree'),
  worker_id: t.string(), agent_id: t.string(), generation: t.u64(),
  lease_expires_us: t.u64(), status: t.string(), watchdog_id: t.u64(), updated_at: t.timestamp(),
});
export const studio_task_watch = table({ public: false }, {
  key: t.string().primaryKey(), workspace_id: t.string().index('btree'),
  task_id: t.string(), revision: t.u32(), status: t.string(), timeout_seconds: t.u32(),
  deadline_us: t.u64(), watchdog_id: t.u64(), updated_at: t.timestamp(),
});
export const studio_watchdog = table({ public: false }, {
  scheduledId: t.u64().primaryKey().autoInc(), scheduledAt: t.scheduleAt(),
  workspace_id: t.string(), kind: t.string(), subject_id: t.string(), generation: t.u64(),
});
export const studio_attention = table({ public: false }, {
  key: t.string().primaryKey(), workspace_id: t.string().index('btree'),
  kind: t.string(), subject_id: t.string(), generation: t.u64(), message: t.string(),
  status: t.string(), created_at: t.timestamp(), updated_at: t.timestamp(),
});
export const studio_job = table({ public: false }, {
  key: t.string().primaryKey(), workspace_id: t.string().index('btree'),
  job_id: t.string(), enabled: t.bool(), schedule: t.string(), retention_days: t.u32(),
  next_run_us: t.u64(), last_run_us: t.u64(), run_count: t.u64(),
  last_error: t.string(), scheduled_id: t.u64(), last_processed: t.u32(),
});
export const studio_job_trigger = table({ public: false }, {
  scheduledId: t.u64().primaryKey().autoInc(), scheduledAt: t.scheduleAt(), job_key: t.string(),
});
export const studio_metric = table({ public: false }, {
  key: t.string().primaryKey(), workspace_id: t.string().index('btree'),
  event_id: t.string(), metric: t.string(), value: t.f64(),
  occurred_us: t.u64(), aggregated: t.bool(),
});
export const studio_rollup = table({ public: false }, {
  key: t.string().primaryKey(), workspace_id: t.string().index('btree'),
  metric: t.string(), bucket_us: t.u64(), sample_count: t.u64(),
  total: t.f64(), minimum: t.f64(), maximum: t.f64(), updated_at: t.timestamp(),
});
export const studioTables = { studio_workspace, studio_presence, studio_task_watch,
  studio_watchdog, studio_attention, studio_job, studio_job_trigger, studio_metric, studio_rollup };
