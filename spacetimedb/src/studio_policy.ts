export const HOUR_US = 3_600_000_000n;
export const DAY_US = 24n * HOUR_US;
export const STUDIO_LIMITS = { workspaces: 64, presence: 256, tasks: 512,
  attention: 1024, metrics: 8192, rollups: 8192, batch: 256 } as const;
export const STUDIO_METRICS = new Set(['task_created', 'task_completed', 'task_failed',
  'task_cancelled', 'worker_heartbeat', 'worker_stale', 'task_overdue', 'dispatch_latency_ms',
  'report_latency_ms', 'coordination_latency_ms', 'exchange_latency_ms', 'database_latency_ms',
  'llm_latency_ms', 'exchange_bytes']);
export const STUDIO_JOBS = new Set(['health', 'rollup', 'cleanup']);
export const STUDIO_STATUSES = new Set(['queued', 'running', 'blocked', 'succeeded', 'failed', 'cancelled']);
export function studioTerminal(status: string): boolean {
  return ['succeeded', 'failed', 'cancelled'].includes(status);
}
// Delayed execution skips missed calendar ticks; it never replays a burst of jobs.
export function nextStudioBoundary(now: bigint, schedule: string): bigint {
  if (schedule !== 'hourly' && schedule !== 'daily') throw new Error('Schedule must be hourly or daily (UTC)');
  const period = schedule === 'hourly' ? HOUR_US : DAY_US;
  return (now / period + 1n) * period;
}
export function studioTaskDeadline(now: bigint, timeout: number,
  prior?: { timeout_seconds: number; deadline_us: bigint }): bigint {
  if (timeout === 0) return 0n;
  return prior && prior.timeout_seconds === timeout ? prior.deadline_us : now + BigInt(timeout) * 1_000_000n;
}
