/**
 * Authoritative barriers for one local, synchronous, two-robot simulator.
 *
 * A reducer transaction never executes a robot action. It durably authorizes
 * one batch or commits observed results. An executing batch must never be
 * replayed after a process crash: the simulator may already have moved.
 */
import { schema, table, t, SenderError, type ReducerCtx } from 'spacetimedb/server';

const module_owner = table({ public: false }, {
  id: t.u8().primaryKey(), identity: t.identity(),
});
const task = table({ public: false }, {
  task_id: t.string().primaryKey(),
  intent: t.string(), spec_json: t.string(), max_steps: t.u32(),
  status: t.string(), phase: t.string(), next_tick: t.u32(),
  worker_id: t.string(), last_feedback_tick: t.i64(),
  reason: t.string(), outcome_json: t.string(),
  created_at: t.timestamp(), updated_at: t.timestamp(),
});
const assignment = table({ public: false }, {
  assignment_id: t.string().primaryKey(), task_id: t.string().index('btree'),
  agent_id: t.string(), role: t.string(), worker_id: t.string(),
  last_feedback_json: t.string(), last_feedback_tick: t.i64(),
  context_text: t.string(), context_critical: t.bool(),
});
const plan_frame = table({ public: false }, {
  frame_id: t.string().primaryKey(), task_id: t.string().index('btree'),
  tick: t.u32(), worker_id: t.string(), picker_json: t.string(), packer_json: t.string(),
  created_at: t.timestamp(),
});
const action_frame = table({ public: false }, {
  frame_id: t.string().primaryKey(), task_id: t.string().index('btree'),
  tick: t.u32(), worker_id: t.string(), picker_json: t.string(), packer_json: t.string(),
  created_at: t.timestamp(),
});
const feedback_frame = table({ public: false }, {
  frame_id: t.string().primaryKey(), task_id: t.string().index('btree'),
  tick: t.u32(), worker_id: t.string(),
  picker_report_json: t.string(), packer_report_json: t.string(), outcome_json: t.string(),
  created_at: t.timestamp(),
});

const spacetime = schema({ module_owner, task, assignment, plan_frame, action_frame, feedback_frame });
export default spacetime;
type Ctx = ReducerCtx<typeof spacetime.schemaType>;
type Json = Record<string, unknown>;
const ROBOTS = ['picker', 'packer'] as const;
const COLORS = ['coral', 'mint', 'gold'];

function require(condition: unknown, message: string): asserts condition {
  if (!condition) throw new SenderError(message);
}
function identifier(value: string, name: string): void {
  require(/^[A-Za-z0-9_-]{1,128}$/.test(value), `Invalid ${name}`);
}
function text(value: unknown, name: string, max = 4096, empty = false): asserts value is string {
  require(typeof value === 'string' && value.length <= max && (empty || value.trim().length > 0), `Invalid ${name}`);
}
function object(value: unknown, name: string): Json {
  require(value !== null && typeof value === 'object' && !Array.isArray(value), `Invalid ${name}`);
  return value as Json;
}
function parse(value: string, name: string): Json {
  text(value, name, 65_536);
  try { return object(JSON.parse(value), name); }
  catch { throw new SenderError(`Invalid ${name} JSON`); }
}
function number(value: unknown, name: string): asserts value is number {
  require(typeof value === 'number' && Number.isFinite(value), `Invalid ${name}`);
}
function vector(value: unknown, name: string): number[] {
  require(Array.isArray(value) && value.length === 3, `Invalid ${name}`);
  value.forEach(item => number(item, name));
  return value as number[];
}
function order(value: unknown): string[] {
  require(Array.isArray(value) && value.length === 3 && new Set(value).size === 3 && value.every(c => COLORS.includes(c)), 'Invalid relay order');
  return value as string[];
}
function owner(ctx: Ctx): void {
  const config = ctx.db.module_owner.id.find(0);
  require(config && config.identity.isEqual(ctx.sender), 'Only the module owner may coordinate robots');
}
function ownedTask(ctx: Ctx, task_id: string) {
  owner(ctx);
  identifier(task_id, 'task ID');
  const value = ctx.db.task.task_id.find(task_id);
  require(value, 'Unknown task');
  return value;
}
function workerTask(ctx: Ctx, task_id: string, worker_id: string) {
  const value = ownedTask(ctx, task_id);
  identifier(worker_id, 'worker ID');
  require(value.worker_id === worker_id, 'Task belongs to a different worker; automatic takeover is disabled');
  return value;
}
function phase(value: ReturnType<typeof workerTask>, tick: number, expected: string) {
  require(value.status === 'running', 'Task is not running');
  require(value.next_tick === tick, 'Stale or future tick');
  require(value.phase === expected, `Task phase must be ${expected}`);
  require(tick < value.max_steps, 'Task execution budget exhausted');
}
function observation(value: unknown, robot: string, tick: number): Json {
  const obs = object(value, 'observation');
  require(obs.agent_id === robot && obs.tick === tick, 'Observation robot/tick mismatch');
  vector(obs.ee_position, 'observed end-effector position');
  number(obs.gripper_opening, 'observed gripper opening');
  require(obs.gripper_opening >= 0 && obs.gripper_opening <= 1, 'Invalid observed gripper opening');
  require(obs.holding === null || (typeof obs.holding === 'string' && COLORS.includes(obs.holding)), 'Invalid observed attachment');
  require(!('rgb' in obs) && !('wrist_rgb' in obs), 'Images must remain local');
  return obs;
}
function context(value: unknown): { text: string; critical: boolean } {
  const ctx = object(value, 'feedback context');
  text(ctx.text, 'feedback text', 850, true);
  require(typeof ctx.critical === 'boolean', 'Invalid feedback critical flag');
  return { text: ctx.text, critical: ctx.critical };
}
function plan(value: string, robot: string, tick: number) {
  const data = parse(value, 'plan');
  observation(data.observation, robot, tick);
  const instruction = object(data.instruction, 'instruction');
  require(['pick', 'place', 'wait'].includes(instruction.kind as string), 'Invalid instruction kind');
  text(instruction.text, 'instruction text');
  const feedback = context(data.context);
  require(instruction.feedback === feedback.text && instruction.replan === feedback.critical, 'Instruction and persisted context disagree');
  return feedback;
}
function action(value: string): Json {
  const data = parse(value, 'action proposal');
  const proposal = object(data.action, 'action');
  const delta = vector(proposal.delta, 'action delta');
  // The simulator applies its own 35 mm norm limit. Proposals are retained as
  // requested so the interpreter can report clipping instead of hiding it.
  require(delta.every(v => Math.abs(v) <= 10), 'Action proposal outside simulation bounds');
  require([-1, 0, 1].includes(proposal.gripper as number), 'Invalid gripper command');
  return proposal;
}
function report(value: string, task_id: string, robot: string, tick: number, proposed: Json, planned: string) {
  const data = parse(value, 'observed report');
  const local = object(data.report, 'action report');
  require(local.schema === 'relay-lab.action-feedback.v1', 'Unsupported action report schema');
  require(local.task_id === task_id && local.agent_id === robot && local.before_tick === tick && local.tick === tick + 1, 'Report task/robot/tick mismatch');
  const after = observation(data.observation, robot, tick + 1);
  const before = object(parse(planned, 'stored plan').observation, 'stored observation');
  const request = vector(local.requested_delta, 'reported requested delta');
  const expected = vector(proposed.delta, 'stored action delta');
  require(request.every((v, i) => v === expected[i]) && local.gripper_command === proposed.gripper, 'Report does not match the authorized action');
  const delta = vector(local.observed_delta, 'observed delta');
  const beforePosition = vector(before.ee_position, 'before position');
  const afterPosition = vector(after.ee_position, 'after position');
  require(delta.every((v, i) => Math.abs(v - (afterPosition[i] - beforePosition[i])) <= 1e-8), 'Reported motion disagrees with observations');
  require(local.holding_before === before.holding && local.holding_after === after.holding, 'Reported attachment disagrees with observations');
  require(local.gripper_opening_before === before.gripper_opening && local.gripper_opening_after === after.gripper_opening, 'Reported gripper disagrees with observations');
  require(typeof local.motion_limited === 'boolean' && typeof local.critical === 'boolean', 'Invalid action report flags');
  text(local.feedback, 'action feedback', 4096);
  require(Array.isArray(local.local_events) && local.local_events.length <= 64, 'Invalid local event list');
  require(Array.isArray(data.execution_events) && data.execution_events.length <= 64, 'Invalid execution event list');
  for (const event of data.execution_events) {
    const e = object(event, 'execution event');
    require(e.task_id === task_id && e.agent_id === robot && e.tick === tick + 1, 'Execution event scope mismatch');
  }
  return context({ text: data.text, critical: data.critical });
}

export const init = spacetime.init(ctx => {
  ctx.db.module_owner.insert({ id: 0, identity: ctx.sender });
});

export const create_task = spacetime.reducer({
  task_id: t.string(), intent: t.string(), spec_json: t.string(), max_steps: t.u32(),
}, (ctx, args) => {
  owner(ctx);
  const { task_id, intent, spec_json, max_steps } = args;
  identifier(task_id, 'task ID');
  text(intent, 'task intent');
  const spec = parse(spec_json, 'task specification');
  require(spec.kind === 'relay', 'Unsupported task adapter');
  order(spec.order);
  require(max_steps >= 1 && max_steps <= 100_000, 'Invalid execution budget');
  const prior = ctx.db.task.task_id.find(task_id);
  if (prior) {
    require(prior.intent === intent && prior.spec_json === spec_json && prior.max_steps === max_steps, 'Conflicting duplicate task');
    return;
  }
  ctx.db.task.insert({ ...args, status: 'queued', phase: 'ready', next_tick: 0,
    worker_id: '', last_feedback_tick: -1n, reason: '', outcome_json: '',
    created_at: ctx.timestamp, updated_at: ctx.timestamp });
  for (const robot of ROBOTS) ctx.db.assignment.insert({
    assignment_id: `${task_id}:${robot}`, task_id, agent_id: robot, role: robot,
    worker_id: '', last_feedback_json: '', last_feedback_tick: -1n,
    context_text: '', context_critical: false,
  });
});

export const claim_task = spacetime.reducer({ task_id: t.string(), worker_id: t.string() }, (ctx, args) => {
  const value = ownedTask(ctx, args.task_id);
  identifier(args.worker_id, 'worker ID');
  if (value.worker_id) {
    require(value.worker_id === args.worker_id, 'Task belongs to a different worker; automatic takeover is disabled');
    return;
  }
  require(value.status === 'queued', 'Task cannot be claimed');
  ctx.db.task.task_id.update({ ...value, worker_id: args.worker_id, status: 'running', updated_at: ctx.timestamp });
  for (const robot of ROBOTS) {
    const row = ctx.db.assignment.assignment_id.find(`${args.task_id}:${robot}`)!;
    ctx.db.assignment.assignment_id.update({ ...row, worker_id: args.worker_id });
  }
});

const batchArgs = { task_id: t.string(), worker_id: t.string(), tick: t.u32(), picker_json: t.string(), packer_json: t.string() };

export const plan_tick = spacetime.reducer(batchArgs, (ctx, args) => {
  const value = workerTask(ctx, args.task_id, args.worker_id);
  const frame_id = `${args.task_id}:${args.tick}`;
  const prior = ctx.db.plan_frame.frame_id.find(frame_id);
  if (prior) {
    require(prior.picker_json === args.picker_json && prior.packer_json === args.packer_json, 'Conflicting duplicate plan');
    return;
  }
  phase(value, args.tick, 'ready');
  const contexts = [plan(args.picker_json, 'picker', args.tick), plan(args.packer_json, 'packer', args.tick)];
  ctx.db.plan_frame.insert({ frame_id, ...args, created_at: ctx.timestamp });
  ROBOTS.forEach((robot, i) => {
    const row = ctx.db.assignment.assignment_id.find(`${args.task_id}:${robot}`)!;
    ctx.db.assignment.assignment_id.update({ ...row, context_text: contexts[i].text, context_critical: contexts[i].critical });
  });
  ctx.db.task.task_id.update({ ...value, phase: 'planned', updated_at: ctx.timestamp });
});

export const prepare_tick = spacetime.reducer(batchArgs, (ctx, args) => {
  const value = workerTask(ctx, args.task_id, args.worker_id);
  const frame_id = `${args.task_id}:${args.tick}`;
  const prior = ctx.db.action_frame.frame_id.find(frame_id);
  if (prior) {
    require(prior.picker_json === args.picker_json && prior.packer_json === args.packer_json, 'Conflicting duplicate action');
    return;
  }
  phase(value, args.tick, 'planned');
  action(args.picker_json); action(args.packer_json);
  ctx.db.action_frame.insert({ frame_id, ...args, created_at: ctx.timestamp });
  ctx.db.task.task_id.update({ ...value, phase: 'executing', updated_at: ctx.timestamp });
});

export const commit_tick = spacetime.reducer({
  task_id: t.string(), worker_id: t.string(), tick: t.u32(),
  picker_report_json: t.string(), packer_report_json: t.string(), outcome_json: t.string(),
}, (ctx, args) => {
  const value = workerTask(ctx, args.task_id, args.worker_id);
  const frame_id = `${args.task_id}:${args.tick}`;
  const prior = ctx.db.feedback_frame.frame_id.find(frame_id);
  if (prior) {
    require(prior.picker_report_json === args.picker_report_json && prior.packer_report_json === args.packer_report_json && prior.outcome_json === args.outcome_json, 'Conflicting duplicate observed result');
    return;
  }
  phase(value, args.tick, 'executing');
  const actions = ctx.db.action_frame.frame_id.find(frame_id)!;
  const plans = ctx.db.plan_frame.frame_id.find(frame_id)!;
  const contexts = [
    report(args.picker_report_json, args.task_id, 'picker', args.tick, action(actions.picker_json), plans.picker_json),
    report(args.packer_report_json, args.task_id, 'packer', args.tick, action(actions.packer_json), plans.packer_json),
  ];
  const outcome = parse(args.outcome_json, 'observed outcome');
  const expected = order(parse(value.spec_json, 'stored task specification').order);
  const actual = order(outcome.order);
  require(actual.every((color, i) => color === expected[i]), 'Observed order differs from the assigned task');
  require(outcome.tick === args.tick + 1, 'Outcome tick mismatch');
  const packed = outcome.packed;
  require(Array.isArray(packed) && packed.length === 3 && packed.every(c => c === null || COLORS.includes(c)), 'Invalid observed packed slots');
  require(new Set(packed.filter(c => c !== null)).size === packed.filter(c => c !== null).length, 'An object cannot occupy multiple slots');
  const success = packed.every((color, i) => color === expected[i]);
  require(outcome.success === success, 'Claimed success differs from observed tray slots');
  const packerObservation = object(parse(args.packer_report_json, 'packer report').observation, 'packer observation');
  require(JSON.stringify(packerObservation.packed) === JSON.stringify(packed) && JSON.stringify(packerObservation.order) === JSON.stringify(actual), 'Global result disagrees with packer observation');
  ctx.db.feedback_frame.insert({ frame_id, ...args, created_at: ctx.timestamp });
  const reports = [args.picker_report_json, args.packer_report_json];
  ROBOTS.forEach((robot, i) => {
    const row = ctx.db.assignment.assignment_id.find(`${args.task_id}:${robot}`)!;
    ctx.db.assignment.assignment_id.update({ ...row,
      last_feedback_json: reports[i], last_feedback_tick: BigInt(args.tick),
      context_text: contexts[i].text, context_critical: contexts[i].critical });
  });
  const status = success ? 'succeeded' : args.tick + 1 >= value.max_steps ? 'exhausted' : 'running';
  ctx.db.task.task_id.update({ ...value, status, phase: status === 'running' ? 'ready' : 'terminal',
    next_tick: args.tick + 1, last_feedback_tick: BigInt(args.tick), outcome_json: args.outcome_json,
    reason: status === 'exhausted' ? 'Execution budget exhausted without observed task success' : '', updated_at: ctx.timestamp });
});

export const abort_task = spacetime.reducer({ task_id: t.string(), worker_id: t.string(), reason: t.string() }, (ctx, args) => {
  const value = workerTask(ctx, args.task_id, args.worker_id);
  text(args.reason, 'abort reason', 2048);
  if (value.status === 'failed') {
    require(value.reason === args.reason, 'Conflicting duplicate abort');
    return;
  }
  require(value.status === 'running', 'Cannot abort a terminal task');
  ctx.db.task.task_id.update({ ...value, status: 'failed', phase: 'terminal', reason: args.reason, updated_at: ctx.timestamp });
});
