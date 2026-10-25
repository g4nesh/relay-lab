/**
 * Authoritative barriers for one local, synchronous, two-robot simulator.
 *
 * A reducer transaction never executes a robot action. It durably authorizes
 * one batch or commits observed results. An executing batch must never be
 * replayed after a process crash: the simulator may already have moved.
 */
import { schema, table, t, SenderError, type ReducerCtx } from 'spacetimedb/server';
import { ScheduleAt } from 'spacetimedb';
import { studioTables, studio_watchdog, studio_job_trigger } from './studio_tables';
import { DAY_US, HOUR_US, STUDIO_LIMITS, STUDIO_METRICS, STUDIO_JOBS, STUDIO_STATUSES,
  studioTerminal, nextStudioBoundary, studioTaskDeadline } from './studio_policy';

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

// Factory coordination is additive; the original two-arm tables remain intact.
const factory_run = table({ public: false }, {
  run_id: t.string().primaryKey(), worker_id: t.string(), spec_json: t.string(),
  state_json: t.string(), revision: t.u32(), tick: t.u32(), max_ticks: t.u32(),
  phase: t.string(), status: t.string(), created_at: t.timestamp(), updated_at: t.timestamp(),
});
const factory_robot = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'), robot_id: t.string(),
  capabilities_json: t.string(), last_sequence: t.i64(),
});
const factory_batch = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'), revision: t.u32(),
  prepare_json: t.string(), result_json: t.string(), created_at: t.timestamp(),
});
const factory_command = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'), command_id: t.string(),
  robot_id: t.string(), job_id: t.string(), command_json: t.string(), status: t.string(),
});
const factory_resource = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'), resource_id: t.string(),
  holder: t.string(), command_id: t.string(), expires_tick: t.u32(),
});
const factory_event = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'), message_id: t.string(),
  sender: t.string(), recipient: t.string(), sequence: t.u32(), created_tick: t.u32(),
  expires_tick: t.u32(), priority: t.u8(), acknowledged: t.bool(), payload_json: t.string(),
});

const factory_embedding_space = table({ public: false }, {
  run_id: t.string().primaryKey(), space_id: t.string(), checkpoint_sha256: t.string(),
  config_json: t.string(),
});
const factory_embedding = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'), robot_scope: t.string().index('btree'),
  embedding_id: t.string(), robot_id: t.string(), tick: t.u32(), expires_tick: t.u32(),
  space_id: t.string(), checkpoint_sha256: t.string(), dimension: t.u8(), modality: t.string(),
  command_id: t.string(), job_id: t.string(), vector: t.array(t.f32()), record_json: t.string(),
});
const factory_semantic_channel = table({ public: false }, {
  run_id: t.string().primaryKey(), config_json: t.string(),
});
const factory_semantic_packet = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'),
  job_id: t.string(), recipient_scope: t.string().index('btree'),
  emitted_tick: t.u32(), expires_tick: t.u32(), mode: t.string(),
  vector: t.array(t.f32()), text_payload: t.string(), packet_json: t.string(),
});

// Private two-arm cloth channels. Stored arrival ticks are a simulated network
// schedule; reducer wall-clock latency does not advance the MuJoCo world.
const cloth_channel = table({ public: false }, {
  run_id: t.string().primaryKey(), worker_id: t.string(), config_json: t.string(),
  last_tick: t.i64(), created_at: t.timestamp(), updated_at: t.timestamp(),
});
const cloth_frame = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'),
  tick: t.u32(), payload_json: t.string(), created_at: t.timestamp(),
});
const cloth_packet = table({ public: false }, {
  key: t.string().primaryKey(), run_id: t.string().index('btree'),
  recipient_scope: t.string().index('btree'), sender: t.string(), observation_tick: t.u32(),
  deliver_tick: t.u32(), expires_tick: t.u32(), dropped: t.bool(), packet_json: t.string(),
});

// Sole durable control-plane checkpoint; never exposed to browser subscriptions.
const studio_state = table({ public: false }, {
  id: t.u8().primaryKey(), revision: t.u32(), payload: t.string(), last_operation: t.string(),
});
const studio_lease = table({ public: false }, {
  id: t.u8().primaryKey(), holder: t.string(), expires_us: t.u64(), last_operation: t.string(),
});

const spacetime = schema({ studio_lease, studio_state, module_owner, task, assignment, plan_frame, action_frame, feedback_frame, factory_run, factory_robot, factory_batch, factory_command, factory_resource, factory_event, factory_embedding_space, factory_embedding, factory_semantic_channel, factory_semantic_packet, cloth_channel, cloth_frame, cloth_packet, ...studioTables });

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

const CLOTH_FEATURES = ['ee_x', 'ee_y', 'ee_z', 'hem_x', 'hem_y', 'hem_z',
  'sleeve_x', 'sleeve_y', 'sleeve_z', 'shoulder_x', 'shoulder_y', 'shoulder_z',
  'grasped', 'obstacle_x', 'obstacle_y', 'obstacle_z', 'obstacle_radius', 'obstacle_active',
  'hem_visible', 'sleeve_visible', 'shoulder_visible', 'grasp_sleeve', 'grasp_hem'];
const CLOTH_SAFETY = ['grasped', 'obstacle_x', 'obstacle_y', 'obstacle_z', 'obstacle_radius', 'obstacle_active',
  'hem_visible', 'sleeve_visible', 'shoulder_visible', 'grasp_sleeve', 'grasp_hem'];
function clothKeys(value: Json, keys: string[], label: string) {
  require(Object.keys(value).length === keys.length && keys.every(key => key in value), `${label} fields mismatch`);
}
function clothFacts(value: unknown, keys: string[]) {
  const facts = object(value, 'cloth features');
  clothKeys(facts, keys, 'Cloth feature');
  for (const key of keys) {
    number(facts[key], 'cloth feature');
    require(Math.abs(facts[key] as number) <= 100, 'Cloth feature exceeds bounds');
  }
}
function clothConfiguration(config_json: string) {
  const config = parse(config_json, 'cloth channel configuration');
  clothKeys(config, ['schema', 'mode', 'model', 'network', 'features', 'hybrid_fields'], 'Cloth configuration');
  const learnedLatent = config.schema === 'relay.cloth.learned-latent.v1';
  require(learnedLatent || config.schema === 'relay.cloth.communication.v2', 'Unknown cloth channel schema');
  require(['embeddings', 'structured_text', 'hybrid', 'none'].includes(config.mode as string), 'Unknown cloth channel mode');
  require(JSON.stringify(config.features) === JSON.stringify(CLOTH_FEATURES), 'Cloth observation schema mismatch');
  require(JSON.stringify(config.hybrid_fields) === JSON.stringify(CLOTH_SAFETY), 'Cloth safety schema mismatch');
  const model = object(config.model, 'cloth feature codec');
  text(model.space_id, 'cloth codec space', 128);
  require(typeof model.checkpoint_sha256 === 'string' && /^[a-f0-9]{64}$/.test(model.checkpoint_sha256), 'Invalid cloth codec hash');
  if (learnedLatent) {
    // The learned mixed-backbone protocol has its own version. The original
    // measured-state PCA contract and all recorded channels remain unchanged.
    require(model.kind === 'cross-vla-adapter-v1' && model.dimension === 64,
      'Invalid learned cloth adapter kind or dimension');
    const dimensions = object(model.source_dimensions, 'cloth backbone dimensions');
    clothKeys(dimensions, ['left', 'right'], 'Cloth backbone dimensions');
    require(dimensions.left === 960 && dimensions.right === 1024, 'Invalid cloth backbone dimensions');
    const extractors = object(model.source_extractor_ids, 'cloth feature extractor identities');
    clothKeys(extractors, ['left', 'right'], 'Cloth feature extractor identities');
    for (const arm of ['left', 'right']) {
      require(typeof extractors[arm] === 'string' && /^[a-f0-9]{64}$/.test(extractors[arm] as string),
        'Invalid cloth feature extractor identity');
    }
  } else {
    require(model.kind === 'standardized-pca-v1' && model.source_dimension === CLOTH_FEATURES.length &&
      Number.isInteger(model.dimension) && (model.dimension as number) >= 1 && (model.dimension as number) < CLOTH_FEATURES.length,
    'Invalid cloth codec dimensions');
  }
  const network = object(config.network, 'cloth network');
  clothKeys(network, ['seed', 'min_delay_ticks', 'max_delay_ticks', 'drop_probability', 'ttl_ticks'], 'Cloth network');
  require(Number.isInteger(network.seed) && (network.seed as number) >= 0 && (network.seed as number) < 4294967296, 'Invalid network seed');
  require(Number.isInteger(network.min_delay_ticks) && Number.isInteger(network.max_delay_ticks) &&
    (network.min_delay_ticks as number) >= 1 && (network.min_delay_ticks as number) <= (network.max_delay_ticks as number) &&
    (network.max_delay_ticks as number) <= 10000, 'Invalid cloth delay limits');
  require(Number.isInteger(network.ttl_ticks) && (network.ttl_ticks as number) >= 1 && (network.ttl_ticks as number) <= 10000, 'Invalid cloth observation TTL');
  number(network.drop_probability, 'cloth packet drop probability');
  require(network.drop_probability >= 0 && network.drop_probability <= 1, 'Invalid cloth packet drop probability');
  return { config, model, network };
}

export const create_cloth_channel = spacetime.reducer({
  run_id: t.string(), worker_id: t.string(), config_json: t.string(),
}, (ctx, { run_id, worker_id, config_json }) => {
  owner(ctx);
  identifier(run_id, 'cloth run'); identifier(worker_id, 'cloth worker');
  clothConfiguration(config_json);
  const old = ctx.db.cloth_channel.run_id.find(run_id);
  if (old) {
    require(old.worker_id === worker_id && old.config_json === config_json, 'Cloth channel creation conflict');
    return;
  }
  ctx.db.cloth_channel.insert({ run_id, worker_id, config_json, last_tick: -1n,
    created_at: ctx.timestamp, updated_at: ctx.timestamp });
});

export const commit_cloth_frame = spacetime.reducer({
  run_id: t.string(), worker_id: t.string(), tick: t.u32(), payload_json: t.string(),
}, (ctx, { run_id, worker_id, tick, payload_json }) => {
  owner(ctx);
  identifier(run_id, 'cloth run'); identifier(worker_id, 'cloth worker');
  const channel = ctx.db.cloth_channel.run_id.find(run_id);
  require(channel && channel.worker_id === worker_id, 'Cloth channel belongs to another worker');
  require(tick <= 100000, 'Cloth run exceeds supported simulation horizon');
  const payload = parse(payload_json, 'cloth frame');
  clothKeys(payload, ['packets', 'state'], 'Cloth frame');
  const state = object(payload.state, 'cloth recorded state');
  require(!('tick' in state) || state.tick === tick, 'Cloth recorded state tick mismatch');
  const key = `${run_id}:${tick}`, old = ctx.db.cloth_frame.key.find(key);
  if (old) {
    require(old.payload_json === payload_json, 'Cloth frame retry changes committed content');
    return;
  }
  require(BigInt(tick) > channel.last_tick, 'Cloth frame cannot move backwards');
  require(Array.isArray(payload.packets) && payload.packets.length <= 2, 'Too many cloth packets');
  const { config, model, network } = clothConfiguration(channel.config_json);
  const senders = new Set<string>();
  for (const raw of payload.packets) {
    const packet = object(raw, 'cloth packet');
    clothKeys(packet, ['schema', 'run_id', 'mode', 'packet_id', 'sender', 'recipient', 'observation_tick',
      'emitted_tick', 'deliver_tick', 'expires_tick', 'dropped', 'source_sha256', 'checkpoint_sha256', 'space_id', 'content'], 'Cloth packet');
    require(config.mode !== 'none' && packet.schema === config.schema && packet.mode === config.mode && packet.run_id === run_id,
      'Cloth packet run/schema/mode mismatch');
    require(packet.sender === 'left' || packet.sender === 'right', 'Invalid cloth sender');
    const sender = packet.sender as string;
    require(!senders.has(sender), 'Duplicate cloth sender in batch'); senders.add(sender);
    require(packet.recipient === (sender === 'left' ? 'right' : 'left'), 'Cloth packet must go to the peer');
    require(packet.observation_tick === tick && packet.emitted_tick === tick && packet.packet_id === `${sender}-${tick}`,
      'Cloth observation timestamp or identity mismatch');
    require(Number.isInteger(packet.deliver_tick) && (packet.deliver_tick as number) >= tick + (network.min_delay_ticks as number) &&
      (packet.deliver_tick as number) <= tick + (network.max_delay_ticks as number), 'Cloth arrival outside configured delay range');
    require(packet.expires_tick === tick + (network.ttl_ticks as number) && typeof packet.dropped === 'boolean', 'Invalid cloth expiration or drop flag');
    require(typeof packet.source_sha256 === 'string' && /^[a-f0-9]{64}$/.test(packet.source_sha256), 'Invalid cloth source provenance');
    require(packet.checkpoint_sha256 === model.checkpoint_sha256 && packet.space_id === model.space_id, 'Cloth codec identity mismatch');
    const content = object(packet.content, 'cloth message content');
    clothKeys(content, config.mode === 'embeddings' ? ['vector'] : config.mode === 'structured_text' ? ['text'] : ['vector', 'text'], 'Cloth content');
    if ('vector' in content) {
      require(Array.isArray(content.vector) && content.vector.length === model.dimension, 'Invalid cloth latent dimension');
      const bound = config.schema === 'relay.cloth.learned-latent.v1' ? 1.00001 : 1000;
      for (const value of content.vector) { number(value, 'cloth latent'); require(Math.abs(value) <= bound, 'Cloth latent exceeds bounds'); }
    }
    if ('text' in content) {
      text(content.text, 'cloth structured text', 6000);
      clothFacts(parse(content.text, 'cloth structured facts'), config.mode === 'structured_text' ? CLOTH_FEATURES : CLOTH_SAFETY);
    }
    ctx.db.cloth_packet.insert({ key: `${run_id}:${packet.packet_id}`, run_id,
      recipient_scope: `${run_id}:${packet.recipient}`, sender, observation_tick: tick,
      deliver_tick: packet.deliver_tick as number, expires_tick: packet.expires_tick as number,
      dropped: packet.dropped as boolean, packet_json: JSON.stringify(packet) });
  }
  ctx.db.cloth_frame.insert({ key, run_id, tick, payload_json, created_at: ctx.timestamp });
  ctx.db.cloth_channel.run_id.update({ ...channel, last_tick: BigInt(tick), updated_at: ctx.timestamp });
});
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

// ----- Heterogeneous factory coordination protocol v1 -----
const FACTORY_STAGES = ['queued', 'loaded', 'at_cell', 'in_fixture', 'processed', 'outbound_loaded', 'at_dispatch', 'completed'];
const FACTORY_CAPABILITIES = ['transfer', 'navigate', 'transfer', 'process', 'transfer', 'navigate', 'transfer'];
function factoryRun(ctx: Ctx, run_id: string, worker_id: string) {
  owner(ctx); identifier(run_id, 'factory run'); identifier(worker_id, 'factory worker');
  const run = ctx.db.factory_run.run_id.find(run_id);
  require(run && run.worker_id === worker_id, 'Unknown factory run or wrong worker');
  return run;
}
function array(value: unknown, label: string, max = 64): unknown[] {
  require(Array.isArray(value) && value.length <= max, `Invalid ${label}`); return value;
}
function integer(value: unknown, label: string, min: number, max: number): asserts value is number {
  require(Number.isInteger(value) && (value as number) >= min && (value as number) <= max, `Invalid ${label}`);
}
function factoryMessages(ctx: Ctx, run_id: string, messages: unknown, tick: number) {
  for (const value of array(messages, 'messages', 64)) {
    const msg = object(value, 'message');
    text(msg.message_id, 'message ID', 128); identifier(msg.message_id, 'message ID');
    text(msg.sender, 'message sender', 128); text(msg.recipient, 'message recipient', 128);
    const key = `${run_id}:${msg.message_id}`, payload_json = JSON.stringify(msg);
    const prior = ctx.db.factory_event.key.find(key);
    if (prior) { require(prior.payload_json === payload_json, 'Conflicting duplicate message'); continue; }
    const sender = ctx.db.factory_robot.key.find(`${run_id}:${msg.sender}`);
    require(sender, 'Unknown message sender');
    require(msg.recipient === 'broadcast' || ctx.db.factory_robot.key.find(`${run_id}:${msg.recipient}`), 'Unknown message recipient');
    integer(msg.sequence, 'message sequence', 0, 4_294_967_295);
    require(BigInt(msg.sequence) === sender.last_sequence + 1n, 'Stale or out-of-order message');
    integer(msg.expires_tick, 'message expiry', tick + 1, tick + 10_000);
    integer(msg.priority, 'message priority', 0, 10); number(msg.confidence, 'confidence');
    require(msg.confidence >= 0 && msg.confidence <= 1, 'Confidence outside range');
    require(['intent', 'belief', 'request', 'ack', 'outcome', 'blocked'].includes(msg.kind as string), 'Unknown message kind');
    object(msg.payload, 'message payload');
    let parent;
    if (msg.correlation_id !== null && msg.correlation_id !== undefined) {
      text(msg.correlation_id, 'causal parent', 128);
      parent = ctx.db.factory_event.key.find(`${run_id}:${msg.correlation_id}`);
      require(parent, 'Unknown causal parent');
    }
    if (msg.kind === 'ack') {
      require(parent && parent.recipient === msg.sender && parent.sender === msg.recipient && parent.expires_tick > tick, 'Invalid ACK scope or expired message');
      ctx.db.factory_event.key.update({ ...parent, acknowledged: true });
    } else {
      const targets = [...ctx.db.factory_robot.run_id.filter(run_id)].filter(r =>
        r.robot_id !== msg.sender && (msg.recipient === 'broadcast' || r.robot_id === msg.recipient));
      const live = [...ctx.db.factory_event.run_id.filter(run_id)].filter(e =>
        !e.acknowledged && e.expires_tick > tick && parse(e.payload_json, 'stored message').kind !== 'ack');
      for (const target of targets) require(live.filter(e =>
        e.sender !== target.robot_id && (e.recipient === target.robot_id || e.recipient === 'broadcast')).length < 64, 'Recipient queue full');
    }
    ctx.db.factory_event.insert({ key, run_id, message_id: msg.message_id, sender: msg.sender,
      recipient: msg.recipient, sequence: msg.sequence, created_tick: tick, expires_tick: msg.expires_tick,
      priority: msg.priority, acknowledged: false, payload_json });
    ctx.db.factory_robot.key.update({ ...sender, last_sequence: BigInt(msg.sequence) });
  }
}

const create_factory_run_args = { run_id: t.string(), worker_id: t.string(), spec_json: t.string(), state_json: t.string(), };
function create_factory_run_handler(ctx: Ctx, args: { run_id: string; worker_id: string; spec_json: string; state_json: string }) {
  owner(ctx); identifier(args.run_id, 'factory run'); identifier(args.worker_id, 'factory worker');
  const spec = parse(args.spec_json, 'factory spec'), state = parse(args.state_json, 'initial factory state');
  const prior = ctx.db.factory_run.run_id.find(args.run_id);
  if (prior) {
    require(prior.worker_id === args.worker_id && prior.spec_json === args.spec_json, 'Conflicting factory creation');
    require(prior.revision === 0 && prior.phase === 'ready' && prior.state_json === args.state_json, 'Existing execution cannot be resumed with a new world'); return;
  }
  integer(spec.max_ticks, 'factory tick budget', 1, 100_000);
  require(state.tick === 0 && state.success === false, 'Invalid initial factory state');
  const robots = array(spec.robots, 'factory robots', 32); require(robots.length > 0, 'Missing robots');
  const ids = new Set<string>();
  for (const item of robots) {
    const robot = object(item, 'robot'); text(robot.robot_id, 'robot ID', 128); identifier(robot.robot_id, 'robot ID');
    require(!ids.has(robot.robot_id) && !['mediator', 'judge'].includes(robot.robot_id), 'Duplicate or reserved robot ID'); ids.add(robot.robot_id);
    const caps = array(robot.capabilities, 'capabilities', 32); require(caps.length > 0, 'No capabilities'); caps.forEach(c => text(c, 'capability', 128));
    ctx.db.factory_robot.insert({ key: `${args.run_id}:${robot.robot_id}`, run_id: args.run_id,
      robot_id: robot.robot_id, capabilities_json: JSON.stringify(caps), last_sequence: -1n });
  }
  for (const id of ['mediator', 'judge']) ctx.db.factory_robot.insert({ key: `${args.run_id}:${id}`,
    run_id: args.run_id, robot_id: id, capabilities_json: '[]', last_sequence: -1n });
  const jobs = array(spec.jobs, 'factory jobs', 64); require(jobs.length > 0, 'Missing jobs');
  if (state.task === 'dispatch-routing') require(jobs.length <= 3, 'Routing supports one to three jobs');
  const observedJobs = object(state.jobs, 'initial job inventory'), jobIds = new Set<string>();
  for (const item of jobs) {
    const job = object(item, 'job'); text(job.job_id, 'job ID', 128); identifier(job.job_id, 'job ID');
    require(!jobIds.has(job.job_id), 'Duplicate job'); jobIds.add(job.job_id);
    const observed = object(observedJobs[job.job_id], 'initial job');
    require(observed.stage === job.stage && observed.stage !== 'completed', 'Invalid initial job stage');
    if (state.task === 'dispatch-routing') require(observed.routing_correct === null || observed.routing_correct === undefined, 'Initial route correctness cannot be prefilled');
  }
  require(Object.keys(observedJobs).length === jobs.length, 'Wrong initial job inventory');
  const resources = array(spec.resources, 'factory resources', 128);
  require(new Set(resources).size === resources.length, 'Duplicate resource registry entry');
  for (const resource of new Set([...resources, ...ids])) {
    text(resource, 'resource', 128); identifier(resource, 'resource');
    ctx.db.factory_resource.insert({ key: `${args.run_id}:${resource}`, run_id: args.run_id,
      resource_id: resource, holder: '', command_id: '', expires_tick: 0 });
  }
  ctx.db.factory_run.insert({ ...args, revision: 0, tick: 0, max_ticks: spec.max_ticks,
    phase: 'ready', status: 'running', created_at: ctx.timestamp, updated_at: ctx.timestamp });
}
export const create_factory_run = spacetime.reducer(create_factory_run_args, (ctx, args) => {
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  create_factory_run_handler(ctx, args);
});
export const studio_create_factory_run = spacetime.reducer({
  holder: t.string(), ...create_factory_run_args,
}, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  create_factory_run_handler(ctx, args);
});

const prepare_factory_batch_args = { run_id: t.string(), worker_id: t.string(), revision: t.u32(), prepare_json: t.string(), };
function prepare_factory_batch_handler(ctx: Ctx, args: { run_id: string; worker_id: string; revision: number; prepare_json: string }) {
  const run = factoryRun(ctx, args.run_id, args.worker_id), key = `${args.run_id}:${args.revision}`;
  const prior = ctx.db.factory_batch.key.find(key);
  if (prior) {
    require(prior.prepare_json === args.prepare_json, 'Conflicting duplicate factory batch');
    require(run.status === 'running' && run.phase === 'executing' && run.revision === args.revision && !prior.result_json, 'Old or terminal batch cannot authorize physics'); return;
  }
  require(run.phase === 'ready' && run.status === 'running' && run.revision === args.revision, 'Factory phase or revision conflict');
  const batch = parse(args.prepare_json, 'factory batch'), state = parse(run.state_json, 'stored factory state');
  const commands = array(batch.commands, 'factory commands', 32), seenRobots = new Set<string>(), seenJobs = new Set<string>();
  const jobs = object(state.jobs, 'stored factory jobs');
  factoryMessages(ctx, args.run_id, batch.messages ?? [], run.tick);
  const included = new Set(commands.map(c => object(c, 'command').command_id));
  for (const c of ctx.db.factory_command.run_id.filter(args.run_id)) require(c.status !== 'running' || included.has(c.command_id), 'Running commands cannot be omitted');
  for (const value of commands) {
    const command = object(value, 'factory command');
    text(command.command_id, 'command ID', 128); identifier(command.command_id, 'command ID');
    const oldCommand = ctx.db.factory_command.key.find(`${args.run_id}:${command.command_id}`);
    if (oldCommand) require(oldCommand.status === 'running' && oldCommand.command_json === JSON.stringify(command), 'Terminal or changed command replay');
    text(command.robot_id, 'command robot', 128);
    const robot = ctx.db.factory_robot.key.find(`${args.run_id}:${command.robot_id}`);
    require(robot && !seenRobots.has(command.robot_id), 'Unknown or duplicate robot'); seenRobots.add(command.robot_id);
    require((JSON.parse(robot.capabilities_json) as unknown[]).includes(command.capability), 'Robot lacks capability');
    const payload = object(command.payload ?? {}, 'command payload');
    if (payload.handoff_message_id) {
      text(payload.handoff_message_id, 'handoff message', 128);
      require(ctx.db.factory_event.key.find(`${args.run_id}:${payload.handoff_message_id}`)?.acknowledged, 'Handoff not acknowledged');
    }
    const job_id = command.job_id ?? ''; text(job_id, 'command job', 128, true);
    if (job_id) {
      const job = object(jobs[job_id], 'command job');
      require(!seenJobs.has(job_id), 'Conflicting job custody'); seenJobs.add(job_id);
      if (!oldCommand && job.custodian && job.custodian !== command.robot_id && ctx.db.factory_robot.key.find(`${args.run_id}:${job.custodian}`)) {
        const handoff = typeof payload.handoff_message_id === 'string' ? ctx.db.factory_event.key.find(`${args.run_id}:${payload.handoff_message_id}`) : undefined;
        const parent = handoff ? parse(handoff.payload_json, 'handoff request') : {};
        require(command.capability === 'transfer' && handoff?.acknowledged && parent.kind === 'request' &&
          parent.sender === command.robot_id && parent.recipient === job.custodian &&
          object(parent.payload, 'handoff payload').job_id === job_id &&
          object(parent.payload, 'handoff payload').command_id === command.command_id && (oldCommand || handoff.expires_tick > run.tick), 'Robot custody requires acknowledged handoff');
      }
      const expected = command.expected_stage ?? payload.expected_stage;
      require(expected === undefined || expected === job.stage, 'Stale job stage');
    }
    const lease = command.lease_ticks ?? 1; integer(lease, 'resource lease', 1, 1000);
    const resources = array(command.resources ?? [], 'command resources', 32);
    require(new Set(resources).size === resources.length, 'Duplicate resource');
    const required: unknown[] = ['transfer', 'process'].includes(command.capability as string)
      ? [payload.source, payload.target] : command.capability === 'navigate' ? [payload.target, 'aisle-main'] : [];
    if (job_id) required.push(object(jobs[job_id], 'job').cargo_id);
    for (const resource of required) if (typeof resource === 'string' && ctx.db.factory_resource.key.find(`${args.run_id}:${resource}`)) {
      require(resources.includes(resource), 'Command omitted required station, aisle, or cargo resources');
    }
    // The actor lease is implicit and cannot be omitted to bypass a peer's
    // reservation of this robot during an in-progress handoff.
    for (const resource of new Set([command.robot_id, ...resources])) {
      text(resource, 'resource ID', 128);
      const lock = ctx.db.factory_resource.key.find(`${args.run_id}:${resource}`);
      require(lock && (!lock.holder || lock.expires_tick <= run.tick || lock.holder === command.robot_id), 'Resource unavailable');
      ctx.db.factory_resource.key.update({ ...lock, holder: command.robot_id,
        command_id: command.command_id, expires_tick: run.tick + lease });
    }
    const row = { key: `${args.run_id}:${command.command_id}`, run_id: args.run_id,
      command_id: command.command_id, robot_id: command.robot_id, job_id, command_json: JSON.stringify(command), status: 'authorized' };
    if (oldCommand) ctx.db.factory_command.key.update(row); else ctx.db.factory_command.insert(row);
  }
  ctx.db.factory_batch.insert({ key, run_id: args.run_id, revision: args.revision,
    prepare_json: args.prepare_json, result_json: '', created_at: ctx.timestamp });
  ctx.db.factory_run.run_id.update({ ...run, phase: 'executing', updated_at: ctx.timestamp });
}
export const prepare_factory_batch = spacetime.reducer(prepare_factory_batch_args, (ctx, args) => {
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  prepare_factory_batch_handler(ctx, args);
});
export const studio_prepare_factory_batch = spacetime.reducer({
  holder: t.string(), ...prepare_factory_batch_args,
}, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  prepare_factory_batch_handler(ctx, args);
});

const commit_factory_batch_args = { run_id: t.string(), worker_id: t.string(), revision: t.u32(), result_json: t.string(), };
function commit_factory_batch_handler(ctx: Ctx, args: { run_id: string; worker_id: string; revision: number; result_json: string }) {
  const run = factoryRun(ctx, args.run_id, args.worker_id), key = `${args.run_id}:${args.revision}`;
  const batch = ctx.db.factory_batch.key.find(key); require(batch, 'Factory batch not authorized');
  if (batch.result_json) { require(batch.result_json === args.result_json, 'Conflicting duplicate factory outcome'); return; }
  require(run.phase === 'executing' && run.revision === args.revision, 'Factory result revision conflict');
  const result = parse(args.result_json, 'factory result'), state = object(result.state, 'factory observed state');
  require(state.tick === run.tick + 1, 'Factory observed tick conflict');
  const previousState = parse(run.state_json, 'previous factory state');
  require(state.task === previousState.task, 'Factory task is immutable');
  const routing = state.task === 'dispatch-routing';
  const commands = array(parse(batch.prepare_json, 'stored batch').commands, 'stored commands');
  const outcomes = array(result.outcomes, 'factory outcomes', 32);
  require(outcomes.length === commands.length, 'Missing factory outcomes');
  const byId = new Map(commands.map(c => { const x = object(c, 'command'); return [x.command_id, x]; }));
  const seen = new Set<unknown>();
  for (const item of outcomes) {
    const outcome = object(item, 'factory outcome'), command = byId.get(outcome.command_id);
    require(command && !seen.has(outcome.command_id), 'Unknown or duplicate outcome'); seen.add(outcome.command_id);
    require(outcome.robot_id === command.robot_id, 'Factory outcome robot mismatch');
    require(['running', 'completed', 'failed'].includes(outcome.status as string), 'Invalid factory outcome status');
    object(outcome.observed, 'observed skill evidence');
    const row = ctx.db.factory_command.key.find(`${args.run_id}:${command.command_id}`)!;
    ctx.db.factory_command.key.update({ ...row, status: outcome.status as string });
    if (outcome.status !== 'running') for (const lock of ctx.db.factory_resource.run_id.filter(args.run_id)) {
      if (lock.command_id === command.command_id) ctx.db.factory_resource.key.update({ ...lock, holder: '', command_id: '', expires_tick: 0 });
    }
  }
  const jobs = object(state.jobs, 'observed jobs'), previous = object(parse(run.state_json, 'previous state').jobs, 'previous jobs');
  require(Object.keys(jobs).length === Object.keys(previous).length && Object.keys(jobs).every(id => id in previous), 'Factory job inventory conflict');
  for (const [jid, item] of Object.entries(jobs)) {
    const job = object(item, 'observed job'), old = object(previous[jid], 'previous job');
    text(job.stage, 'observed stage', 128);
    const associated = outcomes.map(x => object(x, 'outcome')).filter(o => byId.get(o.command_id)!.job_id === jid);
    if (routing) {
      if (job.stage === 'completed') {
        require(typeof job.routing_correct === 'boolean', 'Completed routing job requires measured correctness');
        if (old.stage === 'completed') require(job.routing_correct === old.routing_correct, 'Completed routing correctness is immutable');
        else require(associated.length === 1 && associated[0].status === 'completed' && object(associated[0].observed, 'route evidence').routing_correct === job.routing_correct, 'Routing correctness disagrees with terminal evidence');
      } else require(job.routing_correct === null || job.routing_correct === undefined, 'Routing correctness is only observed after completion');
    }
    if (job.stage !== old.stage) {
      require(associated.length === 1 && associated[0].status === 'completed', 'Unobserved factory job transition');
      const outcome = associated[0], command = byId.get(outcome.command_id)!;
      const index = FACTORY_STAGES.indexOf(old.stage as string);
      require(index >= 0 && index < 7 && FACTORY_STAGES[index + 1] === job.stage && FACTORY_CAPABILITIES[index] === command.capability, 'Invalid factory process transition');
      require((command.next_stage ?? object(command.payload ?? {}, 'payload').next_stage) === job.stage && object(outcome.observed, 'skill evidence').stage === job.stage, 'Unauthorized or inconsistent job stage');
    }
    require(job.custodian === null || job.custodian === undefined || job.custodian === '' ||
      (typeof job.custodian === 'string' && (!!ctx.db.factory_robot.key.find(`${args.run_id}:${job.custodian}`) ||
        !!ctx.db.factory_resource.key.find(`${args.run_id}:${job.custodian}`))), 'Unknown job custodian');
    if (job.custodian !== old.custodian) {
      require(associated.length === 1 && object(associated[0].observed, 'custody evidence').custodian === job.custodian, 'Unobserved custody change');
      const command = byId.get(associated[0].command_id)!, payload = object(command.payload ?? {}, 'transfer payload');
      require(command.capability === 'transfer', 'Only a transfer may change cargo custody');
      require([null, undefined, '', command.robot_id, payload.source, payload.target].includes(job.custodian), 'Custody recipient outside authorized transfer');
    }
  }
  const complete = Object.values(jobs).every(item => object(item, 'job').stage === 'completed');
  const success = complete && (!routing || Object.values(jobs).every(item => object(item, 'job').routing_correct === true));
  require(state.success === success, 'Factory success disagrees with observed job stages');
  storeFactoryEmbeddings(ctx, run, result.embeddings ?? [], state, byId);
  storeFactorySemanticPackets(ctx, run, result.semantic_packets ?? [], state, previousState, byId);
  factoryMessages(ctx, args.run_id, result.messages ?? [], state.tick as number);
  ctx.db.factory_batch.key.update({ ...batch, result_json: args.result_json });
  const status = success ? 'succeeded' : complete && routing ? 'rejected' : (state.tick as number) >= run.max_ticks ? 'exhausted' : 'running';
  ctx.db.factory_run.run_id.update({ ...run, state_json: JSON.stringify(state), tick: state.tick as number,
    revision: run.revision + 1, phase: status === 'running' ? 'ready' : 'terminal', status, updated_at: ctx.timestamp });
}
export const commit_factory_batch = spacetime.reducer(commit_factory_batch_args, (ctx, args) => {
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  commit_factory_batch_handler(ctx, args);
});
export const studio_commit_factory_batch = spacetime.reducer({
  holder: t.string(), ...commit_factory_batch_args,
}, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  commit_factory_batch_handler(ctx, args);
});


const abort_factory_run_args = { run_id: t.string(), worker_id: t.string(), reason: t.string(), };
function abort_factory_run_handler(ctx: Ctx, args: { run_id: string; worker_id: string; reason: string }) {
  const run = factoryRun(ctx, args.run_id, args.worker_id);
  text(args.reason, 'factory abort reason', 2048);
  const state = parse(run.state_json, 'factory state');
  if (run.status === 'failed') { require(state.abort_reason === args.reason, 'Conflicting duplicate abort'); return; }
  require(run.status === 'running', 'Cannot abort terminal factory run');
  for (const lock of ctx.db.factory_resource.run_id.filter(args.run_id)) {
    ctx.db.factory_resource.key.update({ ...lock, holder: '', command_id: '', expires_tick: 0 });
  }
  for (const command of ctx.db.factory_command.run_id.filter(args.run_id)) {
    if (['authorized', 'running'].includes(command.status)) ctx.db.factory_command.key.update({ ...command, status: 'aborted' });
  }
  ctx.db.factory_run.run_id.update({ ...run, status: 'failed', phase: 'terminal',
    state_json: JSON.stringify({ ...state, abort_reason: args.reason }), updated_at: ctx.timestamp });
}
export const abort_factory_run = spacetime.reducer(abort_factory_run_args, (ctx, args) => {
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  abort_factory_run_handler(ctx, args);
});
export const studio_abort_factory_run = spacetime.reducer({
  holder: t.string(), ...abort_factory_run_args,
}, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  abort_factory_run_handler(ctx, args);
});


// Learned embeddings are a typed, bounded mailbox, not authoritative task facts.
// Native indexes are B-trees. The client computes exact cosine after selecting
// the newest eligible record from each peer (not ANN or historical-pose search).
export const configure_factory_embeddings = spacetime.reducer({
  run_id: t.string(), worker_id: t.string(), config_json: t.string(),
}, (ctx, args) => {
  const run = factoryRun(ctx, args.run_id, args.worker_id), config = parse(args.config_json, 'embedding config');
  require(config.schema === 'relay.factory.embedding.v1' && config.dimension === 32 && config.modality === 'vla-hidden' && config.max_per_robot === 16, 'Unsupported embedding schema');
  text(config.space_id, 'embedding space', 128); identifier(config.space_id, 'embedding space');
  text(config.checkpoint_sha256, 'checkpoint SHA256', 64);
  require(/^[a-f0-9]{64}$/.test(config.checkpoint_sha256), 'Invalid checkpoint SHA256');
  const prior = ctx.db.factory_embedding_space.run_id.find(args.run_id);
  if (prior) { require(prior.config_json === args.config_json, 'Embedding space is immutable'); return; }
  require(run.revision === 0 && run.phase === 'ready', 'Configure embedding space before execution');
  ctx.db.factory_embedding_space.insert({ run_id: args.run_id, config_json: args.config_json, space_id: config.space_id, checkpoint_sha256: config.checkpoint_sha256 });
});

function storeFactoryEmbeddings(ctx: Ctx, run: ReturnType<typeof factoryRun>, records: unknown,
  state: Json, commands: Map<unknown, Json>) {
  const config = ctx.db.factory_embedding_space.run_id.find(run.run_id);
  const batch = array(records, 'embedding batch', 32);
  require(config || batch.length === 0, 'Embedding space not configured');
  if (!config) return;
  const tick = state.tick as number;
  const robotIds = array(parse(run.spec_json, 'factory spec').robots, 'robots').map(r => object(r, 'robot').robot_id as string);
  const seen = new Set<string>();
  for (const item of batch) {
    const record = object(item, 'embedding'), record_json = JSON.stringify(record);
    require(record_json.length <= 4096, 'Embedding record too large');
    require(record.schema === 'relay.factory.embedding.v1' && record.dimension === 32 && record.modality === 'vla-hidden', 'Embedding schema mismatch');
    require(record.space_id === config.space_id && record.checkpoint_sha256 === config.checkpoint_sha256, 'Embedding model or space mismatch');
    require(record.run_id === run.run_id, 'Cross-run embedding');
    text(record.robot_id, 'embedding robot', 128);
    require(robotIds.includes(record.robot_id) && !seen.has(record.robot_id), 'Unknown or duplicate embedding robot'); seen.add(record.robot_id);
    require(record.tick === tick && record.embedding_id === `${record.robot_id}-${tick}`, 'Stale/future embedding or inconsistent identity');
    integer(record.expires_tick, 'embedding expiry', tick + 1, tick + 64);
    const vector = array(record.vector, 'embedding vector', 32); require(vector.length === 32, 'Embedding dimension mismatch');
    vector.forEach(v => number(v, 'embedding component'));
    require(Math.abs((vector as number[]).reduce((sum, x) => sum + x*x, 0) - 1) <= 0.002, 'Embedding must be unit normalized');
    const decoder = object(record.decoder, 'semantic decoder');
    text(decoder.text, 'decoded text', 512, true); text(decoder.version, 'decoder version', 128); number(decoder.confidence, 'decoder confidence');
    require(decoder.confidence >= 0 && decoder.confidence <= 1, 'Invalid decoder confidence');
    const observed = array(state.robots, 'observed robots', 32).map(r => object(r, 'robot')).find(r => (r.id ?? r.robot_id) === record.robot_id);
    require(observed, 'Embedding has no observed robot');
    if (record.command_id === null) {
      require(record.job_id === null && !observed.active_command, 'Idle embedding claims active robot');
    } else {
      const command = commands.get(record.command_id);
      require(command && command.robot_id === record.robot_id && command.job_id === record.job_id, 'Embedding command or job scope mismatch');
      require(!observed.active_command || observed.active_command === record.command_id, 'Embedding differs from observed active command');
    }
    const key = `${run.run_id}:${record.embedding_id}`, prior = ctx.db.factory_embedding.key.find(key);
    if (prior) { require(prior.record_json === record_json, 'Conflicting duplicate embedding'); continue; }
    ctx.db.factory_embedding.insert({ key, run_id: run.run_id, robot_scope: `${run.run_id}:${record.robot_id}`,
      embedding_id: record.embedding_id as string, robot_id: record.robot_id, tick, expires_tick: record.expires_tick,
      space_id: config.space_id, checkpoint_sha256: config.checkpoint_sha256, dimension: 32, modality: 'vla-hidden',
      command_id: (record.command_id ?? '') as string, job_id: (record.job_id ?? '') as string,
      vector: vector as number[], record_json });
  }
  for (const row of ctx.db.factory_embedding.run_id.filter(run.run_id)) if (row.expires_tick <= tick) ctx.db.factory_embedding.key.delete(row.key);
  for (const robot of robotIds) {
    const rows = [...ctx.db.factory_embedding.robot_scope.filter(`${run.run_id}:${robot}`)].sort((a,b) => b.tick-a.tick);
    for (const row of rows.slice(16)) ctx.db.factory_embedding.key.delete(row.key);
  }
}

// Job beliefs are independent of short-lived pose embeddings. Retransmissions
// preserve the original observation/payload; fresh emission never means fresh RGB.
export const configure_factory_semantic_channel = spacetime.reducer({
  run_id: t.string(), worker_id: t.string(), config_json: t.string(),
}, (ctx, args) => {
  const run = factoryRun(ctx, args.run_id, args.worker_id), config = parse(args.config_json, 'semantic config');
  require(parse(run.state_json, 'factory state').task === 'dispatch-routing', 'Semantic channel requires routing task');
  require(config.schema === 'relay.factory.semantic.v1' && config.space_id === 'factory-route-v1', 'Unsupported semantic schema');
  require(['embeddings', 'text', 'none'].includes(config.mode as string), 'Unsupported semantic mode');
  require(typeof config.checkpoint_sha256 === 'string' && /^[a-f0-9]{64}$/.test(config.checkpoint_sha256), 'Invalid semantic checkpoint');
  integer(config.period_ticks, 'semantic cadence', 1, 64); integer(config.ttl_ticks, 'semantic TTL', 15, 64);
  require(config.period_ticks <= config.ttl_ticks, 'Semantic cadence exceeds TTL');
  const prior = ctx.db.factory_semantic_channel.run_id.find(run.run_id);
  if (prior) { require(prior.config_json === args.config_json, 'Semantic configuration is immutable'); return; }
  require(run.revision === 0 && run.phase === 'ready', 'Configure semantic channel before execution');
  const robots = array(parse(run.spec_json, 'factory spec').robots, 'robots').map(r => object(r, 'robot').robot_id);
  require(robots.includes('humanoid-01') && robots.includes('gantry-01'), 'Routing semantic peers missing');
  ctx.db.factory_semantic_channel.insert({ run_id: run.run_id, config_json: args.config_json });
});

function semanticOrigin(packet: Json) {
  return JSON.stringify(Object.fromEntries(Object.entries(packet).filter(([key]) => !['packet_id', 'emitted_tick', 'expires_tick'].includes(key)).sort(([a], [b]) => a.localeCompare(b))));
}

// Studio operations are private advisory projections. They never dispatch,
// retry or change the authoritative robot execution journal.
function studioScope(ctx: Ctx, workspace_id: string): void {
  identifier(workspace_id, 'workspace ID');
  if (!ctx.db.studio_workspace.workspace_id.find(workspace_id)) {
    require(ctx.db.studio_workspace.count() < BigInt(STUDIO_LIMITS.workspaces), 'Workspace capacity reached');
    ctx.db.studio_workspace.insert({ workspace_id, created_at: ctx.timestamp });
  }
}
function studioCapacity<T>(rows: Iterable<T>, maximum: number, label: string): void {
  let count = 0;
  for (const _row of rows) { count++; require(count < maximum, `${label} capacity reached; run retention cleanup`); }
}
function studioScheduler(ctx: Ctx): void {
  require(ctx.sender.isEqual(ctx.identity), 'Only the database scheduler may execute this function');
}
function studioAlert(ctx: Ctx, workspace_id: string, kind: string, subject_id: string,
  generation: bigint, message: string): void {
  const key = `${workspace_id}:${kind}:${subject_id}`;
  const prior = ctx.db.studio_attention.key.find(key);
  if (prior?.status === 'open' && prior.generation === generation) return;
  if (!prior) studioCapacity(ctx.db.studio_attention.workspace_id.filter(workspace_id), STUDIO_LIMITS.attention, 'Attention');
  const row = { key, workspace_id, kind, subject_id, generation, message,
    status: 'open', created_at: ctx.timestamp, updated_at: ctx.timestamp };
  if (prior) ctx.db.studio_attention.key.update(row); else ctx.db.studio_attention.insert(row);
}
function studioResolve(ctx: Ctx, workspace_id: string, kind: string, subject_id: string): void {
  const row = ctx.db.studio_attention.key.find(`${workspace_id}:${kind}:${subject_id}`);
  if (row && row.status !== 'resolved') ctx.db.studio_attention.key.update({ ...row, status: 'resolved', updated_at: ctx.timestamp });
}
function studioGoal(ctx: Ctx, workspace_id: string, task_id: string): Json | undefined {
  const run = ctx.db.factory_run.run_id.find(task_id);
  if (!run) return;
  const state = parse(run.state_json, 'task state');
  require(state.task === 'studio-objective-v1' && state.registry_id === workspace_id, 'Task does not belong to this workspace');
  return object(state.goal, 'goal');
}
function studioCheckPresence(ctx: Ctx, key: string, generation: bigint): void {
  const row = ctx.db.studio_presence.key.find(key);
  if (!row || row.generation !== generation || row.status !== 'online' || row.lease_expires_us > ctx.timestamp.microsSinceUnixEpoch) return;
  ctx.db.studio_presence.key.update({ ...row, status: 'stale', watchdog_id: 0n, updated_at: ctx.timestamp });
  studioAlert(ctx, row.workspace_id, 'worker_stale', row.worker_id, generation,
    'Runtime heartbeat expired. Inspect the runtime before resuming execution.');
}
function studioCheckTask(ctx: Ctx, key: string, revision: bigint): void {
  const row = ctx.db.studio_task_watch.key.find(key);
  if (!row || BigInt(row.revision) !== revision || studioTerminal(row.status)) return;
  const goal = studioGoal(ctx, row.workspace_id, row.task_id);
  if (goal && studioTerminal(String(goal.status))) {
    ctx.db.studio_watchdog.scheduledId.delete(row.watchdog_id);
    ctx.db.studio_task_watch.key.update({ ...row, status: String(goal.status), deadline_us: 0n, watchdog_id: 0n, updated_at: ctx.timestamp });
    studioResolve(ctx, row.workspace_id, 'task_overdue', row.task_id);
    return;
  }
  if (row.deadline_us === 0n || row.deadline_us > ctx.timestamp.microsSinceUnixEpoch) return;
  ctx.db.studio_task_watch.key.update({ ...row, watchdog_id: 0n, updated_at: ctx.timestamp });
  studioAlert(ctx, row.workspace_id, 'task_overdue', row.task_id, revision,
    'Task deadline passed. Review progress; no robot action was retried.');
}

function studioHandleHeartbeat(ctx: Ctx, args: { workspace_id: string, worker_id: string, agent_id: string, lease_seconds: number }) {
  owner(ctx); studioScope(ctx, args.workspace_id);
  identifier(args.worker_id, 'worker ID'); identifier(args.agent_id, 'agent ID');
  require(args.lease_seconds >= 1 && args.lease_seconds <= 300, 'Lease must be 1 to 300 seconds');
  const key = `${args.workspace_id}:${args.worker_id}`, prior = ctx.db.studio_presence.key.find(key);
  require(!prior || prior.agent_id === args.agent_id, 'Worker agent binding is immutable');
  require(prior?.status !== 'revoked', 'Worker credential is revoked');
  if (!prior) studioCapacity(ctx.db.studio_presence.workspace_id.filter(args.workspace_id), STUDIO_LIMITS.presence, 'Presence');
  if (prior) ctx.db.studio_watchdog.scheduledId.delete(prior.watchdog_id);
  const generation = (prior?.generation ?? 0n) + 1n;
  const lease_expires_us = ctx.timestamp.microsSinceUnixEpoch + BigInt(args.lease_seconds) * 1_000_000n;
  const timer = ctx.db.studio_watchdog.insert({ scheduledId: 0n, scheduledAt: ScheduleAt.time(lease_expires_us),
    workspace_id: args.workspace_id, kind: 'presence', subject_id: args.worker_id, generation });
  const row = { key, workspace_id: args.workspace_id, worker_id: args.worker_id, agent_id: args.agent_id,
    generation, lease_expires_us, status: 'online', watchdog_id: timer.scheduledId, updated_at: ctx.timestamp };
  if (prior) ctx.db.studio_presence.key.update(row); else ctx.db.studio_presence.insert(row);
  studioResolve(ctx, args.workspace_id, 'worker_stale', args.worker_id);
}
export const studio_heartbeat = spacetime.reducer({ workspace_id: t.string(), worker_id: t.string(),
  agent_id: t.string(), lease_seconds: t.u32() }, (ctx, args) => {
  owner(ctx);
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  studioHandleHeartbeat(ctx, args);
});
export const studio_fenced_heartbeat = spacetime.reducer({ holder: t.string(), ...{ workspace_id: t.string(), worker_id: t.string(),
  agent_id: t.string(), lease_seconds: t.u32() } }, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  studioHandleHeartbeat(ctx, args);
});

function studioHandleRevokeWorker(ctx: Ctx, args: { workspace_id: string, worker_id: string }) {
  owner(ctx); studioScope(ctx, args.workspace_id); identifier(args.worker_id, 'worker ID');
  const row = ctx.db.studio_presence.key.find(`${args.workspace_id}:${args.worker_id}`);
  if (!row) {
    studioCapacity(ctx.db.studio_presence.workspace_id.filter(args.workspace_id), STUDIO_LIMITS.presence, 'Presence');
    ctx.db.studio_presence.insert({ key: `${args.workspace_id}:${args.worker_id}`, ...args, agent_id: '',
      generation: 1n, lease_expires_us: 0n, status: 'revoked', watchdog_id: 0n, updated_at: ctx.timestamp });
    return;
  }
  if (row.status === 'revoked') return;
  ctx.db.studio_watchdog.scheduledId.delete(row.watchdog_id);
  ctx.db.studio_presence.key.update({ ...row, generation: row.generation + 1n, status: 'revoked', watchdog_id: 0n, updated_at: ctx.timestamp });
  studioResolve(ctx, args.workspace_id, 'worker_stale', args.worker_id);
}
export const studio_revoke_worker = spacetime.reducer({ workspace_id: t.string(), worker_id: t.string() }, (ctx, args) => {
  owner(ctx);
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  studioHandleRevokeWorker(ctx, args);
});
export const studio_fenced_revoke_worker = spacetime.reducer({ holder: t.string(), ...{ workspace_id: t.string(), worker_id: t.string() } }, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  studioHandleRevokeWorker(ctx, args);
});

function studioHandleSyncTask(ctx: Ctx, args: { workspace_id: string, task_id: string, revision: number, status: string, timeout_seconds: number }) {
  owner(ctx); studioScope(ctx, args.workspace_id); identifier(args.task_id, 'task ID');
  require(STUDIO_STATUSES.has(args.status), 'Unknown task status');
  require(args.timeout_seconds <= 86400, 'Deadline must be at most 86400 seconds');
  const key = `${args.workspace_id}:${args.task_id}`, prior = ctx.db.studio_task_watch.key.find(key);
  if (prior && args.revision < prior.revision) return;
  const goal = studioGoal(ctx, args.workspace_id, args.task_id);
  const status = goal && studioTerminal(String(goal.status)) ? String(goal.status) : args.status;
  if (prior && args.revision === prior.revision) {
    require(prior.timeout_seconds === args.timeout_seconds && (prior.status === status || studioTerminal(status)), 'Conflicting task projection');
    if (prior.status === status) return;
  }
  require(!prior || !studioTerminal(prior.status) || prior.status === status, 'Terminal task cannot reopen or change outcome');
  if (!prior) studioCapacity(ctx.db.studio_task_watch.workspace_id.filter(args.workspace_id), STUDIO_LIMITS.tasks, 'Task');
  if (prior) ctx.db.studio_watchdog.scheduledId.delete(prior.watchdog_id);
  const created = typeof goal?.created_at === 'number' && Number.isFinite(goal.created_at) && goal.created_at > 0
    ? BigInt(Math.floor(goal.created_at * 1_000_000)) : ctx.timestamp.microsSinceUnixEpoch;
  const deadline_us = studioTerminal(status) ? 0n : studioTaskDeadline(created, args.timeout_seconds, prior ?? undefined);
  let watchdog_id = 0n;
  if (deadline_us > 0n) watchdog_id = ctx.db.studio_watchdog.insert({ scheduledId: 0n,
    scheduledAt: ScheduleAt.time(deadline_us > ctx.timestamp.microsSinceUnixEpoch ? deadline_us : ctx.timestamp.microsSinceUnixEpoch + 1n),
    workspace_id: args.workspace_id, kind: 'task', subject_id: args.task_id, generation: BigInt(args.revision) }).scheduledId;
  const row = { key, ...args, status, deadline_us, watchdog_id, updated_at: ctx.timestamp };
  if (prior) ctx.db.studio_task_watch.key.update(row); else ctx.db.studio_task_watch.insert(row);
  if (studioTerminal(status) || deadline_us === 0n || deadline_us > ctx.timestamp.microsSinceUnixEpoch)
    studioResolve(ctx, args.workspace_id, 'task_overdue', args.task_id);
}
export const studio_sync_task = spacetime.reducer({ workspace_id: t.string(), task_id: t.string(),
  revision: t.u32(), status: t.string(), timeout_seconds: t.u32() }, (ctx, args) => {
  owner(ctx);
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  studioHandleSyncTask(ctx, args);
});
export const studio_fenced_sync_task = spacetime.reducer({ holder: t.string(), ...{ workspace_id: t.string(), task_id: t.string(),
  revision: t.u32(), status: t.string(), timeout_seconds: t.u32() } }, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  studioHandleSyncTask(ctx, args);
});

export const studio_expire_watchdog = spacetime.reducer({ onSchedule: studio_watchdog },
  { arg: studio_watchdog.rowType }, (ctx, { arg }) => {
    studioScheduler(ctx);
    const key = `${arg.workspace_id}:${arg.subject_id}`;
    if (arg.kind === 'presence') studioCheckPresence(ctx, key, arg.generation);
    if (arg.kind === 'task') studioCheckTask(ctx, key, arg.generation);
  });

function studioHandleRecordMetric(ctx: Ctx, args: { workspace_id: string, event_id: string, metric: string, value: number }) {
  owner(ctx); studioScope(ctx, args.workspace_id); identifier(args.event_id, 'metric event ID');
  require(STUDIO_METRICS.has(args.metric), 'Unknown metric');
  require(Number.isFinite(args.value) && args.value >= 0 && args.value <= 1e12, 'Metric must be finite and between 0 and 1e12');
  const key = `${args.workspace_id}:${args.event_id}`, prior = ctx.db.studio_metric.key.find(key);
  if (prior) { require(prior.metric === args.metric && prior.value === args.value, 'Conflicting metric event'); return; }
  studioCapacity(ctx.db.studio_metric.workspace_id.filter(args.workspace_id), STUDIO_LIMITS.metrics, 'Metric');
  ctx.db.studio_metric.insert({ key, ...args, occurred_us: ctx.timestamp.microsSinceUnixEpoch, aggregated: false });
}
export const studio_record_metric = spacetime.reducer({ workspace_id: t.string(), event_id: t.string(),
  metric: t.string(), value: t.f64() }, (ctx, args) => {
  owner(ctx);
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  studioHandleRecordMetric(ctx, args);
});
export const studio_fenced_record_metric = spacetime.reducer({ holder: t.string(), ...{ workspace_id: t.string(), event_id: t.string(),
  metric: t.string(), value: t.f64() } }, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  studioHandleRecordMetric(ctx, args);
});

function studioJobWork(ctx: Ctx, workspace_id: string, job_id: string, retention_days: number) {
  let processed = 0, pending = false, error = '';
  const now = ctx.timestamp.microsSinceUnixEpoch;
  if (job_id === 'health') {
    // Workspace quotas bound this sweep; alerts are upserts, never duplicate actions.
    for (const row of ctx.db.studio_presence.workspace_id.filter(workspace_id)) {
      studioCheckPresence(ctx, row.key, row.generation); processed++;
    }
    for (const row of ctx.db.studio_task_watch.workspace_id.filter(workspace_id)) {
      studioCheckTask(ctx, row.key, BigInt(row.revision)); processed++;
    }
  } else if (job_id === 'rollup') {
    let rollupCount = [...ctx.db.studio_rollup.workspace_id.filter(workspace_id)].length;
    for (const row of ctx.db.studio_metric.workspace_id.filter(workspace_id)) {
      if (row.aggregated) continue;
      if (processed >= STUDIO_LIMITS.batch) { pending = true; break; }
      const bucket_us = row.occurred_us / HOUR_US * HOUR_US;
      const key = `${workspace_id}:${row.metric}:${bucket_us}`;
      const prior = ctx.db.studio_rollup.key.find(key);
      if (!prior && rollupCount >= STUDIO_LIMITS.rollups) {
        error = 'Rollup capacity reached; run cleanup or reduce retention'; break;
      }
      const aggregate = { key, workspace_id, metric: row.metric, bucket_us,
        sample_count: (prior?.sample_count ?? 0n) + 1n, total: (prior?.total ?? 0) + row.value,
        minimum: Math.min(prior?.minimum ?? row.value, row.value), maximum: Math.max(prior?.maximum ?? row.value, row.value), updated_at: ctx.timestamp };
      if (prior) ctx.db.studio_rollup.key.update(aggregate);
      else { ctx.db.studio_rollup.insert(aggregate); rollupCount++; }
      ctx.db.studio_metric.key.update({ ...row, aggregated: true }); processed++;
    }
  } else if (job_id === 'cleanup') {
    const cutoff = now - BigInt(retention_days) * DAY_US;
    const remove = (eligible: boolean, action: () => void) => {
      if (!eligible) return;
      if (processed >= STUDIO_LIMITS.batch) { pending = true; return; }
      action(); processed++;
    };
    // Only aggregated samples are deleted. Dedupe lasts for this retention window.
    for (const row of ctx.db.studio_metric.workspace_id.filter(workspace_id))
      remove(row.aggregated && row.occurred_us < cutoff, () => { ctx.db.studio_metric.key.delete(row.key); });
    for (const row of ctx.db.studio_rollup.workspace_id.filter(workspace_id))
      remove(row.bucket_us + HOUR_US < cutoff, () => { ctx.db.studio_rollup.key.delete(row.key); });
    for (const row of ctx.db.studio_attention.workspace_id.filter(workspace_id))
      remove(row.status === 'resolved' && row.updated_at.microsSinceUnixEpoch < cutoff, () => { ctx.db.studio_attention.key.delete(row.key); });
    for (const row of ctx.db.studio_task_watch.workspace_id.filter(workspace_id))
      remove(studioTerminal(row.status) && row.updated_at.microsSinceUnixEpoch < cutoff, () => { ctx.db.studio_watchdog.scheduledId.delete(row.watchdog_id); ctx.db.studio_task_watch.key.delete(row.key); });
    // Revocation tombstones are retained: cleanup must never re-enable a credential.
  }
  return { processed, pending, error };
}
function studioScheduleJob(ctx: Ctx, key: string, at: bigint): bigint {
  return ctx.db.studio_job_trigger.insert({ scheduledId: 0n, scheduledAt: ScheduleAt.time(at), job_key: key }).scheduledId;
}
function studioRunJob(ctx: Ctx, key: string): void {
  const job = ctx.db.studio_job.key.find(key);
  require(job, 'Unknown job');
  const result = studioJobWork(ctx, job.workspace_id, job.job_id, job.retention_days);
  ctx.db.studio_job_trigger.scheduledId.delete(job.scheduled_id);
  const now = ctx.timestamp.microsSinceUnixEpoch;
  // Large backlogs use bounded batches, yielding between transactions. A disabled
  // job can be run manually once without silently enabling recurring execution.
  const next_run_us = !job.enabled ? 0n : result.pending ? now + 1_000_000n : nextStudioBoundary(now, job.schedule);
  const scheduled_id = next_run_us ? studioScheduleJob(ctx, key, next_run_us) : 0n;
  ctx.db.studio_job.key.update({ ...job, next_run_us, scheduled_id, last_run_us: now,
    run_count: job.run_count + 1n, last_processed: result.processed, last_error: result.error });
}

function studioHandleConfigureJob(ctx: Ctx, args: { workspace_id: string, job_id: string, enabled: boolean, schedule: string, retention_days: number }) {
  owner(ctx); studioScope(ctx, args.workspace_id);
  require(STUDIO_JOBS.has(args.job_id), 'Unknown job');
  require(args.schedule === 'hourly' || args.schedule === 'daily', 'Schedule must be hourly or daily (UTC)');
  require(args.retention_days >= 1 && args.retention_days <= 90, 'Retention must be 1 to 90 days');
  const key = `${args.workspace_id}:${args.job_id}`, prior = ctx.db.studio_job.key.find(key);
  if (prior && prior.enabled === args.enabled && prior.schedule === args.schedule && prior.retention_days === args.retention_days) return;
  if (prior) ctx.db.studio_job_trigger.scheduledId.delete(prior.scheduled_id);
  const next_run_us = args.enabled ? nextStudioBoundary(ctx.timestamp.microsSinceUnixEpoch, args.schedule) : 0n;
  const scheduled_id = next_run_us ? studioScheduleJob(ctx, key, next_run_us) : 0n;
  const row = { key, ...args, next_run_us, scheduled_id, last_run_us: prior?.last_run_us ?? 0n,
    run_count: prior?.run_count ?? 0n, last_processed: prior?.last_processed ?? 0, last_error: prior?.last_error ?? '' };
  if (prior) ctx.db.studio_job.key.update(row); else ctx.db.studio_job.insert(row);
}
export const studio_configure_job = spacetime.reducer({ workspace_id: t.string(), job_id: t.string(),
  enabled: t.bool(), schedule: t.string(), retention_days: t.u32() }, (ctx, args) => {
  owner(ctx);
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  studioHandleConfigureJob(ctx, args);
});
export const studio_fenced_configure_job = spacetime.reducer({ holder: t.string(), ...{ workspace_id: t.string(), job_id: t.string(),
  enabled: t.bool(), schedule: t.string(), retention_days: t.u32() } }, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  studioHandleConfigureJob(ctx, args);
});
function studioHandleRunJob(ctx: Ctx, args: { workspace_id: string, job_id: string }) {
  owner(ctx); studioScope(ctx, args.workspace_id); require(STUDIO_JOBS.has(args.job_id), 'Unknown job');
  studioRunJob(ctx, `${args.workspace_id}:${args.job_id}`);
}
export const studio_run_job = spacetime.reducer({ workspace_id: t.string(), job_id: t.string() }, (ctx, args) => {
  owner(ctx);
  require(!ctx.db.studio_state.id.find(0), 'Studio storage requires a fenced request');
  studioHandleRunJob(ctx, args);
});
export const studio_fenced_run_job = spacetime.reducer({ holder: t.string(), ...{ workspace_id: t.string(), job_id: t.string() } }, (ctx, { holder, ...args }) => {
  studioLease(ctx, holder);
  studioHandleRunJob(ctx, args);
});
export const studio_execute_job = spacetime.reducer({ onSchedule: studio_job_trigger },
  { arg: studio_job_trigger.rowType }, (ctx, { arg }) => {
    studioScheduler(ctx);
    const job = ctx.db.studio_job.key.find(arg.job_key);
    if (!job || !job.enabled || job.scheduled_id !== arg.scheduledId) return;
    studioRunJob(ctx, arg.job_key);
  });

function storeFactorySemanticPackets(ctx: Ctx, run: ReturnType<typeof factoryRun>, values: unknown,
  state: Json, previous: Json, commands: Map<unknown, Json>) {
  const packets = array(values, 'semantic packet batch', 3);
  const configured = ctx.db.factory_semantic_channel.run_id.find(run.run_id);
  require(configured || packets.length === 0, 'Semantic channel not configured');
  if (!configured) return;
  const config = parse(configured.config_json, 'semantic config'), tick = state.tick as number;
  const jobs = object(state.jobs, 'observed jobs'), oldJobs = object(previous.jobs, 'previous jobs');
  const seen = new Set<string>();
  for (const value of packets) {
    const packet = object(value, 'semantic packet'); require(config.mode !== 'none', 'Semantic channel is disabled');
    const content = config.mode === 'embeddings' ? 'vector' : 'text';
    const fields = ['schema', 'space_id', 'checkpoint_sha256', 'mode', 'run_id', 'job_id', 'sender', 'recipient', 'topic', 'packet_id', 'emitted_tick', 'expires_tick', 'observation_tick', 'source_command_id', 'source_frame_sha256', content];
    require(Object.keys(packet).length === fields.length && Object.keys(packet).every(key => fields.includes(key)), 'Semantic fields disclose mixed or unsupported data');
    const packet_json = JSON.stringify(packet); require(packet_json.length <= 4096, 'Semantic packet exceeds limit');
    for (const key of ['schema', 'space_id', 'checkpoint_sha256', 'mode']) require(packet[key] === config[key], 'Semantic packet schema/model/mode mismatch');
    text(packet.job_id, 'semantic job', 128);
    require(packet.run_id === run.run_id && packet.job_id in jobs && !seen.has(packet.job_id), 'Semantic run/job scope or duplicate mismatch');
    seen.add(packet.job_id);
    require(packet.sender === 'humanoid-01' && packet.recipient === 'gantry-01' && packet.topic === 'dispatch_route', 'Semantic peer/topic mismatch');
    require(typeof packet.source_frame_sha256 === 'string' && /^[a-f0-9]{64}$/.test(packet.source_frame_sha256), 'Invalid semantic source frame');
    text(packet.source_command_id, 'semantic source command', 128); identifier(packet.source_command_id, 'semantic source command');
    require(packet.emitted_tick === tick && packet.packet_id === `humanoid-01-${packet.job_id}-${tick}`, 'Semantic emission tick or identity mismatch');
    integer(packet.observation_tick, 'semantic observation tick', 0, tick - 1);
    require(packet.expires_tick === tick + (config.ttl_ticks as number), 'Semantic TTL differs from configured delivery');
    const job = object(jobs[packet.job_id], 'semantic job'); require(job.stage !== 'completed', 'Cannot publish completed-job beliefs');
    if (content === 'vector') {
      const vector = array(packet.vector, 'semantic vector', 32); require(vector.length === 32, 'Semantic vector dimension mismatch');
      vector.forEach(value => number(value, 'semantic component'));
      require(Math.abs((vector as number[]).reduce((sum, x) => sum + x*x, 0) - 1) <= 0.002, 'Semantic vector must be unit normalized');
    } else text(packet.text, 'semantic text', 256);
    const key = `${run.run_id}:${packet.job_id}`, prior = ctx.db.factory_semantic_packet.key.find(key);
    if (prior) {
      require(tick % (config.period_ticks as number) === 0, 'Semantic retransmission outside cadence');
      require(semanticOrigin(parse(prior.packet_json, 'old semantic packet')) === semanticOrigin(packet), 'Semantic evidence and payload are immutable');
    } else {
      const command = commands.get(packet.source_command_id);
      require(packet.observation_tick === tick - 1, 'Initial semantic observation must precede this physical tick');
      require(object(oldJobs[packet.job_id], 'previous semantic job').stage === 'queued' && command && command.robot_id === 'humanoid-01' && command.job_id === packet.job_id && command.capability === 'transfer', 'Initial semantic evidence requires this job authorized humanoid transfer');
    }
    const row = { key, run_id: run.run_id, job_id: packet.job_id, recipient_scope: `${run.run_id}:gantry-01`, emitted_tick: tick,
      expires_tick: packet.expires_tick as number, mode: config.mode as string, vector: content === 'vector' ? packet.vector as number[] : [],
      text_payload: content === 'text' ? packet.text as string : '', packet_json };
    if (prior) ctx.db.factory_semantic_packet.key.update(row); else ctx.db.factory_semantic_packet.insert(row);
  }
  for (const row of ctx.db.factory_semantic_packet.run_id.filter(run.run_id)) {
    if (object(jobs[row.job_id], 'semantic job').stage === 'completed') ctx.db.factory_semantic_packet.key.delete(row.key);
  }
}


// A small, separate lease row avoids rewriting the checkpoint on read-only requests.
// Task mutations and checkpoint commits check this same database-side fence.
const STUDIO_LEASE_US = 180_000_000n;
function studioLease(ctx: Ctx, holder: string) {
  owner(ctx); identifier(holder, 'Studio request');
  const row = ctx.db.studio_lease.id.find(0);
  require(row && row.holder === holder && row.expires_us > ctx.timestamp.microsSinceUnixEpoch,
    'Studio request lease expired or superseded');
  return row;
}
export const studio_acquire = spacetime.reducer({ holder: t.string() }, (ctx, args) => {
  owner(ctx); identifier(args.holder, 'Studio request');
  const prior = ctx.db.studio_lease.id.find(0), now = ctx.timestamp.microsSinceUnixEpoch;
  require(!prior || !prior.holder || prior.holder === args.holder || prior.expires_us <= now,
    'Studio workspace is busy');
  if (!ctx.db.studio_state.id.find(0)) ctx.db.studio_state.insert({
    id: 0, revision: 0, payload: '', last_operation: '',
  });
  const row = { id: 0, last_operation: prior?.last_operation ?? '',
    holder: args.holder, expires_us: now + STUDIO_LEASE_US };
  if (prior) ctx.db.studio_lease.id.update(row); else ctx.db.studio_lease.insert(row);
});
export const studio_commit_state = spacetime.reducer({
  holder: t.string(), revision: t.u32(), payload: t.string(), operation_id: t.string(),
}, (ctx, args) => {
  owner(ctx); identifier(args.operation_id, 'Studio operation');
  const prior = ctx.db.studio_state.id.find(0);
  if (prior?.last_operation === args.operation_id) {
    require(prior.revision === args.revision + 1 && prior.payload === args.payload,
      'Conflicting duplicate Studio commit'); return;
  }
  const lease = studioLease(ctx, args.holder);
  require(prior && prior.revision === args.revision && prior.revision < 4_294_967_295, 'Studio revision conflict');
  require(args.payload.length > 0 && args.payload.length <= 1_048_576 &&
    /^[A-Za-z0-9+/]+={0,2}$/.test(args.payload), 'Invalid Studio checkpoint');
  ctx.db.studio_state.id.update({ ...prior, revision: prior.revision + 1,
    payload: args.payload, last_operation: args.operation_id });
  ctx.db.studio_lease.id.update({ ...lease, holder: '', expires_us: 0n });
});
export const studio_release = spacetime.reducer({ holder: t.string() }, (ctx, args) => {
  owner(ctx); identifier(args.holder, 'Studio request');
  const row = ctx.db.studio_lease.id.find(0);
  if (row?.holder === args.holder) ctx.db.studio_lease.id.update({ ...row, holder: '', expires_us: 0n });
});
export const studio_finish_read = spacetime.reducer({
  holder: t.string(), revision: t.u32(),
}, (ctx, args) => {
  owner(ctx);
  const row = ctx.db.studio_state.id.find(0), lease = ctx.db.studio_lease.id.find(0);
  require(row?.revision === args.revision, 'Studio read revision conflict');
  if (lease?.last_operation === args.holder) return;
  const held = studioLease(ctx, args.holder);
  ctx.db.studio_lease.id.update({ ...held, holder: '', expires_us: 0n, last_operation: args.holder });
});
