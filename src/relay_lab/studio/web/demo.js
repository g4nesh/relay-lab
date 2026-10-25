/* Public, illustrative fixtures only. No network, credentials or persistent storage. */
(() => {
  "use strict";
  const provenance = "Illustrative demo data. No live robot execution or measured performance.";
  const timestamp = Date.parse("2026-10-25T09:00:00Z") / 1000;
  const agents = ["left", "right"].map((id, index) => ({
    id, name: index ? "Hem alignment arm" : "Sleeve folding arm", robot: "arm", family: "smolvla",
    source: {kind: "huggingface", ref: "lerobot/smolvla_base", revision: ""},
    action_contract: "Cartesian pose + gripper · illustrative configuration", rig: null,
  }));
  const workspace = {name: "Two-arm folding · demo", objective: "Fold a T-shirt together, sharing local observations and waiting when a peer is blocked.", mode: "mediated", direction: "both", agents};
  const models = agents.map((spec, i) => ({id: `demo-model-${spec.id}`, spec, revision: i + 1, status: "released", updated_at: timestamp - 3600}));
  const user = {id: "demo-visitor", name: "Demo visitor", username: "visitor", role: "viewer"};
  // team.manage here permits rendering sample team rows only. It grants no server access.
  const bootstrap = {workspace, models, user, token: "", permissions: ["team.manage"], release_ready: true, example: null, capabilities: {experiments_enabled: false, demo: true}};
  const workers = agents.map((a, i) => ({id: `demo-worker-${a.id}`, name: `${i ? "Right" : "Left"} arm · example runtime`, agent_id: a.id, model_id: models[i].id, active: true, status: "online", execution_kind: "simulation", last_seen: timestamp, release_revision: models[i].revision}));
  const scenarios = [
    {id: "demo-fold-handoff", title: "Fold the shirt · shared handoff", status: "running", progress: 65, summary: ["Sleeve folded inward. Holding the left edge while the peer aligns the hem.", "Hem visible in my local camera. Aligning to the shared fold line before the final lift."]},
    {id: "demo-obstacle", title: "Wait for a clear work area", status: "blocked", progress: 30, summary: ["A new obstacle covers the left approach. Pausing and sharing the blocked region.", "Peer reported a blocked approach. Holding position until a fresh all-clear arrives."]},
    {id: "demo-complete", title: "Inspect the finished fold", status: "succeeded", progress: 100, summary: ["Example outcome: both sleeves lie within the shirt body; left gripper is clear.", "Example outcome: hem is aligned; right gripper is clear of the shared workcell."]},
  ];
  const tasks = scenarios.map((s, i) => ({
    id: s.id, title: s.title, objective: workspace.objective, status: s.status, target: "both", revision: 4,
    created_by: "demo-operator", created_at: timestamp - i * 1800 - 180, updated_at: timestamp - i * 1800,
    readback_verified: false, success_criteria: ["Both sleeves folded inward.", "Hem aligned and both grippers clear."],
    assignments: Object.fromEntries(agents.map((a, j) => [a.id, {agent_id: a.id, name: a.name, status: s.status, progress: s.progress, summary: s.summary[j], worker_id: workers[j].id, execution_kind: "simulation", release_revision: models[j].revision, instruction: j ? "Use your local view to align the hem. Wait for the peer's sleeve-clear report before lifting." : "Fold the visible sleeve inward. Share the new edge position and wait for hem alignment.", evidence: {kind: "illustrative", note: provenance}}])),
    events: [{type: "assigned", summary: "Sample objective routed to both agents", at: timestamp - i * 1800 - 180, revision: 1}, {type: "peer_context", summary: "Example mediator context: sleeve clear; align hem before the shared lift", at: timestamp - i * 1800 - 30, revision: 3}, {type: "report", summary: s.status === "blocked" ? "Example obstacle report pauses both agents" : "Illustrative agent progress received", at: timestamp - i * 1800, revision: 4}],
  }));
  const team = [{id: "demo-manager", name: "Demo manager", username: "manager-example", role: "manager", active: true}, {id: "demo-operator", name: "Demo operator", username: "operator-example", role: "employee", active: true}];
  const jobs = ["Workspace health", "Telemetry rollup", "History cleanup"].map((label, i) => ({job_id: `demo-job-${i}`, label, enabled: true, schedule: i === 2 ? "daily" : "hourly", next_run_at: timestamp + 3600, last_run_at: timestamp - 1800}));
  const routes = {
    "/api/bootstrap": bootstrap, "/api/workspace": workspace,
    "/api/models": {models, release_ready: true}, "/api/tasks": {tasks}, "/api/workers": {workers},
    "/api/team": {users: team}, "/api/requests": {requests: []},
    "/api/audit": {events: [{action: "model.release", actor: "Demo manager", subject: "Illustrative SmolVLA configuration", created_at: timestamp - 3600, reason: "Sample approval record; no model was deployed."}]},
    "/api/health": {database: {available: false}, sample_available: false},
    "/api/coordination": {enabled: true, available: true, realtime: {state: "demo"}, jobs, presence: workers, rollups: [], attention: [{status: "open", kind: "obstacle", subject_id: "demo-obstacle", message: "Example: left approach obstructed; both arms waiting", created_at: timestamp}]},
    "/api/analytics": {generated_at: timestamp, demo: true, provenance, governance: {models: {total: 2, released: 2, draft: 0, retired: 0}}, operations: {modes: []}, definitions: {demo: provenance, task_status: "Sample task states illustrate running, blocked and complete workflows.", runtime_status: "Example connections, not authenticated live heartbeats.", performance: "No speed, latency, productivity or physical-success claim is made."}, sources: [{id: "public-demo-fixtures", status: "ILLUSTRATIVE", path: "/demo.js", grain: "Hand-authored example records"}]},
  };
  for (const task of tasks) routes[`/api/tasks/${task.id}`] = task;
  function shouldEnter(location) {
    const query = new URLSearchParams(location.search);
    if (query.get("workspace") === "1") return false;
    return query.get("demo") === "1" || !["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  }
  async function request(path, body) {
    if (body !== undefined) throw new Error("This demo is read-only. Sign in to your workspace to make changes.");
    if (!Object.hasOwn(routes, path)) throw new Error("This record is not part of the public demo.");
    return structuredClone({...routes[path], demo: true, provenance});
  }
  window.RelayDemo = Object.freeze({request, shouldEnter, provenance});
})();
