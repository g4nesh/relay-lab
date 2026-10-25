/* Customer task control. Only signed-in APIs and agent reports change task state. */
(() => {
  "use strict";
  const el = (id) => document.getElementById(id);
  const html = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const stable = (value) => JSON.stringify(value, (_key, item) => item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a],[b]) => a.localeCompare(b))) : item);
  const terminal = (status) => ["succeeded", "failed", "cancelled"].includes(status);
  const labels = {queued:"Queued", running:"In progress", blocked:"Needs attention", succeeded:"Reported complete", failed:"Reported failed", cancelled:"Cancelled", pending:"Awaiting agent", claimed:"Assigned", idle:"Ready", active:"Active", revoked:"Revoked"};
  const pill = (status) => `<span class="tk-status ${html(["queued","running","blocked","succeeded","failed","cancelled","pending","claimed","idle","active","revoked"].includes(status) ? status : "pending")}"><i></i>${html(labels[status] || status || "Awaiting agent")}</span>`;
  const time = (value, full = false) => { const d = new Date(typeof value === "number" ? value * 1000 : value); return Number.isNaN(d.getTime()) ? "Not reported" : d.toLocaleString(undefined, full ? {month:"short",day:"numeric",hour:"numeric",minute:"2-digit",second:"2-digit"} : {month:"short",day:"numeric",hour:"numeric",minute:"2-digit"}); };
  const pretty = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2);
  let options, generation = 0, request = 0, active = false, timer, loading = false, tasks = [], workers = [], boot = null, selected = null, detail = null, filter = "all", credential = null, createRequest = null, loaded = false, workersLoaded = false, creating = false, issuing = false, listSignature = "", teamSignature = "", workerSignature = "", cancelTarget = null;
  const cancelAttempts = new Map();
  let simulations = null, simulationError = "", simulationRequest = null, simulationBusy = false, simulationPreparing = false, simulationSignature = "", runSignature = "", recipeReview = null;
  const simulationTerminal = (status) => ["succeeded", "failed", "cancelled", "interrupted"].includes(status);
  const simulationLabels = {starting:"Starting runtime",running:"Simulation running",verifying:"Verifying action replay",succeeded:"Simulation complete",failed:"Simulation failed",cancelled:"Simulation cancelled",interrupted:"Runtime interrupted"};
  const localSimulationHost = () => ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);

  function skeleton() {
    return `<div class="tk-heading"><div><div class="eyebrow"><span class="tiny-square"></span> WORKSPACE OPERATIONS</div><h1>Give your team a goal.</h1><p>Assign an objective. Follow each agent. Keep the next step in sync.</p></div><button class="button" id="tk-refresh" aria-label="Refresh tasks">↻ <span>Refresh</span></button></div>
    <div class="tk-state-line"><span><i class="status-dot neutral" id="tk-backend-dot"></i><span id="tk-sync-label">Connecting to your task workspace…</span></span><span>Powered by SpacetimeDB</span></div>
    <div id="tk-error" class="tk-error" role="alert" hidden></div>
    <div class="tk-overview" aria-label="Task overview"><div><span>Active tasks</span><strong id="tk-active-count">—</strong><small>Queued, running or blocked</small></div><div><span>Agent-reported complete</span><strong id="tk-complete-count">—</strong><small>Across loaded task history</small></div><div><span>Approved models</span><strong id="tk-model-count">—</strong><small>Current saved configuration</small></div><div><span>Connected runtimes</span><strong id="tk-worker-count">—</strong><small>Recent authenticated heartbeat</small></div></div>
    <section class="tk-simulation tk-panel" id="tk-simulation" aria-label="Two-arm simulation"><p class="tk-muted">Checking the local simulation runtime…</p></section>
    <div id="tk-simulation-error" class="tk-error" role="alert" hidden></div>
    <div class="tk-create-grid"><section class="tk-compose tk-panel" aria-labelledby="tk-compose-title"><div class="tk-panel-heading"><div><div class="section-overline">01 / DEFINE THE OUTCOME</div><h2 id="tk-compose-title">New task</h2></div><span class="tk-mini-label">OBJECTIVE → AGENT</span></div><form id="tk-create-form"><label for="tk-objective">What should the robots accomplish?</label><textarea id="tk-objective" rows="3" maxlength="2000" minlength="5" placeholder="Move the inspected components to the assembly tray. Coordinate the handoff and stop if either path is blocked." required></textarea><div class="tk-form-row"><label>Task name <span>optional</span><input id="tk-title" maxlength="100" placeholder="e.g. Prepare the assembly tray"></label><label>Assign to<select id="tk-target"><option value="both">Both agents · shared objective</option><option value="left">Agent A</option><option value="right">Agent B</option></select></label></div><label for="tk-criteria">How will the agents know they are done?</label><textarea id="tk-criteria" rows="2" maxlength="3000" placeholder="All components are in the tray.&#10;Both grippers are clear of the work area." required></textarea><p class="tk-field-help">One criterion per line, up to 8 criteria (500 characters each). The exact objective and criteria are delivered to each assigned runtime.</p><div id="tk-create-error" class="tk-error" role="alert" hidden></div><div class="tk-compose-footer"><span id="tk-readiness">Checking released models…</span><button class="button primary" id="tk-create" type="submit" disabled>Assign task ${icon("arrow")}</button></div></form></section>
    <section class="tk-team tk-panel" aria-labelledby="tk-team-title"><div class="tk-panel-heading"><div><div class="section-overline">02 / ROUTE TO YOUR TEAM</div><h2 id="tk-team-title">Assigned agents</h2></div>${icon("link")}</div><div id="tk-team-list"><p class="tk-muted">Loading saved models…</p></div><div class="tk-routing-note">${icon("db")}<p><strong>One shared task state.</strong>SpacetimeDB records assignments and reports. Your runtime receives the goal and returns what happened.</p></div><button class="text-button" id="tk-configure">Manage models & channels ${icon("arrow")}</button></section></div>
    <div class="tk-section-heading"><div><div class="section-overline">03 / FOLLOW THE WORK</div><h2>Task activity</h2><p>Progress comes from your connected agents.</p></div><label class="tk-filter">Show<select id="tk-filter" aria-label="Filter tasks"><option value="all">All tasks</option><option value="active">Active tasks</option><option value="succeeded">Reported complete</option><option value="attention">Needs attention</option><option value="cancelled">Cancelled</option></select></label></div>
    <div class="tk-activity-grid"><section class="tk-task-list tk-panel" aria-label="Task list" id="tk-task-list"><div class="tk-empty">${icon("clock")}<strong>Loading task history</strong><p>Reading the shared task state.</p></div></section><section class="tk-detail tk-panel" id="tk-detail" aria-label="Selected task"><div class="tk-empty">${icon("check")}<strong>The work, step by step.</strong><p>Select a task to see instructions, agent updates, and the recorded outcome.</p></div></section></div>
    <section class="tk-runtime tk-panel" aria-labelledby="tk-runtime-title"><div class="tk-panel-heading"><div><div class="section-overline">Runtime connections</div><h2 id="tk-runtime-title">Bring your agent to the task.</h2><p>Run the worker beside your policy or simulator. It connects outward to this workspace.</p></div>${icon("code")}</div><div class="tk-runtime-body"><div id="tk-workers"></div><div id="tk-worker-manager" hidden><form id="tk-worker-form" class="tk-worker-form"><label>Connection name<input id="tk-worker-name" maxlength="80" placeholder="e.g. Assembly arm runtime" required></label><label>Approved agent<select id="tk-worker-agent"><option value="left">Agent A</option><option value="right">Agent B</option></select></label><label>Runtime purpose<select id="tk-worker-kind"><option value="observation">Observation only</option><option value="simulation">Simulation</option></select></label><button class="button" id="tk-issue" type="submit">Create connection</button></form><p class="tk-field-help">Each credential is restricted to one approved model revision. Revoking it prevents future API access.</p><div id="tk-worker-error" class="tk-error" role="alert" hidden></div></div><p id="tk-worker-employee" class="tk-muted" hidden>A manager can create a runtime credential for an approved model.</p></div></section>
    <div class="tk-footer"><span>Relay Studio / task control</span><p>Task outcomes are agent-reported. A completed status does not independently verify physical execution.</p></div>
    <dialog id="tk-credential-dialog" class="tk-credential-dialog" aria-labelledby="tk-credential-title"><div class="dialog-heading"><div class="eyebrow">Runtime connection</div><button class="close-button" id="tk-close-credential" aria-label="Close connection credential">×</button></div><h2 id="tk-credential-title">Connect your agent.</h2><p class="dialog-subtitle">Save this credential now. It is shown once and belongs only to the selected model revision.</p><div class="tk-credential-scope" id="tk-credential-scope"></div><label class="field-label" for="tk-credential-value">Worker credential</label><textarea id="tk-credential-value" readonly rows="3" spellcheck="false" aria-label="One-time worker credential"></textarea><div class="tk-credential-actions"><button class="button" id="tk-copy-credential">Copy credential</button><button class="button primary" id="tk-download-config">Download token file</button></div><div class="tk-setup-guide"><strong>Next, start your runtime bridge.</strong><p>The worker fetches its assignment and reports observed outcomes. Your adapter remains responsible for policy inference and safe robot control.</p><pre id="tk-worker-command"></pre></div><p class="tk-field-help">Keep worker-token private and restrict access to it. Never commit this credential to Git.</p></dialog>
    <dialog id="tk-cancel-dialog" aria-labelledby="tk-cancel-title"><form id="tk-cancel-form"><div class="dialog-heading"><div class="eyebrow">Task control</div><button type="button" class="close-button" id="tk-close-cancel" aria-label="Close cancellation">×</button></div><h2 id="tk-cancel-title">Cancel this task?</h2><p class="dialog-subtitle">The cancellation is recorded in SpacetimeDB. Agents must observe it and stop through their own control system; this is not an emergency stop.</p><div class="tk-cancel-target" id="tk-cancel-target"></div><label class="field-label" for="tk-cancel-reason">Reason</label><textarea id="tk-cancel-reason" rows="3" maxlength="1000" minlength="3" required placeholder="Explain why this task should stop."></textarea><div id="tk-cancel-error" class="tk-error" role="alert" hidden></div><div class="dialog-footer"><button class="button" type="button" id="tk-keep-task">Keep task</button><button class="button primary" id="tk-confirm-cancel" type="submit">Cancel task</button></div></form></dialog>
    <dialog id="tk-prepare-dialog" aria-labelledby="tk-prepare-title"><div class="dialog-heading"><div class="eyebrow">Prepare simulation models</div><button type="button" class="close-button" id="tk-close-prepare" aria-label="Close simulation setup">×</button></div><h2 id="tk-prepare-title">Prepare the two-arm workcell.</h2><p class="dialog-subtitle">This replaces the active workspace pair and communication configuration with the simulation recipe. Existing model records and task history are retained. You will review and release both model configurations separately.</p><div id="tk-prepare-models" class="tk-prepare-models"></div><p class="tk-field-help">These are original task-trained, state-assisted neural controllers. This recipe does not load SmolVLA or X-VLA, and it does not control hardware.</p><div id="tk-prepare-error" class="tk-error" role="alert" hidden></div><div class="dialog-footer"><button class="button" id="tk-keep-pair">Keep current pair</button><button class="button primary" id="tk-confirm-prepare">Save pair & review releases</button></div></dialog>`;
  }
  function mount(config) {
    options = config; el("tasks-view").innerHTML = skeleton();
    el("tk-refresh").addEventListener("click", () => void refresh());
    el("tk-configure").addEventListener("click", options.onWorkspace);
    el("tk-create-form").addEventListener("submit", createTask);
    el("tk-create-form").addEventListener("input", () => { createRequest = null; el("tk-create-error").hidden = true; });
    el("tk-target").addEventListener("change", renderReadiness);
    el("tk-filter").addEventListener("change", () => { filter = el("tk-filter").value; const visible = matchingTasks(); if (!visible.some((task) => task.id === selected)) { selected = visible[0]?.id || null; detail = null; if (selected) void selectTask(selected); else renderDetail(); } renderList(); });
    el("tk-task-list").addEventListener("click", (event) => { const button = event.target.closest("[data-task]"); if (button) void selectTask(button.dataset.task); });
    el("tk-detail").addEventListener("click", (event) => { if (event.target.closest("[data-cancel-task]")) openCancellation(); });
    el("tk-cancel-form").addEventListener("submit", cancelTask);
    for (const id of ["tk-close-cancel", "tk-keep-task"]) el(id).addEventListener("click", () => el("tk-cancel-dialog").close());
    el("tk-cancel-dialog").addEventListener("close", () => { cancelTarget = null; el("tk-cancel-form").reset(); el("tk-cancel-target").replaceChildren(); });
    el("tk-worker-form").addEventListener("submit", createWorker);
    el("tk-worker-agent").addEventListener("change", renderReadiness);
    el("tk-workers").addEventListener("click", (event) => { const button = event.target.closest("[data-revoke-worker]"); if (button) void revokeWorker(button.dataset.revokeWorker, button); });
    el("tk-close-credential").addEventListener("click", () => el("tk-credential-dialog").close());
    el("tk-credential-dialog").addEventListener("close", clearCredential);
    el("tk-copy-credential").addEventListener("click", async () => { if (!credential) return; try { await navigator.clipboard.writeText(credential.token); options.notify("Worker credential copied"); } catch { el("tk-credential-value").focus(); el("tk-credential-value").select(); options.notify("Select and copy the credential"); } });
    el("tk-download-config").addEventListener("click", downloadConfig);
    el("tk-simulation").addEventListener("click", (event) => {
      if (event.target.closest("[data-sim-prepare]")) openRecipePreparation();
      if (event.target.closest("[data-sim-review]")) options.onWorkspace();
      if (event.target.closest("[data-sim-launch]")) void launchSimulation();
      const view = event.target.closest("[data-sim-view]");
      if (view) void viewSimulation(view.dataset.simView);
    });
    for (const id of ["tk-close-prepare", "tk-keep-pair"]) el(id).addEventListener("click", () => el("tk-prepare-dialog").close());
    el("tk-confirm-prepare").addEventListener("click", prepareSimulation);
    el("tk-prepare-dialog").addEventListener("close", () => { recipeReview = null; });
    document.addEventListener("visibilitychange", () => { clearTimeout(timer); if (active && !document.hidden) void refresh(); });
  }
  function clearCredential() { credential = null; el("tk-credential-value").value = ""; el("tk-worker-command").textContent = ""; el("tk-credential-scope").textContent = ""; }
  function reset() {
    hide(); generation++; request++; loading = false; tasks = []; workers = []; boot = null; selected = null; detail = null; loaded = false; workersLoaded = false; createRequest = null; creating = false; issuing = false; listSignature = ""; teamSignature = ""; workerSignature = ""; cancelTarget = null; cancelAttempts.clear();
    simulations = null; simulationError = ""; simulationRequest = null; simulationBusy = false; simulationPreparing = false; simulationSignature = ""; runSignature = ""; recipeReview = null;
    for (const id of ["tk-credential-dialog", "tk-cancel-dialog", "tk-prepare-dialog"]) if (el(id)?.open) el(id).close(); clearCredential();
    el("tk-create-form")?.reset(); el("tk-worker-form")?.reset(); el("tk-cancel-form")?.reset();
    for (const input of el("tasks-view").querySelectorAll("input,textarea,select,button")) input.disabled = false;
    for (const id of ["tk-task-list","tk-detail","tk-team-list","tk-workers"]) el(id)?.replaceChildren();
    for (const id of ["tk-error","tk-create-error","tk-worker-error","tk-cancel-error","tk-simulation-error","tk-prepare-error"]) if (el(id)) el(id).hidden = true;
    if (el("tk-simulation")) el("tk-simulation").innerHTML = '<p class="tk-muted">Checking the local simulation runtime…</p>';
    for (const id of ["tk-active-count","tk-complete-count","tk-model-count","tk-worker-count"]) if (el(id)) el(id).textContent = "—";
    if (el("tk-create")) el("tk-create").disabled = true;
    if (el("tk-refresh")) el("tk-refresh").disabled = false;
    filter = "all"; if (el("tk-filter")) el("tk-filter").value = "all";
  }
  function show() {
    active = true;
    const linkedTask = new URLSearchParams(location.search).get("task");
    if (!selected && /^goal-[a-f0-9]{16}$/.test(linkedTask || "")) selected = linkedTask;
    if (!loading) void refresh();
  }
  function hide() { active = false; clearTimeout(timer); }
  function current() { return boot || options.context(); }
  function manager() { return (current().permissions || []).includes("models.manage"); }
  function canCancel(task) { const userId = current().user?.id; return manager() || (!!userId && task?.created_by === userId); }
  function agents() { return current().workspace?.agents || []; }
  function release(agent) { return (current().models || []).find((model) => stable(model.spec) === stable(agent)); }
  function live(worker) { return !worker.revoked && worker.active !== false && worker.status === "online"; }
  function relevantWorkers(id) { return workers.filter((worker) => worker.agent_id === id && !worker.revoked && worker.active !== false); }
  function fresh(identity) { return identity === generation && options.authenticated(); }
  function errorAt(id, error) { if (!error.stale && options.authenticated()) { el(id).textContent = error.message; el(id).hidden = false; } }
  function schedule() { clearTimeout(timer); if (active && options.authenticated() && !document.hidden) timer = setTimeout(() => void refresh(), 5000); }
  async function refresh() {
    if (!active || !options.authenticated() || loading) return;
    const identity = generation, revision = ++request; loading = true; el("tk-refresh").disabled = true;
    try {
      const results = await Promise.allSettled([options.api("/api/tasks"), options.api("/api/workers"), options.api("/api/bootstrap"), options.api("/api/simulations")]);
      if (!fresh(identity) || revision !== request) return;
      const errors = [];
      if (results[0].status === "fulfilled") { tasks = results[0].value.tasks || []; loaded = true; } else errors.push(`Task state: ${results[0].reason.message}`);
      if (results[1].status === "fulfilled") { workers = results[1].value.workers || []; workersLoaded = true; } else errors.push(`Runtime connections: ${results[1].reason.message}`);
      if (results[2].status === "fulfilled") boot = results[2].value; else errors.push(`Model approvals: ${results[2].reason.message}`);
      if (results[3].status === "fulfilled") { simulations = results[3].value; simulationError = ""; }
      else simulationError = results[3].reason.message;
      el("tk-error").textContent = errors.join(" · "); el("tk-error").hidden = !errors.length;
      el("tk-backend-dot").classList.toggle("neutral", !!errors.length);
      el("tk-sync-label").textContent = errors.length ? "Some information could not refresh. Last loaded values remain visible." : `Task state refreshed ${new Date().toLocaleTimeString(undefined, {hour:"numeric",minute:"2-digit",second:"2-digit"})}`;
      renderTeam(); renderWorkers(); renderList(); renderOverview(); renderSimulation();
      if (selected && !matchingTasks().some((task) => task.id === selected)) { selected = null; detail = null; }
      if (!selected && matchingTasks().length) selected = matchingTasks()[0].id;
      if (selected) { try { const value = await options.api(`/api/tasks/${encodeURIComponent(selected)}`); if (fresh(identity) && revision === request && value.id === selected) { const changed = stable(detail) !== stable(value); detail = value; if (changed) renderDetail(); renderList(); } } catch (error) { if (fresh(identity)) errorAt("tk-error", error); } }
      else renderDetail();
      renderSimulationRun();
    } catch (error) { if (fresh(identity)) errorAt("tk-error", error); }
    finally { if (fresh(identity) && revision === request) { loading = false; el("tk-refresh").disabled = false; schedule(); } }
  }
  function renderOverview() {
    el("tk-active-count").textContent = loaded ? tasks.filter((t) => !terminal(t.status)).length : "—";
    el("tk-complete-count").textContent = loaded ? tasks.filter((t) => t.status === "succeeded").length : "—";
    el("tk-model-count").textContent = `${agents().filter((a) => release(a)?.status === "released").length} / ${agents().length}`;
    el("tk-worker-count").textContent = workersLoaded ? workers.filter(live).length : "—";
  }
  function renderTeam() {
    const signature = stable([agents(), current().models, workersLoaded, workers.map((w) => [w.id,w.status,w.active])]);
    if (signature === teamSignature) { renderReadiness(); return; } teamSignature = signature;
    el("tk-team-list").innerHTML = ["left","right"].map((id) => { const a = agents().find((agent) => agent.id === id), model = a && release(a), ready = model?.status === "released", connected = relevantWorkers(id).some(live); return `<article class="tk-agent"><span class="tk-agent-letter">${id === "left" ? "A" : "B"}</span><div><strong>${html(a?.name || "No model registered")}</strong><span>${html(a?.family?.toUpperCase() || "Configure a model to assign tasks")}</span><div class="tk-agent-meta"><span class="${ready ? "tk-approved" : "tk-pending"}">${ready ? `Approved · revision ${html(model.revision)}` : "Awaiting model release"}</span><span>${connected ? "Runtime connected" : workersLoaded ? "No connected runtime" : "Runtime status unavailable"}</span></div></div></article>`; }).join("");
    const target = el("tk-target").value, workerAgent = el("tk-worker-agent").value;
    el("tk-target").innerHTML = `<option value="both">Both agents · shared objective</option>${["left","right"].map((id) => `<option value="${id}">${html(agents().find((a) => a.id === id)?.name || (id === "left" ? "Agent A" : "Agent B"))}</option>`).join("")}`; el("tk-target").value = target;
    el("tk-worker-agent").innerHTML = ["left","right"].map((id) => `<option value="${id}">${html(agents().find((a) => a.id === id)?.name || (id === "left" ? "Agent A" : "Agent B"))}</option>`).join(""); el("tk-worker-agent").value = workerAgent;
    renderReadiness();
  }
  function renderReadiness() {
    const chosen = el("tk-target").value === "both" ? ["left","right"] : [el("tk-target").value];
    const ready = ["left","right"].every((id) => { const a = agents().find((agent) => agent.id === id); return a && release(a)?.status === "released"; });
    const online = chosen.every((id) => relevantWorkers(id).some(live));
    el("tk-readiness").textContent = !ready ? "A manager must release both models in the saved workspace before tasks can be assigned." : online ? "Approved models · connected runtimes can claim this task." : "Approved models · task will wait for a connected runtime.";
    el("tk-create").disabled = !ready || !loaded || !options.authenticated() || creating;
    el("tk-worker-manager").hidden = !manager(); el("tk-worker-employee").hidden = manager();
    const a = agents().find((item) => item.id === el("tk-worker-agent").value);
    el("tk-issue").disabled = !manager() || !ready || !a || release(a)?.status !== "released" || issuing;
  }
  function simulationRuns() {
    const all = [simulations?.active, ...(simulations?.runs || [])].filter(Boolean);
    return [...new Map(all.map((run) => [run.goal_id || run.task_id, run])).values()];
  }
  function renderSimulation() {
    const signature = stable([simulations, simulationError, manager(), simulationBusy, simulationPreparing, simulationRequest, current().workspace]);
    if (signature === simulationSignature) return;
    simulationSignature = signature;
    const recipe = simulations?.recipe, activeRun = simulations?.active;
    const verifying = activeRun && (activeRun.status === "verifying" || activeRun.result?.status === "verifying");
    const installed = recipe && stable(current().workspace) === stable(recipe.workspace);
    const ready = !!simulations?.release_ready, available = !!simulations?.available && !simulationError;
    const message = simulationError ? `Runtime check unavailable: ${simulationError}` : !simulations ? "Checking this installation…" : !available ? simulations.reason || "The simulation runtime is unavailable on this installation." : verifying ? "A fresh-world replay re-executes the recorded actions and checks the final geometry. This is verification, not a second neural run." : activeRun ? "Both controllers share one simulated workcell. Open the task to follow their progress." : !ready ? simulations.release_error || "Prepare and release the two task-trained models before starting." : "Ready for a fresh, measured rollout. The fixed recipe uses the same seed and configuration on every run.";
    const label = simulationError || !simulations ? "Status unavailable" : !available ? "Runtime unavailable" : verifying ? "Verifying recording" : activeRun ? "Workcell active" : ready ? "Ready to run" : "Model approval required";
    const last = simulationRuns()[0];
    el("tk-simulation").innerHTML = `<div class="tk-sim-scene" aria-label="Schematic of two arms sharing a cloth workcell"><svg viewBox="0 0 270 180" role="img" aria-label="Two-arm cloth workcell schematic"><path class="tk-sim-grid" d="M15 150 135 82 255 150M15 126 135 58 255 126M15 102 135 34 255 102M45 167 45 84M90 167 90 59M135 167 135 34M180 167 180 59M225 167 225 84"/><path class="tk-sim-table" d="m42 96 93-54 93 54-93 54z"/><path class="tk-sim-shirt" d="m105 80 17-10 13 5 13-5 17 10 14 20-17 10-9-9v26h-36v-26l-9 9-17-10z"/><path class="tk-sim-seam" d="M117 108h36M135 78v46"/><path class="tk-sim-arm" d="m33 120 11-39 27-19 24 27"/><path class="tk-sim-arm tk-sim-arm-b" d="m237 120-11-39-27-19-24 27"/><g class="tk-sim-joint"><circle cx="44" cy="81" r="7"/><circle cx="71" cy="62" r="7"/><circle cx="226" cy="81" r="7"/><circle cx="199" cy="62" r="7"/></g><path class="tk-sim-gripper" d="m90 89 1 10 9 1m80-11-1 10-9 1"/><path class="tk-sim-base" d="m19 119 14-8 14 8v10l-14 8-14-8zm204 0 14-8 14 8v10l-14 8-14-8z"/></svg><span>Shared MuJoCo workcell</span></div><div class="tk-sim-copy"><div class="tk-sim-eyebrow"><span class="tiny-square"></span> TWO-ARM SIMULATION <span class="tk-sim-availability ${available ? "ready" : ""}">${html(label)}</span></div><h2>${html(recipe?.title || "One shirt. Two perspectives.")}</h2><p>Run a complete cloth-folding task with separate local observations, changing obstacles and delayed peer messages.</p><div class="tk-sim-stack"><span>Left · CNN controller</span><span>Right · attention controller</span><span>Qwen phase review</span><span>SpacetimeDB</span></div><p class="tk-sim-boundary">Task-trained, state-assisted policies · measured simulation · no hardware control</p><div class="tk-sim-controls"><div class="tk-sim-buttons">${manager() && localSimulationHost() && recipe ? `${!ready ? `<button class="button" ${simulationPreparing || !!activeRun ? "disabled" : ""} ${installed ? "data-sim-review" : "data-sim-prepare"}>${installed ? "Review model releases" : "Prepare model pair"}</button>` : ""}<button class="button primary" data-sim-launch ${!ready || !available || !!activeRun || simulationBusy || simulationPreparing ? "disabled" : ""}>${simulationBusy ? "Starting…" : simulationRequest ? "Retry same launch" : "Run folding task"} ${icon("arrow")}</button>` : ""}${activeRun ? `<button class="button" data-sim-view="${html(activeRun.goal_id || activeRun.task_id)}">View live task ${icon("arrow")}</button>` : last ? `<button class="text-button" data-sim-view="${html(last.goal_id || last.task_id)}">View latest result →</button>` : ""}</div><p>${html(message)}${!manager() ? " A manager can prepare and launch this recipe." : !localSimulationHost() ? " Use an outbound runtime for this hosted workspace." : ""}</p></div></div>`;
  }
  function openRecipePreparation() {
    if (!manager() || !localSimulationHost() || !simulations?.recipe || simulationPreparing) return;
    recipeReview = structuredClone(simulations.recipe);
    el("tk-prepare-models").innerHTML = (recipeReview.workspace?.agents || []).map((agent) => `<div><span class="tk-agent-letter">${agent.id === "left" ? "A" : "B"}</span><p><strong>${html(agent.name)}</strong><span>${html(agent.id === "left" ? "CNN task controller" : "Attention task controller")}</span><code>${html(agent.source?.ref || "")}</code></p></div>`).join("");
    el("tk-prepare-error").hidden = true; el("tk-confirm-prepare").disabled = false; el("tk-prepare-dialog").showModal();
  }
  async function prepareSimulation() {
    if (!recipeReview || simulationPreparing || !manager()) return;
    const identity = generation, recipe = structuredClone(recipeReview);
    simulationPreparing = true; el("tk-confirm-prepare").disabled = true; el("tk-prepare-error").hidden = true;
    for (const id of ["tk-close-prepare", "tk-keep-pair"]) el(id).disabled = true;
    try {
      const workspace = await options.api("/api/workspace", recipe.workspace);
      if (!fresh(identity)) return;
      boot = {...current(), workspace};
      el("tk-prepare-dialog").close();
      await options.onWorkspace(workspace);
      if (fresh(identity)) options.notify("Simulation pair saved. Review and release both exact model configurations.");
    } catch (error) { if (fresh(identity)) errorAt("tk-prepare-error", error); }
    finally {
      if (fresh(identity)) { simulationPreparing = false; for (const id of ["tk-confirm-prepare", "tk-close-prepare", "tk-keep-pair"]) el(id).disabled = false; renderSimulation(); }
    }
  }
  async function launchSimulation() {
    if (!manager() || !simulations?.available || !simulations.release_ready || simulations.active || simulationBusy || simulationError) return;
    const identity = generation;
    simulationRequest ||= crypto.randomUUID(); simulationBusy = true; el("tk-simulation-error").hidden = true; renderSimulation();
    try {
      const run = await options.api("/api/simulations", {request_id:simulationRequest});
      if (!fresh(identity)) return;
      simulations = {...simulations, active:simulationTerminal(run.status) ? null : run, runs:[run, ...(simulations.runs || []).filter((item) => item.goal_id !== run.goal_id)]};
      simulationRequest = null;
      options.notify(simulationTerminal(run.status) ? "Opened the recorded simulation result." : "Simulation task assigned. The local team runtime is starting.");
      await viewSimulation(run.goal_id || run.task_id);
    } catch (error) {
      if (fresh(identity)) errorAt("tk-simulation-error", new Error(`${error.message} The launch request is retained; retrying it does not intentionally start a second run.`));
    } finally { if (fresh(identity)) { simulationBusy = false; renderSimulation(); schedule(); } }
  }
  async function viewSimulation(id) {
    if (!/^goal-[a-f0-9]{16}$/.test(id || "")) return;
    filter = "all"; el("tk-filter").value = "all";
    await selectTask(id);
    if (selected === id && active && options.authenticated()) el("tk-detail").scrollIntoView({behavior:"smooth", block:"start"});
  }
  function simulationArtifact(run, name) {
    if (!["overview.mp4", "left.mp4", "right.mp4", "summary.json", "verification.json"].includes(name)) return null;
    const raw = run.artifacts?.[name], id = run.goal_id || run.task_id;
    if (typeof raw !== "string" || !/^goal-[a-f0-9]{16}$/.test(id || "")) return null;
    try {
      const url = new URL(raw, location.origin);
      if (url.origin !== location.origin || url.username || url.password || url.search || url.hash || !url.pathname.startsWith(`/api/simulations/${id}/`) || !url.pathname.endsWith(`/${name}`)) return null;
      return url.pathname;
    } catch { return null; }
  }
  function renderSimulationRun() {
    const host = el("tk-simulation-run"); if (!host || !detail || selected !== detail.id) return;
    const run = simulationRuns().find((item) => (item.goal_id || item.task_id) === detail.id);
    const signature = stable([detail.id, run, simulationError]);
    if (signature === runSignature && host.childElementCount) return;
    runSignature = signature;
    if (!run) { host.hidden = true; host.replaceChildren(); return; }
    host.hidden = false;
    const progress = run.progress || {}, result = run.result || {}, metrics = result.metrics && Object.keys(result.metrics).length ? result.metrics : progress.metrics || {}, phases = progress.phases || {};
    const reason = result.reason || run.reason;
    const status = run.status === "running" && result.status === "verifying" ? "verifying" : run.status;
    const length = (value) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)} cm` : "—";
    const worst = (value) => value && typeof value === "object" && Object.values(value).length && Object.values(value).every((item) => typeof item === "number" && Number.isFinite(item)) ? Math.max(...Object.values(value)) : null;
    const metricItems = [["Hem alignment · worst arm", length(worst(metrics.hem_alignment_m))], ["Sleeve overhang · worst arm", length(worst(metrics.sleeve_overhang_m))], ["Cloth height · maximum", length(metrics.max_height_m)], ["Geometry check", metrics.fold_geometry_ok === true ? "Passed" : metrics.fold_geometry_ok === false ? "Not passed" : "Pending"]];
    const video = simulationArtifact(run, "overview.mp4"), priorVideo = host.querySelector("video"), priorSource = priorVideo?.getAttribute("src");
    const links = ["summary.json", "verification.json", "left.mp4", "right.mp4"].map((name) => ({name,url:simulationArtifact(run,name)})).filter((item) => item.url);
    const ended = simulationTerminal(status), statusClass = status === "succeeded" ? "succeeded" : ["failed","interrupted"].includes(status) ? "failed" : status === "cancelled" ? "cancelled" : "running";
    host.innerHTML = `<div class="tk-sim-result-heading"><div><span class="section-overline">Measured simulation</span><h4>Two arms. One recorded run.</h4></div><span class="tk-status ${statusClass}"><i></i>${html(simulationLabels[status] || status)}</span></div>${video ? `<video class="tk-sim-video" controls preload="metadata" playsinline src="${html(video)}" aria-label="Recorded MuJoCo cloth-folding rollout">Your browser does not support the recorded video.</video>` : `<div class="tk-sim-video-pending">${icon(ended ? "clock" : "link")}<span>${ended ? "No overview recording is available for this run." : status === "verifying" ? "Re-executing saved actions in a fresh world to verify geometry. This does not run the neural policies again." : "The replay will appear after the recording is finalized."}</span></div>`}<div class="tk-sim-phases">${["left","right"].map((arm) => `<div><span>${arm === "left" ? "A · CNN" : "B · attention"}</span><strong>${html(String(phases[arm] || (ended ? "See recorded evidence" : "Awaiting runtime")).replaceAll("_", " "))}</strong>${progress.hold_reasons?.[arm] ? `<small>${html(progress.hold_reasons[arm])}</small>` : ""}</div>`).join("")}</div><dl class="tk-sim-metrics">${metricItems.map(([label,value]) => `<div><dt>${html(label)}</dt><dd>${html(value)}</dd></div>`).join("")}</dl>${Number.isInteger(progress.tick) ? `<p class="tk-sim-tick">SIMULATION TICK ${html(progress.tick)}${Number.isInteger(progress.total_phases) ? ` · ${html(progress.total_phases)} PHASES PER ARM` : ""}</p>` : ""}${reason ? `<p class="tk-sim-reason">${html(reason)}</p>` : ""}${links.length ? `<div class="tk-sim-evidence">${links.map(({name,url}) => `<a href="${html(url)}" target="_blank" rel="noopener">${html({"summary.json":"Run summary", "verification.json":"Verification record", "left.mp4":"Agent A camera", "right.mp4":"Agent B camera"}[name])} ↗</a>`).join("")}</div>` : ""}<details class="tk-sim-verification"><summary>Controller provenance & verification</summary><p>Original task-trained, state-assisted neural controllers with Qwen phase review. Geometry is measured in the simulator; task completion is reported by the team runtime. This is not a foundation-VLA fold or proof of hardware performance.</p>${result.verification ? `<pre>${html(pretty(result.verification))}</pre>` : '<p>The final verification record is not yet available.</p>'}${result.controller_provenance ? `<pre>${html(pretty(result.controller_provenance))}</pre>` : ""}</details>${simulationError ? '<p class="tk-sim-reason">Live simulation metadata could not refresh. These are the last loaded values.</p>' : ""}`;
    if (priorVideo && video && priorSource === video) host.querySelector("video")?.replaceWith(priorVideo);
  }
  function matchingTasks() { return tasks.filter((task) => filter === "all" || (filter === "active" ? !terminal(task.status) : filter === "attention" ? ["blocked","failed"].includes(task.status) : task.status === filter)); }
  function renderList() {
    const signature = stable([tasks,filter,selected,loaded]); if (signature === listSignature) return; listSignature = signature;
    const visible = matchingTasks();
    if (!visible.length) { el("tk-task-list").innerHTML = `<div class="tk-empty">${icon("check")}<strong>${!loaded ? "Task history is unavailable" : tasks.length ? "No matching tasks" : "Your first task starts here."}</strong><p>${!loaded ? "Refresh to reconnect to the task store." : tasks.length ? "Choose another filter to see your task history." : "Define an objective above. Once assigned, its progress and results will appear here."}</p></div>`; return; }
    el("tk-task-list").innerHTML = `<div class="tk-list-caption"><span>${visible.length} TASK${visible.length === 1 ? "" : "S"}</span><span>Latest first</span></div>${visible.map((task) => `<button class="tk-task-row ${selected === task.id ? "selected" : ""}" data-task="${html(task.id)}" aria-pressed="${selected === task.id}"><div>${pill(task.status)}<time>${html(time(task.created_at))}</time></div><strong>${html(task.title || task.objective)}</strong><p>${html(task.objective)}</p><span class="tk-task-target">${icon("users")}${html(Object.values(task.assignments || {}).map((a) => a.name || a.agent_id).join(" + ") || (task.target === "both" ? "Both agents" : task.target))}<span>View task →</span></span></button>`).join("")}`;
  }
  async function selectTask(id) {
    const identity = generation; selected = id; detail = null; renderList(); el("tk-detail").innerHTML = `<div class="tk-empty">${icon("clock")}<strong>Loading task</strong><p>Reading the recorded instructions and agent feedback.</p></div>`;
    try { const value = await options.api(`/api/tasks/${encodeURIComponent(id)}`); if (fresh(identity) && selected === id) { detail = value; renderDetail(); } }
    catch (error) { if (fresh(identity) && selected === id) { el("tk-detail").innerHTML = `<div class="tk-empty">${icon("clock")}<strong>Task could not load</strong><p>${html(error.message)}</p></div>`; } }
  }
  function renderDetail() {
    if (!detail) { el("tk-detail").innerHTML = `<div class="tk-empty">${icon("check")}<strong>The work, step by step.</strong><p>Select a task to see instructions, agent updates, and the recorded outcome.</p></div>`; return; }
    const previousSimulationPanel = el("tk-simulation-run");
    const t = detail, expanded = new Set([...el("tk-detail").querySelectorAll("details[open][data-tk-detail]")].map((node) => node.dataset.tkDetail));
    el("tk-detail").innerHTML = `<div class="tk-detail-heading"><div>${pill(t.status)}<h3>${html(t.title || "Task objective")}</h3></div>${!terminal(t.status) && canCancel(t) ? '<button class="text-button" data-cancel-task>Cancel task</button>' : ""}</div>${t.pending_update ? '<div class="tk-error">An update is awaiting confirmation. The recorded state remains authoritative until the commit is verified.</div>' : ""}<p class="tk-objective">${html(t.objective)}</p><div class="tk-criteria"><strong>Success criteria</strong><ul>${(t.success_criteria || []).map((c) => `<li>${html(c)}</li>`).join("")}</ul></div><div class="tk-detail-meta"><span>${icon("db")}SpacetimeDB · ${t.readback_verified ? "verified readback · " : ""}rev ${html(t.revision)}</span><span>${html(time(t.updated_at))}</span></div><section id="tk-simulation-run" class="tk-sim-result" aria-label="Simulation evidence" hidden></section><div class="tk-assignment-list">${Object.entries(t.assignments || {}).map(([id,a]) => { const progress = Number.isFinite(a.progress) ? Math.round(Math.max(0,Math.min(100,a.progress)) * 10) / 10 : null; return `<article class="tk-assignment"><div><span class="tk-agent-letter">${id === "left" ? "A" : "B"}</span><strong>${html(a.name || id)}</strong>${pill(a.status)}</div>${a.worker_id ? `<div class="tk-assignment-runtime">${html(workers.find((w) => w.id === a.worker_id)?.name || a.worker_id)} · ${html(a.execution_kind === "observation" ? "Observation only" : a.execution_kind === "simulation" ? "Simulation" : "External runtime")} · release ${html(a.release_revision)}</div>` : ""}${progress !== null ? `<label class="tk-progress">Agent-reported progress <span>${progress}%</span><progress max="100" value="${progress}">${progress}%</progress></label>` : ""}<p>${html(a.summary || "Waiting for the assigned runtime to claim this objective.")}</p><details data-tk-detail="instruction-${html(id)}"><summary>Instructions sent to this agent</summary><pre>${html(pretty(a.instruction || t.objective))}</pre></details>${a.evidence && (typeof a.evidence === "string" ? a.evidence.trim() : Object.keys(a.evidence).length) ? `<details data-tk-detail="evidence-${html(id)}"><summary>Agent-reported evidence</summary><pre>${html(pretty(a.evidence))}</pre></details>` : ""}</article>`; }).join("")}</div><div class="tk-events"><strong>Activity log</strong>${(t.events || []).slice(-12).reverse().map((event) => `<div class="tk-event"><i></i><div><strong>${html(event.summary || String(event.type || "Task updated").replaceAll("_"," "))}</strong><span>${html(event.agent_id ? `${event.agent_id === "left" ? "Agent A" : "Agent B"} · ` : "")}${html(time(event.at, true))} · revision ${html(event.revision)}</span></div></div>`).join("") || '<p class="tk-muted">No events recorded yet.</p>'}</div>${terminal(t.status) ? `<div class="tk-outcome-note">${icon("shield")}<span>${t.status === "succeeded" ? "All assigned agents reported success. Physical completion has not been independently verified." : t.status === "cancelled" ? "Cancellation is recorded. Confirm that your runtime has stopped before servicing the workcell." : "The runtime reported a failed outcome. Review its feedback before assigning another task."}</span></div>` : ""}`;
    if (previousSimulationPanel && previousSimulationPanel.dataset.task === t.id) el("tk-simulation-run").replaceWith(previousSimulationPanel);
    el("tk-simulation-run").dataset.task = t.id;
    renderSimulationRun();
    for (const node of el("tk-detail").querySelectorAll("details[data-tk-detail]")) node.open = expanded.has(node.dataset.tkDetail);
  }
  async function createTask(event) {
    event.preventDefault(); if (el("tk-create").disabled) return;
    const identity = generation, payload = {request_id: createRequest || (createRequest = crypto.randomUUID()), objective: el("tk-objective").value.trim(), title: el("tk-title").value.trim(), success_criteria: el("tk-criteria").value.split("\n").map((v) => v.trim()).filter(Boolean), target: el("tk-target").value};
    if (!payload.success_criteria.length || payload.success_criteria.length > 8 || payload.success_criteria.some((c) => c.length > 500)) { errorAt("tk-create-error", new Error("Add 1–8 success criteria, with at most 500 characters per line.")); return; }
    creating = true; el("tk-create").disabled = true; el("tk-create-error").hidden = true;
    for (const input of el("tk-create-form").querySelectorAll("input,select,textarea")) input.disabled = true;
    try { const value = await options.api("/api/tasks", payload); if (!fresh(identity)) return; selected = value.id; detail = value; tasks = [value, ...tasks.filter((t) => t.id !== value.id)]; createRequest = null; el("tk-create-form").reset(); filter = "all"; el("tk-filter").value = "all"; renderDetail(); renderList(); renderOverview(); options.notify("Task assigned. Waiting for the selected agents to report progress."); el("tk-detail").scrollIntoView({behavior:"smooth",block:"start"}); }
    catch (error) { if (fresh(identity)) errorAt("tk-create-error", error); }
    finally { if (fresh(identity)) { creating = false; for (const input of el("tk-create-form").querySelectorAll("input,select,textarea")) input.disabled = false; renderReadiness(); } }
  }
  function openCancellation() {
    if (!detail || terminal(detail.status) || !canCancel(detail)) return;
    // The modal owns its task identity. Polling or filters may select another task.
    cancelTarget = cancelAttempts.get(detail.id) || {id:detail.id, snapshot:structuredClone(detail), phase:"editing", payload:null, busy:false};
    cancelAttempts.set(cancelTarget.id, cancelTarget);
    el("tk-cancel-form").reset(); el("tk-cancel-reason").value = cancelTarget.payload?.reason || "";
    el("tk-cancel-error").hidden = true; renderCancellation(); el("tk-cancel-dialog").showModal();
  }
  function renderCancellation() {
    const attempt = cancelTarget; if (!attempt) return;
    const task = attempt.snapshot;
    el("tk-cancel-title").textContent = "Cancel this task?";
    el("tk-cancel-target").innerHTML = `<strong>${html(task.title || task.objective)}</strong><span>${html(task.id)} · revision ${html(task.revision)} · ${html(labels[task.status] || task.status)}</span><p>${html(task.objective)}</p>`;
    el("tk-cancel-reason").readOnly = attempt.busy || attempt.phase === "pending" || ["terminal","denied"].includes(attempt.phase);
    el("tk-confirm-cancel").disabled = attempt.busy || ["terminal","denied"].includes(attempt.phase);
    el("tk-confirm-cancel").textContent = attempt.phase === "review" ? `Cancel revision ${task.revision}` : attempt.phase === "pending" ? "Retry same cancellation" : "Cancel task";
    el("tk-keep-task").textContent = attempt.payload ? "Close" : "Keep task";
  }
  function updateRecordedTask(value) {
    tasks = tasks.map((task) => task.id === value.id ? value : task);
    if (selected === value.id) { detail = value; renderDetail(); }
    renderList(); renderOverview();
  }
  async function cancelTask(event) {
    event.preventDefault(); const attempt = cancelTarget;
    if (!attempt || attempt.busy || ["terminal","denied"].includes(attempt.phase)) return;
    const identity = generation;
    if (!attempt.payload || attempt.phase === "review") {
      // A revised cancellation requires another explicit click after conflict review.
      attempt.payload = {request_id:crypto.randomUUID(), expected_revision:attempt.snapshot.revision, reason:el("tk-cancel-reason").value.trim()};
    }
    attempt.phase = "pending"; attempt.busy = true; renderCancellation(); el("tk-cancel-error").hidden = true;
    try {
      const value = await options.api(`/api/tasks/${encodeURIComponent(attempt.id)}/cancel`, attempt.payload);
      if (!fresh(identity)) return;
      cancelAttempts.delete(attempt.id); updateRecordedTask(value);
      if (cancelTarget === attempt) el("tk-cancel-dialog").close();
      options.notify("Task cancellation recorded");
    } catch (error) {
      if (!fresh(identity)) return;
      if (error.status === 409) {
        try {
          const latest = await options.api(`/api/tasks/${encodeURIComponent(attempt.id)}`);
          if (!fresh(identity)) return;
          attempt.snapshot = latest; updateRecordedTask(latest);
          if (terminal(latest.status)) { attempt.phase = "terminal"; cancelAttempts.delete(attempt.id); }
          else if (!latest.pending_update) attempt.phase = "review";
          if (cancelTarget === attempt) errorAt("tk-cancel-error", new Error(terminal(latest.status) ? `This task is now ${labels[latest.status] || latest.status}. No further cancellation will be sent.` : latest.pending_update ? "A task update is awaiting confirmation. Retry sends the exact same cancellation request; no action is automatically repeated." : `The task changed to revision ${latest.revision}. Review the task above, then explicitly confirm cancellation at this revision.`));
        } catch (readError) {
          if (fresh(identity) && cancelTarget === attempt) errorAt("tk-cancel-error", new Error(`The original task could not be refreshed. Its cancellation request has been retained unchanged. ${readError.message}`));
        }
      } else {
        if (error.status === 400) { attempt.phase = "editing"; attempt.payload = null; }
        if (error.status === 403 || error.status === 404) attempt.phase = "denied";
        if (cancelTarget === attempt) errorAt("tk-cancel-error", new Error(`${error.message}${attempt.phase === "pending" ? " The outcome is not confirmed. Retry sends this exact request for the same task." : ""}`));
      }
    } finally {
      attempt.busy = false;
      if (fresh(identity) && cancelTarget === attempt) renderCancellation();
    }
  }
  function renderWorkers() {
    const signature = stable([workers, workersLoaded, manager(), agents()]); if (signature === workerSignature) { renderReadiness(); return; } workerSignature = signature;
    el("tk-workers").innerHTML = workers.length ? workers.map((worker) => `<div class="tk-worker"><span class="tk-worker-icon">${icon("code")}</span><div><strong>${html(worker.name || worker.id)}</strong><span>${html(agents().find((a) => a.id === worker.agent_id)?.name || worker.agent_id)} · ${worker.last_seen ? `Last seen ${html(time(worker.last_seen))}` : "No heartbeat received"}</span></div><span class="tk-worker-state ${live(worker) ? "connected" : ""}"><i></i>${({online:"Connected",offline:"Offline",never_connected:"Awaiting connection",revoked:"Revoked",expired:"Expired",release_blocked:"Release changed"}[worker.status] || "Not connected")}</span>${manager() && worker.status !== "revoked" && worker.active !== false ? `<button class="text-button" data-revoke-worker="${html(worker.id)}">Revoke</button>` : ""}</div>`).join("") : `<div class="tk-runtime-empty">${icon("link")}<div><strong>${workersLoaded ? "No runtimes connected yet." : "Runtime status is unavailable."}</strong><p>${workersLoaded ? "Create a scoped connection, then start your agent worker. Tasks can wait safely in the queue." : "Refresh to check the workspace’s registered connections."}</p></div></div>`;
    renderReadiness();
  }
  async function createWorker(event) {
    event.preventDefault(); if (!manager() || issuing) return; const identity = generation;
    issuing = true; el("tk-issue").disabled = true; el("tk-worker-error").hidden = true;
    try { const value = await options.api("/api/workers", {name:el("tk-worker-name").value.trim(), agent_id:el("tk-worker-agent").value, execution_kind:el("tk-worker-kind").value}); if (!fresh(identity)) return; credential = value; el("tk-credential-value").value = value.token; el("tk-credential-title").textContent = `Connect ${value.worker?.name || "your agent"}.`; el("tk-credential-scope").textContent = `${value.worker?.execution_kind === "simulation" ? "Simulation" : "Observation"} · ${value.worker?.agent_id === "right" ? "Agent B" : "Agent A"} · revision ${value.worker?.release_revision ?? "—"} · expires ${time(value.expires_at || value.worker?.expires_at)}`; el("tk-worker-command").textContent = `chmod 600 ./worker-token\npython -m relay_lab.studio.worker \\\n  --url ${JSON.stringify(location.origin)} \\\n  --token-file ./worker-token \\\n  --handler your_module:handle`; el("tk-credential-dialog").showModal(); el("tk-worker-form").reset(); if (value.worker) { workers.unshift(value.worker); renderWorkers(); } options.notify("Runtime connection created. Save the credential before closing."); }
    catch (error) { if (fresh(identity)) errorAt("tk-worker-error", error); }
    finally { if (fresh(identity)) { issuing = false; renderReadiness(); } }
  }
  async function revokeWorker(id, button) {
    const identity = generation; button.disabled = true;
    try { await options.api(`/api/workers/${encodeURIComponent(id)}/revoke`, {}); if (!fresh(identity)) return; workers = workers.map((w) => w.id === id ? {...w,revoked:true,active:false,status:"revoked"} : w); renderWorkers(); renderOverview(); options.notify("Runtime credential revoked"); }
    catch (error) { if (fresh(identity)) { errorAt("tk-worker-error", error); button.disabled = false; } }
  }
  function downloadConfig() {
    if (!credential) return;
    const link = document.createElement("a"), url = URL.createObjectURL(new Blob([credential.token + "\n"], {type:"text/plain"})); link.href = url; link.download = "worker-token"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); options.notify("Token file downloaded. Move it beside your runtime and keep it private.");
  }
  window.RelayTasks = {mount, show, hide, reset};
})();
