"use strict";

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const families = {smolvla: "SmolVLA", xvla: "X-VLA", openvla: "OpenVLA", groot: "GR00T", custom: "Custom VLA"};
const robotNames = {arm: "Robot arm", mobile: "Drive unit", humanoid: "Humanoid", drone: "Drone", custom: "Custom"};
const modes = {
  mediated: {title: "LLM mediator", description: "Turns each local report into useful peer context.", route: "A’s report → Qwen → B<br>B’s report → Qwen → A", label: "PEER CONTEXT"},
  orchestrated: {title: "LLM orchestrator", description: "Reads both reports and assigns the next instructions.", route: "A + B → <b>Qwen planner</b><br>Individual instructions → A + B", label: "SHARED PLANNING"},
  latent: {title: "Learned latents", description: "Exchanges features through trained CoLA-inspired adapters.", route: "Native features → <b>64D message</b><br>Peer decoder → 32D context", label: "LEARNED INTERFACE"},
  hybrid: {title: "Hybrid", description: "Delivers Qwen instructions alongside learned messages.", route: "<b>Qwen instructions</b> + <b>64D latents</b><br>Two parallel paths to each receiver", label: "TWO CHANNELS"},
};
const robotPaths = {
  arm: `<ellipse cx="65" cy="85" rx="33" ry="5" fill="currentColor" opacity=".07"/><path d="M39 81h43l-4-8H44l-5 8Z" fill="#e5ebde"/><path d="M48 73V57l14-20 16 8-15 19v9" stroke-width="10" stroke="#e5ebde"/><path d="m53 58 15-20 19-20 10 9-20 22-15 14M51 72V60m11 13V61"/><path d="m88 19 12 8 9-8-12-8-9 8Z" fill="#e6ecdf"/><path d="m108 19 8 6-4 8m-10-9 6 5-4 7M49 73h16"/><circle cx="57" cy="58" r="8" fill="#f9fbf6"/><circle cx="57" cy="58" r="3"/><circle cx="72" cy="40" r="8" fill="#f7faf2"/><circle cx="72" cy="40" r="3"/><circle cx="92" cy="22" r="6" fill="#f7faf2"/><path d="M44 81v3h34v-3M97 15l7-4 9 5-5 4"/>`,
  mobile: `<ellipse cx="65" cy="83" rx="39" ry="5" fill="currentColor" opacity=".07"/><path d="m25 55 25-12 54 13-25 15-54-16Z" fill="#e6ecde"/><path d="M25 55v18l54 15 25-16V56M79 71v17M42 47V31l35-10 17 7v25M42 31l17 6 35-9M59 37v14"/><path d="m45 49 32 8 14-7"/><circle cx="40" cy="75" r="5" fill="#f9fbf5"/><circle cx="91" cy="78" r="5" fill="#f9fbf5"/><path d="m61 65 11 3M50 28V18m-5-1h11"/><circle cx="51" cy="15" r="3"/>`,
  humanoid: `<ellipse cx="65" cy="88" rx="28" ry="4" fill="currentColor" opacity=".07"/><rect x="53" y="8" width="24" height="17" rx="7" fill="#edf2e5"/><path d="M58 16h14M61 25v5m8-5v5M51 31h28l-4 26H55l-4-26Z" fill="#e7edde"/><path d="m51 34-10 7-3 20m41-27 10 7 3 20M57 57l-2 18-6 11h13l3-27m8-2 3 18 5 11H68l-3-27M37 62l-2 8m5-8 2 7M91 62l-2 7m5-7 3 8"/><circle cx="44" cy="40" r="5" fill="#f7faf2"/><circle cx="86" cy="40" r="5" fill="#f7faf2"/><path d="M58 36h14M62 40h6M58 54h14"/>`,
  drone: `<ellipse cx="65" cy="83" rx="33" ry="4" fill="currentColor" opacity=".07"/><path d="m48 38 34 17M48 55l34-17" stroke-width="7" stroke="#dce6d1"/><path d="M47 40 27 29m57 11 20-11M47 54 27 65m57-11 20 11" stroke-width="3"/><ellipse cx="27" cy="26" rx="19" ry="5"/><ellipse cx="104" cy="26" rx="19" ry="5"/><ellipse cx="27" cy="62" rx="19" ry="5"/><ellipse cx="104" cy="62" rx="19" ry="5"/><path d="M27 26v8m77-8v8M27 62v8m77-8v8M50 40l15-7 15 7v15l-15 7-15-7V40Z" fill="#edf2e5"/><path d="m50 40 15 7 15-7M65 47v15M56 60l-4 14m22-14 4 14m-29 0h7m18 0h7"/><circle cx="65" cy="56" r="3"/>`,
  custom: `<path d="m36 32 29-14 29 14v35L65 81 36 67V32Zm0 0 29 15 29-15M65 47v34" fill="#eef3e7"/><path d="m50 39 29-14M50 39v12M54 61l11 6 11-6"/><circle cx="65" cy="49" r="4" fill="#f9fcf3"/>`,
};
const robot = (kind) => `<svg viewBox="0 0 130 96" aria-hidden="true">${robotPaths[kind] || robotPaths.custom}</svg>`;
let state = {workspace: null, example: null, capabilities: {}, token: "", session: null, busy: false, dirty: false, upload: null, robot: "arm", results: []};
let toastTimer, polling, historySignature = "", activeUpload, diagnosticRevision = 0, authEpoch = 0;
Object.assign(state, {authenticated: false, user: null, permissions: [], models: [], team: [], requests: [], savedWorkspace: null, release_ready: false, sessionReleaseReady: false, authMode: "login", health: null, diagnostics: null, sessions: [], history: null, selectedMessage: null, readOnly: false, failed: false});

async function api(path, body) {
  const epoch = authEpoch;
  let response, data;
  try {
    response = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-Relay-Token": state.token}, body: JSON.stringify(body)});
    data = await response.json();
  } catch (error) { if (epoch !== authEpoch) error.stale = true; throw error; }
  if (epoch !== authEpoch) {
    const error = new Error("This request belongs to an earlier sign-in session."); error.stale = true; throw error;
  }
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith("/api/auth/")) {
      if (state.profileMode) showAccessRecovery("Your profile session changed. Retry to reopen the workspace.");
      else showAuth("login", "Your session expired or access changed. Please sign in again.");
    }
    const error = new Error(data.error || (response.status === 403 ? "Your account does not have permission for this action." : `Request failed (${response.status})`));
    error.status = response.status; throw error;
  }
  return data;
}
function notify(message, success = false) { if (!state.authenticated) return; $("notice").textContent = message; $("notice").className = `notice${success ? " success" : ""}`; $("notice").hidden = false; }
function toast(message) { if (!state.authenticated) return; clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false; toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500); }
function clearSession() {
  historySignature = "";
  state.session = null; state.sessionReleaseReady = false; state.results = []; state.history = null; state.selectedMessage = null; state.readOnly = false; state.failed = false; state.diagnostics = null; clearInterval(polling);
  $("receiver-panel").hidden = true;
  void refreshDiagnostics();
  $("stream-events").replaceChildren(); $("stream-count").textContent = "0 PACKETS";
  $("stream-empty").hidden = false; $("stream-metrics").hidden = true;
}
function changed() { clearSession(); state.dirty = true; $("notice").hidden = true; render(); }
const experimentsEnabled = () => state.capabilities.experiments_enabled !== false;
function isExample() {
  return experimentsEnabled() && !!state.example && state.workspace.agents.length === 2 && state.workspace.agents.every((a, i) => {
    const b = state.example.agents[i];
    return a.id === b.id && a.robot === "arm" && a.family === b.family && JSON.stringify(a.source) === JSON.stringify(b.source);
  });
}
function applyDeploymentProfile() {
  const research = experimentsEnabled();
  document.body.classList.toggle("control-plane", !research);
  $("deployment-guide").hidden = research;
  const copy = (selector, text) => {
    const node = document.querySelector(selector); if (!node) return;
    if (node.dataset.researchCopy === undefined) node.dataset.researchCopy = node.innerHTML;
    if (research) node.innerHTML = node.dataset.researchCopy; else node.textContent = text;
  };
  copy("#nav-team > span", "Models & channels");
  copy(".page-heading h1", "Workspace");
  copy(".page-heading p", "Register your policies, approve revisions, and connect the runtimes that do the work.");
  copy(".page-heading .preview-tag", "PRIVATE WORKSPACE");
  copy(".release-heading p", "Approve each model configuration before assigning goals to its runtime.");
  copy(".release-footnote > span:last-child", "EXACT CONFIGURATION · MANAGER APPROVED");
  copy(".canvas-caption .mono", "YOUR POLICIES / SHARED TASK STATE");
  copy(".canvas-coordinate", "CUSTOMER WORKSPACE");
  copy(".canvas-safe", "INFERENCE ON AGENT HOSTS");
  copy(".connection-summary > div > span", "Objective → scoped assignment → observed outcome");
  copy(".bridge-section h2", "Keep your models close to the work.");
  copy(".bridge-section p", "Your agent bridge connects outward to this workspace. It receives assigned goals and reports progress from your own policy or simulator.");
  copy(".bridge-code .code-comment", "# Your agents. One shared task state.");
  copy(".bridge-code code", "receive(goal) → execute() → report()");
  copy(".bridge-code > div", "SCOPED CREDENTIALS · PERSISTENT FEEDBACK");
  copy("#workspace-view > footer > span:last-child", "Model inference runs on your agent hosts. Task outcomes are reported by those agents.");
  copy(".permission-matrix article:last-child h3", "Operate approved agents.");
  copy(".permission-matrix article:last-child p", "Assign objectives, inspect agent progress, and request changes to model configurations and releases.");
  copy("#model-dialog .rig-details .field-help", "Saved for your runtime adapter. The dashboard does not connect to the rig or load its policy.");
  copy("#bridge-dialog .eyebrow", "OUTBOUND AGENT BRIDGE / v1");
  copy("#bridge-dialog .dialog-subtitle", "Create a runtime connection in Tasks, then install the agent bridge beside your policy. Your adapter owns inference, execution, and observed results.");
  $("source-kind").querySelector('[value="upload"]').disabled = !research;
  $("source-kind").querySelector('[value="upload"]').hidden = !research;
}
function renderModel(arm) {
  const a = state.workspace.agents.find((m) => m.id === arm), label = arm === "left" ? "A" : "B";
  if (!a) {
    $(`slot-${arm}`).innerHTML = `<button class="empty-card" data-edit="${arm}"><span class="add-mark">${icon("plus")}</span><strong>Add model ${label}</strong><small>Weights, repository or endpoint</small></button>`;
    return;
  }
  const available = isExample() && state.capabilities.sample_available;
  const native = available ? (state.capabilities.adapter?.source_dimensions?.[arm] || (arm === "left" ? 960 : 1024)) : null;
  $(`slot-${arm}`).innerHTML = `<article class="model-card" data-agent="${arm}"><div class="model-card-top"><span class="agent-letter">${label}</span><strong>${esc(a.name)}</strong><button class="edit-button" data-edit="${arm}" aria-label="Edit ${esc(a.name)}">···</button></div><div class="robot-illustration">${robot(a.robot)}</div><div class="model-spec"><div class="model-family">${families[a.family]}<span>${a.source.kind === "upload" ? "WEIGHTS" : "POLICY"}</span></div><div class="model-repo" title="${esc(a.source.ref)}">${esc(a.source.ref)}</div><div class="model-native"><span>${experimentsEnabled() ? "FEATURE CONTRACT" : "EXECUTION"}</span><strong>${experimentsEnabled() ? (native ? `${native}D → 64D` : "UNCONFIGURED") : "YOUR RUNTIME"}</strong></div></div><div class="model-card-footer"><span><i class="status-dot ${available ? "" : "neutral"}"></i>${available ? "Pinned feature contract" : "Registered · not loaded"}</span><span>${a.rig ? "Rig registered" : "No live rig"}</span></div></article>`;
}
function render() {
  if (!state.authenticated || !state.workspace) return;
  applyDeploymentProfile(); renderAccessIdentity(); renderReleases(); renderRequestModels();
  const w = state.workspace, mode = modes[w.mode], exact = isExample();
  const activeSession = !!state.session && !state.readOnly && !state.failed && state.sessionReleaseReady;
  const releaseBlocked = !!state.session && !state.sessionReleaseReady;
  $("breadcrumb-name").textContent = w.name;
  $("objective").value = w.objective;
  $("save-state").textContent = state.dirty ? "Unsaved changes" : currentWorkspaceSelected() ? "Saved workspace" : "Session configuration";
  $("model-count").textContent = `${w.agents.length} / 2 models`;
  $("nav-model-count").textContent = String(w.agents.length);
  renderPipeline();
  $("direction").value = w.direction;
  for (const arm of ["left", "right"]) renderModel(arm);
  $("channel-label").textContent = mode.title;
  $("direction-label").textContent = {both: "A ↔ B", left: "A → B", right: "B → A"}[w.direction];
  $("mode-options").innerHTML = Object.entries(modes).map(([key, m]) => `<button class="mode-option ${key === w.mode ? "active" : ""}" data-mode="${key}" aria-pressed="${key === w.mode}"><span class="radio-mark"></span><span><strong>${m.title}${key === "latent" ? "<em>CoLA-inspired</em>" : ""}</strong><small>${m.description}</small></span></button>`).join("");
  $("route-preview").innerHTML = mode.route;
  const latentPath = ["latent", "hybrid"].includes(w.mode);
  $("contract-input").textContent = latentPath ? "Frozen features" : "Local reports";
  $("contract-message").textContent = latentPath ? "64 dimensions" : "Qwen text";
  $("contract-context").textContent = latentPath ? "32 dimensions" : "Worker supplied";
  $("contract-preview").textContent = latentPath ? "Replacement head" : "Not evaluated";
  $("session-badge").textContent = state.readOnly ? "ARCHIVED VIEW" : releaseBlocked ? "RELEASE BLOCKED" : state.failed ? "CHANNEL STOPPED" : state.session ? "CHANNEL OPEN" : "DRAFT";
  $("connection-state").textContent = state.readOnly ? "Read-only history" : releaseBlocked ? "Model approval changed" : state.session ? "Session confirmed" : "Not connected";
  $("canvas-status").textContent = state.readOnly ? "Read-only journal · open a new channel to exchange" : releaseBlocked ? "Model approval changed · this session cannot exchange" : state.failed ? "Channel stopped · open a new session to continue" : state.session ? "Communication session open · hardware inactive" : w.agents.length === 2 ? (["latent", "hybrid"].includes(w.mode) && !exact ? "A trained adapter is needed for this pair" : (currentPairReleased() ? "Approved pair · ready to open a channel" : "Pair configured · manager release required")) : "Add two models to create a channel";
  $("canvas-status-dot").classList.toggle("neutral", !state.session);
  const needsAdapter = ["latent", "hybrid"].includes(w.mode) && !exact;
  let note = "Add your models to check the available communication paths.";
  if (w.agents.length === 2) {
    note = needsAdapter ? "These checkpoints need trained latent adapters. Try the LLM paths with local reports, or load the supported example pair." : exact ? "Example adapters match these pinned checkpoints. Captured features are available for a real communication probe." : "Text reports can use Qwen now. Your policy worker must load the weights and consume the returned context.";
  }
  $("compatibility").querySelector("p").textContent = note;
  $("connect-button").disabled = state.busy || !allowed("sessions.run") || !currentPairReleased() || (!allowed("workspace.edit") && !currentWorkspaceSelected()) || w.agents.length !== 2 || needsAdapter || (state.diagnostics && !state.diagnostics.ready) || activeSession;
  $("connect-button").innerHTML = icon(activeSession ? "check" : "link") + (state.readOnly || state.failed || releaseBlocked ? "Open a new channel" : activeSession ? "Pair connected" : "Connect pair");
  $("probe-button").disabled = state.busy || !state.session;
  $("probe-source").querySelector('[value="captured"]').disabled = !exact || !state.capabilities.sample_available;
  if (!exact || !state.capabilities.sample_available) $("probe-source").value = "manual";
  $("qwen-state").textContent = state.capabilities.llm_available ? "Installed" : "Setup needed";
  $("canvas").classList.toggle("busy", state.busy);
  $("export-button").disabled = state.busy;
  for (const id of ["save-button", "example-button", "direction", "objective"]) {
    $(id).disabled = state.busy || !allowed("workspace.edit");
    $(id).title = allowed("workspace.edit") ? "" : "Only managers may change the workspace or its models";
  }
  document.querySelectorAll("[data-mode], [data-edit]").forEach((button) => {
    button.disabled = state.busy || !allowed("workspace.edit");
    button.title = allowed("workspace.edit") ? "" : "Model and channel configuration is managed by your manager";
  });
  $("connect-button").title = !currentPairReleased() ? "A manager must release both exact model revisions first" : "Start an approved communication session";
  $("probe-button").title = state.session && !state.sessionReleaseReady ? "This session uses a model revision that is no longer released" : "Run a communication exchange";
  if (!experimentsEnabled()) {
    $("channel-label").textContent = "Shared task state";
    $("direction-label").textContent = "A ↔ workspace ↔ B";
    $("session-badge").textContent = currentPairReleased() ? "RELEASED" : "CONFIGURATION";
    $("canvas-status").textContent = currentPairReleased() ? "Approved models · connect your runtimes in Tasks" : "Register and release the models that will receive your tasks";
    $("connection-state").textContent = "Runtime connections in Tasks";
    $("channel-node").setAttribute("aria-label", "Open runtime connections");
    $("connect-button").disabled = true; $("probe-button").disabled = true;
  }
  updateProbeSource();
}
function updateProbeSource() {
  if (!experimentsEnabled()) { $("observations-panel").hidden = true; $("probe-button").disabled = true; return; }
  const captured = $("probe-source").value === "captured";
  $("condition-field").hidden = !captured;
  $("manual-reports").hidden = captured;
  const latent = ["latent", "hybrid"].includes(state.workspace.mode);
  $("probe-note").textContent = captured ? "Captured foundation features; fresh adapter / Qwen inference; verified Spacetime readback. No robot motion is executed." : latent ? "Latent mode also needs native feature vectors. Use captured observations or the API bridge to supply features; this text-only form cannot generate them." : "These are reports you supply, not sensor measurements. Qwen processes them and Spacetime commits the resulting peer messages.";
  if (!captured && latent) $("probe-button").disabled = true;
  else $("probe-button").disabled = state.busy || !state.session || state.readOnly || state.failed || !state.sessionReleaseReady || !allowed("sessions.run");
  updateObservations();
}
async function save() {
  if (!allowed("workspace.edit")) throw new Error("Only managers may edit the workspace.");
  state.workspace = await api("/api/workspace", state.workspace);
  state.savedWorkspace = structuredClone(state.workspace); state.dirty = false;
  await refreshModels(); render();
}
function robotPicker() {
  $("robot-picker").innerHTML = Object.entries(robotNames).map(([key, name]) => `<button type="button" class="robot-option ${state.robot === key ? "active" : ""}" data-robot="${key}" aria-pressed="${state.robot === key}">${robot(key)}${name}</button>`).join("");
}
function sourceFields() {
  const source = $("source-kind").value;
  $("source-ref-field").hidden = source === "upload";
  $("upload-field").hidden = source !== "upload";
  $("revision-field").hidden = source !== "huggingface";
  $("source-ref").required = source !== "upload";
  $("source-ref-label").textContent = {huggingface: "Repository ID", local: "Checkpoint path", endpoint: "Inference endpoint", upload: "Weights"}[source];
  $("source-ref").placeholder = {huggingface: "owner/model", local: "/path/on-your-agent-host/model", endpoint: experimentsEnabled() ? "http://127.0.0.1:8000/policy" : "https://your-policy-service.example/infer", upload: ""}[source];
}
function editModel(arm) {
  if (!allowed("models.manage")) return;
  const a = state.workspace.agents.find((m) => m.id === arm);
  $("model-form").reset(); state.upload = null;
  $("agent-id").value = arm; $("model-dialog-slot").textContent = `MODEL ${arm === "left" ? "A" : "B"}`;
  $("agent-name").value = a?.name || ""; state.robot = a?.robot || "arm";
  $("agent-family").value = a?.family || "smolvla";
  $("source-kind").value = !experimentsEnabled() && a?.source.kind === "upload" ? "local" : a?.source.kind || "huggingface";
  $("source-ref").value = a?.source.ref || ""; $("source-revision").value = a?.source.revision || "";
  $("rig-endpoint").value = a?.rig || ""; $("action-contract").value = a?.action_contract || "";
  $("form-error").hidden = true; $("remove-model").hidden = !a;
  $("register-model").textContent = a ? "Save model" : "Add model";
  $("upload-name").textContent = a?.source.kind === "upload" ? "Previously uploaded weights" : "Choose a weights file";
  if (a?.source.kind === "upload") state.upload = {asset: a.source.ref};
  $("upload-status").textContent = "Stored locally. Loading requires a matching runtime adapter and model configuration.";
  $("upload-progress").hidden = true;
  robotPicker(); sourceFields(); $("model-dialog").showModal();
}
async function register(event) {
  event.preventDefault(); $("register-model").disabled = true; $("form-error").hidden = true;
  try {
    const kind = $("source-kind").value;
    if (!experimentsEnabled() && kind === "upload") throw new Error("Register a policy on your agent host or an inference endpoint.");
    if (kind === "upload" && !state.upload) throw new Error("Choose and upload your safetensors weights first.");
    const a = {id: $("agent-id").value, name: $("agent-name").value.trim(), robot: state.robot,
      family: $("agent-family").value, source: {kind, ref: kind === "upload" ? state.upload.asset : $("source-ref").value.trim(), revision: kind === "huggingface" ? $("source-revision").value.trim() : ""},
      rig: $("rig-endpoint").value.trim(), action_contract: $("action-contract").value.trim()};
    const next = {...state.workspace, agents: [...state.workspace.agents.filter((m) => m.id !== a.id), a].sort((x, y) => x.id.localeCompare(y.id))};
    const value = await api("/api/workspace", next);
    state.workspace = value; state.savedWorkspace = structuredClone(value); clearSession(); state.dirty = false; await refreshModels(); render();
    $("model-dialog").close(); toast(`${a.name} registered`);
  } catch (error) { $("form-error").textContent = error.message; $("form-error").hidden = false; }
  finally { $("register-model").disabled = false; }
}
function upload(file) {
  if (!experimentsEnabled() || !file || !allowed("models.manage")) return;
  if (!file.name.endsWith(".safetensors") || file.size > 4 * 1024 ** 3) {
    $("form-error").textContent = "Choose a .safetensors file under 4 GiB, or register a local path."; $("form-error").hidden = false; return;
  }
  state.upload = null; $("register-model").disabled = true; $("weight-file").disabled = true;
  $("upload-progress").hidden = false; $("upload-progress").value = 0; $("form-error").hidden = true;
  $("upload-name").textContent = file.name; $("upload-status").textContent = "Uploading to this local workspace…";
  const request = new XMLHttpRequest(); activeUpload = request; request.open("POST", "/api/upload");
  request.setRequestHeader("X-Relay-Token", state.token); request.setRequestHeader("X-Filename", encodeURIComponent(file.name));
  request.upload.onprogress = (e) => { if (e.lengthComputable) $("upload-progress").value = e.loaded / e.total * 100; };
  request.onload = () => {
    try {
      const data = JSON.parse(request.responseText);
      if (request.status === 401) showAuth("login", "Your session expired. Please sign in again.");
      if (request.status !== 201) throw new Error(data.error || "Upload failed");
      state.upload = data; $("upload-status").textContent = `${data.tensors} tensors · ${formatBytes(data.bytes)} · Stored, not loaded`;
    } catch (error) { $("form-error").textContent = error.message; $("form-error").hidden = false; $("upload-status").textContent = "Upload failed. Choose a valid weights file to retry."; }
  };
  request.onerror = () => { $("form-error").textContent = "Upload interrupted. Try again."; $("form-error").hidden = false; };
  request.onloadend = () => { activeUpload = null; $("register-model").disabled = false; $("weight-file").disabled = false; $("upload-progress").hidden = true; };
  request.send(file);
}
function formatBytes(n) { return n >= 1024 ** 2 ? `${(n / 1024 ** 2).toFixed(1)} MiB` : n >= 1024 ? `${(n / 1024).toFixed(1)} KiB` : `${n} B`; }
function ms(n) { if (n == null || !Number.isFinite(n)) return "Not retained"; return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${n.toFixed(1)} ms`; }
function renderHistory(data) {
  const previousReady = state.sessionReleaseReady, previousFailed = state.failed;
  state.failed = !!data.failed;
  state.history = data;
  state.sessionReleaseReady = data.release_ready === true;
  if (data.release_error) { notify(data.release_error); $("probe-button").disabled = true; }
  if (previousReady && !state.sessionReleaseReady && state.authenticated) {
    void refreshApprovalState();
  }
  if (state.authenticated && (previousReady !== state.sessionReleaseReady || previousFailed !== state.failed)) render();
  const signature = JSON.stringify(data);
  if (signature === historySignature) return;
  historySignature = signature;
  const messages = data.messages || [];
  $("stream-count").textContent = `${messages.length} PACKETS`;
  $("stream-empty").hidden = messages.length > 0;
  if (messages.length) {
    if (!messages.some((m) => m.message_id === state.selectedMessage)) state.selectedMessage = messages.at(-1).message_id;
    renderPackets();
  } else $("stream-events").replaceChildren();
  if (!messages.length) { $("stream-metrics").hidden = true; $("receiver-panel").hidden = true; }
  if (data.failed) { notify("This session stopped after a database failure. Its committed packets remain available. Open a new channel to continue."); $("probe-button").disabled = true; }
}
function renderExchange(result) {
  $("stream-metrics").hidden = !result;
  if (!result) { $("receiver-panel").hidden = true; return; }
  $("metric-roundtrip").textContent = ms(result.elapsed_ms);
  $("metric-db").textContent = ms(result.database_ms);
  $("metric-llm").textContent = result.generations?.length ? ms(result.llm_ms) : "Not used";
  $("metric-llm-note").textContent = result.generations?.length ? "Language path" : "Direct learned path";
  $("metric-bytes").textContent = formatBytes(result.bytes || 0);
  $("receiver-panel").querySelector("h3").textContent = `Receiver predictions · exchange ${result.exchange}`;
  renderReceivers(result);
}
function renderPackets() {
  const messages = state.history?.messages || [];
  const names = Object.fromEntries(state.workspace.agents.map((a) => [a.id, a.name]));
  const selected = messages.find((m) => m.message_id === state.selectedMessage);
  const recent = messages.slice(-8).reverse();
  $("stream-events").innerHTML = `<div class="packet-toolbar"><span>COMMITTED MESSAGE LOG</span><span>${messages.length > 8 ? "LATEST 8 OF " : ""}${messages.length} RECORDS</span></div><div class="packet-list">${recent.map((m) => `<button class="packet-row ${m.message_id === state.selectedMessage ? "active" : ""}" data-packet="${esc(m.message_id)}" aria-pressed="${m.message_id === state.selectedMessage}"><span class="agent-letter ${m.sender === "right" ? "agent-b" : ""}">${m.sender === "left" ? "A" : "B"}</span><span><strong>${esc(names[m.sender] || m.sender)} <span class="muted">→</span> ${esc(names[m.recipient] || m.recipient)}</strong><small>${esc(modes[m.payload.mode]?.title || m.payload.mode)} · ${m.payload.vector ? `${m.payload.vector.length}D latent` : "Text context"}</small></span><span class="packet-status">Committed</span><span class="packet-order">EX ${esc(m.payload.exchange)}</span></button>`).join("")}</div><div id="selected-packet"></div>`;
  if (!selected) return;
  const p = selected.payload;
  const adapter = p.adapter || {};
  const vector = Array.isArray(p.vector) ? p.vector : null;
  const checkpoint = adapter.checkpoint_sha256;
  const result = state.history.results?.find((r) => r.exchange === p.exchange);
  renderExchange(result);
  const conditionLabel = {left_obstacle: "Obstacle visible to A", right_obstacle: "Obstacle visible to B", clear: "Both views clear"}[result?.condition];
  const observed = result?.source === "captured" ? `Captured · ${conditionLabel || "Condition not retained"}` : result?.source === "manual" ? "Supplied local reports" : "Recorded session";
  $("selected-packet").innerHTML = `<article class="packet-inspector"><div class="packet-inspector-header"><strong>Message ${esc(selected.message_id)}</strong><span>${esc(observed)}</span></div><div class="packet-body"><div class="packet-facts"><span>SEQUENCE <b>${esc(selected.sequence)}</b></span><span>ROUTE <b>${selected.sender === "left" ? "A → B" : "B → A"}</b></span>${checkpoint ? `<span>ADAPTER <b>${esc(checkpoint.slice(0,12))}…</b></span>` : ""}</div>${p.context ? `<div class="packet-context-label">DELIVERED LANGUAGE CONTEXT</div><p class="event-message">${esc(p.context)}</p>` : ""}${vector ? `<div class="packet-context-label">LEARNED MESSAGE / ${vector.length} DIMENSIONS</div><div class="vector-display" id="selected-vector" aria-label="Signed learned message components"></div><div class="vector-label"><span>64D encoded → 32D receiver context</span><span class="vector-scale">−1 <i></i> +1</span></div>` : ""}${p.safety_facts && Object.keys(p.safety_facts).length ? `<details class="event-details"><summary>Structured safety context</summary><pre>${esc(JSON.stringify(p.safety_facts, null, 2))}</pre></details>` : ""}<details class="event-details"><summary>Inspect full committed packet</summary><pre>${esc(JSON.stringify(selected, null, 2))}</pre></details></div><div class="receipt-strip">${icon("check")}<span>${result?.readback_verified ? "Exact packet independently verified after SpacetimeDB commit" : "Read from the SpacetimeDB message journal"}</span></div></article>`;
  if (vector) for (let i = 0; i < vector.length; i++) {
    const v = vector[i], cell = document.createElement("i"); cell.className = "vector-cell";
    const magnitude = Math.min(1, Math.abs(v));
    const base = v < 0 ? [79,122,147] : [93,139,54];
    const neutral = [237,244,230];
    const rgb = neutral.map((n,j) => Math.round(n + (base[j] - n) * magnitude));
    cell.style.setProperty("--cell-color", `rgb(${rgb.join(",")})`); cell.title = `Dimension ${i + 1}: ${v.toFixed(5)}`;
    $("selected-vector").append(cell);
  }
}
function renderReceivers(result) {
  const decisions = result.decisions || {};
  $("receiver-panel").hidden = !Object.keys(decisions).length && !result.prediction_error && result.preview_status !== "not_persisted";
  if (result.prediction_error) { $("receiver-cards").innerHTML = `<p class="packet-error">Messages committed, but action preview failed: ${esc(result.prediction_error)}</p>`; return; }
  if (result.preview_status === "not_persisted" && !Object.keys(decisions).length) { $("receiver-cards").innerHTML = `<p class="receiver-caption">This session was recovered from the durable journal. Post-commit action previews and transport timings were process-local and are not retained.</p>`; return; }
  const names = Object.fromEntries(state.workspace.agents.map((a) => [a.id,a.name]));
  $("receiver-cards").innerHTML = Object.entries(decisions).map(([arm,d]) => {
    const p = d.prediction || {}, held = p.learned_hold || p.missing_message_hold;
    const target = Array.isArray(d.action?.target) ? d.action.target : [];
    return `<article class="receiver-card"><div class="receiver-card-heading"><span class="agent-letter ${arm === "right" ? "agent-b" : ""}">${arm === "left" ? "A" : "B"}</span><strong>${esc(names[arm] || arm)}</strong></div><div class="action-state ${held ? "hold" : ""}"><strong>${held ? "HOLD" : "MOVE"}</strong><span>${p.missing_message_hold ? "NO PEER MESSAGE" : "LEARNED HEAD"}</span></div><div class="action-components">${["X","Y","Z"].map((axis,i) => `<div><span>${axis} TARGET / m</span><strong>${Number.isFinite(target[i]) ? target[i].toFixed(4) : "—"}</strong></div>`).join("")}</div><p class="receiver-caption">Gripper ${d.action?.grip ? "closed" : "open"} · Peer age ${p.peer_age_ticks == null ? "unavailable" : `${p.peer_age_ticks} ticks`}</p><details class="event-details"><summary>Raw head & action</summary><pre>${esc(JSON.stringify(d,null,2))}</pre></details></article>`;
  }).join("");
}
async function refreshStream() {
  const session = state.session, epoch = authEpoch;
  if (!session || state.busy) return;
  try { const data = await api(`/api/sessions/${session}`); if (session === state.session) renderHistory(data); }
  catch (error) { if (error.stale || epoch !== authEpoch) return; clearInterval(polling); notify(`Could not refresh the durable stream: ${error.message}`); }
}
async function connect() {
  if (!experimentsEnabled()) return;
  state.busy = true; $("notice").hidden = true; render();
  $("connect-button").textContent = "Opening channel…";
  try {
    if (allowed("workspace.edit") && (state.dirty || !currentWorkspaceSelected())) await save();
    const result = await api("/api/sessions", state.workspace);
    state.sessionReleaseReady = result.release_ready !== false && currentPairReleased();
    clearInterval(polling); state.results = []; state.session = result.session_id; state.readOnly = false; state.failed = false; state.history = null; state.selectedMessage = null; historySignature = "";
    $("stream-events").replaceChildren(); $("stream-empty").hidden = false; $("stream-metrics").hidden = true; $("receiver-panel").hidden = true;
    void refreshSessions();
    toast("Pair connected through SpacetimeDB");
    polling = setInterval(refreshStream, 3000);
  } catch (error) { notify(error.message); }
  finally { state.busy = false; render(); }
}
async function probe() {
  if (!experimentsEnabled()) return;
  state.busy = true; $("notice").hidden = true; render();
  $("probe-button").textContent = state.workspace.mode === "latent" ? "Encoding & committing…" : "Qwen is processing…";
  try {
    const request = {request_id: crypto.randomUUID(), source: $("probe-source").value, condition: $("condition").value,
      reports: {left: $("report-left").value, right: $("report-right").value}};
    const result = await api(`/api/sessions/${state.session}/exchange`, request);
    state.results.push(result); state.selectedMessage = null;
    void refreshSessions();
    renderHistory(await api(`/api/sessions/${state.session}`));
    if (result.prediction_error) notify(`Messages committed; receiver preview failed: ${result.prediction_error}`);
    else if (result.hybrid_scope) notify(result.hybrid_scope, true);
    else toast("Exchange committed and independently read back");
  } catch (error) { notify(error.message); }
  finally { state.busy = false; render(); $("probe-button").innerHTML = icon("play") + "Test exchange"; }
}
function exportWorkspace() {
  const blob = new Blob([JSON.stringify(state.workspace, null, 2) + "\n"], {type: "application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = "relay-team.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Workspace configuration exported");
}
function showBridge() {
  if (!experimentsEnabled()) {
    $("api-example").textContent = `# Save as your_module.py on the host running your policy.
# Create a scoped connection in Tasks and save its token privately.
# Install the relay-studio package on this host, then run:
# chmod 600 ./worker-token
# python -m relay_lab.studio.worker --url ${location.origin} \\
#   --token-file ./worker-token --handler your_module:handle

def handle(context):
    task = context["task"]
    # task contains the assigned objective, success criteria,
    # and peer feedback. Inspect context["worker"] for your identity.
    # Check context["should_stop"]() before and during bounded actions.
    # Call your local policy here; return its observed outcome.
    # This unconfigured adapter deliberately reports a blocked task.
    return {
        "status": "blocked",
        "progress": 0,
        "summary": "Policy adapter needs configuration.",
        "evidence": [],
    }

# Valid statuses: running, blocked, succeeded, failed.
# Only report succeeded with progress 100 after checking the criteria.
# Keep worker-token and the worker's recovery journal private.
`;
    $("bridge-dialog").showModal();
    return;
  }
  $("api-example").textContent = `import getpass, json, uuid
from http.cookiejar import CookieJar
from urllib.request import Request, build_opener, HTTPCookieProcessor

BASE = "${location.origin}"
client = build_opener(HTTPCookieProcessor(CookieJar()))

def call(path, data=None, token=None):
    headers = {"Content-Type": "application/json", "Origin": BASE}
    if token: headers["X-Relay-Token"] = token
    body = json.dumps(data).encode() if data is not None else None
    return json.load(client.open(Request(BASE + path, body, headers)))

# Individual account; password is never written into the example.
call("/api/auth/login", {
    "username": input("Studio username: "),
    "password": getpass.getpass("Studio password: ")
})
boot = call("/api/bootstrap")
assert boot["release_ready"], "A manager must release the configured pair"
# Use the saved configuration approved for this account.
session = call("/api/sessions", boot["workspace"], boot["token"])
sid = session["session_id"]
result = call(f"/api/sessions/{sid}/exchange", {
    "request_id": str(uuid.uuid4()),
    "source": "captured",
    "condition": "left_obstacle"
}, boot["token"])

for packet in result["messages"]:
    print(packet["recipient"], packet["payload"])
    # Feed context into the matching policy adapter.
    # Your runtime remains responsible for actuation.

call("/api/auth/logout", {}, boot["token"])
`;
  $("bridge-dialog").showModal();
}
function bind() {
  const localWorkspace = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  $("workspace-environment").textContent = localWorkspace ? "Local environment" : "Private workspace";
  $("environment-location").textContent = localWorkspace ? "LOCAL" : "PRIVATE";
  $("auth-environment").textContent = `${localWorkspace ? "LOCAL" : "PRIVATE"} WORKSPACE · INDIVIDUAL ACCOUNTS`;
  if (!localWorkspace) {
    document.querySelector(".second-label").hidden = true;
    document.querySelectorAll('a[href^="http://127.0.0.1:"]').forEach((link) => { link.hidden = true; });
  }
  window.RelayAnalytics.mount({api, notify: toast, authenticated: () => state.authenticated, experimentsEnabled});
  window.RelayTasks.mount({api, notify: toast, authenticated: () => state.authenticated, context: () => ({workspace: state.savedWorkspace, models: state.models, user: state.user, permissions: state.permissions, capabilities: state.capabilities}), onWorkspace: async (savedWorkspace = null) => {
    if (savedWorkspace) {
      state.workspace = structuredClone(savedWorkspace);
      state.savedWorkspace = structuredClone(savedWorkspace);
      state.dirty = false;
      clearSession();
      await refreshModels();
      render();
    }
    setStudioView("workspace");
  }});
  const showWorkspace = () => setStudioView("workspace");
  for (const id of ["nav-team", "nav-stream", "nav-history", "nav-access", "nav-bridge", "mobile-workspace"]) $(id).addEventListener("click", showWorkspace);
  for (const id of ["nav-analytics", "mobile-analytics"]) $(id).addEventListener("click", () => setStudioView("analytics"));
  for (const id of ["nav-tasks", "mobile-tasks"]) $(id).addEventListener("click", () => setStudioView("tasks"));
  window.addEventListener("hashchange", () => setStudioView(viewFromHash(), false));
  $("canvas").addEventListener("click", (event) => { const b = event.target.closest("[data-edit]"); if (b) editModel(b.dataset.edit); });
  $("mode-options").addEventListener("click", (event) => { const b = event.target.closest("[data-mode]"); if (b && !state.busy && allowed("workspace.edit")) { state.workspace.mode = b.dataset.mode; changed(); } });
  $("robot-picker").addEventListener("click", (event) => { const b = event.target.closest("[data-robot]"); if (b) { state.robot = b.dataset.robot; robotPicker(); } });
  $("objective").addEventListener("change", () => { state.workspace.objective = $("objective").value; changed(); });
  $("direction").addEventListener("change", () => { state.workspace.direction = $("direction").value; changed(); });
  $("source-kind").addEventListener("change", sourceFields);
  $("model-form").addEventListener("submit", register);
  $("weight-file").addEventListener("change", () => upload($("weight-file").files[0]));
  $("model-dialog").addEventListener("close", () => { if (activeUpload) activeUpload.abort(); });
  for (const id of ["close-model", "cancel-model"]) $(id).addEventListener("click", () => $("model-dialog").close());
  $("remove-model").addEventListener("click", async () => {
    const next = {...state.workspace, agents: state.workspace.agents.filter((a) => a.id !== $("agent-id").value)};
    try { state.workspace = await api("/api/workspace", next); state.savedWorkspace = structuredClone(state.workspace); clearSession(); state.dirty = false; await refreshModels(); render(); $("model-dialog").close(); toast("Model removed from the team"); }
    catch (error) { $("form-error").textContent = error.message; $("form-error").hidden = false; }
  });
  $("example-button").addEventListener("click", async () => {
    if (!experimentsEnabled() || !state.example) return;
    try { const value = await api("/api/workspace", state.example); state.workspace = value; state.savedWorkspace = structuredClone(value); clearSession(); state.dirty = false; await refreshModels(); $("probe-source").value = "captured"; $("notice").hidden = true; render(); toast("Example pair loaded · weights are not loaded into memory"); }
    catch (error) { notify(error.message); }
  });
  $("save-button").addEventListener("click", async () => { try { await save(); toast("Workspace saved"); } catch (e) { notify(e.message); } });
  $("export-button").addEventListener("click", exportWorkspace);
  $("connect-button").addEventListener("click", connect); $("probe-button").addEventListener("click", probe);
  $("probe-source").addEventListener("change", updateProbeSource);
  $("condition").addEventListener("change", updateObservations);
  $("refresh-health").addEventListener("click", async () => { await refreshModels(); await refreshHealth(); await refreshDiagnostics(); });
  $("refresh-sessions").addEventListener("click", refreshSessions);
  $("session-list").addEventListener("click", (event) => { const b = event.target.closest("[data-session]"); if (b && !state.busy) void openSession(b.dataset.session, b.dataset.resume === "true"); });
  $("stream-events").addEventListener("click", (event) => { const b = event.target.closest("[data-packet]"); if (b) { state.selectedMessage = b.dataset.packet; renderPackets(); } });
  $("nav-history").addEventListener("click", () => $("session-section").scrollIntoView({behavior:"smooth", block:"start"}));
  $("channel-node").addEventListener("click", () => { if (!experimentsEnabled()) { setStudioView("tasks"); $("tk-runtime-title").scrollIntoView({behavior:"smooth",block:"start"}); return; } $("mode-options").scrollIntoView({behavior: "smooth", block: "center"}); $("mode-options").querySelector("button.active")?.focus(); });
  $("nav-team").addEventListener("click", () => {
    if (state.session && !state.dirty && state.savedWorkspace) { state.workspace = structuredClone(state.savedWorkspace); clearSession(); render(); }
    window.scrollTo({top: 0, behavior: "smooth"});
  });
  $("nav-stream").addEventListener("click", () => $("stream-panel").scrollIntoView({behavior: "smooth", block: "center"}));
  for (const id of ["nav-bridge", "bridge-button"]) $(id).addEventListener("click", showBridge);
  $("close-bridge").addEventListener("click", () => $("bridge-dialog").close());
  $("copy-api").addEventListener("click", async () => { try { await navigator.clipboard.writeText($("api-example").textContent); toast("Python example copied"); } catch { toast("Select and copy the example above"); } });
  window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
}
async function checkAccess() {
  $("auth-retry").disabled = true;
  let signedIn = false;
  try {
    const auth = await api("/api/auth/status"); signedIn = auth.authenticated;
    state.profileMode = !!auth.account_switching;
    const profile = await window.RelayProfiles.enter(auth);
    if (profile) { state.token = profile.token; signedIn = true; }
    if (signedIn) await loadWorkspace();
    else showAuth(auth.setup_required ? "setup" : "login");
  } catch (error) {
    if (error.stale || error.status === 401) return;
    showAccessRecovery(`${signedIn ? "You are signed in, but the workspace could not load" : "Studio could not check workspace access"}: ${error.message}`);
  } finally { $("auth-retry").disabled = false; }
}
async function init() {
  window.RelayProfiles.mount({api, context: () => state, notify: toast, activate: async (credentials) => {
    showAuth(); state.token = credentials.token;
    await loadWorkspace();
  }});
  bind(); bindAccess(); await checkAccess();
}
function viewFromHash() {
  return location.hash === "#analytics" ? "analytics" : location.hash === "#workspace" ? "workspace" : "tasks";
}
function setStudioView(view, updateHash = true) {
  const selected = state.authenticated ? (view === true ? "analytics" : view === false ? "workspace" : view) : "workspace";
  for (const name of ["workspace", "analytics", "tasks"]) $(`${name}-view`).hidden = name !== selected;
  document.body.classList.toggle("analytics-active", selected === "analytics");
  document.body.classList.toggle("tasks-active", selected === "tasks");
  for (const [id, name] of [["nav-team", "workspace"], ["nav-analytics", "analytics"], ["nav-tasks", "tasks"]]) {
    $(id).classList.toggle("selected", name === selected); $(id).setAttribute("aria-current", name === selected ? "page" : "false");
  }
  for (const name of ["workspace", "analytics", "tasks"]) $(`mobile-${name}`).classList.toggle("active", name === selected);
  if (updateHash) history.replaceState(null, "", location.pathname + location.search + `#${selected}`);
  if (selected === "analytics") window.RelayAnalytics.show();
  if (selected === "tasks") window.RelayTasks.show(); else window.RelayTasks.hide();
  window.scrollTo({top: 0});
}
function renderPipeline() {
  const latent = ["latent","hybrid"].includes(state.workspace.mode);
  const nodes = !experimentsEnabled() ? [["APPROVED MODELS","Exact revisions"],["TASK OBJECTIVE","Scoped assignment"],["SPACETIMEDB","Commit + verify"],["YOUR RUNTIME","Infer + report"]] : latent ? [["FROZEN BACKBONE","960 / 1024D"],["TRAINED ENCODER","64D message"],["SPACETIMEDB","Commit + verify"],["PEER DECODER","32D context"],["REPLACEMENT HEAD","Action preview"]] : [["LOCAL EVIDENCE","Text reports"],["QWEN 3",state.workspace.mode === "orchestrated" ? "Joint plan" : "Peer context"],["SPACETIMEDB","Commit + verify"],["POLICY WORKER","Receive text"]];
  $("pipeline").innerHTML = nodes.map(([label,value],i) => `${i ? "<i>→</i>" : ""}<div class="pipeline-node ${i === 2 ? "emphasis" : ""}"><span>${label}</span><strong>${value}</strong></div>`).join("");
}
function updateObservations() {
  const shown = isExample() && state.capabilities.sample_available && $("probe-source").value === "captured";
  $("observations-panel").hidden = !shown;
  if (!shown) return;
  const condition = $("condition").value;
  for (const arm of ["left","right"]) {
    const img = $(`observation-${arm}`), path = `/api/samples/${condition}/${arm}.png`;
    if (img.getAttribute("src") !== path) {
      img.hidden = false; img.src = path;
      img.onerror = () => { img.hidden = true; $(`observation-${arm}-note`).textContent = "Captured image could not be loaded"; };
    }
    $(`observation-${arm}-note`).textContent = condition === `${arm}_obstacle` ? "Local obstacle is visible" : "No local obstacle in this view";
  }
}
async function refreshHealth() {
  $("refresh-health").disabled = true;
  try {
    const health = await api("/api/health"); state.health = health;
    const db = !!health.database?.available, adapter = !!health.adapter?.verified, llm = !!health.llm?.available;
    state.capabilities.llm_available = llm;
    state.capabilities.sample_available = health.sample_available ?? health.adapter?.available ?? false;
    $("health-db").textContent = db ? "SpacetimeDB connected" : "Database unavailable";
    $("health-db").title = health.database?.error || `${health.database?.database || "Local database"} · ${ms(health.database?.latency_ms)}`;
    $("health-adapter").textContent = adapter ? "CoLA adapters verified" : "Adapter setup needed";
    $("health-adapter").title = health.adapter?.error || "Pinned 960D / 1024D feature contracts · learned 64D messages";
    $("health-qwen").textContent = llm ? (health.llm.loaded ? "Qwen runtime loaded" : "Qwen runtime installed") : "Qwen setup needed";
    for (const [key,ready] of [["db",db],["adapter",adapter],["qwen",llm]]) { $(`health-${key}-dot`).classList.toggle("neutral", !ready); $(`health-${key}-dot`).classList.toggle("error", !ready); }
    $("sidebar-db-dot").classList.toggle("neutral", !db); $("sidebar-db-label").textContent = db ? "SPACETIMEDB ONLINE" : "DATABASE OFFLINE";
    if (health.adapter) state.capabilities.adapter = health.adapter;
    render();
  } catch (error) {
    if (error.stale) return;
    $("health-db").textContent = "Readiness check failed"; $("health-db").title = error.message;
    $("health-adapter").textContent = "Readiness unavailable"; $("health-qwen").textContent = "Readiness unavailable";
    $("sidebar-db-label").textContent = "CHECK UNAVAILABLE";
  } finally { $("refresh-health").disabled = false; }
}
async function refreshDiagnostics() {
  if (!experimentsEnabled()) return;
  if (!state.workspace) return;
  const revision = ++diagnosticRevision;
  try {
    const data = await api("/api/diagnostics", state.workspace);
    if (revision !== diagnosticRevision) return;
    state.diagnostics = data;
    const blocked = data.checks?.filter((c) => c.status === "blocked") || [];
    render();
    $("preflight-details").hidden = !data.checks?.length;
    $("preflight-checks").innerHTML = (data.checks || []).map((c) => `<div class="preflight-check"><span class="status-dot ${c.status === "ready" ? "" : c.status === "blocked" ? "error" : "neutral"}"></span><div><strong>${esc(c.label)}</strong><p>${esc(c.detail)}</p></div></div>`).join("");
    if (blocked.length && state.workspace.agents.length === 2) $("compatibility").querySelector("p").textContent = blocked.map((c) => c.detail || c.label).join(" ");
  } catch (error) { if (revision === diagnosticRevision) { state.diagnostics = null; $("compatibility").querySelector("p").textContent = `Preflight could not be checked: ${error.message}`; } }
}
async function refreshSessions() {
  if (!experimentsEnabled()) return;
  $("refresh-sessions").disabled = true;
  try { const data = await api("/api/sessions"); state.sessions = data.sessions || []; renderSessions(); }
  catch (error) { if (error.stale) return; $("session-list").innerHTML = `<p class="empty-history session-error">Could not read session history: ${esc(error.message)}</p>`; }
  finally { $("refresh-sessions").disabled = false; }
}
function sessionDate(value) {
  if (!value) return "Earlier session";
  const date = new Date(typeof value === "number" ? value * (value < 1e12 ? 1000 : 1) : value);
  return Number.isNaN(date.valueOf()) ? "Earlier session" : date.toLocaleString(undefined,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"});
}
function renderSessions() {
  $("session-list").innerHTML = state.sessions.length ? state.sessions.slice(0,12).map((s) => `<article class="session-row ${s.session_id === state.session ? "current" : ""}"><div><strong>${esc(s.name)}</strong><small>${esc(sessionDate(s.updated_at))} · ${esc(s.session_id.slice(-8))}</small></div><div class="session-mode">${esc(modes[s.mode]?.title || s.mode)}<small>${s.direction === "both" ? "A ↔ B" : s.direction === "left" ? "A → B" : "B → A"} · ${esc(s.resumable ? "Resumable" : "Read only")}</small></div><span class="session-count">${s.tick} exchange${s.tick === 1 ? "" : "s"}</span><button class="button" data-session="${esc(s.session_id)}" data-resume="${!!s.resumable}" ${state.busy ? "disabled" : ""}>${s.session_id === state.session ? "Open" : s.resumable ? "Resume" : "Inspect"}${icon("arrow")}</button></article>`).join("") : '<p class="empty-history">No saved sessions yet. Connect a pair to create the first durable channel.</p>';
}
async function openSession(id, resume) {
  if (!experimentsEnabled()) return;
  state.busy = true; render();
  try {
    const data = await api(`/api/sessions/${id}${resume ? "/resume" : ""}`, resume ? {} : undefined);
    clearInterval(polling); historySignature = ""; state.session = id; state.selectedMessage = null; state.readOnly = data.status === "inspect_only"; state.failed = !!data.failed;
    const latest = data.results?.at(-1);
    if (["clear","left_obstacle","right_obstacle"].includes(latest?.condition)) $("condition").value = latest.condition;
    if (latest?.source) $("probe-source").value = latest.source;
    if (data.workspace) { state.workspace = data.workspace; state.dirty = false; }
    state.diagnostics = null; render(); renderHistory(data); renderSessions();
    if (!state.readOnly) polling = setInterval(refreshStream,3000);
    toast(state.readOnly ? "Inspecting the durable message journal" : "Session resumed from SpacetimeDB");
    $("stream-panel").scrollIntoView({behavior:"smooth",block:"start"});
    void refreshDiagnostics();
  } catch (error) { notify(error.message); }
  finally { state.busy = false; render(); renderSessions(); }
}
const allowed = (permission) => !!state.authenticated && state.permissions.includes(permission);
const stableJSON = (value) => JSON.stringify(value, (_key, item) => item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a],[b]) => a.localeCompare(b))) : item);
const modelFor = (agent) => state.models.find((model) => stableJSON(model.spec) === stableJSON(agent));
const currentPairReleased = () => state.workspace?.agents.length === 2 && state.workspace.agents.every((agent) => modelFor(agent)?.status === "released");
const currentWorkspaceSelected = () => stableJSON(state.workspace) === stableJSON(state.savedWorkspace);
function showAuth(mode = "login", message = "") {
  authEpoch++; diagnosticRevision++;
  window.RelayAnalytics.reset();
  window.RelayTasks.reset();
  setStudioView(false, false);
  clearInterval(polling); clearTimeout(toastTimer); $("toast").hidden = true;
  if (activeUpload) activeUpload.abort();
  state.authenticated = false; state.user = null; state.permissions = []; state.token = "";
  state.workspace = null; state.savedWorkspace = null; state.models = []; state.sessions = []; state.session = null;
  state.results = []; state.history = null; state.selectedMessage = null; state.team = []; state.requests = []; state.sessionReleaseReady = false; state.dirty = false; state.busy = false; historySignature = "";
  state.capabilities = {}; state.example = null; state.health = null; state.diagnostics = null; state.upload = null;
  state.pendingRelease = null; state.pendingRequest = null; state.readOnly = false; state.failed = false; state.release_ready = false;
  for (const id of ["slot-left", "slot-right", "release-list", "stream-events", "receiver-cards", "session-list", "release-review", "request-review", "preflight-checks", "request-model"]) $(id).replaceChildren();
  $("api-example").textContent = ""; $("model-form").reset(); $("invite-form").reset(); $("request-form").reset();
  $("objective").value = ""; $("report-left").value = ""; $("report-right").value = "";
  $("breadcrumb-name").textContent = "Workspace"; $("account-name").textContent = "Account"; $("account-avatar").textContent = "?";
  $("probe-source").value = "captured";
  for (const [id, value] of Object.entries({"nav-model-count":"0", "model-count":"0 / 2 models", "stream-count":"0 PACKETS", "request-count":"REQUESTS", "team-count":"", "save-state":"Sign in required", "role-badge":"ACCOUNT", "account-role":"Signed out", "session-badge":"DRAFT", "connection-state":"Not connected", "canvas-status":"Sign in to use this workspace", "metric-roundtrip":"—", "metric-db":"—", "metric-llm":"—", "metric-bytes":"—"})) $(id).textContent = value;
  for (const id of ["stream-metrics", "receiver-panel", "observations-panel", "preflight-details"]) $(id).hidden = true;
  for (const arm of ["left", "right"]) $("observation-" + arm).removeAttribute("src");
  document.querySelectorAll("dialog[open]").forEach((dialog) => dialog.close());
  document.body.classList.add("auth-pending"); $("auth-screen").hidden = false;
  $("invitation-code").textContent = ""; $("invitation-result").hidden = true;
  $("team-list").replaceChildren(); $("audit-list").replaceChildren(); $("request-list").replaceChildren();
  $("auth-form").reset(); $("auth-screen").classList.remove("auth-recovery"); $("auth-retry").hidden = true; $("auth-recovery-logout").hidden = true; setAuthMode(mode);
  $("auth-error").textContent = message; $("auth-error").hidden = !message;
}
function setAuthMode(mode) {
  state.authMode = mode;
  const setup = mode === "setup", join = mode === "join";
  $("auth-tabs").hidden = setup;
  $("auth-eyebrow").textContent = setup ? "FIRST-TIME WORKSPACE SETUP" : join ? "YOU’RE INVITED" : "WORKSPACE ACCESS";
  $("auth-title").textContent = setup ? "Create the first manager." : join ? "Join your team." : "Welcome back.";
  $("auth-subtitle").textContent = setup ? "The local setup code gives the workspace owner its first manager account." : join ? "Use the one-time invitation your manager shared with you. Your role is assigned by the invitation." : "Sign in with the account created for this workspace.";
  $("auth-code-field").hidden = !(setup || join); $("auth-code").required = setup || join;
  $("auth-code-label").textContent = setup ? "Local setup code" : "Invitation code";
  $("auth-code-help").textContent = setup ? "Find it in .relay-lab/studio/setup-code on the computer hosting Studio." : "Invitation codes expire and can only be used once.";
  $("auth-name-field").hidden = !setup; $("auth-name").required = setup;
  $("auth-username-field").hidden = join; $("auth-username").required = !join;
  $("auth-password").autocomplete = mode === "login" ? "current-password" : "new-password";
  $("auth-password").minLength = mode === "login" ? 1 : 12;
  $("auth-password-help").hidden = mode === "login";
  $("auth-submit").textContent = setup ? "Create manager account" : join ? "Accept invitation" : "Sign in";
  $("auth-submit").disabled = false; $("auth-error").hidden = true;
  document.querySelectorAll("[data-auth-mode]").forEach((button) => { button.classList.toggle("active", button.dataset.authMode === mode); button.setAttribute("aria-pressed", String(button.dataset.authMode === mode)); });
}
async function authenticate(event) {
  event.preventDefault();
  authEpoch++;
  let signedIn = false; $("auth-submit").disabled = true; $("auth-error").hidden = true;
  const mode = state.authMode;
  const payload = {password: $("auth-password").value};
  if (mode !== "join") payload.username = $("auth-username").value.trim();
  if (mode !== "login") payload.code = $("auth-code").value.trim();
  if (mode === "setup") payload.name = $("auth-name").value.trim();
  try {
    const credentials = await api(`/api/auth/${mode}`, payload); signedIn = true; state.token = credentials.token;
    $("auth-password").value = ""; $("auth-code").value = "";
    await loadWorkspace();
  } catch (error) {
    if (error.stale) return;
    if (signedIn && error.status !== 401) showAccessRecovery(`You are signed in, but the workspace could not load: ${error.message}`);
    else { $("auth-error").textContent = error.message; $("auth-error").hidden = false; }
  } finally { $("auth-submit").disabled = false; }
}
async function loadWorkspace() {
  const boot = await api("/api/bootstrap");
  Object.assign(state, boot, {authenticated: true, savedWorkspace: structuredClone(boot.workspace), models: boot.models || [], permissions: boot.permissions || []});
  document.body.classList.remove("auth-pending"); $("auth-screen").classList.remove("auth-recovery"); $("auth-screen").hidden = true; $("auth-retry").hidden = true; $("auth-recovery-logout").hidden = true;
  $("notice").hidden = true; render();
  setStudioView(viewFromHash(), false);
  await Promise.allSettled([refreshHealth(), refreshDiagnostics(), refreshSessions(), refreshRequests()]);
}
async function signOut() {
  if (state.profileMode) { await window.RelayProfiles.show(); return; }
  const buttons = ["logout-button", "logout-top-button", "auth-recovery-logout"];
  buttons.forEach((id) => { $(id).disabled = true; });
  try { await api("/api/auth/logout", {}); showAuth("login"); }
  catch (error) { if (error.stale) return; if (state.authenticated) notify(error.message); else { $("auth-error").textContent = error.message; $("auth-error").hidden = false; } }
  finally { buttons.forEach((id) => { $(id).disabled = false; }); }
}
function renderAccessIdentity() {
  window.RelayProfiles.configure(state.profileMode, state.user);
  const user = state.user || {}, manager = allowed("workspace.edit");
  const name = user.name || user.username || "Workspace member";
  $("account-name").textContent = name; $("account-avatar").textContent = name.split(/\s+/).slice(0,2).map((part) => part[0]).join("").toUpperCase();
  $("account-role").textContent = user.role === "manager" ? "Manager" : "Employee";
  $("role-badge").textContent = String(user.role || "employee").toUpperCase(); $("role-badge").className = `role-badge ${user.role === "manager" ? "manager" : "employee"}`;
  $("permission-title").textContent = manager ? "Manager access" : "Employee access";
  $("permission-detail").textContent = manager ? "Register policies, release reviewed revisions, and manage your team. Changes are recorded in the audit trail." : (experimentsEnabled() ? "Run approved pairs, inspect exchanges, and request model changes. Managers control configuration, releases, and team access." : "Assign tasks to approved models, follow agent reports, and request changes. Managers control model releases and team access.");
  $("manager-access").hidden = !allowed("team.manage"); $("access-limited").hidden = allowed("team.manage");
  $("refresh-access").disabled = state.busy;
  $("refresh-access").title = allowed("team.manage") ? "Refresh requests, team membership, and audit trail" : "Refresh your change requests";
}
function renderReleases() {
  if (!state.workspace) return;
  const released = currentPairReleased(), manager = allowed("models.release");
  $("release-summary").textContent = released ? "PAIR RELEASED" : "APPROVAL REQUIRED";
  $("release-summary").classList.toggle("ready", released);
  $("release-help").textContent = state.dirty ? "Save your changes before reviewing the model revisions." : !currentWorkspaceSelected() ? "Viewing a saved session configuration. Select Team workspace to return to the current pair." : manager ? "Review each policy revision before approving it for workspace use." : "A manager must release both exact model revisions before you can assign tasks.";
  $("release-list").innerHTML = state.workspace.agents.length ? state.workspace.agents.map((agent) => {
    const model = modelFor(agent), status = model?.status || "draft", action = status === "released" ? "retire" : "release";
    const canChange = manager && !!model && !state.dirty && !state.busy;
    const approval = model?.updated_at ? sessionDate(model.updated_at) : "";
    return `<article class="release-card"><div class="release-card-top"><span class="agent-letter ${agent.id === "right" ? "agent-b" : ""}">${agent.id === "left" ? "A" : "B"}</span><strong>${esc(agent.name)}</strong><span class="lifecycle-tag ${status === "released" ? "released" : status === "retired" ? "retired" : ""}">${esc(status.toUpperCase())}</span></div><div class="release-source" title="${esc(agent.source.ref)}">${esc(agent.source.ref)}</div><div class="release-meta">${model ? `Revision ${esc(model.revision)} · ${status === "released" ? "Approved for Studio use" : status === "retired" ? "Removed from active use" : "Awaiting manager review"}` : "Save workspace to register this revision"}${approval ? ` · ${esc(typeof approval === "object" ? approval.name || approval.username || "Manager" : approval)}` : ""}</div><div class="release-card-bottom"><span>${status === "released" ? "Exact model configuration approved" : "Unreleased models cannot receive tasks"}</span><button class="button ${action === "release" ? "primary" : ""}" data-lifecycle="${action}" data-model-id="${esc(model?.id || "")}" ${canChange ? "" : "disabled"} title="${manager ? model ? state.dirty ? "Save workspace changes first" : action === "release" ? "Review and release this revision" : "Retire this revision from active use" : "Save this model first" : "Only managers may release or retire models"}">${action === "release" ? "Review & release" : "Retire release"}</button></div></article>`;
  }).join("") : '<p class="release-empty">Add a policy to begin its review and release workflow.</p>';
}
async function refreshApprovalState() {
  try { await refreshModels(); await refreshDiagnostics(); }
  catch (error) { if (!error.stale) notify(`Model approvals could not be refreshed: ${error.message}`); }
}
async function refreshModels() {
  const result = await api("/api/models"); state.models = result.models || []; state.release_ready = !!result.release_ready;
  render();
}
function reviewRelease(id, action) {
  if (!allowed("models.release")) return;
  const model = state.models.find((item) => item.id === id); if (!model) return;
  state.pendingRelease = {id, action, revision: model.revision};
  $("release-dialog-title").textContent = action === "release" ? "Release this revision?" : "Retire this release?";
  $("release-dialog-description").textContent = action === "release" ? "Approve this exact model configuration for workspace use. This does not deploy weights or enable robot actuation." : "Block this revision from new assignments and runtime access. Existing task history is retained.";
  $("release-review").innerHTML = `<strong>${esc(model.spec.name)} · Revision ${esc(model.revision)}</strong><code>${esc(model.spec.source.ref)}${model.spec.source.revision ? `<br>Source revision: ${esc(model.spec.source.revision)}` : ""}</code><p>${esc(families[model.spec.family] || model.spec.family)} · ${esc(robotNames[model.spec.robot] || model.spec.robot)} · ${esc(model.spec.source.kind)}<br>${esc(model.spec.action_contract || "No action contract attached")}</p>`;
  $("release-reason").value = ""; $("release-error").hidden = true;
  $("confirm-release").textContent = action === "release" ? "Release revision" : "Retire release";
  $("release-dialog").showModal();
}
async function changeRelease(event) {
  event.preventDefault(); const pending = state.pendingRelease; if (!pending) return;
  $("confirm-release").disabled = true; $("release-error").hidden = true;
  try {
    await api(`/api/models/${encodeURIComponent(pending.id)}/${pending.action}`, {expected_revision: pending.revision, reason: $("release-reason").value.trim()});
    $("release-dialog").close(); await refreshModels();
    if (state.session) { const history = await api(`/api/sessions/${state.session}`); renderHistory(history); }
    await refreshDiagnostics(); void refreshAccess(); render();
    toast(pending.action === "release" ? "Revision released for Studio use" : "Release retired · new assignments are blocked");
  } catch (error) { $("release-error").textContent = error.message; $("release-error").hidden = false; if (!$("release-dialog").open) notify(error.message); }
  finally { $("confirm-release").disabled = false; }
}
async function refreshAccess() {
  const epoch = authEpoch;
  await refreshRequests();
  if (epoch !== authEpoch) return;
  if (!allowed("team.manage")) return;
  $("refresh-access").disabled = true;
  const results = await Promise.allSettled([api("/api/team"), api("/api/audit")]);
  if (epoch !== authEpoch) return;
  if (results[0].status === "fulfilled") { state.team = results[0].value.users || []; renderTeam(); }
  else $("team-list").innerHTML = `<p class="audit-empty">Could not load team: ${esc(results[0].reason.message)}</p>`;
  if (results[1].status === "fulfilled") renderAudit(results[1].value.events || []);
  else $("audit-list").innerHTML = `<p class="audit-empty">Could not load audit trail: ${esc(results[1].reason.message)}</p>`;
  $("refresh-access").disabled = false;
}
function renderTeam() {
  $("team-count").textContent = `${state.team.length} MEMBER${state.team.length === 1 ? "" : "S"}`;
  $("team-list").innerHTML = state.team.map((user) => `<div class="team-row" data-team-id="${esc(user.id)}"><div class="team-person"><span class="account-avatar">${esc((user.name || user.username).slice(0,2).toUpperCase())}</span><div><strong>${esc(user.name || user.username)}${user.id === state.user?.id ? '<span class="team-self">YOU</span>' : ""}</strong><small>@${esc(user.username)}</small></div></div><select data-team-role aria-label="Role for ${esc(user.name || user.username)}"><option value="employee" ${user.role === "employee" ? "selected" : ""}>Employee</option><option value="manager" ${user.role === "manager" ? "selected" : ""}>Manager</option></select><label class="team-active"><input type="checkbox" data-team-active ${user.active ? "checked" : ""}>Active</label><button class="button" data-team-save="${esc(user.id)}">Apply</button></div>`).join("");
}
function renderAudit(events) {
  const labels = {"register":"Model draft registered", "release":"Model revision released", "retire":"Model release retired", "model.release":"Model revision released", "model.retire":"Model release retired", "model.draft":"Model draft registered", "model.register":"Model registered", "workspace.update":"Workspace updated", "team.invite":"Teammate invited", "team.update":"Member access updated", "auth.setup":"Workspace manager created", "auth.join":"Invitation accepted", "auth.login":"Account signed in", "auth.logout":"Account signed out"};
  $("audit-list").innerHTML = events.length ? events.slice(0,30).map((event) => {
    const actor = typeof event.actor === "object" && event.actor ? event.actor.name || event.actor.username || event.actor.id : event.actor_name || event.actor_username || event.username || event.actor || event.actor_id || "Workspace";
    const details = event.details || event.detail || {}, subject = event.spec?.name || event.target || event.model_id || event.target_id || event.subject || details.model_id || details.username || "";
    return `<article class="audit-row"><span class="audit-mark">${icon(String(event.action).includes("model") ? "box" : "shield")}</span><div class="audit-content"><strong>${esc(labels[event.action] || String(event.action || "Workspace change").replace(/[._]/g," "))}</strong><span>${esc(actor)}${subject ? ` · ${esc(subject)}` : ""}${event.reason || details.reason ? ` · ${esc(event.reason || details.reason)}` : ""}</span></div><time>${esc(sessionDate(event.created_at || event.timestamp || event.at))}</time></article>`;
  }).join("") : '<p class="audit-empty">Recorded workspace activity will appear here.</p>';
}
async function inviteMember(event) {
  event.preventDefault(); $("invite-submit").disabled = true;
  try {
    const result = await api("/api/team/invitations", {name: $("invite-name").value.trim(), username: $("invite-username").value.trim(), role: $("invite-role").value});
    $("invitation-code").textContent = result.code;
    $("invitation-help").textContent = `Share this code directly with ${$("invite-name").value.trim()}. Expires ${sessionDate(result.expires_at)}. They can use “Accept invitation” on the sign-in page.`;
    $("invitation-result").hidden = false; $("invite-form").reset(); void refreshAccess();
  } catch (error) { notify(error.message); $("notice").scrollIntoView({behavior:"smooth", block:"center"}); }
  finally { $("invite-submit").disabled = false; }
}
async function updateMember(id) {
  const row = [...document.querySelectorAll("[data-team-id]")].find((item) => item.dataset.teamId === id); if (!row) return;
  const button = row.querySelector("[data-team-save]"); button.disabled = true;
  try {
    await api(`/api/team/${encodeURIComponent(id)}`, {role: row.querySelector("[data-team-role]").value, active: row.querySelector("[data-team-active]").checked});
    toast("Team access updated");
    if (id === state.user?.id) { const boot = await api("/api/bootstrap"); state.user = boot.user; state.permissions = boot.permissions; render(); }
    await refreshAccess();
  } catch (error) { notify(error.message); $("notice").scrollIntoView({behavior:"smooth", block:"center"}); }
  finally { button.disabled = false; }
}
const requestKinds = {add: "Add model", edit: "Edit model", release: "Release revision", retire: "Retire release"};
function renderRequestModels() {
  const selected = $("request-model").value;
  $("request-model").innerHTML = '<option value="">No specific model</option>' + state.models.map((model) => `<option value="${esc(model.id)}">${esc(model.spec.name)} · ${esc(model.status)} · r${esc(model.revision)}</option>`).join("");
  if (state.models.some((model) => model.id === selected)) $("request-model").value = selected;
  $("requests-description").textContent = allowed("models.manage") ? "Review team proposals and record the outcome after applying approved changes." : "Submit a proposal to your manager. You can follow the status of your own requests here.";
}
async function refreshRequests() {
  if (!state.authenticated) return;
  try { const result = await api("/api/requests"); state.requests = result.requests || []; renderRequests(); }
  catch (error) { if (error.stale) return; $("request-list").innerHTML = `<p class="audit-empty">Could not load requests: ${esc(error.message)}</p>`; }
}
function renderRequests() {
  const pending = state.requests.filter((request) => request.status === "pending").length;
  $("request-count").textContent = `${pending} PENDING`;
  $("request-list").innerHTML = state.requests.length ? state.requests.slice(0,30).map((request) => {
    const model = state.models.find((item) => item.id === request.model_id);
    return `<article class="change-request"><div class="change-request-header"><strong>${esc(requestKinds[request.kind] || request.kind)}${model ? ` <span>· ${esc(model.spec.name)}</span>` : ""}</strong><span class="lifecycle-tag ${request.status === "resolved" ? "released" : request.status === "declined" ? "retired" : ""}">${esc(request.status.toUpperCase())}</span></div><p>${esc(request.message)}</p><div class="change-request-meta"><span>${esc(request.username || "Team member")} · ${esc(sessionDate(request.created_at))}</span>${request.status === "pending" && allowed("models.manage") ? `<div><button class="button" data-review-request="${esc(request.id)}" data-resolution="declined">Decline</button><button class="button" data-review-request="${esc(request.id)}" data-resolution="resolved">Mark resolved</button></div>` : ""}</div>${request.reason ? `<div class="request-outcome"><strong>Manager response</strong><span>${esc(request.reason)}</span></div>` : ""}</article>`;
  }).join("") : '<p class="audit-empty">No model change requests yet.</p>';
}
async function submitChangeRequest(event) {
  event.preventDefault(); $("request-submit").disabled = true;
  try {
    const body = {kind: $("request-kind").value, message: $("request-message").value.trim()};
    if ($("request-model").value) body.model_id = $("request-model").value;
    await api("/api/requests", body); $("request-form").reset(); await refreshRequests(); toast("Request submitted for manager review");
  } catch (error) { notify(error.message); $("notice").scrollIntoView({behavior:"smooth", block:"center"}); }
  finally { $("request-submit").disabled = false; }
}
function reviewRequest(id, status) {
  const request = state.requests.find((item) => item.id === id); if (!request || !allowed("models.manage")) return;
  state.pendingRequest = {id, status};
  $("request-dialog-title").textContent = status === "resolved" ? "Mark this request resolved?" : "Decline this request?";
  $("request-review").innerHTML = `<strong>${esc(requestKinds[request.kind] || request.kind)} · ${esc(request.username)}</strong><p>${esc(request.message)}</p>`;
  $("confirm-request").textContent = status === "resolved" ? "Mark resolved" : "Decline request";
  $("resolve-reason").value = ""; $("resolve-error").hidden = true; $("request-dialog").showModal();
}
async function resolveRequest(event) {
  event.preventDefault(); const pending = state.pendingRequest; if (!pending) return;
  $("confirm-request").disabled = true; $("resolve-error").hidden = true;
  try {
    await api(`/api/requests/${encodeURIComponent(pending.id)}/resolve`, {status: pending.status, reason: $("resolve-reason").value.trim()});
    $("request-dialog").close(); await refreshAccess(); toast("Request outcome recorded");
  } catch (error) { $("resolve-error").textContent = error.message; $("resolve-error").hidden = false; }
  finally { $("confirm-request").disabled = false; }
}

function openAccess() { $("access-section").scrollIntoView({behavior:"smooth", block:"start"}); void refreshAccess(); }
function showAccessRecovery(message) {
  clearInterval(polling); document.body.classList.add("auth-pending");
  $("auth-screen").hidden = false; $("auth-screen").classList.add("auth-recovery");
  $("auth-eyebrow").textContent = "CONNECTION NEEDS ATTENTION"; $("auth-title").textContent = "Let’s reconnect.";
  $("auth-subtitle").textContent = "Your account has not been changed. Retry when the local Studio backend is available.";
  $("auth-error").textContent = message; $("auth-error").hidden = false; $("auth-retry").hidden = false;
  $("auth-recovery-logout").hidden = !state.token; $("auth-submit").disabled = true;
}
function bindAccess() {
  $("auth-retry").addEventListener("click", checkAccess);
  $("auth-recovery-logout").addEventListener("click", signOut);
  $("auth-form").addEventListener("submit", authenticate);
  document.querySelectorAll("[data-auth-mode]").forEach((button) => button.addEventListener("click", () => setAuthMode(button.dataset.authMode)));
  for (const id of ["logout-button", "logout-top-button"]) $(id).addEventListener("click", signOut);
  for (const id of ["nav-access", "permission-access"]) $(id).addEventListener("click", openAccess);
  $("refresh-access").addEventListener("click", refreshAccess);
  $("refresh-releases").addEventListener("click", async () => { $("refresh-releases").disabled = true; try { await refreshModels(); await refreshDiagnostics(); } catch (error) { notify(error.message); } finally { $("refresh-releases").disabled = false; } });
  $("release-list").addEventListener("click", (event) => { const button = event.target.closest("[data-lifecycle]"); if (button && !button.disabled) reviewRelease(button.dataset.modelId, button.dataset.lifecycle); });
  $("release-form").addEventListener("submit", changeRelease);
  for (const id of ["close-release", "cancel-release"]) $(id).addEventListener("click", () => $("release-dialog").close());
  $("invite-form").addEventListener("submit", inviteMember);
  $("request-form").addEventListener("submit", submitChangeRequest);
  $("request-list").addEventListener("click", (event) => { const button = event.target.closest("[data-review-request]"); if (button) reviewRequest(button.dataset.reviewRequest, button.dataset.resolution); });
  $("resolve-form").addEventListener("submit", resolveRequest);
  for (const id of ["close-request", "cancel-request"]) $(id).addEventListener("click", () => $("request-dialog").close());
  $("team-list").addEventListener("click", (event) => { const button = event.target.closest("[data-team-save]"); if (button) void updateMember(button.dataset.teamSave); });
  $("copy-invitation").addEventListener("click", async () => { try { await navigator.clipboard.writeText($("invitation-code").textContent); toast("Invitation code copied"); } catch { toast("Select and copy the invitation code"); } });
  $("dismiss-invitation").addEventListener("click", () => { $("invitation-code").textContent = ""; $("invitation-result").hidden = true; });
}

init();
