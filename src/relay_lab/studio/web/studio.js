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
let toastTimer, polling, historySignature = "", activeUpload, diagnosticRevision = 0;
Object.assign(state, {health: null, diagnostics: null, sessions: [], history: null, selectedMessage: null, readOnly: false, failed: false});

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-Relay-Token": state.token}, body: JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function notify(message, success = false) { $("notice").textContent = message; $("notice").className = `notice${success ? " success" : ""}`; $("notice").hidden = false; }
function toast(message) { clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false; toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500); }
function clearSession() {
  historySignature = "";
  state.session = null; state.results = []; state.history = null; state.selectedMessage = null; state.readOnly = false; state.failed = false; state.diagnostics = null; clearInterval(polling);
  $("receiver-panel").hidden = true;
  void refreshDiagnostics();
  $("stream-events").replaceChildren(); $("stream-count").textContent = "0 PACKETS";
  $("stream-empty").hidden = false; $("stream-metrics").hidden = true;
}
function changed() { clearSession(); state.dirty = true; $("notice").hidden = true; render(); }
function isExample() {
  return state.workspace.agents.length === 2 && state.workspace.agents.every((a, i) => {
    const b = state.example.agents[i];
    return a.id === b.id && a.robot === "arm" && a.family === b.family && JSON.stringify(a.source) === JSON.stringify(b.source);
  });
}
function renderModel(arm) {
  const a = state.workspace.agents.find((m) => m.id === arm), label = arm === "left" ? "A" : "B";
  if (!a) {
    $(`slot-${arm}`).innerHTML = `<button class="empty-card" data-edit="${arm}"><span class="add-mark">${icon("plus")}</span><strong>Add model ${label}</strong><small>Weights, repository or endpoint</small></button>`;
    return;
  }
  const available = isExample() && state.capabilities.sample_available;
  const native = available ? (state.capabilities.adapter?.source_dimensions?.[arm] || (arm === "left" ? 960 : 1024)) : null;
  $(`slot-${arm}`).innerHTML = `<article class="model-card" data-agent="${arm}"><div class="model-card-top"><span class="agent-letter">${label}</span><strong>${esc(a.name)}</strong><button class="edit-button" data-edit="${arm}" aria-label="Edit ${esc(a.name)}">···</button></div><div class="robot-illustration">${robot(a.robot)}</div><div class="model-spec"><div class="model-family">${families[a.family]}<span>${a.source.kind === "upload" ? "WEIGHTS" : "POLICY"}</span></div><div class="model-repo" title="${esc(a.source.ref)}">${esc(a.source.ref)}</div><div class="model-native"><span>FEATURE CONTRACT</span><strong>${native ? `${native}D → 64D` : "UNCONFIGURED"}</strong></div></div><div class="model-card-footer"><span><i class="status-dot ${available ? "" : "neutral"}"></i>${available ? "Pinned feature contract" : "Registered · not loaded"}</span><span>${a.rig ? "Rig registered" : "No live rig"}</span></div></article>`;
}
function render() {
  if (!state.workspace) return;
  const w = state.workspace, mode = modes[w.mode], exact = isExample();
  $("breadcrumb-name").textContent = w.name;
  $("objective").value = w.objective;
  $("save-state").textContent = state.dirty ? "Unsaved changes" : "Saved locally";
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
  $("session-badge").textContent = state.readOnly ? "ARCHIVED VIEW" : state.failed ? "CHANNEL STOPPED" : state.session ? "CHANNEL OPEN" : "DRAFT";
  $("connection-state").textContent = state.readOnly ? "Read-only history" : state.session ? "Session confirmed" : "Not connected";
  $("canvas-status").textContent = state.readOnly ? "Read-only journal · open a new channel to exchange" : state.failed ? "Channel stopped · open a new session to continue" : state.session ? "Communication session open · hardware inactive" : w.agents.length === 2 ? (["latent", "hybrid"].includes(w.mode) && !exact ? "A trained adapter is needed for this pair" : "Pair configured · ready to open a channel") : "Add two models to create a channel";
  $("canvas-status-dot").classList.toggle("neutral", !state.session);
  const needsAdapter = ["latent", "hybrid"].includes(w.mode) && !exact;
  let note = "Add your models to check the available communication paths.";
  if (w.agents.length === 2) {
    note = needsAdapter ? "These checkpoints need trained latent adapters. Try the LLM paths with local reports, or load the supported example pair." : exact ? "Example adapters match these pinned checkpoints. Captured features are available for a real communication probe." : "Text reports can use Qwen now. Your policy worker must load the weights and consume the returned context.";
  }
  $("compatibility").querySelector("p").textContent = note;
  $("connect-button").disabled = state.busy || w.agents.length !== 2 || needsAdapter || (state.diagnostics && !state.diagnostics.ready) || (!!state.session && !state.readOnly && !state.failed);
  $("connect-button").innerHTML = icon(state.session && !state.readOnly && !state.failed ? "check" : "link") + (state.readOnly || state.failed ? "Open a new channel" : state.session ? "Pair connected" : "Connect pair");
  $("probe-button").disabled = state.busy || !state.session;
  $("probe-source").querySelector('[value="captured"]').disabled = !exact || !state.capabilities.sample_available;
  if (!exact || !state.capabilities.sample_available) $("probe-source").value = "manual";
  $("qwen-state").textContent = state.capabilities.llm_available ? "Installed" : "Setup needed";
  $("canvas").classList.toggle("busy", state.busy);
  for (const id of ["save-button", "example-button", "export-button", "direction", "objective"]) $(id).disabled = state.busy;
  document.querySelectorAll("[data-mode], [data-edit]").forEach((b) => { b.disabled = state.busy; });
  updateProbeSource();
}
function updateProbeSource() {
  const captured = $("probe-source").value === "captured";
  $("condition-field").hidden = !captured;
  $("manual-reports").hidden = captured;
  const latent = ["latent", "hybrid"].includes(state.workspace.mode);
  $("probe-note").textContent = captured ? "Captured foundation features; fresh adapter / Qwen inference; verified Spacetime readback. No robot motion is executed." : latent ? "Latent mode also needs native feature vectors. Use captured observations or the API bridge to supply features; this text-only form cannot generate them." : "These are reports you supply, not sensor measurements. Qwen processes them and Spacetime commits the resulting peer messages.";
  if (!captured && latent) $("probe-button").disabled = true;
  else $("probe-button").disabled = state.busy || !state.session || state.readOnly || state.failed;
  updateObservations();
}
async function save() {
  state.workspace = await api("/api/workspace", state.workspace);
  state.dirty = false; render();
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
  $("source-ref").placeholder = {huggingface: "owner/model", local: "/path/to/weights-or-model-directory", endpoint: "http://127.0.0.1:8000/policy", upload: ""}[source];
}
function editModel(arm) {
  const a = state.workspace.agents.find((m) => m.id === arm);
  $("model-form").reset(); state.upload = null;
  $("agent-id").value = arm; $("model-dialog-slot").textContent = `MODEL ${arm === "left" ? "A" : "B"}`;
  $("agent-name").value = a?.name || ""; state.robot = a?.robot || "arm";
  $("agent-family").value = a?.family || "smolvla";
  $("source-kind").value = a?.source.kind || "huggingface";
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
    if (kind === "upload" && !state.upload) throw new Error("Choose and upload your safetensors weights first.");
    const a = {id: $("agent-id").value, name: $("agent-name").value.trim(), robot: state.robot,
      family: $("agent-family").value, source: {kind, ref: kind === "upload" ? state.upload.asset : $("source-ref").value.trim(), revision: kind === "huggingface" ? $("source-revision").value.trim() : ""},
      rig: $("rig-endpoint").value.trim(), action_contract: $("action-contract").value.trim()};
    const next = {...state.workspace, agents: [...state.workspace.agents.filter((m) => m.id !== a.id), a].sort((x, y) => x.id.localeCompare(y.id))};
    const value = await api("/api/workspace", next);
    state.workspace = value; clearSession(); state.dirty = false; render();
    $("model-dialog").close(); toast(`${a.name} registered`);
  } catch (error) { $("form-error").textContent = error.message; $("form-error").hidden = false; }
  finally { $("register-model").disabled = false; }
}
function upload(file) {
  if (!file) return;
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
  state.history = data;
  state.failed = !!data.failed;
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
  const session = state.session;
  if (!session || state.busy) return;
  try { const data = await api(`/api/sessions/${session}`); if (session === state.session) renderHistory(data); }
  catch (error) { clearInterval(polling); notify(`Could not refresh the durable stream: ${error.message}`); }
}
async function connect() {
  state.busy = true; $("notice").hidden = true; render();
  $("connect-button").textContent = "Opening channel…";
  try {
    await save();
    const result = await api("/api/sessions", state.workspace);
    clearInterval(polling); state.results = []; state.session = result.session_id; state.readOnly = false; state.failed = false; state.history = null; state.selectedMessage = null; historySignature = "";
    $("stream-events").replaceChildren(); $("stream-empty").hidden = false; $("stream-metrics").hidden = true; $("receiver-panel").hidden = true;
    void refreshSessions();
    toast("Pair connected through SpacetimeDB");
    polling = setInterval(refreshStream, 3000);
  } catch (error) { notify(error.message); }
  finally { state.busy = false; render(); }
}
async function probe() {
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
  $("api-example").textContent = `import json, uuid\nfrom urllib.request import Request, urlopen\n\nBASE = "${location.origin}"\n\ndef call(path, data=None, token=None):\n    headers = {"Content-Type": "application/json"}\n    if token: headers["X-Relay-Token"] = token\n    body = json.dumps(data).encode() if data is not None else None\n    return json.load(urlopen(Request(BASE + path, body, headers)))\n\nboot = call("/api/bootstrap")\n# The two local reports come from your separate policy workers.\nsession = call("/api/sessions", {**boot["workspace"], "mode": "mediated"}, boot["token"])\nsid = session["session_id"]\nresult = call(f"/api/sessions/{sid}/exchange", {\n    "request_id": str(uuid.uuid4()),\n    "source": "manual",\n    "reports": {"left": "Path blocked.", "right": "Ready to move."}\n}, boot["token"])\n\nfor packet in result["messages"]:\n    print(packet["recipient"], packet["payload"])\n    # Feed context into the matching policy adapter.\n    # Your runtime remains responsible for actuation.\n`;
  $("bridge-dialog").showModal();
}
function bind() {
  $("canvas").addEventListener("click", (event) => { const b = event.target.closest("[data-edit]"); if (b) editModel(b.dataset.edit); });
  $("mode-options").addEventListener("click", (event) => { const b = event.target.closest("[data-mode]"); if (b && !state.busy) { state.workspace.mode = b.dataset.mode; changed(); } });
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
    try { state.workspace = await api("/api/workspace", next); clearSession(); state.dirty = false; render(); $("model-dialog").close(); toast("Model removed from the team"); }
    catch (error) { $("form-error").textContent = error.message; $("form-error").hidden = false; }
  });
  $("example-button").addEventListener("click", async () => {
    try { const value = await api("/api/workspace", state.example); state.workspace = value; clearSession(); state.dirty = false; $("probe-source").value = "captured"; $("notice").hidden = true; render(); toast("Example pair loaded · weights are not loaded into memory"); }
    catch (error) { notify(error.message); }
  });
  $("save-button").addEventListener("click", async () => { try { await save(); toast("Workspace saved locally"); } catch (e) { notify(e.message); } });
  $("export-button").addEventListener("click", exportWorkspace);
  $("connect-button").addEventListener("click", connect); $("probe-button").addEventListener("click", probe);
  $("probe-source").addEventListener("change", updateProbeSource);
  $("condition").addEventListener("change", updateObservations);
  $("refresh-health").addEventListener("click", async () => { await refreshHealth(); await refreshDiagnostics(); });
  $("refresh-sessions").addEventListener("click", refreshSessions);
  $("session-list").addEventListener("click", (event) => { const b = event.target.closest("[data-session]"); if (b && !state.busy) void openSession(b.dataset.session, b.dataset.resume === "true"); });
  $("stream-events").addEventListener("click", (event) => { const b = event.target.closest("[data-packet]"); if (b) { state.selectedMessage = b.dataset.packet; renderPackets(); } });
  $("nav-history").addEventListener("click", () => $("session-section").scrollIntoView({behavior:"smooth", block:"start"}));
  $("channel-node").addEventListener("click", () => { $("mode-options").scrollIntoView({behavior: "smooth", block: "center"}); $("mode-options").querySelector("button.active").focus(); });
  $("nav-team").addEventListener("click", () => window.scrollTo({top: 0, behavior: "smooth"}));
  $("nav-stream").addEventListener("click", () => $("stream-panel").scrollIntoView({behavior: "smooth", block: "center"}));
  for (const id of ["nav-bridge", "bridge-button"]) $(id).addEventListener("click", showBridge);
  $("close-bridge").addEventListener("click", () => $("bridge-dialog").close());
  $("copy-api").addEventListener("click", async () => { try { await navigator.clipboard.writeText($("api-example").textContent); toast("Python example copied"); } catch { toast("Select and copy the example above"); } });
  window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
}
async function init() {
  try { const boot = await api("/api/bootstrap"); Object.assign(state, boot); bind(); render(); await Promise.allSettled([refreshHealth(), refreshDiagnostics(), refreshSessions()]); }
  catch (error) { notify(`Studio could not start: ${error.message}`); }
}
function renderPipeline() {
  const latent = ["latent","hybrid"].includes(state.workspace.mode);
  const nodes = latent ? [["FROZEN BACKBONE","960 / 1024D"],["TRAINED ENCODER","64D message"],["SPACETIMEDB","Commit + verify"],["PEER DECODER","32D context"],["REPLACEMENT HEAD","Action preview"]] : [["LOCAL EVIDENCE","Text reports"],["QWEN 3",state.workspace.mode === "orchestrated" ? "Joint plan" : "Peer context"],["SPACETIMEDB","Commit + verify"],["POLICY WORKER","Receive text"]];
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
    $("health-db").textContent = "Readiness check failed"; $("health-db").title = error.message;
    $("health-adapter").textContent = "Readiness unavailable"; $("health-qwen").textContent = "Readiness unavailable";
    $("sidebar-db-label").textContent = "CHECK UNAVAILABLE";
  } finally { $("refresh-health").disabled = false; }
}
async function refreshDiagnostics() {
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
  $("refresh-sessions").disabled = true;
  try { const data = await api("/api/sessions"); state.sessions = data.sessions || []; renderSessions(); }
  catch (error) { $("session-list").innerHTML = `<p class="empty-history session-error">Could not read session history: ${esc(error.message)}</p>`; }
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
  state.busy = true; render();
  try {
    const data = await api(`/api/sessions/${id}${resume ? "/resume" : ""}`, resume ? {} : undefined);
    clearInterval(polling); historySignature = ""; state.session = id; state.selectedMessage = null; state.readOnly = data.status === "inspect_only"; state.failed = !!data.failed;
    const latest = data.results?.at(-1);
    if (["clear","left_obstacle","right_obstacle"].includes(latest?.condition)) $("condition").value = latest.condition;
    if (latest?.source) $("probe-source").value = latest.source;
    if (data.workspace) { state.workspace = data.workspace; await api("/api/workspace", state.workspace); state.dirty = false; }
    state.diagnostics = null; render(); renderHistory(data); renderSessions();
    if (!state.readOnly) polling = setInterval(refreshStream,3000);
    toast(state.readOnly ? "Inspecting the durable message journal" : "Session resumed from SpacetimeDB");
    $("stream-panel").scrollIntoView({behavior:"smooth",block:"start"});
    void refreshDiagnostics();
  } catch (error) { notify(error.message); }
  finally { state.busy = false; render(); renderSessions(); }
}
init();
