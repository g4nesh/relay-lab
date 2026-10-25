/* Analytics reads authenticated aggregates. It never infers robot success from a fast exchange. */
(() => {
  "use strict";
  const el = (id) => document.getElementById(id);
  const html = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const number = (v, digits = 0) => Number.isFinite(v) ? v.toLocaleString(undefined, {maximumFractionDigits: digits}) : "—";
  const percent = (v) => Number.isFinite(v) ? `${number(v * 100, 1)}%` : "—";
  const duration = (v) => Number.isFinite(v) ? (v >= 1000 ? `${number(v / 1000, 2)} s` : `${number(v, 2)} ms`) : "Not retained";
  const bytes = (v) => Number.isFinite(v) ? (v >= 1024 ? `${number(v / 1024, 1)} KiB` : `${number(v)} B`) : "—";
  const labels = {none:"No private messages", embeddings:"Embeddings", text:"Structured text", structured_text:"Structured text", hybrid:"Hybrid", latent:"Learned latents", mediated:"LLM mediator", orchestrated:"LLM orchestrator"};
  const colors = {none:"#c3c9ce", embeddings:"#183e4a", latent:"#183e4a", text:"#5b97a6", structured_text:"#5b97a6", hybrid:"#8d99ae", mediated:"#5b97a6", orchestrated:"#8d99ae"};
  const chip = (label, kind = "") => `<span class="an-chip ${kind}">${html(label)}</span>`;
  const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const empty = (title, detail) => `<div class="an-empty"><span>${icon("chart")}</span><strong>${html(title)}</strong><p>${html(detail)}</p></div>`;
  const list = (items) => `<ul class="an-caveats">${(items || []).map((x) => `<li>${html(x)}</li>`).join("")}</ul>`;
  let options, snapshot = null, generation = 0, request = 0, estimateRequest = 0, loading = false, estimate = null;
  let filters = {cohort:"factory", mode:"all", scenario:"all"};

  function skeleton() {
    return `<div class="an-heading"><div><div class="eyebrow"><span class="tiny-square"></span> PERFORMANCE INTELLIGENCE</div><h1>Coordination, <span>measured.</span></h1><p>Follow the evidence from robot decisions to the infrastructure behind them.</p></div><div class="an-heading-actions"><button class="button" id="an-sources">${icon("code")}Evidence & method</button><button class="button" id="an-refresh" aria-label="Refresh analytics">↻ <span>Refresh</span></button></div></div>
    <div class="an-snapshot"><span><i class="status-dot neutral" id="an-snapshot-dot"></i><span id="an-snapshot-label">Loading authenticated workspace evidence…</span></span><button class="text-button" id="an-export" disabled>Export this view ${icon("arrow")}</button></div>
    <div id="an-error" class="an-error" role="alert" hidden></div>
    <div id="an-content" aria-busy="true"><div class="an-loading"><div class="an-loading-card"></div><div class="an-loading-card"></div><div class="an-loading-card"></div><div class="an-loading-plot"></div></div></div>
    <dialog id="an-evidence-dialog" class="an-evidence-dialog" aria-labelledby="an-evidence-title"><div class="dialog-heading"><div class="eyebrow">Reproducible evidence</div><button class="close-button" id="an-close-evidence" aria-label="Close evidence">×</button></div><h2 id="an-evidence-title">What is behind the numbers?</h2><p class="dialog-subtitle">Artifacts, definitions, and scope travel with every export.</p><div id="an-evidence-body"></div><div class="an-evidence-footer"><button class="button primary" id="an-export-evidence">Export current evidence</button></div></dialog>`;
  }
  function mount(config) {
    options = config;
    el("analytics-view").innerHTML = skeleton();
    el("an-refresh").addEventListener("click", () => void refresh());
    el("an-export").addEventListener("click", exportView);
    el("an-export-evidence").addEventListener("click", exportView);
    el("an-sources").addEventListener("click", showEvidence);
    el("an-close-evidence").addEventListener("click", () => el("an-evidence-dialog").close());
    el("an-content").addEventListener("change", changeFilter);
    el("an-content").addEventListener("input", (event) => {
      if (!event.target.closest("#an-effort-form")) return;
      estimate = null; estimateRequest++;
      el("an-calculate").disabled = false;
      el("an-estimate-error").hidden = true;
      el("an-estimate-result").innerHTML = empty("Assumptions updated", "Calculate again to see the result for these inputs. Previous results are excluded from the export.");
    });
    el("an-content").addEventListener("submit", (event) => { if (event.target.id === "an-effort-form") { event.preventDefault(); void calculateEstimate(); } });
    el("an-content").addEventListener("click", (event) => { if (event.target.closest("[data-an-evidence]")) showEvidence(); });
  }
  function reset() {
    generation++; request++; estimateRequest++;
    snapshot = null; estimate = null; loading = false; filters = {cohort:"factory", mode:"all", scenario:"all"};
    const dialog = el("an-evidence-dialog"); if (dialog?.open) dialog.close();
    el("an-evidence-body")?.replaceChildren();
    el("an-content")?.replaceChildren();
    if (el("an-export")) el("an-export").disabled = true;
    if (el("an-snapshot-label")) el("an-snapshot-label").textContent = "Sign in to read workspace evidence.";
    if (el("an-error")) el("an-error").hidden = true;
  }
  function show() { if (!snapshot && !loading) void refresh(); }
  async function refresh() {
    if (!options.authenticated()) return;
    const revision = ++request, identity = generation;
    loading = true; el("an-refresh").disabled = true; el("an-error").hidden = true;
    el("an-content").setAttribute("aria-busy", "true");
    if (!snapshot) el("an-content").innerHTML = '<div class="an-loading"><div class="an-loading-card"></div><div class="an-loading-card"></div><div class="an-loading-card"></div><div class="an-loading-plot"></div></div>';
    try {
      const data = await options.api("/api/analytics");
      if (revision !== request || identity !== generation || !options.authenticated()) return;
      snapshot = data; estimate = null; estimateRequest++; render();
      el("an-snapshot-dot").classList.remove("neutral");
      const time = data.generated_at ? new Date(typeof data.generated_at === "number" ? data.generated_at * 1000 : data.generated_at).toLocaleString(undefined, {month:"short",day:"numeric",hour:"numeric",minute:"2-digit"}) : "just now";
      el("an-snapshot-label").textContent = `Evidence snapshot · ${time} · each section has its own population`;
      el("an-export").disabled = false;
    } catch (error) {
      if (identity !== generation || error.stale || !options.authenticated()) return;
      el("an-error").textContent = `Analytics could not refresh. ${error.message}${snapshot ? " The last loaded snapshot is still shown." : " Use Refresh to try again."}`;
      el("an-error").hidden = false;
      if (!snapshot) el("an-content").innerHTML = empty("Evidence is unavailable", "No estimates or success claims are substituted for missing measurements.");
    } finally {
      if (revision === request && identity === generation) { loading = false; el("an-refresh").disabled = false; el("an-content").setAttribute("aria-busy", "false"); }
    }
  }
  function applyVisualValues() {
    el("an-content").querySelectorAll("[data-an-color], [data-an-width], [data-an-left]").forEach((node) => {
      if (/^#[a-f0-9]{6}$/i.test(node.dataset.anColor || "")) node.style.backgroundColor = node.dataset.anColor;
      for (const [key, property] of [["anWidth", "width"], ["anLeft", "left"]]) {
        if (node.dataset[key] !== undefined && Number.isFinite(Number(node.dataset[key]))) node.style[property] = Math.max(0, Math.min(100, Number(node.dataset[key]))) + "%";
      }
    });
  }
  function metric(label, value, detail, tag, extra = "") {
    return `<article class="an-metric ${extra}"><div class="an-metric-label">${html(label)}${chip(tag)}</div><strong>${html(value)}</strong><p>${html(detail)}</p></article>`;
  }
  function render() {
    const data = snapshot, headline = data.headline || {}, ops = data.operations || {}, factory = data.factory || {};
    const gain = !Number.isFinite(headline.value) ? "Not established" : ["pp", "percentage_points", "percentage points"].includes(headline.unit) ? `${headline.value > 0 ? "+" : ""}${number(headline.value, 1)} pp` : Number.isFinite(headline.value) ? number(headline.value, 1) + (headline.unit ? ` ${headline.unit}` : "") : "Not established";
    const totalOps = ops.status !== "unavailable" && Number.isFinite(ops.exchanges) ? number(ops.exchanges) : "Unavailable";
    el("an-content").innerHTML = `<div class="an-metrics">${metric(headline.label || "Coordination benefit", gain, headline.detail || "Matched task trials are needed before a benefit can be claimed.", Number.isFinite(headline.value) ? "Measured" : "Unavailable", "an-metric-highlight")}${metric("Observed exchanges", totalOps, ops.status === "unavailable" ? "The current journal could not be measured. Historical evidence is labeled separately below." : `${number(ops.sessions)} Studio sessions · observed committed journal exchanges`, "Observed")}${metric("Engineering time saved", "Model your scenario", "No human engineering baseline has been timed. Adjust explicit assumptions below.", "Estimate")}</div>
      <div class="an-section-heading"><div><div class="section-overline">01 / ROBOT COORDINATION</div><h2>Better together? Test the difference.</h2><p>Compare communication channels inside a matched experiment.</p></div>${chip("Simulation / captured inputs")}</div>
      <section class="an-panel an-coordination"><div class="an-filters"><label>Evidence cohort<select id="an-cohort"><option value="factory">Factory · private routing</option><option value="cloth">Two arms · cloth task</option><option value="learned-static">Learned adapters · held-out decisions</option><option value="learned-transport">Learned adapters · delayed transport</option></select></label><label>Communication channel<select id="an-mode"></select></label><label id="an-scenario-field">Condition<select id="an-scenario"></select></label><span class="an-filter-scope">Filters apply to this experiment<br>and its exported rows.</span></div><div id="an-cohort-body"></div></section>
      <div class="an-section-heading"><div><div class="section-overline">02 / COORDINATION INFRASTRUCTURE</div><h2>Where the milliseconds go.</h2><p>Measured exchange timings across the Spacetime coordination loop.</p></div>${chip("SPACETIMEDB")}</div>
      ${renderOperations(ops)}
      <div class="an-section-heading"><div><div class="section-overline">03 / WORKSPACE OPERATIONS</div><h2>From model registration to approved use.</h2><p>Current model inventory and observed session activity, kept separate from task outcomes.</p></div>${chip("Current workspace")}</div>${renderGovernance(data.governance || {}, ops)}
      <div class="an-section-heading"><div><div class="section-overline">04 / ENGINEERING EFFORT</div><h2>Make the assumptions visible.</h2><p>A planning calculator for integration work. Replace example inputs with your own estimates.</p></div>${chip("Illustrative · not measured", "scenario")}</div>${renderEffort(data.effort_model || {})}
      <div class="an-bottom-note"><span>${icon("shield")} Evidence before efficiency claims.</span><p>Simulation outcomes, transport speed, and engineering hours answer different questions. Faster database access alone does not establish a faster or safer robot task.</p><button class="text-button" data-an-evidence>Inspect sources & definitions ↗</button></div><footer><span>Relay Studio <span class="muted">/</span> ANALYTICS</span><span>${number(viewSources().length)} source artifacts · reproducible snapshots · no production ROI claim</span></footer>`;
    renderCohort();
    applyVisualValues();
    el("an-estimate-result").innerHTML = empty("Your scenario, not a benchmark", "Adjust the assumptions and calculate a planning estimate.");
  }
  function cohort() {
    const map = {factory:snapshot.factory, cloth:snapshot.cloth, "learned-static":snapshot.learned, "learned-transport":snapshot.learned};
    const data = map[filters.cohort] || {};
    const raw = filters.cohort === "learned-static" ? data.static : filters.cohort === "learned-transport" ? data.transport : data.rows;
    return {data, rows:Array.isArray(raw) ? raw : [], learned:filters.cohort.startsWith("learned"), title:{factory:"Factory routing task",cloth:"Two-arm cloth task","learned-static":"Held-out peer decisions","learned-transport":"Peer decisions with delayed messages"}[filters.cohort]};
  }
  function selectedRows() { return cohort().rows.filter((r) => (filters.mode === "all" || r.mode === filters.mode) && (filters.scenario === "all" || r.scenario === filters.scenario)); }
  function changeFilter(event) {
    if (!snapshot) return;
    if (event.target.id === "an-cohort") { filters = {cohort:event.target.value, mode:"all", scenario:"all"}; renderCohort(); }
    if (event.target.id === "an-mode") { filters.mode = event.target.value; renderCohort(); }
    if (event.target.id === "an-scenario") { filters.scenario = event.target.value; renderCohort(); }
  }
  function renderCohort() {
    const {data, rows, learned, title} = cohort();
    el("an-cohort").value = filters.cohort;
    const modes = [...new Set(rows.map((r) => r.mode))];
    if (!modes.includes(filters.mode)) filters.mode = "all";
    el("an-mode").innerHTML = '<option value="all">All channels</option>' + modes.map((m) => `<option value="${html(m)}">${html(labels[m] || m)}</option>`).join("");
    el("an-mode").value = filters.mode;
    const scenarios = [...new Set(rows.map((r) => r.scenario).filter(Boolean))];
    if (!scenarios.includes(filters.scenario)) filters.scenario = scenarios.length ? scenarios[0] : "all";
    el("an-scenario-field").hidden = scenarios.length === 0;
    el("an-scenario").innerHTML = scenarios.map((s) => `<option value="${html(s)}">${html(s.replaceAll("_", " "))}</option>`).join("");
    el("an-scenario").value = filters.scenario;
    const selected = selectedRows();
    if (!selected.length) { el("an-cohort-body").innerHTML = empty("No matching evidence", data.reason || "This cohort has no measurements for the selected channel and condition."); return; }
    const count = selected.reduce((sum, row) => sum + (learned ? row.hidden_decisions || 0 : row.attempts || 0), 0);
    const sourceTag = learned ? "Decision accuracy · not task success" : "Task success rate";
    const caveats = data.caveats || [];
    const scopeNote = filters.cohort === "factory" ? "The control removes the private peer channel. Robots still share the task runtime; this is not a team of fully independent robots." : learned ? "Repeated decisions within an episode are correlated. High accuracy on captured observations does not demonstrate successful live manipulation." : "Task-controller evidence is separate from the learned VLA controller. Do not attribute these fold outcomes to the learned adapter.";
    const baseline = rows.find((r) => r.mode === "none" && (!r.scenario || r.scenario === filters.scenario));
    const best = selected.filter((r) => r.mode !== "none").sort((a,b) => (learned ? b.hidden_accuracy : b.success_rate) - (learned ? a.hidden_accuracy : a.success_rate))[0];
    const baselineRate = learned ? baseline?.hidden_accuracy : baseline?.success_rate;
    const bestRate = learned ? best?.hidden_accuracy : best?.success_rate;
    const matched = filters.cohort !== "factory" || data.pairing_verified === true;
    const gain = matched && Number.isFinite(bestRate) && Number.isFinite(baselineRate) ? (bestRate - baselineRate) * 100 : null;
    el("an-cohort-body").innerHTML = `<div class="an-cohort-topline"><div><h3>${html(title)}</h3><p>${number(selected.length)} channel groups · ${number(count)} ${learned ? "hidden-peer decisions" : "task attempts"}${filters.cohort === "factory" && data.pairing_verified ? ` · ${number(data.paired_cases)} matched cases per channel` : ""}${filters.scenario !== "all" ? ` · ${html(filters.scenario.replaceAll("_", " "))}` : ""}${data.backend ? ` · ${html(data.backend)} backend` : ""}${data.planner ? ` / ${html(data.planner)} planner` : ""}</p></div>${chip(learned ? "Captured / offline" : "Simulated tasks")}</div><div class="an-comparison-grid"><div class="an-chart-pane"><div class="an-chart-caption"><strong>${sourceTag}</strong><span>Higher is better · identical scale</span></div>${barChart(selected, learned)}<div class="an-chart-key"><i></i><span>${learned ? "Point estimates only; decisions are not independent trials." : "Whiskers show a descriptive 95% Wilson interval when available."}</span></div></div><aside class="an-reading"><div class="section-overline">How to read this</div><strong>${gain == null ? "Keep the scope in view." : gain > 0 ? `+${number(gain, 1)} percentage points` : gain < 0 ? `${number(gain, 1)} percentage points` : "No observed lift"}</strong><p>${gain == null ? "A comparable no-channel baseline is not available in this slice." : `${labels[best.mode] || best.mode} versus no private messages in this condition. Reference control: ${number(learned ? baseline.hidden_correct : baseline.successes)} / ${number(learned ? baseline.hidden_decisions : baseline.attempts)}. This is a difference in ${learned ? "decision accuracy" : "task success"}, not a speedup.`}</p><div class="an-reading-rule"></div><p>${html(scopeNote)}</p><button class="text-button" data-an-evidence>Trace the evidence ↗</button></aside></div><div class="an-task-details">${detailRows(selected, learned)}</div>${learned && data.live ? `<div class="an-live-note an-live-prominent"><strong>Closed-loop task not established</strong><span>Live controller: ${number(data.live.grasp_count)} grasps · ${number(data.live.missed_hazard_holds)} / ${number(data.live.hazard_decisions)} required hazard holds missed · full task ${data.live.full_task_success ? "completed" : "not completed"}.</span></div>` : ""}<details class="an-method-details"><summary>Experiment conditions & limitations <span>${number(caveats.length)} notes</span></summary>${list(caveats)}${data.diagnostics ? `<p class="an-method-note">Additional baseline diagnostics are retained in the evidence export; they are not pooled into this comparison.</p>` : ""}</details>`;
    applyVisualValues();
  }
  function barChart(rows, learned) {
    const width = 720, left = 150, right = 64, top = 28, rowHeight = 64, plot = width - left - right, height = top + rowHeight * rows.length + 35;
    const grid = [0,.25,.5,.75,1].map((v) => `<text x="${left+plot*v}" y="${height-8}" text-anchor="middle" class="an-svg-axis">${v*100}%</text>`).join("");
    const marks = rows.map((r, index) => {
      const value = learned ? r.hidden_accuracy : r.success_rate, y = top + index * rowHeight;
      const n = learned ? r.hidden_decisions : r.attempts, correct = learned ? r.hidden_correct : r.successes;
      const rate = Number.isFinite(value) ? Math.max(0,Math.min(1,value)) : 0;
      const label = labels[r.mode] || r.label || r.mode;
      const interval = Array.isArray(r.wilson95) ? r.wilson95 : r.wilson95 ? [r.wilson95.low, r.wilson95.high] : null;
      const whisker = !learned && interval?.every(Number.isFinite) ? `<path d="M${left + interval[0]*plot} ${y+14}H${left+interval[1]*plot}M${left+interval[0]*plot} ${y+9}v10M${left+interval[1]*plot} ${y+9}v10" stroke="#263d32" stroke-width="1.3"/>` : "";
      return `<g tabindex="0" class="an-chart-mark" role="img" aria-label="${html(label)}: ${html(percent(value))}, ${number(correct)} of ${number(n)} ${learned ? "hidden-peer decisions" : "task attempts"}${interval ? `; 95% interval ${percent(interval[0])} to ${percent(interval[1])}` : ""}"><title>${html(label)}: ${html(percent(value))} · ${number(correct)} / ${number(n)}${interval ? ` · Wilson 95% ${percent(interval[0])}–${percent(interval[1])}` : ""}</title><text x="${left-16}" y="${y+12}" text-anchor="end" class="an-svg-label">${html(label)}</text><text x="${left-16}" y="${y+28}" text-anchor="end" class="an-svg-small">${number(correct)} / ${number(n)}</text><rect x="${left}" y="${y}" width="${plot}" height="28" rx="4" fill="#f3f6f1"/><rect x="${left}" y="${y}" width="${Math.max(rate*plot,1)}" height="28" rx="4" fill="${colors[r.mode] || "#758576"}"/>${whisker}<text x="${left+plot+12}" y="${y+19}" class="an-svg-value">${html(percent(value))}</text></g>`;
    }).join("");
    return `<svg class="an-bar-chart" viewBox="0 0 ${width} ${height}" role="group" aria-label="${learned ? "Hidden peer decision accuracy" : "Task success"} by communication channel">${grid}${marks}</svg>`;
  }
  function detailRows(rows, learned) {
    const showWall = !learned && rows.some((r) => Number.isFinite(r.mean_wall_seconds));
    const showSim = !learned && rows.some((r) => Number.isFinite(r.mean_simulation_seconds));
    return `<div class="an-detail-caption"><span>${learned ? "Decision & message details" : "Execution & message details"}</span><span>${learned ? "The action preview may still fail in closed loop." : "Simulation time and wall time are different measures."}</span></div><div class="an-table-scroll"><table class="an-table"><thead><tr><th scope="col">Channel</th>${learned ? '<th scope="col">Hold accuracy</th><th scope="col">Action error</th><th scope="col">Unsafe GO</th>' : `<th scope="col">Failed / total</th><th scope="col">Mean ticks</th>${showSim ? '<th scope="col">Mean sim time</th>' : ""}${showWall ? '<th scope="col">Mean wall time</th>' : ""}`}<th scope="col">Packet bytes</th></tr></thead><tbody>${rows.map((r) => `<tr><th scope="row"><i data-an-color="${colors[r.mode] || "#758576"}"></i>${html(labels[r.mode] || r.mode)}</th>${learned ? `<td>${percent(r.hold_accuracy)}<small>${number(r.hold_correct)} / ${number(r.decisions)}</small></td><td>${Number.isFinite(r.action_error_mm) ? `${number(r.action_error_mm, 2)} mm` : "—"}</td><td>${number(r.unsafe_go_count)}</td>` : `<td>${number(r.failures)} / ${number(r.attempts)}</td><td>${number(r.mean_ticks, 1)}</td>${showSim ? `<td>${number(r.mean_simulation_seconds, 2)} s</td>` : ""}${showWall ? `<td>${number(r.mean_wall_seconds, 2)} s</td>` : ""}`}<td>${bytes(r.packet_bytes ?? r.payload_bytes)}<small>reported aggregate</small></td></tr>`).join("")}</tbody></table></div>`;
  }
  function renderOperations(ops, archived = false) {
    const rows = ops.modes || [];
    const archive = !archived && ops.archive?.modes?.length ? renderOperations(ops.archive, true) : "";
    if (!rows.length) return `<section class="an-panel an-runtime">${empty(ops.status === "unavailable" ? "Live workspace telemetry is unavailable" : "No live exchange timings yet", ops.status === "unavailable" ? "The current journal could not be measured. Historical verification below is separate from live workspace totals." : "Create an approved Studio session and run an exchange. Journal history and retained timing samples will appear here.")}</section>${archive}`;
    const totalSamples = rows.reduce((sum,r) => sum + (r.elapsed_ms?.n || 0),0);
    const title = archived ? "Archived channel verification" : "Studio exchange latency";
    return `<section class="an-panel an-runtime"><div class="an-panel-heading"><div><h3>${title}</h3><p>${number(ops.exchanges)} ${archived ? "historical captured-feature exchanges" : "observed committed journal exchanges"} · ${number(totalSamples)} retained total-latency samples</p></div>${chip(archived ? "Archive · not live totals" : "Local journal")}</div><div class="an-latency-grid">${rows.map((r) => {
      const total = r.elapsed_ms || {}, db = r.database_ms || {}, llm = r.llm_ms || {};
      return `<article class="an-latency-card"><div><i data-an-color="${colors[r.mode] || "#758576"}"></i><h4>${html(labels[r.mode] || r.mode)}</h4><span>${number(r.exchanges)} exchanges</span></div><strong>${duration(total.p50)}<small>p50 total</small></strong><dl><div><dt>p95 total</dt><dd>${duration(total.p95)}</dd></div><div><dt>p50 database</dt><dd>${duration(db.p50)}</dd></div><div><dt>p50 Qwen</dt><dd>${duration(llm.p50)}</dd></div><div><dt>Timing coverage</dt><dd>${number(total.n)} / ${number(r.exchanges)}</dd></div><div><dt>Verified readbacks</dt><dd>${number(r.verified_readbacks)}</dd></div></dl></article>`;
    }).join("")}</div><div class="an-inline-note">${icon("clock")}<span>${archived ? "These six historical probes used captured foundation features with fresh adapter / Qwen inference. They exclude foundation vision inference, execute no robot motion, and are not pooled into live totals. A single sample is not a latency distribution." : `Component percentiles are calculated separately and do not add up to total p50. ${number(ops.coverage?.journal_only_exchanges)} exchanges were reconstructed from journal messages only; their historical timing and receiver verification are unknown. Missing timings remain missing after recovery.`}</span></div>${ops.caveats?.length ? `<details class="an-method-details"><summary>Timing coverage & interpretation</summary>${list(ops.caveats)}</details>` : ""}</section>${archive}`;
  }
  function renderGovernance(governance, ops) {
    const m = governance.models || {}, total = m.total;
    const operational = ops.status === "unavailable" ? {} : ops;
    const healthCoverage = ops.status === "unavailable" ? "Session health is unavailable." : `${number(ops.known_failed_sessions)} known failed; ${number(ops.coverage?.session_health_checked)} of ${number(ops.sessions)} sessions health-checked. ${number(ops.coverage?.session_health_unknown)} remain unknown.`;
    const states = [["released", "Released", "#1e7a4c"],["draft", "Draft", "#c98a00"],["retired", "Retired", "#8a9399"]];
    let offset=0;
    const segments=states.map(([key,label,color])=>{const value=Number.isFinite(total)&&total>0?Math.max(0,m[key]||0)/total*100:0;const mark=`<circle cx="80" cy="80" r="60" pathLength="100" fill="none" stroke="${color}" stroke-width="20" stroke-dasharray="${value} ${100-value}" stroke-dashoffset="${-offset}" transform="rotate(-90 80 80)"/>`;offset+=value;return mark}).join("");
    const donut=`<div class="an-donut"><svg viewBox="0 0 160 160" role="img" aria-label="Model lifecycle: ${states.map(([key,label])=>`${number(m[key])} ${label}`).join(', ')}"><circle cx="80" cy="80" r="60" fill="none" stroke="#fff" stroke-width="20"/>${segments}<text x="80" y="86" text-anchor="middle" font-size="30">${number(total)}</text></svg><div>${states.map(([key,label])=>`<p>${label} <b>${number(m[key])}</b></p>`).join('')}</div></div>`;
    return `<div class="an-operations-grid"><section class="an-panel an-model-inventory"><div class="an-panel-heading"><div><h3>Model lifecycle</h3><p>Exact registered revisions</p></div><strong>${number(total)}<small>models</small></strong></div>${donut}<div class="an-inventory-strip" role="img" aria-label="${states.map(([key,label])=>`${number(m[key])} ${label}`).join(", ")}">${states.map(([key,,color]) => Number.isFinite(total) && total > 0 && m[key] > 0 ? `<i data-an-width="${m[key]/total*100}" data-an-color="${color}"></i>` : "").join("")}</div><div class="an-inventory-legend">${states.map(([key,label,color]) => `<span><i data-an-color="${color}"></i>${label}<strong>${number(m[key])}</strong></span>`).join("")}</div><p>Manager approval authorizes Studio use. It does not certify robot task performance.</p></section><section class="an-panel an-session-inventory"><div class="an-panel-heading"><div><h3>Session health</h3><p>Journal observations · current snapshot</p></div>${icon("db")}</div><div class="an-session-stats"><div><strong>${number(operational.sessions)}</strong><span>sessions</span></div><div><strong>${number(operational.exchanges)}</strong><span>exchanges</span></div><div><strong>${number(operational.failed_sessions)}</strong><span>failed sessions</span></div></div><p class="an-session-coverage">${html(healthCoverage)}</p><p>Observed exchanges are journal entries, not completed physical tasks.</p></section></div>`;
  }
  const effortFields = [
    ["robots","Robots",6,1,100,1], ["tasks","Task types",4,1,1000,1],
    ["manual_per_robot_hours","Manual setup / robot",8,0,10000,.5], ["manual_pair_hours","Manual integration / pair",2,0,10000,.5], ["manual_task_hours","Manual coding / task",6,0,10000,.5],
    ["relay_per_robot_hours","Relay setup / robot",2,0,10000,.5], ["relay_task_hours","Relay configuration / task",2,0,10000,.5], ["hourly_rate","Hourly engineering rate",100,0,10000,1]
  ];
  function renderEffort(model) {
    const defaults = model.defaults || {};
    return `<section class="an-effort"><div class="an-effort-top"><div><span class="an-chip scenario">Planning scenario</span><h3>What would integration take?</h3><p>Compare explicit hours for custom pair integration and a reusable coordination interface. These inputs are examples, not measured savings.</p></div><div class="an-effort-symbol" aria-hidden="true">${icon("link")}</div></div><div class="an-effort-layout"><form id="an-effort-form"><div class="an-effort-inputs">${effortFields.map(([key,label,fallback,min,max,step]) => `<label>${html(label)}<span class="an-effort-input-wrap"><input type="number" name="${key}" value="${html(defaults[key] ?? fallback)}" min="${min}" max="${max}" step="${step}" required><span>${key === "hourly_rate" ? "$/h" : key === "robots" || key === "tasks" ? "count" : "hours"}</span></span></label>`).join("")}</div><div class="an-estimate-actions"><span>All-to-all pairs: n × (n − 1) / 2</span><button type="submit" class="button" id="an-calculate">Calculate scenario ${icon("arrow")}</button></div><p class="an-estimate-error" id="an-estimate-error" role="alert" hidden></p></form><div id="an-estimate-result" aria-live="polite"></div></div><details class="an-effort-assumptions"><summary>Included work & assumptions</summary><p>Manual hours = robots × setup hours + robot pairs × integration hours + tasks × task coding hours. Relay hours = robots × setup hours + tasks × configuration hours.</p>${list(model.caveats || ["Training data, fine-tuning, hardware commissioning, validation, maintenance, and infrastructure costs are outside this simple model.","Actual labor savings require timed work on comparable tasks with comparable teams."])}<p>Time and quality are evaluated separately above. An engineering scenario cannot validate model success or safety.</p></details></section>`;
  }
  async function calculateEstimate() {
    const inputs = Object.fromEntries(new FormData(el("an-effort-form")).entries());
    Object.keys(inputs).forEach((key) => { inputs[key] = Number(inputs[key]); });
    const current = generation, revision = ++estimateRequest;
    el("an-calculate").disabled = true; el("an-estimate-error").hidden = true;
    try {
      const value = await options.api("/api/analytics/estimate", inputs);
      if (current !== generation || revision !== estimateRequest || !options.authenticated()) return;
      estimate = value;
      const saved = value.hours_saved, positive = Number.isFinite(saved) && saved >= 0;
      el("an-estimate-result").innerHTML = `<div class="an-estimate-output"><span class="section-overline">Your modeled scenario</span><strong>${number(Math.abs(saved),1)}<span>hours ${positive ? "less" : "more"}</span></strong><p>${Number.isFinite(value.savings_percent) ? `${number(Math.abs(value.savings_percent),1)}% ${positive ? "less" : "more"} modeled effort` : "No percentage baseline"} · ${number(value.edges)} robot pairs</p><div class="an-effort-compare"><div><span>Custom integration</span><strong>${number(value.manual_hours,1)} h</strong></div><div><span>Relay coordination</span><strong>${number(value.relay_hours,1)} h</strong></div><div><span>Labor cost difference</span><strong>${value.cost_saved < 0 ? "−" : ""}$${number(Math.abs(value.cost_saved),0)}</strong></div></div><p class="an-estimate-disclaimer">Illustrative inputs. No human team was timed; this result is not a measured productivity claim.</p></div>`;
    } catch (error) {
      if (current !== generation || error.stale || !options.authenticated()) return;
      el("an-estimate-error").textContent = error.message; el("an-estimate-error").hidden = false;
    } finally { if (current === generation && revision === estimateRequest && el("an-calculate")) el("an-calculate").disabled = false; }
  }
  function viewSources() { return snapshot?.sources || []; }
  function showEvidence() {
    if (!snapshot) return;
    const sources = viewSources();
    el("an-evidence-body").innerHTML = `<div class="an-evidence-scope">${chip("Authenticated snapshot")}${chip(`${sources.length} SOURCE ARTIFACTS`)}<p>Snapshot generated ${html(snapshot.generated_at ? new Date(typeof snapshot.generated_at === "number" ? snapshot.generated_at * 1000 : snapshot.generated_at).toLocaleString() : "at request time")}. Current comparison: ${html(cohort().title)} · ${html(filters.mode === "all" ? "all channels" : labels[filters.mode] || filters.mode)}${filters.scenario !== "all" ? ` · ${html(filters.scenario)}` : ""}.</p></div><div class="an-definitions"><h3>Metric definitions</h3>${typeof snapshot.definitions === "object" ? Object.entries(snapshot.definitions || {}).map(([key,value]) => `<div><strong>${html(key.replaceAll("_"," "))}</strong><p>${html(typeof value === "string" ? value : JSON.stringify(value))}</p></div>`).join("") : `<p>${html(snapshot.definitions || "See the source artifacts for experiment-level definitions.")}</p>`}</div><h3 class="an-source-title">Source lineage</h3>${sources.map((source) => `<article class="an-source"><div><strong>${html(source.id)}</strong>${chip(source.status || "Recorded")}</div><code>${html(source.path)}</code><p>${html(source.grain || "Artifact-level evidence")} · ${bytes(source.bytes)}</p>${source.sha256 ? `<details><summary>SHA-256</summary><code>${html(source.sha256)}</code></details>` : ""}</article>`).join("")}<div class="an-evidence-note">Export includes the selected comparison rows, the displayed operations sections, source hashes, assumptions, and any calculated effort scenario.</div>`;
    el("an-evidence-dialog").showModal();
  }
  function exportView() {
    if (!snapshot || !options.authenticated()) return;
    const comparisonCohort = cohort(), visibleRows = selectedRows();
    const baseline = comparisonCohort.rows.find((row) => row.mode === "none" && (!row.scenario || row.scenario === filters.scenario));
    const reference = baseline && !visibleRows.includes(baseline) ? {label:"Contextual no-private-messages control; excluded from filtered chart rows", evidence_id:comparisonCohort.data.evidence_id, included_in_filtered_rows:false, row:baseline} : null;
    const data = {schema:"relay.analytics.view.v1",generated_at:snapshot.generated_at,filters:{...filters},comparison:{title:cohort().title,evidence_id:cohort().data.evidence_id,rows:visibleRows,reference_control:reference,diagnostics:cohort().data.diagnostics,live:cohort().data.live,caveats:cohort().data.caveats || []},headline:snapshot.headline,operations:snapshot.operations,governance:snapshot.governance,effort_scenario:estimate,effort_model:snapshot.effort_model,sources:viewSources(),definitions:snapshot.definitions,scope_note:"Comparison filters apply only to comparison rows. Other sections retain their independently labeled populations."};
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)], {type:"application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = `relay-analytics-${filters.cohort}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000); options.notify("Analytics view exported with source hashes and assumptions");
  }
  window.RelayAnalytics = {mount, show, reset};
})();
