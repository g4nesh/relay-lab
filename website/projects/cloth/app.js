"use strict";

const $ = (id) => document.getElementById(id);
const labels = {embeddings: "Embeddings", structured_text: "Structured text", hybrid: "Hybrid", none: "No communication"};
const descriptions = {
  embeddings: "A trained feature bottleneck carries local geometry and obstacle observations. Decoded facts are approximate.",
  structured_text: "Every measured local feature is sent at full precision, with its observation timestamp and visibility flags.",
  hybrid: "A trained feature bottleneck shares geometry. Exact structured fields preserve obstacle, grasp and visibility information.",
  none: "Each arm keeps its own observation. No private peer evidence crosses the channel.",
};
const state = {manifest: null, run: null, summary: {}, trace: {frames: []}, events: [], index: -1, request: 0, controller: null};
const video = $("main-video");
let animationRequest = null;
let comparisonRequest = 0;
const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
const firstNumber = (...values) => values.find(isNumber);
const human = (value) => String(value ?? "—").replaceAll("_", " ");
const number = (value, places = 0) => isNumber(value) ? value.toLocaleString(undefined, {maximumFractionDigits: places}) : "—";
const metres = (value) => isNumber(value) ? `${(value * 1000).toFixed(1)} mm` : "—";
const bytes = (value) => !isNumber(value) ? "—" : value < 1024 ? `${Math.round(value)} B` : value < 1048576 ? `${(value / 1024).toFixed(1)} KB` : `${(value / 1048576).toFixed(2)} MB`;
const clock = (seconds) => !isNumber(seconds) ? "00:00" : `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
function text(id, value) { $(id).textContent = value; }
function node(tag, className, content) { const element = document.createElement(tag); if (className) element.className = className; if (content !== undefined) element.textContent = content; return element; }
function artifactURL(value) {
  if (typeof value !== "string" || !value) return null;
  const url = new URL(value, location.href);
  return url.origin === location.origin && url.pathname.startsWith("/projects/cloth/runs/") ? url.href : null;
}
async function readJSON(url, signal) {
  const response = await fetch(url, {signal, cache: "no-store"});
  if (!response.ok) throw new Error(`Could not load recorded evidence (HTTP ${response.status}).`);
  return response.json();
}
function showError(message) { text("global-error", message); $("global-error").hidden = !message; }
function setView(view) {
  $("replay-view").hidden = view !== "replay";
  $("comparison-view").hidden = view !== "comparison";
  document.querySelectorAll("[data-view]").forEach((button) => {
    const selected = button.dataset.view === view;
    button.classList.toggle("active", selected); button.setAttribute("aria-pressed", String(selected));
  });
  if (view !== "replay") video.pause();
  history.replaceState(null, "", view === "comparison" ? "#comparison" : "#workcell");
}
function alignment(metrics = {}) {
  const errors = metrics.hem_errors || (typeof metrics.hem_alignment_m === "object" ? metrics.hem_alignment_m : null);
  return firstNumber(metrics.alignment_m, metrics.alignment_error_m, metrics.mean_alignment_m,
    metrics.hem_alignment_m, metrics.hem_alignment_error,
    errors && typeof errors === "object" && Object.values(errors).every(isNumber)
      ? Math.max(...Object.values(errors)) : undefined);
}
function channelMetrics(frame = {}) {
  const raw = frame.communication || state.summary.communication || {};
  return {...(state.summary.communication?.metrics || state.summary.communication || {}),
    ...(raw.metrics || raw.channel_metrics || raw)};
}
function peerBeliefs(frame) {
  const communication = frame.communication || {};
  return frame.beliefs || communication.beliefs || communication.received || communication.peers ||
    {left: frame.decisions?.left?.peer, right: frame.decisions?.right?.peer};
}
function compileEvents(trace) {
  const all = [], seen = new Set();
  for (const [fallback, events] of [
    [0, trace.events || []],
    ...trace.frames.map((frame) => [frame.tick, [...(frame.events || []), ...(frame.communication?.events || [])]]),
  ]) {
    for (const raw of events) {
      const event = typeof raw === "string" ? {event: raw} : raw;
      if (!event || typeof event !== "object") continue;
      const record = {...event, tick: isNumber(event.tick) ? event.tick : fallback};
      const identity = JSON.stringify(record);
      if (!seen.has(identity)) { seen.add(identity); all.push(record); }
    }
  }
  return all.sort((a, b) => a.tick - b.tick);
}
function renderEvents(tick) {
  const current = state.events.filter((event) => event.tick <= tick);
  text("event-count", `${current.length} events`);
  const list = $("event-list"); list.replaceChildren();
  if (!current.length) { list.append(node("li", "event-placeholder", "No recorded events before this point. Move through the timeline to inspect changes.")); return; }
  current.slice(-7).reverse().forEach((event) => {
    const item = node("li"); item.append(node("div", "event-tick", `TICK ${String(event.tick).padStart(4, "0")}`));
    item.append(node("div", "event-title", human(event.event || event.kind || event.type || "Observed update")));
    const detail = event.reason || event.detail || event.message || [event.side || event.sender || event.agent, event.packet_id].filter(Boolean).join(" · ");
    if (detail) item.append(node("div", "event-detail", typeof detail === "string" ? detail : JSON.stringify(detail)));
    list.append(item);
  });
}
function renderFrame(index, force = false) {
  if (!state.trace.frames.length || (index === state.index && !force)) return;
  state.index = Math.min(Math.max(index, 0), state.trace.frames.length - 1);
  const frame = state.trace.frames[state.index];
  const metrics = frame.metrics || {};
  const channel = channelMetrics(frame);
  const peers = peerBeliefs(frame);
  text("tick-label", `TICK ${String(frame.tick ?? state.index).padStart(4, "0")}`);
  text("alignment-value", metres(alignment(metrics)));
  const phases = typeof frame.phase === "object" ? frame.phase : {left: frame.phase, right: frame.phase};
  text("phase-value", phases?.left === phases?.right ? human(phases?.left) : "Coordinating");
  const traffic = firstNumber(channel.packet_bytes, channel.transport_bytes, channel.bytes, state.summary.communication?.packet_bytes);
  text("bytes-value", bytes(traffic));
  const ages = Object.values(peers).filter(Boolean).map((peer) => firstNumber(peer.age_ticks, peer.observation_age_ticks)).filter(isNumber);
  text("age-value", ages.length ? `${Math.max(...ages)} ticks` : "—");
  for (const name of ["delivered", "dropped", "expired"]) text(`${name}-value`, number(channel[name]));
  text("reordered-value", number(channel.out_of_order));
  text("channel-status", channel.backend?.durable === true ? "Durable" : channel.backend?.kind === "memory" ? "Reference" : "Recorded");
  text("backend-label", channel.backend?.kind === "spacetimedb" ? "SpacetimeDB · exact retrieval" : channel.backend?.kind === "memory" ? "Memory reference channel" : "Confirmed delivery boundary");
  const provenance = {tick: frame.tick, mode: state.run.mode,
    received: peers, model: channel.model || state.summary.communication?.model || null,
    robot_models: state.summary.models || state.summary.model_info || null,
    playback_source: state.run.playback_source || "original simulation recording",
    schedule: channel.schedule || null};
  text("provenance-data", JSON.stringify(provenance, null, 2));
  for (const arm of ["left", "right"]) {
    const robot = frame.robots?.[arm] || {};
    const decision = frame.decisions?.[arm] || {};
    text(`${arm}-state`, decision.hold === true ? "Holding position" : "Local control");
    text(`${arm}-phase`, human(phases?.[arm] || decision.phase));
    text(`${arm}-grasp`, robot.holding != null && robot.holding !== false ? "Attached" : robot.holding === null || robot.holding === false ? "Open" : "—");
    text(`${arm}-clearance`, metres(firstNumber(decision.clearance, decision.clearance_m)));
    const ee = robot.ee || robot.ee_position;
    text(`${arm}-position`, Array.isArray(ee) ? ee.map((value, i) => `${"XYZ"[i]} ${isNumber(value) ? value.toFixed(3) : "—"}`).join(" · ") : "X — · Y — · Z —");
    text(`${arm}-reason`, decision.reason || decision.hold_reason || (peers[arm]
      ? `Peer evidence is ${peers[arm].age_ticks ?? "—"} ticks old. ${peers[arm].precision || "Timestamped local observation."}`
      : "No valid peer observation is available at this point."));
  }
  renderEvents(frame.tick ?? state.index);
}
function updateFromVideo() {
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const ratio = duration ? Math.min(1, video.currentTime / duration) : 0;
  $("timeline").value = String(Math.round(ratio * 1000));
  text("time-label", `${clock(video.currentTime)} / ${clock(duration)}`);
  text("play-button", video.paused ? "▶" : "Ⅱ");
  $("play-button").setAttribute("aria-label", video.paused ? "Play recording" : "Pause recording");
  let index = Math.round(ratio * Math.max(0, state.trace.frames.length - 1));
  if (state.trace.frames.length && isNumber(state.trace.frames[0].video_time)) {
    index = 0;
    while (index + 1 < state.trace.frames.length && state.trace.frames[index + 1].video_time <= video.currentTime) index++;
  }
  renderFrame(index);
  for (const arm of ["left", "right"]) {
    const local = $(`${arm}-video`);
    if (!local.hidden && Number.isFinite(local.duration) && local.duration > 0) {
      const target = ratio * local.duration;
      if (Math.abs(local.currentTime - target) > 0.12) local.currentTime = target;
    }
  }
  if (!video.paused && !video.ended && animationRequest === null) {
    animationRequest = requestAnimationFrame(() => { animationRequest = null; updateFromVideo(); });
  }
}
async function selectRun(id) {
  const run = state.manifest.runs.find((entry) => entry.id === id);
  if (!run) return;
  const request = ++state.request;
  state.controller?.abort(); state.controller = new AbortController();
  video.pause(); state.run = run; state.index = -1; state.trace = {frames: []}; state.events = [];
  video.removeAttribute("src"); video.removeAttribute("poster"); video.load();
  $("video-empty").hidden = false; $("play-button").disabled = true;
  for (const id of ["alignment-value", "phase-value", "bytes-value", "age-value", "delivered-value", "dropped-value", "reordered-value", "expired-value"]) text(id, "—");
  text("provenance-data", "Loading this episode’s evidence…"); renderEvents(0);
  for (const arm of ["left", "right"]) {
    const local = $(`${arm}-video`); local.pause(); local.removeAttribute("src"); local.load(); local.hidden = true;
    text(`${arm}-phase`, "Loading observation…"); text(`${arm}-state`, "—"); text(`${arm}-grasp`, "—"); text(`${arm}-clearance`, "—");
    text(`${arm}-position`, "X — · Y — · Z —"); text(`${arm}-reason`, "Awaiting this episode’s local evidence.");
  }
  $("run-select").value = run.id;
  showError(""); text("verdict", "Loading evidence…"); $("verdict").className = "verdict";
  text("mode-tag", labels[run.mode] || human(run.mode)); text("scenario-tag", human(run.scenario)); text("seed-tag", `Seed ${run.seed ?? "—"}`);
  text("channel-kind", labels[run.mode] || "Shared evidence"); text("representation-note", descriptions[run.mode] || "Timestamped local observations are shared through the recorded channel.");
  try {
    const summaryURL = artifactURL(run.summary_url), traceURL = artifactURL(run.trace_url);
    if (!summaryURL || !traceURL) throw new Error("This episode has no valid local summary or trace URL.");
    const [summary, trace] = await Promise.all([readJSON(summaryURL, state.controller.signal), readJSON(traceURL, state.controller.signal)]);
    if (request !== state.request) return;
    if (!trace || !Array.isArray(trace.frames)) throw new Error("The recorded trace has no frame list.");
    state.summary = summary; state.trace = trace; state.events = compileEvents(trace);
    window.dispatchEvent(new CustomEvent("relay:episode", {detail:{summary,trace,run}}));
    $("summary-link").href = summaryURL; $("trace-link").href = traceURL;
    text("verdict", summary.success === true ? "Evaluator · completed" : summary.success === false ? "Evaluator · not completed" : "Outcome not recorded");
    $("verdict").className = `verdict${summary.success === true ? " success" : summary.success === false ? " failed" : ""}`;
    const physicsURL = artifactURL(run.physics_replay_url);
    $("physics-link").hidden = !physicsURL; $("physics-link").href = physicsURL || "#";
    const videoURL = artifactURL(run.video_url), posterURL = artifactURL(run.poster_url);
    if (posterURL) video.poster = posterURL; else video.removeAttribute("poster");
    if (videoURL) { video.src = videoURL; video.load(); } else { video.removeAttribute("src"); video.load(); }
    $("video-empty").hidden = Boolean(videoURL || posterURL); $("play-button").disabled = !videoURL;
    text("recording-label", `${run.video_note ? run.video_note + " · " : ""}${summary.planner?.mode === "llm" ? "Qwen review" : "Fixed phase guards"} · ${human(run.scenario)} · seed ${run.seed ?? "—"}`);
    for (const arm of ["left", "right"]) {
      const local = $(`${arm}-video`), url = artifactURL(run[`${arm}_video_url`]);
      local.hidden = !url;
      if (url) local.src = url; else local.removeAttribute("src");
      local.load();
    }
    renderFrame(0, true);
    if (typeof summary.error === "string" && summary.error) showError(`This recorded attempt stopped: ${summary.error}`);
    if (!trace.frames.length) showError("This episode has no recorded frames. Its summary is available, but no physical trace can be inspected.");
  } catch (error) {
    if (error.name === "AbortError") return;
    if (request === state.request) { showError(error.message); text("verdict", "Evidence unavailable"); }
  }
}
async function selectComparison(value) {
  const url = artifactURL(value), request = ++comparisonRequest;
  if (!url) return;
  $("comparison-cards").replaceChildren(); $("comparison-table-card").hidden = true;
  $("episodes-card").hidden = true; text("comparison-note", "Loading this experiment’s results…");
  try {
    const report = await readJSON(url);
    if (request === comparisonRequest) {
      renderComparison(report, url);
      const suite = state.manifest.comparisons?.find(item => artifactURL(item.url) === url);
      if (typeof suite?.note === "string") text("comparison-note", `${report.conclusion || ""} ${suite.note}`);
    }
  } catch (error) { if (request === comparisonRequest) text("comparison-note", error.message); }
}
function renderComparison(report, url) {
  const summaries = report.mode_summaries || report.summaries || report.modes || report.results || [];
  const modes = Array.isArray(summaries) ? summaries : Object.entries(summaries).map(([mode, value]) => ({mode, ...value}));
  const valid = modes.filter((row) => row && typeof row === "object" && isNumber(row.runs));
  text("compared-modes", valid.map(row => labels[row.mode] || human(row.mode)).join(" / ") || "Recorded conditions");
  $("comparison-empty").hidden = valid.length > 0;
  $("comparison-table-card").hidden = !valid.length;
  $("comparison-source").href = url;
  if (typeof report.conclusion === "string") text("comparison-note", report.conclusion);
  else if (typeof report.note === "string" || typeof report.description === "string") text("comparison-note", report.note || report.description);
  const cards = $("comparison-cards"), table = $("comparison-table"); cards.replaceChildren(); table.replaceChildren();
  valid.forEach((row) => {
    const card = node("article", "result-card"); card.dataset.mode = row.mode;
    card.append(node("div", "result-label", labels[row.mode] || human(row.mode)));
    const score = node("div", "result-score", number(row.successes)); score.append(node("span", "", ` / ${number(row.runs)}`));
    card.append(score, node("div", "result-caption", "episodes completed"), node("p", "result-detail", descriptions[row.mode] || "Recorded communication condition."));
    const stats = node("div", "result-stats");
    for (const [label, value] of [["Mean worst-arm error", metres(row.mean_alignment_m)], ["Mean traffic", bytes(row.mean_bytes)]]) {
      const stat = node("div"); stat.append(node("span", "", label), node("strong", "", value)); stats.append(stat);
    }
    card.append(stats); cards.append(card);
    const tr = node("tr");
    const rate = isNumber(row.success_rate) ? row.success_rate : row.runs > 0 && isNumber(row.successes) ? row.successes / row.runs : undefined;
    for (const value of [labels[row.mode] || human(row.mode), `${number(row.successes)} / ${number(row.runs)}`, isNumber(rate) ? `${(rate * 100).toFixed(1)}%` : "—", metres(row.mean_alignment_m), bytes(row.mean_bytes), number(row.mean_ticks, 1)]) tr.append(node("td", "", value));
    table.append(tr);
  });
  const episodes = Array.isArray(report.episodes) ? report.episodes : [];
  $("episodes-card").hidden = !episodes.length;
  const body = $("episodes-table"); body.replaceChildren();
  episodes.forEach((episode, index) => {
    const row = node("tr");
    for (const value of [episode.label || episode.id || `${human(episode.scenario)} · ${index + 1}`, labels[episode.mode] || human(episode.mode), number(episode.seed)]) row.append(node("td", "", value));
    row.append(node("td", episode.success === true ? "outcome-pass" : episode.success === false ? "outcome-fail" : "", episode.success === true ? "Completed" : episode.success === false ? "Not completed" : "Unrecorded"));
    row.append(node("td", "", metres(firstNumber(episode.alignment_m, episode.alignment_error_m, alignment(episode.metrics)))));
    const action = node("td");
    const replay = state.manifest.runs.find((run) => run.id === (episode.run_id || episode.id) || (run.mode === episode.mode && run.seed === episode.seed && run.scenario === episode.scenario));
    if (replay) { const button = node("button", "episode-button", "Replay ↗"); button.addEventListener("click", () => { setView("replay"); selectRun(replay.id); window.scrollTo({top: 0, behavior: "smooth"}); }); action.append(button); }
    else action.append(node("span", "quiet-label", "Metrics only"));
    row.append(action); body.append(row);
  });
  text("experiment-count", number(episodes.length || valid.reduce((sum, row) => sum + row.runs, 0)));
}
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
$("compare-link").addEventListener("click", (event) => { event.preventDefault(); setView("comparison"); });
$("back-to-workcell").addEventListener("click", () => setView("replay"));
$("run-select").addEventListener("change", () => selectRun($("run-select").value));
$("suite-select").addEventListener("change", () => selectComparison($("suite-select").value));
$("play-button").addEventListener("click", () => { if (video.paused) video.play().catch((error) => showError(`Playback could not start: ${error.message}`)); else video.pause(); });
$("speed").addEventListener("change", () => { video.playbackRate = Number($("speed").value); });
$("timeline").addEventListener("input", () => {
  const ratio = Number($("timeline").value) / 1000;
  if (Number.isFinite(video.duration) && video.duration > 0) video.currentTime = video.duration * ratio;
  else renderFrame(Math.round(ratio * Math.max(0, state.trace.frames.length - 1)));
});
for (const event of ["timeupdate", "loadedmetadata", "seeked", "play", "pause", "ended"]) video.addEventListener(event, updateFromVideo);
for (const arm of ["left", "right"]) $(`${arm}-video`).addEventListener("loadedmetadata", updateFromVideo);
video.addEventListener("error", () => { if (video.getAttribute("src")) showError("The simulation video could not be decoded. The recorded summary and trace remain available."); });
window.addEventListener("hashchange", () => setView(location.hash === "#comparison" ? "comparison" : "replay"));
(async () => {
  try {
    const manifest = await readJSON("/projects/cloth/manifest.json");
    if (!manifest || !Array.isArray(manifest.runs)) throw new Error("No valid recording manifest is available.");
    state.manifest = manifest;
    if (manifest.title) text("page-title", manifest.title);
    if (manifest.description) text("page-description", manifest.description);
    const select = $("run-select"); select.replaceChildren();
    manifest.runs.forEach((run) => { const option = node("option", "", run.label || `${labels[run.mode] || human(run.mode)} · ${human(run.scenario)} · ${run.seed}`); option.value = run.id; select.append(option); });
    if (manifest.runs.length) await selectRun(manifest.runs[0].id);
    else { select.append(node("option", "", "No recordings available")); select.disabled = true; }
    const suites = Array.isArray(manifest.comparisons) ? manifest.comparisons.filter(s => artifactURL(s.url)) : [];
    const suiteSelect = $("suite-select"); suiteSelect.replaceChildren();
    for (const suite of suites) { const option = node("option", "", suite.label); option.value = suite.url; suiteSelect.append(option); }
    $("suite-toolbar").hidden = suites.length < 2;
    await selectComparison(suites[0]?.url || manifest.comparison_url);
    setView(location.hash === "#comparison" ? "comparison" : "replay");
  } catch (error) { showError(error.message); }
})();
