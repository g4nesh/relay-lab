"use strict";

(() => {
  const RESULT_PATH = "/runs/learned-latent/results.json";
  const MODES = ["none", "structured_text", "embeddings", "hybrid"];
  const LABELS = {
    none: "No communication",
    structured_text: "Structured text",
    embeddings: "Learned embeddings",
    hybrid: "Hybrid",
  };
  const byId = (id) => document.getElementById(id);
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const percent = (value) => finite(value) && value >= 0 && value <= 1
    ? `${(value * 100).toFixed(1)}%` : null;
  const decimal = (value) => finite(value) && value >= 0 ? value.toFixed(2) : null;
  const count = (value) => Number.isInteger(value) && value >= 0 ? value.toLocaleString() : null;
  const bytes = (value) => finite(value) && value >= 0
    ? `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} B` : null;
  const humanize = (value) => String(value).replace(/[_-]+/g, " ").replace(/^./, (letter) => letter.toUpperCase());
  let transportRows = [];

  function measurementCell(value) {
    const cell = document.createElement("td");
    cell.textContent = value ?? "Not measured";
    if (value === null) cell.className = "latent-unknown";
    return cell;
  }

  function modeCell(mode) {
    const cell = document.createElement("th");
    cell.scope = "row";
    const label = document.createElement("span");
    label.className = "latent-mode";
    label.dataset.mode = mode;
    const dot = document.createElement("span");
    dot.className = "latent-mode-dot";
    dot.setAttribute("aria-hidden", "true");
    label.append(dot, document.createTextNode(LABELS[mode]));
    cell.append(label);
    return cell;
  }

  function indexedRows(rows) {
    const result = new Map();
    for (const row of Array.isArray(rows) ? rows : []) {
      if (!row || typeof row !== "object" || !MODES.includes(row.mode)) continue;
      // Duplicate entries are ambiguous; never silently select the best value.
      result.set(row.mode, result.has(row.mode) ? null : row);
    }
    return result;
  }

  function renderStatic(rows) {
    const indexed = indexedRows(rows);
    const body = byId("static-results");
    body.replaceChildren();
    let measured = 0;
    for (const mode of MODES) {
      const row = indexed.get(mode) ?? {};
      if (percent(row.hold_accuracy) !== null) measured += 1;
      const tr = document.createElement("tr");
      tr.append(
        modeCell(mode),
        measurementCell(percent(row.hold_accuracy)),
        measurementCell(percent(row.hidden_peer_accuracy)),
        measurementCell(decimal(row.mean_action_error_mm)),
        measurementCell(percent(row.zero_peer_accuracy)),
        measurementCell(percent(row.shuffled_peer_accuracy)),
      );
      body.append(tr);
    }
    const status = byId("static-status");
    status.textContent = measured ? `${measured} of 4 modes measured` : "Awaiting measurements";
    status.classList.toggle("ready", measured === MODES.length);
  }

  function renderTransport(scenario) {
    const indexed = indexedRows(transportRows.filter((row) => row.scenario === scenario));
    const body = byId("transport-results");
    body.replaceChildren();
    for (const mode of MODES) {
      const row = indexed.get(mode) ?? {};
      const tr = document.createElement("tr");
      tr.append(
        modeCell(mode),
        measurementCell(percent(row.hold_accuracy)),
        measurementCell(percent(row.hazard_hold_rate)),
        measurementCell(percent(row.clear_go_rate)),
        measurementCell(bytes(row.mean_packet_bytes)),
      );
      body.append(tr);
    }
  }

  function loadTransport(rows) {
    transportRows = (Array.isArray(rows) ? rows : []).filter((row) => row &&
      typeof row === "object" && typeof row.scenario === "string" && row.scenario.trim() && MODES.includes(row.mode));
    const scenarios = [...new Set(transportRows.map((row) => row.scenario))];
    const select = byId("transport-scenario");
    select.replaceChildren();
    if (!scenarios.length) {
      const option = document.createElement("option");
      option.textContent = "No measurements yet";
      select.append(option);
      select.disabled = true;
    } else {
      for (const scenario of scenarios) {
        const option = document.createElement("option");
        option.value = scenario;
        option.textContent = humanize(scenario);
        select.append(option);
      }
      select.disabled = false;
    }
    byId("transport-count").textContent = scenarios.length
      ? `${scenarios.length} recorded ${scenarios.length === 1 ? "condition" : "conditions"}`
      : "No conditions loaded";
    renderTransport(scenarios[0]);
  }

  function renderFindings(notes) {
    const entries = (Array.isArray(notes) ? notes : typeof notes === "string" ? [notes] : [])
      .filter((entry) => typeof entry === "string" && entry.trim());
    const container = byId("findings-content");
    container.replaceChildren();
    for (const entry of entries) {
      const paragraph = document.createElement("p");
      paragraph.textContent = entry;
      container.append(paragraph);
    }
    byId("findings-card").hidden = entries.length === 0;
  }

  function artifactUrl(value) {
    if (typeof value !== "string" || !value || value.includes("\\") || value.includes("\0")) return null;
    const path = value.startsWith("/runs/") ? value : value.startsWith("learned-latent/") ? `/runs/${value}` : null;
    if (!path || path.split("/").some((part) => part === "." || part === ".." || part.startsWith("."))) return null;
    const url = new URL(path, window.location.origin);
    return url.origin === window.location.origin && url.pathname.startsWith("/runs/") ? url.href : null;
  }

  function renderLive(live) {
    const row = live && typeof live === "object" && !Array.isArray(live) ? live : {};
    byId("live-calls").textContent = count(row.calls_per_arm) ?? "Not reported";
    byId("live-ticks").textContent = count(row.ticks) ?? "Not reported";
    byId("live-grasps").textContent = count(row.grasp_count) ?? "Not reported";
    byId("live-fold").textContent = row.full_task_success === false ? "Not completed"
      : row.full_task_success === true ? "Reported in source" : "Not established";
    byId("live-status").textContent = typeof row.status === "string" && row.status.trim()
      ? humanize(row.status) : "Trial status unreported";
    const url = artifactUrl(row.video);
    const video = byId("latent-video");
    if (url && new URL(url).pathname.endsWith(".mp4")) {
      video.src = url;
      video.hidden = false;
      byId("latent-video-empty").hidden = true;
    }
  }

  function evidenceUrl(value) {
    if (typeof value !== "string") return null;
    const local = artifactUrl(value);
    if (local) return local;
    if (!value.startsWith("https://")) return null;
    try {
      const url = new URL(value);
      if (url.username || url.password) return null;
      return url.href;
    } catch {
      return null;
    }
  }

  function renderSources(sources) {
    const container = byId("source-links");
    container.replaceChildren();
    if (!sources || typeof sources !== "object" || Array.isArray(sources)) return;
    for (const [key, value] of Object.entries(sources)) {
      const url = evidenceUrl(typeof value === "string" ? value : value?.url ?? value?.path);
      if (!url) continue;
      const link = document.createElement("a");
      link.href = url;
      link.textContent = `${typeof value?.label === "string" ? value.label : humanize(key)} ↗`;
      if (new URL(url).origin !== window.location.origin) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      container.append(link);
    }
    container.hidden = container.childElementCount === 0;
  }

  byId("transport-scenario").addEventListener("change", (event) => renderTransport(event.target.value));
  byId("latent-video").addEventListener("loadedmetadata", (event) => {
    const duration = event.target.duration;
    if (finite(duration) && duration > 0) byId("trial-duration").textContent = `${duration.toFixed(1)} s · short execution check`;
  });
  byId("latent-video").addEventListener("error", () => {
    byId("latent-video").hidden = true;
    byId("latent-video-empty").hidden = false;
    byId("latent-video-empty").querySelector("strong").textContent = "Recording could not be loaded";
    byId("latent-video-empty").querySelector("p").textContent = "The numerical results remain available above. Check the raw measurements for the recording path.";
  });

  async function load() {
    renderStatic([]);
    loadTransport([]);
    try {
      const response = await fetch(RESULT_PATH, { cache: "no-store" });
      if (!response.ok) throw new Error(response.status === 404 ? "pending" : "unavailable");
      const data = await response.json();
      if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("invalid");
      renderStatic(data.static);
      loadTransport(data.transport);
      renderFindings(data.notes);
      renderLive(data.live);
      renderSources(data.sources);
      if (typeof data.heading === "string" && data.heading.trim()) {
        document.title = `Relay Lab · ${data.heading}`;
      }
    } catch (error) {
      const banner = byId("latent-error");
      banner.textContent = error.message === "pending"
        ? "Results have not been published yet. Unmeasured values are shown explicitly below."
        : "The measurements could not be loaded. This page shows no assumed or example scores.";
      banner.hidden = false;
    }
  }

  load();
})();
