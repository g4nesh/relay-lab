"""Read-only, source-backed Studio analytics; simulations and estimates stay separate.

No inference, robot execution, database mutations, or external network calls are
triggered by this service. Archive paths are an allowlist, not request inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from .access import AuthenticationError

SOURCES = {
    "factory": (
        "examples/factory-communication/results.json",
        "Complete simulated dispatch task; 3 correlated workpieces per seed",
    ),
    "cloth": (
        "examples/cloth-coordination/primary/comparison.json",
        "Complete state-assisted cloth episode per seed/scenario/mode",
    ),
    "cloth_diagnostics": (
        "examples/cloth-coordination/diagnostics/comparison.json",
        "Two separate no-message cloth diagnostics; not pooled with primary",
    ),
    "learned": (
        "examples/cloth-coordination/learned-latent/results.json",
        "Recorded static/transport diagnostics and bounded live trial",
    ),
    "learned_evaluation": (
        "examples/cloth-coordination/learned-latent/models/evaluation.json",
        "Held-out arm decisions; correlated states from one episode",
    ),
    "learned_transport": (
        "examples/cloth-coordination/learned-latent/transport/manifest.json",
        "Frozen-state observation sequence; no actions integrated",
    ),
    "studio_archive": (
        "examples/studio/verification.json",
        "Archived captured-feature Studio exchange; no robot actuation",
    ),
}
LABELS = {
    "none": "No private messages",
    "text": "Structured text",
    "structured_text": "Structured text",
    "embeddings": "Embeddings",
    "hybrid": "Hybrid",
    "latent": "Learned latents",
    "mediated": "LLM mediation",
    "orchestrated": "LLM orchestration",
}
EFFORT_DEFAULTS = {
    "robots": 6,
    "tasks": 4,
    "manual_per_robot_hours": 8,
    "manual_pair_hours": 2,
    "manual_task_hours": 6,
    "relay_per_robot_hours": 2,
    "relay_task_hours": 2,
    "hourly_rate": 100,
}
EFFORT_CAVEATS = [
    "Illustrative planning assumptions, not measured engineering productivity or customer ROI.",
    "Manual pair work assumes every unordered robot pair needs integration; change the input for your architecture.",
    "Hours represent person-hours, not elapsed delivery time. Hardware, training, validation, deployment and maintenance are not included.",
    "No human-team time study exists in the recorded experiments. Negative savings are retained.",
]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    """Linear-interpolated percentiles of raw samples, never averages of percentiles."""
    values = sorted(float(v) for v in values if _number(v) and v >= 0)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None}

    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "n": len(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "mean": sum(values) / len(values),
    }


def _mean(values):
    return distribution(values)["mean"]


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _wilson(successes, attempts):
    if not attempts:
        return [None, None]
    z = 1.959963984540054
    rate = successes / attempts
    centre = rate + z * z / (2 * attempts)
    spread = z * math.sqrt(rate * (1 - rate) / attempts + z * z / (4 * attempts * attempts))
    denominator = 1 + z * z / attempts
    return [max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)]


def estimate_effort(inputs):
    """Transparent editable scenario; never treated as an observed KPI."""
    if not isinstance(inputs, dict) or set(inputs) != set(EFFORT_DEFAULTS):
        raise ValueError("Provide exactly the documented effort scenario inputs")
    values = {}
    for key, value in inputs.items():
        if not _number(value) or value < 0 or value > 100000:
            raise ValueError(f"{key} must be a finite number between 0 and 100000")
        if key in ("robots", "tasks") and (int(value) != value or value < 1 or value > 10000):
            raise ValueError(f"{key} must be a whole number between 1 and 10000")
        values[key] = value
    edges = values["robots"] * (values["robots"] - 1) // 2
    manual = (
        values["robots"] * values["manual_per_robot_hours"]
        + edges * values["manual_pair_hours"]
        + values["tasks"] * values["manual_task_hours"]
    )
    relay = (
        values["robots"] * values["relay_per_robot_hours"]
        + values["tasks"] * values["relay_task_hours"]
    )
    saved = manual - relay
    return {
        "kind": "scenario",
        "measured": False,
        "inputs": values,
        "edges": edges,
        "manual_hours": manual,
        "relay_hours": relay,
        "hours_saved": saved,
        "savings_percent": saved / manual * 100 if manual else None,
        "manual_cost": manual * values["hourly_rate"],
        "relay_cost": relay * values["hourly_rate"],
        "cost_saved": saved * values["hourly_rate"],
        "currency": "USD",
        "caveats": EFFORT_CAVEATS,
    }


def _task_row(mode, runs):
    successes = sum(r.get("success") is True for r in runs)
    return {
        "mode": mode,
        "label": LABELS.get(mode, mode),
        "attempts": len(runs),
        "successes": successes,
        "failures": len(runs) - successes,
        "success_rate": _rate(successes, len(runs)),
        "wilson95": _wilson(successes, len(runs)),
        "mean_ticks": _mean(r.get("ticks") for r in runs),
        "mean_wall_seconds": _mean(r.get("wall_seconds") for r in runs),
        "wall_ms": distribution(
            r["wall_seconds"] * 1000 for r in runs if _number(r.get("wall_seconds"))
        ),
    }


def summarize_factory(data):
    if not data:
        return {
            "status": "unavailable",
            "rows": [],
            "caveats": ["Factory evidence is unavailable."],
        }
    runs = data.get("runs", [])
    modes = ("embeddings", "text", "none")
    grouped = {mode: [r for r in runs if r.get("mode") == mode] for mode in modes}
    expected = set(data.get("seeds", []))
    paired = []
    reasons = []
    # Independently validate identities in the raw run records, in addition to
    # the benchmark's sensor/schedule parity audit. Never silently drop failures.
    checks = {c["seed"]: c for c in data.get("paired_checks", [])}
    for seed in sorted(expected):
        cohort = [r for r in runs if r.get("seed") == seed]
        check = checks.get(seed, {})
        communication = [r.get("communication", {}) for r in cohort]
        codec_hashes = {c.get("model", {}).get("checkpoint_sha256") for c in communication}
        sensor_hashes = {
            tuple(
                (o.get("job_id"), o.get("source_frame_sha256"))
                for o in c.get("observation_provenance", [])
            )
            for c in communication
        }
        valid = (
            len(cohort) == 3
            and {r.get("mode") for r in cohort} == set(modes)
            and len({r.get("initial_physical_state_sha256") for r in cohort}) == 1
            and all(r.get("initial_physical_state_sha256") for r in cohort)
            and len({r.get("actor_sha256") for r in cohort}) == 1
            and all(r.get("actor_sha256") for r in cohort)
            and len(codec_hashes) == 1
            and None not in codec_hashes
            and len(sensor_hashes) == 1
            and () not in sensor_hashes
            and all(
                len(c.get("observation_provenance", [])) == data.get("orders")
                for c in communication
            )
            and len({(c.get("period_ticks"), c.get("ttl_ticks")) for c in communication}) == 1
            and all(r.get("orders") == data.get("orders") for r in cohort)
            and all(
                check.get(k) is True
                for k in (
                    "identical_initial_physics",
                    "complete_private_observations",
                    "identical_private_sensor_inputs",
                    "identical_actor_weights",
                    "identical_codec_weights",
                )
            )
            and not check.get("setup_failed_modes")
        )
        if valid:
            paired.append(seed)
        else:
            reasons.append(f"Seed {seed}: incomplete or unmatched evidence")
    verified = bool(expected) and len(paired) == len(expected) and len(runs) == len(expected) * 3
    rows = []
    for mode in modes:
        cohort = grouped[mode]
        row = _task_row(mode, cohort)
        row.update(
            correct_routes=sum(r.get("correct_routes", 0) for r in cohort),
            total_routes=sum(r.get("orders", 0) for r in cohort),
            payload_bytes=sum(r.get("source_payload_bytes", 0) for r in cohort),
            packet_bytes=sum(
                r.get("communication", {}).get("metrics", {}).get("packet_payload_write_bytes", 0)
                for r in cohort
            ),
            encode_ms=distribution(
                v for r in cohort for v in r.get("communication", {}).get("encode_ms", [])
            ),
            decode_ms=distribution(
                v for r in cohort for v in r.get("communication", {}).get("decode_ms", [])
            ),
            mean_simulation_seconds=None,
        )
        rows.append(row)
    baseline = rows[2]
    gains = (
        {r["mode"]: (r["success_rate"] - baseline["success_rate"]) * 100 for r in rows[:2]}
        if verified
        else {}
    )
    return {
        "status": "measured",
        "title": "Factory dispatch · matched private-information experiment",
        "evidence_id": "factory",
        "backend": data.get("coordination"),
        "planner": data.get("planner"),
        "attempts": len(runs),
        "paired_cases": len(paired),
        "pairing_verified": verified,
        "pairing_errors": reasons,
        "seeds": sorted(expected),
        "rows": rows,
        "comparison": {"baseline": "none", "success_gain_pp": gains, "time_saved_seconds": None},
        "caveats": [
            "No-private-message robots still share public stages, custody, leases and handoff ACKs. This is not fully independent operation.",
            "The motor actor and routing codec are original task-trained models, not a general foundation VLA.",
            "This cohort uses an in-memory coordination backend; it cannot measure a SpacetimeDB speedup.",
            "All attempts and failures are retained. Success intervals use complete tasks, not correlated workpieces.",
            "Wall time includes rendering, logging and model startup. It is not robot cycle time or human engineering effort.",
            "Simulation ticks are recorded; a per-run simulation timestep is absent, so simulation seconds are not inferred.",
        ],
    }


def summarize_cloth(primary, diagnostic):
    rows = []
    for scenario in (primary or {}).get("scenarios", []):
        for mode in ("embeddings", "structured_text", "hybrid"):
            runs = [
                r
                for r in primary.get("episodes", [])
                if r.get("mode") == mode and r.get("scenario") == scenario
            ]
            row = _task_row(mode, runs)
            row.update(
                scenario=scenario,
                packet_bytes=sum(r.get("communication", {}).get("packet_bytes", 0) for r in runs),
                mean_simulation_seconds=None,
            )
            rows.append(row)
    diagnostic_runs = (diagnostic or {}).get("episodes", [])
    return {
        "status": "measured" if primary else "unavailable",
        "title": "Two-arm cloth · task-trained controllers",
        "evidence_id": "cloth",
        "rows": rows,
        "paired_cases": len((primary or {}).get("paired_checks", [])),
        "backend": (primary or {}).get("coordination"),
        "simulation_tick_seconds": 0.04,
        "diagnostics": {
            "status": "measured" if diagnostic else "unavailable",
            "evidence_id": "cloth_diagnostics",
            "rows": [_task_row("none", diagnostic_runs)] if diagnostic_runs else [],
            "pooled_with_primary": False,
            "conclusion": (diagnostic or {}).get("conclusion"),
        },
        "caveats": [
            "Primary modes share seeds, physics, observation boundaries and action budget; only 2 seeds × 2 scenarios per mode.",
            "The two no-message attempts are separate diagnostics and are not pooled into a primary communication-effect estimate.",
            "Policies use measured local landmarks, geometric task scaffolding, a vacuum fixture and idealized grasps; this is not foundation-VLA folding.",
            "PCA compresses measured local features. Structured text carries full-precision measurements; the channel content differs.",
            "Synthetic delays advance with simulation; wall-clock inference and database calls pause the worker.",
        ],
    }


def summarize_learned(summary, evaluation, transport):
    static_rows = []
    for mode, item in (evaluation or {}).get("modes", {}).items():
        test = item.get("test", {})
        arms = list(test.get("arms", {}).values())
        hidden_count = sum(a.get("hidden_peer_count", 0) for a in arms)
        hidden_correct = sum(
            round(a.get("hidden_peer_accuracy", 0) * a.get("hidden_peer_count", 0)) for a in arms
        )
        decisions = test.get("pairs", 0) * len(arms)
        correct = sum(
            test.get("pairs", 0) - a.get("unsafe_go_count", 0) - a.get("false_hold_count", 0)
            for a in arms
        )
        static_rows.append(
            {
                "mode": mode,
                "label": LABELS.get(mode, mode),
                "hidden_correct": hidden_correct,
                "hidden_decisions": hidden_count,
                "hidden_accuracy": _rate(hidden_correct, hidden_count),
                "hold_correct": correct,
                "decisions": decisions,
                "hold_accuracy": _rate(correct, decisions),
                "action_error_mm": _mean(a.get("mean_action_error_mm") for a in arms),
            }
        )
    transport_rows = []
    attempts = {
        (a.get("mode"), a.get("scenario")): a for a in (transport or {}).get("attempts", [])
    }
    for row in (summary or {}).get("transport", []):
        attempt = attempts.get((row["mode"], row["scenario"]), {})
        arms = list(attempt.get("metrics", {}).values())
        count = sum(a.get("decisions", 0) for a in arms)
        incorrect = sum(a.get("unsafe_go_count", 0) + a.get("false_hold_count", 0) for a in arms)
        hidden_count = row.get("posthoc_hidden_peer_count", 0)
        hidden_correct = round(row.get("posthoc_hidden_peer_accuracy", 0) * hidden_count)
        comm = attempt.get("communication", {})
        transport_rows.append(
            {
                "mode": row["mode"],
                "label": LABELS.get(row["mode"], row["mode"]),
                "scenario": row["scenario"],
                "hidden_correct": hidden_correct,
                "hidden_decisions": hidden_count,
                "hidden_accuracy": _rate(hidden_correct, hidden_count),
                "hold_correct": count - incorrect,
                "decisions": count,
                "hold_accuracy": _rate(count - incorrect, count),
                "unsafe_go_count": sum(a.get("unsafe_go_count", 0) for a in arms),
                "packet_bytes": comm.get("packet_bytes"),
                "mean_packet_bytes": row.get("mean_packet_bytes"),
                "published": comm.get("published"),
                "delivered": comm.get("delivered"),
                "dropped": comm.get("dropped"),
                "expired": comm.get("expired"),
                "stale_reads": comm.get("stale_reads"),
            }
        )
    live = (summary or {}).get("live", {})
    return {
        "status": "measured" if summary else "unavailable",
        "title": "Mixed foundation VLA adapters · diagnostic evidence",
        "evidence_id": "learned",
        "static": static_rows,
        "transport": transport_rows,
        "live": {
            "status": live.get("status", "unavailable"),
            "grasp_count": live.get("grasp_count"),
            "full_task_success": live.get("full_task_success"),
            "ticks": live.get("ticks"),
            "calls_per_arm": live.get("calls_per_arm"),
            **live.get("hold_audit", {}),
        },
        "caveats": (summary or {}).get("notes", [])
        + [
            "Decision accuracy is not a task success rate. No reliable complete fold by the mixed foundation VLA pair has been demonstrated."
        ],
    }


def _journal_observations(history):
    """Observe committed message groups without inventing the original ACK/timing.

    inspect() deliberately does not rebuild results or run model inference. Its
    bounded event read is still evidence that a numbered exchange was committed.
    """
    tick = history.get("tick")
    if type(tick) is not int or tick < 1:
        return []
    grouped = defaultdict(dict)
    for message in history.get("messages", []):
        if not isinstance(message, dict) or not isinstance(message.get("payload"), dict):
            continue
        number = message["payload"].get("exchange")
        if type(number) is not int or not max(1, tick - 49) <= number <= tick:
            continue
        identity = message.get("message_id")
        if not isinstance(identity, str) or not identity:
            continue
        grouped[number][identity] = message
    observations = []
    for number, messages_by_id in sorted(grouped.items(), reverse=True):
        messages = sorted(messages_by_id.values(), key=lambda m: m["message_id"])
        modes = {m["payload"].get("mode") for m in messages}
        mode = next(iter(modes)) if len(modes) == 1 else "unknown"
        if mode not in ("mediated", "orchestrated", "latent", "hybrid"):
            mode = "unknown"
        provenance = {m["payload"].get("provenance") for m in messages}
        source = (
            "captured"
            if provenance == {"captured foundation features"}
            else "manual"
            if provenance == {"user / external worker report"}
            else "unknown"
        )
        observations.append(
            {
                "exchange": number,
                "mode": mode,
                "source": source,
                "bytes": len(json.dumps(messages, separators=(",", ":")).encode()),
                "elapsed_ms": None,
                "database_ms": None,
                "llm_ms": None,
                "readback_verified": None,
                "actions_executed": None,
                "journal_only": True,
            }
        )
    return observations


def _operational_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("mode", "unknown")].append(row)
    return [
        {
            "mode": mode,
            "label": LABELS.get(mode, mode),
            "exchanges": len(items),
            "bytes": sum(r.get("bytes", 0) for r in items if _number(r.get("bytes"))),
            "elapsed_ms": distribution(r.get("elapsed_ms") for r in items),
            "database_ms": distribution(r.get("database_ms") for r in items),
            "llm_ms": distribution(r.get("llm_ms") for r in items),
            "verified_readbacks": sum(
                r.get("readback_verified") is True or r.get("readback_verified") == 1 for r in items
            ),
            "actions_executed": sum(
                r.get("actions_executed") is True or r.get("actions_executed") == 1 for r in items
            ),
            "captured_exchanges": sum(r.get("source") == "captured" for r in items),
        }
        for mode, items in sorted(groups.items())
    ]


class AnalyticsService:
    """Bounded aggregates, five-second cache, fresh authorization on every call."""

    def __init__(self, root, studio_service, registry, access=None, *, cache_seconds=5):
        self.root = Path(root).resolve()
        self.studio = studio_service
        self.registry = registry
        self.access = access
        self.cache_seconds = cache_seconds
        self._cache = None
        self._cached_at = 0.0
        self._lock = threading.Lock()

    def _sources(self):
        if not getattr(self.studio, "experiments_enabled", True):
            return {}, []
        values, manifest = {}, []
        for identifier, (relative, grain) in SOURCES.items():
            item = {
                "id": identifier,
                "path": relative,
                "grain": grain,
                "status": "unavailable",
                "sha256": None,
                "bytes": None,
            }
            try:
                path = self.root / relative
                if (
                    not path.resolve().is_relative_to(self.root)
                    or path.stat().st_size > 8 * 1024 * 1024
                ):
                    raise ValueError("Evidence path or size outside allowed bounds")
                raw = path.read_bytes()
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise TypeError("Expected object")
                values[identifier] = data
                item.update(
                    status="available", sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)
                )
            except (OSError, ValueError, TypeError):
                # No filesystem exceptions or absolute local paths reach users.
                item["reason"] = "Evidence missing or invalid"
            manifest.append(item)
        return values, manifest

    def _operations(self, archive):
        rows, session_rows, notes = {}, [], []
        telemetry_total, telemetry_limit = 0, 500
        telemetry = getattr(self.studio, "telemetry", None)
        if telemetry is not None:
            try:
                measured = telemetry.snapshot(limit=telemetry_limit)
                telemetry_total = measured.get("total_rows", 0)
                for row in measured.get("rows", [])[:telemetry_limit]:
                    rows[(row["session_id"], row["exchange"])] = row
                if measured.get("unavailable_reason"):
                    notes.append("Local exchange telemetry is unavailable.")
            except (OSError, ValueError, RuntimeError, sqlite3.Error):
                notes.append("Local exchange telemetry is unavailable.")
        unreadable = 0
        history_sessions = 0
        inventory_available = False
        session_health = {}
        try:
            # A control-plane workspace has no research sessions. In particular,
            # never backfill its telemetry from a database-wide research journal.
            research = getattr(self.studio, "experiments_enabled", True)
            listed = self.studio.list_sessions() if research else {"sessions": []}
            session_rows = listed.get("sessions", [])[:200]
            inventory_available = research
            session_health = {
                s["session_id"]: True if s.get("status") == "failed" else None for s in session_rows
            }
            for session in session_rows[:20]:
                try:
                    history = self.studio.history(session["session_id"])
                    history_sessions += 1
                    failure = history.get("failed")
                    if type(failure) is bool:
                        session_health[session["session_id"]] = (
                            session_health[session["session_id"]] is True or failure
                        )
                    for row in history.get("results", [])[-50:]:
                        key = (session["session_id"], row["exchange"])
                        if key not in rows and len(rows) < 500:
                            rows[key] = row
                    for row in _journal_observations(history):
                        key = (session["session_id"], row["exchange"])
                        if key not in rows and len(rows) < 500:
                            rows[key] = row
                except (OSError, ValueError, RuntimeError, sqlite3.Error):
                    unreadable += 1
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            notes.append(
                "SpacetimeDB session inventory is unavailable; archived evidence remains separate."
            )
        archive_rows = [
            a["result"]
            for a in (archive or {}).get("attempts", [])
            if isinstance(a.get("result"), dict)
        ]
        values = list(rows.values())
        known_failed = sum(failed is True for failed in session_health.values())
        unknown_health = sum(failed is None for failed in session_health.values())
        return {
            "status": "measured" if values or session_rows else "unavailable",
            "sessions": len(session_rows),
            "exchanges": len(values),
            "failed_sessions": known_failed
            if inventory_available and unknown_health == 0
            else None,
            "known_failed_sessions": known_failed,
            "modes": _operational_rows(values),
            "backend": "SpacetimeDB",
            "coverage": {
                "session_inventory_limit": 200,
                "session_inventory_may_be_truncated": len(session_rows) == 200,
                "history_sessions_read": history_sessions,
                "history_session_limit": 20,
                "history_window_per_session": 50,
                "unreadable_sessions": unreadable,
                "exchange_window_limit": 500,
                "telemetry_total_rows": telemetry_total,
                "journal_only_exchanges": sum(r.get("journal_only") is True for r in values),
                "session_health_checked": len(session_health) - unknown_health,
                "session_health_unknown": unknown_health,
                "failed_session_count_complete": inventory_available and unknown_health == 0,
                "timed_exchanges": sum(_number(r.get("elapsed_ms")) for r in values),
                "database_timed_exchanges": sum(_number(r.get("database_ms")) for r in values),
                "llm_timed_exchanges": sum(_number(r.get("llm_ms")) for r in values),
                "timing_coverage": _rate(
                    sum(_number(r.get("elapsed_ms")) for r in values), len(values)
                ),
            },
            "archive": {
                "status": "measured" if archive_rows else "unavailable",
                "evidence_id": "studio_archive",
                "exchanges": len(archive_rows),
                "modes": _operational_rows(archive_rows),
                "included_in_live_totals": False,
            },
            "caveats": notes
            + [
                "Message-only journal groups establish observed committed exchanges, not the original readback verification, inference timings or action execution. Those fields remain unknown.",
                "Session failures reconcile the database inventory with inspected worker state. A complete failure count is unknown when session health is unreadable or outside the history window.",
                "Latency percentiles use individual exchange samples. Missing post-ACK timings remain unknown after restart; zero is not substituted.",
                "A maximum of 500 exchanges and 20 session histories is inspected. Session inventory is bounded to 200; this is not a lifetime total.",
                "An idempotent retry is counted once by session/exchange. Live telemetry and journal entries are deduplicated. Failed requests without committed exchanges have no latency record.",
                "Captured-feature exchanges exclude foundation vision inference. These channel timings do not demonstrate physical task completion.",
                "Studio elapsed time measures execution through the prediction preview. It excludes time queued behind the governance lock, local telemetry persistence, and HTTP response delivery; it is not client-perceived latency.",
                "Timing coverage and archive provenance are explicit; archived verification is never counted as current traffic.",
            ],
        }

    def snapshot(self, actor):
        if (
            not isinstance(actor, dict)
            or actor.get("role") not in ("manager", "employee")
            or not actor.get("id")
        ):
            raise AuthenticationError("Sign in to view analytics")
        # Read catalog using the current actor even on a cache hit. Only lifecycle
        # counts leave this method; never specs, names, account IDs or requests.
        catalog = self.registry.catalog(actor)
        statuses = Counter(m.get("status") for m in catalog.get("models", []))
        governance = {
            "status": "measured",
            "models": {
                "total": sum(statuses.values()),
                **{s: statuses[s] for s in ("draft", "released", "retired")},
            },
            "modeled": False,
            "scope": "Current model catalog; access policy remains enforced by the release registry.",
        }
        with self._lock:
            if self._cache is not None and time.monotonic() - self._cached_at < self.cache_seconds:
                result = copy.deepcopy(self._cache)
                result["governance"] = governance
                return result
            data, sources = self._sources()
            factory = summarize_factory(data.get("factory"))
            gain = factory.get("comparison", {}).get("success_gain_pp", {}).get("text")
            headline = {
                "label": "Observed dispatch success gain",
                "value": gain,
                "unit": "percentage points",
                "evidence_id": "factory",
                "detail": "Structured text vs no private messages on the same 12 simulated tasks. Public coordination remains enabled.",
                "time_saved": None,
                "physical_factory_efficiency_measured": False,
            }
            result = {
                "schema": "relay.studio.analytics.v1",
                "experiments_enabled": getattr(self.studio, "experiments_enabled", True),
                "generated_at": time.time(),
                "scope": "Original simulation evidence, bounded operational telemetry, and explicitly modeled engineering scenarios. No real-factory productivity claim.",
                "headline": headline,
                "factory": factory,
                "cloth": summarize_cloth(data.get("cloth"), data.get("cloth_diagnostics")),
                "learned": summarize_learned(
                    data.get("learned"),
                    data.get("learned_evaluation"),
                    data.get("learned_transport"),
                ),
                "operations": self._operations(data.get("studio_archive")),
                "governance": governance,
                "effort_model": {
                    "kind": "scenario",
                    "measured": False,
                    "defaults": EFFORT_DEFAULTS,
                    "formulas": {
                        "edges": "robots × (robots − 1) / 2",
                        "manual_hours": "robots × manual_per_robot_hours + edges × manual_pair_hours + tasks × manual_task_hours",
                        "relay_hours": "robots × relay_per_robot_hours + tasks × relay_task_hours",
                        "hours_saved": "manual_hours − relay_hours",
                        "cost_saved": "hours_saved × hourly_rate",
                    },
                    "caveats": EFFORT_CAVEATS,
                },
                "sources": sources,
                "definitions": {
                    "success_rate": "Successful complete attempts / all attempted tasks in the same cohort; failures retained.",
                    "paired_cases": "Unique seed/scenario cases with matching initial conditions, policies and communication schedule.",
                    "latency": "Milliseconds of measured wall time for the named operation; p50/p95 linearly interpolated from raw samples.",
                    "payload_bytes": "Unique UTF-8 source message content; excludes common envelope and retransmission.",
                    "packet_bytes": "UTF-8 application envelopes, including retransmissions; excludes network headers and SQL readback unless separately stated.",
                    "hidden_accuracy": "Correct hold/move decisions where the receiver's local view cannot distinguish peer hazards; not full task success.",
                    "simulation_time": "Simulator progress and wall-clock compute time are separate; unavailable simulated durations remain null.",
                    "engineering_effort": "Editable labor scenario; no empirical human-team benchmark has been conducted.",
                    "spacetime": "Shared ordered state, ownership-checked commits and exact indexed readback.",
                },
            }
            if not result["experiments_enabled"]:
                result["scope"] = (
                    "Current workspace operations and model governance. "
                    "Archived research fixtures are disabled in this deployment."
                )
                result["headline"] = {
                    "label": "Workspace operations",
                    "value": result["operations"]["exchanges"],
                    "unit": "recorded exchanges",
                    "evidence_id": None,
                    "detail": "Bounded current-workspace telemetry; no archived benchmark data.",
                    "time_saved": None,
                    "physical_factory_efficiency_measured": False,
                }
            self._cache = copy.deepcopy(result)
            self._cached_at = time.monotonic()
            return result
