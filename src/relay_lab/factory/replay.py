"""Compact replay export from committed trace events, never a fabricated live run."""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from copy import deepcopy
from pathlib import Path

SCHEMA = "relay.factory.replay.v1"
_OMIT = {"rgb", "rgba", "wrist_rgb", "image", "image_bytes", "pixels", "sensor_images"}


def _compact(value):
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key not in _OMIT}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def export_factory_replay(log_path, dest, summary=None, *, sample_every=5):
    """Export recorded dashboard frames; retain stage changes and terminal outcomes.

    Only ``tick`` rows are committed frames. ``execution_observed`` rows precede
    database confirmation and are deliberately excluded. The complete JSONL
    remains the canonical audit trail, including any unconfirmed final action.
    """
    if type(sample_every) is not int or not 1 <= sample_every <= 1000:
        raise ValueError("Replay sample period must be an integer from 1 to 1000")
    log_path, dest = Path(log_path), Path(dest)
    if log_path.resolve() == dest.resolve():
        raise ValueError("Replay output must not overwrite its full audit trace")
    summary = _compact(deepcopy(summary)) if summary is not None else None
    scene = None
    run_id = None
    backend, policy, planner = (
        {},
        {},
        {
            "mode": (summary or {}).get("planner", {}).get("mode", "unknown"),
            "reason": "Recorded initialization",
        },
    )
    messages, events = deque(maxlen=80), deque(maxlen=40)
    latest_embeddings = {}
    communication = None
    frames = []
    last_state, last_tick, last_stages = None, -1, None
    committed_ticks = decisions = completed_commands = failed_commands = 0
    sampling = sample_every
    trace_failure = None

    def dashboard(state, trace=None):
        trace = trace or {}
        tick = state["tick"]
        orders = list(state.get("jobs", {}).values())
        success = bool(state.get("success"))
        latent_records = [
            deepcopy(record)
            for record in latest_embeddings.values()
            if record.get("expires_tick", tick + 1) > tick
        ]
        return {
            **_compact(deepcopy(state)),
            "run_id": run_id,
            "status": "replay",
            "recorded_status": "succeeded" if success else "running",
            "recorded": True,
            "live": False,
            "sim_time": state.get("time", tick * state.get("dt", 0.1)),
            "orders": orders,
            "mission": {
                "completed": state.get(
                    "completed", sum(j.get("stage") == "completed" for j in orders)
                ),
                "total": state.get("total", len(orders)),
            },
            "stations": scene.get("stations", []) if scene else [],
            "messages": list(deepcopy(messages)),
            "events": list(deepcopy(events)),
            "planner": {**deepcopy(planner), "recorded": True, "calls": decisions},
            "policy": deepcopy((state.get("motor_policy") or {}).get("model") or policy),
            "coordination": {**backend, "connected": False, "recorded": True},
            "resources": [],
            "resource_telemetry": "not recorded in tick trace",
            "embeddings": {
                "enabled": bool(latest_embeddings),
                "latest": latent_records,
                "recorded": True,
                "metrics": {
                    "available": False,
                    "reason": "Per-frame retrieval latency is not in the trace",
                },
            },
            "communication": deepcopy(communication),
            "motor_actions": _compact(deepcopy(trace.get("motor_actions", []))),
            "causality": {
                "label": "Recorded peer embedding IDs used by motor inference",
                "motor_inputs": [
                    {
                        "robot_id": action.get("robot_id"),
                        "command_id": action.get("command_id"),
                        "peer_record_ids": action.get("peer_record_ids", []),
                    }
                    for action in trace.get("motor_actions", [])
                ],
            },
            "metrics": {
                "completed_commands": completed_commands,
                "failed_commands": failed_commands,
                "recorded": True,
            },
            "error": None,
        }

    with log_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed trace at line {line_number}") from error
            if run_id is not None and entry.get("run_id") != run_id:
                raise ValueError("Replay cannot combine different run IDs")
            kind = entry.get("type")
            if kind == "created":
                if scene is not None:
                    raise ValueError("Trace contains multiple initializations")
                run_id, scene = entry.get("run_id"), _compact(entry["scene"])
                backend, policy = entry.get("backend", {}), entry.get("policy", {})
                communication = _compact(entry.get("communication"))
                if not policy and summary:
                    policy = summary.get("policy", {})
                state = entry["state"]
                frames.append((True, dashboard(state)))
                last_state, last_tick = state, state["tick"]
                last_stages = {
                    key: value.get("stage") for key, value in state.get("jobs", {}).items()
                }
            elif kind == "decision":
                planner = {
                    **_compact(entry),
                    "reason": entry.get("reason", "Recorded planner decision"),
                }
                planner.pop("run_id", None)
                decisions += 1
            elif kind == "tick":
                if scene is None:
                    raise ValueError("Committed frame precedes initialization")
                state = entry["state"]
                if state["tick"] <= last_tick or entry.get("tick") != state["tick"]:
                    raise ValueError("Committed replay ticks must strictly increase")
                last_tick = state["tick"]
                committed_ticks += 1
                if entry.get("communication") is not None:
                    communication = {**(communication or {}), **_compact(entry["communication"])}
                for message in entry.get("messages", []):
                    messages.append(
                        {
                            **_compact(message),
                            "id": message.get("message_id"),
                            "tick": state["tick"],
                            "text": message.get("payload", {}).get("text", ""),
                        }
                    )
                events.extend(_compact(entry.get("events", [])))
                for embedding in entry.get("embeddings", []):
                    latest_embeddings[embedding["robot_id"]] = _compact(embedding)
                terminal = [
                    outcome
                    for outcome in entry.get("outcomes", [])
                    if outcome.get("status") in {"completed", "failed"}
                ]
                completed_commands += sum(outcome["status"] == "completed" for outcome in terminal)
                failed_commands += sum(outcome["status"] == "failed" for outcome in terminal)
                stages = {key: value.get("stage") for key, value in state.get("jobs", {}).items()}
                important = bool(terminal) or stages != last_stages or bool(state.get("success"))
                if important or state["tick"] % sampling == 0:
                    frames.append((important, dashboard(state, entry)))
                last_state, last_stages = state, stages
                # Bound memory even for a user-specified 100k-tick trace.
                if len(frames) > 4000:
                    sampling *= 2
                    frames = [
                        (key, frame)
                        for key, frame in frames
                        if key or frame["tick"] % sampling == 0
                    ]
            elif kind == "failure":
                trace_failure = {
                    "status": "failed",
                    "tick": entry.get("tick"),
                    "error": entry.get("error"),
                }
            # Pre-commit observation and failure rows are evidence, but never
            # promote a simulator observation into a committed replay frame.
    if scene is None or last_state is None:
        raise ValueError("Trace has no factory initialization")
    if not frames or frames[-1][1]["tick"] != last_state["tick"]:
        frames.append((True, dashboard(last_state)))
    important_count = sum(important for important, _ in frames)
    if important_count > 2000:
        raise ValueError("Trace has too many critical transitions for a bounded replay")
    if len(frames) > 2000:
        regular = [i for i, (important, _) in enumerate(frames) if not important]
        budget = 2000 - important_count
        selected = (
            {regular[(i * len(regular)) // budget] for i in range(budget)} if budget else set()
        )
        frames = [frame for i, frame in enumerate(frames) if frame[0] or i in selected]
    if summary is not None and summary.get("run_id", run_id) != run_id:
        raise ValueError("Replay summary belongs to another run")
    if summary is None:
        summary = {
            "run_id": run_id,
            "status": "succeeded" if last_state.get("success") else "recorded",
            "tick": last_tick,
            "mission": frames[-1][1]["mission"],
        }
    if trace_failure and summary.get("status") == "recorded":
        summary = {**summary, **trace_failure}
    final_status = summary.get("status")
    if final_status == "succeeded" and (
        not last_state.get("success") or summary.get("tick", last_tick) != last_tick
    ):
        raise ValueError("Summary claims success without matching committed success evidence")
    if final_status in {"succeeded", "rejected", "failed", "cancelled", "exhausted"}:
        final_frame = frames[-1][1]
        final_frame["recorded_status"] = final_status
        final_frame["error"] = summary.get("error")
        final_frame["terminal_annotation"] = {
            "source": "recorded run summary",
            "status": final_status,
            "reported_tick": summary.get("tick", last_tick),
            "last_committed_tick": last_tick,
        }
        if summary.get("tick") == last_tick and summary.get("embeddings"):
            # These measurements summarize the whole recorded run. Earlier
            # frames never pretend to have unrecorded per-frame timings.
            recorded = summary["embeddings"]
            final_frame["embeddings"].update(
                {
                    key: deepcopy(recorded[key])
                    for key in ("published", "space_id", "ttl_ticks", "metrics")
                    if key in recorded
                }
            )
            final_frame["embeddings"]["metrics_scope"] = "whole recorded run"
        if summary.get("tick") == last_tick and summary.get("communication"):
            final_frame["communication"] = deepcopy(summary["communication"])
            final_frame["communication"]["metrics_scope"] = "whole recorded run"
    summary = {**summary, "recorded": True, "live": False}
    if "coordination" in summary:
        summary["coordination"] = {**summary["coordination"], "connected": False, "recorded": True}
    result = {
        "schema": SCHEMA,
        "run_id": run_id,
        "recorded": True,
        "live": False,
        "scene": scene,
        "frames": [frame for _, frame in frames],
        "summary": summary,
        "source": {
            "trace": log_path.name,
            "committed_only": True,
            "committed_ticks": committed_ticks,
            "sample_every": sample_every,
            "full_trace_retained": True,
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, separators=(",", ":"), allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=dest.parent, prefix=dest.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(dest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
