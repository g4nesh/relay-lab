#!/usr/bin/env python3
"""Paired, full-physics partial-observation experiment; never selects successful seeds.

Defaults: 12 held-out seeds, three jobs each, identical frozen VLA, a fixed
5000-tick budget, deterministic public scheduler, and the in-memory equivalent
of Spacetime's transactional barrier. Use --coordination spacetime --planner
llm for the actual Qwen/Spacetime integration comparison. Runs are sequential;
mode order rotates by seed. Failures remain in the denominator and recordings.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from relay_lab.factory.runtime import FactoryRun

MODES = ("embeddings", "text", "none")


def wilson(wins, total):
    if not total:
        return [0.0, 1.0]
    z = 1.96
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - half), min(1, center + half)]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def aggregate(rows):
    result = {}
    for mode in MODES:
        matching = [row for row in rows if row["mode"] == mode]
        successes = sum(row["success"] for row in matching)
        correct = sum(row["correct_routes"] for row in matching)
        total = sum(row["orders"] for row in matching)
        result[mode] = {
            "runs": len(matching),
            "successful_tasks": successes,
            "task_success_rate": successes / len(matching) if matching else None,
            "task_success_wilson_95": wilson(successes, len(matching)),
            "correct_routes": correct,
            "total_routes": total,
            "route_accuracy": correct / total if total else None,
            "physically_completed": sum(row["completed"] for row in matching),
            "failed_skills": sum(row["failed_skills"] for row in matching),
            "mean_ticks": statistics.mean(row["ticks"] for row in matching) if matching else None,
            "mean_wall_seconds": statistics.mean(row["wall_seconds"] for row in matching)
            if matching
            else None,
            "packet_count": sum(row["communication"]["published"] for row in matching),
            "setup_failures": sum(row["status"] == "setup_failed" for row in matching),
            "packet_envelope_write_bytes": sum(
                row["communication"].get("metrics", {}).get("packet_payload_write_bytes", 0)
                for row in matching
            ),
            "refresh_transport_body_bytes": sum(
                row["communication"].get("metrics", {}).get("transport_bytes", 0)
                for row in matching
            ),
            "source_payload_measured_runs": sum("source_payload_bytes" in row for row in matching),
            "source_payload_bytes": sum(row["source_payload_bytes"] for row in matching)
            if matching and all("source_payload_bytes" in row for row in matching)
            else None,
            "source_payload_count": sum(row["source_payload_count"] for row in matching)
            if matching and all("source_payload_count" in row for row in matching)
            else None,
            "source_encoding_ms_total": sum(
                sum(row["communication"].get("encode_ms", [])) for row in matching
            ),
            "semantic_decode_parse_ms_total": sum(
                sum(row["communication"].get("decode_ms", [])) for row in matching
            ),
            "semantic_decode_parse_ms_per_attempt": statistics.mean(
                sum(row["communication"].get("decode_ms", [])) for row in matching
            )
            if matching
            else None,
            "semantic_decode_parse_scope": {
                "embeddings": "receiver neural decoding; total per attempted run, not per call",
                "text": "sender neural decoding plus receiver JSON parsing; total per attempted run, not per call",
                "none": "no route decoding or parsing",
            }[mode],
            "source_payload_scope": "Unique source content generated for enabled channel; excludes common envelope and retransmissions; generated content may not have committed",
        }
    return result


def paired_checks(rows):
    def same_real_hash(values):
        return (
            all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in values)
            and len(set(values)) == 1
        )

    results = []
    for seed in sorted({row["seed"] for row in rows}):
        group = {row["mode"]: row for row in rows if row["seed"] == seed}
        if set(group) != set(MODES):
            continue
        hashes = [row["initial_physical_state_sha256"] for row in group.values()]
        sensor_hashes = [
            sorted(
                (v["job_id"], v["source_frame_sha256"])
                for v in row["communication"]["observation_provenance"]
            )
            for row in group.values()
        ]
        a, b = group["embeddings"], group["text"]
        complete_observations = all(
            len(row["communication"]["observation_provenance"]) == row["orders"]
            and len({v["job_id"] for v in row["communication"]["observation_provenance"]})
            == row["orders"]
            and all(
                isinstance(v["source_frame_sha256"], str)
                and re.fullmatch(r"[a-f0-9]{64}", v["source_frame_sha256"])
                for v in row["communication"]["observation_provenance"]
            )
            for row in group.values()
        )
        result = {
            "seed": seed,
            "setup_failed_modes": [
                mode for mode, row in group.items() if row["status"] == "setup_failed"
            ],
            "identical_initial_physics": same_real_hash(hashes),
            "complete_private_observations": complete_observations,
            "identical_private_sensor_inputs": complete_observations
            and all(value == sensor_hashes[0] for value in sensor_hashes),
            "identical_actor_weights": same_real_hash(
                [row["actor_sha256"] for row in group.values()]
            ),
            "identical_codec_weights": same_real_hash(
                [row["communication"]["model"]["checkpoint_sha256"] for row in group.values()]
            ),
            "embedding_text_decisions_equal": len(a["communication"]["decisions"]) == a["orders"]
            and len(b["communication"]["decisions"]) == b["orders"]
            and [(v["job_id"], v["route"]) for v in a["communication"]["decisions"]]
            == [(v["job_id"], v["route"]) for v in b["communication"]["decisions"]],
            "none_packets": group["none"]["communication"]["published"],
            "embedding_success": a["success"],
            "text_success": b["success"],
            "none_success": group["none"]["success"],
        }
        results.append(result)
    return results


def run_trial(args, seed, mode, directory):
    started = time.perf_counter()
    try:
        run = FactoryRun(
            seed=seed,
            orders=args.orders,
            planner=args.planner,
            coordination=args.coordination,
            max_ticks=args.max_ticks,
            output=directory,
            task="dispatch-routing",
            communication=mode,
            embeddings=False,
        )
    except Exception as error:  # noqa: BLE001 - setup failures remain in the planned denominator
        detail = f"{type(error).__name__}: {error}"
        print(f"ERROR seed={seed} mode={mode}: {detail}", flush=True)
        row = {
            "seed": seed,
            "mode": mode,
            "orders": args.orders,
            "run_id": None,
            "status": "setup_failed",
            "success": False,
            "correct_routes": 0,
            "completed": 0,
            "ticks": 0,
            "wall_seconds": time.perf_counter() - started,
            "failed_skills": 0,
            "initial_physical_state_sha256": None,
            "actor_sha256": None,
            "motor_calls": {},
            "communication": {
                "mode": mode,
                "model": {"checkpoint_sha256": None},
                "local_observations": 0,
                "observation_provenance": [],
                "decisions": [],
                "published": 0,
                "encode_ms": [],
                "decode_ms": [],
                "metrics": {},
            },
            "source_payload_bytes": 0,
            "source_payload_count": 0,
            "oracle": None,
            "error": detail,
            "directory": str(directory.resolve()),
        }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "setup_failure.json").write_text(json.dumps(row, indent=2))
        return row
    initial = {key: run.state[key] for key in ("robots", "cargo", "jobs", "tick")}
    initial_hash = digest(initial)
    try:
        while run.status not in {"succeeded", "rejected", "exhausted", "failed"}:
            run.tick()
    except Exception as error:  # noqa: BLE001 - every failed run is scored and retained
        print(f"ERROR seed={seed} mode={mode}: {error}", flush=True)
    finally:
        run.close()
    summary = json.loads((directory / "summary.json").read_text())
    memories = run.routing_exchange.memories if mode != "none" else {}
    source_bytes = sum(
        len(json.dumps(memory["vector"], separators=(",", ":")).encode())
        if mode == "embeddings"
        else len(memory["text"].encode())
        for memory in memories.values()
    )
    return {
        "seed": seed,
        "mode": mode,
        "orders": args.orders,
        "run_id": summary["run_id"],
        "status": summary["status"],
        "success": summary["success"] and summary["status"] == "succeeded",
        "correct_routes": sum(
            job.get("routing_correct") is True for job in summary["jobs"].values()
        ),
        "completed": summary["completed"],
        "ticks": summary["tick"],
        "wall_seconds": summary["metrics"]["wall_seconds"],
        "failed_skills": summary["metrics"]["failed_commands"],
        "initial_physical_state_sha256": initial_hash,
        "actor_sha256": summary["policy"]["checkpoint_sha256"],
        "motor_calls": summary["motor_policy"]["metrics"],
        "communication": summary["communication"],
        "source_payload_bytes": source_bytes,
        "source_payload_count": len(memories),
        "oracle": summary.get("routing_oracle"),
        "error": summary.get("error"),
        "directory": str(directory.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 113)))
    parser.add_argument("--orders", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-ticks", type=int, default=5000)
    parser.add_argument("--coordination", choices=("memory", "spacetime"), default="memory")
    parser.add_argument("--planner", choices=("rules", "llm"), default="rules")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(
        seed < 0 or seed >= 2**32 for seed in args.seeds
    ):
        parser.error("Seeds must be unique nonnegative 32-bit integers")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "relay.factory.communication-benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "dispatch-routing",
        "seeds": args.seeds,
        "orders": args.orders,
        "max_ticks": args.max_ticks,
        "planner": args.planner,
        "coordination": args.coordination,
        "mode_order": "rotate embeddings/text/none by seed index; sequential execution",
        "policy": "frozen factory-vla actor; no analytic motor fallback",
        "partial_information": "Only humanoid receives private RGB routing cue. Gantry sees local dispatch scene. Central planner gets public stage/custody and action candidates only.",
        "parity": "Same learned visual prediction; embedding decoding at receiver versus exact JSON route text decoded at sender. Same cadence/TTL/job scope. Control-plane leases and stage ACKs remain in all modes.",
        "no_communication_prior": "Always A, independent of seed, cue and job ID",
        "limitations": [
            "Two-class immutable routing cue in calibrated synthetic inspection imagery; task-specific learned codec, not a general semantic foundation embedding.",
            "The motor policy consumes the decoded symbolic slot; this tests communication for a local subgoal, not end-to-end latent motor learning.",
            "All failures are retained; Wilson interval is over complete tasks, not correlated workpieces.",
            "Wall time includes rendering, logging, model startup and coordination; channel timings are reported separately.",
        ],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        order = MODES[seed_index % 3 :] + MODES[: seed_index % 3]
        for mode in order:
            directory = args.output / f"seed-{seed}-{mode}"
            row = run_trial(args, seed, mode, directory)
            rows.append(row)
            # Keep every full trace; gzip is lossless and does not select frames.
            for filename in ("events.jsonl", "replay.json"):
                path = directory / filename
                if path.exists():
                    with (
                        path.open("rb") as src,
                        gzip.open(str(path) + ".gz", "wb", compresslevel=3) as dst,
                    ):
                        while chunk := src.read(1024 * 1024):
                            dst.write(chunk)
                    path.unlink()
            result = {
                **protocol,
                "runs": rows,
                "aggregate": aggregate(rows),
                "paired_checks": paired_checks(rows),
            }
            temporary = args.output / "results.json.tmp"
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(args.output / "results.json")
            print(
                f"seed={seed} mode={mode}: {row['status']} {row['correct_routes']}/{args.orders} correct, {row['completed']} delivered, {row['ticks']} ticks",
                flush=True,
            )
    print(json.dumps(aggregate(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
