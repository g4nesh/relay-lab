"""Predeclared, paired representation comparison; retain every attempted run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .runtime import ClothConfig, run_cloth, write_json

MODES = ("embeddings", "structured_text", "hybrid")


def validate_evaluation_seeds(seeds):
    # Training collection reserves 10000+, development reserves 0..99.
    # Fail closed rather than silently labeling development seeds as held out.
    if any(type(seed) is not int or not 100 <= seed < 10000 for seed in seeds):
        raise ValueError(
            "Evaluation seeds must be in [100, 9999]; development and training are reserved"
        )


def summarize(episodes):
    result = []
    for mode in MODES:
        rows = [r for r in episodes if r["mode"] == mode]
        alignments = [
            max(r["metrics"]["hem_alignment_m"].values())
            for r in rows
            if "hem_alignment_m" in r.get("metrics", {})
        ]
        result.append(
            {
                "mode": mode,
                "runs": len(rows),
                "successes": sum(r["success"] for r in rows),
                "success_rate": sum(r["success"] for r in rows) / len(rows) if rows else 0.0,
                "mean_alignment_m": float(np.mean(alignments)) if alignments else None,
                "mean_bytes": float(
                    np.mean([r.get("communication", {}).get("packet_bytes", 0) for r in rows])
                )
                if rows
                else 0.0,
                "mean_ticks": float(np.mean([r["ticks"] for r in rows])) if rows else 0.0,
                "failed_runs": sum(r["status"] == "failed" for r in rows),
            }
        )
    return result


def paired_checks(episodes):
    groups = {}
    for row in episodes:
        groups.setdefault((row["seed"], row["scenario"]), []).append(row)
    checks = []
    for (seed, scenario), rows in sorted(groups.items()):
        hashes = [r.get("initial_physics_sha256") for r in rows]
        rgb = [r.get("initial_rgb_sha256") for r in rows]
        model = [{a: m["checkpoint_sha256"] for a, m in r.get("models", {}).items()} for r in rows]
        codecs = [
            r.get("communication", {}).get("model", {}).get("checkpoint_sha256") for r in rows
        ]
        checks.append(
            {
                "seed": seed,
                "scenario": scenario,
                "runs": len(rows),
                "complete": len(rows) == len(MODES) and {r["mode"] for r in rows} == set(MODES),
                "same_initial_physics": bool(all(hashes)) and len(set(hashes)) == 1,
                "same_initial_rgb": bool(all(rgb)) and all(v == rgb[0] for v in rgb),
                "same_motor_models": bool(all(model)) and all(v == model[0] for v in model),
                "same_codec": bool(all(codecs)) and len(set(codecs)) == 1,
            }
        )
    return checks


def benchmark(
    *,
    output,
    seeds=(101, 102, 103, 104),
    scenarios=("clear", "obstacles", "delayed"),
    coordination="memory",
    planner="rules",
    max_ticks=1800,
    video=True,
):
    if (
        not seeds
        or not scenarios
        or len(set(seeds)) != len(seeds)
        or len(set(scenarios)) != len(scenarios)
    ):
        raise ValueError("Benchmark requires unique seeds and scenarios")
    validate_evaluation_seeds(seeds)
    for seed in seeds:
        for scenario in scenarios:
            ClothConfig(
                seed=seed,
                scenario=scenario,
                coordination=coordination,
                planner=planner,
                max_ticks=max_ticks,
            )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "relay.cloth.experiment.v1",
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "modes": list(MODES),
        "max_ticks": max_ticks,
        "coordination": coordination,
        "planner": planner,
        "planned_runs": len(seeds) * len(scenarios) * len(MODES),
        "training_excluded": "All experiment seeds are held out from model and codec fitting.",
        "seed_partitions": {
            "development": [0, 99],
            "evaluation": [100, 9999],
            "training": [10000, 4294967295],
        },
        "task": "Fold sleeves inward, then fold hem toward shoulders; independent geometric scoring.",
        "fairness": "Same seeds, initial cloth, sensor boundaries, models, scaffolding, obstacle schedule, message schedule and action budget.",
        "baseline": "Structured text carries full precision local measurements; hybrid adds exact safety fields to a learned feature bottleneck.",
        "interpretation": "Small task-specific, state-assisted simulation experiment; no foundation-VLA, photorealism or real-fabric transfer claim.",
    }
    write_json(output / "protocol.json", protocol)
    protocol_hash = hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()
    episodes, runs = [], []

    def report():
        value = {
            **protocol,
            "protocol_sha256": protocol_hash,
            "recorded_runs": len(episodes),
            "modes": summarize(episodes),
            "by_scenario": {
                s: summarize([r for r in episodes if r["scenario"] == s]) for s in scenarios
            },
            "paired_checks": paired_checks(episodes),
            "episodes": episodes,
            "conclusion": "Pending completion of all predeclared conditions.",
        }
        if len(episodes) == protocol["planned_runs"]:
            rates = {r["mode"]: r["success_rate"] for r in value["modes"]}
            best = max(rates.values())
            leaders = [m for m, rate in rates.items() if rate == best]
            value["conclusion"] = (
                "No mode completed the task; control/physics limitations prevent a meaningful success winner."
                if best == 0
                else f"Observed highest completion: {', '.join(leaders)} ({best:.0%}). Small paired sample; not a general superiority claim."
            )
            if not all(
                all(v for k, v in check.items() if k.startswith("same_") or k == "complete")
                for check in value["paired_checks"]
            ):
                value["conclusion"] = (
                    "Pairing validation failed; these attempts cannot establish a valid comparative winner."
                )
        write_json(output / "comparison.json", value)
        write_json(
            output / "manifest.json",
            {
                "title": "Two arms. One changing task.",
                "description": "Measured MuJoCo cloth folding with partial observations, moving obstacles and delayed communication.",
                "runs": runs,
                "comparison_url": "/runs/comparison.json",
            },
        )

    report()
    for i, seed in enumerate(seeds):
        for j, scenario in enumerate(scenarios):
            rotation = (i + j) % len(MODES)
            for mode in MODES[rotation:] + MODES[:rotation]:
                ident = f"{scenario}-{seed}-{mode}"
                config = ClothConfig(
                    seed=seed,
                    scenario=scenario,
                    mode=mode,
                    coordination=coordination,
                    planner=planner,
                    max_ticks=max_ticks,
                    video=video,
                )
                try:
                    row = run_cloth(config, output / ident)
                except Exception as exc:  # noqa: BLE001 - artifact/setup failures remain in denominator
                    row = {
                        "config": vars(config),
                        "mode": mode,
                        "seed": seed,
                        "scenario": scenario,
                        "status": "failed",
                        "success": False,
                        "ticks": 0,
                        "metrics": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                episodes.append(row)
                runs.append(
                    {
                        "id": ident,
                        "label": f"{mode.replace('_', ' ')} · {scenario} · seed {seed}",
                        "mode": mode,
                        "scenario": scenario,
                        "seed": seed,
                        "summary_url": f"/runs/{ident}/summary.json",
                        "trace_url": f"/runs/{ident}/trace.json",
                        "video_url": f"/runs/{ident}/overview.mp4" if video else None,
                        "poster_url": f"/runs/{ident}/poster.png" if video else None,
                        "left_video_url": f"/runs/{ident}/left.mp4" if video else None,
                        "right_video_url": f"/runs/{ident}/right.mp4" if video else None,
                    }
                )
                report()
                print(
                    f"{len(episodes)}/{protocol['planned_runs']} {ident}: {row['status']} "
                    f"({row['ticks']} ticks) {row['error'] or ''}",
                    flush=True,
                )
    return json.loads((output / "comparison.json").read_text())
