"""Collect original cloth-training observations, or replay recorded actions exactly.

This is an explicitly geometric teaching controller, never a learned-policy
benchmark. Replay re-executes MuJoCo physics and obtains grasp identity from
actual local observations; it does not infer identity from the intended phase.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from relay_lab.cloth.communication import FEATURES, SCHEMA, observation_features
from relay_lab.cloth.world import ClothWorld


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect(output, *, seed=10004, phase_ticks=45, settle_ticks=100, replay=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(replay) if replay else None
    source_summary = json.loads((source / "summary.json").read_text()) if source else None
    if source_summary:
        seed = source_summary["seed"]
    world = ClothWorld(seed=seed, disturbances=True, fixture=True)
    started = perf_counter()
    rows, stages = [], []
    try:
        Image.fromarray(world.render(width=1280, height=800)).save(output / "initial.png")
        with gzip.open(output / "physics.jsonl.gz", "wt") as log:

            def execute(phase, actions, pause=False):
                observations = {a: world.observe(a) for a in ("left", "right")}
                if world.tick % 4 == 0:
                    rows.extend(
                        {
                            "tick": world.tick,
                            "agent": a,
                            "phase": phase,
                            "observation": observation,
                            "features": observation_features(observation),
                        }
                        for a, observation in observations.items()
                    )
                snapshot = world.step(actions)
                log.write(
                    json.dumps(
                        {
                            "tick": world.tick,
                            "phase": phase,
                            "pause": pause,
                            "actions": actions,
                            "metrics": snapshot["metrics"],
                            "robots": snapshot["robots"],
                            "fixture_active": world.fixture_active,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )

            if source:
                previous_phase = None
                with gzip.open(source / "physics.jsonl.gz", "rt") as source_log:
                    for line in source_log:
                        step = json.loads(line)
                        if step["tick"] != world.tick + 1:
                            raise ValueError("reference actions are not consecutive")
                        if world.fixture_active and not step["fixture_active"]:
                            world.release_fixture()
                        if previous_phase is not None and previous_phase != step["phase"]:
                            stages.append(
                                {
                                    "phase": previous_phase,
                                    "tick": world.tick,
                                    "metrics": world.metrics(),
                                }
                            )
                            print(
                                previous_phase,
                                world.tick,
                                world.metrics()["physics_valid"],
                                flush=True,
                            )
                        execute(step["phase"], step["actions"], step.get("pause", False))
                        previous_phase = step["phase"]
                stages.append(
                    {"phase": previous_phase, "tick": world.tick, "metrics": world.metrics()}
                )
            else:
                for phase in world.PHASES:
                    if phase == "settle":
                        world.release_fixture()
                    active_ticks = 0
                    while active_ticks < (settle_ticks if phase == "settle" else phase_ticks):
                        observations = {a: world.observe(a) for a in ("left", "right")}
                        # Deliberately privileged teaching coordinator; comparison
                        # policies use only their own observations + delivered messages.
                        pause = any(o["obstacle"] for o in observations.values())
                        if pause:
                            actions = {
                                a: {"target": o["ee_position"], "grip": o["grasped"]}
                                for a, o in observations.items()
                            }
                        else:
                            actions = {a: world.reference_target(a, phase) for a in observations}
                            active_ticks += 1
                        execute(phase, actions, pause)
                    stages.append({"phase": phase, "tick": world.tick, "metrics": world.metrics()})
                    print(phase, world.tick, world.metrics(), flush=True)
        snapshot = world.snapshot()
        parity = None
        if source:
            previous = np.asarray(json.loads((source / "snapshot.json").read_text())["vertices"])
            current = np.asarray(snapshot["vertices"])
            parity = bool(np.array_equal(previous, current))
            if not parity:
                raise RuntimeError(
                    f"action replay changed final physical vertices: max error {np.max(np.abs(previous - current))}"
                )
        summary = {
            "kind": "original_reference_training_action_replay"
            if source
            else "original_reference_training_rollout",
            "seed": seed,
            "split": "training",
            "learned_policy": False,
            "fixture": True,
            "disturbances": True,
            "controller": "geometric teacher with coordinated workspace-interlock holds",
            "steps": world.tick,
            "wall_seconds": perf_counter() - started,
            "metrics": world.metrics(),
            "physics_config": snapshot["physics_config"],
            "schema": SCHEMA,
            "features": list(FEATURES),
            "samples": len(rows),
            "source_action_sha256": hashlib.sha256(
                (source / "physics.jsonl.gz").read_bytes()
            ).hexdigest()
            if source
            else None,
            "source_run": str(source) if source else None,
            "final_vertices_bitwise_equal": parity,
            "stages": stages,
        }
        write_json(output / "summary.json", summary)
        write_json(
            output / "communication-features.json",
            {"seed": seed, "split": "training", "schema": SCHEMA, "rows": rows},
        )
        write_json(output / "snapshot.json", snapshot)
        Image.fromarray(world.render(width=1280, height=800)).save(output / "final.png")
        Image.fromarray(world.render("top", width=960, height=720)).save(output / "top.png")
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in ("seed", "steps", "metrics", "samples", "final_vertices_bitwise_equal")
                }
            )
        )
        return summary
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=10004)
    parser.add_argument("--phase-ticks", type=int, default=45)
    parser.add_argument("--settle-ticks", type=int, default=100)
    parser.add_argument("--replay", help="Original reference run whose exact actions to re-execute")
    args = parser.parse_args()
    if args.phase_ticks < 1 or args.settle_ticks < 15 or not 0 <= args.seed < 2**32:
        parser.error("Use positive phase ticks, at least 15 settling ticks, and a uint32 seed")
    collect(
        args.output,
        seed=args.seed,
        phase_ticks=args.phase_ticks,
        settle_ticks=args.settle_ticks,
        replay=args.replay,
    )


if __name__ == "__main__":
    main()
