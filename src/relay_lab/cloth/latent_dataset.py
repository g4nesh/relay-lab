"""Original, paired one-step cloth coordination data from real MuJoCo states.

This is a small action-imitation diagnostic, not a successful-fold corpus. A
privileged geometric teacher supplies labels. The actor inputs contain only one
arm's local RGB, local measurements and phase instruction. At each physical
snapshot we render clear/left-hazard/right-hazard counterfactuals without stepping
the simulator. A hazard on either side changes BOTH teacher labels to hold.

The private-camera restriction explicitly removes a peer-side obstacle while
rendering the other arm's camera. This also removes indirect visual leakage such
as shadows; it is a controlled observation mask, not a claim about raw cameras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .world import ClothWorld

SCHEMA = "relay.cloth.latent-dataset.v1"
AGENTS = ("left", "right")
DEFAULT_EPISODES = (("train", 10010), ("train", 10011), ("dev", 11), ("test", 104))
MIDPHASES = ("lift_sleeve", "carry_sleeve", "lift_hem", "carry_hem")
COUNTERFACTUALS = (None, "left", "right")
CAMERA_RESTRICTION = (
    "Own narrow overhead camera; peer-side obstacle geometry is explicitly removed "
    "during rendering, including its shadows. Own measurements are unchanged. "
    "This controlled visibility mask is not a raw-camera occlusion claim."
)


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_episodes(episodes):
    episodes = tuple(episodes)
    if not episodes:
        raise ValueError("at least one episode is required")
    seeds = set()
    for split, seed in episodes:
        if split not in ("train", "dev", "test"):
            raise ValueError("split must be train, dev or test")
        if type(seed) is not int or seed < 0 or seed >= 2**32 or seed in seeds:
            raise ValueError("episode seeds must be unique uint32 values")
        if (split == "train" and seed < 10000) or (split == "dev" and seed >= 100):
            raise ValueError("training seeds must be >=10000; development seeds must be <100")
        if split == "test" and not 100 <= seed < 10000:
            raise ValueError("test seeds must be in [100, 10000)")
        seeds.add(seed)
    return episodes


def _physics_signature(world):
    """Independent integration state, excluding recomputed forward dynamics."""
    return canonical_hash(
        {
            "qpos": array_hash(world.data.qpos),
            "qvel": array_hash(world.data.qvel),
            "qacc_warmstart": array_hash(world.data.qacc_warmstart),
            "ctrl": array_hash(world.data.ctrl),
            "eq_active": array_hash(world.data.eq_active),
            "time": float(world.data.time),
            "tick": world.tick,
            "holding": world.holding,
            "targets": {a: world._targets[a].tolist() for a in AGENTS},
        }
    )


@contextmanager
def frozen_obstacle(world, side):
    """Move only exogenous hazard geometry; restore it even on renderer failure."""
    if side not in COUNTERFACTUALS:
        raise ValueError("hazard side must be left, right or None")
    position = world.data.mocap_pos[world._obstacle_mocap].copy()
    active, previous_side = world._obstacle_active, world._obstacle_side
    signature = _physics_signature(world)
    try:
        world._obstacle_active = side is not None
        world._obstacle_side = side
        world.data.mocap_pos[world._obstacle_mocap] = (
            [(-1 if side == "left" else 1) * 0.28, -0.04, 0.18]
            if side is not None
            else [2.3, 0, 0.18]
        )
        mujoco.mj_forward(world.model, world.data)
        yield
    finally:
        world.data.mocap_pos[world._obstacle_mocap] = position
        world._obstacle_active, world._obstacle_side = active, previous_side
        mujoco.mj_forward(world.model, world.data)
        if _physics_signature(world) != signature:
            raise RuntimeError("counterfactual rendering changed the physical integration state")


def private_rgb(world, agent, size):
    """Explicitly mask hidden hazard geometry, preserving the requested own view."""
    if world._obstacle_active and world._obstacle_side != agent:
        with frozen_obstacle(world, None):
            return world.render(agent, width=size, height=size)
    return world.render(agent, width=size, height=size)


def capture_counterfactuals(world, *, split, seed, phase, snapshot_index, image_size=128):
    """Return six paired rows/images at exactly the same arm and cloth state."""
    if phase not in world.PHASES:
        raise ValueError("unsupported phase")
    if type(image_size) is not int or not 16 <= image_size <= 1024:
        raise ValueError("image size must be an integer between 16 and 1024")
    episode_id = f"{split}-{seed}"
    snapshot_id = f"{episode_id}-s{snapshot_index:03d}"
    state_sha256 = _physics_signature(world)
    nominal = {a: world.reference_target(a, phase) for a in AGENTS}
    rows, images, clear = [], [], {}
    for side in COUNTERFACTUALS:
        counterfactual = "clear" if side is None else f"{side}_obstacle"
        pair_id = f"{snapshot_id}-{counterfactual}"
        with frozen_obstacle(world, side):
            for agent in AGENTS:
                observation = world.observe(agent)
                rgb = private_rgb(world, agent, image_size)
                if rgb.shape != (image_size, image_size, 3) or rgb.dtype != np.uint8:
                    raise ValueError("renderer must return uint8 HxWx3 RGB")
                rgb_sha256 = array_hash(rgb)
                observation_sha256 = canonical_hash(observation)
                if side is None:
                    clear[agent] = (rgb_sha256, observation_sha256)
                elif agent != side and clear[agent] != (rgb_sha256, observation_sha256):
                    raise RuntimeError("hidden peer hazard leaked into a local actor input")
                action = (
                    {"target": observation["ee_position"], "grip": observation["grasped"]}
                    if side is not None
                    else nominal[agent]
                )
                rows.append(
                    {
                        "id": f"{pair_id}-{agent}",
                        "rgb_index": len(images),
                        "rgb_sha256": rgb_sha256,
                        "observation_sha256": observation_sha256,
                        "episode_id": episode_id,
                        "split": split,
                        "seed": seed,
                        "pair_id": pair_id,
                        "snapshot_id": snapshot_id,
                        "physical_state_sha256": state_sha256,
                        "counterfactual": counterfactual,
                        "phase": phase,
                        "instruction": "Fold the T-shirt. " + phase.replace("_", " ") + ".",
                        "agent": agent,
                        "observation": observation,
                        "action": action,
                        "labels": {
                            "local_obstacle": side == agent,
                            "peer_obstacle": side is not None and side != agent,
                            "coordinated_hold": side is not None,
                            "nominal_action": nominal[agent],
                            "privileged_teacher": True,
                        },
                    }
                )
                images.append(rgb)
    if _physics_signature(world) != state_sha256:
        raise RuntimeError("capturing counterfactuals advanced the physical state")
    return rows, images


def _collect_episode(arguments):
    output, episode, image_size, phase_ticks = arguments
    return collect(
        output, episodes=(episode,), image_size=image_size, phase_ticks=phase_ticks, workers=1
    )


def collect(output, *, episodes=DEFAULT_EPISODES, image_size=128, phase_ticks=45, workers=1):
    """Collect disjoint whole episodes using only this project's original teacher."""
    episodes = validate_episodes(episodes)
    if type(phase_ticks) is not int or phase_ticks < 15:
        raise ValueError("phase_ticks must be an integer of at least 15")
    if type(image_size) is not int or not 16 <= image_size <= 1024:
        raise ValueError("image size must be an integer between 16 and 1024")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    rows, images, reports = [], [], []
    if workers > 1:
        arguments = [
            (output / "episodes" / f"{split}-{seed}", (split, seed), image_size, phase_ticks)
            for split, seed in episodes
        ]
        with ProcessPoolExecutor(max_workers=min(workers, len(episodes))) as pool:
            # executor.map preserves the specified episode order regardless of
            # completion time. Each worker owns a separate world and output shard.
            summaries = list(pool.map(_collect_episode, arguments))
        for (directory, _, _, _), summary in zip(arguments, summaries, strict=True):
            shard_rows = json.loads((directory / "rows.json").read_text())
            with np.load(directory / "rgb.npz", allow_pickle=False) as shard:
                shard_images = shard["rgb"]
            for row in shard_rows:
                row["rgb_index"] += len(images)
            rows.extend(shard_rows)
            images.extend(shard_images)
            reports.extend(summary["episodes"])
        return _save(output, rows, images, reports, started, workers)
    for split, seed in episodes:
        with ClothWorld(seed=seed, disturbances=False, fixture=True) as world:
            initial_state = _physics_signature(world)
            initial_vertices = array_hash(world.vertices)
            snapshot_index = 0
            phases = []
            for phase in world.PHASES:
                if phase == "settle":
                    world.release_fixture()
                ticks = 160 if phase == "settle" else 90 if phase == "fold_hem" else phase_ticks
                for active_tick in range(ticks):
                    if active_tick == 0 or (phase in MIDPHASES and active_tick == 12):
                        new_rows, new_images = capture_counterfactuals(
                            world,
                            split=split,
                            seed=seed,
                            phase=phase,
                            snapshot_index=snapshot_index,
                            image_size=image_size,
                        )
                        for row in new_rows:
                            row["rgb_index"] += len(images)
                        rows.extend(new_rows)
                        images.extend(new_images)
                        snapshot_index += 1
                    # Counterfactuals are diagnostics on the clear trajectory.
                    # They do not advance physics and do not become training
                    # demonstrations of successful obstacle avoidance.
                    world.step({a: world.reference_target(a, phase) for a in AGENTS})
                phases.append({"phase": phase, "end_tick": world.tick})
            reports.append(
                {
                    "episode_id": f"{split}-{seed}",
                    "split": split,
                    "seed": seed,
                    "initial_state_sha256": initial_state,
                    "initial_vertices_sha256": initial_vertices,
                    "snapshots": snapshot_index,
                    "ticks": world.tick,
                    "phases": phases,
                    "final_metrics": world.metrics(),
                    "physics_config": world.PHYSICS_CONFIG,
                }
            )
            print(
                json.dumps({"episode": f"{split}-{seed}", "snapshots": snapshot_index}), flush=True
            )
    return _save(output, rows, images, reports, started, workers)


def _save(output, rows, images, reports, started, workers):
    rgb = np.stack(images)
    np.savez_compressed(output / "rgb.npz", rgb=rgb)
    (output / "rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    metadata = {
        "schema": SCHEMA,
        "kind": "original paired one-step action-imitation and peer-hazard diagnostic",
        "rgb_shape": list(rgb.shape),
        "rows": len(rows),
        "pairs": len(rows) // 2,
        "workers": workers,
        "merge_order": [episode["episode_id"] for episode in reports],
        "episodes": reports,
        "camera_restriction": CAMERA_RESTRICTION,
        "teacher": "Privileged geometric reference_target; either hazard makes both labels hold.",
        "actor_inputs": ["local RGB", "own observation", "phase instruction"],
        "label_only_fields": ["labels", "action", "counterfactual"],
        "split_policy": "Whole episodes; seeds train>=10000, dev<100, test in [100,10000).",
        "counterfactual_physics": "No stepping; qpos/qvel/qacc_warmstart/ctrl/constraints/time "
        "and controller targets verified unchanged.",
        "hidden_peer_input_parity": "Exact RGB and observation hashes match the clear case.",
        "limitations": [
            "A tiny diagnostic dataset, not a generalization benchmark or successful-fold corpus.",
            "Local measured state is available; vision-only control is not established.",
            "Counterfactual hazards change sensing and one-step teacher labels, not the trajectory.",
            "Privileged teacher labels must never be passed as policy inputs.",
            "Phase instructions are supplied; no language planning is learned here.",
        ],
        "files": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("rgb.npz", "rows.json")
        },
        "wall_seconds": perf_counter() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/cloth-latent-dataset-v1"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--phase-ticks", type=int, default=45)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    result = collect(
        args.output,
        image_size=args.image_size,
        phase_ticks=args.phase_ticks,
        workers=args.workers,
    )
    print(json.dumps({k: result[k] for k in ("rows", "pairs", "wall_seconds")}))


if __name__ == "__main__":
    main()
