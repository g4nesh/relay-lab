"""Reproducible training/evaluation for the original small Relay VLA.

All oracle coordinates below are training labels or evaluation metrics. The
runtime policy imports none of this module. Run with ``python -m
relay_lab.train_relay_vla collect|train|evaluate``; no external dataset is used.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .relay_vla import (
    DEFAULT_CHECKPOINT,
    WORDS,
    RelayVLABackend,
    RelayVLAPolicy,
    language_features,
    make_network,
)
from .types import AGENTS, COLORS, MAX_DELTA


def collect(path: Path, scenes=400, seed=2000):
    import mujoco

    from .environment import RelayEnv

    rng = np.random.default_rng(seed)
    images, masks, coords, roles = [], [], [], []
    started = perf_counter()
    with RelayEnv(seed=seed) as env:
        # Workcell positions are measured training labels, never runtime inputs.
        targets = [
            [env.TARGETS[name] for name in ("relay", "slot_0", "slot_1", "slot_2")]
            + [env.HOME[agent]]
            for agent in AGENTS
        ]
        for index in range(scenes):
            role = index % 2
            agent = AGENTS[role]
            env.reset(seed + index)
            for color in COLORS:
                # Include source, relay, and packing regions with arbitrary
                # colour placement and jitter, beyond the evaluation reset range.
                if role == 0:
                    x, y = rng.uniform(-0.65, -0.30), rng.uniform(-0.28, 0.28)
                    if rng.random() < 0.2:
                        x, y = rng.uniform(-0.025, 0.025, 2)
                else:
                    x, y = rng.uniform(0.30, 0.65), rng.uniform(-0.28, 0.28)
                    if rng.random() < 0.4:
                        x, y = rng.uniform(-0.025, 0.025, 2)
                env._set_object_position(color, (x, y, 0.03))
            for robot in AGENTS:
                low, high = np.asarray(env.BOUNDS[robot])
                env.data.mocap_pos[env._mocap[robot]] = rng.uniform(low, high)
            mujoco.mj_forward(env.model, env.data)
            rgb = env.render(camera=agent, width=128, height=128)
            renderer = env._renderers[128, 128]
            renderer.enable_segmentation_rendering()
            segmentation = env.render(camera=agent, width=128, height=128)
            renderer.disable_segmentation_rendering()
            labels = np.zeros((128, 128), dtype=np.uint8)
            for c, color in enumerate(COLORS, start=1):
                labels[
                    (segmentation[:, :, 0] == env._object_geoms[color])
                    & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                ] = c
            images.append(rgb)
            masks.append(labels)
            coords.append([env._object_position(c)[:2] for c in COLORS])
            roles.append(role)
            if (index + 1) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "collect",
                            "scenes": index + 1,
                            "seconds": round(perf_counter() - started, 2),
                        }
                    ),
                    flush=True,
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.stack(images),
        masks=np.stack(masks),
        coords=np.array(coords),
        roles=np.array(roles),
        targets=np.array(targets),
        seed=seed,
    )
    print(
        json.dumps(
            {
                "dataset": str(path),
                "scenes": scenes,
                "seed": seed,
                "source": "original Relay Lab synthetic renders",
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )


def language_examples():
    examples = []
    for color_index, color in enumerate(COLORS):
        for text in (
            f"Pick the {color} cube from the source.",
            f"Collect {color} from the relay pad.",
            f"Pick {color}.",
            f"Pick up the {color} parcel.",
        ):
            examples.append((text, color_index, 0))
        examples += [
            (f"Place {color} on the shared relay pad.", 3, 1),
            (f"Place the {color} cube at relay.", 3, 1),
        ]
        for slot in range(3):
            examples += [
                (f"Place {color} in packing slot {slot + 1}.", 4 + slot, 1),
                (f"Place the {color} parcel in slot {slot + 1}.", 4 + slot, 1),
            ]
    for text in (
        "Keep the relay clear while the packer collects.",
        "Wait for a new color request from the packer.",
        "Packing complete; return home.",
        "Wait at home.",
        "Wait for the picker to clear the relay.",
        *(f"Wait for {c} at the relay pad." for c in COLORS),
        *(f"Wait for the picker to verify {c} is ready and clear." for c in COLORS),
    ):
        examples.append((text, 7, 2))
    return examples


def teacher_batch(rng, count):
    """Generate original expert supervision across perturbed robot states.

    This reactive teacher is used during training only. Inference executes the
    neural network's motor outputs without a waypoint or gripper override.
    """
    role = rng.integers(0, 2, count)
    kind = rng.integers(0, 3, count)
    goal = np.column_stack(
        [rng.uniform(-0.65, 0.65, count), rng.uniform(-0.28, 0.28, count), np.full(count, 0.035)]
    )
    goal[:, 0] = np.where(role == 0, -np.abs(goal[:, 0]), np.abs(goal[:, 0]))
    wait = kind == 2
    goal[wait, 0] = np.where(role[wait] == 0, -0.54, 0.48)
    goal[wait, 1:] = [0, 0.30]
    position = np.column_stack(
        [
            rng.uniform(-0.70, 0.70, count),
            rng.uniform(-0.35, 0.35, count),
            rng.uniform(0.035, 0.40, count),
        ]
    )
    position[:, 0] = np.where(role == 0, -np.abs(position[:, 0]), np.abs(position[:, 0]))
    # Oversample approaching/arrived states and the travel altitude. These are
    # the difficult grasp/release transitions omitted by uniform sampling.
    close = rng.random(count) < 0.50
    position[close, :2] = goal[close, :2] + rng.normal(0, 0.035, (close.sum(), 2))
    near = rng.random(count) < 0.35
    position[near] = goal[near] + rng.normal(0, 0.018, (near.sum(), 3))
    travel = rng.random(count) < 0.20
    position[travel, 2] = rng.normal(0.30, 0.01, travel.sum())
    position[:, 2] = np.maximum(0.035, position[:, 2])
    diff = goal - position
    distance_xy = np.linalg.norm(diff[:, :2], axis=1)
    # Smooth expert field allows small corrections throughout descent instead
    # of a discontinuous lift/descend boundary that traps a cloned controller.
    horizontal_gate = np.clip((position[:, 2] - 0.18) / 0.09, 0, 1)
    horizontal_gate = np.maximum(horizontal_gate, (distance_xy < 0.06).astype(float))
    travel_weight = np.clip((distance_xy - 0.018) / 0.055, 0, 1)
    desired_z = goal[:, 2] * (1 - travel_weight) + 0.30 * travel_weight
    delta = np.column_stack(
        [diff[:, 0] * horizontal_gate, diff[:, 1] * horizontal_gate, desired_z - position[:, 2]]
    )
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, MAX_DELTA / np.maximum(length, 1e-8))[:, None]
    distance = np.linalg.norm(diff, axis=1)
    grip = np.ones(count, dtype=np.int64)  # maintain = class 1
    grip[kind == 0] = 2  # open while approaching
    grip[(kind == 0) & (distance < 0.040)] = 0  # close
    grip[(kind == 1) & (distance < 0.035)] = 2  # release
    return (
        position.astype("float32"),
        goal.astype("float32"),
        np.eye(3, dtype="float32")[kind],
        (delta / MAX_DELTA).astype("float32"),
        grip,
    )


def train(path: Path, output: Path, seed=17, pixel_steps=1500, control_steps=10000, device="cpu"):
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.load(path)
    network = make_network().to(device)
    started = perf_counter()
    # Balanced pixel supervision comes from simulator segmentation only at
    # training time; evaluation sees RGB and invokes the learned classifier.
    flat_rgb = data["images"].reshape(-1, 3).astype("float32") / 255
    flat_labels = data["masks"].ravel()
    pools = [np.flatnonzero(flat_labels == c) for c in range(4)]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("Every parcel colour must occur in training images")
    pixel_optimizer = torch.optim.Adam(network.pixels.parameters(), lr=0.008)
    for step in range(pixel_steps):
        indices = np.concatenate([rng.choice(pool, 1024) for pool in pools])
        x = torch.from_numpy(flat_rgb[indices]).to(device)[:, :, None, None]
        y = torch.from_numpy(flat_labels[indices].astype("int64")).to(device)
        features = x[:, :, 0, 0]
        for layer in network.pixels:
            if isinstance(layer, torch.nn.Conv2d):
                features = F.linear(features, layer.weight[:, :, 0, 0], layer.bias)
            else:
                features = layer(features)
        loss = F.cross_entropy(features, y)
        pixel_optimizer.zero_grad()
        loss.backward()
        pixel_optimizer.step()
        if (step + 1) % 500 == 0:
            print(
                json.dumps(
                    {
                        "stage": "pixels",
                        "step": step + 1,
                        "loss": loss.item(),
                        "seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    # Fit camera calibration using the classifier's own estimated centroids.
    # This avoids an inference-time perfect-mask assumption.
    network.eval()
    rows = [[], []]
    labels = [[], []]
    with torch.no_grad():
        for start in range(0, len(data["images"]), 16):
            images = (
                torch.from_numpy(data["images"][start : start + 16])
                .to(device)
                .permute(0, 3, 1, 2)
                .float()
                / 255
            )
            probabilities = network.pixels(images).softmax(dim=1)[:, 1:].pow(12).cpu().numpy()
            for offset, probs in enumerate(probabilities):
                i = start + offset
                role = int(data["roles"][i])
                for color in range(3):
                    if (data["masks"][i] == color + 1).sum() < 5:
                        continue
                    p = probs[color]
                    mass = p.sum()
                    u = (p * np.linspace(-1, 1, 128)[None, :]).sum() / mass
                    v = (p * np.linspace(-1, 1, 128)[:, None]).sum() / mass
                    rows[role].append([1, u, v, u * v, u * u, v * v])
                    labels[role].append(data["coords"][i, color])
        calibration_errors = []
        for role in range(2):
            matrix = np.asarray(rows[role])
            expected = np.asarray(labels[role])
            fitted = np.linalg.lstsq(matrix, expected, rcond=None)[0]
            network.calibration[role].copy_(
                torch.tensor(fitted, device=device, dtype=torch.float32)
            )
            calibration_errors.append(
                float(np.linalg.norm(matrix @ fitted - expected, axis=1).mean())
            )
        network.targets.copy_(torch.tensor(data["targets"], device=device, dtype=torch.float32))
    print(
        json.dumps({"stage": "calibration", "mean_xy_error_metres": calibration_errors}), flush=True
    )
    examples = language_examples()
    words = torch.tensor(np.stack([language_features(x[0]) for x in examples]), device=device)
    goals = torch.tensor([x[1] for x in examples], device=device)
    kinds = torch.tensor([x[2] for x in examples], device=device)
    optimizer = torch.optim.Adam(
        [*network.language.parameters(), *network.kind.parameters()], lr=0.05
    )
    for _ in range(1000):
        loss = F.cross_entropy(network.language(words), goals) + F.cross_entropy(
            network.kind(words), kinds
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        language_accuracy = float((network.language(words).argmax(-1) == goals).float().mean())
    optimizer = torch.optim.AdamW(network.control.parameters(), lr=0.001, weight_decay=1e-6)
    for step in range(control_steps):
        position, goal, kind, delta, grip = teacher_batch(rng, 1024)
        position, goal, kind, delta, grip = [
            torch.from_numpy(x).to(device) for x in (position, goal, kind, delta, grip)
        ]
        prediction = network.action(position, goal, kind)
        motion = F.mse_loss(prediction[:, :3], delta)
        grasp = F.cross_entropy(prediction[:, 3:], grip)
        loss = motion + grasp * 0.15
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 1000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "control",
                        "step": step + 1,
                        "motion_loss": motion.item(),
                        "grip_loss": grasp.item(),
                        "seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    training = {
        "seed": seed,
        "dataset": str(path.resolve()),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scene_seed_start": int(data["seed"]),
        "scenes": len(data["images"]),
        "pixel_steps": pixel_steps,
        "control_steps": control_steps,
        "teacher": "original synthetic RGB segmentation/calibration and reactive expert action labels; no demonstration dataset downloaded",
        "language_training_accuracy": language_accuracy,
        "calibration_mean_error_metres": calibration_errors,
        "training_seconds": perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "relay-vla-v1",
            "words": list(WORDS),
            "state_dict": {k: v.cpu() for k, v in network.state_dict().items()},
            "training": training,
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps(training, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(output), **training}), flush=True)


def evaluate(checkpoint: Path, seeds, max_steps=600, output: Path | None = None):
    import torch

    from .communication import LocalMessageBus
    from .environment import RelayEnv
    from .policies import RuleOrchestrator

    torch.set_num_threads(2)
    backend = RelayVLABackend(checkpoint)
    results = []
    for seed in seeds:
        policies = {a: RelayVLAPolicy(a, backend) for a in AGENTS}
        planners = {a: RuleOrchestrator(a) for a in AGENTS}
        bus = LocalMessageBus()
        records = []
        with RelayEnv(seed=seed) as env:
            for _ in range(max_steps):
                observations = {a: env.observe(a, rgb=True) for a in AGENTS}
                decisions = {
                    a: planners[a].decide(observations[a], bus.receive(a, env.tick)) for a in AGENTS
                }
                for a in AGENTS:
                    if decisions[a].message:
                        bus.send(a, decisions[a].message, env.tick)
                actions = {
                    a: policies[a].act(observations[a], decisions[a].instruction) for a in AGENTS
                }
                events = env.step(actions)
                records.append(
                    {
                        "tick": env.tick,
                        "state": env.state(),
                        "actions": {a: asdict(actions[a]) for a in AGENTS},
                        "instructions": {a: asdict(decisions[a].instruction) for a in AGENTS},
                        "predictions": {a: policies[a].last_prediction for a in AGENTS},
                        "events": events,
                    }
                )
                if env.success:
                    break
            result = {
                "seed": seed,
                "success": env.success,
                "steps": env.tick,
                "packed": env.packed,
                "order": env.order,
            }
            print(json.dumps(result), flush=True)
            results.append(result)
        if output:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"seed-{seed}.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in records)
            )
    if output:
        (output / "summary.json").write_text(
            json.dumps({"results": results, "model": backend.model_info}, indent=2) + "\n"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--output", type=Path, default=Path("datasets/relay-vla/original.npz"))
    p.add_argument("--scenes", type=int, default=400)
    p.add_argument("--seed", type=int, default=2000)
    p = sub.add_parser("train")
    p.add_argument("--dataset", type=Path, default=Path("datasets/relay-vla/original.npz"))
    p.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--pixel-steps", type=int, default=1500)
    p.add_argument("--control-steps", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 101])
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--output", type=Path, default=Path("runs/relay-vla-evaluation"))
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output, args.scenes, args.seed)
    elif args.command == "train":
        train(
            args.dataset, args.output, args.seed, args.pixel_steps, args.control_steps, args.device
        )
    else:
        evaluate(args.checkpoint, args.seeds, args.max_steps, args.output)


if __name__ == "__main__":
    main()
