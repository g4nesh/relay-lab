"""Train two original cloth controllers; all reference geometry stays here.

Local-state augmentation supplies coverage beyond a handful of rendered scenes.
This is a state-assisted imitation baseline, not foundation-VLA fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .policies import (
    ARCHITECTURES,
    DEFAULT_DIRECTORY,
    FORMAT,
    PHASES,
    make_network,
    observation_features,
)


def collect(output: Path, seeds: int = 12):
    import mujoco

    from .communication import FEATURES
    from .communication import observation_features as communication_features
    from .world import ClothWorld

    output.mkdir(parents=True, exist_ok=True)
    images, observations, communication = (
        {a: [] for a in ARCHITECTURES},
        {a: [] for a in ARCHITECTURES},
        [],
    )
    started = perf_counter()
    with ClothWorld(
        seed=10000, disturbances=True, obstacle_schedule=((6, 20, "left"), (28, 42, "right"))
    ) as world:
        for seed in range(10000, 10000 + seeds):
            world.reset(seed)
            for agent in ARCHITECTURES:
                obs = world.observe(agent, rgb=True)
                images[agent].append(obs.pop("rgb"))
                observations[agent].append(obs)
                communication.append(communication_features(obs))
            # Real, mechanically executed examples contain lifted/occluded cloth.
            for phase in PHASES:
                for t in range(3):
                    world.step({a: world.reference_target(a, phase) for a in ARCHITECTURES})
                    if t == 2:
                        for agent in ARCHITECTURES:
                            obs = world.observe(agent, rgb=True)
                            images[agent].append(obs.pop("rgb"))
                            observations[agent].append(obs)
                            communication.append(communication_features(obs))
            # Original randomized initial-contact states cover positive grasp
            # observations. Reposition only the training robot joints, then let
            # the real proximity gate and MuJoCo constraints establish contact.
            for contact_agents in (("left",), ("right",), ("left", "right")):
                world.reset(seed)
                actions = {}
                for agent in contact_agents:
                    target = np.asarray(world.observe(agent)["landmarks"]["hem"]) + [0, 0, 0.008]
                    world._targets[agent] = target.copy()
                    q = world.inverse_kinematics(agent, target)
                    world.data.qpos[world._joints[agent]] = q
                    start = 0 if agent == "left" else 6
                    world.data.ctrl[start : start + 6] = q
                    actions[agent] = {"target": target, "grip": True}
                mujoco.mj_forward(world.model, world.data)
                world.step(actions)
                for agent in ARCHITECTURES:
                    obs = world.observe(agent, rgb=True)
                    images[agent].append(obs.pop("rgb"))
                    observations[agent].append(obs)
                    communication.append(communication_features(obs))
            print(
                json.dumps(
                    {
                        "stage": "cloth_training_render",
                        "seed": seed,
                        "seconds": round(perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
            # Preserve completed collection if a later physical sample fails.
            np.save(
                output / "communication-training.npy",
                np.asarray(
                    [[row[name] for name in FEATURES] for row in communication], dtype=np.float32
                ),
            )
    for agent in ARCHITECTURES:
        np.savez_compressed(output / f"{agent}-rgb.npz", images=np.stack(images[agent]))
        (output / f"{agent}-observations.json").write_text(json.dumps(observations[agent]))
    np.save(
        output / "communication-training.npy",
        np.asarray([[row[name] for name in FEATURES] for row in communication], dtype=np.float32),
    )
    (output / "communication-training.json").write_text(json.dumps(communication, indent=2) + "\n")
    (output / "collection.json").write_text(
        json.dumps(
            {
                "training_seeds": list(range(10000, 10000 + seeds)),
                "stage": "original reference demonstration collection",
                "obstacle_schedule": [[6, 20, "left"], [28, 42, "right"]],
                "evaluation_seeds_used": False,
            },
            indent=2,
        )
    )


def synthetic_examples(agent: str, count: int, seed: int):
    """Original analytic demonstrations over locally measurable states.

    Teacher uses fixed workcell dimensions and local keypoint coordinates; no
    evaluation seed, outcome, or peer observation enters training.
    """
    rng = np.random.default_rng(seed)
    sign = -1 if agent == "left" else 1
    phases = rng.integers(0, len(PHASES), count)
    words = np.eye(len(PHASES), dtype=np.float32)[phases]
    states, actions = [], []
    for index, phase_id in enumerate(phases):
        phase = PHASES[phase_id]
        jitter = rng.uniform(-0.025, 0.025, 2)
        ee = np.array([sign * 0.3, 0, 0.16]) + rng.uniform(
            [-0.20, -0.35, -0.15], [0.20, 0.35, 0.25]
        )
        sleeve = np.array([sign * 0.44, 0.16, 0.008]) + [*jitter, 0]
        hem = np.array([sign * 0.24, -0.32, 0.008]) + [*jitter, 0]
        shoulder = np.array([sign * 0.24, 0.28, 0.008]) + [*jitter, 0]
        # Locally observed cloth can deform substantially before contact.
        sleeve += rng.normal(0, [0.07, 0.06, 0.02])
        hem += rng.normal(0, [0.035, 0.035, 0.02])
        shoulder += rng.normal(0, [0.05, 0.16, 0.025])
        grasped = bool(rng.random() < 0.5)
        clearance = float(rng.uniform(0.10, 0.40))
        obs = {
            "ee_position": ee,
            "grasped": grasped,
            "reference_origin": jitter,
            "landmarks": {"sleeve": sleeve, "hem": hem, "shoulder": shoulder},
        }
        state = observation_features(obs, clearance)
        if phase == "approach_sleeve_above":
            target = sleeve + [0, 0, 0.17]
        elif phase in ("approach_sleeve", "grasp_sleeve"):
            target = sleeve + [0, 0, 0.008] if not grasped else ee.copy()
        elif phase == "lift_sleeve":
            target = np.array([sign * 0.44, 0.16, 0.14]) + [*jitter, 0]
        elif phase == "carry_sleeve":
            target = np.array([sign * 0.04, 0.16, 0.14]) + [*jitter, 0]
        elif phase in ("fold_sleeve", "release_sleeve"):
            target = np.array([sign * 0.04, 0.16, 0.018]) + [*jitter, 0]
        elif phase == "retreat_sleeve":
            target = np.array([sign * 0.04, 0.16, 0.30]) + [*jitter, 0]
        elif phase == "approach_hem_above":
            target = hem + [0, 0, 0.17]
        elif phase in ("approach_hem", "grasp_hem"):
            target = hem + [0, 0, 0.008] if not grasped else ee.copy()
        elif phase == "lift_hem":
            target = np.array([sign * 0.24, -0.04, max(0.25, clearance)]) + [*jitter, 0]
        elif phase == "carry_hem":
            target = np.array([shoulder[0], shoulder[1] - 0.12, 0.18])
        elif phase in ("fold_hem", "release_hem"):
            target = shoulder + [0, -0.04, 0.020]
        else:
            target = np.array([sign * 0.38, -0.28, 0.28])
        grip = phase in (
            "grasp_sleeve",
            "lift_sleeve",
            "carry_sleeve",
            "fold_sleeve",
            "grasp_hem",
            "lift_hem",
            "carry_hem",
            "fold_hem",
        )
        states.append(state)
        actions.append([*target, 5.0 if grip else -5.0])
    return np.asarray(states), words, np.asarray(actions, dtype=np.float32)


def train(data: Path, output: Path = DEFAULT_DIRECTORY, steps: int = 400):
    import torch

    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for role, agent in enumerate(ARCHITECTURES):
        started = perf_counter()
        torch.manual_seed(4100 + role)
        network = make_network(ARCHITECTURES[agent])
        state, words, labels = synthetic_examples(agent, 18000, 40000 + role)
        rgb = np.load(data / f"{agent}-rgb.npz")["images"]
        pixels = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255
        pixels = torch.nn.functional.interpolate(
            pixels, (32, 32), mode="bilinear", align_corners=False
        )
        # Independently fitted expert weights supply the accurate geometric term.
        for phase in range(len(PHASES)):
            selected = words[:, phase] > 0.5
            design = np.column_stack([state[selected], np.ones(selected.sum())])
            weights = np.linalg.lstsq(design, labels[selected], rcond=1e-7)[0]
            with torch.no_grad():
                network.experts[phase].copy_(torch.tensor(weights, dtype=torch.float32))
        network.experts.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [p for p in network.parameters() if p.requires_grad], lr=0.002
        )
        rng = np.random.default_rng(42000 + role)
        states, languages, targets = map(torch.from_numpy, (state, words, labels))
        for step in range(steps):
            ids = torch.tensor(rng.integers(0, len(states), 128))
            rgb_ids = torch.tensor(rng.integers(0, len(pixels), 128))
            pred = network(pixels[rgb_ids], states[ids], languages[ids])
            loss = ((pred - targets[ids]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        test_state, test_words, expected = synthetic_examples(agent, 3000, 50000 + role)
        with torch.inference_mode():
            prediction = network(
                pixels[torch.arange(len(test_state)) % len(pixels)],
                torch.from_numpy(test_state),
                torch.from_numpy(test_words),
            ).numpy()
        error = np.linalg.norm(prediction[:, :3] - expected[:, :3], axis=-1)
        training = {
            "source": "original MuJoCo RGB contexts and local-state-augmented analytic demonstrations",
            "external_demonstrations": False,
            "foundation_weights": False,
            "control_recipe": "reachable-arc-v2",
            "state_examples": len(state),
            "rendered_rgb_contexts": len(rgb),
            "steps": steps,
            "training_seeds": [40000 + role, 42000 + role],
            "heldout_state_seed": 50000 + role,
            "note": "RGB contexts are randomized auxiliary inputs; these tests do not establish that RGB is required.",
        }
        payload = {
            "format": FORMAT,
            "feature_basis": "reachable-arc-v2",
            "agent": agent,
            "architecture": ARCHITECTURES[agent],
            "phases": list(PHASES),
            "state_dict": network.state_dict(),
            "training": training,
        }
        target_path = output / f"{agent}.pt"
        torch.save(payload, target_path)
        reports[agent] = {
            "architecture": ARCHITECTURES[agent],
            "training": training,
            "checkpoint_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in network.parameters()),
            "heldout_examples": len(error),
            "position_rmse_m": float(np.sqrt(np.mean(error**2))),
            "position_max_error_m": float(error.max()),
            "grip_accuracy": float(np.mean((prediction[:, 3] > 0) == (expected[:, 3] > 0))),
            "training_seconds": perf_counter() - started,
        }
        print(
            json.dumps({"stage": "cloth_policy_trained", "agent": agent, **reports[agent]}),
            flush=True,
        )
    (output / "evaluation.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "train", "all"))
    parser.add_argument("--data", type=Path, default=Path("datasets/cloth-policies"))
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect(args.data, args.seeds)
    if args.stage in ("train", "all"):
        train(args.data, args.output, args.steps)


if __name__ == "__main__":
    main()
