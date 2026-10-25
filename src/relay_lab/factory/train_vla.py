"""Original rendered data, behavior cloning and held-out checks for Factory VLA.

All state-derived targets and analytic teacher fields are confined to this
training/evaluation module. Runtime imports none of them. No public demo
corpus, model download or external training service is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .vision import (
    ROBOT_IDS,
    TAG_OFFSETS,
    TARGETS,
    FactoryCameraRig,
    camera_center,
    sensor_signature,
)
from .vla import (
    DEFAULT_CHECKPOINT,
    SPACE_ID,
    WORDS,
    FactoryVLABackend,
    language_features,
    make_network,
)

DEFAULT_DATA = Path("datasets/factory-vla/original.npz")


def collect(path=DEFAULT_DATA, scenes=600, seed=8123):
    rng = np.random.default_rng(seed)
    rig = FactoryCameraRig()
    images, masks, coordinates, roles = [], [], [], []
    started = perf_counter()
    try:
        for i in range(scenes):
            role = i % len(ROBOT_IDS)
            robot_id = ROBOT_IDS[role]
            center, span = camera_center(robot_id)
            positions = {}
            for name in TARGETS:
                positions[name] = [
                    center[0] + rng.uniform(-span * 0.85, span * 0.85),
                    center[1] + rng.uniform(-span * 0.38, span * 0.38),
                    rng.uniform(0.35, 2.7),
                ]
            rig.set_positions(positions)
            rgb, mask = rig.render(robot_id, segmentation=True)
            images.append(rgb)
            masks.append(mask)
            coordinates.append([positions[name] for name in TARGETS])
            roles.append(role)
            if (i + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "render",
                            "scenes": i + 1,
                            "seconds": round(perf_counter() - started, 2),
                        }
                    ),
                    flush=True,
                )
    finally:
        rig.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.asarray(images),
        masks=np.asarray(masks),
        coords=np.asarray(coordinates, dtype=np.float32),
        marker_coords=np.asarray(coordinates, dtype=np.float32)
        + np.asarray([TAG_OFFSETS[name] for name in TARGETS], dtype=np.float32)[None],
        sensor_signature=sensor_signature(),
        roles=np.asarray(roles),
        seed=seed,
    )
    print(
        json.dumps(
            {
                "dataset": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )


def language_examples():
    examples = []
    for i, name in enumerate(TARGETS):
        if name.startswith("part-"):
            examples += [(f"pick {name}", i, 1), (f"process {name}", i, 3)]
        else:
            examples += [(f"navigate to {name}", i, 0)]
            for cargo in TARGETS[:6]:
                examples.append((f"place {cargo} on {name}", i, 2))
    examples.append(("wait", 0, 4))
    return examples


def teacher_batch(rng, count):
    body_id = rng.integers(0, 4, count)
    kind_id = rng.choice([1, 2, 3, 4], count, p=[0.4, 0.4, 0.15, 0.05])
    mobile = body_id == 1
    kind_id[mobile] = rng.choice([0, 4], mobile.sum(), p=[0.95, 0.05])
    centers = np.array([-7.0, 0.0, 0.0, 6.9])[body_id]
    goal = np.column_stack(
        [
            centers + rng.uniform(-1.15, 1.15, count),
            rng.uniform(1.0, 3.0, count),
            rng.uniform(0.55, 1.05, count),
        ]
    )
    goal[body_id == 2, 0] = rng.uniform(-1.95, 1.95, (body_id == 2).sum())
    goal[body_id == 3, 2] = rng.uniform(0.3, 0.65, (body_id == 3).sum())
    goal[mobile, 0] = rng.choice([-6.0, -1.8, 1.8, 6.0], mobile.sum()) + rng.normal(
        0, 0.015, mobile.sum()
    )
    goal[mobile, 1] = 2.0 + rng.normal(0, 0.08, mobile.sum())
    goal[mobile, 2] = 0.62
    position = goal + rng.normal(0, [1.1, 0.65, 0.75], (count, 3))
    position[:, 2] = np.clip(position[:, 2], 0.25, 2.6)
    position[mobile] = np.column_stack(
        [
            rng.uniform(-7.0, 7.0, mobile.sum()),
            rng.uniform(-1.7, 2.2, mobile.sum()),
            np.zeros(mobile.sum()),
        ]
    )
    near = rng.random(count) < 0.5
    position[near] = goal[near] + rng.normal(0, [0.15, 0.15, 0.15], (near.sum(), 3))
    arrived = rng.random(count) < 0.25
    position[arrived] = goal[arrived] + rng.normal(0, 0.025, (arrived.sum(), 3))
    position[mobile, 2] = 0
    junction = mobile & (rng.random(count) < 0.30)
    position[junction, 0] = goal[junction, 0] + rng.uniform(-0.6, 0.6, junction.sum())
    position[junction, 1] = rng.uniform(-1.65, 2.15, junction.sum())
    corridor = mobile & (rng.random(count) < 0.5)
    position[corridor, 1] = -1.5 + rng.normal(0, 0.04, corridor.sum())
    clearance = np.array([1.5, 0.0, 1.9, 1.75])[body_id]
    cruising = (~mobile) & (rng.random(count) < 0.25)
    position[cruising, 2] = clearance[cruising] + rng.normal(0, 0.03, cruising.sum())
    effective = goal.copy()
    effective[kind_id == 3, 2] += 0.30
    diff = effective - position
    xy = np.linalg.norm(diff[:, :2], axis=1)
    travel_weight = np.clip((xy - 0.025) / 0.09, 0, 1)
    desired_z = effective[:, 2] * (1 - travel_weight) + clearance * travel_weight
    gate = np.clip((position[:, 2] - (clearance - 0.30)) / 0.22, 0, 1)
    # The near-goal horizontal servo stays active throughout descent; a hard
    # clearance gate creates a stable but incorrect hover equilibrium.
    gate = np.maximum(gate, np.clip(1 - xy / 0.5, 0, 1))
    vector = np.column_stack(
        [diff[:, 0] * gate * 2.4, diff[:, 1] * gate * 2.4, (desired_z - position[:, 2]) * 2.4]
    )
    # A smooth training-only lane field: descend to the reserved shared aisle,
    # traverse it, then enter the requested docking lane. No waypoint array.
    dx = goal[mobile, 0] - position[mobile, 0]
    align = np.clip((0.14 - np.abs(dx)) / 0.07, 0, 1)
    desired_y = -1.5 * (1 - align) + goal[mobile, 1] * align
    horizontal = np.clip(1 - np.abs(position[mobile, 1] + 1.5) / 0.20, 0, 1)
    # Keep correcting X throughout the docking column. Blending this gate
    # with only ``align`` creates a false equilibrium near the lane junction.
    horizontal = np.maximum(horizontal, np.clip(1 - np.abs(dx) / 0.70, 0, 1))
    vector[mobile, 0] = np.clip(dx * 3.0, -1.0, 1.0) * horizontal
    vector[mobile, 1] = np.clip((desired_y - position[mobile, 1]) * 3.0, -1.0, 1.0)
    vector[mobile, 2] = 0
    norms = np.linalg.norm(vector, axis=1)
    limits = np.where(mobile, 1.0, 0.70)
    vector *= np.minimum(1, limits / np.maximum(norms, 1e-8))[:, None]
    vector[kind_id == 4] = 0
    yaw = rng.uniform(-np.pi, np.pi, count)
    angle = np.arctan2(vector[:, 1], vector[:, 0])
    error = (angle - yaw + np.pi) % (2 * np.pi) - np.pi
    yaw_action = np.where(
        mobile & (np.linalg.norm(vector[:, :2], axis=1) > 0.03), np.clip(error / 0.12, -1, 1), 0
    )
    dist = np.linalg.norm(diff, axis=1)
    grip = np.ones(count, dtype=np.int64)
    grip[kind_id == 1] = 2
    grip[(kind_id == 1) & (dist < 0.045)] = 0
    grip[(kind_id == 2) & (dist < 0.045)] = 2
    holding = (kind_id == 2).astype(np.float32)
    holding[mobile] = rng.integers(0, 2, mobile.sum())
    velocity = rng.normal(0, 0.2, (count, 3))
    return tuple(
        np.asarray(x, dtype=np.float32)
        for x in [
            position,
            goal,
            np.eye(5)[kind_id],
            np.eye(4)[body_id],
            yaw[:, None],
            holding[:, None],
            velocity,
            vector,
            yaw_action[:, None],
        ]
    ) + (grip,)


def train(
    data_path=DEFAULT_DATA,
    output=DEFAULT_CHECKPOINT,
    seed=91,
    pixel_steps=1500,
    control_steps=16000,
    codec_steps=5000,
):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.load(data_path)
    if str(data["sensor_signature"]) != sensor_signature():
        raise ValueError("Training data does not match current factory sensors")
    net = make_network()
    started = perf_counter()
    flat_rgb = data["images"].reshape(-1, 3)
    flat_mask = data["masks"].ravel()
    pools = [np.flatnonzero(flat_mask == i) for i in range(len(TARGETS) + 1)]
    if any(len(p) == 0 for p in pools):
        raise ValueError("Missing physical tag training class")
    opt = torch.optim.Adam(net.pixels.parameters(), lr=0.004)
    for step in range(pixel_steps):
        indices = np.concatenate([rng.choice(pool, 96) for pool in pools])
        x = torch.from_numpy(flat_rgb[indices].astype(np.float32) / 255)
        y = torch.from_numpy(flat_mask[indices].astype(np.int64))
        features = x
        for layer in net.pixels:
            features = (
                F.linear(features, layer.weight[:, :, 0, 0], layer.bias)
                if isinstance(layer, torch.nn.Conv2d)
                else layer(features)
            )
        loss = F.cross_entropy(features, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
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
    # Camera calibration is fitted to the learned classifier's own centroids.
    polys, masses = [], []
    with torch.no_grad():
        for i in range(0, len(data["images"]), 4):
            rgb = (
                torch.from_numpy(data["images"][i : i + 4].copy()).permute(0, 3, 1, 2).float() / 255
            )
            role = torch.tensor(data["roles"][i : i + 4], dtype=torch.long)
            _, mass, poly = net.locate(rgb, role)
            polys.append(poly.numpy())
            masses.append(mass.numpy())
    polys = np.concatenate(polys)
    masses = np.concatenate(masses)
    valid = masses.min(axis=-1) > 0.25
    offsets = np.asarray([TAG_OFFSETS[n] for n in TARGETS])
    with torch.no_grad():
        for role in range(len(ROBOT_IDS)):
            mask = valid & (data["roles"][:, None] == role)
            X = polys[mask]
            Y = (data["coords"] + offsets[None])[mask]
            coef = np.linalg.lstsq(X, Y, rcond=1e-6)[0]
            net.calibration[role].copy_(torch.from_numpy(coef))
        # Mean tag-to-reference displacement is learned from paired labels.
        net.target_offsets.copy_(
            torch.from_numpy((data["coords"] - (data["coords"] + offsets[None])).mean(0))
        )
    print(
        json.dumps(
            {
                "stage": "calibration",
                "visible_targets": int(valid.sum()),
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )
    examples = language_examples()
    words = torch.from_numpy(np.stack([language_features(t) for t, _, _ in examples]))
    targets = torch.tensor([i for _, i, _ in examples])
    kinds = torch.tensor([i for _, _, i in examples])
    opt = torch.optim.Adam(list(net.language.parameters()) + list(net.kind.parameters()), lr=0.08)
    for step in range(1500):
        loss = F.cross_entropy(net.language(words), targets) + F.cross_entropy(
            net.kind(words), kinds
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(
        json.dumps({"stage": "language", "loss": loss.item(), "seconds": perf_counter() - started}),
        flush=True,
    )
    opt = torch.optim.Adam(net.control.parameters(), lr=0.0015)
    for step in range(control_steps):
        batch = teacher_batch(rng, 768)
        ts = [torch.from_numpy(x) for x in batch[:-1]]
        position, goal, kinds, body, yaw, holding, velocity, delta, yaw_target = ts
        pred = net.motor(position, goal, kinds, body, yaw, holding, velocity)
        grip = torch.from_numpy(batch[-1])
        loss = (
            F.mse_loss(pred[:, :3], delta) * 8
            + F.mse_loss(pred[:, 3:4], yaw_target) * 0.15
            + F.cross_entropy(pred[:, 4:], grip) * 0.3
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == int(control_steps * 0.7):
            for group in opt.param_groups:
                group["lr"] = 0.0004
        if (step + 1) % 2000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "control",
                        "step": step + 1,
                        "loss": loss.item(),
                        "seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    opt = torch.optim.Adam(
        list(net.encoder.parameters()) + list(net.decoder.parameters()), lr=0.0015
    )
    for step in range(codec_steps):
        batch = teacher_batch(rng, 1024)
        position = torch.from_numpy(batch[0])
        goal = torch.from_numpy(batch[1])
        kinds = torch.from_numpy(batch[2])
        body = torch.from_numpy(batch[3])
        # Publish tool/deck pose for AMRs, with independent goal/intent.
        position[:, 2] += body[:, 1] * 0.62
        target = torch.cat([position / 10, goal / 10, kinds, body], -1)
        _latent, pred = net.encode(position, goal, kinds, body)
        loss = F.mse_loss(pred[:, :6], target[:, :6]) * 50 + F.mse_loss(pred[:, 6:], target[:, 6:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == int(codec_steps * 0.7):
            for group in opt.param_groups:
                group["lr"] = 0.0003
        if (step + 1) % 1000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "codec",
                        "step": step + 1,
                        "loss": loss.item(),
                        "seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    # Learn noise-weighted fusion using independently perturbed camera and peer
    # observations. Both pathways get nonzero trained weights.
    goal = rng.uniform([-8, -2, 0.3], [8, 3, 2.5], (30000, 3)).astype(np.float32)
    visual = goal + rng.normal(0, 0.012, goal.shape)
    peer = goal + rng.normal(0, 0.025, goal.shape)
    coef = np.linalg.lstsq(np.concatenate([visual, peer], 1), goal, rcond=None)[0]
    with torch.no_grad():
        net.peer_fusion.weight.copy_(torch.from_numpy(coef.T))
    validation_batch = teacher_batch(np.random.default_rng(seed + 917), 12000)
    with torch.no_grad():
        pos = torch.from_numpy(validation_batch[0])
        go = torch.from_numpy(validation_batch[1])
        ki = torch.from_numpy(validation_batch[2])
        bo = torch.from_numpy(validation_batch[3])
        pos[:, 2] += bo[:, 1] * 0.62
        _, decoded = net.encode(pos, go, ki, bo)
        pose_errors = (decoded[:, :3] * 10 - pos).norm(dim=-1).numpy()
    codec_validation = {
        "seed": seed + 917,
        "samples": 12000,
        "mean_m": float(pose_errors.mean()),
        "p95_m": float(np.percentile(pose_errors, 95)),
        "within_65mm": float((pose_errors < 0.065).mean()),
    }
    training = {
        "codec_validation": codec_validation,
        "seed": seed,
        "rendered_scenes": len(data["images"]),
        "render_seed": int(data["seed"]),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source": "original MuJoCo tagged factory RGB and original behavior-cloning teacher",
        "pixel_steps": pixel_steps,
        "control_steps": control_steps,
        "codec_steps": codec_steps,
        "teacher_examples": control_steps * 768,
        "seconds": perf_counter() - started,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modalities": ["RGB", "instruction", "proprioception", "compatible peer latent"],
        "peer_fusion": "learned least-squares camera/peer fusion; typed same-checkpoint carrier identity required",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": SPACE_ID,
            "sensor_signature": sensor_signature(),
            "words": list(WORDS),
            "targets": list(TARGETS),
            "state_dict": net.state_dict(),
            "training": training,
        },
        output,
    )
    manifest = {
        "model_id": "relay-lab/factory-vla-v1",
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in net.parameters()),
        "training": training,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def refine(checkpoint=DEFAULT_CHECKPOINT, steps=6000, seed=919):
    """Continue original behavior cloning after a measured closed-loop failure."""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    parent_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = make_network()
    net.load_state_dict(payload["state_dict"])
    opt = torch.optim.Adam(net.control.parameters(), lr=0.00035)
    started = perf_counter()
    for step in range(steps):
        batch = teacher_batch(rng, 1024)
        ts = [torch.from_numpy(x) for x in batch[:-1]]
        pred = net.motor(*ts[:7])
        loss = (
            F.mse_loss(pred[:, :3], ts[7]) * 12
            + F.mse_loss(pred[:, 3:4], ts[8]) * 0.1
            + F.cross_entropy(pred[:, 4:], torch.from_numpy(batch[-1])) * 0.3
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == int(steps * 0.75):
            for group in opt.param_groups:
                group["lr"] = 0.0001
        if (step + 1) % 2000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "refine",
                        "step": step + 1,
                        "loss": loss.item(),
                        "seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    payload["state_dict"] = net.state_dict()
    payload["training"].setdefault("refinements", []).append(
        {
            "seed": seed,
            "steps": steps,
            "examples": steps * 1024,
            "parent_checkpoint_sha256": parent_sha,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seconds": perf_counter() - started,
        }
    )
    torch.save(payload, checkpoint)
    manifest = {
        "model_id": "relay-lab/factory-vla-v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in net.parameters()),
        "training": payload["training"],
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def recalibrate_offsets(checkpoint=DEFAULT_CHECKPOINT, data_path=DEFAULT_DATA):
    """Fit changed physical tag mounts from original paired render labels.

    Camera geometry, pixels, language, motor and communication weights are kept
    unchanged. Only learned reference-frame offsets and sensor provenance change.
    """
    import torch

    data = np.load(data_path)
    if str(data["sensor_signature"]) != sensor_signature():
        raise ValueError("Rendered calibration data and current sensor contract differ")
    parent_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    offsets = (data["coords"] - data["marker_coords"]).mean(axis=0)
    payload["state_dict"]["target_offsets"] = torch.from_numpy(offsets.astype(np.float32))
    payload["sensor_signature"] = sensor_signature()
    payload["training"].setdefault("sensor_refinements", []).append(
        {
            "parent_checkpoint_sha256": parent_sha,
            "sensor_signature": sensor_signature(),
            "rendered_scenes": len(data["images"]),
            "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "changed_parameters": ["target_offsets"],
            "reason": "Distinct physical manipulator tag mounts prevent dock-tag interpenetration after placement",
        }
    )
    torch.save(payload, checkpoint)
    manifest = {
        "model_id": "relay-lab/factory-vla-v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "sensor_signature": sensor_signature(),
        "parameters": sum(value.numel() for value in payload["state_dict"].values()),
        "training": payload["training"],
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def evaluate(checkpoint=DEFAULT_CHECKPOINT, scenes=100, seed=49271):
    import torch

    from .vision import FactoryCameraRig, camera_center

    backend = FactoryVLABackend(checkpoint)
    net = backend.network
    rng = np.random.default_rng(seed)
    rig = FactoryCameraRig()
    errors = []
    by_role = {r: [] for r in ROBOT_IDS}
    try:
        for i in range(scenes):
            role = i % len(ROBOT_IDS)
            robot_id = ROBOT_IDS[role]
            center, span = camera_center(robot_id)
            positions = {
                name: [
                    center[0] + rng.uniform(-span * 0.75, span * 0.75),
                    center[1] + rng.uniform(-span * 0.32, span * 0.32),
                    rng.uniform(0.5, 2.5),
                ]
                for name in TARGETS
            }
            rig.set_positions(positions)
            rgb = rig.render(robot_id)
            with torch.inference_mode():
                image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
                pose, mass, _ = net.locate(image, torch.tensor([role]))
                valid = mass[0].min(-1).values.numpy() > 0.25
                error = np.linalg.norm(
                    pose[0].numpy() - np.asarray(list(positions.values())), axis=-1
                )[valid]
                errors.extend(error.tolist())
                by_role[robot_id].extend(error.tolist())
        b = teacher_batch(rng, 8000)
        ts = [torch.from_numpy(x) for x in b[:-1]]
        with torch.inference_mode():
            pred = net.motor(*ts[:7])
            motor_rmse = float(((pred[:, :3] - ts[7]) ** 2).mean().sqrt()) * 0.1
            codec_pos = ts[0].clone()
            codec_pos[:, 2] += ts[3][:, 1] * 0.62
            _latent, decoded = net.encode(codec_pos, ts[1], ts[2], ts[3])
            codec_err = (decoded[:, :3] * 10 - codec_pos).norm(dim=-1).numpy()
            znet = make_network()
            for p in znet.parameters():
                p.zero_()
            zero = znet.motor(*ts[:7])
            zero_rmse = float(((zero[:, :3] - ts[7]) ** 2).mean().sqrt()) * 0.1
        with torch.inference_mode():
            blank = torch.zeros((1, 3, 128, 512))
            _, blank_mass, _ = net.locate(blank, torch.tensor([0]))
            examples = language_examples()
            words = torch.from_numpy(np.stack([language_features(text) for text, _, _ in examples]))
            target_labels = torch.tensor([target for _, target, _ in examples])
            kind_labels = torch.tensor([kind for _, _, kind in examples])
            language_accuracy = float(
                (net.language(words).argmax(-1) == target_labels).float().mean()
            )
            kind_accuracy = float((net.kind(words).argmax(-1) == kind_labels).float().mean())
        report = {
            "checkpoint_sha256": backend.checkpoint_sha256,
            "heldout_seed": seed,
            "rendered_scenes": scenes,
            "visible_samples": len(errors),
            "total_target_observations": scenes * len(TARGETS),
            "visibility_threshold_mass": 0.25,
            "ablations": {
                "blank_rgb_detected_targets": int((blank_mass.min(-1).values > 0.2).sum()),
                "language_target_accuracy": language_accuracy,
                "language_skill_accuracy": kind_accuracy,
            },
            "vision_mean_m": float(np.mean(errors)),
            "vision_p95_m": float(np.percentile(errors, 95)),
            "vision_by_robot": {
                r: {"mean_m": float(np.mean(e)), "p95_m": float(np.percentile(e, 95))}
                for r, e in by_role.items()
            },
            "motor_delta_rmse_m": motor_rmse,
            "zero_weights_motor_delta_rmse_m": zero_rmse,
            "codec_pose_mean_m": float(codec_err.mean()),
            "codec_pose_p95_m": float(np.percentile(codec_err, 95)),
        }
        checkpoint.with_name("evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return report
    finally:
        rig.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--scenes", type=int, default=600)
    p.add_argument("--seed", type=int, default=8123)
    p.add_argument("--output", type=Path, default=DEFAULT_DATA)
    p = sub.add_parser("train")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--pixel-steps", type=int, default=1500)
    p.add_argument("--control-steps", type=int, default=16000)
    p.add_argument("--codec-steps", type=int, default=5000)
    p = sub.add_parser("refine")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=919)
    p = sub.add_parser("recalibrate-offsets")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--scenes", type=int, default=100)
    p.add_argument("--seed", type=int, default=49271)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output, args.scenes, args.seed)
    elif args.command == "train":
        train(
            args.data,
            args.output,
            args.seed,
            args.pixel_steps,
            args.control_steps,
            args.codec_steps,
        )
    elif args.command == "refine":
        refine(args.checkpoint, args.steps, args.seed)
    elif args.command == "recalibrate-offsets":
        recalibrate_offsets(args.checkpoint, args.data)
    else:
        evaluate(args.checkpoint, args.scenes, args.seed)


if __name__ == "__main__":
    main()
