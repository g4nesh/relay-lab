"""Original MuJoCo cue rendering and supervised private-route codec training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .routing_codec import (
    DEFAULT_CHECKPOINT,
    MODEL_ID,
    SENSOR_SIGNATURE,
    SPACE_ID,
    InspectionCamera,
    RoutingCodec,
    make_network,
)

DATA_DIR = Path(".cache/factory-routing")


def collect(output=DATA_DIR, train_scenes=3072, holdout_scenes=768):
    output.mkdir(parents=True, exist_ok=True)
    camera = InspectionCamera()
    started = perf_counter()
    try:
        for name, count, base in (
            ("train", train_scenes, 100000),
            ("validation", holdout_scenes, 200000),
            ("test", holdout_scenes, 300000),
        ):
            images = []
            labels = []
            for index in range(count):
                label = index % 3
                image = camera.render(label if label < 2 else None, base + index)
                # Original sensor-dropout augmentation adds explicit unreadable
                # examples; it never uses a cue bit in neural inference.
                if label == 2 and index % 15 == 2:
                    value = int(np.random.default_rng(base + index).integers(0, 220))
                    image[:] = value
                images.append(image)
                labels.append(label)
            path = output / f"{name}.npz"
            np.savez_compressed(
                path,
                images=np.asarray(images),
                labels=np.asarray(labels, dtype=np.int64),
                seed_base=base,
                sensor_signature=SENSOR_SIGNATURE,
            )
            print(
                json.dumps(
                    {
                        "stage": "render",
                        "split": name,
                        "scenes": count,
                        "seed_base": base,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    finally:
        camera.close()


def measure(net, images, labels, batch_size=128):
    import torch

    probabilities = []
    latents = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            tensor = (
                torch.from_numpy(images[start : start + batch_size].copy())
                .permute(0, 3, 1, 2)
                .float()
                / 255
            )
            latent, logits = net(tensor)
            probabilities.append(logits.softmax(-1).numpy())
            latents.append(latent.numpy())
    probabilities = np.concatenate(probabilities)
    latents = np.concatenate(latents)
    predicted = probabilities.argmax(-1)
    confidence = probabilities.max(-1)
    valid = labels < 2
    return (
        {
            "samples": len(images),
            "classification_accuracy": float((predicted == labels).mean()),
            "route_samples": int(valid.sum()),
            "route_accuracy": float((predicted[valid] == labels[valid]).mean()),
            "route_accepted_accuracy": float(
                ((predicted[valid] == labels[valid]) & (confidence[valid] >= 0.90)).mean()
            ),
            "unreadable_recall": float((predicted[~valid] == 2).mean()),
            "route_min_confidence": float(confidence[valid].min()),
            "max_latent_norm_error": float(np.abs(np.linalg.norm(latents, axis=1) - 1).max()),
        },
        probabilities,
        latents,
    )


def train(data_dir=DATA_DIR, output=DEFAULT_CHECKPOINT, steps=1200, seed=317):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = np.load(data_dir / "train.npz")
    validation = np.load(data_dir / "validation.npz")
    if (
        str(data["sensor_signature"]) != SENSOR_SIGNATURE
        or str(validation["sensor_signature"]) != SENSOR_SIGNATURE
    ):
        raise ValueError("Cue training data has a different physical camera contract")
    images = data["images"]
    labels = data["labels"]
    net = make_network()
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
    started = perf_counter()
    for step in range(steps):
        indices = rng.integers(0, len(images), 96)
        x = torch.from_numpy(images[indices].copy()).permute(0, 3, 1, 2).float() / 255
        y = torch.from_numpy(labels[indices])
        latent, logits = net(x)
        contrast = 0
        for label in range(3):
            selected = latent[y == label]
            if len(selected) > 1:
                center = F.normalize(selected.mean(0, keepdim=True), dim=-1)
                contrast = contrast + (1 - (selected * center).sum(-1)).mean()
        loss = F.cross_entropy(logits, y) + contrast * 0.025
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == int(steps * 0.7):
            for group in opt.param_groups:
                group["lr"] = 0.0004
        if (step + 1) % 300 == 0:
            print(
                json.dumps(
                    {
                        "stage": "train",
                        "step": step + 1,
                        "loss": loss.item(),
                        "seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    net.eval()
    metrics, _, _ = measure(net, validation["images"], validation["labels"])
    if metrics["route_accepted_accuracy"] < 0.99 or metrics["unreadable_recall"] < 0.98:
        raise RuntimeError(f"Routing codec did not pass held-out validation: {metrics}")
    training = {
        "seed": seed,
        "steps": steps,
        "examples_seen": steps * 96,
        "training_scenes": len(images),
        "train_seed_base": int(data["seed_base"]),
        "validation_seed_base": int(validation["seed_base"]),
        "validation": metrics,
        "training_data_sha256": hashlib.sha256((data_dir / "train.npz").read_bytes()).hexdigest(),
        "validation_data_sha256": hashlib.sha256(
            (data_dir / "validation.npz").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": "original MuJoCo inspection-panel RGB; independent illumination/pose/background/LED noise; unreadable sensor augmentation",
        "seconds": perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": SPACE_ID,
            "sensor_signature": SENSOR_SIGNATURE,
            "state_dict": net.state_dict(),
            "training": training,
        },
        output,
    )
    manifest = {
        "model_id": MODEL_ID,
        "space_id": SPACE_ID,
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sensor_signature": SENSOR_SIGNATURE,
        "parameters": sum(p.numel() for p in net.parameters()),
        "training": training,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def evaluate(data_dir=DATA_DIR, checkpoint=DEFAULT_CHECKPOINT):
    import torch

    codec = RoutingCodec(checkpoint)
    data = np.load(data_dir / "test.npz")
    if str(data["sensor_signature"]) != SENSOR_SIGNATURE:
        raise ValueError("Test data sensor mismatch")
    metrics, _probabilities, latents = measure(codec.network, data["images"], data["labels"])
    # Transport float32 serializes all components; the strong text baseline is
    # produced by the same decoder, so no perception advantage is smuggled in.
    valid = data["labels"] < 2
    text_predictions = []
    latent_predictions = []
    for vector in latents[valid]:
        sender = codec.decode(vector.tolist())
        text_predictions.append(json.loads(sender["text"])["route"])
        transported = json.loads(json.dumps(vector.tolist(), separators=(",", ":")))
        latent_predictions.append(codec.decode(transported)["route"])
    with torch.inference_mode():
        zero = codec.network(torch.zeros((1, 3, 64, 64)))[1].softmax(-1)[0]
    report = {
        "checkpoint_sha256": codec.checkpoint_sha256,
        "sensor_signature": SENSOR_SIGNATURE,
        "test_seed_base": int(data["seed_base"]),
        "test_data_sha256": hashlib.sha256((data_dir / "test.npz").read_bytes()).hexdigest(),
        **metrics,
        "text_embedding_decision_agreement": float(
            np.mean(np.asarray(text_predictions) == np.asarray(latent_predictions))
        ),
        "blank_rgb_prediction": "unreadable"
        if int(zero.argmax()) == 2
        else ("A", "B")[int(zero.argmax())],
        "blank_rgb_unreadable_score": float(zero[2]),
        "text_payload_bytes": len(json.dumps({"route": "A"}, separators=(",", ":")).encode()),
        "embedding_payload_bytes_mean": float(
            np.mean(
                [
                    len(json.dumps(v.tolist(), separators=(",", ":")).encode())
                    for v in latents[valid]
                ]
            )
        ),
    }
    checkpoint.with_name("evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    # Small actual held-out RGB fixtures keep no-GL inference tests reproducible.
    picks = np.r_[
        np.flatnonzero(data["labels"] == 0)[:4],
        np.flatnonzero(data["labels"] == 1)[:4],
        np.flatnonzero(data["labels"] == 2)[:4],
    ]
    np.savez_compressed(
        checkpoint.with_name("verification_rgb.npz"),
        images=data["images"][picks],
        labels=data["labels"][picks],
        seed_base=int(data["seed_base"]),
    )
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--output", type=Path, default=DATA_DIR)
    p.add_argument("--train-scenes", type=int, default=3072)
    p.add_argument("--holdout-scenes", type=int, default=768)
    p = sub.add_parser("train")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=317)
    p = sub.add_parser("evaluate")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output, args.train_scenes, args.holdout_scenes)
    elif args.command == "train":
        train(args.data_dir, args.output, args.steps, args.seed)
    else:
        evaluate(args.data_dir, args.checkpoint)


if __name__ == "__main__":
    main()
