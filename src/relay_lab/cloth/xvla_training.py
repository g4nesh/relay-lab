"""Adapt native X-VLA input projections and soft prompts with frozen Florence.

No new action network is introduced. Training calls the publisher's complete
24-block native transformer on cached, exact Florence outputs. Only domain-5
rows of its action encoder, soft prompts, and executed decoder columns change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .action_dataset import load_episode, phase_instruction
from .policies import PHASES, validate_rgb
from .pretrained import XVLA_DOMAIN_ID, _tokenization_evidence
from .xvla_adaptation import (
    CHANNELS,
    NOISE_SEED,
    NativeXVLA,
    measured_local_state,
    native_contract,
    save_adapter,
)

SCHEMA = "relay-xvla-frozen-florence-training-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_source(path):
    """Preserve separate semantics for successful episodes and one-step recovery."""
    metadata = json.loads((Path(path) / "manifest.json").read_text())
    if metadata.get("schema") == "relay.cloth.native-action-recovery.v1":
        from .action_dataset import load_recovery

        return load_recovery(path)
    if metadata.get("schema") == "relay.cloth.native-action-corrections.v1":
        from .action_dataset import load_corrections

        result = load_corrections(path)
        if result[0].get("agent") != "right":
            raise ValueError("X-VLA requires recorded right-arm corrections")
        return result
    return load_episode(path)


def cache_florence(episode, output, *, native=None, device="mps", stride=8):
    """Capture actual frozen native VLM outputs; no teacher features enter."""
    if type(stride) is not int or stride < 1:
        raise ValueError("Invalid native feature stride")
    episode, output = Path(episode), Path(output)
    manifest, episodes = load_training_source(episode)
    arrays = episodes["right"]
    recovery = manifest["schema"] in {
        "relay.cloth.native-action-recovery.v1",
        "relay.cloth.native-action-corrections.v1",
    }
    source_file = (
        "corrections.json"
        if manifest["schema"] == "relay.cloth.native-action-corrections.v1"
        else "right-samples.json"
        if recovery
        else "right.npz"
    )
    available = np.arange(len(arrays["tick"])) if recovery else np.flatnonzero(~arrays["hold"])
    selected = set(available[:: (1 if recovery else stride)].tolist())
    for phase in np.unique(arrays["phase"]):
        ids = available[arrays["phase"][available] == phase]
        if len(ids):
            selected.update((int(ids[0]), int(ids[-1])))
    indices = sorted(selected)
    if not indices:
        raise ValueError("No active native training frames")
    output.mkdir(parents=True, exist_ok=False)
    native = native or NativeXVLA(device=device)
    probe, torch = native.probe, native.probe.torch
    rows = {key: [] for key in ("vlm_features", "aux_visual_inputs", "proprio", "targets")}
    started = perf_counter()
    for count, index in enumerate(indices):
        observation = {
            "agent": "right",
            **{
                key: arrays[key][index]
                for key in ("rgb", "ee_position", "ee_rotation", "grasped", "gripper_closed")
            },
        }
        rgb = validate_rgb(observation["rgb"])
        prompt = phase_instruction("right", PHASES[int(arrays["phase"][index])])
        with torch.inference_mode():
            prepared = probe.preprocess(
                {
                    "observation.images.image": torch.from_numpy(rgb.copy())
                    .permute(2, 0, 1)
                    .float()
                    / 255,
                    "observation.state": torch.from_numpy(measured_local_state(observation)),
                    "task": prompt,
                }
            )
            _tokenization_evidence(prepared, probe.tokenizer, prompt)
            prepared["domain_id"] = torch.tensor([XVLA_DOMAIN_ID], device=probe.device)
            inputs = probe.model._build_model_inputs(prepared)
            encoded = probe.model.model.forward_vlm(
                inputs["input_ids"], inputs["image_input"], inputs["image_mask"]
            )
        for key in ("vlm_features", "aux_visual_inputs"):
            value = encoded[key][0].detach().cpu().numpy().copy()
            if value.ndim != 2 or value.shape[1] != 1024 or not np.isfinite(value).all():
                raise ValueError("Invalid native frozen Florence boundary")
            rows[key].append(value)
        rows["proprio"].append(inputs["proprio"][0].detach().cpu().numpy().copy())
        rows["targets"].append([*arrays["target"][index], float(arrays["grip"][index])])
        if count % 20 == 0 or count + 1 == len(indices):
            print(
                json.dumps(
                    {
                        "event": "florence_cache",
                        "seed": manifest["seed"],
                        "count": count + 1,
                        "total": len(indices),
                        "seconds": round(perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
    archive = output / "features.npz"
    np.savez_compressed(
        archive,
        **{key: np.asarray(values, dtype=np.float32) for key, values in rows.items()},
        tick=np.asarray(arrays["tick"][indices], np.int64),
        source_index=np.asarray(indices, np.int64),
        phase=arrays["phase"][indices],
    )
    metadata = {
        "schema": SCHEMA,
        "native_contract": native_contract(),
        "source_manifest_sha256": sha(episode / "manifest.json"),
        "source_right_file": source_file,
        "source_right_sha256": manifest["files"][source_file],
        "source_schema": manifest["schema"],
        "source_kind": manifest.get("kind", "original_simulation_episode"),
        "source_status": manifest.get("status"),
        "source_success": manifest.get("success"),
        "label_semantics": (
            "One-step teacher correction at an exploration state; counterfactual action outcome unmeasured"
            if recovery
            else "Executed teacher command at the recorded pre-step observation"
        ),
        "future_action_labels_used": False,
        "seed": manifest["seed"],
        "split": "train",
        "samples": len(indices),
        "archive_sha256": sha(archive),
        "noise_device": probe.device,
        "cache_boundary": "XVLAModel.forward_vlm full Florence token tensors; own measured proprioception; no cloth landmarks",
        "seconds": perf_counter() - started,
    }
    (output / "cache.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_caches(paths):
    """Read hash-pinned, finite native features without pickle."""
    sources, chunks = [], []
    for path in map(Path, paths):
        metadata = json.loads((path / "cache.json").read_text())
        if (
            metadata.get("schema") != SCHEMA
            or metadata.get("native_contract") != native_contract()
            or metadata.get("split") != "train"
        ):
            raise ValueError("Incompatible frozen Florence training cache")
        if sha(path / "features.npz") != metadata["archive_sha256"]:
            raise ValueError("Frozen Florence feature archive changed")
        with np.load(path / "features.npz", allow_pickle=False) as saved:
            rows = {key: saved[key] for key in saved.files}
        expected = {
            "vlm_features",
            "aux_visual_inputs",
            "proprio",
            "targets",
            "tick",
            "phase",
        }
        if set(rows) not in (expected, expected | {"source_index"}):
            raise ValueError("Unexpected native training feature fields")
        # Original full-episode caches used the physical tick as its row index.
        rows.setdefault("source_index", rows["tick"].copy())
        n = metadata["samples"]
        if any(len(value) != n or not np.isfinite(value).all() for value in rows.values()):
            raise ValueError("Invalid frozen Florence samples")
        for key in ("tick", "source_index", "phase"):
            if rows[key].shape != (n,) or rows[key].dtype.kind not in "iu" or np.any(rows[key] < 0):
                raise ValueError("Invalid native cache source index or phase")
        if np.any(rows["phase"] >= len(PHASES)):
            raise ValueError("Invalid native cache phase")
        if rows["proprio"].shape != (n, 20) or rows["targets"].shape != (n, 4):
            raise ValueError("Invalid native pose/action target dimensions")
        if not np.isin(rows["targets"][:, 3], [0, 1]).all() or np.count_nonzero(
            rows["proprio"][:, :10]
        ):
            raise ValueError("Invalid grip targets or privileged peer state")
        for key in ("vlm_features", "aux_visual_inputs"):
            if rows[key].ndim != 3 or rows[key].shape[2] != 1024:
                raise ValueError("Invalid Florence token dimensions")
        sources.append(metadata)
        chunks.append(rows)
    if not chunks or len({row["seed"] for row in sources}) != len(sources):
        raise ValueError("Unique original training episodes are required")
    return sources, {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}


def configure_native_gradients(transformer):
    """Train only permitted existing domain rows; preserve all other weights."""
    import torch

    transformer.eval().requires_grad_(False)
    allowed = [
        ("action_encoder.weight", transformer.action_encoder.fc.weight),
        ("action_encoder.bias", transformer.action_encoder.bias.weight),
        ("soft_prompt_hub.weight", transformer.soft_prompt_hub.weight),
        ("action_decoder.weight", transformer.action_decoder.fc.weight),
        ("action_decoder.bias", transformer.action_decoder.bias.weight),
    ]
    handles, groups = [], []
    for name, parameter in allowed:
        parameter.requires_grad_(True)
        mask = torch.zeros_like(parameter)
        if name == "action_decoder.weight":
            row = mask[XVLA_DOMAIN_ID].reshape(transformer.hidden_size, 20)
            row[:, CHANNELS] = 1
        elif name == "action_decoder.bias":
            mask[XVLA_DOMAIN_ID, list(CHANNELS)] = 1
        else:
            mask[XVLA_DOMAIN_ID] = 1
        handles.append(parameter.register_hook(lambda grad, selection=mask: grad * selection))
        groups.append({"params": [parameter], "name": name})
    return groups, handles


def native_loss(prediction, targets):
    """Selected native EE6D endpoint XYZ and native gripper-logit loss."""
    import torch.nn.functional as F

    xyz = ((prediction[:, 0, 10:13] - targets[:, :3]) ** 2).mean() * 500
    grip = F.binary_cross_entropy_with_logits(prediction[:, 0, 19], targets[:, 3])
    return xyz + grip, {"xyz_loss": xyz, "grip_loss": grip}


def native_forward(model, batch, noise):
    """The official action preprocessing and transformer, without a new head."""
    import torch

    count = len(batch["proprio"])
    state, noisy = model.action_space.preprocess(batch["proprio"], noise.expand(count, -1, -1))
    return model.transformer(
        domain_id=torch.full((count,), XVLA_DOMAIN_ID, device=state.device, dtype=torch.long),
        action_with_noise=noisy,
        t=torch.ones(count, device=state.device),
        proprio=state,
        vlm_features=batch["vlm_features"],
        aux_visual_inputs=batch["aux_visual_inputs"],
    )


def _batch(torch, data, indices, device):
    return {
        key: torch.from_numpy(data[key][indices]).to(device)
        for key in ("vlm_features", "aux_visual_inputs", "proprio", "targets")
    }


def _metrics(predicted, targets):
    errors = np.linalg.norm(predicted[:, :3] - targets[:, :3], axis=1)
    return {
        "samples": len(errors),
        "xyz_norm_mean_metres": float(errors.mean()),
        "xyz_norm_p95_metres": float(np.percentile(errors, 95)),
        "grip_accuracy": float(np.mean((predicted[:, 3] > 0) == targets[:, 3])),
    }


def evaluate_native(model, data, noise, *, device="mps", batch_size=4):
    import torch

    outputs = []
    with torch.no_grad():
        for start in range(0, len(data["targets"]), batch_size):
            batch = _batch(torch, data, slice(start, start + batch_size), device)
            predicted = native_forward(model, batch, noise)
            outputs.append(predicted[:, 0, list(CHANNELS)].detach().cpu().numpy())
    predicted = np.concatenate(outputs)
    return {
        "overall": _metrics(predicted, data["targets"]),
        "phases": {
            PHASES[int(phase)]: _metrics(
                predicted[data["phase"] == phase], data["targets"][data["phase"] == phase]
            )
            for phase in np.unique(data["phase"])
        },
    }


def verify_cached_equivalence(native, cache, episode, *, sample_index=0, tolerance=2e-5):
    """Compare a real full native inference with the cached training boundary."""
    sources, data = load_caches([cache])
    episode = Path(episode)
    if sha(episode / "manifest.json") != sources[0]["source_manifest_sha256"]:
        raise ValueError("Equivalence observation does not match cached source episode")
    _, episodes = load_training_source(episode)
    arrays = episodes["right"]
    if type(sample_index) is not int or not 0 <= sample_index < len(data["tick"]):
        raise ValueError("Invalid equivalence sample index")
    index = int(data["source_index"][sample_index])
    observation = {
        "agent": "right",
        **{
            key: arrays[key][index]
            for key in ("rgb", "ee_position", "ee_rotation", "grasped", "gripper_closed")
        },
    }
    prompt = phase_instruction("right", PHASES[int(arrays["phase"][index])])
    complete, _ = native.predict(observation, prompt)
    torch, model, device = native.probe.torch, native.probe.model.model, native.probe.device
    torch.manual_seed(NOISE_SEED)
    if device == "mps":
        torch.mps.manual_seed(NOISE_SEED)
    noise = torch.randn((1, 30, 20), device=device)
    with torch.inference_mode():
        batch = _batch(torch, data, slice(sample_index, sample_index + 1), device)
        predicted = (
            model.action_space.postprocess(native_forward(model, batch, noise))
            .detach()
            .cpu()
            .numpy()[0]
        )
    maximum_error = float(np.abs(complete - predicted).max())
    return {
        "kind": "actual_native_vs_cached_florence_equivalence",
        "verified": maximum_error <= tolerance,
        "maximum_action_error": maximum_error,
        "tolerance": tolerance,
        "native_chunk_shape": list(complete.shape),
        "native_chunk_sha256": hashlib.sha256(complete.tobytes()).hexdigest(),
        "cached_chunk_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
        "cache_sha256": sources[0]["archive_sha256"],
        "episode_seed": sources[0]["seed"],
        "tick": int(arrays["tick"][index]),
        "source_index": index,
        "adapter_sha256": native.model_info.get("adapter_sha256"),
        "native_predict_action_chunk_executed": True,
        "simulation_actions_executed": 0,
    }


def training_lineage(adapter):
    """Include inherited adaptation seeds when enforcing a held-out split."""
    if adapter is None:
        return set()
    training = adapter.get("training", {})
    seeds = set(training.get("training_seeds", []))
    seeds.update(training.get("accumulated_training_seeds", []))
    if "episode_seed" in training:
        seeds.add(training["episode_seed"])
    for item in training.get("sources", []):
        seed = item.get("seed", item.get("source", {}).get("seed"))
        if type(seed) is not int:
            raise ValueError("Native training source has no explicit seed identity")
        seeds.add(seed)
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("Warm-start native adapter has no verified training-seed lineage")
    return seeds


def train_cached(
    caches,
    output,
    *,
    validation_caches=(),
    steps=10,
    batch_size=1,
    device="mps",
    learning_rate=0.0002,
    warm_start=None,
    native=None,
    equivalence_evidence=None,
):
    """Train permitted native rows through all frozen native transformer blocks."""
    import torch

    if (
        type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or not 1 <= batch_size <= 8
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Invalid native adaptation step or batch count")
    sources, data = load_caches(caches)
    validation_sources, validation = (
        load_caches(validation_caches) if validation_caches else ([], None)
    )
    if {item["seed"] for item in sources} & {item["seed"] for item in validation_sources}:
        raise ValueError("Validation episodes overlap training seeds")
    if any(item["noise_device"] != device for item in [*sources, *validation_sources]):
        raise ValueError("Native feature cache backend differs from training")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if native is not None and warm_start is not None:
        raise ValueError("Pass an existing native model or a warm-start package, not both")
    native = native or NativeXVLA(device=device, adapter=warm_start)
    warm_start_sha256 = native.model_info.get("adapter_sha256")
    accumulated_training_seeds = training_lineage(native.adapter) | {
        item["seed"] for item in sources
    }
    if accumulated_training_seeds & {item["seed"] for item in validation_sources}:
        raise ValueError("Validation episodes overlap inherited native training seeds")
    model = native.probe.model.model
    # Florence is frozen and already cached. Moving it out of the GPU allocator
    # leaves room for native-transformer activation gradients on small machines.
    model.vlm.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()
    groups, handles = configure_native_gradients(model.transformer)
    optimizer = torch.optim.Adam(groups, lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=learning_rate / 20
    )
    torch.manual_seed(NOISE_SEED)
    if device == "mps":
        torch.mps.manual_seed(NOISE_SEED)
    noise = torch.randn((1, 30, 20), device=device)
    rng = np.random.default_rng(917)
    phases = sorted(np.unique(data["phase"]).tolist())
    phase_rows = {phase: np.flatnonzero(data["phase"] == phase) for phase in phases}
    losses, started = [], perf_counter()
    for step in range(steps):
        chosen_phases = rng.choice(phases, size=batch_size)
        indices = np.asarray([rng.choice(phase_rows[phase]) for phase in chosen_phases])
        batch = _batch(torch, data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        predicted = native_forward(model, batch, noise)
        loss, parts = native_loss(predicted, batch["targets"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite native adaptation loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([group["params"][0] for group in groups], 10)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite native adaptation gradient")
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if step < 10 or (step + 1) % 25 == 0 or step + 1 == steps:
            memory = (
                {
                    "mps_allocated_gib": torch.mps.current_allocated_memory() / 2**30,
                    "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30,
                }
                if device == "mps"
                else {}
            )
            print(
                json.dumps(
                    {
                        "event": "native_training_step",
                        "step": step + 1,
                        "steps": steps,
                        "loss": losses[-1],
                        "mean_last_25": float(np.mean(losses[-25:])),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "xyz_loss": float(parts["xyz_loss"].detach().cpu()),
                        "grip_loss": float(parts["grip_loss"].detach().cpu()),
                        "seconds": perf_counter() - started,
                        **memory,
                    }
                ),
                flush=True,
            )
    model.transformer.requires_grad_(False)
    for handle in handles:
        handle.remove()
    train_metrics = evaluate_native(model, data, noise, device=device, batch_size=batch_size)
    val_metrics = (
        evaluate_native(model, validation, noise, device=device, batch_size=batch_size)
        if validation is not None
        else None
    )
    decoder = model.transformer.action_decoder
    weight = decoder.fc.weight[XVLA_DOMAIN_ID].detach().cpu().numpy().reshape(1024, 20).copy()
    bias = decoder.bias.weight[XVLA_DOMAIN_ID].detach().cpu().numpy().copy()
    native_inputs = {
        "action_encoder.weight": model.transformer.action_encoder.fc.weight[XVLA_DOMAIN_ID]
        .detach()
        .cpu()
        .numpy()
        .reshape(72, 1024)
        .copy(),
        "action_encoder.bias": model.transformer.action_encoder.bias.weight[XVLA_DOMAIN_ID]
        .detach()
        .cpu()
        .numpy()
        .copy(),
        "soft_prompt_hub.weight": model.transformer.soft_prompt_hub.weight[XVLA_DOMAIN_ID]
        .detach()
        .cpu()
        .numpy()
        .copy(),
    }
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    metadata = save_adapter(
        output,
        weight,
        bias,
        dataset_sha256=source_digest,
        native_inputs=native_inputs,
        training={
            "method": "supervised gradient adaptation of original native action encoder, soft prompts and executed decoder columns; frozen Florence and all 24 transformer blocks",
            "noise_device": device,
            "noise_seed": NOISE_SEED,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_schedule": "cosine decay to learning_rate / 20",
            "warm_start_adapter_sha256": warm_start_sha256,
            "sources": sources,
            "training_seeds": [item["seed"] for item in sources],
            "accumulated_training_seeds": sorted(accumulated_training_seeds),
            "validation_sources": validation_sources,
            "training_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "losses": losses,
            "seconds": perf_counter() - started,
            "closed_loop_task_success": None,
            "cached_equivalence_evidence": equivalence_evidence,
        },
    )
    # Leave a caller-supplied policy usable through its normal full native API.
    # Training moved only the frozen VLM to CPU; deployment must encode new RGB.
    del optimizer, scheduler
    model.vlm.to(device)
    native.adapter = metadata
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--episode", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--stride", type=int, default=8)
    train = commands.add_parser("train")
    train.add_argument("--cache", action="append", required=True)
    train.add_argument("--validation-cache", action="append", default=[])
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=0.0002)
    train.add_argument("--warm-start")
    args = parser.parse_args()
    result = (
        cache_florence(args.episode, args.output, stride=args.stride)
        if args.command == "cache"
        else train_cached(
            args.cache,
            args.output,
            validation_caches=args.validation_cache,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warm_start=args.warm_start,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
