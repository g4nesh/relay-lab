"""Fit X-VLA's original final action-transformer block, with a frozen prefix.

This is a training optimization only. NativeXVLA still runs the complete
publisher policy on each fresh RGB/proprioceptive observation at deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .policies import PHASES
from .pretrained import XVLA_DOMAIN_ID, XVLA_ID, XVLA_REVISION, XVLA_WEIGHT_SHA256
from .xvla_adaptation import (
    CHANNELS,
    NOISE_SEED,
    NativeXVLA,
    _read_adapter_tensors,
    fit_native_columns,
    save_adapter,
    supervised_targets,
)
from .xvla_training import (
    _metrics,
    configure_native_gradients,
    load_caches,
    native_forward,
    native_loss,
    sha,
    training_lineage,
    verify_cached_equivalence,
)

SCHEMA = "relay-xvla-native-final-block-prefix-v1"


def prefix_identity(native):
    """Identify all variable native inputs to the unchanged first 23 blocks."""
    transformer = native.probe.model.model.transformer
    digest = hashlib.sha256()
    for parameter in (
        transformer.action_encoder.fc.weight,
        transformer.action_encoder.bias.weight,
        transformer.soft_prompt_hub.weight,
    ):
        digest.update(parameter[XVLA_DOMAIN_ID].detach().cpu().numpy().tobytes())
    return {
        "base_checkpoint_sha256": XVLA_WEIGHT_SHA256,
        "native_input_parameters_sha256": digest.hexdigest(),
        "noise_seed": NOISE_SEED,
        "device": native.probe.device,
        "frozen_native_blocks": list(range(23)),
        "cache_boundary": "input tokens to original transformer.blocks.23",
    }


def forward_final(transformer, prefix):
    """Exactly the native final block, native LayerNorm, and native decoder."""
    import torch

    hidden = transformer.blocks[-1](prefix)
    return transformer.action_decoder(
        transformer.norm(hidden[:, :30]),
        torch.full((len(prefix),), XVLA_DOMAIN_ID, dtype=torch.long, device=prefix.device),
    )


def configure_final_gradients(transformer):
    """Retain original blocks and modules; train last block and own decoder columns."""
    groups, handles = configure_native_gradients(transformer)
    retained = []
    for group in groups:
        if group["name"].startswith("action_decoder."):
            retained.append(group)
        else:
            group["params"][0].requires_grad_(False)
    transformer.blocks[-1].requires_grad_(True)
    retained.append({"params": list(transformer.blocks[-1].parameters()), "name": "blocks.23"})
    return retained, handles


def cache_prefix(native, source, output):
    """Cache measured-observation tokens, never targets, into the native prefix."""
    # This cache version binds the publisher multimodal projection through its
    # base checkpoint, and separately hashes adapted action inputs. A changed
    # VLM projection needs a new cache identity; never label it as that base.
    if native.adapter and native.adapter.get("adapted_native_multimodal"):
        raise ValueError("Final-block prefix cache does not support an adapted VLM projection")
    sources, arrays = load_caches([source])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch, model = native.probe.torch, native.probe.model.model
    device = native.probe.device
    torch.manual_seed(NOISE_SEED)
    if device == "mps":
        torch.mps.manual_seed(NOISE_SEED)
    noise = torch.randn((1, 30, 20), device=device)
    captured = []

    def capture(_module, args):
        captured.append(args[0].detach())

    handle = model.transformer.blocks[23].register_forward_pre_hook(capture)
    features, maximum_error, started = [], 0.0, perf_counter()
    try:
        with torch.inference_mode():
            for index in range(len(arrays["targets"])):
                batch = {
                    key: torch.from_numpy(arrays[key][index : index + 1]).to(device)
                    for key in ("vlm_features", "aux_visual_inputs", "proprio")
                }
                captured.clear()
                complete = native_forward(model, batch, noise)
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one original final-block invocation")
                prefix = captured[0]
                predicted = forward_final(model.transformer, prefix)
                # The verification call itself invokes the original hooked block.
                if len(captured) != 2:
                    raise RuntimeError("Native final-block verification did not execute")
                error = float((complete - predicted).abs().max().cpu())
                maximum_error = max(maximum_error, error)
                if error > 2e-5:
                    raise RuntimeError(
                        "Native final-block cache boundary differs from full transformer"
                    )
                features.append(prefix[0].cpu().numpy().copy())
                if index % 25 == 0 or index + 1 == len(arrays["targets"]):
                    print(
                        json.dumps(
                            {
                                "event": "native_final_prefix_cache",
                                "seed": sources[0]["seed"],
                                "count": index + 1,
                                "total": len(arrays["targets"]),
                                "seconds": perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
    finally:
        handle.remove()
    np.save(output / "prefix.npy", np.asarray(features, np.float32), allow_pickle=False)
    np.savez_compressed(
        output / "labels.npz",
        **{key: arrays[key] for key in ("targets", "phase", "tick", "source_index")},
    )
    metadata = {
        "schema": SCHEMA,
        "prefix_identity": prefix_identity(native),
        "source": sources[0],
        "samples": len(features),
        "prefix_sha256": sha(output / "prefix.npy"),
        "labels_sha256": sha(output / "labels.npz"),
        "native_transformer_equivalence_max_error": maximum_error,
        "simulation_actions_executed": 0,
    }
    (output / "cache.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_prefix_caches(paths):
    sources, chunks = [], []
    for path in map(Path, paths):
        metadata = json.loads((path / "cache.json").read_text())
        if (
            metadata.get("schema") != SCHEMA
            or metadata.get("native_transformer_equivalence_max_error", 1) > 2e-5
        ):
            raise ValueError("Unverified native final-block prefix cache")
        if (
            sha(path / "prefix.npy") != metadata["prefix_sha256"]
            or sha(path / "labels.npz") != metadata["labels_sha256"]
        ):
            raise ValueError("Native prefix cache changed")
        prefix = np.load(path / "prefix.npy", allow_pickle=False, mmap_mode="r")
        with np.load(path / "labels.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if set(arrays) != {"targets", "phase", "tick", "source_index"}:
            raise ValueError("Unexpected native prefix label fields")
        n = metadata["samples"]
        if (
            prefix.ndim != 3
            or prefix.shape[0] != n
            or prefix.shape[2] != 1024
            or prefix.dtype != np.float32
        ):
            raise ValueError("Invalid original native token dimensions")
        if arrays["targets"].shape != (n, 4) or not np.isin(arrays["targets"][:, 3], [0, 1]).all():
            raise ValueError("Invalid native final-block targets")
        if not np.isfinite(prefix).all() or any(
            len(value) != n or not np.isfinite(value).all() for value in arrays.values()
        ):
            raise ValueError("Invalid native final-block data")
        for key in ("phase", "tick", "source_index"):
            if (
                arrays[key].shape != (n,)
                or arrays[key].dtype.kind not in "iu"
                or np.any(arrays[key] < 0)
            ):
                raise ValueError("Invalid native prefix source indices")
        if np.any(arrays["phase"] >= len(PHASES)):
            raise ValueError("Invalid native prefix phase")
        arrays["prefix"] = prefix
        sources.append(metadata)
        chunks.append(arrays)
    if not chunks or len({item["source"]["seed"] for item in sources}) != len(sources):
        raise ValueError("Unique native prefix source episodes are required")
    if any(item["prefix_identity"] != sources[0]["prefix_identity"] for item in sources):
        raise ValueError("Native prefix caches used different frozen models")
    return sources, {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}


def evaluate_final(transformer, data, *, device="mps", batch_size=4):
    import torch

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data["targets"]), batch_size):
            context = torch.from_numpy(data["prefix"][start : start + batch_size]).to(device)
            predicted = forward_final(transformer, context)
            outputs.append(predicted[:, 0, list(CHANNELS)].cpu().numpy())
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


def train_final(
    native,
    caches,
    output,
    *,
    validation_caches=(),
    steps=5000,
    batch_size=4,
    learning_rate=0.0001,
    equivalence_evidence=None,
):
    """Fine-tune the native final block with the complete inference path preserved."""
    import torch

    if (
        type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or not 1 <= batch_size <= 16
    ):
        raise ValueError("Invalid native final-block training budget")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid native final-block learning rate")
    if not equivalence_evidence or equivalence_evidence.get("verified") is not True:
        raise ValueError("Verified full native inference equivalence is required before fitting")
    sources, data = load_prefix_caches(caches)
    validation_sources, validation = (
        load_prefix_caches(validation_caches) if validation_caches else ([], None)
    )
    identity = prefix_identity(native)
    if (
        equivalence_evidence.get("kind") != "actual_native_vs_cached_florence_equivalence"
        or equivalence_evidence.get("native_predict_action_chunk_executed") is not True
        or equivalence_evidence.get("adapter_sha256") != native.model_info.get("adapter_sha256")
        or not any(
            equivalence_evidence.get("cache_sha256") == item["source"]["archive_sha256"]
            and equivalence_evidence.get("episode_seed") == item["source"]["seed"]
            for item in sources
        )
    ):
        raise ValueError(
            "Full native equivalence evidence is not bound to this model and training cache"
        )
    if any(item["prefix_identity"] != identity for item in [*sources, *validation_sources]):
        raise ValueError("Training model differs from the frozen native cache prefix")
    inherited = training_lineage(native.adapter)
    seen = inherited | {item["source"]["seed"] for item in sources}
    if seen & {item["source"]["seed"] for item in validation_sources}:
        raise ValueError("Native final-block validation overlaps training lineage")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device, model = native.probe.device, native.probe.model.model
    transformer = model.transformer
    warm_start_sha = native.model_info.get("adapter_sha256")
    model.vlm.to("cpu")
    for block in transformer.blocks[:-1]:
        block.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()
    groups, handles = configure_final_gradients(transformer)
    optimizer = torch.optim.Adam(groups, lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=learning_rate / 20
    )
    rng = np.random.default_rng(1701)
    phases = sorted(np.unique(data["phase"]).tolist())
    phase_rows = {phase: np.flatnonzero(data["phase"] == phase) for phase in phases}
    losses, started = [], perf_counter()
    for step in range(steps):
        chosen = rng.choice(phases, size=batch_size)
        indices = np.asarray([rng.choice(phase_rows[phase]) for phase in chosen])
        prefix = torch.from_numpy(data["prefix"][indices]).to(device)
        targets = torch.from_numpy(data["targets"][indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted = forward_final(transformer, prefix)
        loss, _ = native_loss(predicted, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite native final-block training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for group in groups for parameter in group["params"]], 10
        )
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite native final-block training gradient")
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if step < 5 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(
                json.dumps(
                    {
                        "event": "native_final_block_step",
                        "step": step + 1,
                        "steps": steps,
                        "loss": losses[-1],
                        "mean_last_100": float(np.mean(losses[-100:])),
                        "seconds": perf_counter() - started,
                        "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30
                        if device == "mps"
                        else None,
                    }
                ),
                flush=True,
            )
    transformer.requires_grad_(False)
    for handle in handles:
        handle.remove()
    training_metrics = evaluate_final(transformer, data, device=device, batch_size=batch_size)
    validation_metrics = (
        evaluate_final(transformer, validation, device=device, batch_size=batch_size)
        if validation is not None
        else None
    )
    weight, bias = native.decoder_parameters()
    block_parameters = {
        f"blocks.23.{name}": value.detach().cpu().numpy().copy()
        for name, value in transformer.blocks[23].state_dict().items()
    }
    inputs = None
    if native.adapter and native.adapter.get("adapted_native_inputs"):
        inputs = {
            "action_encoder.weight": transformer.action_encoder.fc.weight[5]
            .detach()
            .cpu()
            .numpy()
            .reshape(72, 1024)
            .copy(),
            "action_encoder.bias": transformer.action_encoder.bias.weight[5]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "soft_prompt_hub.weight": transformer.soft_prompt_hub.weight[5]
            .detach()
            .cpu()
            .numpy()
            .copy(),
        }
    metadata = save_adapter(
        output,
        weight,
        bias,
        dataset_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        native_inputs=inputs,
        native_block=block_parameters,
        training={
            "method": "Original native final transformer block and native decoder columns; frozen original Florence and first 23 blocks",
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_schedule": "cosine decay to learning_rate / 20",
            "noise_device": device,
            "noise_seed": NOISE_SEED,
            "warm_start_adapter_sha256": warm_start_sha,
            "training_seeds": [item["source"]["seed"] for item in sources],
            "accumulated_training_seeds": sorted(seen),
            "sources": [{"seed": item["source"]["seed"], "cache": item} for item in sources],
            "validation_sources": validation_sources,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "losses": losses,
            "seconds": perf_counter() - started,
            "closed_loop_task_success": None,
            "cached_equivalence_evidence": equivalence_evidence,
        },
    )
    del optimizer, scheduler
    model.to(device)
    native.adapter = metadata
    return metadata


def refit_final_decoder(
    adapter, caches, output, *, validation_caches=(), ridge=0.1, standardize_features=False
):
    """CPU-only fixed-ridge fit of the existing output columns after block tuning."""
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.xvla.soft_transformer import TransformerBlock
    from safetensors import safe_open

    torch.set_num_threads(3)
    metadata, tensors = _read_adapter_tensors(adapter)
    if metadata.get("adapted_native_blocks") != [23]:
        raise ValueError("Expected the already-trained original final block")
    sources, data = load_prefix_caches(caches)
    val_sources, val = load_prefix_caches(validation_caches) if validation_caches else ([], None)
    seen = training_lineage(metadata) | {item["source"]["seed"] for item in sources}
    if seen & {item["source"]["seed"] for item in val_sources}:
        raise ValueError("Native output refit validation overlaps prior training")
    snapshot = snapshot_download(
        XVLA_ID,
        revision=XVLA_REVISION,
        cache_dir=Path(".cache/huggingface/hub"),
        local_files_only=True,
        allow_patterns=["model.safetensors"],
    )
    checkpoint = Path(snapshot) / "model.safetensors"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != XVLA_WEIGHT_SHA256:
        raise ValueError("Native output refit base LayerNorm checkpoint pin changed")
    norm = torch.nn.LayerNorm(1024).eval().requires_grad_(False)
    input_digest = hashlib.sha256()
    with safe_open(checkpoint, framework="np") as saved:
        norm.load_state_dict(
            {
                name: torch.from_numpy(
                    tensors.get(f"norm.{name}", saved.get_tensor(f"model.transformer.norm.{name}"))
                )
                for name in ("weight", "bias")
            }
        )
        for name, base_name in (
            ("action_encoder.weight", "action_encoder.fc.weight"),
            ("action_encoder.bias", "action_encoder.bias.weight"),
            ("soft_prompt_hub.weight", "soft_prompt_hub.weight"),
        ):
            value = (
                tensors[name]
                if name in tensors
                else saved.get_tensor(f"model.transformer.{base_name}")[5]
            )
            input_digest.update(value.tobytes())
    identity = sources[0]["prefix_identity"]
    if (
        identity["base_checkpoint_sha256"] != XVLA_WEIGHT_SHA256
        or identity["native_input_parameters_sha256"] != input_digest.hexdigest()
        or any(item["prefix_identity"] != identity for item in [*sources, *val_sources])
    ):
        raise ValueError("Output refit context differs from pinned native prefix")
    block = TransformerBlock(1024, 16, mlp_ratio=4).eval().requires_grad_(False)
    block.load_state_dict(
        {
            name.removeprefix("blocks.23."): torch.from_numpy(value)
            for name, value in tensors.items()
            if name.startswith("blocks.23.")
        },
        strict=True,
    )

    def encode(rows):
        features = []
        with torch.inference_mode():
            for start in range(0, len(rows["targets"]), 4):
                prefix = torch.from_numpy(rows["prefix"][start : start + 4])
                features.append(norm(block(prefix)[:, 0]).numpy().copy())
        return np.concatenate(features)

    started = perf_counter()
    features = encode(data)
    validation_features = encode(val) if val is not None else None
    targets = supervised_targets(data["targets"][:, :3], data["targets"][:, 3])
    weight, bias = fit_native_columns(
        features,
        targets,
        tensors["action_decoder.weight"],
        tensors["action_decoder.bias"],
        ridge=ridge,
        standardize_features=standardize_features,
    )

    def metrics(rows, encoded):
        predicted = encoded @ weight[:, CHANNELS] + bias[list(CHANNELS)]
        return {
            "overall": _metrics(predicted, rows["targets"]),
            "phases": {
                PHASES[int(phase)]: _metrics(
                    predicted[rows["phase"] == phase], rows["targets"][rows["phase"] == phase]
                )
                for phase in np.unique(rows["phase"])
            },
        }

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        features=features,
        targets=data["targets"],
        phase=data["phase"],
        validation_features=validation_features
        if validation_features is not None
        else np.empty((0, 1024)),
        validation_targets=val["targets"] if val is not None else np.empty((0, 4)),
    )
    feature_bytes = buffer.getvalue()
    inputs = {
        name: value
        for name, value in tensors.items()
        if name.startswith(("action_encoder.", "soft_prompt_hub."))
    }
    result = save_adapter(
        output,
        weight,
        bias,
        dataset_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        native_block={
            name: value for name, value in tensors.items() if name.startswith("blocks.23.")
        },
        native_inputs=inputs or None,
        native_norm={name: value for name, value in tensors.items() if name.startswith("norm.")}
        or None,
        training={
            "method": "Fixed-regularizer least-squares refit of existing native decoder columns on adapted original final-block features",
            "ridge": ridge,
            "training_feature_preconditioning_folded_into_native_weights": standardize_features,
            "warm_start_adapter_sha256": metadata["decoder_sha256"],
            "training_seeds": [item["source"]["seed"] for item in sources],
            "accumulated_training_seeds": sorted(seen),
            "validation_sources": val_sources,
            "sources": [{"seed": item["source"]["seed"], "cache": item} for item in sources],
            "noise_device": identity["device"],
            "noise_seed": NOISE_SEED,
            "training_metrics": metrics(data, features),
            "validation_metrics": metrics(val, validation_features) if val is not None else None,
            "refit_features_sha256": hashlib.sha256(feature_bytes).hexdigest(),
            "seconds": perf_counter() - started,
            "closed_loop_task_success": None,
            "full_native_post_install_equivalence": "Required before deployment; this fit used the exact native final block on CPU only.",
        },
    )
    (Path(output) / "refit-features.npz").write_bytes(feature_bytes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--florence-cache", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--adapter")
    train = commands.add_parser("train")
    train.add_argument("--cache", action="append", required=True)
    train.add_argument("--validation-cache", action="append", default=[])
    train.add_argument("--equivalence-cache", required=True)
    train.add_argument("--equivalence-episode", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--adapter")
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=0.0001)
    refit = commands.add_parser("refit")
    refit.add_argument("--adapter", required=True)
    refit.add_argument("--cache", action="append", required=True)
    refit.add_argument("--validation-cache", action="append", default=[])
    refit.add_argument("--output", required=True)
    refit.add_argument("--ridge", type=float, default=0.1)
    refit.add_argument("--standardize-features", action="store_true")
    args = parser.parse_args()
    if args.command == "refit":
        result = refit_final_decoder(
            args.adapter,
            args.cache,
            args.output,
            validation_caches=args.validation_cache,
            ridge=args.ridge,
            standardize_features=args.standardize_features,
        )
        print(
            json.dumps(
                {
                    "adapter_sha256": result["decoder_sha256"],
                    "training_metrics": result["training"]["training_metrics"],
                    "validation_metrics": result["training"]["validation_metrics"],
                },
                indent=2,
            )
        )
        return
    native = NativeXVLA(device="mps", adapter=args.adapter, action_horizon=1)
    if args.command == "cache":
        print(json.dumps(cache_prefix(native, args.florence_cache, args.output), indent=2))
        return
    evidence = verify_cached_equivalence(native, args.equivalence_cache, args.equivalence_episode)
    metadata = train_final(
        native,
        args.cache,
        args.output,
        validation_caches=args.validation_cache,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        equivalence_evidence=evidence,
    )
    installed = NativeXVLA(
        device="mps",
        probe=native.probe,
        adapter=args.output,
        expected_sha256=metadata["decoder_sha256"],
        action_horizon=1,
    )
    after = verify_cached_equivalence(installed, args.equivalence_cache, args.equivalence_episode)
    (Path(args.output) / "native-equivalence.json").write_text(json.dumps([after], indent=2) + "\n")
    if not after["verified"]:
        raise RuntimeError("Saved native final block failed full policy inference equivalence")
    print(
        json.dumps({"adapter_sha256": metadata["decoder_sha256"], "equivalence": after}, indent=2)
    )


if __name__ == "__main__":
    main()
