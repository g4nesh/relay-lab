"""Train the existing X-VLA final pointwise MLP on exact native residuals.

Only a training optimization: deployed policies still execute Florence and all
24 original action-transformer blocks through predict_action_chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .action_dataset import phase_instruction
from .policies import PHASES
from .pretrained import XVLA_DOMAIN_ID, XVLA_ID, XVLA_REVISION, XVLA_WEIGHT_SHA256
from .xvla_adaptation import CHANNELS, NativeXVLA, _read_adapter_tensors, save_adapter
from .xvla_final_block import load_prefix_caches
from .xvla_training import (
    _metrics,
    load_caches,
    load_training_source,
    sha,
    training_lineage,
    verify_cached_equivalence,
)


def matched_motion_labels(prefix_source, prefix_rows, florence_path):
    """Join exact source rows by bound archive, seed, physical tick and row index."""
    sources, data = load_caches([florence_path])
    if sources[0] != prefix_source["source"]:
        raise ValueError("Pointwise cache Florence source identity differs from prefix source")
    for key in ("targets", "phase", "tick", "source_index"):
        if not np.array_equal(data[key], prefix_rows[key]):
            raise ValueError(f"Pointwise source row mismatch: {key}")
    own_xyz = data["proprio"][:, 10:13].copy()
    return own_xyz, np.linalg.norm(data["targets"][:, :3] - own_xyz, axis=1)


def native_residual(block, prefix):
    """Original final attention, with token positions and every context retained."""
    return (prefix + block.attn(block.norm1(prefix)))[:, :1]


def forward_pointwise(parts, residual):
    """Original native tokenwise modules, without an alternative action head."""
    import torch

    hidden = residual + parts.block.mlp(parts.block.norm2(residual))
    return parts.decoder(
        parts.norm(hidden),
        torch.full((len(hidden),), XVLA_DOMAIN_ID, dtype=torch.long, device=hidden.device),
    )


def configure_pointwise(parts, *, freeze_norm2=False):
    import torch

    parts.block.requires_grad_(False)
    parts.block.norm2.requires_grad_(not freeze_norm2)
    parts.block.mlp.requires_grad_(True)
    parts.norm.requires_grad_(True)
    parts.decoder.requires_grad_(True)
    width = parts.norm.normalized_shape[0]
    weight_mask = torch.zeros_like(parts.decoder.fc.weight)
    weight_mask[XVLA_DOMAIN_ID].view(width, 20)[:, list(CHANNELS)] = 1
    bias_mask = torch.zeros_like(parts.decoder.bias.weight)
    bias_mask[XVLA_DOMAIN_ID, list(CHANNELS)] = 1
    hooks = [
        parts.decoder.fc.weight.register_hook(lambda grad: grad * weight_mask),
        parts.decoder.bias.weight.register_hook(lambda grad: grad * bias_mask),
    ]
    parameters = [
        parameter
        for module in (parts.block, parts.norm, parts.decoder)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return parameters, hooks


def condition_norm2(parts, residual, *, minimum_scale=1e-3):
    """Fold training-only feature centering/scaling into original LayerNorm affine."""
    import torch

    if minimum_scale < 1e-3 or not np.isfinite(minimum_scale):
        raise ValueError("Native LayerNorm conditioning scale must be at least 1e-3")
    layer = parts.block.norm2
    with torch.no_grad():
        outputs = layer(residual)
        mean = outputs.mean(dim=(0, 1))
        deviation = outputs.std(dim=(0, 1), unbiased=False)
        scale = deviation.clamp_min(minimum_scale)
        expected = (outputs - mean) / scale
        layer.weight.div_(scale)
        layer.bias.sub_(mean).div_(scale)
        error = float((layer(residual) - expected).abs().max().cpu())
        if error > 0.01:
            raise RuntimeError(
                "Original LayerNorm affine conditioning failed numerical equivalence"
            )
    return {
        "method": "Original blocks.23.norm2 gamma divided by training std; beta shifted by training mean and divided by std",
        "minimum_scale": minimum_scale,
        "feature_std_percentiles": np.percentile(
            deviation.cpu().numpy(), [0, 1, 10, 50, 90, 99, 100]
        ).tolist(),
        "training_mean_scale_sha256": hashlib.sha256(
            mean.cpu().numpy().tobytes() + scale.cpu().numpy().tobytes()
        ).hexdigest(),
        "affine_equivalence_max_error": error,
        "new_runtime_modules": 0,
    }


def whiten_native_fc1(parts, residual, *, minimum_scale=1e-3):
    """Fold a training covariance preconditioner into the ORIGINAL native fc1."""
    import torch

    if minimum_scale < 1e-3 or not np.isfinite(minimum_scale):
        raise ValueError("Native covariance conditioning scale must be at least 1e-3")
    layer = parts.block.mlp.fc1
    with torch.no_grad():
        features = (
            parts.block.norm2(residual)
            .cpu()
            .numpy()
            .reshape(-1, layer.in_features)
            .astype(np.float64)
        )
        mean = features.mean(0)
        centered = features - mean
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(centered))
        scales = np.sqrt(np.maximum(eigenvalues, minimum_scale**2))
        whitening = (eigenvectors / scales[None]) @ eigenvectors.T
        original_weight = layer.weight.cpu().numpy().astype(np.float64)
        original_bias = layer.bias.cpu().numpy().astype(np.float64)
        weight = original_weight @ whitening
        bias = original_bias - weight @ mean
        expected = (centered @ whitening) @ original_weight.T + original_bias
        layer.weight.copy_(torch.from_numpy(weight.astype(np.float32)).to(layer.weight.device))
        layer.bias.copy_(torch.from_numpy(bias.astype(np.float32)).to(layer.bias.device))
        actual = layer(parts.block.norm2(residual)).cpu().numpy().reshape(expected.shape)
        error = float(np.abs(actual - expected).max())
        if error > 0.02:
            raise RuntimeError("Original native fc1 whitening failed numerical affine equivalence")
    return {
        "method": "Training covariance inverse square root folded into original blocks.23.mlp.fc1 weight and bias",
        "norm2_initialization": "Unmodified publisher native LayerNorm",
        "minimum_eigenvalue_std": minimum_scale,
        "eigenvalue_std_percentiles": np.percentile(
            np.sqrt(np.maximum(eigenvalues, 0)), [0, 1, 10, 50, 90, 99, 100]
        ).tolist(),
        "mean_whitening_sha256": hashlib.sha256(mean.tobytes() + whitening.tobytes()).hexdigest(),
        "affine_equivalence_max_error": error,
        "new_runtime_modules": 0,
    }


def load_parts(adapter):
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.xvla.soft_transformer import DomainAwareLinear, TransformerBlock
    from safetensors import safe_open

    metadata, tensors = _read_adapter_tensors(adapter)
    if metadata.get("adapted_native_blocks") != [23]:
        raise ValueError("Pointwise fitting requires the recorded original final block")
    checkpoint = (
        Path(
            snapshot_download(
                XVLA_ID,
                revision=XVLA_REVISION,
                cache_dir=Path(".cache/huggingface/hub"),
                local_files_only=True,
                allow_patterns=["model.safetensors"],
            )
        )
        / "model.safetensors"
    )
    if sha(checkpoint) != XVLA_WEIGHT_SHA256:
        raise ValueError("Pinned native base checkpoint changed")
    block = TransformerBlock(1024, 16, mlp_ratio=4).eval().requires_grad_(False)
    block.load_state_dict(
        {
            name.removeprefix("blocks.23."): torch.from_numpy(value)
            for name, value in tensors.items()
            if name.startswith("blocks.23.")
        },
        strict=True,
    )
    norm = torch.nn.LayerNorm(1024).eval().requires_grad_(False)
    digest = hashlib.sha256()
    with safe_open(checkpoint, framework="np") as saved:
        # Reset the complete native tail to the publisher checkpoint. The
        # previous joint-attention fit is evidence, not this fit's initializer.
        for name in list(tensors):
            if name.startswith("blocks.23."):
                tensors[name] = saved.get_tensor(f"model.transformer.{name}")
        tensors["action_decoder.weight"] = saved.get_tensor(
            "model.transformer.action_decoder.fc.weight"
        )[5].reshape(1024, 20)
        tensors["action_decoder.bias"] = saved.get_tensor(
            "model.transformer.action_decoder.bias.weight"
        )[5]
        block.load_state_dict(
            {
                name.removeprefix("blocks.23."): torch.from_numpy(value)
                for name, value in tensors.items()
                if name.startswith("blocks.23.")
            },
            strict=True,
        )
        norm.load_state_dict(
            {
                name: torch.from_numpy(saved.get_tensor(f"model.transformer.norm.{name}"))
                for name in ("weight", "bias")
            },
            strict=True,
        )
        for name, base_name in (
            ("action_encoder.weight", "action_encoder.fc.weight"),
            ("action_encoder.bias", "action_encoder.bias.weight"),
            ("soft_prompt_hub.weight", "soft_prompt_hub.weight"),
        ):
            digest.update(
                tensors.get(name, saved.get_tensor(f"model.transformer.{base_name}")[5]).tobytes()
            )
    # This is the publisher's original DomainAwareLinear. Other domain slots
    # are never accessed in training; deployment retains all original slots.
    decoder = DomainAwareLinear(1024, 20, num_domains=6).eval().requires_grad_(False)
    with torch.no_grad():
        decoder.fc.weight[5].copy_(torch.from_numpy(tensors["action_decoder.weight"].reshape(-1)))
        decoder.bias.weight[5].copy_(torch.from_numpy(tensors["action_decoder.bias"]))
    return (
        metadata,
        tensors,
        SimpleNamespace(block=block, norm=norm, decoder=decoder),
        digest.hexdigest(),
    )


def build_residuals(parts, prefix_paths, florence_paths, input_sha):
    import torch

    if len(prefix_paths) != len(florence_paths) or not prefix_paths:
        raise ValueError("Each prefix cache requires its matching Florence cache")
    sources, chunks, identity = [], [], None
    device = parts.block.norm1.weight.device
    for path, florence in zip(prefix_paths, florence_paths, strict=True):
        bound, data = load_prefix_caches([path])
        source = bound[0]
        if source["prefix_identity"]["native_input_parameters_sha256"] != input_sha:
            raise ValueError("Native input parameters differ from prefix cache")
        if source["prefix_identity"]["base_checkpoint_sha256"] != XVLA_WEIGHT_SHA256:
            raise ValueError("Native base differs from prefix cache")
        if identity is not None and source["prefix_identity"] != identity:
            raise ValueError("Different native prefix identities")
        identity = source["prefix_identity"]
        own_xyz, motion = matched_motion_labels(source, data, florence)
        residuals, error = [], 0.0
        with torch.inference_mode():
            for index in range(len(data["targets"])):
                prefix = torch.from_numpy(data["prefix"][index : index + 1]).to(device)
                residual = native_residual(parts.block, prefix)
                expected = parts.decoder(
                    parts.norm(parts.block(prefix)[:, :1]),
                    torch.full((len(prefix),), 5, dtype=torch.long, device=device),
                )
                actual = forward_pointwise(parts, residual)
                error = max(error, float((actual - expected).abs().max()))
                if error > 2e-5:
                    raise RuntimeError(
                        "Original native pointwise cache differs from original block"
                    )
                residuals.append(residual[:, 0].cpu().numpy().copy())
        sources.append({**source, "pointwise_equivalence_max_error": error})
        chunks.append(
            {
                **{key: data[key] for key in ("targets", "phase", "tick", "source_index")},
                "residual": np.concatenate(residuals),
                "own_xyz": own_xyz,
                "motion": motion,
                "seed": np.full(len(motion), source["source"]["seed"], np.int64),
            }
        )
        print(
            json.dumps(
                {
                    "event": "pointwise_cache",
                    "seed": source["source"]["seed"],
                    "samples": len(motion),
                    "maximum_error": error,
                }
            ),
            flush=True,
        )
    if len({source["source"]["seed"] for source in sources}) != len(sources):
        raise ValueError("Duplicate pointwise source seeds")
    return sources, {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def sample_balanced(rng, data, count):
    """Half movement observations, half uniform phase then uniform observation."""
    moving = np.flatnonzero(data["motion"] > 0.03)
    if not len(moving):
        raise ValueError("Motion-balanced training requires movement observations")
    phases = np.unique(data["phase"])
    n_moving = count // 2
    result = list(rng.choice(moving, n_moving, replace=True))
    for phase in rng.choice(phases, count - n_moving, replace=True):
        result.append(rng.choice(np.flatnonzero(data["phase"] == phase)))
    return np.asarray(result, np.int64)


def evaluate(parts, data, device):
    import torch

    predictions = []
    with torch.inference_mode():
        for index in range(0, len(data["targets"]), 64):
            residual = torch.from_numpy(data["residual"][index : index + 64]).to(device)[:, None]
            predictions.append(
                forward_pointwise(parts, residual)[:, 0, list(CHANNELS)].cpu().numpy()
            )
    predicted = np.concatenate(predictions)
    first_rows = []
    for seed in np.unique(data["seed"]):
        for phase in np.unique(data["phase"][data["seed"] == seed]):
            rows = np.flatnonzero((data["seed"] == seed) & (data["phase"] == phase))
            first_rows.append(int(rows[np.argmin(data["tick"][rows])]))
    return {
        "overall": _metrics(predicted, data["targets"]),
        "movement_over_30mm": _metrics(
            predicted[data["motion"] > 0.03], data["targets"][data["motion"] > 0.03]
        ),
        "phase_first": _metrics(predicted[first_rows], data["targets"][first_rows]),
        "phases": {
            PHASES[int(phase)]: _metrics(
                predicted[data["phase"] == phase], data["targets"][data["phase"] == phase]
            )
            for phase in np.unique(data["phase"])
        },
        "phase_first_predictions": [
            {
                "seed": int(data["seed"][i]),
                "phase": PHASES[int(data["phase"][i])],
                "tick": int(data["tick"][i]),
                "xyz_error_metres": float(
                    np.linalg.norm(predicted[i, :3] - data["targets"][i, :3])
                ),
                "target": data["targets"][i].tolist(),
                "predicted": predicted[i].tolist(),
            }
            for i in first_rows
        ],
    }


def optimize(parts, data, *, steps, learning_rate, device, seed, label, freeze_norm2=False):
    import torch
    import torch.nn.functional as F

    parameters, hooks = configure_pointwise(parts, freeze_norm2=freeze_norm2)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0)
    residual = torch.from_numpy(data["residual"]).to(device)[:, None]
    targets = torch.from_numpy(data["targets"]).to(device)
    rng, started = np.random.default_rng(seed), perf_counter()
    try:
        for step in range(steps):
            ids = torch.from_numpy(sample_balanced(rng, data, 64)).to(device)
            predicted = forward_pointwise(parts, residual[ids])[:, 0, list(CHANNELS)]
            expected = targets[ids]
            xyz = F.mse_loss(predicted[:, :3], expected[:, :3])
            grip = F.binary_cross_entropy_with_logits(predicted[:, 3], expected[:, 3])
            loss = xyz * 500 + grip
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite native pointwise gradient")
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * (
                    0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (step + 1) / steps))
                )
            if step % 250 == 0 or step + 1 == steps:
                print(
                    json.dumps(
                        {
                            "event": label,
                            "step": step + 1,
                            "loss": float(loss.detach().cpu()),
                            "xyz_rmse_metres": float(xyz.detach().sqrt().cpu()),
                            "grip_bce": float(grip.detach().cpu()),
                            "seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        for hook in hooks:
            hook.remove()
    return evaluate(parts, data, device)


def train(
    adapter,
    caches,
    florence,
    output,
    *,
    validation_caches=(),
    validation_florence=(),
    steps=3000,
    learning_rate=3e-5,
    device="mps",
    condition_features=False,
    sanity_only=False,
    whiten_features=False,
):
    import torch

    torch.set_num_threads(3)
    metadata, tensors, parts, input_sha = load_parts(adapter)
    for module in (parts.block, parts.norm, parts.decoder):
        module.to(device).eval()
    sources, data = build_residuals(parts, caches, florence, input_sha)
    val_sources, val = (
        build_residuals(parts, validation_caches, validation_florence, input_sha)
        if validation_caches
        else ([], None)
    )
    training_seeds = training_lineage(metadata) | set(map(int, np.unique(data["seed"])))
    if val is not None and training_seeds & set(map(int, np.unique(val["seed"]))):
        raise ValueError("Pointwise validation overlaps training lineage")
    for module in (parts.block, parts.norm, parts.decoder):
        module.to(device).eval()
    if condition_features and whiten_features:
        raise ValueError("Choose one explicit native feature conditioning method")
    conditioning = (
        condition_norm2(parts, torch.from_numpy(data["residual"]).to(device)[:, None])
        if condition_features
        else None
    )
    if whiten_features:
        conditioning = whiten_native_fc1(
            parts, torch.from_numpy(data["residual"]).to(device)[:, None]
        )
    initial = [
        {name: value.detach().clone() for name, value in module.state_dict().items()}
        for module in (parts.block, parts.norm, parts.decoder)
    ]
    # Two rows per phase: earliest transition and highest-motion observation.
    sanity_rows = []
    for phase in np.unique(data["phase"]):
        rows = np.flatnonzero(data["phase"] == phase)
        sanity_rows.extend([int(rows[0]), int(rows[1:][np.argmax(data["motion"][rows[1:]])])])
    sanity = {key: value[sanity_rows] for key, value in data.items()}
    sanity_metrics = optimize(
        parts,
        sanity,
        steps=1000 if condition_features or whiten_features else 600,
        learning_rate=1e-4,
        device=device,
        seed=712,
        label="pointwise_sanity32",
        freeze_norm2=whiten_features,
    )
    print(
        json.dumps({"event": "pointwise_sanity_finished", "metrics": sanity_metrics["overall"]}),
        flush=True,
    )
    if sanity_only or (
        (condition_features or whiten_features)
        and sanity_metrics["overall"]["xyz_norm_p95_metres"] >= 0.025
    ):
        report = {
            "event": "pointwise_sanity_only",
            "conditioning": conditioning,
            "sanity_metrics": sanity_metrics,
            "full_training_started": False,
            "sources": [{"seed": item["source"]["seed"], "cache": item} for item in sources],
        }
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "sanity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return report
    # The diagnostic fit never initializes the full-data optimizer.
    for module, state in zip((parts.block, parts.norm, parts.decoder), initial, strict=True):
        module.load_state_dict(state, strict=True)
    started = perf_counter()
    training_metrics = optimize(
        parts,
        data,
        steps=steps,
        learning_rate=learning_rate,
        device=device,
        seed=713,
        label="pointwise_train",
        freeze_norm2=whiten_features,
    )
    validation_metrics = evaluate(parts, val, device) if val is not None else None
    native_block = {
        f"blocks.23.{name}": value.detach().cpu().numpy().copy()
        for name, value in parts.block.state_dict().items()
    }
    for name, value in native_block.items():
        if name.startswith(("blocks.23.attn.", "blocks.23.norm1.")) and not np.array_equal(
            value, tensors[name]
        ):
            raise RuntimeError("Frozen native final attention changed")
        if (
            whiten_features
            and name.startswith("blocks.23.norm2.")
            and not np.array_equal(value, tensors[name])
        ):
            raise RuntimeError("Whitening requires the original norm2 affine to remain frozen")
    weight = parts.decoder.fc.weight[5].detach().cpu().numpy().reshape(1024, 20).copy()
    bias = parts.decoder.bias.weight[5].detach().cpu().numpy().copy()
    unused = [i for i in range(20) if i not in CHANNELS]
    if not np.array_equal(
        weight[:, unused], tensors["action_decoder.weight"][:, unused]
    ) or not np.array_equal(bias[unused], tensors["action_decoder.bias"][unused]):
        raise RuntimeError("Unsupervised native decoder columns changed")
    boundary_buffer = io.BytesIO()
    np.savez_compressed(
        boundary_buffer, **{key: data[key] for key in ("residual", "seed", "tick", "source_index")}
    )
    boundary_bytes = boundary_buffer.getvalue()
    result = save_adapter(
        output,
        weight,
        bias,
        dataset_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        native_block=native_block,
        native_norm={
            f"norm.{name}": value.detach().cpu().numpy().copy()
            for name, value in parts.norm.state_dict().items()
        },
        native_inputs={
            name: value
            for name, value in tensors.items()
            if name.startswith(("action_encoder.", "soft_prompt_hub."))
        }
        or None,
        training={
            "method": "Original final norm2/MLP, final LayerNorm and native decoder columns; cached exact post-attention token0 residual",
            "prefix_adapter_sha256": metadata["decoder_sha256"],
            "native_tail_initialization": "Publisher checkpoint, including original final attention; previous adapted tail discarded",
            "training_seeds": sorted(set(map(int, np.unique(data["seed"])))),
            "accumulated_training_seeds": sorted(training_seeds),
            "sources": [{"seed": item["source"]["seed"], "cache": item} for item in sources],
            "validation_sources": val_sources,
            "noise_device": sources[0]["prefix_identity"]["device"],
            "noise_seed": 712,
            "steps": steps,
            "learning_rate": learning_rate,
            "batch_size": 64,
            "sampling": "50% uniform rows with target-minus-measured-own-XYZ norm >0.03m; 50% uniform phase then uniform row",
            "native_feature_conditioning": conditioning,
            "native_norm2_frozen": whiten_features,
            "native_norm2_freeze_reason": "Keep the training-whitening basis fixed; prior joint affine updates amplified through the whitened fc1 and destroyed feature conditioning"
            if whiten_features
            else None,
            "native_training_boundary_sha256": hashlib.sha256(boundary_bytes).hexdigest(),
            "native_training_boundary_device": device,
            "training_motion_rows": int(np.sum(data["motion"] > 0.03)),
            "sanity_32_metrics": sanity_metrics,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "native_modules_trained": ([] if whiten_features else ["blocks.23.norm2"])
            + [
                "blocks.23.mlp",
                "norm",
                "action_decoder domain5 columns10,11,12,19",
            ],
            "effective_trainable_parameters": sum(
                parameter.numel()
                for module in (parts.block.norm2, parts.block.mlp, parts.norm)
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            + 4100,
            "seconds": perf_counter() - started,
            "closed_loop_task_success": None,
            "full_native_post_install_equivalence": "Required before deployment",
        },
    )
    (Path(output) / "native-training-boundary.npz").write_bytes(boundary_bytes)
    print(
        json.dumps(
            {
                "event": "pointwise_finished",
                "adapter_sha256": result["decoder_sha256"],
                "training": training_metrics,
                "validation": validation_metrics,
            }
        ),
        flush=True,
    )
    return result


def representative_indices(phases, maximum=32):
    phases = np.asarray(phases)
    if phases.ndim != 1 or not len(phases) or maximum < len(np.unique(phases)):
        raise ValueError("Representative proof budget must include every observed phase")
    selected = [int(np.flatnonzero(phases == phase)[0]) for phase in np.unique(phases)]
    for index in np.linspace(0, len(phases) - 1, min(len(phases), maximum), dtype=int):
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) == maximum:
            break
    return sorted(selected)


def verify_pointwise_bridge(
    native,
    prefix_cache,
    florence_cache,
    episode,
    *,
    indices=None,
    tolerance=1e-4,
    training_boundary=None,
):
    """Compare stored training residuals directly to actual full native actions."""
    import torch

    sources, data = load_prefix_caches([prefix_cache])
    matched_motion_labels(sources[0], data, florence_cache)
    if sha(Path(episode) / "manifest.json") != sources[0]["source"]["source_manifest_sha256"]:
        raise ValueError("Pointwise proof episode identity mismatch")
    _, episodes = load_training_source(episode)
    rows = episodes["right"]
    transformer = native.probe.model.model.transformer
    parts = SimpleNamespace(
        block=transformer.blocks[23],
        norm=transformer.norm,
        decoder=transformer.action_decoder,
    )
    if (
        training_boundary is None
        or sha(training_boundary) != native.adapter["training"]["native_training_boundary_sha256"]
    ):
        raise ValueError("Pointwise proof requires hash-bound training residuals")
    with np.load(training_boundary, allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    indices = representative_indices(data["phase"]) if indices is None else indices
    evidence = []
    for index in indices:
        source_index = int(data["source_index"][index])
        observation = {
            "agent": "right",
            **{
                key: rows[key][source_index]
                for key in ("rgb", "ee_position", "ee_rotation", "grasped", "gripper_closed")
            },
        }
        actual, _ = native.predict(
            observation, phase_instruction("right", PHASES[int(data["phase"][index])])
        )
        with torch.inference_mode():
            selected = np.flatnonzero(
                (saved["seed"] == sources[0]["source"]["seed"])
                & (saved["source_index"] == source_index)
                & (saved["tick"] == data["tick"][index])
            )
            if len(selected) != 1:
                raise ValueError("Stored pointwise boundary source row is missing or ambiguous")
            residual = torch.from_numpy(saved["residual"][selected]).to(native.probe.device)[
                :, None
            ]
            predicted = (
                forward_pointwise(parts, residual)[0, 0, list(CHANNELS)].cpu().numpy().copy()
            )
            predicted[3] = 1 / (1 + np.exp(-predicted[3]))
        error = float(np.abs(predicted - actual[0, list(CHANNELS)]).max())
        if error > tolerance:
            raise RuntimeError(
                f"Stored pointwise training boundary differs from full deployed native output: {error}"
            )
        evidence.append(
            {
                "kind": "stored_pointwise_training_boundary_vs_actual_full_native",
                "verified": True,
                "maximum_executed_action_error": error,
                "tolerance": tolerance,
                "adapter_sha256": native.adapter["decoder_sha256"],
                "source_manifest_sha256": sources[0]["source"]["source_manifest_sha256"],
                "prefix_sha256": sources[0]["prefix_sha256"],
                "sample_index": index,
                "source_index": source_index,
                "tick": int(data["tick"][index]),
                "native_predict_action_chunk_executed": True,
                "simulation_actions_executed": 0,
            }
        )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--florence-cache", action="append", required=True)
    parser.add_argument("--validation-cache", action="append", default=[])
    parser.add_argument("--validation-florence-cache", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--equivalence-cache", required=True)
    parser.add_argument("--equivalence-episode", required=True)
    parser.add_argument("--condition-features", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--whiten-features", action="store_true")
    args = parser.parse_args()
    result = train(
        args.adapter,
        args.cache,
        args.florence_cache,
        args.output,
        validation_caches=args.validation_cache,
        validation_florence=args.validation_florence_cache,
        steps=args.steps,
        learning_rate=args.learning_rate,
        condition_features=args.condition_features,
        sanity_only=args.sanity_only,
        whiten_features=args.whiten_features,
    )
    if result.get("full_training_started") is False:
        return
    import gc

    import torch

    gc.collect()
    torch.mps.empty_cache()
    native = NativeXVLA(
        device="mps",
        adapter=args.output,
        expected_sha256=result["decoder_sha256"],
        action_horizon=1,
    )
    _, equivalence_data = load_caches([args.equivalence_cache])
    evidence = [
        verify_cached_equivalence(
            native, args.equivalence_cache, args.equivalence_episode, sample_index=index
        )
        for index in sorted(
            {0, len(equivalence_data["tick"]) // 2, len(equivalence_data["tick"]) - 1}
        )
    ]
    evidence.extend(
        verify_pointwise_bridge(
            native,
            args.cache[0],
            args.equivalence_cache,
            args.equivalence_episode,
            training_boundary=Path(args.output) / "native-training-boundary.npz",
        )
    )
    if not all(item["verified"] for item in evidence):
        raise RuntimeError(
            "Saved pointwise native adaptation failed public full-policy equivalence"
        )
    (Path(args.output) / "native-equivalence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(
        json.dumps({"event": "pointwise_post_install_verified", "evidence": evidence}), flush=True
    )


if __name__ == "__main__":
    main()
