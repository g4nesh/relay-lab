"""Adapt X-VLA's native multimodal projection through its frozen action stack.

The original 24 transformer blocks and action encoder remain untouched. There
is no replacement controller, whitening transform, or extra inference module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .xvla_adaptation import CHANNELS, NOISE_SEED, NativeXVLA, save_adapter
from .xvla_pointwise import representative_indices, sample_balanced
from .xvla_training import (
    _batch,
    configure_native_gradients,
    evaluate_native,
    load_caches,
    native_forward,
    native_loss,
    verify_cached_equivalence,
)


def configure_multimodal(transformer):
    """Only original multimodal projection, own prompts/head, and final norm."""
    groups, hooks = configure_native_gradients(transformer)
    retained = []
    for group in groups:
        if group["name"].startswith("action_encoder."):
            group["params"][0].requires_grad_(False)
        else:
            retained.append(group)
    if transformer.use_hetero_proj:
        raise ValueError("Pinned native multimodal contract requires the original shared Linear")
    for name, module in (("vlm_proj", transformer.vlm_proj), ("norm", transformer.norm)):
        module.requires_grad_(True)
        retained.append({"params": list(module.parameters()), "name": name})
    return retained, hooks


def frozen_hash(transformer):
    digest = hashlib.sha256()
    for name, parameter in transformer.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def optimize(
    model, data, noise, *, steps, batch_size, learning_rate, device, label, memory_limit_gib=9
):
    import torch

    groups, hooks = configure_multimodal(model.transformer)
    parameters = [parameter for group in groups for parameter in group["params"]]
    optimizer = torch.optim.Adam(groups, lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, steps, eta_min=learning_rate / 10
    )
    rng, started, losses = np.random.default_rng(712), perf_counter(), []
    try:
        for step in range(steps):
            indices = sample_balanced(rng, data, batch_size)
            batch = _batch(torch, data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = native_forward(model, batch, noise)
            loss, components = native_loss(prediction, batch["targets"])
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 10)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Nonfinite native multimodal gradient")
            memory = torch.mps.driver_allocated_memory() / 2**30 if device == "mps" else 0
            if memory > memory_limit_gib:
                raise RuntimeError(f"Native multimodal trial exceeds memory bound: {memory:.2f}GiB")
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
                print(
                    json.dumps(
                        {
                            "event": label,
                            "step": step + 1,
                            "steps": steps,
                            "loss": losses[-1],
                            "xyz_rmse_metres": float(
                                (components["xyz_loss"] / 500).detach().sqrt().cpu()
                            ),
                            "grip_bce": float(components["grip_loss"].detach().cpu()),
                            "gradient_norm": float(gradient_norm.cpu()),
                            "mps_driver_gib": memory,
                            "seconds": perf_counter() - started,
                            "projection_gradient_norm": float(
                                model.transformer.vlm_proj.weight.grad.norm().cpu()
                            ),
                        }
                    ),
                    flush=True,
                )
    finally:
        for hook in hooks:
            hook.remove()
    return {"losses": losses, "seconds": perf_counter() - started}


def train(
    native,
    caches,
    output,
    *,
    validation_caches=(),
    steps=2000,
    batch_size=8,
    learning_rate=1e-4,
    equivalence_evidence=None,
):
    import torch

    if native.adapter is not None:
        raise ValueError("Multimodal adaptation must initialize the original publisher policy")
    if not equivalence_evidence or equivalence_evidence.get("verified") is not True:
        raise ValueError(
            "Full native/cache equivalence is required before adapting multimodal input"
        )
    if not 1 <= batch_size <= 8 or not 1 <= steps <= 2000:
        raise ValueError("Native multimodal trial exceeds bounded batch or step budget")
    sources, data = load_caches(caches)
    val_sources, validation = load_caches(validation_caches) if validation_caches else ([], None)
    if {source["seed"] for source in sources} & {source["seed"] for source in val_sources}:
        raise ValueError("Native multimodal validation overlaps training")
    device, model = native.probe.device, native.probe.model.model
    if any(source["noise_device"] != device for source in [*sources, *val_sources]):
        raise ValueError("Native multimodal training/cache devices differ")
    data["motion"] = np.linalg.norm(data["targets"][:, :3] - data["proprio"][:, 10:13], axis=1)
    model.vlm.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()
    _groups, hooks = configure_multimodal(model.transformer)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.transformer.named_parameters()
        if parameter.requires_grad
    }
    for hook in hooks:
        hook.remove()
    original_frozen_sha = frozen_hash(model.transformer)
    original_weight, original_bias = native.decoder_parameters()
    torch.manual_seed(NOISE_SEED)
    if device == "mps":
        torch.mps.manual_seed(NOISE_SEED)
    noise = torch.randn((1, 30, 20), device=device)
    selected = []
    for phase in np.unique(data["phase"]):
        rows = np.flatnonzero(data["phase"] == phase)
        selected.extend([int(rows[0]), int(rows[1:][np.argmax(data["motion"][rows[1:]])])])
    sanity = {key: value[selected] for key, value in data.items()}

    def restore():
        with torch.no_grad():
            for name, parameter in model.transformer.named_parameters():
                if name in initial:
                    parameter.copy_(initial[name])

    smoke = optimize(
        model,
        sanity,
        noise,
        steps=1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        label="multimodal_gradient_smoke",
    )
    restore()
    sanity_fit = optimize(
        model,
        sanity,
        noise,
        steps=600,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        label="multimodal_sanity32",
    )
    sanity_metrics = evaluate_native(model, sanity, noise, device=device, batch_size=batch_size)
    print(
        json.dumps({"event": "multimodal_sanity_finished", "metrics": sanity_metrics}), flush=True
    )
    if (
        sanity_metrics["overall"]["xyz_norm_p95_metres"] >= 0.025
        or sanity_metrics["overall"]["grip_accuracy"] < 1
    ):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "native_modules": [
                "vlm_proj",
                "soft_prompt_hub domain5",
                "norm",
                "action_decoder domain5 XYZ/grip",
            ],
            "sources": sources,
            "smoke": smoke,
            "sanity_fit": sanity_fit,
            "sanity_metrics": sanity_metrics,
            "full_training_started": False,
            "frozen_parameters_unchanged": frozen_hash(model.transformer) == original_frozen_sha,
            "native_equivalence_before_training": equivalence_evidence,
        }
        (output / "sanity.json").write_text(json.dumps(report, indent=2) + "\n")
        restore()
        model.transformer.requires_grad_(False)
        model.vlm.to(device)
        return report
    restore()
    fit = optimize(
        model,
        data,
        noise,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        label="multimodal_train",
    )
    train_metrics = evaluate_native(model, data, noise, device=device, batch_size=batch_size)
    validation_metrics = (
        evaluate_native(model, validation, noise, device=device, batch_size=batch_size)
        if validation is not None
        else None
    )
    if frozen_hash(model.transformer) != original_frozen_sha:
        raise RuntimeError("Frozen native action encoder/24 transformer blocks changed")
    weight, bias = native.decoder_parameters()
    unused = [index for index in range(20) if index not in CHANNELS]
    if not np.array_equal(weight[:, unused], original_weight[:, unused]) or not np.array_equal(
        bias[unused], original_bias[unused]
    ):
        raise RuntimeError("Native unsupervised decoder columns changed")
    transformer = model.transformer
    metadata = save_adapter(
        output,
        weight,
        bias,
        dataset_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        native_multimodal={
            "vlm_proj.weight": transformer.vlm_proj.weight.detach().cpu().numpy().copy(),
            "vlm_proj.bias": transformer.vlm_proj.bias.detach().cpu().numpy().copy(),
            "soft_prompt_hub.weight": transformer.soft_prompt_hub.weight[5]
            .detach()
            .cpu()
            .numpy()
            .copy(),
        },
        native_norm={
            f"norm.{name}": value.detach().cpu().numpy().copy()
            for name, value in transformer.norm.state_dict().items()
        },
        training={
            "method": "Native multimodal projection and own soft prompts restore instruction conditioning through 24 frozen native transformer blocks",
            "effective_trainable_parameters": 1088516,
            "sources": sources,
            "training_seeds": [source["seed"] for source in sources],
            "accumulated_training_seeds": [source["seed"] for source in sources],
            "validation_sources": val_sources,
            "noise_device": device,
            "noise_seed": NOISE_SEED,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "steps": steps,
            "native_modules_trained": [
                "vlm_proj",
                "soft_prompt_hub domain5",
                "norm",
                "action_decoder domain5 XYZ/grip",
            ],
            "frozen_parameters_sha256": original_frozen_sha,
            "frozen_parameters_unchanged": True,
            "gradient_smoke": smoke,
            "sanity_fit": sanity_fit,
            "sanity_metrics": sanity_metrics,
            "fit": fit,
            "training_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "cached_equivalence_before_training": equivalence_evidence,
            "closed_loop_task_success": None,
        },
    )
    # Reinstall below starts from publisher parameters, so saved-weight proof
    # cannot accidentally pass by retaining an unsaved in-memory adaptation.
    restore()
    model.transformer.requires_grad_(False)
    model.vlm.to(device)
    native.adapter = None
    print(
        json.dumps(
            {
                "event": "multimodal_fit_finished",
                "adapter_sha256": metadata["decoder_sha256"],
                "training": train_metrics,
                "validation": validation_metrics,
            }
        ),
        flush=True,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--validation-cache", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--equivalence-cache", required=True)
    parser.add_argument("--equivalence-episode", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    native = NativeXVLA(device="mps", action_horizon=1)
    before = verify_cached_equivalence(native, args.equivalence_cache, args.equivalence_episode)
    result = train(
        native,
        args.cache,
        args.output,
        validation_caches=args.validation_cache,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        equivalence_evidence=before,
    )
    if result.get("full_training_started") is False:
        return
    installed = NativeXVLA(
        device="mps",
        probe=native.probe,
        adapter=args.output,
        expected_sha256=result["decoder_sha256"],
        action_horizon=1,
    )
    _, rows = load_caches([args.equivalence_cache])
    evidence = [
        verify_cached_equivalence(
            installed, args.equivalence_cache, args.equivalence_episode, sample_index=index
        )
        for index in representative_indices(rows["phase"])
    ]
    if not all(row["verified"] for row in evidence):
        raise RuntimeError("Saved native multimodal adapter failed full-policy equivalence")
    (Path(args.output) / "native-equivalence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "multimodal_post_install_verified",
                "maximum_error": max(item["maximum_action_error"] for item in evidence),
                "samples": len(evidence),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
