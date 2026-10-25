"""Adapt SmolVLA's original final expert feed-forward block from native caches.

The cached value is the original final attention residual, before its norm/MLP.
Deployment runs the entire pinned foundation model and its unchanged native API.
No cache, teacher, landmark, auxiliary network, or lookup is used at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .smol_adaptation import (
    BASE_WEIGHT_SHA256,
    FINAL_PARAMETER_SHAPES,
    NATIVE_PARAMETER_SHAPES,
    SmolNativePolicy,
    fixed_noise,
    install_native_head,
    install_native_parameters,
    load_native_adapter,
    native_predict,
    normalize,
    orientation_contract,
    referenced_orientation,
    save_native_adapter,
    unnormalize,
)
from .smol_native_training import adaptation_lineage, phase_balanced_schedule, select_phase_samples

CACHE_CONTRACT = "smolvla-original-final-attention-residual-token0-v1"
FINAL_NAMES = tuple(name for name in FINAL_PARAMETER_SHAPES if name not in NATIVE_PARAMETER_SHAPES)


def frozen_upstream_identity(adapter):
    """Only parameters before the cache boundary determine reusable residuals."""
    digest = hashlib.sha256(BASE_WEIGHT_SHA256.encode())
    digest.update(str(adapter["noise_seed"]).encode())
    for name in sorted(NATIVE_PARAMETER_SHAPES):
        tensor = adapter.get("native_parameters", {}).get(name)
        if tensor is not None:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def final_native_velocity(backend, residual):
    """Exactly the original model's final feed-forward/residual/norm/head path."""
    flow = backend._model.model
    expert = flow.vlm_with_expert.lm_expert
    layer = expert.layers[31]
    hidden = layer.mlp(layer.post_attention_layernorm(residual)) + residual
    return flow.action_out_proj(expert.norm(hidden).float())


def capture_final_residual(backend, rgb, state, prompt, normalization):
    """Hook an original module; numerically prove the cached token0 action."""
    captured = []
    layer = backend._model.model.vlm_with_expert.lm_expert.layers[31]
    hook = layer.post_attention_layernorm.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        direct, _ = native_predict(backend, rgb, state, prompt, normalization)
    finally:
        hook.remove()
    if len(captured) != 1 or tuple(captured[0].shape) != (1, 50, 480):
        raise RuntimeError("pinned final expert residual boundary changed")
    residual = captured[0][:, :1].contiguous()
    with backend._torch.no_grad():
        velocity = final_native_velocity(backend, residual)
        cached = (fixed_noise(backend._torch, backend.device)[:, :1] - velocity)[0, 0, :7]
        difference = float(np.max(np.abs(cached.cpu().numpy() - direct[0])))
    if not np.isfinite(difference) or difference > 2e-4:
        raise RuntimeError(f"final native block/cache equivalence failed: {difference}")
    return residual[0, 0].detach().cpu().numpy().copy(), difference


def load_source(directory, agent="left"):
    from .action_dataset import (
        CORRECTION_SCHEMA,
        RECOVERY_SCHEMA,
        load_corrections,
        load_episode,
        load_recovery,
    )
    from .mixed_dataset import MIXED_SCHEMA, load_mixed_episode

    schema = json.loads((Path(directory) / "manifest.json").read_text()).get("schema")
    if schema == MIXED_SCHEMA:
        manifest, episodes = load_mixed_episode(directory)
        kind = "mixed_native_teacher_executed_one_step"
    elif schema == CORRECTION_SCHEMA:
        manifest, episodes = load_corrections(directory)
        kind = "development_corrections"
    elif schema == RECOVERY_SCHEMA:
        manifest, episodes = load_recovery(directory)
        kind = "training_recovery"
    else:
        manifest, episodes = load_episode(directory)
        kind = "expert_demonstration"
    if agent not in ("left", "right") or agent not in episodes:
        raise ValueError("native training source does not contain the requested arm")
    return manifest, episodes[agent], kind


def build_final_cache(policy, directory, output, *, per_phase=10):
    from .action_dataset import phase_instruction, smol_action, smol_state
    from .policies import PHASES

    if per_phase < 1:
        raise ValueError("native final cache needs positive phase sampling")
    source, arrays, kind = load_source(directory, policy.agent)
    source_sha = hashlib.sha256((Path(directory) / "manifest.json").read_bytes()).hexdigest()
    cache_path, meta_path = Path(output).with_suffix(".npz"), Path(output).with_suffix(".json")
    expected = {
        "agent": policy.agent,
        "source_manifest_sha256": source_sha,
        "frozen_upstream_sha256": frozen_upstream_identity(policy.adapter),
        "base_weight_sha256": BASE_WEIGHT_SHA256,
        "base_model_id": policy.backend.model_id,
        "base_revision": policy.backend.revision,
        "cache_contract": CACHE_CONTRACT,
        "normalization": policy.adapter["normalization"],
        "samples_per_phase": per_phase,
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        # Existing residual caches predate arm generalization and are left-only.
        if any(
            old.get(key, "left" if key == "agent" else None) != value
            for key, value in expected.items()
        ):
            raise ValueError("native final cache belongs to a different source/model/contract")
        if hashlib.sha256(cache_path.read_bytes()).hexdigest() != old["cache_sha256"]:
            raise ValueError("native final residual cache bytes changed")
        return old
    selected = (
        select_phase_samples(arrays, per_phase)
        if kind == "expert_demonstration"
        else list(range(len(arrays["tick"])))
    )
    hidden, labels, phases, ticks, hashes, differences = [], [], [], [], [], []
    for number, index in enumerate(selected):
        state = smol_state(
            arrays["ee_position"][index],
            arrays["ee_rotation"][index],
            arrays["gripper_closed"][index],
        )
        prompt = phase_instruction(policy.agent, PHASES[int(arrays["phase"][index])])
        residual, difference = capture_final_residual(
            policy.backend, arrays["rgb"][index], state, prompt, policy.adapter["normalization"]
        )
        hidden.append(residual)
        labels.append(
            referenced_orientation(
                smol_action(
                    arrays["target"][index], arrays["ee_rotation"][index], arrays["grip"][index]
                ),
                orientation_contract(policy.adapter["normalization"], agent=policy.agent),
            )
        )
        phases.append(int(arrays["phase"][index]))
        ticks.append(int(arrays["tick"][index]))
        hashes.append(hashlib.sha256(arrays["rgb"][index].tobytes()).hexdigest())
        differences.append(difference)
        if number % 20 == 0 or number + 1 == len(selected):
            print(
                json.dumps(
                    {
                        "event": "native_final_cache",
                        "seed": source["seed"],
                        "sample": number + 1,
                        "total": len(selected),
                        "max_equivalence_error": max(differences),
                    }
                ),
                flush=True,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        residual=np.asarray(hidden, dtype=np.float32),
        actions=np.asarray(labels, dtype=np.float32),
        phases=np.asarray(phases),
        ticks=np.asarray(ticks),
        rgb_sha256=np.asarray(hashes),
    )
    manifest = {
        **expected,
        "initial_adapter_sha256": policy.adapter["sha256"],
        "seed": source["seed"],
        "manifest_sha256": source_sha,
        "source_kind": kind,
        "one_step_labels_only": True,
        "source_success": source.get("success", source.get("source_success")),
        "rows": len(hidden),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "max_native_equivalence_error": max(differences),
        "scope": "original native final attention residual; no teacher or cache at deployment",
    }
    meta_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_final_caches(paths):
    rows, manifests = [], []
    for path in map(Path, paths):
        manifest = json.loads(path.with_suffix(".json").read_text())
        cache = path.with_suffix(".npz")
        if hashlib.sha256(cache.read_bytes()).hexdigest() != manifest["cache_sha256"]:
            raise ValueError("native final residual cache bytes changed")
        if (
            manifest.get("cache_contract") != CACHE_CONTRACT
            or manifest.get("base_weight_sha256") != BASE_WEIGHT_SHA256
        ):
            raise ValueError("native final residual cache contract changed")
        with np.load(cache, allow_pickle=False) as values:
            rows.append({key: values[key].copy() for key in values.files})
        manifests.append(manifest)
    if not rows:
        raise ValueError("native final block training requires source observations")
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}, manifests


def enable_final_training(backend):
    backend._model.eval().requires_grad_(False)
    names = (*FINAL_NAMES, "action_out_proj.weight", "action_out_proj.bias")
    parameters = [backend._model.model.get_parameter(name) for name in names]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def final_metrics(backend, rows, normalization):
    from .policies import PHASES

    torch = backend._torch
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(rows["residual"]), 64):
            residual = torch.from_numpy(rows["residual"][offset : offset + 64, None]).to(
                backend.device
            )
            velocity = final_native_velocity(backend, residual)[:, 0]
            prediction = fixed_noise(torch, backend.device)[0, 0] - velocity
            predictions.append(
                unnormalize(prediction[:, :7].cpu().numpy(), normalization["action"])
            )
    prediction = np.concatenate(predictions)
    error = np.linalg.norm(prediction[:, :3] - rows["actions"][:, :3], axis=1)
    correct = (prediction[:, 6] > 0) == (rows["actions"][:, 6] > 0)

    def metrics(mask):
        return {
            "rows": int(mask.sum()),
            "mean_target_error_m": float(error[mask].mean()),
            "p95_target_error_m": float(np.percentile(error[mask], 95)),
            "gripper_accuracy": float(correct[mask].mean()),
        }

    return {
        **metrics(np.ones(len(error), dtype=bool)),
        "per_phase": {
            name: metrics(rows["phases"] == i)
            for i, name in enumerate(PHASES)
            if np.any(rows["phases"] == i)
        },
        "scope": "native one-step prediction; not closed-loop task success",
    }


def unused_phase_samples(arrays, *, training_per_phase=10, diagnostic_per_phase=2):
    """Choose recorded active frames omitted by the declared expert-cache sampler."""
    if diagnostic_per_phase < 1:
        raise ValueError("unused-frame diagnostic needs a positive sample count")
    excluded = set(select_phase_samples(arrays, training_per_phase))
    chosen = []
    for phase in np.unique(arrays["phase"]):
        available = [
            int(index)
            for index in np.flatnonzero((arrays["phase"] == phase) & ~arrays["hold"])
            if int(index) not in excluded
        ]
        if available:
            chosen.extend(
                available[i]
                for i in np.linspace(
                    0, len(available) - 1, min(diagnostic_per_phase, len(available)), dtype=int
                )
            )
    return chosen


def evaluate_unused_frames(policy, directory, *, training_per_phase=10, was_training_source=True):
    """Report native predictions on unused frames after the fixed fit; never optimize on them."""
    from .action_dataset import phase_instruction, smol_action, smol_state
    from .policies import PHASES

    manifest, arrays, kind = load_source(directory, policy.agent)
    if kind != "expert_demonstration":
        raise ValueError("unused-frame diagnostic requires a complete original demonstration")
    indices = unused_phase_samples(arrays, training_per_phase=training_per_phase)
    if not indices:
        raise ValueError("no unused expert observations remain for the diagnostic")
    predictions, targets, records = [], [], []
    for index in indices:
        state = smol_state(
            arrays["ee_position"][index],
            arrays["ee_rotation"][index],
            arrays["gripper_closed"][index],
        )
        normalized, _ = native_predict(
            policy.backend,
            arrays["rgb"][index],
            state,
            phase_instruction(policy.agent, PHASES[int(arrays["phase"][index])]),
            policy.adapter["normalization"],
        )
        action = unnormalize(normalized[0], policy.adapter["normalization"]["action"])
        target = smol_action(
            arrays["target"][index], arrays["ee_rotation"][index], arrays["grip"][index]
        )
        predictions.append(action)
        targets.append(target)
        records.append(
            {
                "source_index": index,
                "tick": int(arrays["tick"][index]),
                "phase": PHASES[int(arrays["phase"][index])],
                "rgb_sha256": hashlib.sha256(arrays["rgb"][index].tobytes()).hexdigest(),
                "xyz_error_m": float(np.linalg.norm(action[:3] - target[:3])),
            }
        )
    prediction, target = np.asarray(predictions), np.asarray(targets)
    error = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=1)
    return {
        "scope": "post-fit same-seed unused recorded frames; not independent generalization or closed-loop success",
        "agent": policy.agent,
        "seed": manifest["seed"],
        "rows": len(indices),
        "source_manifest_sha256": hashlib.sha256(
            (Path(directory) / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_was_used_for_this_fit": bool(was_training_source),
        "excluded_sampler": {"samples_per_phase": training_per_phase},
        "diagnostic_samples_per_phase": 2,
        "used_for_optimization": False,
        "mean_target_error_m": float(error.mean()),
        "p95_target_error_m": float(np.percentile(error, 95)),
        "max_target_error_m": float(error.max()),
        "gripper_accuracy": float(np.mean((prediction[:, 6] > 0) == (target[:, 6] > 0))),
        "records": records,
    }


def train_final_block(
    policy,
    train_caches,
    validation_caches,
    output,
    *,
    steps=3000,
    batch_size=64,
    learning_rate=3e-5,
):
    if steps < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid native final block optimizer parameters")
    torch, backend, adapter = policy.backend._torch, policy.backend, policy.adapter
    train, manifests = load_final_caches(train_caches)
    validation, val_manifests = load_final_caches(validation_caches)
    lineage = adaptation_lineage(adapter, manifests, val_manifests)
    if any(
        m["frozen_upstream_sha256"] != frozen_upstream_identity(adapter)
        or m["normalization"] != adapter["normalization"]
        for m in manifests + val_manifests
    ):
        raise ValueError("native final cache frozen upstream parameters differ")
    parameters = enable_final_training(backend)
    x = torch.from_numpy(train["residual"][:, None]).to(backend.device)
    desired = torch.zeros((len(x), 1, 32), dtype=torch.float32, device=backend.device)
    desired[:, 0, :7] = torch.from_numpy(
        normalize(train["actions"], adapter["normalization"]["action"])
    ).to(backend.device)
    truth = fixed_noise(torch, backend.device)[:, :1] - desired
    weights = torch.full((32,), 0.01, device=backend.device)
    weights[:3] = 1.0
    weights[6] = 0.3
    schedule = phase_balanced_schedule(train["phases"], steps=steps, batch_size=batch_size)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0)
    started, losses = perf_counter(), []
    for step, indices in enumerate(schedule):
        scale = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / max(1, steps - 1)))
        optimizer.param_groups[0]["lr"] = learning_rate * scale
        ids = torch.from_numpy(indices).to(backend.device)
        optimizer.zero_grad(set_to_none=True)
        velocity = final_native_velocity(backend, x[ids])
        loss = ((velocity - truth[ids]).square() * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("native final block loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step % 100 == 0 or step + 1 == steps:
            print(
                json.dumps(
                    {
                        "event": "native_final_step",
                        "step": step + 1,
                        "steps": steps,
                        "loss": losses[-1],
                        "elapsed_seconds": perf_counter() - started,
                        "memory": backend.model_info["memory"],
                    }
                ),
                flush=True,
            )
    backend._model.requires_grad_(False)
    training = {
        **adapter["training"],
        **lineage,
        "method": "original final native expert norm/MLP/residual/final norm/output projection gradients",
        "train_seeds": [m["seed"] for m in manifests],
        "train_manifest_sha256s": [m["manifest_sha256"] for m in manifests],
        "validation_seeds": [m["seed"] for m in val_manifests],
        "validation_seed": val_manifests[0]["seed"] if len(val_manifests) == 1 else None,
        "validation_manifest_sha256s": [m["manifest_sha256"] for m in val_manifests],
        "validation_manifest_sha256": val_manifests[0]["manifest_sha256"]
        if len(val_manifests) == 1
        else None,
        "training_prediction": final_metrics(backend, train, adapter["normalization"]),
        "validation": final_metrics(backend, validation, adapter["normalization"]),
        "training_source_kinds": [
            {"seed": m["seed"], "kind": m["source_kind"], "one_step_labels_only": True}
            for m in manifests
        ],
        "initial_adapter_sha256": adapter["sha256"],
        "output_refit_horizon": 1,
        "loss_history": losses,
        "optimizer_wall_seconds": perf_counter() - started,
    }
    flow = backend._model.model
    fit = {
        "method": "original final expert block adaptation",
        "horizon": 1,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_learning_rate_ratio": 0.1,
        "updated_parameter_count": sum(p.numel() for p in parameters),
    }
    digest = save_native_adapter(
        output,
        adapter["agent"],
        {
            "weight": flow.action_out_proj.weight.detach().cpu().numpy(),
            "bias": flow.action_out_proj.bias.detach().cpu().numpy(),
            "fit": fit,
        },
        adapter["normalization"],
        training,
        native_parameters={
            name: flow.get_parameter(name).detach().cpu().clone() for name in FINAL_PARAMETER_SHAPES
        },
    )
    report = {"adapter_sha256": digest, "fit": fit, "training": training}
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    from .action_dataset import phase_instruction, smol_state
    from .policies import PHASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--agent", choices=("left", "right"), default="left")
    parser.add_argument("--train-episode", required=True, action="append", type=Path)
    parser.add_argument("--validation-episode", required=True, action="append", type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-phase", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--device", default="mps", choices=("cpu", "mps"))
    parser.add_argument("--interpolation-episode", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = SmolNativePolicy(
        args.agent, args.adapter, device=args.device, action_horizon=1, allow_initializer=True
    )
    caches = {}
    for split, sources in (
        ("training", args.train_episode),
        ("validation", args.validation_episode),
    ):
        caches[split] = []
        for source in sources:
            path = args.cache_root / source.name
            build_final_cache(policy, source, path, per_phase=args.samples_per_phase)
            caches[split].append(path)
    report = train_final_block(
        policy,
        caches["training"],
        caches["validation"],
        args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    # Reinstall the exact serialized original-module tensors before proof.
    saved = load_native_adapter(args.output, args.agent, expected_sha256=report["adapter_sha256"])
    install_native_parameters(policy.backend, saved["native_parameters"])
    install_native_head(policy.backend, saved["weight"].numpy(), saved["bias"].numpy())
    policy.adapter = saved
    _, arrays, _ = load_source(args.validation_episode[0], args.agent)
    state = smol_state(
        arrays["ee_position"][0], arrays["ee_rotation"][0], arrays["gripper_closed"][0]
    )
    proof = []
    for dx in (0.0, 0.01, -0.015, 0.0):
        changed = state.copy()
        changed[0] += dx
        _, error = capture_final_residual(
            policy.backend,
            arrays["rgb"][0],
            changed,
            phase_instruction(args.agent, PHASES[int(arrays["phase"][0])]),
            saved["normalization"],
        )
        proof.append(
            {
                "agent": args.agent,
                "orientation_contract": orientation_contract(saved["normalization"]),
                "state_x_delta": dx,
                "max_normalized_action_difference": error,
                "verified": True,
            }
        )
    report["deployment_equivalence"] = proof
    if args.interpolation_episode is not None:
        report["unused_frame_diagnostic"] = evaluate_unused_frames(
            policy,
            args.interpolation_episode,
            training_per_phase=args.samples_per_phase,
            was_training_source=args.interpolation_episode.resolve()
            in {source.resolve() for source in args.train_episode},
        )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "adapter_sha256": report["adapter_sha256"],
                "fit": report["fit"],
                "training_prediction": report["training"]["training_prediction"],
                "validation": report["training"]["validation"],
                "deployment_equivalence": proof,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
