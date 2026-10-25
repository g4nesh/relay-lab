"""Native SmolVLA projection training with a frozen visual/language KV cache.

Image/language tokens cannot attend to state in SmolVLA's prefix mask. We cache
those tokens, append the trainable own-state projection through the original VLM,
then run the original action expert and output projection. No learned substitute
controller or object-coordinate label enters inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from .smol_adaptation import (
    BASE_WEIGHT_SHA256,
    CHUNK_SIZE,
    DEFAULT_NOISE_SEED,
    FINAL_FORMAT,
    GRADIENT_FORMAT,
    NATIVE_PARAMETER_SHAPES,
    _prepared_batch,
    fit_native_head,
    fixed_noise,
    install_native_head,
    install_native_parameters,
    load_native_adapter,
    normalize,
    orientation_contract,
    referenced_orientation,
    save_native_adapter,
    unnormalize,
    verify_base_checkpoint,
)

CACHE_CONTRACT = "smolvla-image-language-prefix-with-native-state-append-v1"


def adaptation_lineage(adapter, manifests, validation_manifests):
    """Warm starts retain every seed that influenced parameters or normalization."""
    agent = adapter.get("agent", "left")
    if agent not in ("left", "right") or any(
        manifest.get("agent", "left") != agent for manifest in [*manifests, *validation_manifests]
    ):
        raise ValueError("native adaptation cache belongs to a different arm")
    previous = adapter["training"]
    seeds = set(previous.get("train_seeds", [])) | set(previous.get("adaptation_train_seeds", []))
    for key in ("train_seed", "normalization_training_seed"):
        if previous.get(key) is not None:
            seeds.add(previous[key])
    seeds.update(m["seed"] for m in manifests)
    if seeds & {m["seed"] for m in validation_manifests}:
        raise ValueError("native adaptation lineage overlaps validation seeds")
    hashes = set(previous.get("train_manifest_sha256s", []))
    hashes.update(previous.get("adaptation_manifest_sha256s", []))
    if previous.get("train_manifest_sha256"):
        hashes.add(previous["train_manifest_sha256"])
    hashes.update(m["manifest_sha256"] for m in manifests)
    return {"adaptation_train_seeds": sorted(seeds), "adaptation_manifest_sha256s": sorted(hashes)}


def validate_cache_source(manifest, directory, backend, normalization, *, agent="left"):
    """A matching directory basename is insufficient to reuse frozen features."""
    if agent not in ("left", "right") or manifest.get("agent", "left") != agent:
        raise ValueError("native prefix cache belongs to a different arm")
    expected = {
        "manifest_sha256": hashlib.sha256(
            (Path(directory) / "manifest.json").read_bytes()
        ).hexdigest(),
        "normalization": normalization,
        "model_id": backend.model_id,
        "revision": backend.revision,
        "base_weight_sha256": BASE_WEIGHT_SHA256,
        "cache_contract": CACHE_CONTRACT,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("native prefix cache source or pinned inference contract changed")
    proof = manifest.get("equivalence", {})
    if not proof.get("verified") or not np.isfinite(
        proof.get("max_normalized_action_difference", np.inf)
    ):
        raise ValueError("native prefix cache has no verified inference equivalence")
    if proof["max_normalized_action_difference"] > 2e-4:
        raise ValueError("native prefix cache inference equivalence tolerance exceeded")
    return manifest


def clone_cache(pairs, device):
    """Fresh cache containers prevent suffix mutation or graph reuse between steps."""
    from transformers import DynamicCache

    cache = DynamicCache()
    for index, (keys, values) in enumerate(pairs):
        cache.update(keys.to(device).clone(), values.to(device).clone(), index)
    return cache


def capture_visual_language_cache(backend, rgb, state8, prompt, normalization):
    """No-gradient original VLM prefill, excluding only the final state token."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    torch = backend._torch
    policy, flow = backend._model, backend._model.model
    if flow.prefix_length != 0 or backend._config.n_obs_steps != 1:
        raise ValueError("native cache boundary requires one unpadded own-state token")
    with torch.no_grad():
        batch = _prepared_batch(backend, rgb, state8, prompt, normalization)
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        embs, masks, attention = flow.embed_prefix(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state=state,
        )
        if not bool(attention[0, -1]) or bool(attention[0, :-1].any()):
            raise ValueError("pinned SmolVLA visual/language/state attention boundary changed")
        base_embs, base_masks, base_attention = embs[:, :-1], masks[:, :-1], attention[:, :-1]
        _, cache = flow.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(base_masks, base_attention),
            position_ids=torch.cumsum(base_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[base_embs, None],
            use_cache=True,
        )
        pairs = [
            (layer.keys.detach().cpu().clone(), layer.values.detach().cpu().clone())
            for layer in cache.layers
        ]
        return {
            "pairs": pairs,
            "prefix_mask": base_masks.cpu().clone(),
            "state": state.cpu().clone(),
            "prompt": prompt,
            "rgb_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
        }


def native_cached_velocity(backend, cached, *, noise_seed=DEFAULT_NOISE_SEED):
    """Differentiable native state append + native expert flow velocity at t=1."""
    torch, flow = backend._torch, backend._model.model
    cache = clone_cache(cached["pairs"], backend.device)
    mask = cached["prefix_mask"].to(backend.device)
    state = cached["state"].to(backend.device)
    state_token = flow.state_proj(state)[:, None, :]
    whole_mask = torch.cat(
        (mask, torch.ones((1, 1), dtype=torch.bool, device=backend.device)), dim=1
    )
    backbone = flow.vlm_with_expert
    old_mode = backbone.attention_mode
    try:
        # The official prefix prefill uses VLM self-attention at EVERY layer,
        # including layers where the action expert later uses cross-attention.
        backbone.attention_mode = "self_attn"
        _, cache = backbone.forward(
            attention_mask=whole_mask[:, None, :],
            position_ids=mask.sum(dim=1, keepdim=True),
            past_key_values=cache,
            inputs_embeds=[state_token, None],
            use_cache=True,
        )
    finally:
        backbone.attention_mode = old_mode
    velocity = flow.denoise_step(
        prefix_pad_masks=whole_mask,
        past_key_values=cache,
        x_t=fixed_noise(torch, backend.device, noise_seed),
        timestep=torch.ones(1, device=backend.device),
    )
    return velocity


def verify_native_cache_equivalence(backend, rgb, state, prompt, normalization, *, atol=2e-4):
    """Check the cached training route against full native inference numerically."""
    from .smol_adaptation import native_predict

    cached = capture_visual_language_cache(backend, rgb, state, prompt, normalization)
    with backend._torch.no_grad():
        velocity = native_cached_velocity(backend, cached)
        cached_action = (
            (fixed_noise(backend._torch, backend.device) - velocity)[0, :, :7].cpu().numpy()
        )
    direct, _ = native_predict(backend, rgb, state, prompt, normalization)
    error = float(np.max(np.abs(cached_action - direct)))
    if not np.isfinite(error) or error > atol:
        raise RuntimeError(f"cached native training path differs from full native model: {error}")
    return cached, {
        "max_normalized_action_difference": error,
        "tolerance": atol,
        "verified": True,
        "orientation_contract": orientation_contract(normalization),
    }


def select_phase_samples(arrays, per_phase=3):
    selected = []
    for phase in range(16):
        indices = np.flatnonzero((arrays["phase"] == phase) & ~arrays["hold"])
        if len(indices):
            selected.extend(
                indices[
                    np.linspace(0, len(indices) - 1, min(per_phase, len(indices)), dtype=int)
                ].tolist()
            )
    return sorted(set(selected))


def build_prefix_cache(backend, directory, agent, normalization, output, *, per_phase=3):
    from .action_dataset import (
        CORRECTION_SCHEMA,
        RECOVERY_SCHEMA,
        future_chunk,
        load_corrections,
        load_episode,
        phase_instruction,
        smol_action,
        smol_state,
    )
    from .mixed_dataset import MIXED_SCHEMA, load_mixed_episode
    from .policies import PHASES

    source_manifest = json.loads((Path(directory) / "manifest.json").read_text())
    recovery = source_manifest.get("schema") == RECOVERY_SCHEMA
    mixed = source_manifest.get("schema") == MIXED_SCHEMA
    correction = source_manifest.get("schema") == CORRECTION_SCHEMA or recovery or mixed
    if mixed:
        manifest, episodes = load_mixed_episode(directory)
    elif recovery:
        from .action_dataset import load_recovery

        manifest, episodes = load_recovery(directory)
    else:
        manifest, episodes = load_corrections(directory) if correction else load_episode(directory)
    if agent not in ("left", "right") or agent not in episodes:
        raise ValueError("native training source does not contain the requested arm")
    arrays = episodes[agent]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    selected = (
        list(range(len(arrays["tick"]))) if correction else select_phase_samples(arrays, per_phase)
    )
    for number, index in enumerate(selected):
        state = smol_state(
            arrays["ee_position"][index],
            arrays["ee_rotation"][index],
            arrays["gripper_closed"][index],
        )
        prompt = phase_instruction(agent, PHASES[int(arrays["phase"][index])])
        if number == 0:
            cached, equivalence = verify_native_cache_equivalence(
                backend, arrays["rgb"][index], state, prompt, normalization
            )
        else:
            cached = capture_visual_language_cache(
                backend, arrays["rgb"][index], state, prompt, normalization
            )
        if correction:
            action = smol_action(
                arrays["target"][index], arrays["ee_rotation"][index], arrays["grip"][index]
            )
            chunk = {
                "actions": np.repeat(action[None], CHUNK_SIZE, axis=0),
                "action_is_pad": np.arange(CHUNK_SIZE) > 0,
            }
        else:
            chunk = future_chunk(arrays, index, CHUNK_SIZE, family="smolvla", agent=agent)
        cached["actions"] = backend._torch.from_numpy(
            referenced_orientation(
                chunk["actions"], orientation_contract(normalization, agent=agent)
            )
        )
        cached["action_is_pad"] = backend._torch.from_numpy(chunk["action_is_pad"])
        cached["phase"], cached["tick"] = int(arrays["phase"][index]), int(arrays["tick"][index])
        path = output / f"sample-{number:04d}.pt"
        backend._torch.save(cached, path)
        records.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tick": cached["tick"],
                "phase": cached["phase"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "prefix_cached",
                    "seed": manifest["seed"],
                    "sample": number + 1,
                    "tick": cached["tick"],
                    "bytes": path.stat().st_size,
                }
            ),
            flush=True,
        )
    summary = {
        "agent": agent,
        "seed": manifest["seed"],
        "manifest_sha256": hashlib.sha256(
            (Path(directory) / "manifest.json").read_bytes()
        ).hexdigest(),
        "equivalence": equivalence,
        "records": records,
        "normalization": normalization,
        "model_id": backend.model_id,
        "revision": backend.revision,
        "base_weight_sha256": BASE_WEIGHT_SHA256,
        "cache_contract": CACHE_CONTRACT,
        "cache_scope": "frozen original VLM image/language tokens, excluding state",
        "source_kind": "mixed_native_teacher_executed_one_step"
        if mixed
        else "training_recovery"
        if recovery
        else "development_corrections"
        if correction
        else "expert_demonstration",
        "one_step_labels_only": correction,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def enable_native_projection_training(backend):
    flow = backend._model.model
    backend._model.eval().requires_grad_(False)
    names = [*NATIVE_PARAMETER_SHAPES, "action_out_proj.weight", "action_out_proj.bias"]
    for name in names:
        flow.get_parameter(name).requires_grad_(True)
    return [flow.get_parameter(name) for name in names]


def load_cache_records(torch, directories):
    entries, manifests = [], []
    for directory in map(Path, directories):
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests.append(manifest)
        for record in manifest["records"]:
            path = directory / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("cached original VLM prefix changed")
            entries.append(path)
    return entries, manifests


def phase_balanced_schedule(phases, *, steps, batch_size, seed=741):
    """Deterministic phase coverage without using validation loss for sampling."""
    phases = np.asarray(phases)
    unique = np.unique(phases)
    if not len(unique) or steps < 1 or batch_size < 1:
        raise ValueError("balanced native training requires nonempty phases and positive sizes")
    rng = np.random.default_rng(seed)
    order, result = [], []
    for _ in range(steps * batch_size):
        if not order:
            order = rng.permutation(unique).tolist()
        phase = order.pop()
        result.append(int(rng.choice(np.flatnonzero(phases == phase))))
    return np.asarray(result).reshape(steps, batch_size)


def evaluate_projection_model(backend, entries, normalization, *, horizon=8):
    from .policies import PHASES

    torch = backend._torch
    errors, grippers, phases = [], [], []
    with torch.no_grad():
        for path in entries:
            cached = torch.load(path, map_location="cpu", weights_only=True)
            velocity = native_cached_velocity(backend, cached)
            actions = unnormalize(
                (fixed_noise(torch, backend.device) - velocity)[0, :, :7].cpu().numpy(),
                normalization["action"],
            )
            truth = cached["actions"].numpy()
            valid = ~cached["action_is_pad"].numpy()
            valid[horizon:] = False
            errors.extend(np.linalg.norm(actions[valid, :3] - truth[valid, :3], axis=1).tolist())
            grippers.extend(((actions[valid, 6] > 0) == (truth[valid, 6] > 0)).tolist())
            phases.extend([cached["phase"]] * int(valid.sum()))
    error, grip, phase = np.asarray(errors), np.asarray(grippers), np.asarray(phases)

    def metrics(mask):
        return {
            "rows": int(mask.sum()),
            "mean_target_error_m": float(error[mask].mean()),
            "p95_target_error_m": float(np.percentile(error[mask], 95)),
            "gripper_accuracy": float(grip[mask].mean()),
        }

    return {
        **metrics(np.ones(len(error), dtype=bool)),
        "per_phase": {
            name: metrics(phase == i) for i, name in enumerate(PHASES) if np.any(phase == i)
        },
        "scope": "demonstration prediction; no closed-loop success claim",
    }


def cached_native_feature_rows(backend, entries, normalization, *, horizon=8):
    """Capture the existing native output projection input after state adaptation."""
    torch = backend._torch
    features, targets, phases, actions, chunk_indices, noises = [], [], [], [], [], []
    noise = fixed_noise(torch, "cpu")[0].numpy()
    for path in entries:
        cached = torch.load(path, map_location="cpu", weights_only=True)
        captured = []
        handle = backend._model.model.action_out_proj.register_forward_pre_hook(
            lambda _module, args, captured=captured: captured.append(
                args[0].detach().cpu().numpy().copy()
            )
        )
        try:
            with torch.no_grad():
                native_cached_velocity(backend, cached)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (1, CHUNK_SIZE, 480):
            raise RuntimeError("original native output projection feature boundary changed")
        valid = ~cached["action_is_pad"].numpy()
        valid[horizon:] = False
        desired = np.zeros((CHUNK_SIZE, 32), dtype=np.float32)
        desired[:, :7] = normalize(cached["actions"].numpy(), normalization["action"])
        features.append(captured[0][0, valid])
        targets.append((noise - desired)[valid])
        noises.append(noise[valid])
        actions.append(cached["actions"].numpy()[valid])
        chunk_indices.extend(np.flatnonzero(valid).tolist())
        phases.extend([cached["phase"]] * int(valid.sum()))
    return {
        "features": np.concatenate(features),
        "velocity_targets": np.concatenate(targets),
        "noise": np.concatenate(noises),
        "actions": np.concatenate(actions),
        "phases": np.asarray(phases),
        "chunk_indices": np.asarray(chunk_indices),
    }


def refit_projection_output(
    backend,
    initial_adapter,
    train_caches,
    output,
    *,
    validation_caches=(),
    ridge=0.1,
    horizon=8,
    agent="left",
):
    """Condition the existing native output layer after native input adaptation."""
    from .smol_adaptation import evaluate_native_head

    adapter = load_native_adapter(initial_adapter, agent)
    if adapter["format"] not in (GRADIENT_FORMAT, FINAL_FORMAT):
        raise ValueError("post-gradient output refit requires a native projection adapter")
    install_native_parameters(backend, adapter["native_parameters"])
    install_native_head(backend, adapter["weight"].numpy(), adapter["bias"].numpy())
    backend._model.eval().requires_grad_(False)
    entries, manifests = load_cache_records(backend._torch, train_caches)
    validation_entries, validation_manifests = load_cache_records(backend._torch, validation_caches)
    lineage = adaptation_lineage(adapter, manifests, validation_manifests)
    if not entries or any(
        m["normalization"] != adapter["normalization"] for m in manifests + validation_manifests
    ):
        raise ValueError("native output refit needs consistent cached training observations")
    train = cached_native_feature_rows(backend, entries, adapter["normalization"], horizon=horizon)
    validation = (
        cached_native_feature_rows(
            backend, validation_entries, adapter["normalization"], horizon=horizon
        )
        if validation_entries
        else None
    )
    fitted = fit_native_head(train["features"], train["velocity_targets"], ridge=ridge)
    train_metrics = evaluate_native_head(fitted, train, adapter["normalization"])
    train_metrics["scope"] = "training prediction; not held-out or closed-loop task success"
    val_metrics = (
        evaluate_native_head(fitted, validation, adapter["normalization"])
        if validation is not None
        else {"status": "not_evaluated"}
    )
    training = {
        **adapter["training"],
        **lineage,
        "method": "native projection gradients followed by native output ridge conditioning",
        "output_refit_source_sha256": adapter["sha256"],
        "training_prediction": train_metrics,
        "validation": val_metrics,
        "output_refit_horizon": horizon,
        "train_seeds": [m["seed"] for m in manifests],
        "train_manifest_sha256s": [m["manifest_sha256"] for m in manifests],
        "validation_seeds": [m["seed"] for m in validation_manifests],
        "validation_manifest_sha256s": [m["manifest_sha256"] for m in validation_manifests],
        "training_source_kinds": [
            {
                "seed": m["seed"],
                "kind": m["source_kind"],
                "one_step_labels_only": m["one_step_labels_only"],
            }
            for m in manifests
        ],
    }
    digest = save_native_adapter(
        output,
        adapter["agent"],
        fitted,
        adapter["normalization"],
        training,
        native_parameters=adapter["native_parameters"],
    )
    np.savez_compressed(
        Path(output).with_suffix(".features.npz"),
        **{f"training_{k}": v for k, v in train.items()},
        **({f"validation_{k}": v for k, v in validation.items()} if validation is not None else {}),
    )
    report = {"adapter_sha256": digest, "fit": fitted["fit"], "training": training}
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_native_projections(
    backend,
    initial_adapter,
    train_caches,
    output,
    *,
    validation_caches=(),
    steps=300,
    learning_rate=1e-5,
    horizon=8,
    seed=741,
    batch_size=1,
    balance_phases=False,
    final_learning_rate_ratio=1.0,
    agent="left",
):
    """Backpropagate through the original VLM state token and native action expert."""
    if (
        steps < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
        or not 1 <= horizon <= CHUNK_SIZE
        or batch_size < 1
        or not 0 < final_learning_rate_ratio <= 1
    ):
        raise ValueError("native training requires positive steps/rate and a valid chunk horizon")
    torch = backend._torch
    adapter = load_native_adapter(initial_adapter, agent)
    if adapter["format"] == FINAL_FORMAT:
        raise ValueError("a final-block adapter must continue through native final-block training")
    install_native_head(backend, adapter["weight"].numpy(), adapter["bias"].numpy())
    if "native_parameters" in adapter:
        install_native_parameters(backend, adapter["native_parameters"])
    parameters = enable_native_projection_training(backend)
    entries, manifests = load_cache_records(torch, train_caches)
    validation_entries, validation_manifests = load_cache_records(torch, validation_caches)
    if not entries:
        raise ValueError("native projection training requires observations")
    lineage = adaptation_lineage(adapter, manifests, validation_manifests)
    if any(
        m["normalization"] != adapter["normalization"] for m in [*manifests, *validation_manifests]
    ):
        raise ValueError("native prefix caches use inconsistent input normalization")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0)
    rng = np.random.default_rng(seed)
    if balance_phases:
        schedule = phase_balanced_schedule(
            [r["phase"] for m in manifests for r in m["records"]],
            steps=steps,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        schedule = rng.integers(len(entries), size=(steps, batch_size))
    noise = fixed_noise(torch, backend.device)
    started, losses = perf_counter(), []
    for step in range(steps):
        progress = step / max(1, steps - 1)
        lr_scale = final_learning_rate_ratio + (1 - final_learning_rate_ratio) * 0.5 * (
            1 + np.cos(np.pi * progress)
        )
        optimizer.param_groups[0]["lr"] = learning_rate * lr_scale
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for row in schedule[step]:
            cached = torch.load(entries[int(row)], map_location="cpu", weights_only=True)
            desired = torch.zeros_like(noise)
            desired[0, :, :7] = torch.from_numpy(
                normalize(cached["actions"].numpy(), adapter["normalization"]["action"])
            ).to(backend.device)
            target_velocity = noise - desired
            valid = (~cached["action_is_pad"]).to(backend.device)
            valid[horizon:] = False
            velocity = native_cached_velocity(backend, cached)
            weights = torch.full((32,), 0.01, device=backend.device)
            weights[:3], weights[6] = 1, 0.3
            loss = (((velocity[0] - target_velocity[0]) ** 2) * weights)[valid].mean()
            if not torch.isfinite(loss):
                raise RuntimeError("native projection training produced nonfinite loss")
            (loss / batch_size).backward()
            batch_loss += float(loss.detach().cpu()) / batch_size
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(batch_loss)
        if (step + 1) % 10 == 0 or step == 0:
            print(
                json.dumps(
                    {
                        "event": "native_gradient_step",
                        "step": step + 1,
                        "steps": steps,
                        "loss": losses[-1],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": perf_counter() - started,
                        "memory": backend.model_info["memory"],
                    }
                ),
                flush=True,
            )
    backend._model.eval().requires_grad_(False)
    train_metrics = evaluate_projection_model(
        backend, entries, adapter["normalization"], horizon=horizon
    )
    validation_metrics = (
        evaluate_projection_model(
            backend, validation_entries, adapter["normalization"], horizon=horizon
        )
        if validation_entries
        else {"status": "not_evaluated"}
    )
    flow = backend._model.model
    fitted = {
        "weight": flow.action_out_proj.weight.detach().cpu().numpy(),
        "bias": flow.action_out_proj.bias.detach().cpu().numpy(),
        "fit": {
            "method": "native projection supervised gradient adaptation",
            "steps": steps,
            "learning_rate": learning_rate,
            "horizon": horizon,
            "updated_parameter_count": sum(p.numel() for p in parameters),
            "last_loss": losses[-1],
            "batch_size": batch_size,
            "phase_balanced_sampling": balance_phases,
            "final_learning_rate_ratio": final_learning_rate_ratio,
        },
    }
    training = {
        **adapter["training"],
        **lineage,
        "method": "fixed-noise t=1 supervised native VLA projection gradients",
        "train_seeds": [m["seed"] for m in manifests],
        "validation_seeds": [m["seed"] for m in validation_manifests],
        "validation_seed": validation_manifests[0]["seed"]
        if len(validation_manifests) == 1
        else None,
        "validation_manifest_sha256": validation_manifests[0]["manifest_sha256"]
        if len(validation_manifests) == 1
        else None,
        "train_manifest_sha256s": [m["manifest_sha256"] for m in manifests],
        "validation_manifest_sha256s": [m["manifest_sha256"] for m in validation_manifests],
        "training_prediction": train_metrics,
        "validation": validation_metrics,
        "native_cache_equivalence": [m["equivalence"] for m in manifests],
        "initial_adapter_sha256": adapter["sha256"],
        "wall_seconds": perf_counter() - started,
        "loss_history": losses,
        "training_source_kinds": [
            {
                "seed": m["seed"],
                "kind": m["source_kind"],
                "one_step_labels_only": m["one_step_labels_only"],
            }
            for m in manifests
        ],
    }
    native_parameters = {
        name: flow.get_parameter(name).detach().cpu().clone() for name in NATIVE_PARAMETER_SHAPES
    }
    digest = save_native_adapter(
        output,
        adapter["agent"],
        fitted,
        adapter["normalization"],
        training,
        native_parameters=native_parameters,
    )
    report = {
        "format": GRADIENT_FORMAT,
        "adapter_sha256": digest,
        "fit": fitted["fit"],
        "training": training,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    from ..smolvla import SmolVLABackend

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--agent", choices=("left", "right"), default="left")
    parser.add_argument("--train-episode", action="append", required=True, type=Path)
    parser.add_argument("--validation-episode", action="append", default=[], type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--samples-per-phase", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--balance-phases", action="store_true")
    parser.add_argument("--final-learning-rate-ratio", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--wait-episode-seconds", type=int, default=600)
    parser.add_argument("--device", default="mps", choices=("cpu", "mps"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    adapter = load_native_adapter(args.adapter, args.agent)
    verify_base_checkpoint()
    backend = SmolVLABackend(device=args.device, denoising_steps=1)
    caches = {"training": [], "validation": []}
    for split, directories in (
        ("training", args.train_episode),
        ("validation", args.validation_episode),
    ):
        for directory in directories:
            deadline = perf_counter() + max(0, args.wait_episode_seconds)
            while not (directory / "manifest.json").exists() and perf_counter() < deadline:
                sleep(1)
            destination = args.cache_root / directory.name
            if not (destination / "manifest.json").exists():
                build_prefix_cache(
                    backend,
                    directory,
                    args.agent,
                    adapter["normalization"],
                    destination,
                    per_phase=args.samples_per_phase,
                )
            validate_cache_source(
                json.loads((destination / "manifest.json").read_text()),
                directory,
                backend,
                adapter["normalization"],
                agent=args.agent,
            )
            caches[split].append(destination)
    print(
        json.dumps(
            train_native_projections(
                backend,
                args.adapter,
                caches["training"],
                args.output,
                validation_caches=caches["validation"],
                steps=args.steps,
                learning_rate=args.learning_rate,
                horizon=args.horizon,
                batch_size=args.batch_size,
                balance_phases=args.balance_phases,
                final_learning_rate_ratio=args.final_learning_rate_ratio,
                agent=args.agent,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
