"""Fine-tune SmolVLA's existing flow output projection for this simulated rig.

The complete released vision/language backbone and action expert run at inference.
Adapters update the native output projection, optional native state/action/time
projections, and in v3 the original final expert feed-forward/norm block. There
is no alternate controller, feature-to-policy network, teacher call, or target
lookup in the live policy.
One Euler step and a fixed noise sample are an explicit training/deployment
contract, not a claim that an unadapted LIBERO checkpoint can fold cloth.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import deque
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from ..smolvla import DEFAULT_MODEL_ID, DEFAULT_REVISION
from .policies import validate_rgb

FORMAT = "relay-smolvla-native-head-v1"
GRADIENT_FORMAT = "relay-smolvla-native-projections-v2"
FINAL_FORMAT = "relay-smolvla-native-final-block-v3"
NATIVE_ACTION_DIM = 32
ACTION_DIM = 7
STATE_DIM = 8
HIDDEN_DIM = 480
CHUNK_SIZE = 50
DEFAULT_NOISE_SEED = 2718
BASE_WEIGHT_BYTES = 1_218_047_032
BASE_WEIGHT_SHA256 = "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"
NATIVE_PARAMETER_SHAPES = {
    "state_proj.weight": (960, 32),
    "state_proj.bias": (960,),
    "action_in_proj.weight": (480, 32),
    "action_in_proj.bias": (480,),
    "action_time_mlp_in.weight": (480, 960),
    "action_time_mlp_in.bias": (480,),
    "action_time_mlp_out.weight": (480, 480),
    "action_time_mlp_out.bias": (480,),
}
FINAL_PARAMETER_SHAPES = {
    **NATIVE_PARAMETER_SHAPES,
    "vlm_with_expert.lm_expert.layers.31.post_attention_layernorm.weight": (480,),
    "vlm_with_expert.lm_expert.layers.31.mlp.down_proj.weight": (480, 1280),
    "vlm_with_expert.lm_expert.layers.31.mlp.gate_proj.weight": (1280, 480),
    "vlm_with_expert.lm_expert.layers.31.mlp.up_proj.weight": (1280, 480),
    "vlm_with_expert.lm_expert.norm.weight": (480,),
}


def verify_base_checkpoint(cache_dir=None):
    """Verify the exact pinned local foundation weights before loading them."""
    cache = Path(cache_dir or os.environ.get("HF_HOME", Path.cwd() / ".cache/huggingface"))
    checkpoint = (
        cache
        / "hub"
        / "models--HuggingFaceVLA--smolvla_libero"
        / "snapshots"
        / DEFAULT_REVISION
        / "model.safetensors"
    )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if checkpoint.stat().st_size != BASE_WEIGHT_BYTES or digest.hexdigest() != BASE_WEIGHT_SHA256:
        raise ValueError("SmolVLA foundation checkpoint bytes do not match the pinned SHA-256")
    return BASE_WEIGHT_SHA256


def _finite(values, shape_tail, name):
    value = np.asarray(values, dtype=np.float32)
    if value.shape[-len(shape_tail) :] != tuple(shape_tail) or not np.isfinite(value).all():
        raise ValueError(f"{name} has an invalid shape or nonfinite values")
    return value


WORLD_EULER_CONTRACT = "world_xyz_euler_v1"
RIGHT_YAW_CONTRACT = "right_yaw_pi_v1"


def orientation_contract(normalization, *, agent=None):
    """Legacy adapters retain world Euler angles; the right rig may use a pi chart."""
    contract = normalization.get("orientation_contract", WORLD_EULER_CONTRACT)
    if contract not in (WORLD_EULER_CONTRACT, RIGHT_YAW_CONTRACT):
        raise ValueError("unrecognized native Smol orientation contract")
    if contract == RIGHT_YAW_CONTRACT and agent not in (None, "right"):
        raise ValueError("right yaw orientation contract requires the right arm")
    return contract


def referenced_orientation(values, contract=WORLD_EULER_CONTRACT):
    """Encode measured yaw around a fixed reference; XYZ and grip are untouched.

    Both the eight-dimensional proprioception and seven-dimensional action
    convention place measured Euler yaw at index 5. This changes its coordinate
    chart only. Native predicted rotations remain unexecuted by the workcell.
    """
    values = np.asarray(values, dtype=np.float32)
    if (
        values.ndim < 1
        or values.shape[-1] not in (STATE_DIM, ACTION_DIM)
        or not np.isfinite(values).all()
    ):
        raise ValueError("native orientation encoding needs finite state or action vectors")
    if contract == WORLD_EULER_CONTRACT:
        return values
    if contract != RIGHT_YAW_CONTRACT:
        raise ValueError("unrecognized native Smol orientation contract")
    result = values.copy()
    result[..., 5] = np.remainder(values[..., 5].astype(np.float64), 2 * np.pi) - np.pi
    return result


def training_normalization(states, actions, *, orientation=WORLD_EULER_CONTRACT):
    """Compute stats only from the training split supplied by the caller."""
    states = referenced_orientation(
        _finite(states, (STATE_DIM,), "training states").reshape(-1, STATE_DIM), orientation
    )
    actions = referenced_orientation(
        _finite(actions, (ACTION_DIM,), "training actions").reshape(-1, ACTION_DIM), orientation
    )
    if len(states) < 2 or len(actions) < 2:
        raise ValueError("normalization requires at least two training observations/actions")

    # Constant orientation/finger axes retain a unit scale. Tiny nonzero spreads
    # must not amplify sensor rounding into enormous model inputs.
    def stats(value):
        std = value.std(axis=0)
        return {"mean": value.mean(axis=0).tolist(), "std": np.where(std < 1e-4, 1, std).tolist()}

    result = {"state": stats(states), "action": stats(actions), "source": "training_split_only"}
    if orientation != WORLD_EULER_CONTRACT:
        result["orientation_contract"] = orientation
    return result


def validate_normalization(normalization, *, agent=None):
    if not isinstance(normalization, dict) or normalization.get("source") != "training_split_only":
        raise ValueError("adapter normalization must come from its training split")
    orientation_contract(normalization, agent=agent)
    for name, dimension in (("state", STATE_DIM), ("action", ACTION_DIM)):
        entry = normalization.get(name, {})
        for key in ("mean", "std"):
            value = _finite(entry.get(key, []), (dimension,), f"{name} {key}")
            if value.ndim != 1 or (key == "std" and np.any(value <= 0)):
                raise ValueError(f"invalid {name} normalization")
    return normalization


def normalize(values, stats):
    return (
        np.asarray(values, dtype=np.float32) - np.asarray(stats["mean"], dtype=np.float32)
    ) / np.asarray(stats["std"], dtype=np.float32)


def unnormalize(values, stats):
    return np.asarray(values, dtype=np.float32) * np.asarray(
        stats["std"], dtype=np.float32
    ) + np.asarray(stats["mean"], dtype=np.float32)


def fixed_noise(torch, device, seed=DEFAULT_NOISE_SEED):
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("invalid native flow noise seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, CHUNK_SIZE, NATIVE_ACTION_DIM), generator=generator).to(device)


def velocity_labels(actions, normalization, noise):
    """At native flow time=1, u=noise-action and one Euler step yields action."""
    validate_normalization(normalization)
    actions = _finite(actions, (CHUNK_SIZE, ACTION_DIM), "action chunks")
    noise = _finite(noise, (CHUNK_SIZE, NATIVE_ACTION_DIM), "native noise")
    padded = np.zeros((*actions.shape[:-1], NATIVE_ACTION_DIM), dtype=np.float32)
    padded[..., :ACTION_DIM] = normalize(actions, normalization["action"])
    return noise - padded


def _prepared_batch(backend, rgb, state, instruction, normalization):
    torch = backend._torch
    rgb = validate_rgb(rgb)
    state = _finite(state, (STATE_DIM,), "native state")
    if state.ndim != 1 or not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("one own-robot state and nonempty instruction are required")
    state = referenced_orientation(state, orientation_contract(normalization))
    prompt = instruction.rstrip() + "\n"
    count = len(backend._tokenizer(prompt)["input_ids"])
    if count > backend._config.tokenizer_max_length:
        raise ValueError("native cloth instruction exceeds the model token budget")
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
    batch = backend._preprocess(
        {
            "observation.images.image": image,
            "observation.images.image2": image,
            "observation.state": torch.from_numpy(state.copy()),
            "task": prompt,
        }
    )
    # Retain the official image/token processing, but replace LIBERO's state
    # statistics with the new embodiment's train-only statistics explicitly.
    batch["observation.state"] = torch.from_numpy(normalize(state, normalization["state"])).to(
        backend.device
    )[None]
    backend.last_model_prompt, backend.last_prompt_tokens = prompt, count
    return batch


def native_predict(
    backend,
    rgb,
    state,
    instruction,
    normalization,
    *,
    noise_seed=DEFAULT_NOISE_SEED,
    capture_features=False,
):
    """Run the actual native chunk API; optionally observe its projection input."""
    validate_normalization(normalization)
    if (
        backend._config.chunk_size != CHUNK_SIZE
        or backend._config.max_action_dim != NATIVE_ACTION_DIM
    ):
        raise ValueError("incompatible SmolVLA native action geometry")
    if backend._config.num_steps != 1:
        raise ValueError("this native adapter was trained for exactly one Euler step")
    torch = backend._torch
    projection = backend._model.model.action_out_proj
    if tuple(projection.weight.shape) != (NATIVE_ACTION_DIM, HIDDEN_DIM):
        raise ValueError("incompatible native output projection")
    captured = []
    hook = None
    if capture_features:
        hook = projection.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().cpu().numpy().copy())
        )
    try:
        with torch.inference_mode():
            batch = _prepared_batch(backend, rgb, state, instruction, normalization)
            prediction = (
                backend._model.predict_action_chunk(
                    batch, noise=fixed_noise(torch, backend.device, noise_seed)
                )
                .detach()
                .cpu()
                .numpy()[0]
            )
    finally:
        if hook is not None:
            hook.remove()
    prediction = _finite(prediction, (CHUNK_SIZE, ACTION_DIM), "native prediction")
    features = None
    if capture_features:
        if len(captured) != 1 or captured[0].shape != (1, CHUNK_SIZE, HIDDEN_DIM):
            raise RuntimeError("expected exactly one native output projection invocation")
        features = _finite(captured[0][0], (CHUNK_SIZE, HIDDEN_DIM), "native features")
    return prediction, features


def fit_native_head(features, velocities, *, ridge=0.01):
    """Closed-form supervised fit of the model's existing linear projection.

    Standardizing its inputs improves conditioning only while solving. Returned
    parameters act directly on the native 480D expert outputs, unchanged.
    """
    features = _finite(features, (HIDDEN_DIM,), "native features").reshape(-1, HIDDEN_DIM)
    velocities = _finite(velocities, (NATIVE_ACTION_DIM,), "flow labels").reshape(
        -1, NATIVE_ACTION_DIM
    )
    if (
        len(features) != len(velocities)
        or len(features) < 2
        or not np.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("invalid native-head training rows or ridge penalty")
    x, y = features.astype(np.float64), velocities.astype(np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1
    design = (x - mean) / scale
    y_mean = y.mean(0)
    gram = design.T @ design
    gram.flat[:: HIDDEN_DIM + 1] += ridge
    fitted = np.linalg.solve(gram, design.T @ (y - y_mean)) / scale[:, None]
    bias = y_mean - mean @ fitted
    prediction = x @ fitted + bias
    return {
        "weight": fitted.T.astype(np.float32),
        "bias": bias.astype(np.float32),
        "fit": {
            "rows": len(x),
            "ridge": float(ridge),
            "velocity_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "updated_parameter_count": NATIVE_ACTION_DIM * (HIDDEN_DIM + 1),
        },
    }


def install_native_head(backend, weight, bias):
    torch = backend._torch
    weight = _finite(weight, (NATIVE_ACTION_DIM, HIDDEN_DIM), "native head weight")
    bias = _finite(bias, (NATIVE_ACTION_DIM,), "native head bias")
    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("invalid native projection ranks")
    with torch.no_grad():
        projection = backend._model.model.action_out_proj
        projection.weight.copy_(torch.from_numpy(weight.copy()).to(projection.weight))
        projection.bias.copy_(torch.from_numpy(bias.copy()).to(projection.bias))


def save_native_adapter(
    path,
    agent,
    fitted,
    normalization,
    training,
    *,
    noise_seed=DEFAULT_NOISE_SEED,
    native_parameters=None,
):
    import torch

    if (
        agent not in ("left", "right")
        or not isinstance(training, dict)
        or noise_seed != DEFAULT_NOISE_SEED
    ):
        raise ValueError("native adapter requires an arm and training provenance")
    validate_normalization(normalization, agent=agent)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final = native_parameters is not None and set(native_parameters) == set(FINAL_PARAMETER_SHAPES)
    payload = {
        "format": FINAL_FORMAT
        if final
        else GRADIENT_FORMAT
        if native_parameters is not None
        else FORMAT,
        "agent": agent,
        "base_model_id": DEFAULT_MODEL_ID,
        "base_revision": DEFAULT_REVISION,
        "base_weight_sha256": BASE_WEIGHT_SHA256,
        "native_module": "model.action_out_proj",
        "noise_seed": noise_seed,
        "denoising_steps": 1,
        "normalization": normalization,
        "weight": torch.from_numpy(fitted["weight"]),
        "bias": torch.from_numpy(fitted["bias"]),
        "fit": fitted["fit"],
        "training": training,
        "action_contract": (
            "absolute table xyz, measured XYZ Euler with yaw relative to pi, gripper closed +1/open -1"
            if orientation_contract(normalization) == RIGHT_YAW_CONTRACT
            else "absolute table xyz, measured XYZ Euler, gripper closed +1/open -1"
        ),
    }
    if orientation_contract(normalization) != WORLD_EULER_CONTRACT:
        payload["orientation_contract"] = orientation_contract(normalization)
    if native_parameters is not None:
        payload["native_parameters"] = validate_native_parameters(
            native_parameters, include_final=final
        )
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_native_adapter(path, agent, *, expected_sha256=None):
    import torch

    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("approved SmolVLA native adapter hash does not match its bytes")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") not in (FORMAT, GRADIENT_FORMAT, FINAL_FORMAT)
        or any(
            payload.get(key) != value
            for key, value in {
                "agent": agent,
                "base_model_id": DEFAULT_MODEL_ID,
                "base_revision": DEFAULT_REVISION,
                "base_weight_sha256": BASE_WEIGHT_SHA256,
                "native_module": "model.action_out_proj",
                "denoising_steps": 1,
                "noise_seed": DEFAULT_NOISE_SEED,
            }.items()
        )
    ):
        raise ValueError("incompatible SmolVLA native adapter")
    validate_normalization(payload["normalization"], agent=agent)
    if payload.get("orientation_contract", WORLD_EULER_CONTRACT) != orientation_contract(
        payload["normalization"]
    ):
        raise ValueError("native adapter orientation contract differs from normalization")
    for name, shape in (
        ("weight", (NATIVE_ACTION_DIM, HIDDEN_DIM)),
        ("bias", (NATIVE_ACTION_DIM,)),
    ):
        tensor = payload.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid native output projection tensor: {name}")
    if payload["format"] in (GRADIENT_FORMAT, FINAL_FORMAT):
        validate_native_parameters(
            payload.get("native_parameters"), include_final=payload["format"] == FINAL_FORMAT
        )
    elif "native_parameters" in payload:
        raise ValueError("native head-only adapter cannot contain additional model parameters")
    payload["sha256"] = sha
    return payload


def validate_native_parameters(parameters, *, include_final=False):
    import torch

    shapes = FINAL_PARAMETER_SHAPES if include_final else NATIVE_PARAMETER_SHAPES
    if not isinstance(parameters, dict) or set(parameters) != set(shapes):
        raise ValueError("native projection adapter has unapproved parameter names")
    for name, shape in shapes.items():
        tensor = parameters[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid native projection tensor: {name}")
    return parameters


def install_native_parameters(backend, parameters):
    validate_native_parameters(
        parameters, include_final=set(parameters) == set(FINAL_PARAMETER_SHAPES)
    )
    with backend._torch.no_grad():
        for name, tensor in parameters.items():
            destination = backend._model.model.get_parameter(name)
            destination.copy_(tensor.to(destination))


def extract_episode_features(
    backend,
    directory,
    agent,
    normalization,
    *,
    stride=8,
    noise_seed=DEFAULT_NOISE_SEED,
    maximum_samples=None,
    on_progress=None,
    allow_rejected_for_feature_cache=False,
):
    """Observe native expert outputs on an original training-only MuJoCo episode."""
    from .action_dataset import future_chunk, load_episode, phase_instruction, smol_state
    from .policies import PHASES

    if agent not in ("left", "right") or not isinstance(stride, int) or stride < 1:
        raise ValueError("invalid native feature sampling configuration")
    manifest, episodes = load_episode(
        directory, require_success=not allow_rejected_for_feature_cache
    )
    arrays = episodes[agent]
    count = len(arrays["tick"])
    selected = [
        i
        for i in range(count)
        if not arrays["hold"][i]
        and (
            i % stride == 0
            or i == 0
            or arrays["phase"][i] != arrays["phase"][i - 1]
            or arrays["hold"][i - 1]
        )
    ]
    if maximum_samples is not None and len(selected) > maximum_samples:
        # Deterministic, evenly spread sampling includes all portions of the run.
        selected = [
            selected[i] for i in np.linspace(0, len(selected) - 1, maximum_samples, dtype=int)
        ]
    if not selected:
        raise ValueError("episode has no actionable observations")
    features, targets, noises, sample_ticks, chunk_indices, phases = [], [], [], [], [], []
    noise = fixed_noise(backend._torch, "cpu", noise_seed).numpy()[0]
    for number, index in enumerate(selected):
        state = smol_state(
            arrays["ee_position"][index],
            arrays["ee_rotation"][index],
            arrays["gripper_closed"][index],
        )
        prompt = phase_instruction(agent, PHASES[int(arrays["phase"][index])])
        _, native_features = native_predict(
            backend,
            arrays["rgb"][index],
            state,
            prompt,
            normalization,
            noise_seed=noise_seed,
            capture_features=True,
        )
        chunk = future_chunk(arrays, index, CHUNK_SIZE, family="smolvla", agent=agent)
        valid = ~np.asarray(chunk["action_is_pad"], dtype=bool)
        features.append(native_features[valid])
        targets.append(
            referenced_orientation(
                np.asarray(chunk["actions"])[valid],
                orientation_contract(normalization, agent=agent),
            )
        )
        noises.append(noise[valid])
        sample_ticks.extend([int(arrays["tick"][index])] * int(valid.sum()))
        chunk_indices.extend(np.flatnonzero(valid).tolist())
        phases.extend([int(arrays["phase"][index])] * int(valid.sum()))
        if on_progress:
            on_progress(
                {"sample": number + 1, "samples": len(selected), "tick": int(arrays["tick"][index])}
            )
    return {
        "features": np.concatenate(features),
        "actions": np.concatenate(targets),
        "noise": np.concatenate(noises),
        "ticks": np.asarray(sample_ticks),
        "chunk_indices": np.asarray(chunk_indices),
        "phases": np.asarray(phases),
        "episode_seed": manifest.get("seed"),
        "episode_manifest_sha256": hashlib.sha256(
            (Path(directory) / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def episode_normalization(
    directory, agent, *, allow_rejected_for_feature_cache=False, orientation=WORLD_EULER_CONTRACT
):
    from .action_dataset import load_episode, smol_action, smol_state

    _, episodes = load_episode(directory, require_success=not allow_rejected_for_feature_cache)
    arrays = episodes[agent]
    indices = np.flatnonzero(~arrays["hold"])
    states = [
        smol_state(arrays["ee_position"][i], arrays["ee_rotation"][i], arrays["gripper_closed"][i])
        for i in indices
    ]
    actions = [
        smol_action(arrays["target"][i], arrays["ee_rotation"][i], arrays["grip"][i])
        for i in indices
    ]
    orientation_contract({"orientation_contract": orientation}, agent=agent)
    return training_normalization(states, actions, orientation=orientation)


def prepare_right_initializer(
    source_adapter, normalization_episode, output, *, expected_sha256=None
):
    """Transfer existing native weights as initialization, never as a fitted right policy.

    Own right training statistics and its measured yaw chart change the initial
    function. No function-preservation or right-arm performance is asserted.
    The existing native head and every adapted native tensor are copied exactly.
    """
    from .action_dataset import load_episode
    from .smol_native_training import adaptation_lineage

    source = load_native_adapter(source_adapter, "left", expected_sha256=expected_sha256)
    if source["format"] != FINAL_FORMAT:
        raise ValueError("right initialization requires the approved native v3 source")
    manifest, _ = load_episode(normalization_episode)
    if manifest.get("split") != "train" or manifest["seed"] < 10000:
        raise ValueError("right normalization requires an original training episode")
    previous_validation = set(source["training"].get("validation_seeds", []))
    previous_validation.add(source["training"].get("validation_seed"))
    if manifest["seed"] in previous_validation:
        raise ValueError("right normalization cannot consume the source validation episode")
    source_hash = hashlib.sha256(
        (Path(normalization_episode) / "manifest.json").read_bytes()
    ).hexdigest()
    stats = episode_normalization(normalization_episode, "right", orientation=RIGHT_YAW_CONTRACT)
    lineage = adaptation_lineage(
        source, [{"agent": "left", "seed": manifest["seed"], "manifest_sha256": source_hash}], []
    )
    training = {
        **source["training"],
        **lineage,
        "method": "cross-arm native weight transfer initialization; right-arm fitting required",
        "cross_arm_source_agent": "left",
        "cross_arm_source_adapter_sha256": source["sha256"],
        "cross_arm_source_native_weights_unchanged": True,
        "initial_function_preserved": False,
        "normalization_training_seed": manifest["seed"],
        "normalization_manifest_sha256": source_hash,
        "normalization_agent": "right",
        "normalization_orientation_contract": RIGHT_YAW_CONTRACT,
        "historical_source_validation": source["training"].get("validation"),
        "training_prediction": {"status": "not_fitted_for_right_arm"},
        "validation": {"status": "not_evaluated_for_right_arm"},
        "output_refit_horizon": 1,
    }
    fitted = {
        "weight": source["weight"].numpy().copy(),
        "bias": source["bias"].numpy().copy(),
        "fit": {
            "method": "cross-arm native initialization only",
            "horizon": 1,
            "initialization_only": True,
        },
    }
    digest = save_native_adapter(
        output,
        "right",
        fitted,
        stats,
        training,
        native_parameters={
            key: tensor.clone() for key, tensor in source["native_parameters"].items()
        },
    )
    report = {
        "adapter_sha256": digest,
        "initialization_only": True,
        "source_adapter_sha256": source["sha256"],
        "normalization": stats,
        "training": training,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_settled_rebind(original, settled, agent):
    """Rebind only byte-identical old learner rows plus new held settling rows."""
    from .action_dataset import load_episode

    old_manifest, old_data = load_episode(original, require_success=False)
    new_manifest, new_data = load_episode(settled)
    if old_manifest["seed"] != new_manifest["seed"]:
        raise ValueError("settled continuation changed the training seed")
    old, new = old_data[agent], new_data[agent]
    length = len(old["tick"])
    if len(new["tick"]) < length or set(old) != set(new):
        raise ValueError("settled continuation changed the demonstration schema/prefix")
    if any(not np.array_equal(value, new[key][:length]) for key, value in old.items()):
        raise ValueError("settled continuation changed recorded learner inputs or labels")
    if np.any(~new["hold"][length:]):
        raise ValueError("settled continuation contains uncached new nonhold training rows")
    return hashlib.sha256((Path(settled) / "manifest.json").read_bytes()).hexdigest()


def _row_velocity_labels(rows, normalization):
    padded = np.zeros_like(rows["noise"])
    padded[:, :ACTION_DIM] = normalize(rows["actions"], normalization["action"])
    return rows["noise"] - padded


def evaluate_native_head(fitted, rows, normalization):
    from .policies import PHASES

    features = np.asarray(rows["features"], dtype=np.float64)
    velocity = features @ np.asarray(fitted["weight"], dtype=np.float64).T + fitted["bias"]
    predicted = unnormalize((rows["noise"] - velocity)[:, :ACTION_DIM], normalization["action"])
    target = np.asarray(rows["actions"])
    error = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1)

    def metrics(mask):
        if not np.any(mask):
            return {"rows": 0}
        return {
            "rows": int(np.sum(mask)),
            "xyz_rmse_m": float(np.sqrt(np.mean((predicted[mask, :3] - target[mask, :3]) ** 2))),
            "target_distance_mean_m": float(error[mask].mean()),
            "target_distance_p95_m": float(np.percentile(error[mask], 95)),
            "gripper_accuracy": float(np.mean((predicted[mask, 6] > 0) == (target[mask, 6] > 0))),
        }

    result = {
        "rows": len(error),
        "xyz_rmse_m": float(np.sqrt(np.mean((predicted[:, :3] - target[:, :3]) ** 2))),
        "target_distance_mean_m": float(error.mean()),
        "target_distance_p95_m": float(np.percentile(error, 95)),
        "gripper_accuracy": float(np.mean((predicted[:, 6] > 0) == (target[:, 6] > 0))),
        "scope": "held-out demonstration prediction; not closed-loop task success",
    }
    if "chunk_indices" in rows:
        execute = np.asarray(rows["chunk_indices"]) < 8
        result["executed_indices_0_to_7"] = metrics(execute)
        result["per_phase_executed"] = {
            phase: metrics(execute & (np.asarray(rows["phases"]) == index))
            for index, phase in enumerate(PHASES)
        }
    return result


def refit_cached_adapter(
    source_adapter,
    output,
    *,
    fit_horizon=8,
    ridge=0.01,
    include_validation_as_training=False,
    reserved_validation_seed=None,
):
    """Refit the same native projection without re-running frozen GPU forwards."""
    source_adapter, output = Path(source_adapter), Path(output)
    if not 1 <= fit_horizon <= CHUNK_SIZE or output.exists():
        raise ValueError("invalid fit horizon or existing adapter output")
    report = json.loads(source_adapter.with_suffix(".json").read_text())
    cache = source_adapter.with_suffix(".features.npz")
    if hashlib.sha256(cache.read_bytes()).hexdigest() != report["feature_cache_sha256"]:
        raise ValueError("native feature cache hash does not match extraction provenance")
    import torch

    raw = torch.load(io.BytesIO(source_adapter.read_bytes()), map_location="cpu", weights_only=True)
    adapter = load_native_adapter(
        source_adapter, raw.get("agent"), expected_sha256=report["adapter_sha256"]
    )
    with np.load(cache, allow_pickle=False) as archive:
        rows = {
            split: {
                key.removeprefix(split + "_"): archive[key]
                for key in archive.files
                if key.startswith(split + "_")
            }
            for split in ("training", "validation")
        }
    previous_validation = adapter["training"]["validation"]
    if include_validation_as_training:
        if reserved_validation_seed in (
            None,
            adapter["training"]["train_seed"],
            adapter["training"]["validation_seed"],
        ):
            raise ValueError("reclassification requires a new distinct reserved validation seed")
        rows["training"] = {
            key: np.concatenate([rows["training"][key], rows["validation"][key]])
            for key in rows["training"]
        }
    mask = rows["training"]["chunk_indices"] < fit_horizon
    fitted = fit_native_head(
        rows["training"]["features"][mask],
        _row_velocity_labels(rows["training"], adapter["normalization"])[mask],
        ridge=ridge,
    )
    validation = evaluate_native_head(fitted, rows["validation"], adapter["normalization"])
    training = {
        **adapter["training"],
        "fit_horizon": fit_horizon,
        "validation": validation,
        "refit_source_adapter_sha256": adapter["sha256"],
        "feature_cache_sha256": report["feature_cache_sha256"],
    }
    if include_validation_as_training:
        train_prediction = evaluate_native_head(fitted, rows["training"], adapter["normalization"])
        train_prediction["scope"] = (
            "in-sample training prediction; not held-out or closed-loop success"
        )
        training.update(
            {
                "train_seeds": [
                    adapter["training"]["train_seed"],
                    adapter["training"]["validation_seed"],
                ],
                "train_manifest_sha256s": [
                    adapter["training"]["train_manifest_sha256"],
                    adapter["training"]["validation_manifest_sha256"],
                ],
                "validation_seed": reserved_validation_seed,
                "validation_manifest_sha256": None,
                "validation": {
                    "status": "pending_new_heldout_episode",
                    "reserved_seed": reserved_validation_seed,
                },
                "reclassified_old_validation_seed": adapter["training"]["validation_seed"],
                "historical_validation_before_reclassification": previous_validation,
                "normalization_training_seed": adapter["training"]["train_seed"],
                "training_prediction": train_prediction,
            }
        )
    digest = save_native_adapter(
        output,
        adapter["agent"],
        fitted,
        adapter["normalization"],
        training,
        noise_seed=adapter["noise_seed"],
    )
    result = {
        "adapter_sha256": digest,
        "format": FORMAT,
        "fit": fitted["fit"],
        "training": training,
        "feature_cache_sha256": report["feature_cache_sha256"],
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episode", required=True, type=Path)
    parser.add_argument("--validation-episode", required=True, type=Path)
    parser.add_argument("--agent", choices=("left", "right"), default="left")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--fit-horizon", type=int, default=CHUNK_SIZE)
    parser.add_argument("--wait-validation-seconds", type=int, default=1200)
    parser.add_argument(
        "--preextract-episode",
        type=Path,
        help="Cache forwards from a rejected prefix; adapter publication requires an exact successful continuation.",
    )
    args = parser.parse_args(argv)
    if args.train_episode.resolve() == args.validation_episode.resolve():
        raise ValueError("training and validation episodes must be separate")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.fit_horizon <= CHUNK_SIZE:
        raise ValueError("fit horizon must be between 1 and 50")
    from ..smolvla import SmolVLABackend

    feature_source = args.preextract_episode or args.train_episode
    normalization = episode_normalization(
        feature_source,
        args.agent,
        allow_rejected_for_feature_cache=args.preextract_episode is not None,
    )
    verify_base_checkpoint()
    backend = SmolVLABackend(device=args.device, denoising_steps=1)
    print(json.dumps({"event": "native_model_loaded", "model": backend.model_info}), flush=True)
    rows = {}
    for split, directory in (
        ("training", feature_source),
        ("validation", args.validation_episode),
    ):
        if split == "validation":
            deadline = perf_counter() + max(0, args.wait_validation_seconds)
            while not (directory / "manifest.json").exists() and perf_counter() < deadline:
                sleep(1)

        def progress(value, split=split):
            print(json.dumps({"split": split, **value}), flush=True)

        rows[split] = extract_episode_features(
            backend,
            directory,
            args.agent,
            normalization,
            stride=args.stride,
            maximum_samples=args.maximum_samples,
            on_progress=progress,
            allow_rejected_for_feature_cache=split == "training"
            and args.preextract_episode is not None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        split_cache = args.output.with_suffix(f".{split}.features.npz")
        np.savez_compressed(
            split_cache,
            **{
                name: rows[split][name]
                for name in ("features", "actions", "noise", "ticks", "chunk_indices", "phases")
            },
        )
        print(
            json.dumps(
                {
                    "event": "features_saved",
                    "split": split,
                    "path": str(split_cache),
                    "memory": backend.model_info["memory"],
                }
            ),
            flush=True,
        )
        if split == "training" and args.preextract_episode is not None:
            deadline = perf_counter() + max(0, args.wait_validation_seconds)
            while not (args.train_episode / "manifest.json").exists() and perf_counter() < deadline:
                sleep(1)
            rows[split]["feature_source_manifest_sha256"] = rows[split]["episode_manifest_sha256"]
            rows[split]["episode_manifest_sha256"] = verify_settled_rebind(
                args.preextract_episode, args.train_episode, args.agent
            )
    if rows["training"]["episode_seed"] == rows["validation"]["episode_seed"]:
        raise ValueError("training and validation seeds must differ")
    fitting = rows["training"]["chunk_indices"] < args.fit_horizon
    fitted = fit_native_head(
        rows["training"]["features"][fitting],
        _row_velocity_labels(rows["training"], normalization)[fitting],
        ridge=args.ridge,
    )
    validation = evaluate_native_head(fitted, rows["validation"], normalization)
    training = {
        "method": "ridge regression of existing native action_out_proj on frozen action-expert outputs",
        "source": "original MuJoCo demonstrations; teacher only in offline collection",
        "train_manifest_sha256": rows["training"]["episode_manifest_sha256"],
        "validation_manifest_sha256": rows["validation"]["episode_manifest_sha256"],
        "train_seed": rows["training"]["episode_seed"],
        "validation_seed": rows["validation"]["episode_seed"],
        "validation": validation,
        "stride": args.stride,
        "fit_horizon": args.fit_horizon,
        "feature_source_manifest_sha256": rows["training"].get("feature_source_manifest_sha256"),
        "successful_prefix_rebind": args.preextract_episode is not None,
    }
    digest = save_native_adapter(args.output, args.agent, fitted, normalization, training)
    cache_path = args.output.with_suffix(".features.npz")
    np.savez_compressed(
        cache_path,
        **{
            f"{split}_{name}": rows[split][name]
            for split in rows
            for name in ("features", "actions", "noise", "ticks", "chunk_indices", "phases")
        },
    )
    report = {
        "adapter_sha256": digest,
        "format": FORMAT,
        "fit": fitted["fit"],
        "training": training,
        "feature_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


class SmolNativePolicy:
    """An independently adapted native SmolVLA arm with its own action queue."""

    def __init__(
        self,
        agent,
        adapter_path,
        *,
        device="mps",
        expected_sha256=None,
        action_horizon=8,
        action_repeat=1,
        backend=None,
        allow_initializer=False,
    ):
        from ..smolvla import SmolVLABackend

        if (
            agent not in ("left", "right")
            or not isinstance(action_horizon, int)
            or not 1 <= action_horizon <= CHUNK_SIZE
            or not isinstance(action_repeat, int)
            or not 1 <= action_repeat <= CHUNK_SIZE
            or (action_repeat > 1 and action_horizon != 1)
        ):
            raise ValueError("invalid native SmolVLA arm or action horizon")
        self.agent, self.action_horizon = agent, action_horizon
        self.action_repeat = action_repeat
        self.adapter = load_native_adapter(adapter_path, agent, expected_sha256=expected_sha256)
        if self.adapter["fit"].get("initialization_only", False) and not allow_initializer:
            raise ValueError("cross-arm initialization is not a fitted deployment policy")
        trained_horizon = self.adapter["training"].get(
            "output_refit_horizon", self.adapter["fit"].get("horizon", CHUNK_SIZE)
        )
        if action_horizon > trained_horizon:
            raise ValueError(
                "requested action horizon exceeds the native adapter's supervised horizon"
            )
        verify_base_checkpoint(getattr(backend, "cache_dir", None))
        self.backend = backend or SmolVLABackend(device=device, denoising_steps=1)
        if self.backend.model_id != DEFAULT_MODEL_ID or self.backend.revision != DEFAULT_REVISION:
            raise ValueError("native adapter base model does not match the loaded model")
        install_native_head(
            self.backend, self.adapter["weight"].numpy(), self.adapter["bias"].numpy()
        )
        if self.adapter["format"] in (GRADIENT_FORMAT, FINAL_FORMAT):
            install_native_parameters(self.backend, self.adapter["native_parameters"])
        self.calls, self.seconds, self._queue, self._instruction = 0, 0.0, deque(), None
        self._last_inference_seconds = 0.0
        self.last_prediction = {}

    @property
    def model_info(self):
        return {
            **self.backend.model_info,
            "policy_class": "foundation VLA with fine-tuned native flow projections",
            "foundation_vla": True,
            "native_action_head_executed": self.calls > 0,
            "native_action_head_adapted": True,
            "agent": self.agent,
            "adapter_format": self.adapter["format"],
            "adapted_native_parameters": [
                "action_out_proj.weight",
                "action_out_proj.bias",
                *self.adapter.get("native_parameters", {}),
            ],
            "checkpoint_sha256": self.adapter["sha256"],
            "native_module": "model.action_out_proj",
            "native_denoising_steps": 1,
            "base_weight_sha256": BASE_WEIGHT_SHA256,
            "frame_translation_metres": {self.agent: [0.0, 0.0, 0.0]},
            "action_translation_scale_metres": None,
            "max_delta_norm_metres": None,
            "noise_stream": f"Fixed native Gaussian sample, seed {self.adapter['noise_seed']}; identical training and inference.",
            "adaptation": self.adapter["fit"],
            "training": self.adapter["training"],
            "action_horizon": self.action_horizon,
            "action_reuse": "first_target" if self.action_repeat > 1 else "native_chunk",
            "action_repeat_steps": self.action_repeat,
            "action_contract": self.adapter["action_contract"],
            "orientation_contract": orientation_contract(self.adapter["normalization"]),
            "inference_calls": self.calls,
            "inference_seconds": self.seconds,
            "total_inference_seconds": self.seconds,
            "last_inference_seconds": getattr(self, "_last_inference_seconds", 0.0),
            "inputs": [
                "local RGB duplicated into two camera slots",
                "own measured 8D proprioception",
                "phase language",
            ],
            "limitations": [
                "Task adaptation changes native flow projections and the original final expert feed-forward/norm layers; the foundation vision/language backbone and earlier action-expert layers remain frozen."
                if self.adapter["format"] == FINAL_FORMAT
                else "Task adaptation changes native flow projections; the foundation vision/language backbone and transformer action expert remain frozen.",
                "One fixed-noise Euler step differs from the released LIBERO inference configuration.",
                "Task-specific simulation adaptation; real garment or physical robot transfer is unvalidated.",
                "Tool rotation is retained in the native output audit but the simulator uses fixed-orientation grippers.",
            ],
        }

    def act(self, observation, instruction, clearance=0.20, hold=False):
        from .action_dataset import phase_instruction, smol_state

        if observation.get("agent") != self.agent:
            raise ValueError("native SmolVLA received another arm's observation")
        _finite(observation.get("ee_position"), (3,), "own tool position")
        ee = np.asarray(observation["ee_position"], dtype=float)
        if hold:
            self._queue.clear()
            self.last_prediction = {
                "source": "coordination_safety_hold",
                "native_action_executed": False,
                "coordinator_hold": True,
                "inference_id": self.calls,
                "inference_age_ticks": 0,
            }
            return {
                "target": ee.tolist(),
                "grip": bool(observation.get("gripper_closed", observation.get("grasped"))),
            }
        prompt = phase_instruction(self.agent, instruction)
        if prompt != self._instruction:
            self._queue.clear()
            self._instruction = prompt
        if not self._queue:
            state = smol_state(
                observation["ee_position"],
                observation["ee_rotation"],
                observation["gripper_closed"],
            )
            started = perf_counter()
            normalized, _ = native_predict(
                self.backend,
                observation["rgb"],
                state,
                prompt,
                self.adapter["normalization"],
                noise_seed=self.adapter["noise_seed"],
            )
            actions = unnormalize(normalized, self.adapter["normalization"]["action"])
            self._last_inference_seconds = perf_counter() - started
            self.seconds += self._last_inference_seconds
            self.calls += 1
            repeat = getattr(self, "action_repeat", 1)
            if repeat > 1:
                self._queue.extend((age, actions[0].copy()) for age in range(repeat))
            else:
                self._queue.extend(
                    (i, action.copy()) for i, action in enumerate(actions[: self.action_horizon])
                )
            self.last_prediction = {
                "source": "native_smolvla_flow",
                "native_action_executed": True,
                "coordinator_hold": False,
                "inference_id": self.calls,
                "raw_chunk": normalized.tolist(),
                "normalized_native_chunk": normalized.tolist(),
                "world_action_chunk": actions.tolist(),
                "rgb_sha256": hashlib.sha256(
                    validate_rgb(observation["rgb"]).tobytes()
                ).hexdigest(),
                "model_language_prompt": self.backend.last_model_prompt,
                "language_tokens": self.backend.last_prompt_tokens,
                "native_module": "model.action_out_proj",
                "noise_seed": self.adapter["noise_seed"],
                "orientation_contract": orientation_contract(self.adapter["normalization"]),
            }
        age, action = self._queue.popleft()
        index = 0 if getattr(self, "action_repeat", 1) > 1 else age
        if not np.isfinite(action).all():
            raise RuntimeError("native SmolVLA generated a nonfinite command")
        # This is an explicit workspace boundary; no landmark/teacher modifies
        # the predicted target. The MuJoCo controller separately enforces speed.
        target = np.clip(action[:3], [-0.59, -0.49, 0.008], [0.59, 0.42, 0.52])
        self.last_prediction.update(
            {
                "used_index": index,
                "inference_age_ticks": age,
                "raw_action": action.tolist(),
                "ignored_rotation": action[3:6].tolist(),
                "workspace_clipped": bool(np.any(target != action[:3])),
            }
        )
        return {"target": target.tolist(), "grip": bool(action[6] > 0)}


if __name__ == "__main__":
    main()
