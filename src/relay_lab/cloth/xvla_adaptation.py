"""Inspectable adaptation of X-VLA's *existing* native action modules.

Florence-2 and the first 23 native action-transformer blocks stay frozen. The
compact baseline fits four columns of the original domain-5 ``action_decoder``.
Supported deeper variants adapt native input projections/soft prompts or the
original final transformer block; every additional tensor is explicitly bound
in the artifact manifest. No
auxiliary controller or teacher is evaluated at deployment. Native generation
uses ``predict_action_chunk`` at t=1, with one deterministic denoising step.

This is supervised task adaptation, not zero-shot transfer or a claim that the
unadapted folding checkpoint supports our camera and robot embodiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .policies import PHASES, validate_rgb
from .pretrained import (
    XVLA_DOMAIN_ID,
    XVLA_ID,
    XVLA_REVISION,
    XVLA_WEIGHT_SHA256,
    XVLAClothProbe,
    _tokenization_evidence,
)

FORMAT = "relay-xvla-native-decoder-v1"
NOISE_SEED = 712
CHANNELS = (10, 11, 12, 19)
FEATURE_WIDTH = 1024
GRIP_LOGIT = 5.0
ACTION_HORIZON = 8


def native_contract() -> dict:
    return {
        "format": FORMAT,
        "model_id": XVLA_ID,
        "revision": XVLA_REVISION,
        "base_sha256": XVLA_WEIGHT_SHA256,
        "agent": "right",
        "domain_id": XVLA_DOMAIN_ID,
        "channels": list(CHANNELS),
        "feature_width": FEATURE_WIDTH,
        "noise_seed": NOISE_SEED,
        "denoising_steps": 1,
        "executed_chunk_index": 0,
        "default_action_refresh_ticks": ACTION_HORIZON,
        "action_frame": "absolute Cartesian xyz in MuJoCo table/world metres",
        "action_postprocessor": "identity, checked against native output for every inference",
        "native_layer": "model.transformer.action_decoder domain 5",
        "state_contract": "own measured xyz, rotation first two matrix columns, closed gripper; peer zeros",
        "native_gripper_preprocessing": "official EE6DActionSpace masks gripper channels in proprio and noisy actions before the transformer",
    }


def measured_local_state(observation: dict) -> np.ndarray:
    """Measured local pose only; an absent rotation is an error, not a guess."""
    if observation.get("agent") != "right":
        raise ValueError("X-VLA adaptation is pinned to the right arm")
    xyz = np.asarray(observation.get("ee_position"), dtype=np.float32)
    rotation = np.asarray(observation.get("ee_rotation"), dtype=np.float32)
    if (
        xyz.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(xyz).all()
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
        or not np.isclose(np.linalg.det(rotation), 1, atol=2e-3)
    ):
        raise ValueError("X-VLA requires measured finite xyz and a proper tool rotation")
    state = np.zeros(20, dtype=np.float32)
    state[10:13] = xyz
    state[13:19] = rotation[:, :2].reshape(6)
    state[19] = float(bool(observation.get("gripper_closed", observation.get("grasped"))))
    return state


def supervised_targets(target_xyz, grip) -> np.ndarray:
    """Native t=1 endpoint prediction: absolute position and pre-sigmoid grip."""
    xyz = np.asarray(target_xyz, dtype=np.float64)
    closed = np.asarray(grip)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or closed.shape != (len(xyz),)
        or not np.isfinite(xyz).all()
        or not np.isin(closed, [0, 1]).all()
    ):
        raise ValueError("Expected finite N x 3 native action targets and binary grips")
    return np.column_stack([xyz, (closed.astype(np.float64) * 2 - 1) * GRIP_LOGIT])


def fit_native_columns(
    features, targets, original_weight, original_bias, *, ridge=0.01, standardize_features=False
):
    """Fit a regularized update to existing decoder columns, preserving the rest.

    X-VLA predicts clean endpoints, rather than velocity, in its EE6D action
    space. XYZ uses squared endpoint error; the grip is fitted in logit space
    and receives X-VLA's own sigmoid at generation. Ridge is applied to the
    change from publisher weights; the bias is unpenalized. All solves use
    float64 on CPU; saved tensors retain the original float32 parameter dtype.
    """
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    w = np.asarray(original_weight, dtype=np.float64)
    b = np.asarray(original_bias, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != FEATURE_WIDTH
        or len(x) < 2
        or y.shape != (len(x), len(CHANNELS))
        or w.shape != (FEATURE_WIDTH, 20)
        or b.shape != (20,)
        or not all(np.isfinite(item).all() for item in (x, y, w, b))
        or not np.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("Invalid native decoder training shapes, values, or regularization")
    mean_x = x.mean(0)
    residual = y - (x @ w[:, CHANNELS] + b[list(CHANNELS)])
    mean_residual = residual.mean(0)
    centered_x, centered_y = x - mean_x, residual - mean_residual
    scale = np.ones(FEATURE_WIDTH)
    if standardize_features:
        scale = x.std(0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        centered_x = centered_x / scale
    if len(x) <= FEATURE_WIDTH:
        gram = centered_x @ centered_x.T
        gram.flat[:: len(x) + 1] += ridge
        delta = centered_x.T @ np.linalg.solve(gram, centered_y)
    else:
        gram = centered_x.T @ centered_x
        gram.flat[:: FEATURE_WIDTH + 1] += ridge
        delta = np.linalg.solve(gram, centered_x.T @ centered_y)
    # Feature scaling is an offline solve preconditioner only. Fold it into
    # the existing native projection; runtime receives no additional layer.
    delta = delta / scale[:, None]
    result_w, result_b = w.copy(), b.copy()
    result_w[:, CHANNELS] += delta
    result_b[list(CHANNELS)] += mean_residual - mean_x @ delta
    return result_w.astype(np.float32), result_b.astype(np.float32)


def decoder_metrics(features, targets, weight, bias) -> dict:
    predicted = (
        np.asarray(features) @ np.asarray(weight)[:, CHANNELS] + np.asarray(bias)[list(CHANNELS)]
    )
    targets = np.asarray(targets)
    errors = np.linalg.norm(predicted[:, :3] - targets[:, :3], axis=1)
    return {
        "samples": len(errors),
        "xyz_mae_metres": float(np.abs(predicted[:, :3] - targets[:, :3]).mean()),
        "xyz_norm_mean_metres": float(errors.mean()),
        "xyz_norm_p95_metres": float(np.percentile(errors, 95)),
        "grip_accuracy": float(np.mean((predicted[:, 3] > 0) == (targets[:, 3] > 0))),
    }


def save_adapter(
    directory,
    weight,
    bias,
    *,
    dataset_sha256,
    training: dict,
    native_inputs=None,
    native_block=None,
    native_norm=None,
    native_multimodal=None,
) -> dict:
    """Save tensors without pickle and bind all metadata to immutable hashes."""
    import torch
    from safetensors.torch import save_file

    if len(dataset_sha256) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha256):
        raise ValueError("Dataset must have an immutable SHA-256 identity")
    weight, bias = np.asarray(weight), np.asarray(bias)
    if weight.shape != (FEATURE_WIDTH, 20) or bias.shape != (20,):
        raise ValueError("Wrong native decoder parameter shape")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Nonfinite native decoder parameters")
    if native_inputs is not None:
        _validate_native_inputs(native_inputs)
    if native_block is not None:
        _validate_native_block(native_block)
    if native_norm is not None:
        _validate_native_norm(native_norm)
    if native_multimodal is not None:
        if native_inputs is not None:
            raise ValueError("Native multimodal and input adapters cannot duplicate soft prompts")
        _validate_native_multimodal(native_multimodal)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "native_decoder.safetensors"
    tensors = {
        "action_decoder.weight": torch.from_numpy(weight.astype(np.float32)),
        "action_decoder.bias": torch.from_numpy(bias.astype(np.float32)),
    }
    if native_inputs is not None:
        tensors.update(
            {
                name: torch.from_numpy(np.asarray(value, dtype=np.float32))
                for name, value in native_inputs.items()
            }
        )
    if native_block is not None:
        tensors.update({name: torch.from_numpy(value) for name, value in native_block.items()})
    if native_norm is not None:
        tensors.update({name: torch.from_numpy(value) for name, value in native_norm.items()})
    if native_multimodal is not None:
        tensors.update({name: torch.from_numpy(value) for name, value in native_multimodal.items()})
    save_file(tensors, str(path))
    metadata = {
        **native_contract(),
        "decoder_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_sha256": dataset_sha256,
        "training": training,
        "trainable_parameters": (FEATURE_WIDTH + 1) * len(CHANNELS)
        + (107520 if native_inputs is not None else 0)
        + (sum(value.size for value in native_block.values()) if native_block is not None else 0)
        + (2048 if native_norm is not None else 0)
        + (1082368 if native_multimodal is not None else 0),
        "untouched_output_channels": [i for i in range(20) if i not in CHANNELS],
        "provenance": "Original Relay MuJoCo training observations and supervised action labels; no external demonstration dataset",
    }
    if native_inputs is not None:
        metadata["adapted_native_inputs"] = ["action_encoder", "soft_prompt_hub"]
    if native_block is not None:
        metadata["adapted_native_blocks"] = [23]
        metadata["untouched_output_channels_scope"] = (
            "Decoder parameter columns only; adapted native block can change all native predictions. Only right XYZ/grip are executed."
        )
    if native_norm is not None:
        metadata["adapted_native_final_norm"] = True
    if native_multimodal is not None:
        metadata["adapted_native_multimodal"] = ["vlm_proj", "soft_prompt_hub"]
    metadata["stored_native_parameters"] = sum(value.numel() for value in tensors.values())
    effective = training.get("effective_trainable_parameters")
    if effective is not None:
        if type(effective) is not int or not 0 < effective <= metadata["stored_native_parameters"]:
            raise ValueError("Invalid effective native trainable parameter count")
        metadata["trainable_parameters"] = effective
    (directory / "adapter.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _validate_native_inputs(tensors):
    shapes = {
        "action_encoder.weight": (72, FEATURE_WIDTH),
        "action_encoder.bias": (FEATURE_WIDTH,),
        "soft_prompt_hub.weight": (32 * FEATURE_WIDTH,),
    }
    if set(tensors) != set(shapes):
        raise ValueError("Unexpected native input adaptation tensors")
    for name, shape in shapes.items():
        value = np.asarray(tensors[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid native input adaptation tensor shape or values")


def native_block_shapes():
    """Exact publisher block-23 parameters; no replacement modules are allowed."""
    return {
        f"blocks.23.{name}": shape
        for name, shape in {
            "norm1.weight": (1024,),
            "norm1.bias": (1024,),
            "norm2.weight": (1024,),
            "norm2.bias": (1024,),
            "attn.qkv.weight": (3072, 1024),
            "attn.qkv.bias": (3072,),
            "attn.proj.weight": (1024, 1024),
            "attn.proj.bias": (1024,),
            "mlp.fc1.weight": (4096, 1024),
            "mlp.fc1.bias": (4096,),
            "mlp.fc2.weight": (1024, 4096),
            "mlp.fc2.bias": (1024,),
        }.items()
    }


def _validate_native_block(tensors):
    shapes = native_block_shapes()
    if set(tensors) != set(shapes):
        raise ValueError("Unexpected native final-block adaptation tensors")
    for name, shape in shapes.items():
        value = np.asarray(tensors[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid native final-block tensor shape or values")


def _validate_native_norm(tensors):
    if set(tensors) != {"norm.weight", "norm.bias"}:
        raise ValueError("Unexpected native final LayerNorm tensors")
    for value in tensors.values():
        value = np.asarray(value)
        if value.shape != (1024,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid native final LayerNorm tensor")


def _validate_native_multimodal(tensors):
    shapes = {
        "vlm_proj.weight": (1024, 1024),
        "vlm_proj.bias": (1024,),
        "soft_prompt_hub.weight": (32768,),
    }
    if set(tensors) != set(shapes):
        raise ValueError("Unexpected native multimodal adaptation tensors")
    for name, shape in shapes.items():
        value = np.asarray(tensors[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid native multimodal tensor")


def _read_adapter_tensors(directory, *, expected_sha256=None):
    from safetensors.numpy import load

    directory = Path(directory)
    metadata = json.loads((directory / "adapter.json").read_text())
    if any(metadata.get(key) != value for key, value in native_contract().items()):
        raise ValueError("Incompatible X-VLA native adaptation contract")
    path = directory / "native_decoder.safetensors"
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != metadata.get("decoder_sha256") or (expected_sha256 and digest != expected_sha256):
        raise ValueError("X-VLA native decoder hash mismatch")
    tensors = load(contents)
    expected_keys = {"action_decoder.weight", "action_decoder.bias"}
    if "adapted_native_inputs" in metadata:
        if metadata["adapted_native_inputs"] != ["action_encoder", "soft_prompt_hub"]:
            raise ValueError("Unsupported native input adaptation modules")
        native_inputs = {
            name: value
            for name, value in tensors.items()
            if name.startswith(("action_encoder.", "soft_prompt_hub."))
        }
        _validate_native_inputs(native_inputs)
        expected_keys.update(native_inputs)
    if "adapted_native_blocks" in metadata:
        if metadata["adapted_native_blocks"] != [23]:
            raise ValueError("Unsupported native transformer adaptation blocks")
        native_block = {
            name: value for name, value in tensors.items() if name.startswith("blocks.")
        }
        _validate_native_block(native_block)
        expected_keys.update(native_block)
    if "adapted_native_final_norm" in metadata:
        if metadata["adapted_native_final_norm"] is not True:
            raise ValueError("Unsupported native final LayerNorm adaptation")
        norm = {name: value for name, value in tensors.items() if name.startswith("norm.")}
        _validate_native_norm(norm)
        expected_keys.update(norm)
    if "adapted_native_multimodal" in metadata:
        if (
            metadata["adapted_native_multimodal"] != ["vlm_proj", "soft_prompt_hub"]
            or "adapted_native_inputs" in metadata
        ):
            raise ValueError("Unsupported native multimodal adaptation")
        multimodal = {
            name: value
            for name, value in tensors.items()
            if name.startswith(("vlm_proj.", "soft_prompt_hub."))
        }
        _validate_native_multimodal(multimodal)
        expected_keys.update(multimodal)
    if set(tensors) != expected_keys:
        raise ValueError("Unexpected adapted tensors")
    weight, bias = tensors["action_decoder.weight"], tensors["action_decoder.bias"]
    if weight.shape != (FEATURE_WIDTH, 20) or bias.shape != (20,):
        raise ValueError("Wrong adapted decoder shape")
    if weight.dtype != np.float32 or bias.dtype != np.float32:
        raise ValueError("Adapted native tensors must retain float32 precision")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Nonfinite adapted decoder")
    return metadata, tensors


def load_adapter(directory, *, expected_sha256=None):
    metadata, tensors = _read_adapter_tensors(directory, expected_sha256=expected_sha256)
    return metadata, tensors["action_decoder.weight"], tensors["action_decoder.bias"]


class NativeXVLA:
    """Execute the full native X-VLA path with strictly pinned native adaptation."""

    def __init__(
        self,
        *,
        device="cpu",
        adapter=None,
        expected_sha256=None,
        probe=None,
        action_horizon=ACTION_HORIZON,
    ):
        if type(action_horizon) is not int or not 1 <= action_horizon <= 50:
            raise ValueError("Native action horizon must be an integer between 1 and 50")
        self.action_horizon = action_horizon
        self.probe = probe or XVLAClothProbe("right", device, NOISE_SEED, denoising_steps=1)
        self.agent = "right"
        info = self.probe.model_info
        if (
            info.get("model_id") != XVLA_ID
            or info.get("revision") != XVLA_REVISION
            or info.get("checkpoint_sha256") != XVLA_WEIGHT_SHA256
            or self.probe.agent != self.agent
        ):
            raise ValueError("Adaptation requires the verified pinned right-arm X-VLA")
        self.probe.model.config.num_denoising_steps = 1
        self.probe.model.model.config.num_denoising_steps = 1
        self.probe.model.eval().requires_grad_(False)
        self.adapter = None
        self.last_prediction = {}
        self._cached = None
        if adapter is not None:
            self.adapter, tensors = _read_adapter_tensors(adapter, expected_sha256=expected_sha256)
            weight, bias = tensors["action_decoder.weight"], tensors["action_decoder.bias"]
            trained_device = self.adapter.get("training", {}).get("noise_device")
            if trained_device is not None and trained_device != self.probe.device:
                raise ValueError("Native noise backend must match the adapted feature cache")
            decoder = self.probe.model.model.transformer.action_decoder
            original_w, original_b = self.decoder_parameters()
            unused = [i for i in range(20) if i not in CHANNELS]
            if not np.array_equal(original_w[:, unused], weight[:, unused]) or not np.array_equal(
                original_b[unused], bias[unused]
            ):
                raise ValueError("Adapter modified unapproved native output channels")
            torch = self.probe.torch
            with torch.no_grad():
                decoder.fc.weight[XVLA_DOMAIN_ID].copy_(
                    torch.as_tensor(weight.reshape(-1), device=self.probe.device)
                )
                decoder.bias.weight[XVLA_DOMAIN_ID].copy_(
                    torch.as_tensor(bias, device=self.probe.device)
                )
                if self.adapter.get("adapted_native_inputs"):
                    transformer = self.probe.model.model.transformer
                    for source, parameter in (
                        ("action_encoder.weight", transformer.action_encoder.fc.weight),
                        ("action_encoder.bias", transformer.action_encoder.bias.weight),
                        ("soft_prompt_hub.weight", transformer.soft_prompt_hub.weight),
                    ):
                        parameter[XVLA_DOMAIN_ID].copy_(
                            torch.as_tensor(tensors[source].reshape(-1), device=self.probe.device)
                        )
                if self.adapter.get("adapted_native_blocks"):
                    block = self.probe.model.model.transformer.blocks[23]
                    block.load_state_dict(
                        {
                            name.removeprefix("blocks.23."): torch.as_tensor(
                                value, device=self.probe.device
                            )
                            for name, value in tensors.items()
                            if name.startswith("blocks.23.")
                        },
                        strict=True,
                    )
                if self.adapter.get("adapted_native_final_norm"):
                    self.probe.model.model.transformer.norm.load_state_dict(
                        {
                            name.removeprefix("norm."): torch.as_tensor(
                                value, device=self.probe.device
                            )
                            for name, value in tensors.items()
                            if name.startswith("norm.")
                        },
                        strict=True,
                    )
                if self.adapter.get("adapted_native_multimodal"):
                    transformer = self.probe.model.model.transformer
                    transformer.vlm_proj.load_state_dict(
                        {
                            name.removeprefix("vlm_proj."): torch.as_tensor(
                                value, device=self.probe.device
                            )
                            for name, value in tensors.items()
                            if name.startswith("vlm_proj.")
                        },
                        strict=True,
                    )
                    transformer.soft_prompt_hub.weight[XVLA_DOMAIN_ID].copy_(
                        torch.as_tensor(tensors["soft_prompt_hub.weight"], device=self.probe.device)
                    )

    def decoder_parameters(self):
        decoder = self.probe.model.model.transformer.action_decoder
        return (
            decoder.fc.weight[XVLA_DOMAIN_ID]
            .detach()
            .cpu()
            .numpy()
            .reshape(FEATURE_WIDTH, 20)
            .copy(),
            decoder.bias.weight[XVLA_DOMAIN_ID].detach().cpu().numpy().copy(),
        )

    @property
    def model_info(self):
        return {
            **self.probe.model_info,
            "foundation_vla": True,
            "native_action_head_executed": self.probe.calls > 0,
            "native_action_head_adapted": self.adapter is not None,
            "base_checkpoint_sha256": XVLA_WEIGHT_SHA256,
            "adapter_sha256": self.adapter["decoder_sha256"] if self.adapter else None,
            "adapted_native_inputs": self.adapter.get("adapted_native_inputs", [])
            if self.adapter
            else [],
            "adapted_native_blocks": self.adapter.get("adapted_native_blocks", [])
            if self.adapter
            else [],
            "action_horizon": self.action_horizon,
            "adapted_native_final_norm": bool(
                self.adapter and self.adapter.get("adapted_native_final_norm")
            ),
            "adapted_native_multimodal": self.adapter.get("adapted_native_multimodal", [])
            if self.adapter
            else [],
            "rotation_order": "Measured local-to-world tool matrix, first two columns flattened row-wise; native predicted rotation is not executed by the fixed-orientation rig.",
            "policy_class": (
                "pretrained X-VLA with adapted native multimodal projection, soft prompts, final norm and decoder"
                if self.adapter and self.adapter.get("adapted_native_multimodal")
                else "pretrained X-VLA with adapted native final transformer block and native projections"
                if self.adapter and self.adapter.get("adapted_native_blocks")
                else "pretrained X-VLA with adapted native action encoder, soft prompts, and decoder"
                if self.adapter and self.adapter.get("adapted_native_inputs")
                else "pretrained X-VLA with supervised native decoder adaptation"
            ),
            "adapter": self.adapter,
            "contract": native_contract(),
            "limitations": [
                "Task-adapted native modules; generalization beyond measured episodes is unestablished.",
                "Local RGB, measured own pose, and a phase instruction; unknown peer pose remains zero.",
                "Only right-arm xyz/grip executed; fixed-orientation simulation ignores native rotations.",
            ],
        }

    def predict(self, observation, instruction, *, capture_features=False):
        probe, torch = self.probe, self.probe.torch
        rgb = validate_rgb(observation.get("rgb"))
        state = measured_local_state(observation)
        batch = {
            "observation.images.image": torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
            "observation.state": torch.from_numpy(state),
            "task": instruction,
        }
        features = []

        def capture(_module, args):
            hidden, domain = args
            if tuple(hidden.shape) != (1, 30, FEATURE_WIDTH) or domain.tolist() != [XVLA_DOMAIN_ID]:
                raise RuntimeError("Unexpected native decoder boundary")
            features.append(hidden[0, 0].detach().cpu().numpy().copy())

        hook = (
            probe.model.model.transformer.action_decoder.register_forward_pre_hook(capture)
            if capture_features
            else None
        )
        started = perf_counter()
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=[]):
                # LeRobot generates x1 directly on the configured device. Seed
                # both CPU and MPS so caching and deployment use the same draw.
                torch.manual_seed(NOISE_SEED)
                if probe.device == "mps":
                    torch.mps.manual_seed(NOISE_SEED)
                prepared = probe.preprocess(batch)
                tokenization = _tokenization_evidence(prepared, probe.tokenizer, instruction)
                prepared["domain_id"] = torch.tensor([XVLA_DOMAIN_ID], device=probe.device)
                result = probe.model.predict_action_chunk(prepared)
                native_chunk = result.detach().cpu().numpy().copy()
                chunk = probe.postprocess(result).detach().cpu().numpy()[0]
        finally:
            if hook is not None:
                hook.remove()
        if chunk.shape != (30, 20) or not np.isfinite(chunk).all():
            raise RuntimeError("Invalid native X-VLA action chunk; no fallback")
        if not np.array_equal(native_chunk[0], chunk):
            raise RuntimeError("X-VLA postprocessor changed the calibrated native action frame")
        if capture_features and (len(features) != 1 or not np.isfinite(features[0]).all()):
            raise RuntimeError("Expected exactly one native decoder execution")
        probe.calls += 1
        probe.seconds += perf_counter() - started
        self.last_prediction = {
            "raw_chunk": chunk.tolist(),
            "used_index": 0,
            "selected_arm_offset": 10,
            "native_action_head_executed": True,
            "model_language_prompt": instruction,
            "tokenization": tokenization,
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "native_contract": native_contract(),
        }
        return chunk, (features[0] if capture_features else None)

    def act(self, observation, instruction, *, clearance=0.20, hold=False):
        """Refresh a real native action every eight simulation ticks.

        An instruction change, changed clearance, or coordination hold expires
        the cached action. Holds are an explicit safety/coordination override,
        never called VLA predictions. MuJoCo limits actuator movement per step;
        this wrapper only clips absolute native targets to the work envelope.
        """
        from .action_dataset import phase_instruction

        tick = observation.get("tick")
        if type(tick) is not int or tick < 0:
            raise ValueError("Native action execution requires a nonnegative simulation tick")
        # Validate proprio even on a hold, so missing or stale schemas cannot
        # silently cause a command using invented geometry.
        measured_local_state(observation)
        if not np.isfinite(clearance) or not 0.03 <= clearance <= 0.48:
            raise ValueError("Invalid native execution clearance")
        if self._cached is not None and tick < self._cached["tick"]:
            raise ValueError("Native action cache cannot travel backward in simulation time")
        if hold:
            self._cached = None
            self.last_prediction = {
                "coordinator_hold": True,
                "native_action_head_executed": False,
                "inference_id": None,
                "action_age_ticks": None,
                "reason": "Coordinator requested a stop; cached native action discarded.",
            }
            return {
                "target": np.asarray(observation["ee_position"], dtype=float).tolist(),
                "grip": bool(observation.get("gripper_closed", observation.get("grasped"))),
            }
        prompt = phase_instruction("right", instruction) if instruction in PHASES else instruction
        key = (prompt, float(clearance))
        if (
            self._cached is None
            or self._cached["key"] != key
            or tick - self._cached["tick"] >= self.action_horizon
        ):
            chunk, _ = self.predict(observation, prompt)
            raw = chunk[0]
            command = {
                "target": np.clip(raw[10:13], [-0.59, -0.49, 0.008], [0.59, 0.42, 0.52])
                .astype(float)
                .tolist(),
                "grip": bool(raw[19] > 0.5),
            }
            self._cached = {
                "tick": tick,
                "key": key,
                "command": command,
                "inference_id": self.probe.calls,
                "evidence": dict(self.last_prediction),
            }
        self.last_prediction = {
            **self._cached["evidence"],
            "coordinator_hold": False,
            "inference_id": self._cached["inference_id"],
            "action_age_ticks": tick - self._cached["tick"],
            "action_reused": tick != self._cached["tick"],
        }
        return {
            "target": list(self._cached["command"]["target"]),
            "grip": self._cached["command"]["grip"],
        }


def cache_episode(episode, output, *, device="mps", stride=4, policy=None):
    """Offline only: cache native decoder inputs from a separate training seed."""
    from .action_dataset import load_episode, phase_instruction
    from .policies import PHASES

    if type(stride) is not int or stride < 1:
        raise ValueError("Feature stride must be a positive integer")
    episode, output = Path(episode), Path(output)
    manifest, agents = load_episode(episode)
    arrays = agents["right"]
    # Coordination holds are enforced outside VLA inference and must not teach
    # an ambiguous stop action under an otherwise identical phase instruction.
    available = np.flatnonzero(~arrays["hold"])
    selected = set(available[::stride].tolist())
    for phase in np.unique(arrays["phase"]):
        indices = available[arrays["phase"][available] == phase]
        if len(indices):
            selected.update((int(indices[0]), int(indices[-1])))
    selected = sorted(selected)
    if not selected:
        raise ValueError("No active training observations")
    output.mkdir(parents=True, exist_ok=False)
    native = policy or NativeXVLA(device=device)
    if native.adapter is not None:
        raise ValueError(
            "Decoder-only feature caches require the original frozen native transformer"
        )
    original_weight, original_bias = native.decoder_parameters()
    features, targets = [], []
    started = perf_counter()
    for count, index in enumerate(selected):
        observation = {
            "agent": "right",
            **{
                name: arrays[name][index]
                for name in ("rgb", "ee_position", "ee_rotation", "grasped", "gripper_closed")
            },
        }
        instruction = phase_instruction("right", PHASES[int(arrays["phase"][index])])
        _, hidden = native.predict(observation, instruction, capture_features=True)
        features.append(hidden)
        targets.append(
            supervised_targets(
                arrays["target"][index : index + 1], arrays["grip"][index : index + 1]
            )[0]
        )
        if count % 10 == 0 or count + 1 == len(selected):
            memory = {}
            if native.probe.device == "mps":
                memory = {
                    "mps_allocated_gib": round(
                        native.probe.torch.mps.current_allocated_memory() / 2**30, 3
                    ),
                    "mps_driver_gib": round(
                        native.probe.torch.mps.driver_allocated_memory() / 2**30, 3
                    ),
                }
            print(
                json.dumps(
                    {
                        "family": "xvla",
                        "cached": count + 1,
                        "total": len(selected),
                        "seconds": round(perf_counter() - started, 2),
                        **memory,
                    }
                ),
                flush=True,
            )
    archive = output / "features.npz"
    np.savez_compressed(
        archive,
        features=np.asarray(features, np.float32),
        targets=np.asarray(targets, np.float32),
        original_weight=original_weight,
        original_bias=original_bias,
        tick=np.asarray(selected, np.int64),
        phase=arrays["phase"][selected],
    )
    metadata = {
        **native_contract(),
        "episode_manifest_sha256": hashlib.sha256(
            (episode / "manifest.json").read_bytes()
        ).hexdigest(),
        "episode_seed": manifest["seed"],
        "episode_split": manifest["split"],
        "right_demonstrations_sha256": manifest["files"]["right.npz"],
        "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "samples": len(selected),
        "stride": stride,
        "noise_device": native.probe.device,
        "wall_seconds": perf_counter() - started,
    }
    (output / "cache.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def fit_cache(cache, output, *, ridge=0.01):
    """Fit a native head with an explicit interpolation check, then all train rows.

    This check holds out frames within a training episode. It is not a held-out
    seed rollout and never constitutes successful closed-loop folding evidence.
    """
    cache = Path(cache)
    metadata = json.loads((cache / "cache.json").read_text())
    if any(metadata.get(key) != value for key, value in native_contract().items()):
        raise ValueError("Incompatible native feature cache")
    archive = cache / "features.npz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["cache_sha256"]:
        raise ValueError("Native feature cache digest changed")
    with np.load(archive, allow_pickle=False) as saved:
        x, y = saved["features"], saved["targets"]
        w, b = saved["original_weight"], saved["original_bias"]
        phases = saved["phase"]
    validation = np.zeros(len(x), dtype=bool)
    for phase in np.unique(phases):
        indices = np.flatnonzero(phases == phase)
        if len(indices) >= 5:
            validation[indices[2::5]] = True
    if len(x) < 4 or np.count_nonzero(~validation) < 2 or not validation.any():
        raise ValueError("Insufficient rows for a transparent interpolation check")
    check_w, check_b = fit_native_columns(x[~validation], y[~validation], w, b, ridge=ridge)
    held_metrics = decoder_metrics(x[validation], y[validation], check_w, check_b)
    fitted_w, fitted_b = fit_native_columns(x, y, w, b, ridge=ridge)
    return save_adapter(
        output,
        fitted_w,
        fitted_b,
        dataset_sha256=metadata["episode_manifest_sha256"],
        training={
            "method": "closed-form ridge update of existing native decoder columns; native endpoint XYZ and grip logits",
            "ridge": ridge,
            "native_flow_time": 1.0,
            "source_cache_sha256": metadata["cache_sha256"],
            "episode_seed": metadata["episode_seed"],
            "episode_split": "train",
            "noise_device": metadata.get("noise_device", "mps"),
            "weight_update_l2": float(np.linalg.norm(fitted_w[:, CHANNELS] - w[:, CHANNELS])),
            "maximum_absolute_weight": float(np.abs(fitted_w[:, CHANNELS]).max()),
            "fit": decoder_metrics(x, y, fitted_w, fitted_b),
            "within_training_episode_interpolation": held_metrics,
            "closed_loop_success": None,
            "held_out_seed_evaluation": "Required separately; feature regression does not establish task success.",
        },
    )


def evaluate_cache(cache, adapter):
    """Evaluate immutable cached observations without loading another VLA.

    Unlike the interpolation check in fitting, a separately collected seed can
    be used here. These remain teacher-distribution prediction errors; a new
    closed-loop rollout is still necessary to establish task completion.
    """
    cache = Path(cache)
    metadata = json.loads((cache / "cache.json").read_text())
    if any(metadata.get(key) != value for key, value in native_contract().items()):
        raise ValueError("Incompatible native validation feature cache")
    archive = cache / "features.npz"
    if _file_sha256(archive) != metadata["cache_sha256"]:
        raise ValueError("Native validation feature cache digest changed")
    package, weight, bias = load_adapter(adapter)
    if any(
        package.get(key)
        for key in (
            "adapted_native_inputs",
            "adapted_native_blocks",
            "adapted_native_final_norm",
            "adapted_native_multimodal",
        )
    ):
        raise ValueError(
            "Decoder-only feature caches cannot evaluate changed native feature-producing modules; use frozen Florence caches and the full native transformer"
        )
    if metadata.get("noise_device") != package["training"].get("noise_device"):
        raise ValueError("Native validation noise backend differs from training")
    with np.load(archive, allow_pickle=False) as saved:
        features, targets, phases = saved["features"], saved["targets"], saved["phase"]
    return {
        "kind": "teacher_distribution_native_action_prediction",
        "model_id": XVLA_ID,
        "adapter_sha256": package["decoder_sha256"],
        "cache_sha256": metadata["cache_sha256"],
        "episode_seed": metadata["episode_seed"],
        "held_out_episode": (
            metadata["episode_manifest_sha256"] != package["dataset_sha256"]
            and metadata["episode_seed"] != package["training"]["episode_seed"]
        ),
        "overall": decoder_metrics(features, targets, weight, bias),
        "phases": {
            PHASES[int(phase)]: decoder_metrics(
                features[phases == phase], targets[phases == phase], weight, bias
            )
            for phase in np.unique(phases)
        },
        "closed_loop_task_success": None,
    }


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache")
    cache.add_argument("--episode", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--device", choices=("cpu", "mps"), default="mps")
    cache.add_argument("--stride", type=int, default=4)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--cache", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--ridge", type=float, default=0.01)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--adapter", required=True)
    args = parser.parse_args()
    if args.command == "cache":
        result = cache_episode(args.episode, args.output, device=args.device, stride=args.stride)
    elif args.command == "fit":
        result = fit_cache(args.cache, args.output, ridge=args.ridge)
    else:
        result = evaluate_cache(args.cache, args.adapter)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
