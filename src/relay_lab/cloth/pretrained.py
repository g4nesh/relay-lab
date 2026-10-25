"""Pinned released VLA adapters for explicit out-of-domain transfer probes.

No task-trained policy is called if a released model fails. X-VLA's checkpoint
was trained for another two-arm robot, and SmolVLA LIBERO for rigid manipulation;
neither is claimed to fold this simulation without adaptation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .policies import validate_rgb

XVLA_ID = "lerobot/xvla-folding"
XVLA_REVISION = "c87d6e8fcf47fb19d6aac8b67d35dd20d74bbc8d"
XVLA_WEIGHT_BYTES = 3_519_073_692
XVLA_WEIGHT_SHA256 = "5d9abbcffd042e75347c11b569b8f4f8993062a3c5995b364ac4094d378c1a91"
# Official Soft-FOLD Agilex deployment client selects domain 5.
XVLA_DOMAIN_ID = 5
XVLA_DOMAIN_SOURCE = (
    "https://github.com/2toinf/X-VLA/blob/main/evaluation/SoftFold-Agilex/"
    "deploy/client_eef6d_xvla.py"
)
XVLA_IMAGE_SOURCE = (
    "https://huggingface.co/2toINF/X-VLA-SoftFold/blob/"
    "b8eb3f22026924e3db3e88d47428896c736dc5c9/preprocessor_config.json"
)
XVLA_ROTATION_SOURCE = (
    "https://github.com/2toinf/X-VLA/blob/main/evaluation/SoftFold-Agilex/deploy/utils/rotation.py"
)
BART_ID = "facebook/bart-large"
BART_REVISION = "cb48c1365bd826bd521f650dc2e0940aee54720c"


def _bounded_target(ee, target, maximum=0.018):
    ee, target = np.asarray(ee, dtype=float), np.asarray(target, dtype=float)
    if ee.shape != (3,) or target.shape != (3,) or not np.isfinite([ee, target]).all():
        raise ValueError("invalid pretrained Cartesian action")
    delta = target - ee
    delta *= min(1.0, maximum / max(float(np.linalg.norm(delta)), 1e-12))
    return np.clip(ee + delta, [-0.59, -0.49, 0.008], [0.59, 0.42, 0.52]).tolist()


def _xvla_state(ee, agent, grasped):
    """Approximate downward tool pose using Soft-FOLD's interleaved columns."""
    ee = np.asarray(ee, dtype=np.float32)
    if agent not in ("left", "right") or ee.shape != (3,) or not np.isfinite(ee).all():
        raise ValueError("invalid XVLA local arm proprioception")
    state = np.zeros(20, dtype=np.float32)
    offset = 0 if agent == "left" else 10
    # Rx(pi) first two matrix columns, flattened row-wise, as in the original
    # Soft-FOLD client. This is a fixed approximation, not measured orientation.
    state[offset : offset + 10] = [*ee, 1, 0, 0, -1, 0, 0, float(grasped)]
    return state


def _ensure_xvla_image_normalization(pipeline):
    """Repair the pinned legacy processor's omitted ImageNet normalization."""
    from lerobot.policies.xvla.processor_xvla import XVLAImageNetNormalizeProcessorStep

    count = sum(isinstance(step, XVLAImageNetNormalizeProcessorStep) for step in pipeline.steps)
    if count > 1:
        raise ValueError("XVLA image normalization must run exactly once")
    if not count:
        # Saved VISUAL normalization is IDENTITY. Use the official XVLA step
        # after those saved processors; never normalize again if it is present.
        pipeline.steps = [*pipeline.steps, XVLAImageNetNormalizeProcessorStep()]


def _tokenization_evidence(prepared, tokenizer, expected):
    """Inspect the exact token tensor being passed to the released policy."""
    tokens = prepared["observation.language.tokens"]
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("Expected one local robot instruction")
    ids = tokens[0].detach().cpu().tolist()
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if " ".join(decoded.split()) != " ".join(expected.split()):
        raise ValueError("Model tokenization dropped or changed coordination prompt content")
    return {
        "sequence_tokens_including_padding": len(ids),
        "decoded_model_tokens": decoded,
        "full_prompt_roundtrip_verified": True,
        "tokenizer_truncation": False,
    }


class _ObservedPreprocessor:
    """Observe this adapter instance's actual preprocessing without changing it."""

    def __init__(self, pipeline, tokenizer):
        self.pipeline, self.tokenizer, self.evidence = pipeline, tokenizer, None

    def __call__(self, batch):
        expected = batch["task"]
        prepared = self.pipeline(batch)
        self.evidence = _tokenization_evidence(prepared, self.tokenizer, expected)
        return prepared

    def __getattr__(self, name):
        return getattr(self.pipeline, name)


class SmolVLAClothProbe:
    """Single-arm LIBERO checkpoint with independent local observation only."""

    def __init__(self, agent="left", device="cpu", seed=7):
        import torch

        from ..smolvla import SmolVLABackend

        if agent not in ("left", "right"):
            raise ValueError("unknown arm")
        torch.set_num_threads(3)
        self.agent = agent
        self.backend = SmolVLABackend(device=device, seed=seed)
        self.backend._preprocess = _ObservedPreprocessor(
            self.backend._preprocess, self.backend._tokenizer
        )
        self.last_prediction = {}

    @property
    def model_info(self):
        return {
            **self.backend.model_info,
            "cloth_adapter": "single-arm unadapted transfer probe",
            "agent": self.agent,
            "torch_threads": self.backend._torch.get_num_threads(),
            "frame_translation_metres": {self.agent: [0.0, 0.0, 0.0]},
            "noise_stream": "The same fixed probe seed is reused for each inference call.",
            "max_delta_norm_metres": 0.018,
            "limitations": [
                "Approximate cloth frame/embodiment transfer; no task adaptation or success claim.",
                "Predicted rotation is ignored by the fixed-orientation gripper.",
                "The caller supplies the task/coordination instruction; no phase teacher is invoked.",
            ],
            "cloth_limitations": [
                "LIBERO checkpoint has no training for this cloth scene.",
                "One local camera duplicated as front and wrist; rotation discarded.",
                "Own tool orientation is approximated by fixed downward Euler angles, not estimated from image.",
                "Table-frame xyz passed as approximate LIBERO coordinates; no frame calibration.",
            ],
        }

    def act(self, observation, instruction="Fold the T-shirt neatly using your arm."):
        if observation.get("agent") != self.agent:
            raise ValueError("wrong arm observation")
        rgb = validate_rgb(observation.get("rgb"))
        ee = np.asarray(observation["ee_position"], dtype=float)
        state = np.asarray(
            [
                *ee,
                np.pi,
                0,
                0,
                0.01 if observation["grasped"] else 0.04,
                -0.01 if observation["grasped"] else -0.04,
            ],
            dtype=np.float32,
        )
        chunk = self.backend.predict(rgb, rgb, state, instruction)
        raw = chunk[0]
        target = _bounded_target(ee, ee + np.clip(raw[:3], -1, 1) * 0.05)
        grip = bool(raw[6] > 0.3) if abs(float(raw[6])) > 0.3 else bool(observation["grasped"])
        self.last_prediction = {
            "raw_chunk": chunk.tolist(),
            "used_index": 0,
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "ignored_rotation": raw[3:6].tolist(),
            "model_language_prompt": self.backend.last_model_prompt,
            "language_tokens": self.backend.last_prompt_tokens,
            "tokenizer_truncation": False,
            "tokenization": self.backend._preprocess.evidence,
        }
        return {"target": target, "grip": grip}


class XVLAClothProbe:
    """Run released bimanual weights on a single robot's partial view.

    Only that robot's 10D pose block is populated; unknown peer proprioception
    stays zero. We retain the model's bimanual output but execute only this arm's
    block. This intentionally tests transfer, not faithful Soft-FOLD evaluation.
    """

    def __init__(
        self,
        agent="right",
        device="cpu",
        seed=7,
        denoising_steps=10,
        cache_dir=Path(".cache/huggingface/hub"),
        local_files_only=True,
    ):
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        if agent not in ("left", "right") or device not in ("cpu", "mps"):
            raise ValueError("invalid XVLA arm/device")
        self.agent, self.device, self.seed = agent, device, seed
        self.torch = torch
        torch.set_num_threads(3)
        snapshot = snapshot_download(
            XVLA_ID,
            revision=XVLA_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "model.safetensors"],
        )
        checkpoint = Path(snapshot) / "model.safetensors"
        digest = hashlib.sha256()
        with checkpoint.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()
        if (
            checkpoint.stat().st_size != XVLA_WEIGHT_BYTES
            or self.checkpoint_sha256 != XVLA_WEIGHT_SHA256
        ):
            raise ValueError("X-VLA checkpoint does not match the pinned publisher SHA-256")
        tokenizer = snapshot_download(
            BART_ID,
            revision=BART_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[
                "config.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
        config = PreTrainedConfig.from_pretrained(snapshot)
        config.device = device
        config.dtype = "float32"
        config.num_denoising_steps = denoising_steps
        config.n_action_steps = 1
        started = perf_counter()
        self.model = (
            XVLAPolicy.from_pretrained(snapshot, config=config).eval().requires_grad_(False)
        )
        self.preprocess, self.postprocess = make_pre_post_processors(
            config,
            snapshot,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {
                    "tokenizer_name": tokenizer,
                    "max_length": 50,
                    "truncation": False,
                },
            },
        )
        _ensure_xvla_image_normalization(self.preprocess)
        self.tokenizer = next(
            step.input_tokenizer
            for step in self.preprocess.steps
            if hasattr(step, "input_tokenizer")
        )
        self.load_seconds = perf_counter() - started
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.calls, self.seconds = 0, 0.0
        self.last_prediction = {}

    @property
    def model_info(self):
        return {
            "model_id": XVLA_ID,
            "revision": XVLA_REVISION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "tokenizer_id": BART_ID,
            "tokenizer_revision": BART_REVISION,
            "foundation_vla": True,
            "checkpoint_task": "Soft-FOLD cloth folding",
            "agent": self.agent,
            "device": self.device,
            "parameters": self.parameters,
            "torch_threads": self.torch.get_num_threads(),
            "domain_id": XVLA_DOMAIN_ID,
            "domain_source": XVLA_DOMAIN_SOURCE,
            "load_seconds": self.load_seconds,
            "inference_calls": self.calls,
            "inference_seconds": self.seconds,
            "normalization": "Saved processors plus official XVLA ImageNet normalization once; absent action statistics remain identity.",
            "image_normalization_source": XVLA_IMAGE_SOURCE,
            "rotation_order": "Interleaved first two matrix columns (Soft-FOLD reference); fixed approximate Rx(pi) tool orientation.",
            "rotation_source": XVLA_ROTATION_SOURCE,
            "limitations": [
                "Unadapted robot/camera/action-frame transfer; no cloth success claim.",
                "Only local arm pose and local camera are supplied; missing peer pose is zero.",
                "Local orientation is an approximate fixed downward tool pose, not measured or calibrated.",
                "Domain 5 follows the official Soft-FOLD Agilex deployment client; this simulation is a different embodiment.",
                "Model outputs two Cartesian poses; only selected arm xyz/grip is executed; rotations ignored.",
            ],
        }

    def act(self, observation, instruction="Fold the T-shirt neatly using your arm."):
        if observation.get("agent") != self.agent:
            raise ValueError("wrong arm observation")
        torch = self.torch
        rgb = validate_rgb(observation.get("rgb"))
        ee = np.asarray(observation["ee_position"], dtype=np.float32)
        state = _xvla_state(ee, self.agent, observation["grasped"])
        offset = 0 if self.agent == "left" else 10
        batch = {
            "observation.images.image": torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
            "observation.state": torch.from_numpy(state),
            "task": instruction,
            "domain_id": XVLA_DOMAIN_ID,
        }
        started = perf_counter()
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.calls)
            prepared = self.preprocess(batch)
            tokenization = _tokenization_evidence(prepared, self.tokenizer, instruction)
            # Saved legacy processors drop arbitrary complementary keys, including
            # domain_id. Set it on the final model batch so domain 5 really reaches
            # the learned embodiment-specific action heads and soft prompts.
            prepared["domain_id"] = torch.tensor([XVLA_DOMAIN_ID], device=self.device)
            result = self.model.predict_action_chunk(prepared)
            chunk = self.postprocess(result).detach().cpu().numpy()[0]
        self.seconds += perf_counter() - started
        self.calls += 1
        if chunk.ndim != 2 or chunk.shape[-1] != 20 or not np.isfinite(chunk).all():
            raise RuntimeError("XVLA returned invalid actions; no fallback")
        raw = chunk[0]
        target = _bounded_target(ee, raw[offset : offset + 3])
        self.last_prediction = {
            "raw_chunk": chunk.tolist(),
            "used_index": 0,
            "selected_arm_offset": offset,
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "model_language_prompt": instruction,
            "model_token_sequence_length": int(prepared["observation.language.tokens"].shape[-1])
            if "observation.language.tokens" in prepared
            else None,
            "tokenizer_truncation": False,
            "domain_id": XVLA_DOMAIN_ID,
            "tokenization": tokenization,
        }
        return {"target": target, "grip": bool(raw[offset + 9] > 0.5)}
