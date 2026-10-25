"""Frozen, checkpoint-native RGB/language features for learned communication.

These extractors stop before the native action head. They do not call a teacher,
project simulator state, or align incompatible feature spaces. SmolVLA and X-VLA
have different feature widths; a separately trained adapter must learn how a
receiver uses its peer's features. Load one extractor at a time when preparing
training caches on memory-constrained machines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .policies import validate_rgb
from .pretrained import (
    XVLA_ID,
    XVLA_REVISION,
    SmolVLAClothProbe,
    XVLAClothProbe,
    _tokenization_evidence,
)

FEATURE_SCHEMA = "relay-cloth-frozen-vlm-features-v1"
_FAMILIES = {
    "smolvla": {
        "model_id": "HuggingFaceVLA/smolvla_libero",
        "revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        "width": 960,
        "layer": "SmolVLMWithExpertModel.forward: final normalized prefix hidden states",
        "pooling": "mean of valid image and language tokens; state and padding excluded",
        "preprocessing": "pinned saved SmolVLA processors; one local RGB duplicated in two camera slots",
    },
    "xvla": {
        "model_id": XVLA_ID,
        "revision": XVLA_REVISION,
        "width": 1024,
        "layer": "XVLA.forward_vlm: Florence2 language_model.encoder final hidden states",
        "pooling": "mean of primary image and valid language tokens; language padding excluded",
        "preprocessing": "pinned saved X-VLA processors plus official ImageNet normalization; missing views masked",
    },
}


def feature_contract(family: str) -> dict:
    """Pure pinned boundary identity; does not load a model or access the network."""
    if family not in _FAMILIES:
        raise ValueError("Unknown frozen VLA family")
    spec = dict(_FAMILIES[family])
    identity = hashlib.sha256(
        json.dumps({"schema": FEATURE_SCHEMA, **spec}, sort_keys=True).encode()
    ).hexdigest()
    return {"schema": FEATURE_SCHEMA, "family": family, **spec, "feature_space_id": identity}


@dataclass(frozen=True)
class FrozenVLAFeatures:
    """One local observation encoded by the actual frozen checkpoint."""

    vector: np.ndarray
    metadata: dict


def _pool_hidden(hidden, mask, width):
    """Validate the backbone boundary, then pool only explicitly valid tokens."""
    import torch

    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.ndim != 3
        or hidden.shape[0] != 1
        or hidden.shape[2] != width
        or not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != tuple(hidden.shape[:2])
        or not bool(mask.any())
    ):
        raise ValueError("Invalid frozen VLA feature tensor, width, or pooling mask")
    # Check every valid token, rather than allowing positive/negative overflow to
    # hide in the mean. Padding is never an information-bearing feature.
    selected = hidden[mask]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Nonfinite frozen VLA hidden states; no feature fallback")
    vector = selected.float().mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=True)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite pooled VLA features; no feature fallback")
    vector.flags.writeable = False
    return vector


class FrozenVLAFeatureExtractor:
    """Extract native learned multimodal features without action denoising.

    ``probe`` optionally reuses an already loaded matching cloth probe. Numeric
    observation fields are deliberately never read. Required state slots receive
    zeros for preprocessing compatibility; SmolVLA's native attention mask blocks
    state from image/language tokens, and those state tokens are not pooled.
    """

    def __init__(self, family: str, agent: str, device: str = "cpu", *, probe=None):
        if family not in _FAMILIES or agent not in ("left", "right"):
            raise ValueError("Unknown frozen VLA family or arm")
        self.family, self.agent = family, agent
        self.probe = probe or (
            SmolVLAClothProbe(agent, device)
            if family == "smolvla"
            else XVLAClothProbe(agent, device)
        )
        if self.probe.agent != agent:
            raise ValueError("Frozen VLA extractor received another arm's probe")
        self.spec = dict(_FAMILIES[family])
        info = self.probe.model_info
        if any(info.get(key) != self.spec[key] for key in ("model_id", "revision")):
            raise ValueError("Frozen VLA checkpoint does not match the pinned feature contract")
        self.model = self.probe.backend._model if family == "smolvla" else self.probe.model
        self.model.eval().requires_grad_(False)
        self.calls, self.seconds = 0, 0.0
        self.space_id = feature_contract(family)["feature_space_id"]

    @property
    def width(self):
        return self.spec["width"]

    @property
    def model_info(self):
        return {
            "schema": FEATURE_SCHEMA,
            "family": self.family,
            "agent": self.agent,
            **self.spec,
            "feature_space_id": self.space_id,
            "foundation_features": True,
            "backbone_frozen": True,
            "native_action_head_executed": False,
            "observation_numeric_state_used": False,
            "inference_calls": self.calls,
            "inference_seconds": self.seconds,
            "checkpoint": self.probe.model_info,
        }

    def extract(self, observation: dict, instruction: str) -> FrozenVLAFeatures:
        if observation.get("agent") != self.agent:
            raise ValueError("Wrong arm observation for frozen VLA features")
        rgb = validate_rgb(observation.get("rgb"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Frozen VLA features require a nonempty instruction")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Feature extraction requires an evaluation-mode frozen backbone")
        started = perf_counter()
        vector, tokenization, tokens = (
            self._smolvla(rgb, instruction)
            if self.family == "smolvla"
            else self._xvla(rgb, instruction)
        )
        elapsed = perf_counter() - started
        self.calls += 1
        self.seconds += elapsed
        return FrozenVLAFeatures(
            vector,
            {
                **self.model_info,
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "rgb_shape": list(rgb.shape),
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                "vector_sha256": hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest(),
                "pooled_tokens": tokens,
                "seconds": elapsed,
                "tokenization": tokenization,
            },
        )

    def _smolvla(self, rgb, instruction):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        backend = self.probe.backend
        torch = backend._torch
        prompt = instruction if instruction.endswith("\n") else instruction + "\n"
        if len(backend._tokenizer(prompt)["input_ids"]) > backend._config.tokenizer_max_length:
            raise ValueError("Frozen VLA instruction exceeds the pinned language budget")
        pixels = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        batch = {
            "observation.images.image": pixels,
            "observation.images.image2": pixels.clone(),
            "observation.state": torch.zeros(8, dtype=torch.float32),
            "task": prompt,
        }
        with torch.inference_mode():
            prepared = backend._preprocess(batch)
            evidence = _tokenization_evidence(prepared, backend._tokenizer, prompt)
            prepared = self.model._prepare_batch(prepared)
            images, image_masks = self.model.prepare_images(prepared)
            prefix, valid, blocks = self.model.model.embed_prefix(
                images,
                image_masks,
                prepared["observation.language.tokens"],
                prepared["observation.language.attention_mask"],
                state=self.model.prepare_state(prepared),
            )
            attention = make_att_2d_masks(valid, blocks)
            positions = torch.cumsum(valid, dim=1) - 1
            outputs, _ = self.model.model.vlm_with_expert.forward(
                attention_mask=attention,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix, None],
                # Native prefill uses self-attention to construct its cache. The
                # cache is discarded here; no diffusion/action expert is run.
                use_cache=True,
            )
            visual_language = valid.bool() & (torch.cumsum(blocks, dim=1) == 0)
            vector = _pool_hidden(outputs[0], visual_language, self.width)
        return vector, evidence, int(visual_language.sum())

    def _xvla(self, rgb, instruction):
        probe = self.probe
        torch = probe.torch
        batch = {
            "observation.images.image": torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
            "observation.state": torch.zeros(20, dtype=torch.float32),
            "task": instruction,
        }
        with torch.inference_mode():
            prepared = probe.preprocess(batch)
            evidence = _tokenization_evidence(prepared, probe.tokenizer, instruction)
            images, image_mask = self.model._prepare_images(prepared)
            tokens = prepared["observation.language.tokens"]
            hidden = self.model.model.forward_vlm(tokens, images, image_mask)["vlm_features"]
            image_tokens = hidden.shape[1] - tokens.shape[1]
            if image_tokens < 1:
                raise ValueError("X-VLA returned no primary image tokens")
            language_valid = prepared["observation.language.attention_mask"].bool()
            mask = torch.cat(
                [torch.ones((1, image_tokens), dtype=torch.bool, device=hidden.device), language_valid],
                dim=1,
            )
            vector = _pool_hidden(hidden, mask, self.width)
        evidence = {
            **evidence,
            "native_encoder_language_padding": "Native Florence forward_vlm attends to padded language positions; pooling excludes them.",
        }
        return vector, evidence, int(mask.sum())
