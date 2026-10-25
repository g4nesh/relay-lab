"""Small CoLA-inspired mixed-backbone message encoders and replacement heads.

This is not CoLA's diffusion policy or VLA-Fabric. Native VLA action heads are
not used. Frozen image/language features feed learned, direction-specific
adapters; locally measured state assists a separately trained Cartesian head.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .communication import AGENTS, FEATURES, SAFETY_FIELDS, observation_features
from .policies import PHASES

FORMAT = "relay-cloth-cross-vla-adapter-v1"
WIDTHS = {"left": 960, "right": 1024}
DIMENSION = 64
MODES = ("none", "structured_text", "embeddings", "hybrid")
LOCAL_DIM = len(FEATURES) + len(PHASES)
MAX_DELTA = 0.018


def local_input(observation, phase):
    if phase not in PHASES:
        raise ValueError("Unknown learned cloth instruction")
    facts = observation_features(observation)
    return np.asarray(
        [facts[k] for k in FEATURES] + [float(p == phase) for p in PHASES], np.float32
    )


def bounded_label(observation, action):
    delta = np.asarray(action["target"], np.float32) - observation["ee_position"]
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("Invalid original teaching action")
    delta *= min(1.0, MAX_DELTA / max(float(np.linalg.norm(delta)), 1e-12))
    return np.asarray([*(delta / MAX_DELTA), float(action["grip"])], np.float32)


class DualLatentAdapter(nn.Module):
    def __init__(self, mode="embeddings", widths=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown learned communication mode")
        self.mode, self.widths = mode, dict(widths or WIDTHS)
        self.senders, self.local_encoders, self.receivers, self.heads = (
            nn.ModuleDict() for _ in range(4)
        )
        for arm, width in self.widths.items():
            self.senders[arm] = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 96),
                nn.GELU(),
                nn.Linear(96, DIMENSION),
                nn.Tanh(),
            )
            self.local_encoders[arm] = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 32), nn.Tanh()
            )
            self.receivers[arm] = nn.Sequential(
                nn.Linear(DIMENSION, 64), nn.GELU(), nn.Linear(64, 32), nn.Tanh()
            )
            # xyz delta, gripper logit, coordinated-hold logit. Jointly learned
            # nominal motion and hold supervision are disclosed in the manifest.
            self.heads[arm] = nn.Sequential(
                nn.Linear(32 + LOCAL_DIM + 32 + len(FEATURES) + 2, 96),
                nn.GELU(),
                nn.Linear(96, 64),
                nn.GELU(),
                nn.Linear(64, 5),
            )
            # This pilot trains fresh observations only. Its unused age input
            # must not become a random control signal at deployment. Future
            # historical-message training can learn this column from zero.
            with torch.no_grad():
                self.heads[arm][0].weight[:, -1].zero_()

    def encode(self, arm, features):
        if arm not in AGENTS or features.ndim != 2 or features.shape[1] != self.widths[arm]:
            raise ValueError("Incorrect native VLA feature space")
        return self.senders[arm](features)

    def decode_action(self, arm, own_features, state, message, facts, valid, age):
        # Missing messages must not acquire information from decoder biases.
        mask = valid.reshape(-1, 1).to(own_features.dtype)
        peer = self.receivers[arm](message * mask) * mask
        if self.mode not in ("embeddings", "hybrid"):
            peer = peer * 0
        if self.mode not in ("structured_text", "hybrid"):
            facts = facts * 0
        if self.mode == "hybrid":
            # Hybrid supplies exactly the documented categorical/obstacle safety
            # fields, never silently filling unavailable peer cloth geometry.
            selector = torch.zeros_like(facts)
            selector[:, [FEATURES.index(k) for k in SAFETY_FIELDS]] = 1
            facts = facts * selector
        if self.mode == "none":
            mask, age = mask * 0, age * 0
        joined = torch.cat(
            [
                self.local_encoders[arm](own_features),
                state,
                peer,
                facts * mask,
                mask,
                age.reshape(-1, 1).clamp(0, 1),
            ],
            dim=1,
        )
        return self.heads[arm](joined)

    def forward(self, features, states, facts, valid, age, *, intervention=None):
        messages = {a: self.encode(a, features[a]) for a in AGENTS}
        result = {}
        for arm, other in (("left", "right"), ("right", "left")):
            message, peer_facts = messages[other], facts[other]
            if intervention == "zero":
                message, peer_facts = message * 0, peer_facts * 0
            elif intervention == "shuffle":
                message, peer_facts = message.roll(1, 0), peer_facts.roll(1, 0)
            elif intervention is not None:
                raise ValueError("Unknown peer-message intervention")
            result[arm] = self.decode_action(
                arm, features[arm], states[arm], message, peer_facts, valid, age
            )
        return result


class LearnedLatentPolicy:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT or payload.get("widths") != WIDTHS:
            raise ValueError("Incompatible cross-VLA checkpoint")
        self.network = DualLatentAdapter(payload["mode"]).eval().requires_grad_(False)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.mode, self.training = payload["mode"], payload["training"]
        self.extractor_ids = payload["extractor_ids"]
        if set(self.extractor_ids) != set(AGENTS) or any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in self.extractor_ids.values()
        ):
            raise ValueError("Invalid native feature provenance")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def model_info(self):
        return {
            "kind": "cross-vla-adapter-v1",
            "space_id": "cross-vla-" + self.sha256[:24],
            "checkpoint_sha256": self.sha256,
            "dimension": DIMENSION,
            "source_dimensions": WIDTHS,
            "source_extractor_ids": self.extractor_ids,
            "native_action_heads_used": False,
            "parameters": sum(p.numel() for p in self.network.parameters()),
        }

    def encode(self, arm, features, extractor_id):
        if arm not in AGENTS or extractor_id != self.extractor_ids[arm]:
            raise ValueError("Frozen VLA extractor does not match adapter training")
        vector = np.asarray(features, np.float32)
        if vector.shape != (WIDTHS[arm],) or not np.isfinite(vector).all():
            raise ValueError("Invalid frozen VLA features")
        with torch.inference_mode():
            return self.network.encode(arm, torch.from_numpy(vector.copy())[None])[0].tolist()

    def act(self, observation, phase, own_features, belief, *, extractor_id, ttl=24):
        arm = observation["agent"]
        vector = np.asarray(own_features, np.float32)
        if (
            arm not in AGENTS
            or vector.shape != (WIDTHS[arm],)
            or not np.isfinite(vector).all()
            or extractor_id != self.extractor_ids[arm]
        ):
            raise ValueError("Invalid local native features")
        if belief is not None and (
            belief["recipient"] != arm
            or belief["checkpoint_sha256"] != self.sha256
            or not 0 <= belief["age_ticks"] < ttl
        ):
            raise ValueError("Invalid delivered adapter belief")
        message = np.asarray(belief["vector"] if belief else np.zeros(DIMENSION), np.float32)
        facts = belief.get("features", {}) if belief else {}
        tensors = [
            vector,
            local_input(observation, phase),
            message,
            np.asarray([facts.get(k, 0.0) for k in FEATURES], np.float32),
        ]
        if message.shape != (DIMENSION,) or not all(np.isfinite(v).all() for v in tensors):
            raise ValueError("Invalid received learned message")
        with torch.inference_mode():
            raw = self.network.decode_action(
                arm,
                *(torch.from_numpy(v.copy())[None] for v in tensors),
                torch.tensor([float(belief is not None)]),
                torch.tensor([belief["age_ticks"] / ttl if belief else 1.0]),
            )[0].numpy()
        # Missing/expired communication halts coordinated movement, explicitly
        # outside the learned head. No-communication baseline has no such gate.
        missing_hold = belief is None and self.mode != "none"
        hold = bool(raw[4] >= 0 or missing_hold)
        delta = np.tanh(raw[:3]) * MAX_DELTA
        delta *= min(1.0, MAX_DELTA / max(float(np.linalg.norm(delta)), 1e-12))
        ee = np.asarray(observation["ee_position"])
        target = ee if hold else np.clip(ee + delta, [-0.59, -0.49, 0.008], [0.59, 0.42, 0.52])
        grip = bool(observation["grasped"]) if hold else bool(raw[3] >= 0)
        return {"target": target.tolist(), "grip": grip}, {
            "raw_head": raw.tolist(),
            "learned_hold": bool(raw[4] >= 0),
            "missing_message_hold": missing_hold,
            "replacement_head": True,
            "peer_age_ticks": belief["age_ticks"] if belief else None,
        }
