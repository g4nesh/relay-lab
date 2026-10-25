"""Two independent, original state-assisted vision/language/action policies.

These compact policies are trained on this workcell's original reference data.
They are not foundation VLAs. Local measured cloth landmarks are explicit inputs;
RGB is an auxiliary context input, whose necessity is not presumed. The runtime
does not import the simulator, its teacher, peer observations, or mesh state.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from time import perf_counter

import numpy as np

FORMAT = "relay-cloth-state-assisted-v1"
PHASES = (
    "approach_sleeve_above",
    "approach_sleeve",
    "grasp_sleeve",
    "lift_sleeve",
    "carry_sleeve",
    "fold_sleeve",
    "release_sleeve",
    "retreat_sleeve",
    "approach_hem_above",
    "approach_hem",
    "grasp_hem",
    "lift_hem",
    "carry_hem",
    "fold_hem",
    "release_hem",
    "settle",
)
LANDMARKS = ("sleeve", "hem", "shoulder")
STATE_DIM = 30
ARCHITECTURES = {"left": "spatial-cnn-affine-experts", "right": "patch-attention-affine-experts"}
DEFAULT_DIRECTORY = Path("checkpoints/cloth-policies")


def language_features(instruction: str) -> np.ndarray:
    if not isinstance(instruction, str):
        raise TypeError("cloth instruction must be a string")
    normalized = re.sub(r"\s+", "_", instruction.strip().lower())
    if normalized not in PHASES:
        raise ValueError(f"unsupported cloth instruction: {instruction!r}")
    return np.eye(len(PHASES), dtype=np.float32)[PHASES.index(normalized)]


def observation_features(
    observation: dict, clearance: float = 0.20, feature_basis: str = "reachable-arc-v2"
) -> np.ndarray:
    """Only explicitly supplied local measurements; missing landmarks stay masked."""
    ee = np.asarray(observation.get("ee_position"), dtype=np.float32)
    if ee.shape != (3,) or not np.isfinite(ee).all():
        raise ValueError("cloth policy requires finite own end-effector position")
    if not 0.03 <= clearance <= 0.48 or not np.isfinite(clearance):
        raise ValueError("clearance must be finite and between 0.03 and 0.48 metres")
    result = [*ee, float(bool(observation.get("grasped"))), float(clearance)]
    landmarks = observation.get("landmarks", {})
    for name in LANDMARKS:
        xyz = np.asarray(landmarks.get(name, [0, 0, 0]), dtype=np.float32)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"invalid local landmark: {name}")
        result.extend(xyz)
        result.append(float(name in landmarks))
    origin = np.asarray(observation.get("reference_origin", [0, 0]), dtype=np.float32)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("invalid local reference origin")
    if feature_basis not in ("original-v1", "reachable-arc-v2"):
        raise ValueError("unsupported cloth control feature basis")
    heights = (0.31, 0.29) if feature_basis == "original-v1" else (0.25, 0.18)
    result.extend([max(clearance, heights[0]), max(clearance, heights[1]), *origin])
    grasped = float(bool(observation.get("grasped")))
    for xyz in (ee, landmarks.get("sleeve", [0, 0, 0]), landmarks.get("hem", [0, 0, 0])):
        result.extend(np.asarray(xyz) * grasped)
    return np.asarray(result, dtype=np.float32)


def validate_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("cloth policy requires local HxWx3 uint8 RGB")
    if min(rgb.shape[:2]) < 16:
        raise ValueError("cloth RGB must be at least 16x16")
    return rgb


def make_network(architecture: str):
    import torch
    from torch import nn

    class ClothNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            if architecture == ARCHITECTURES["left"]:
                self.visual = nn.Sequential(
                    nn.Conv2d(3, 12, 5, stride=2, padding=2),
                    nn.SiLU(),
                    nn.Conv2d(12, 24, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d((2, 2)),
                    nn.Flatten(),
                    nn.Linear(96, 24),
                    nn.SiLU(),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(24 + STATE_DIM + len(PHASES), 48), nn.SiLU(), nn.Linear(48, 4)
                )
            elif architecture == ARCHITECTURES["right"]:
                self.patches = nn.Conv2d(3, 24, 8, stride=8)
                self.context = nn.Linear(STATE_DIM + len(PHASES), 24)
                self.attention = nn.MultiheadAttention(24, 3, batch_first=True)
                self.fusion = nn.Sequential(
                    nn.LayerNorm(24), nn.Linear(24, 40), nn.GELU(), nn.Linear(40, 4)
                )
            else:
                raise ValueError(f"unknown cloth architecture: {architecture}")
            # Learned phase-conditioned affine experts keep the small-data model
            # precise enough to contact a 35 mm grasp region. No hand-coded
            # control gains or geometric target formulas appear in inference.
            self.experts = nn.Parameter(torch.zeros(len(PHASES), STATE_DIM + 1, 4))

        def forward(self, rgb, state, words):
            bias = torch.ones((len(state), 1), device=state.device, dtype=state.dtype)
            affine = torch.einsum(
                "bp,psa,bs->ba", words, self.experts, torch.cat([state, bias], -1)
            )
            if architecture == ARCHITECTURES["left"]:
                residual = self.fusion(torch.cat([self.visual(rgb), state, words], -1))
            else:
                patches = self.patches(rgb).flatten(2).transpose(1, 2)
                context = self.context(torch.cat([state, words], -1)).unsqueeze(1)
                visual, _ = self.attention(context, patches, patches, need_weights=False)
                residual = self.fusion((context + visual)[:, 0])
            return affine + residual * 0.01

    return ClothNetwork()


class ClothPolicy:
    def __init__(self, agent: str, checkpoint: str | Path | None = None):
        import torch

        if agent not in ARCHITECTURES:
            raise ValueError("cloth agent must be left or right")
        torch.set_num_threads(2)
        self.torch = torch
        self.agent = agent
        self.path = Path(checkpoint or DEFAULT_DIRECTORY / f"{agent}.pt")
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != FORMAT
            or payload.get("agent") != agent
            or payload.get("phases") != list(PHASES)
            or payload.get("architecture") != ARCHITECTURES[agent]
        ):
            raise ValueError("incompatible cloth policy checkpoint")
        self.network = make_network(ARCHITECTURES[agent]).eval().requires_grad_(False)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.training = payload["training"]
        self.feature_basis = payload.get("feature_basis", "original-v1")
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.calls = 0
        self.seconds = 0.0
        self.last_prediction = {}
        self.reference_origin = None

    @property
    def model_info(self):
        return {
            "model_id": f"relay-lab/cloth-{self.agent}-v1",
            "architecture": ARCHITECTURES[self.agent],
            "policy_class": "original task-trained state-assisted RGB/language/action model",
            "foundation_vla": False,
            "feature_basis": self.feature_basis,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameters": sum(p.numel() for p in self.network.parameters()),
            "inputs": [
                "local RGB",
                "local measured landmarks and visibility masks",
                "own end-effector/grasp state",
                "phase instruction",
                "coordinator clearance",
            ],
            "outputs": ["Cartesian target metres", "grip probability"],
            "training": self.training,
            "inference_calls": self.calls,
            "inference_seconds": self.seconds,
            "limitations": [
                "Not a pretrained foundation VLA; only the fixed instruction vocabulary is supported.",
                "Explicit local landmark measurements assist RGB; vision necessity is not established.",
                "High-level phases, clearance decisions, IK and motor limits are separate components.",
            ],
        }

    def act(
        self, observation: dict, instruction: str, clearance: float = 0.20, hold: bool = False
    ) -> dict:
        if observation.get("agent") != self.agent:
            raise ValueError("policy received another arm's observation")
        rgb = validate_rgb(observation.get("rgb"))
        if self.reference_origin is None:
            shoulder = observation.get("landmarks", {}).get("shoulder")
            if shoulder is None:
                raise ValueError("first local observation needs a shoulder calibration landmark")
            sign = -1 if self.agent == "left" else 1
            self.reference_origin = (np.asarray(shoulder)[:2] - [sign * 0.24, 0.28]).tolist()
        local = {**observation, "reference_origin": self.reference_origin}
        state = observation_features(local, clearance, self.feature_basis)
        words = language_features(instruction)
        # A missing grasp landmark is an unavailable observation, not a zero
        # world coordinate. Fail closed instead of making up the hidden target.
        normalized = PHASES[int(words.argmax())]
        landmark = "sleeve" if "sleeve" in normalized else "hem"
        if normalized.startswith(("approach_", "grasp_")) and landmark not in observation.get(
            "landmarks", {}
        ):
            hold = True
        if normalized in (
            "carry_hem",
            "fold_hem",
            "release_hem",
        ) and "shoulder" not in observation.get("landmarks", {}):
            hold = True
        torch = self.torch
        pixels = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
        pixels = torch.nn.functional.interpolate(
            pixels, size=(32, 32), mode="bilinear", align_corners=False
        )
        started = perf_counter()
        with torch.inference_mode():
            prediction = self.network(
                pixels, torch.from_numpy(state)[None], torch.from_numpy(words)[None]
            )[0].numpy()
        self.calls += 1
        self.seconds += perf_counter() - started
        if prediction.shape != (4,) or not np.isfinite(prediction).all():
            raise RuntimeError("cloth model produced nonfinite action; no fallback")
        probability = float(1 / (1 + np.exp(-np.clip(prediction[3], -40, 40))))
        target = prediction[:3]
        grip = probability >= 0.5
        if hold:
            target = state[:3].copy()
            grip = bool(observation.get("grasped"))
        self.last_prediction = {
            "raw": prediction.tolist(),
            "grip_probability": probability,
            "instruction": normalized,
            "coordinator_hold": bool(hold),
            "checkpoint_sha256": self.checkpoint_sha256,
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        }
        return {"target": target.tolist(), "grip": grip}

    def close(self):
        pass
