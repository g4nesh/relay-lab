"""Small, task-adapted vision/language/action policy trained in Relay Lab.

This is a separate learned policy, not a fine-tuned SmolVLA checkpoint. Its
pixel classifier, image-to-workspace calibration, language routing, target
embeddings, and Cartesian action network are learned from original simulator
examples. Evaluation receives only local RGB, proprioception, and instruction
text. Simulator object positions and the scripted teacher never enter inference.
"""

import hashlib
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .types import AGENTS, MAX_DELTA, Action, Instruction, Observation

DEFAULT_CHECKPOINT = Path("checkpoints/relay-vla/policy.pt")
WORDS = (
    "pick",
    "collect",
    "place",
    "wait",
    "keep",
    "return",
    "coral",
    "mint",
    "gold",
    "relay",
    "slot",
    "1",
    "2",
    "3",
    "home",
    "source",
    "clear",
    "packing",
    "pad",
)


def language_features(text: str) -> np.ndarray:
    """A bounded bag-of-words language encoder; unsupported prose is not a plan."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Relay VLA requires a nonempty instruction")
    tokens = set(re.findall(r"[a-z]+|[123]", text.lower()))
    return np.asarray([float(word in tokens) for word in WORDS], dtype=np.float32)


def make_network():
    """Lazy imports keep the lightweight simulator usable without torch."""
    import torch
    from torch import nn

    class RelayNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.pixels = nn.Sequential(
                nn.Conv2d(3, 24, 1), nn.SiLU(), nn.Conv2d(24, 24, 1), nn.SiLU(), nn.Conv2d(24, 4, 1)
            )
            # Per camera projective calibration, fitted from rendered examples.
            self.calibration = nn.Parameter(torch.zeros(2, 6, 2))
            self.language = nn.Linear(len(WORDS), 8)
            self.kind = nn.Linear(len(WORDS), 3)
            # relay, slot1, slot2, slot3, home; camera/role-specific home.
            self.targets = nn.Parameter(torch.zeros(2, 5, 3))
            self.control = nn.Sequential(
                nn.Linear(12, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 6),
            )

        def locate(self, rgb, role):
            logits = self.pixels(rgb)
            prob = logits.softmax(dim=1)[:, 1:].pow(12)
            height, width = prob.shape[-2:]
            x = torch.linspace(-1, 1, width, device=rgb.device)
            y = torch.linspace(-1, 1, height, device=rgb.device)
            mass = prob.sum(dim=(-1, -2)).clamp_min(1e-6)
            u = (prob * x[None, None, None, :]).sum(dim=(-1, -2)) / mass
            v = (prob * y[None, None, :, None]).sum(dim=(-1, -2)) / mass
            poly = torch.stack([torch.ones_like(u), u, v, u * v, u * u, v * v], dim=-1)
            xy = torch.einsum("bcf,bfd->bcd", poly, self.calibration[role])
            z = torch.full_like(xy[..., :1], 0.035)
            return torch.cat([xy, z], dim=-1), mass

        def goal(self, rgb, words, role):
            objects, mass = self.locate(rgb, role)
            candidates = torch.cat([objects, self.targets[role]], dim=1)
            routing = self.language(words).softmax(dim=-1)
            goal = (routing[..., None] * candidates).sum(dim=1)
            kinds = self.kind(words).softmax(dim=-1)
            return goal, kinds, routing, mass

        def action(self, position, goal, kinds):
            # Relative pose is a fixed representation transform, not a controller.
            inputs = torch.cat([position, goal, goal - position, kinds], dim=-1)
            return self.control(inputs)

    return RelayNetwork()


class RelayVLABackend:
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu"):
        import torch

        path = Path(checkpoint or DEFAULT_CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(f"Train the original Relay VLA first: missing {path}")
        if device == "auto":
            device = "cpu"  # This small network is faster without GPU synchronization.
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be auto, cpu, mps, or cuda")
        self.torch = torch
        self.device = device
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "relay-vla-v1" or payload.get("words") != list(WORDS):
            raise ValueError("Incompatible Relay VLA checkpoint")
        self.network = make_network().to(device).eval().requires_grad_(False)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.checkpoint = str(path.resolve())
        with path.open("rb") as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.training = payload["training"]
        self.inference_calls = 0
        self.total_inference_seconds = 0.0
        self.last_inference_seconds = 0.0
        self.seed = int(self.training["seed"])

    @property
    def model_info(self):
        return {
            "policy": "relay-vla",
            "model_id": "relay-lab/compact-vla-v1",
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameters": sum(p.numel() for p in self.network.parameters()),
            "device": self.device,
            "training": self.training,
            "inference_calls": self.inference_calls,
            "total_inference_seconds": self.total_inference_seconds,
            "inputs": ["local front RGB", "end-effector position", "instruction text"],
            "limitations": [
                "A task-specific learned controller, not SmolVLA or a general-purpose foundation VLA.",
                "Vision is calibrated to Relay Lab's fixed cameras; a new scene requires retraining.",
                "The visually estimated goal is retained for one instruction to avoid gripper occlusion.",
            ],
        }


class RelayVLAPolicy:
    needs_rgb = True

    def __init__(self, agent_id, backend: RelayVLABackend):
        if agent_id not in AGENTS:
            raise ValueError("Unknown agent")
        self.agent_id = agent_id
        self.backend = backend
        self.reset()

    def reset(self):
        self._text = None
        self._goal = None
        self._kinds = None
        self.inference_calls = 0
        self.last_prediction = None
        self._goal_tick = None

    def act(self, observation: Observation, instruction: Instruction) -> Action:
        if observation.agent_id != self.agent_id:
            raise ValueError("Policy received another agent's observation")
        rgb = observation.rgb
        if rgb is None or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Relay VLA requires local RGB")
        position = np.asarray(observation.ee_position, dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Relay VLA requires finite proprioception")
        torch = self.backend.torch
        started = perf_counter()
        visual_refresh = self._text != instruction.text or self._goal is None or instruction.replan
        with torch.inference_mode():
            if visual_refresh:
                image = (
                    torch.from_numpy(rgb.copy())
                    .permute(2, 0, 1)[None]
                    .float()
                    .to(self.backend.device)
                    / 255
                )
                image = torch.nn.functional.interpolate(
                    image, size=(128, 128), mode="bilinear", align_corners=False
                )
                words = torch.from_numpy(language_features(instruction.text))[None].to(
                    self.backend.device
                )
                role = torch.tensor([AGENTS.index(self.agent_id)], device=self.backend.device)
                self._goal, self._kinds, routing, mass = self.backend.network.goal(
                    image, words, role
                )
                self._text = instruction.text
                self._goal_tick = getattr(observation, "tick", None)
                self._routing = routing.cpu().numpy()[0].tolist()
                self._mass = mass.cpu().numpy()[0].tolist()
            state = torch.from_numpy(position)[None].to(self.backend.device)
            output = self.backend.network.action(state, self._goal, self._kinds)[0].cpu().numpy()
            goal = self._goal.cpu().numpy()[0].tolist()
        if not np.isfinite(output).all():
            raise RuntimeError("Relay VLA returned invalid actions; no fallback executed")
        delta = output[:3] * MAX_DELTA
        norm = float(np.linalg.norm(delta))
        if norm > MAX_DELTA:
            delta *= MAX_DELTA / norm
        gripper = int(np.argmax(output[3:])) - 1
        action = Action(tuple(float(x) for x in delta), gripper)
        elapsed = perf_counter() - started
        self.inference_calls += 1
        self.backend.inference_calls += 1
        self.backend.last_inference_seconds = elapsed
        self.backend.total_inference_seconds += elapsed
        self.last_prediction = {
            "model": "relay-vla",
            "model_prompt": instruction.text,
            "goal_estimate": goal,
            "goal_observation_tick": self._goal_tick,
            "visual_refresh": visual_refresh,
            "language_routing": self._routing,
            "pixel_mass": self._mass,
            "raw_network_output": output.tolist(),
            "delta_metres": list(action.delta),
            "gripper": gripper,
            "backend_inference_seconds": elapsed,
            "agent_inference_calls": self.inference_calls,
            "feedback_replan": instruction.replan,
            "chunk_index": 0,
        }
        return action
