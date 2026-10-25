"""Original task-trained RGB/language/action network for five factory robots.

This compact VLA is trained from this project's own synthetic demonstrations.
It is not SmolVLA or a general-purpose foundation model. The inference boundary
contains RGB, text, local proprioception and optional verified peer latents;
no simulator target coordinates, teacher or waypoint generator is imported.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .vision import IMAGE_SHAPE, ROBOT_IDS, TARGETS, VIEW_WIDTH, sensor_signature

DEFAULT_CHECKPOINT = Path("checkpoints/factory-vla/policy.pt")
SPACE_ID = "factory-vla-v1"
KINDS = ("navigate", "pick", "place", "process", "wait")
BODY_KINDS = ("humanoid", "amr", "arm", "gantry")
WORDS = (*KINDS, "on", "to", *TARGETS)


def language_features(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Factory VLA requires an instruction")
    tokens = set(re.findall(r"[a-z]+(?:-\d+|-in|-out)?", text.lower()))
    return np.asarray([float(word in tokens) for word in WORDS], dtype=np.float32)


def make_network():
    import torch
    from torch import nn

    class FactoryNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.pixels = nn.Sequential(
                nn.Conv2d(3, 48, 1),
                nn.SiLU(),
                nn.Conv2d(48, 48, 1),
                nn.SiLU(),
                nn.Conv2d(48, len(TARGETS) + 1, 1),
            )
            self.calibration = nn.Parameter(torch.zeros(len(ROBOT_IDS), 13, 3))
            self.target_offsets = nn.Parameter(torch.zeros(len(TARGETS), 3))
            self.language = nn.Linear(len(WORDS), len(TARGETS))
            self.kind = nn.Linear(len(WORDS), len(KINDS))
            self.control = nn.Sequential(
                nn.Linear(24, 160),
                nn.SiLU(),
                nn.Linear(160, 160),
                nn.SiLU(),
                nn.Linear(160, 160),
                nn.SiLU(),
                nn.Linear(160, 7),
            )
            self.encoder = nn.Sequential(nn.Linear(15, 64), nn.SiLU(), nn.Linear(64, 32))
            self.decoder = nn.Sequential(
                nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 15)
            )
            self.peer_fusion = nn.Linear(6, 3, bias=False)

        def locate(self, rgb, role):
            # Shared learned per-pixel colour classifier over both real RGB views.
            logits = self.pixels(rgb)
            prob = logits.softmax(dim=1)[:, 1:].pow(10)
            views = prob.split(VIEW_WIDTH, dim=-1)
            points, masses = [], []
            for view in views:
                height, width = view.shape[-2:]
                xs = torch.linspace(-1, 1, width, device=rgb.device)
                ys = torch.linspace(-1, 1, height, device=rgb.device)
                mass = view.sum(dim=(-1, -2)).clamp_min(1e-9)
                u = (view * xs[None, None, None, :]).sum(dim=(-1, -2)) / mass
                v = (view * ys[None, None, :, None]).sum(dim=(-1, -2)) / mass
                points += [u, v]
                masses.append(mass)
            a, b, c, d = points
            poly = torch.stack(
                [
                    torch.ones_like(a),
                    a,
                    b,
                    c,
                    d,
                    a * a,
                    b * b,
                    c * c,
                    d * d,
                    a * b,
                    a * d,
                    b * d,
                    c * d,
                ],
                dim=-1,
            )
            pose = (
                torch.einsum("btf,bfd->btd", poly, self.calibration[role])
                + self.target_offsets[None]
            )
            return pose, torch.stack(masses, -1), poly

        def goal(self, rgb, words, role):
            poses, mass, _ = self.locate(rgb, role)
            routing = self.language(words).softmax(-1)
            kinds = self.kind(words).softmax(-1)
            goal = (routing[..., None] * poses).sum(1)
            return goal, kinds, routing, mass

        def motor(self, position, goal, kinds, body, yaw, holding, velocity):
            relative = goal - position
            x = torch.cat(
                [
                    position / 10,
                    goal / 10,
                    relative / 3,
                    kinds,
                    body,
                    yaw.sin(),
                    yaw.cos(),
                    holding,
                    velocity,
                ],
                -1,
            )
            return self.control(x)

        def encode(self, position, goal, kinds, body):
            inputs = torch.cat([position / 10, goal / 10, kinds, body], -1)
            latent = torch.nn.functional.normalize(self.encoder(inputs), dim=-1)
            return latent, self.decoder(latent)

    return FactoryNetwork()


class FactoryVLABackend:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
        import torch

        torch.set_num_threads(4)
        self.torch = torch
        if device not in ("cpu", "auto"):
            raise ValueError("This small calibrated factory VLA currently supports CPU inference")
        self.device = "cpu"
        path = Path(checkpoint or DEFAULT_CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(
                f"Factory VLA checkpoint missing: {path}; run python -m relay_lab.factory.train_vla train"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != SPACE_ID
            or payload.get("words") != list(WORDS)
            or payload.get("targets") != list(TARGETS)
        ):
            raise ValueError("Incompatible factory VLA checkpoint")
        if payload.get("sensor_signature") != sensor_signature():
            raise ValueError(
                "Factory VLA checkpoint and RGB sensor geometry do not match; retrain or recalibrate the original model"
            )
        self.sensor_signature = payload["sensor_signature"]
        self.network = make_network().eval().requires_grad_(False)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.checkpoint = str(path.resolve())
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.training = payload["training"]
        self.cache = {}
        self.inference_calls = 0
        self.total_inference_seconds = 0.0
        self.peer_fusion_calls = 0
        self.last_prediction = {}

    @property
    def model_info(self):
        return {
            "policy": "factory-vla",
            "model_id": "relay-lab/factory-vla-v1",
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameters": sum(p.numel() for p in self.network.parameters()),
            "sensor_signature": self.sensor_signature,
            "latent_space": SPACE_ID,
            "embedding_dimensions": 32,
            "device": self.device,
            "training": self.training,
            "inference_calls": self.inference_calls,
            "total_inference_seconds": self.total_inference_seconds,
            "peer_fusion_calls": self.peer_fusion_calls,
            "sensor": {
                "engine": "MuJoCo",
                "views": ["top", "oblique"],
                "rgb_shape": list(IMAGE_SHAPE),
                "physical_fiducial_classes": len(TARGETS),
                "command_highlights": False,
            },
            "inputs": [
                "MuJoCo top and oblique RGB",
                "instruction text",
                "local proprioception",
                "optional compatible peer embedding",
            ],
            "limitations": [
                "Original task-specific imitation model, not a pretrained foundation VLA.",
                "Calibrated fixed cameras and physical colour fiducials; no general visual reasoning.",
                "Fixed-stance humanoid, holonomic mobile carriers, kinematic manipulator links and idealized proximity-checked grasps.",
                "Visual goal held per instruction while target is stationary; numeric simulator goals never enter inference.",
            ],
        }

    def close(self):
        """Release independent per-robot context; GL sensors belong to the world."""
        self.cache.clear()

    def reset(self, robot_id=None):
        if robot_id is None:
            self.cache.clear()
        else:
            self.cache.pop(robot_id, None)

    def _inputs(self, obs):
        robot_id = obs.get("robot_id")
        body_name = obs.get("kind")
        if robot_id not in ROBOT_IDS or body_name not in BODY_KINDS:
            raise ValueError("Unknown factory embodiment")
        p = obs.get("proprio", {})
        position = np.asarray(
            p.get("position") if body_name == "amr" else p.get("ee_position"), dtype=np.float32
        )
        velocity = np.asarray(p.get("velocity", [0, 0, 0]), dtype=np.float32)
        yaw = float(p.get("yaw", 0))
        ee_position = np.asarray(p.get("ee_position"), dtype=np.float32)
        if ee_position.shape != (3,) or not np.isfinite(ee_position).all():
            raise ValueError("Factory VLA requires finite local proprioception")
        if (
            position.shape != (3,)
            or velocity.shape != (3,)
            or not np.isfinite(np.r_[position, velocity, yaw]).all()
        ):
            raise ValueError("Factory VLA requires finite local proprioception")
        body = np.eye(4, dtype=np.float32)[BODY_KINDS.index(body_name)]
        t = self.torch
        return (
            t.from_numpy(position)[None],
            t.from_numpy(velocity)[None],
            t.tensor([[yaw]]),
            t.from_numpy(body)[None],
            t.tensor([[float(bool(p.get("holding")))]]),
            robot_id,
        )

    def _goal(self, obs, position, body):
        t = self.torch
        robot_id = obs["robot_id"]
        text = obs.get("instruction", "wait")
        words_np = language_features(text)
        if not any(words_np[: len(KINDS)]):
            raise ValueError("Instruction has no supported learned skill")
        words = t.from_numpy(words_np)[None]
        kind = self.network.kind(words).softmax(-1)
        if text.strip() == "wait":
            return position.clone(), kind, None, False
        cached = self.cache.get(robot_id)
        cache_key = (text, obs.get("command_id"))
        if cached is None or cached["key"] != cache_key or obs.get("refresh_visual"):
            rgb = obs.get("rgb")
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != IMAGE_SHAPE:
                raise ValueError(f"Factory VLA requires RGB with shape {IMAGE_SHAPE}")
            image = t.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
            role = t.tensor([ROBOT_IDS.index(robot_id)])
            goal, kind, routing, mass = self.network.goal(image, words, role)
            target_index = int(routing.argmax(-1)[0])
            if float(routing[0, target_index]) < 0.98:
                raise RuntimeError("Ambiguous learned target grounding; no fallback")
            if float(mass[0, target_index].min()) < 0.2:
                raise RuntimeError(
                    f"Target fiducial not visible in both RGB cameras for {robot_id}: "
                    f"{TARGETS[target_index]}, per-view learned mass={mass[0, target_index].tolist()}; no fallback"
                )
            cached = {
                "key": cache_key,
                "goal": goal.detach(),
                "kind": kind.detach(),
                "target": TARGETS[target_index],
                "routing": routing[0].tolist(),
                "mass": mass[0, target_index].tolist(),
                "tick": obs.get("rgb_tick", obs.get("tick")),
            }
            self.cache[robot_id] = cached
        goal = cached["goal"].clone()
        peer_used = False
        peer = obs.get("peer_embedding")
        # Cross-robot fusion is constrained by the typed transport, then checked
        # again against this instruction and checkpoint before affecting motors.
        if (
            isinstance(peer, dict)
            and cached["target"] in ("amr-01", "amr-02")
            and int(kind.argmax(-1)[0]) == KINDS.index("place")
        ):
            values = peer.get("embedding", peer.get("vector"))
            source = peer.get("robot_id", peer.get("sender", peer.get("source_robot")))
            if (
                source == cached["target"]
                and peer.get("checkpoint_sha256") == self.checkpoint_sha256
                and peer.get("space_id", peer.get("latent_space")) == SPACE_ID
            ):
                latent = np.asarray(values, dtype=np.float32)
                if (
                    latent.shape == (32,)
                    and np.isfinite(latent).all()
                    and abs(float(np.linalg.norm(latent)) - 1) < 0.002
                ):
                    decoded = self.network.decoder(t.from_numpy(latent)[None])
                    peer_pose = decoded[:, :3] * 10
                    # Shared pose channel is always tool/deck centre, including AMRs.
                    goal = self.network.peer_fusion(t.cat([goal, peer_pose], -1))
                    peer_used = True
                    self.peer_fusion_calls += 1
        return goal, kind, cached, peer_used

    def act(self, obs):
        started = perf_counter()
        t = self.torch
        with t.inference_mode():
            position, velocity, yaw, body, holding, robot_id = self._inputs(obs)
            goal, kinds, cached, peer_used = self._goal(obs, position, body)
            output = self.network.motor(position, goal, kinds, body, yaw, holding, velocity)[
                0
            ].numpy()
            # The codec publishes own tool/deck centre, not base position.
            ee = t.tensor([obs["proprio"]["ee_position"]], dtype=t.float32)
            latent, _decoded = self.network.encode(ee, goal, kinds, body)
        if not np.isfinite(output).all():
            raise RuntimeError("Nonfinite learned motor output; no fallback")
        dt = float(obs.get("dt", 0.1))
        if not 0.01 <= dt <= 0.2:
            raise ValueError("Unsupported control interval")
        delta = output[:3] * 0.1 * (dt / 0.1)
        yaw_delta = float(np.clip(output[3], -1, 1) * 0.12 * (dt / 0.1))
        grip = int(np.argmax(output[4:])) - 1
        elapsed = perf_counter() - started
        self.inference_calls += 1
        self.total_inference_seconds += elapsed
        prediction = {
            "instruction": obs["instruction"],
            "goal_estimate": goal[0].tolist(),
            "visual_observation_tick": cached.get("tick") if cached else obs.get("tick"),
            "target": cached.get("target") if cached else None,
            "peer_fusion": peer_used,
            "raw_output": output.tolist(),
            "inference_seconds": elapsed,
        }
        self.last_prediction[robot_id] = prediction
        return {
            "delta": delta.tolist(),
            "yaw_delta": yaw_delta,
            "gripper": grip,
            "embedding": latent[0].tolist(),
            "provenance": {
                "model_id": "relay-lab/factory-vla-v1",
                "checkpoint_sha256": self.checkpoint_sha256,
                **prediction,
            },
        }

    def embed(self, obs):
        t = self.torch
        with t.inference_mode():
            position, _, _, body, _, robot_id = self._inputs(obs)
            # Refreshing RGB is not needed to report a local proprioceptive pose;
            # the most recent learned visual goal/intent remain in the same code.
            cached = self.cache.get(robot_id)
            goal = cached["goal"] if cached else position
            words = t.from_numpy(language_features(obs.get("instruction", "wait")))[None]
            kinds = self.network.kind(words).softmax(-1)
            ee = t.tensor([obs["proprio"]["ee_position"]], dtype=t.float32)
            latent, decoded = self.network.encode(ee, goal, kinds, body)
            decoded_np = decoded[0].numpy()
        intent = KINDS[int(np.argmax(decoded_np[6:11]))]
        pose = (decoded_np[:3] * 10).tolist()
        return {
            "embedding": latent[0].tolist(),
            "model_id": "relay-lab/factory-vla-v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "latent_space": SPACE_ID,
            "space_id": SPACE_ID,
            "robot_id": robot_id,
            "decoded": {
                "text": f"{robot_id}: {intent}; learned tool/deck pose {pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f} m",
                "confidence": float(
                    self.training.get("codec_validation", {}).get("within_65mm", 0.0)
                ),
                "version": SPACE_ID,
                "pose": pose,
                "intent": intent,
            },
        }
