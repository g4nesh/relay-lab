"""Learned private RGB route representation for communication experiments.

A sender sees an actual MuJoCo inspection-camera crop, encodes it as a unit
32-vector, and a receiver decodes the hidden representation into route A or B.
The strong text baseline uses the identical encoder and decoder before sending
its structured route text. No route scalar enters neural inference.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from time import perf_counter

import numpy as np

DEFAULT_CHECKPOINT = Path("checkpoints/factory-routing/policy.pt")
SPACE_ID = "factory-route-v1"
MODEL_ID = "relay-lab/private-inspection-codec-v1"
RGB_SHAPE = (64, 64, 3)
ROUTES = ("A", "B")
COLORS = ((0.88, 0.23, 0.045), (0.035, 0.61, 0.88))
SENSOR_SPEC = {
    "version": "private-inspection-mujoco-v1",
    "image_shape": list(RGB_SHAPE),
    "indicator_colors": [list(c) for c in COLORS],
    "camera": {"position": [0, -0.65, 1.10], "xyaxes": [1, 0, 0, 0, 0.852, 0.523], "fovy": 42},
    "indicator_radius": 0.105,
    "fixture_size": [0.36, 0.28, 0.025],
    "randomization": "local pose, illumination, cue intensity, metal/background colours, unrelated status LEDs",
}
SENSOR_SIGNATURE = hashlib.sha256(
    json.dumps(SENSOR_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class RoutingObservationError(ValueError):
    """The learned sensor head cannot support a confident routing instruction."""


class InspectionCamera:
    """Simulator-only physical indicator renderer; never an inference input."""

    def __init__(self):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string("""<mujoco>
          <visual><global offwidth="64" offheight="64"/><quality shadowsize="1024"/>
          <headlight ambient=".25 .25 .25" diffuse=".25 .25 .25" specular=".05 .05 .05"/></visual>
          <asset><material name="lamp" emission=".25" specular=".1" shininess=".3"/></asset>
          <worldbody><light name="inspection-light" pos="-.4 -.4 1.7" diffuse=".8 .8 .8"/>
          <camera name="inspection" pos="0 -.65 1.10" xyaxes="1 0 0 0 .852 .523" fovy="42"/>
          <geom name="bench" type="box" pos="0 0 -.07" size=".8 .8 .05" rgba=".22 .25 .28 1"/>
          <body name="fixture" mocap="true" pos="0 0 0">
            <geom name="panel" type="box" size=".36 .28 .025" rgba=".50 .53 .57 1"/>
            <geom name="bezel" type="cylinder" size=".14 .014" pos="0 0 .037" rgba=".075 .08 .09 1"/>
            <geom name="indicator" type="cylinder" size=".105 .012" pos="0 0 .054" material="lamp" rgba=".88 .23 .045 1"/>
            <geom name="led0" type="sphere" size=".025" pos="-.24 -.16 .045" rgba=".15 .4 .7 1"/>
            <geom name="led1" type="sphere" size=".025" pos=".24 -.16 .045" rgba=".7 .2 .15 1"/>
            <geom name="led2" type="sphere" size=".025" pos="-.24 .16 .045" rgba=".6 .6 .15 1"/>
            <geom name="led3" type="sphere" size=".025" pos=".24 .16 .045" rgba=".2 .6 .3 1"/>
            <geom type="box" size=".16 .014 .006" pos="0 -.205 .035" rgba=".075 .08 .09 1"/>
          </body></worldbody></mujoco>""")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=64, width=64)
        self._fixture = int(self.model.body("fixture").mocapid[0])

    def render(self, route, seed):
        """``None`` turns the physical indicator off for training negatives."""
        if route not in (None, "A", "B", 0, 1):
            raise ValueError("Inspection indicator route must be A or B")
        rng = np.random.default_rng(int(seed))
        label = ROUTES.index(route) if isinstance(route, str) else route
        color = (
            np.array(COLORS[int(label)])
            if label is not None
            else np.full(3, rng.uniform(0.04, 0.5))
        )
        color = np.clip(color * rng.uniform(0.72, 1.12) + rng.normal(0, 0.015, 3), 0.01, 0.98)
        self.model.geom_rgba[self.model.geom("indicator").id, :3] = color
        metal = rng.uniform(0.28, 0.66)
        self.model.geom_rgba[self.model.geom("panel").id, :3] = np.clip(
            metal + rng.normal(0, 0.025, 3), 0.15, 0.8
        )
        self.model.geom_rgba[self.model.geom("bench").id, :3] = rng.uniform(0.08, 0.35, 3)
        for i in range(4):
            self.model.geom_rgba[self.model.geom(f"led{i}").id, :3] = rng.uniform(0.04, 0.95, 3)
        light = rng.uniform(0.45, 1.15)
        self.model.light_diffuse[0] = [light] * 3
        self.data.mocap_pos[self._fixture] = [
            rng.uniform(-0.035, 0.035),
            rng.uniform(-0.025, 0.025),
            rng.uniform(-0.006, 0.006),
        ]
        yaw = rng.uniform(-0.20, 0.20)
        self.data.mocap_quat[self._fixture] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="inspection")
        rgb = self.renderer.render().copy()
        # Sensor noise is independent of the private route and shared by modes.
        noise = rng.normal(0, rng.uniform(0, 1.8), rgb.shape)
        return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def close(self):
        self.renderer.close()


_LOCAL = threading.local()


def render_inspection_cue(route, seed):
    if route not in ("A", "B", 0, 1):
        raise ValueError("Runtime inspection route must be A or B")
    camera = getattr(_LOCAL, "camera", None)
    if camera is None:
        camera = InspectionCamera()
        _LOCAL.camera = camera
    return camera.render(route, seed)


def close_inspection_renderers():
    camera = getattr(_LOCAL, "camera", None)
    if camera is not None:
        camera.close()
        del _LOCAL.camera


def make_network():
    import torch
    from torch import nn

    class RoutingNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Sequential(
                nn.Conv2d(3, 12, 5, 2, 2),
                nn.SiLU(),
                nn.Conv2d(12, 24, 3, 2, 1),
                nn.SiLU(),
                nn.Conv2d(24, 32, 3, 2, 1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 32),
            )
            self.route_head = nn.Linear(32, 3)

        def encode(self, rgb):
            return torch.nn.functional.normalize(self.visual(rgb), dim=-1)

        def decode(self, latent):
            return self.route_head(latent) * 10

        def forward(self, rgb):
            latent = self.encode(rgb)
            return latent, self.decode(latent)

    return RoutingNetwork()


class RoutingCodec:
    """CPU neural perception/representation only; no world handle or route input."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        import torch

        torch.set_num_threads(4)
        self.torch = torch
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Train the private routing codec first: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != SPACE_ID or payload.get("sensor_signature") != SENSOR_SIGNATURE:
            raise ValueError("Routing checkpoint or private camera contract mismatch")
        self.network = make_network().eval().requires_grad_(False)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.checkpoint = str(path.resolve())
        self.training = payload["training"]
        self.space_id = SPACE_ID
        self.encode_calls = self.decode_calls = 0
        self.encode_seconds = self.decode_seconds = 0.0

    @property
    def model_info(self):
        return {
            "model_id": MODEL_ID,
            "space_id": SPACE_ID,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint": self.checkpoint,
            "sensor_signature": SENSOR_SIGNATURE,
            "parameters": sum(p.numel() for p in self.network.parameters()),
            "embedding_dimensions": 32,
            "training": self.training,
            "encode_calls": self.encode_calls,
            "decode_calls": self.decode_calls,
            "encode_seconds": self.encode_seconds,
            "decode_seconds": self.decode_seconds,
            "inputs": ["private 64x64 MuJoCo RGB crop"],
            "limitations": [
                "Task-specific indicator classifier/latent codec, not a pretrained VLA.",
                "Confidence is a neural softmax score, not a calibrated probability.",
                "Binary text baseline uses exactly the same learned visual inference.",
            ],
        }

    def encode(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != RGB_SHAPE:
            raise ValueError("Routing encoder requires uint8 private RGB with shape64x64x3")
        started = perf_counter()
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
        with self.torch.inference_mode():
            latent = self.network.encode(tensor)[0].tolist()
        self.encode_calls += 1
        self.encode_seconds += perf_counter() - started
        return {
            "vector": latent,
            "space_id": SPACE_ID,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": MODEL_ID,
            "source_frame_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
        }

    def decode(self, vector):
        values = np.asarray(vector, dtype=np.float32)
        if (
            values.shape != (32,)
            or not np.isfinite(values).all()
            or abs(float(np.linalg.norm(values)) - 1) > 0.002
        ):
            raise ValueError("Routing decoder requires a finite normalized32D vector")
        started = perf_counter()
        with self.torch.inference_mode():
            probability = (
                self.network.decode(self.torch.from_numpy(values)[None]).softmax(-1)[0].numpy()
            )
        index = int(probability.argmax())
        confidence = float(probability[index])
        self.decode_calls += 1
        self.decode_seconds += perf_counter() - started
        if index == 2 or confidence < 0.90:
            raise RoutingObservationError(
                "Private indicator is unreadable or the learned route is uncertain"
            )
        route = ROUTES[index]
        return {
            "route": route,
            "confidence": confidence,
            "text": json.dumps({"route": route}, separators=(",", ":")),
        }

    def close(self):
        """Renderer lifetime belongs to the worker; this codec owns no graphics."""
