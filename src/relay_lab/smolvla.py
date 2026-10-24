"""Pretrained SmolVLA inference with an explicit, approximate embodiment adapter.

The LIBERO checkpoint predicts 7D normalized OSC commands. Relay Lab can execute
translation and grip only; predicted rotation is retained in the audit log and
discarded. The checkpoint has not been trained on this scene. There is no scripted
fallback, including for wait instructions, and no simulator object pose enters
the VLA. High-level orchestration remains a separate component.
"""

import os
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np

from .types import AGENTS, MAX_DELTA, Action, AgentId, Instruction, Observation

DEFAULT_MODEL_ID = "HuggingFaceVLA/smolvla_libero"
DEFAULT_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
OSC_TRANSLATION_SCALE = 0.05  # robosuite OSC_POSE output_max, metres
GRIPPER_DEADBAND = 0.3
FEEDBACK_TOKEN_BUDGET = 512


def _rgb(value: np.ndarray | None, name: str) -> np.ndarray:
    if (
        value is None
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] != 3
        or min(value.shape[:2]) < 1
    ):
        raise ValueError(f"SmolVLA requires {name} RGB as an H x W x 3 uint8 array")
    return value


class SmolVLABackend:
    """One shared, frozen model; queues belong to the individual robot policies.

    Heavy dependencies and checkpoint loading happen only on construction. Saved
    preprocessing/normalization statistics come from the same pinned snapshot as
    the weights. Downloads use HF_HOME, or this workspace's .cache/huggingface.
    """

    def __init__(
        self,
        device: str = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        cache_dir: str | Path | None = None,
        seed: int = 7,
        denoising_steps: int = 10,
    ) -> None:
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")
        cache = Path(cache_dir or os.environ.get("HF_HOME", Path.cwd() / ".cache/huggingface"))
        os.environ["HF_HOME"] = str(cache.resolve())
        try:
            import torch
            from huggingface_hub import snapshot_download
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLA
        except ImportError as error:
            raise RuntimeError(
                "SmolVLA needs its optional dependencies. Run `uv sync --extra vla` "
                "using Python 3.12 or 3.13, then retry."
            ) from error
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be auto, cpu, mps, or cuda")
        self.device, self.model_id, self.revision = device, model_id, revision
        self.seed, self.denoising_steps = seed, denoising_steps
        self.cache_dir = str(cache.resolve())
        self.inference_calls = 0
        self.total_inference_seconds = 0.0
        self.last_inference_seconds = 0.0
        self._torch = torch
        started = perf_counter()
        snapshot = snapshot_download(
            model_id,
            revision=revision,
            cache_dir=str(cache / "hub"),
            allow_patterns=["*.json", "*.safetensors", "README.md"],
        )
        config = PreTrainedConfig.from_pretrained(snapshot)
        if config.vlm_model_name != VLM_MODEL_ID:
            raise ValueError(f"Unsupported VLM assets: {config.vlm_model_name!r}")
        vlm_snapshot = snapshot_download(
            VLM_MODEL_ID,
            revision=VLM_REVISION,
            cache_dir=str(cache / "hub"),
            allow_patterns=["*.json", "*.txt"],
        )
        config.device = device
        config.load_vlm_weights = False  # The full policy already contains the VLM weights.
        config.vlm_model_name = vlm_snapshot
        config.num_steps = denoising_steps
        # The same budget is used with feedback on/off. Do not silently lose
        # execution evidence at the checkpoint's original 48-token limit.
        config.tokenizer_max_length = FEEDBACK_TOKEN_BUDGET
        self._model = LeRobotSmolVLA.from_pretrained(snapshot, config=config, strict=True)
        self._model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self._preprocess, self._postprocess = make_pre_post_processors(
            config,
            snapshot,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {
                    "tokenizer_name": vlm_snapshot,
                    "max_length": config.tokenizer_max_length,
                    "truncation": False,
                },
            },
        )
        self._config = config
        self._tokenizer = next(
            step.input_tokenizer
            for step in self._preprocess.steps
            if hasattr(step, "input_tokenizer")
        )
        self.last_prompt_tokens = 0
        self.last_model_prompt = ""
        self._parameter_count = sum(parameter.numel() for parameter in self._model.parameters())
        self.load_seconds = perf_counter() - started

    @property
    def model_info(self) -> dict:
        memory = {}
        try:
            import psutil

            memory["process_rss_gib"] = psutil.Process().memory_info().rss / 1024**3
        except ImportError:
            pass
        if self.device == "mps":
            memory["mps_allocated_gib"] = self._torch.mps.current_allocated_memory() / 1024**3
            memory["mps_driver_allocated_gib"] = self._torch.mps.driver_allocated_memory() / 1024**3
        return {
            "policy": "smolvla",
            "model_id": self.model_id,
            "revision": self.revision,
            "license": "Apache-2.0 (checkpoint model card)",
            "lerobot_version": "0.6.1",
            "device": self.device,
            "dtype": "float32",
            "parameters": self._parameter_count,
            "seed": self.seed,
            "noise_stream": "Independent agent/local-inference seeds; policy reset repeats its stream.",
            "denoising_steps": self.denoising_steps,
            "language_token_budget": self._config.tokenizer_max_length,
            "vlm_assets_model_id": VLM_MODEL_ID,
            "vlm_assets_revision": VLM_REVISION,
            "cache_dir": self.cache_dir,
            "load_seconds": self.load_seconds,
            "inference_calls": self.inference_calls,
            "total_inference_seconds": self.total_inference_seconds,
            "last_inference_seconds": self.last_inference_seconds,
            "memory": memory,
            "inputs": ["local front RGB", "local wrist RGB", "8D proprioception", "instruction"],
            "frame_translation_metres": {"picker": [0.35, 0.0, 0.90], "packer": [-0.35, 0.0, 0.90]},
            "action_translation_scale_metres": OSC_TRANSLATION_SCALE,
            "max_delta_norm_metres": MAX_DELTA,
            "limitations": [
                "Approximate frame/embodiment transfer; no Relay Lab fine-tuning or success claim.",
                "Predicted rotation is ignored by the fixed-orientation gripper.",
                "High-level instructions are still provided by the rule orchestrator.",
            ],
            "checkpoint_url": f"https://huggingface.co/{self.model_id}/tree/{self.revision}",
        }

    def predict(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state8: np.ndarray,
        instruction_text: str,
        *,
        noise_seed: int | None = None,
    ) -> np.ndarray:
        """Return [chunk,7] after official dataset unnormalization, before OSC clipping."""
        torch = self._torch
        front, wrist = _rgb(front_rgb, "front"), _rgb(wrist_rgb, "wrist")
        state = np.asarray(state8, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("SmolVLA proprioception must contain eight finite values")
        if not isinstance(instruction_text, str) or not instruction_text.strip():
            raise ValueError("SmolVLA requires a nonempty instruction")
        # LeRobot's preceding newline processor applies this normalization.
        self.last_model_prompt = (
            instruction_text if instruction_text.endswith("\n") else instruction_text + "\n"
        )
        self.last_prompt_tokens = len(self._tokenizer(self.last_model_prompt)["input_ids"])
        if self.last_prompt_tokens > self._config.tokenizer_max_length:
            raise ValueError("Model instruction exceeds the language budget; shorten feedback")
        started = perf_counter()
        batch = {
            "observation.images.image": torch.from_numpy(front.copy()).permute(2, 0, 1).float()
            / 255,
            "observation.images.image2": torch.from_numpy(wrist.copy()).permute(2, 0, 1).float()
            / 255,
            "observation.state": torch.from_numpy(state.copy()),
            "task": self.last_model_prompt,
        }
        with torch.inference_mode():
            prepared = self._preprocess(batch)
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed if noise_seed is None else noise_seed
            )
            noise = torch.randn(
                (1, self._config.chunk_size, self._config.max_action_dim),
                generator=generator,
                dtype=torch.float32,
            ).to(self.device)
            # select_action owns a single queue; deliberately use only the chunk API.
            predicted = self._model.predict_action_chunk(prepared, noise=noise)
            actions = self._postprocess(predicted).detach().cpu().numpy()[0]
        self.last_inference_seconds = perf_counter() - started
        self.total_inference_seconds += self.last_inference_seconds
        self.inference_calls += 1
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise RuntimeError("SmolVLA returned an invalid action chunk; no fallback was executed")
        return actions


class SmolVLAPolicy:
    """Per-agent action chunk state, with strict local vision/proprioception input."""

    needs_rgb = True

    def __init__(self, agent_id: AgentId, backend: SmolVLABackend, action_horizon: int = 5) -> None:
        if agent_id not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_id!r}")
        if not isinstance(action_horizon, int) or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        self.agent_id, self.backend, self.action_horizon = agent_id, backend, action_horizon
        self.reset()

    def reset(self) -> None:
        self._actions: deque[np.ndarray] = deque()
        self._instruction_key: tuple | None = None
        self.last_prediction: dict | None = None
        self.inference_calls = 0
        self._chunk_index = 0
        self._chunk_inference_seconds = 0.0
        self._noise_seed: int | None = None
        self._model_prompt = ""
        self._model_prompt_tokens: int | None = None

    def act(self, observation: Observation, instruction: Instruction) -> Action:
        if observation.agent_id != self.agent_id:
            raise ValueError("SmolVLA policy received another agent's observation")
        front, wrist = _rgb(observation.rgb, "front"), _rgb(observation.wrist_rgb, "wrist")
        key = (instruction.kind, instruction.object_id, instruction.target_id, instruction.text)
        if key != self._instruction_key or instruction.replan:
            self._actions.clear()
            self._instruction_key = key
        if not self._actions:
            position = np.asarray(observation.ee_position, dtype=np.float32)
            # Translation only: world action axes therefore need no inverse rotation.
            position = position + np.array([0.35 if self.agent_id == "picker" else -0.35, 0, 0.90])
            half_opening = observation.gripper_opening / 2
            state = np.asarray(
                [*position, *observation.ee_orientation, half_opening, -half_opening],
                dtype=np.float32,
            )
            # Each robot advances its own stream regardless of its peer's call cadence.
            self._noise_seed = (
                self.backend.seed
                + (AGENTS.index(self.agent_id) + 1) * 1_000_000_007
                + self.inference_calls
            ) % (2**63 - 1)
            self._model_prompt = instruction.model_prompt
            chunk = np.asarray(
                self.backend.predict(
                    front, wrist, state, self._model_prompt, noise_seed=self._noise_seed
                )
            )
            if (
                chunk.ndim != 2
                or chunk.shape[0] == 0
                or chunk.shape[1] != 7
                or not np.isfinite(chunk).all()
            ):
                raise RuntimeError(
                    "SmolVLA returned an invalid action chunk; no fallback was executed"
                )
            self._actions.extend(row.copy() for row in chunk[: self.action_horizon])
            self._model_prompt_tokens = getattr(self.backend, "last_prompt_tokens", None)
            self._model_prompt = getattr(self.backend, "last_model_prompt", self._model_prompt)
            self.inference_calls += 1
            self._chunk_index = 0
            self._chunk_inference_seconds = self.backend.last_inference_seconds
        raw = self._actions.popleft()
        clipped = np.clip(raw, -1.0, 1.0)
        delta = clipped[:3] * OSC_TRANSLATION_SCALE
        length = float(np.linalg.norm(delta))
        delta *= min(1.0, MAX_DELTA / length) if length else 1.0
        # Panda +1 closes; Relay Lab -1 closes. Near-zero means maintain current grip.
        gripper = (
            -1 if clipped[6] > GRIPPER_DEADBAND else (1 if clipped[6] < -GRIPPER_DEADBAND else 0)
        )
        action = Action(tuple(float(value) for value in delta), gripper)
        self.last_prediction = {
            "raw_osc_action": raw.tolist(),
            "clipped_osc_action": clipped.tolist(),
            "discarded_rotation": raw[3:6].tolist(),
            "delta_metres": list(action.delta),
            "gripper": gripper,
            "chunk_index": self._chunk_index,
            "agent_inference_calls": self.inference_calls,
            "backend_inference_seconds": self._chunk_inference_seconds,
            "noise_seed": self._noise_seed,
            "model_prompt": self._model_prompt,
            "model_prompt_tokens": self._model_prompt_tokens,
            "feedback_replan": instruction.replan,
        }
        self._chunk_index += 1
        return action
