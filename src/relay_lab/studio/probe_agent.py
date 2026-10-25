"""Observation-only reference adapter for one real, pinned SmolVLA inference.

Run via relay_lab.studio.worker with an observation connection for the released
SmolVLA left slot. This checks the objective-to-policy path, not robot execution.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

OBJECTIVE = "Predict one action chunk for holding the left arm in its current position. Do not execute the actions."
CRITERIA = [
    "The released SmolVLA checkpoint returns a finite action chunk with seven channels.",
    "No predicted action is executed in the simulator or on hardware.",
]
_backend = None


def handle(context):
    global _backend
    task, worker = context["task"], context["worker"]
    blocked = {"status": "blocked", "progress": 0, "evidence": []}
    if task["objective"] != OBJECTIVE or task["success_criteria"] != CRITERIA:
        return {
            **blocked,
            "summary": "This reference adapter only supports its documented observation-only policy probe objective and criteria.",
        }
    from relay_lab.smolvla import DEFAULT_MODEL_ID, DEFAULT_REVISION, SmolVLABackend

    assignment = task["assignments"][worker["agent_id"]]
    source = assignment["spec"]["source"]
    if (
        worker["agent_id"] != "left"
        or worker["execution_kind"] != "observation"
        or source != {"kind": "huggingface", "ref": DEFAULT_MODEL_ID, "revision": DEFAULT_REVISION}
    ):
        return {
            **blocked,
            "summary": "Use an observation connection for the released pinned SmolVLA left agent.",
        }
    if context["should_stop"]():
        return {**blocked, "summary": "Task authorization changed before inference."}

    import numpy as np

    from relay_lab.environment import RelayEnv

    if _backend is None:
        _backend = SmolVLABackend(device=os.environ.get("RELAY_AGENT_DEVICE", "auto"))
    # Fresh local cameras + measured proprioception. No spectator object state.
    with RelayEnv(seed=41) as env:
        obs = env.observe("picker", rgb=True)
        position = np.asarray(obs.ee_position) + np.array([0.35, 0, 0.90])
        half = obs.gripper_opening / 2
        proprioception = np.asarray([*position, *obs.ee_orientation, half, -half], dtype=np.float32)
        chunk = _backend.predict(
            obs.rgb, obs.wrist_rgb, proprioception, task["objective"], noise_seed=41
        )
        unchanged_tick = env.tick == 0
    finite = bool(np.isfinite(chunk).all() and chunk.ndim == 2 and chunk.shape[1] == 7)
    proof = {
        "task_id": task["id"],
        "objective": task["objective"],
        "agent_id": worker["agent_id"],
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_REVISION,
        "observation": "fresh MuJoCo local front/wrist RGB + 8D proprioception",
        "actions": chunk.tolist(),
        "shape": list(chunk.shape),
        "finite": finite,
        "inference_seconds": _backend.last_inference_seconds,
        "model_prompt": _backend.last_model_prompt,
        "simulator_tick_unchanged": unchanged_tick,
        "executed_actions": 0,
        "physical_task_completed": False,
    }
    evidence = Path(os.environ.get("RELAY_AGENT_EVIDENCE_DIR", ".relay-agent/evidence"))
    evidence.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(proof, indent=2, allow_nan=False).encode()
    evidence.joinpath(task["id"] + ".json").write_bytes(encoded)
    passed = finite and unchanged_tick
    return {
        "status": "succeeded" if passed else "failed",
        "progress": 100 if passed else 0,
        "summary": f"SmolVLA received the assigned objective and returned a finite {chunk.shape[0]}×7 action chunk in {_backend.last_inference_seconds:.2f}s. No actions were executed; this completes the inference probe only.",
        "evidence": [
            f"Checkpoint: {DEFAULT_MODEL_ID}@{DEFAULT_REVISION}",
            f"Local evidence SHA-256: {hashlib.sha256(encoded).hexdigest()}",
            "Fresh local MuJoCo observations; simulator tick unchanged; executed actions: 0.",
        ],
    }
