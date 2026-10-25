"""Offline right-arm imitation in a native SmolVLA left-arm distribution.

This is a mixed-controller data collector, never a two-VLA task demonstration.
Only the right teacher's actually executed, single-step labels are exported.
The existing runtime owns phase guards, cancellation, dynamics and scoring.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .action_dataset import phase_instruction, smol_action, smol_state
from .policies import DEFAULT_DIRECTORY, PHASES
from .runtime import ClothConfig, run_cloth

MIXED_SCHEMA = "relay.cloth.mixed-native-teacher-imitation.v1"
SEED = 10007
MAX_TICKS = 1800
SAMPLE_EVERY = 4
SMOL_FORMAT = "relay-smolvla-native-final-block-v3"
LEARNER_KEYS = frozenset(
    (
        "rgb",
        "ee_position",
        "ee_rotation",
        "grasped",
        "gripper_closed",
        "phase",
        "target",
        "grip",
        "tick",
    )
)
SOURCE_FILES = (
    "config.json",
    "summary.json",
    "events.jsonl.gz",
    "final.json",
    "models/left.pt",
    "models/right.pt",
    "models/communication.json",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _check_stop(should_stop):
    if should_stop is not None and should_stop():
        raise RuntimeError("Mixed imitation collection cancelled")


def _require_offline_assets():
    from huggingface_hub.constants import HF_HUB_OFFLINE

    if not HF_HUB_OFFLINE:
        raise RuntimeError(
            "Start mixed collection with HF_HUB_OFFLINE=1; model downloads are outside its budget"
        )


class _RecordingTeacher:
    """Record local inputs before the teacher call; export only committed steps."""

    def __init__(self, teacher, samples):
        self.teacher, self.samples = teacher, samples
        self.last_prediction = {}
        self._last_sample_phase = None

    @property
    def model_info(self):
        return self.teacher.model_info

    def act(self, observation, instruction, *, clearance, hold):
        tick = observation.get("tick")
        if observation.get("agent") != "right" or type(tick) is not int or tick < 0:
            raise ValueError("Mixed imitation requires a current right-arm observation")
        if instruction not in PHASES:
            raise ValueError("Unknown right-arm imitation phase")
        sample = not hold and (tick % SAMPLE_EVERY == 0 or instruction != self._last_sample_phase)
        # Select before invoking the privileged offline teacher. No landmark,
        # peer state, future observation or teacher activation enters this row.
        own = None
        if sample:
            rgb = np.asarray(observation["rgb"])
            if rgb.shape != (128, 128, 3) or rgb.dtype != np.uint8:
                raise ValueError("Mixed imitation camera must be the actual 128px native view")
            own = {
                "rgb": rgb.copy(),
                "ee_position": np.asarray(observation["ee_position"], dtype=np.float64).copy(),
                "ee_rotation": np.asarray(observation["ee_rotation"], dtype=np.float64).copy(),
                "grasped": observation["grasped"],
                "gripper_closed": observation["gripper_closed"],
                "phase": PHASES.index(instruction),
                "tick": tick,
            }
            if any(type(own[key]) is not bool for key in ("grasped", "gripper_closed")):
                raise ValueError("Invalid own gripper or attachment observation")
            smol_state(own["ee_position"], own["ee_rotation"], own["gripper_closed"])
        action = self.teacher.act(observation, instruction, clearance=clearance, hold=hold)
        self.last_prediction = dict(self.teacher.last_prediction)
        if own is not None and not self.last_prediction.get("coordinator_hold"):
            if type(action.get("grip")) is not bool:
                raise ValueError("Teacher gripper label must be a boolean")
            smol_action(action["target"], own["ee_rotation"], action["grip"])
            own.update(
                target=np.asarray(action["target"], dtype=np.float64).copy(), grip=action["grip"]
            )
            self.samples.append(own)
            self._last_sample_phase = instruction
            self.last_prediction["dataset_sample_rgb_sha256"] = hashlib.sha256(
                own["rgb"].tobytes()
            ).hexdigest()
        return action


def _read_source(directory):
    captured = {name: (directory / name).read_bytes() for name in SOURCE_FILES}
    summary = json.loads(captured["summary.json"])
    records = [
        json.loads(line) for line in gzip.decompress(captured["events.jsonl.gz"]).splitlines()
    ]
    if summary["ticks"] != len(records):
        raise ValueError("Mixed source tick count differs from executed trace")
    final = json.loads(captured["final.json"])
    if final.get("tick") != summary["ticks"] or final.get("metrics") != summary.get("metrics"):
        raise ValueError("Mixed source summary differs from measured final physics")
    return (
        summary,
        records,
        {name: hashlib.sha256(value).hexdigest() for name, value in captured.items()},
    )


def _physical_success(summary):
    return (
        summary.get("status") == "succeeded"
        and summary.get("success") is True
        and summary.get("metrics", {}).get("success") is True
        and summary.get("metrics", {}).get("physics_valid") is True
        and summary.get("error") is None
    )


def _valid_replay(replay, hashes, ticks, metrics):
    return (
        replay.get("kind") == "independent_mujoco_action_reexecution"
        and replay.get("verified") is True
        and replay.get("reexecutes_neural_models") is False
        and replay.get("steps") == ticks
        and replay.get("final_metrics") == metrics
        and metrics.get("success") is True
        and metrics.get("physics_valid") is True
        and all(
            replay.get(key) is True
            for key in (
                "final_vertices_bitwise_equal",
                "same_final_metrics",
                "same_fixture_state",
                "same_tick_count",
            )
        )
        and replay.get("source_artifact_sha256")
        == {name: hashes[name] for name in ("summary.json", "events.jsonl.gz", "final.json")}
    )


def _validate_source_identity(summary, hashes, expected):
    models, config = summary["models"], summary["config"]
    if (
        set(expected) != {"left", "right"}
        or config["seed"] != SEED
        or type(config["max_ticks"]) is not int
        or not 1 <= config["max_ticks"] <= MAX_TICKS
        or any(
            config[key] != value
            for key, value in {
                "mode": "structured_text",
                "scenario": "delayed",
                "coordination": "memory",
                "planner": "rules",
                "fixture": True,
            }.items()
        )
        or models["left"].get("foundation_vla") is not True
        or models["left"].get("adapter_format") != SMOL_FORMAT
        or models["left"].get("action_horizon") != 1
        or models["left"].get("action_repeat_steps") != 4
        or models["right"].get("foundation_vla") is not False
        or any(models[a]["checkpoint_sha256"] != expected[a] for a in expected)
        or any(hashes[f"models/{a}.pt"] != expected[a] for a in expected)
    ):
        raise ValueError("Mixed source controller identities differ from approved inputs")


def _align_samples(samples, records):
    """Reject label/observation drift and discard candidates never executed."""
    by_tick = {}
    before_phase = PHASES[0]
    for tick, record in enumerate(records):
        if record["decision_tick"] != tick or record["tick"] != tick + 1:
            raise ValueError("Mixed source must contain consecutive actual physics steps")
        after_phase = record["phase"]["right"]
        if after_phase not in PHASES or PHASES.index(after_phase) not in (
            PHASES.index(before_phase),
            PHASES.index(before_phase) + 1,
        ):
            raise ValueError("Mixed source has an invalid phase transition")
        by_tick[tick] = (record, before_phase)
        before_phase = after_phase
    aligned, previous = [], -1
    for row in samples:
        tick = int(row["tick"])
        if tick <= previous:
            raise ValueError("Mixed sample ticks must be strictly increasing")
        previous = tick
        if tick not in by_tick:
            # A cancellation can occur after prediction and before world.step.
            # That pending label has no measured outcome and is not exported.
            if tick != len(records):
                raise ValueError("Sample does not belong to the recorded source")
            continue
        record, phase = by_tick[tick]
        action, observation = record["actions"]["right"], record["observations"]["right"]
        prediction = record["predictions"]["right"]
        if (
            record["decisions"]["right"]["hold"]
            or prediction.get("coordinator_hold")
            or int(row["phase"]) != PHASES.index(phase)
            or not np.array_equal(row["target"], action["target"])
            or bool(row["grip"]) != action["grip"]
            or prediction.get("dataset_sample_rgb_sha256")
            != hashlib.sha256(row["rgb"].tobytes()).hexdigest()
        ):
            raise ValueError("Mixed label is not the actual executed teacher command/input")
        for key in ("ee_position", "ee_rotation", "grasped", "gripper_closed"):
            if not np.array_equal(row[key], observation[key]):
                raise ValueError(f"Mixed own observation differs from source: {key}")
        aligned.append(row)
    return aligned, len(samples) - len(aligned)


def _verify_physics(source, output, should_stop):
    """Run the existing independent replay with cancellation before every action."""
    _check_stop(should_stop)
    script = Path(__file__).resolve().parents[3] / "scripts/replay_cloth_actions.py"
    spec = importlib.util.spec_from_file_location("relay_mixed_imitation_replay", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Independent cloth replay implementation is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = module.ClothWorld

    class CancellableWorld(base):
        def __init__(self, **kwargs):
            _check_stop(should_stop)
            super().__init__(**kwargs)

        def release_fixture(self):
            _check_stop(should_stop)
            return super().release_fixture()

        def step(self, actions):
            _check_stop(should_stop)
            return super().step(actions)

    module.ClothWorld = CancellableWorld
    result = module.replay(source, output, video=False)
    _check_stop(should_stop)
    return result


def collect_mixed_episode(
    output,
    *,
    left_adapter,
    expected_left_sha256,
    max_ticks=MAX_TICKS,
    should_stop=None,
    on_progress=None,
):
    """One seed10007 scene, native Smol v3 left and offline CPU teacher right.

    No training, model download or retries occur here. This function uses MPS
    and must be called only after the integration owner grants its model lease.
    """
    if type(max_ticks) is not int or not 1 <= max_ticks <= MAX_TICKS:
        raise ValueError("Mixed collection is bounded to at most 1800 physics ticks")
    if (
        not isinstance(expected_left_sha256, str)
        or len(expected_left_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_left_sha256)
    ):
        raise ValueError("An exact Smol v3 adapter SHA-256 is required")
    _check_stop(should_stop)
    _require_offline_assets()
    left_adapter, output = Path(left_adapter), Path(output)
    right_adapter = DEFAULT_DIRECTORY / "right.pt"
    expected = {"left": expected_left_sha256, "right": _sha(right_adapter)}
    if _sha(left_adapter) != expected_left_sha256:
        raise ValueError("Smol adapter differs from the requested immutable candidate")
    config = ClothConfig(
        seed=SEED,
        mode="structured_text",
        scenario="delayed",
        coordination="memory",
        planner="rules",
        max_ticks=max_ticks,
        video=False,
    )
    samples = []

    def factory(arm, snapshot):
        _check_stop(should_stop)
        if arm == "left":
            from .smol_adaptation import SmolNativePolicy

            policy = SmolNativePolicy(
                "left",
                snapshot,
                device="mps",
                expected_sha256=expected["left"],
                action_horizon=1,
                action_repeat=4,
            )
            if policy.model_info.get("adapter_format") != SMOL_FORMAT:
                raise ValueError("Mixed collection requires the original native Smol v3 modules")
            return policy
        from .policies import ClothPolicy

        if arm != "right" or _sha(snapshot) != expected["right"]:
            raise ValueError("Offline teacher snapshot changed")
        teacher = ClothPolicy("right", snapshot)
        teacher.torch.set_num_threads(1)
        return _RecordingTeacher(teacher, samples)

    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    started = perf_counter()
    summary = run_cloth(
        config,
        source,
        checkpoint_paths={"left": left_adapter, "right": right_adapter},
        expected_checkpoint_hashes=expected,
        policy_factory=factory,
        should_stop=should_stop,
        on_progress=on_progress,
    )
    error, replay, source_hashes, aligned = None, {"verified": False}, {}, []
    discarded, source_success = 0, _physical_success(summary)
    try:
        summary, records, source_hashes = _read_source(source)
        _validate_source_identity(summary, source_hashes, expected)
        aligned, discarded = _align_samples(samples, records)
        if source_success:
            replay = _verify_physics(source, output / "verification", should_stop)
            if not _valid_replay(replay, source_hashes, summary["ticks"], summary["metrics"]):
                raise ValueError("Mixed imitation source lacks matching independent physics proof")
        if any(_sha(source / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("Mixed source artifacts changed during verification")
        _check_stop(should_stop)
    except Exception as exc:  # noqa: BLE001 - retain failed source and ineligible observations
        error = f"{type(exc).__name__}: {exc}"
    arrays = {key: np.asarray([row[key] for row in aligned]) for key in LEARNER_KEYS}
    np.savez_compressed(output / "right.npz", **arrays)
    verification = {
        "source_artifact_sha256": source_hashes,
        "physics": replay,
        "alignment_verified": error is None,
        "discarded_unexecuted_candidates": discarded,
        "teacher": "right only; labels are actual recorded commands, never unexecuted futures",
        "native_models_reexecuted": False,
        "error": error,
    }
    _json(output / "verification.json", verification)
    files = {f"source/{name}": digest for name, digest in source_hashes.items()}
    files.update({name: _sha(output / name) for name in ("right.npz", "verification.json")})
    replay_path = output / "verification/verification.json"
    if replay_path.exists():
        files["verification/verification.json"] = _sha(replay_path)
    manifest = {
        "schema": MIXED_SCHEMA,
        "kind": "right_imitation_with_native_left_controller",
        "seed": SEED,
        "split": "train",
        "agents": ["right"],
        "supervised_agent": "right",
        "samples": len(aligned),
        "sample_every": SAMPLE_EVERY,
        "sampling": "every four physics ticks and first unheld observation of each phase",
        "future_action_chunks_available": False,
        "source_status": summary.get("status", "failed"),
        "source_success": source_success,
        "source_ticks": summary.get("ticks", 0),
        "source_config": summary.get("config", {}),
        "source_models": summary.get("models", {}),
        "source_metrics": summary.get("metrics", {}),
        "source_artifact_sha256": source_hashes,
        "approved_checkpoint_sha256": expected,
        "success": source_success
        and replay.get("verified") is True
        and error is None
        and bool(aligned),
        "two_vla_success": False,
        "error": error,
        "files": files,
        "learner_inputs": [
            "own right RGB",
            "own measured pose",
            "own gripper command",
            "phase language",
        ],
        "instructions": [phase_instruction("right", phase) for phase in PHASES],
        "action_frame": "absolute MuJoCo world/table xyz in metres; binary commanded grip",
        "wall_seconds": perf_counter() - started,
        "limitations": [
            "Only the left arm uses a foundation VLA; the right arm is an offline state-assisted teacher.",
            "A successful mixed source is training data, not evidence of two-VLA task success.",
            "Sparse labels are one-step executed commands; no measured future action chunk is supplied.",
            "Native model execution is recorded, not independently repeated by the physics replay.",
        ],
    }
    _json(output / "manifest.json", manifest)
    return manifest


def load_mixed_episode(path, *, require_success=True):
    """Return right-only own inputs/one-step executed labels; failures reject by default."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("schema") != MIXED_SCHEMA
        or manifest.get("seed") != SEED
        or manifest.get("split") != "train"
        or manifest.get("agents") != ["right"]
        or manifest.get("supervised_agent") != "right"
        or manifest.get("two_vla_success") is not False
        or manifest.get("future_action_chunks_available") is not False
    ):
        raise ValueError("Not a right-only mixed-controller imitation dataset")
    if require_success and manifest.get("success") is not True:
        raise ValueError("Mixed source did not complete verified physical folding")
    required = {"right.npz", "verification.json", *(f"source/{name}" for name in SOURCE_FILES)}
    if not required <= set(manifest["files"]):
        raise ValueError("Mixed imitation evidence is incomplete")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or _sha(path / relative) != digest:
            raise ValueError("Mixed imitation artifact digest changed")
    summary, records, hashes = _read_source(path / "source")
    _validate_source_identity(summary, hashes, manifest["approved_checkpoint_sha256"])
    if any(
        manifest.get(key) != value
        for key, value in {
            "source_config": summary["config"],
            "source_models": summary["models"],
            "source_metrics": summary["metrics"],
            "source_status": summary["status"],
            "source_ticks": summary["ticks"],
            "source_success": _physical_success(summary),
        }.items()
    ):
        raise ValueError("Mixed source outcome differs from immutable raw run")
    proof = json.loads((path / "verification.json").read_text())
    if hashes != manifest["source_artifact_sha256"] or hashes != proof["source_artifact_sha256"]:
        raise ValueError("Mixed source and verification identities differ")
    if proof.get("alignment_verified") is not True or proof.get("error") is not None:
        raise ValueError("Mixed imitation source alignment is unverified")
    if require_success and not (
        _physical_success(summary)
        and _valid_replay(proof.get("physics", {}), hashes, summary["ticks"], summary["metrics"])
        and "verification/verification.json" in manifest["files"]
        and json.loads((path / "verification/verification.json").read_text()) == proof["physics"]
    ):
        raise ValueError("Mixed source lacks independently verified physical success")
    with np.load(path / "right.npz", allow_pickle=False) as data:
        if set(data.files) != LEARNER_KEYS:
            raise ValueError("Unexpected learner fields, possibly privileged state")
        arrays = {key: data[key].copy() for key in data.files}
    n = manifest["samples"]
    if type(n) is not int or n < 1:
        raise ValueError("Invalid mixed imitation sample count")
    shapes = {
        "rgb": (n, 128, 128, 3),
        "ee_position": (n, 3),
        "ee_rotation": (n, 3, 3),
        "target": (n, 3),
        **{key: (n,) for key in ("grasped", "gripper_closed", "phase", "grip", "tick")},
    }
    if n < 1 or any(
        value.shape != shapes[key] or not np.isfinite(value).all() for key, value in arrays.items()
    ):
        raise ValueError("Invalid mixed imitation input shapes or values")
    if arrays["rgb"].dtype != np.uint8 or any(
        arrays[key].dtype.kind != "b" for key in ("grasped", "gripper_closed", "grip")
    ):
        raise ValueError("Invalid mixed camera or gripper dtype")
    if any(arrays[key].dtype.kind not in "iu" for key in ("phase", "tick")) or np.any(
        (arrays["phase"] < 0) | (arrays["phase"] >= len(PHASES))
    ):
        raise ValueError("Invalid mixed phase/tick indices")
    rows = [{key: values[index] for key, values in arrays.items()} for index in range(n)]
    aligned, discarded = _align_samples(rows, records)
    if discarded or len(aligned) != n:
        raise ValueError("Mixed labels include unexecuted predictions")
    for row in aligned:
        smol_state(row["ee_position"], row["ee_rotation"], row["gripper_closed"])
        smol_action(row["target"], row["ee_rotation"], row["grip"])
    return manifest, {"right": arrays}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-adapter", type=Path, required=True)
    parser.add_argument("--left-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    args = parser.parse_args()
    result = collect_mixed_episode(
        args.output,
        left_adapter=args.left_adapter,
        expected_left_sha256=args.left_sha256,
        max_ticks=args.max_ticks,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "samples", "success", "source_status", "error")}
        )
    )
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
