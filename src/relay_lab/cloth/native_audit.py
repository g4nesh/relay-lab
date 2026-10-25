"""Check native-VLA command provenance in a recorded cloth execution.

This is an artifact consistency check. It does not rerun neural models, attest
to process execution, or independently verify the final cloth geometry. The
separate MuJoCo action replay is needed for physical execution verification.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from ..smolvla import DEFAULT_MODEL_ID, DEFAULT_REVISION
from .action_dataset import phase_instruction
from .policies import PHASES
from .pretrained import XVLA_ID, XVLA_REVISION, XVLA_WEIGHT_SHA256
from .smol_adaptation import (
    BASE_WEIGHT_SHA256,
    WORLD_EULER_CONTRACT,
    load_native_adapter,
    orientation_contract,
    unnormalize,
)
from .xvla_adaptation import load_adapter

AGENTS = ("left", "right")
LOWER = [-0.59, -0.49, 0.008]
UPPER = [0.59, 0.42, 0.52]


def _model_family(model, arm):
    """Choose only a supported publisher model, never a caller-supplied flag."""
    model_id = model.get("model_id")
    if model_id == DEFAULT_MODEL_ID:
        return "smol"
    if model_id == XVLA_ID and arm == "right":
        return "xvla"
    raise ValueError(f"Unsupported foundation model for {arm}: {model_id!r}")


def _required_sha256(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Native snapshot needs an explicit SHA-256 pin")
    return value


def _chunk_digest(chunk):
    return hashlib.sha256(
        json.dumps(chunk, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def audit_native_records(
    summary, records, *, smol_normalization=None, smol_normalizations=None
):
    """Validate all commands, caching, and explicit coordinator overrides."""
    issues = []
    state = {
        arm: {"id": 0, "age": None, "chunk": None, "phase": None, "held": False} for arm in AGENTS
    }
    counts = {arm: {"inferences": 0, "native_commands": 0, "holds": 0} for arm in AGENTS}
    phases = {arm: PHASES[0] for arm in AGENTS}
    tick = 0

    def check(condition, message, arm=None):
        if not condition:
            issues.append({"tick": tick, "agent": arm, "issue": message})
        return bool(condition)

    models = summary.get("models", {})
    normalizations = dict(smol_normalizations or {})
    check(set(normalizations) <= set(AGENTS), "normalization supplied for an unknown arm")
    if smol_normalization is not None:
        if "left" in normalizations:
            check(
                normalizations["left"] == smol_normalization,
                "conflicting left-arm native normalizations",
                "left",
            )
        else:
            normalizations["left"] = smol_normalization
    families = {}
    for arm in AGENTS:
        model = models.get(arm, {})
        check(model.get("foundation_vla") is True, "controller is not a foundation VLA", arm)
        try:
            family = _model_family(model, arm)
        except (ValueError, AttributeError) as exc:
            check(False, f"wrong foundation model pin: {exc}", arm)
            continue
        families[arm] = family
        is_smol = family == "smol"
        revision = DEFAULT_REVISION if is_smol else XVLA_REVISION
        base = BASE_WEIGHT_SHA256 if is_smol else XVLA_WEIGHT_SHA256
        check(
            model.get("revision") == revision,
            "wrong foundation model pin",
            arm,
        )
        key = "base_weight_sha256" if is_smol else "base_checkpoint_sha256"
        check(model.get(key) == base, "wrong foundation weight digest", arm)
        if is_smol and "native_module" in model:
            check(
                model["native_module"] == "model.action_out_proj",
                "wrong native action head for foundation model",
                arm,
            )
        check(
            model.get("agent", arm) == arm,
            "foundation model is assigned to the wrong arm",
            arm,
        )
        check(is_smol or arm not in normalizations, "Smol normalization assigned to X-VLA", arm)
        if is_smol and arm in normalizations:
            try:
                check(
                    model.get("orientation_contract", WORLD_EULER_CONTRACT)
                    == orientation_contract(normalizations[arm], agent=arm),
                    "model orientation contract differs from its native normalization",
                    arm,
                )
            except ValueError as exc:
                check(False, f"invalid native normalization contract: {exc}", arm)
        horizon = model.get("action_horizon")
        check(type(horizon) is int and 1 <= horizon <= 50, "invalid action horizon", arm)
        if is_smol and model.get("action_reuse") == "first_target":
            repeat = model.get("action_repeat_steps")
            check(
                horizon == 1 and type(repeat) is int and 1 <= repeat <= 50,
                "invalid first-target native action reuse",
                arm,
            )

    for record in records:
        check(
            record.get("decision_tick") == tick and record.get("tick") == tick + 1,
            "missing or nonconsecutive physics tick",
        )
        for arm in AGENTS:
            if arm not in families:
                continue
            is_smol = families[arm] == "smol"
            previous = state[arm]
            try:
                prediction = record["predictions"][arm]
                observation, action = record["observations"][arm], record["actions"][arm]
                target = np.asarray(action["target"], dtype=float)
                if (
                    target.shape != (3,)
                    or not np.isfinite(target).all()
                    or type(action["grip"]) is not bool
                ):
                    raise ValueError("invalid executed action")
                held = prediction.get("coordinator_hold") is True
                check(
                    held == bool(record["decisions"][arm]["hold"]),
                    "coordinator hold is not reflected in the executed action",
                    arm,
                )
                if held:
                    check(
                        np.array_equal(target, np.asarray(observation["ee_position"], dtype=float)),
                        "hold moved away from measured tool position",
                        arm,
                    )
                    check(
                        action["grip"] == bool(observation["gripper_closed"]),
                        "hold changed the commanded gripper",
                        arm,
                    )
                    previous["held"] = True
                    counts[arm]["holds"] += 1
                    continue
                native_flag = (
                    "native_action_executed" if is_smol else "native_action_head_executed"
                )
                check(
                    prediction.get(native_flag) is True,
                    "non-hold command lacks native inference provenance",
                    arm,
                )
                expected_prompt = phase_instruction(arm, phases[arm])
                check(
                    prediction.get("model_language_prompt", "").strip() == expected_prompt,
                    "native instruction does not match the active phase",
                    arm,
                )
                inference_id = prediction["inference_id"]
                age_key = "inference_age_ticks" if is_smol else "action_age_ticks"
                age = prediction[age_key]
                horizon = models[arm]["action_horizon"]
                repeat_first = is_smol and models[arm].get("action_reuse") == "first_target"
                if repeat_first:
                    horizon = models[arm]["action_repeat_steps"]
                if (
                    type(inference_id) is not int
                    or inference_id < 1
                    or type(age) is not int
                    or not 0 <= age < horizon
                ):
                    raise ValueError("invalid inference id or expired native action")
                chunk_key = "world_action_chunk" if is_smol else "raw_chunk"
                chunk = np.asarray(prediction[chunk_key], dtype=float)
                expected_shape = (50, 7) if is_smol else (30, 20)
                if chunk.shape != expected_shape or not np.isfinite(chunk).all():
                    raise ValueError("invalid native action chunk shape or values")
                if is_smol and arm in normalizations:
                    check(
                        prediction.get("orientation_contract", WORLD_EULER_CONTRACT)
                        == orientation_contract(normalizations[arm], agent=arm),
                        "prediction orientation contract differs from its native normalization",
                        arm,
                    )
                    normalized = np.asarray(prediction["normalized_native_chunk"], dtype=np.float32)
                    if normalized.shape != (50, 7) or not np.isfinite(normalized).all():
                        raise ValueError("invalid normalized native SmolVLA chunk")
                    check(
                        np.array_equal(
                            chunk, unnormalize(normalized, normalizations[arm]["action"])
                        ),
                        "world actions differ from the adapted native normalization",
                        arm,
                    )
                if is_smol and "native_module" in prediction:
                    check(
                        prediction["native_module"] == "model.action_out_proj",
                        "prediction names the wrong native action head",
                        arm,
                    )
                if not is_smol:
                    check(
                        bool(((chunk[:, [9, 19]] >= 0) & (chunk[:, [9, 19]] <= 1)).all()),
                        "X-VLA gripper outputs are outside native sigmoid range",
                        arm,
                    )
                digest = _chunk_digest(prediction[chunk_key])
                new_inference = inference_id != previous["id"]
                if new_inference:
                    check(
                        inference_id == previous["id"] + 1,
                        "native inference ids skipped or reversed",
                        arm,
                    )
                    check(age == 0, "new inference begins with a stale action", arm)
                    counts[arm]["inferences"] += 1
                else:
                    check(
                        not previous["held"],
                        "cached native action resumed after a coordination hold",
                        arm,
                    )
                    check(
                        previous["phase"] == phases[arm],
                        "cached native action crossed an instruction change",
                        arm,
                    )
                    check(
                        age == previous["age"] + 1,
                        "native action age did not advance one physics tick",
                        arm,
                    )
                    check(
                        digest == previous["chunk"],
                        "native chunk mutated without a new inference",
                        arm,
                    )
                index = prediction.get("used_index")
                check(
                    index == (age if is_smol and not repeat_first else 0),
                    "wrong native action chunk index",
                    arm,
                )
                if type(index) is not int or not 0 <= index < len(chunk):
                    raise ValueError("invalid native chunk index")
                raw = chunk[index]
                xyz = raw[:3] if is_smol else raw[10:13]
                grip = bool(raw[6] > 0) if is_smol else bool(raw[19] > 0.5)
                check(
                    np.array_equal(target, np.clip(xyz, LOWER, UPPER)),
                    "executed target differs from bounded native VLA output",
                    arm,
                )
                check(
                    action["grip"] == grip, "executed gripper differs from native VLA output", arm
                )
                state[arm] = {
                    "id": inference_id,
                    "age": age,
                    "chunk": digest,
                    "phase": phases[arm],
                    "held": False,
                }
                counts[arm]["native_commands"] += 1
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                check(False, f"malformed native provenance: {type(exc).__name__}: {exc}", arm)
        for arm in AGENTS:
            next_phase = record.get("phase", {}).get(arm)
            if check(next_phase in PHASES, "invalid recorded phase", arm):
                old_index, new_index = PHASES.index(phases[arm]), PHASES.index(next_phase)
                check(
                    new_index in (old_index, old_index + 1),
                    "phase sequence skipped or reversed",
                    arm,
                )
                phases[arm] = next_phase
        tick += 1
    check(tick == summary.get("ticks"), "summary tick count differs from trace")
    for arm in AGENTS:
        check(counts[arm]["inferences"] > 0, "arm never executed its foundation VLA", arm)
        check(
            counts[arm]["inferences"] == models.get(arm, {}).get("inference_calls"),
            "native inference count disagrees with model summary",
            arm,
        )
    return {
        "kind": "native_vla_command_consistency",
        "verified": not issues,
        "ticks": tick,
        "counts": counts,
        "model_families": families,
        "issues": issues,
        "reported_task_success": summary.get("success") is True,
        "reexecutes_neural_models": False,
        "reexecutes_physics": False,
        "limitations": "Checks recorded native commands and immutable adapter identities; independent physics replay and new model inference are separate evidence.",
    }


def audit_native_run(directory):
    """Bind each supported arm to its own adapter and normalization snapshot."""
    directory = Path(directory)
    # Parse and hash the same immutable bytes. A concurrent producer must not
    # make the reported digest describe a different execution than we audited.
    source_bytes = {
        name: (directory / name).read_bytes()
        for name in ("summary.json", "events.jsonl.gz", "final.json")
    }
    summary = json.loads(source_bytes["summary.json"])
    snapshot_issues = []
    pins = {}
    normalizations = {}
    model_bytes = {}

    def capture(path):
        raw = path.read_bytes()
        model_bytes[path] = raw
        return raw

    for arm in AGENTS:
        try:
            model = summary["models"][arm]
            family = _model_family(model, arm)
            snapshot = directory / "models" / f"{arm}.pt"
            raw = capture(snapshot)
            if family == "smol":
                adapter = load_native_adapter(
                    snapshot, arm,
                    expected_sha256=_required_sha256(model["checkpoint_sha256"]),
                )
                if hashlib.sha256(raw).hexdigest() != adapter["sha256"]:
                    raise ValueError("SmolVLA snapshot changed during adapter loading")
                normalizations[arm] = adapter["normalization"]
                pins[f"{arm}_adapter_sha256"] = adapter["sha256"]
            else:
                package = directory / "models" / "xvla"
                manifest = capture(package / "adapter.json")
                weights = capture(package / "native_decoder.safetensors")
                if raw != manifest:
                    raise ValueError("X-VLA snapshot manifest differs from loaded package")
                metadata, _, _ = load_adapter(
                    package, expected_sha256=_required_sha256(model["adapter_sha256"])
                )
                if model.get("adapter") != metadata or json.loads(manifest) != metadata:
                    raise ValueError("X-VLA summary differs from exact adapter manifest")
                if hashlib.sha256(weights).hexdigest() != metadata["decoder_sha256"]:
                    raise ValueError("X-VLA weights changed during adapter loading")
                pins[f"{arm}_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
                pins[f"{arm}_decoder_sha256"] = metadata["decoder_sha256"]
        except Exception as exc:  # noqa: BLE001 - damaged files fail closed in the report
            snapshot_issues.append(
                {
                    "agent": arm,
                    "issue": f"Invalid native model snapshots: {type(exc).__name__}: {exc}",
                }
            )
    event_lines = gzip.decompress(source_bytes["events.jsonl.gz"]).decode("utf-8").splitlines()
    report = audit_native_records(
        summary, (json.loads(line) for line in event_lines), smol_normalizations=normalizations
    )
    for path, captured in model_bytes.items():
        try:
            unchanged = path.read_bytes() == captured
        except OSError:
            unchanged = False
        if not unchanged:
            snapshot_issues.append(
                {"issue": f"Native model snapshot changed during validation: {path.name}"}
            )
    for name, captured in source_bytes.items():
        try:
            unchanged = (directory / name).read_bytes() == captured
        except OSError:
            unchanged = False
        if not unchanged:
            snapshot_issues.append(
                {"issue": f"Native audit source changed during validation: {name}"}
            )
    report["issues"] = snapshot_issues + report["issues"]
    report["verified"] = not report["issues"]
    report["model_snapshot_sha256"] = pins
    report["source_sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()
    }
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = audit_native_run(args.directory)
    (args.directory / "native-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["verified"] else 1)


if __name__ == "__main__":
    main()
