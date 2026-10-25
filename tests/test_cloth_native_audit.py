"""Proof checks reject controller substitution and stale native action records."""

import copy
import gzip
import hashlib
import json

import numpy as np
import pytest

from relay_lab.cloth.action_dataset import phase_instruction
from relay_lab.cloth.native_audit import audit_native_records, audit_native_run
from relay_lab.cloth.pretrained import XVLA_ID, XVLA_REVISION, XVLA_WEIGHT_SHA256
from relay_lab.cloth.smol_adaptation import (
    BASE_WEIGHT_SHA256,
    RIGHT_YAW_CONTRACT,
    save_native_adapter,
    unnormalize,
)
from relay_lab.cloth.xvla_adaptation import save_adapter
from relay_lab.smolvla import DEFAULT_MODEL_ID, DEFAULT_REVISION


def example():
    phase = "approach_sleeve_above"
    summary = {
        "ticks": 5,
        "success": False,
        "models": {
            "left": {
                "model_id": DEFAULT_MODEL_ID,
                "revision": DEFAULT_REVISION,
                "base_weight_sha256": BASE_WEIGHT_SHA256,
            },
            "right": {
                "model_id": XVLA_ID,
                "revision": XVLA_REVISION,
                "base_checkpoint_sha256": XVLA_WEIGHT_SHA256,
            },
        },
    }
    for model in summary["models"].values():
        model.update(foundation_vla=True, action_horizon=8, inference_calls=2)
    records = []
    for tick in range(5):
        held = tick == 2
        inference = 1 if tick < 3 else 2
        age = tick if tick < 2 else tick - 3
        record = {
            "decision_tick": tick,
            "tick": tick + 1,
            "phase": {a: phase for a in ("left", "right")},
            "predictions": {},
            "observations": {},
            "actions": {},
            "decisions": {},
        }
        for arm in ("left", "right"):
            observation = {"ee_position": [0.1, 0.2, 0.3], "gripper_closed": True}
            record["observations"][arm] = observation
            record["decisions"][arm] = {"hold": held}
            if held:
                record["actions"][arm] = {"target": observation["ee_position"], "grip": True}
                record["predictions"][arm] = {"coordinator_hold": True}
                continue
            chunk = np.zeros((50, 7) if arm == "left" else (30, 20))
            if arm == "left":
                chunk[:, :3] = [0.1, 0.2, 0.3]
                chunk[:, 6] = 1
            else:
                chunk[:, 10:13] = [0.1, 0.2, 0.3]
                chunk[:, 19] = 1
            record["actions"][arm] = {"target": [0.1, 0.2, 0.3], "grip": True}
            record["predictions"][arm] = {
                "coordinator_hold": False,
                "native_action_executed" if arm == "left" else "native_action_head_executed": True,
                "inference_id": inference,
                "inference_age_ticks" if arm == "left" else "action_age_ticks": age,
                "world_action_chunk" if arm == "left" else "raw_chunk": chunk.tolist(),
                "model_language_prompt": phase_instruction(arm, phase),
                "used_index": age if arm == "left" else 0,
            }
        records.append(record)
    return summary, records


def smol_pair_example():
    summary, records = example()
    summary["models"]["right"] = copy.deepcopy(summary["models"]["left"])
    for arm in ("left", "right"):
        summary["models"][arm].update(agent=arm, native_module="model.action_out_proj")
    for record in records:
        record["predictions"]["right"] = copy.deepcopy(record["predictions"]["left"])
        if not record["predictions"]["right"]["coordinator_hold"]:
            record["predictions"]["right"]["model_language_prompt"] = phase_instruction(
                "right", "approach_sleeve_above"
            )
    return summary, records


def test_native_commands_and_explicit_holds_are_consistent_without_task_success_claim():
    summary, records = example()
    report = audit_native_records(summary, records)
    assert report["verified"]
    assert report["counts"]["left"] == {"inferences": 2, "native_commands": 4, "holds": 1}
    assert not report["reported_task_success"]
    assert not report["reexecutes_neural_models"]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_canonical_commands_cannot_be_replaced_by_a_teacher(arm):
    summary, records = example()
    records[0]["actions"][arm]["target"][0] += 0.01
    report = audit_native_records(summary, records)
    assert not report["verified"]
    assert any("bounded native" in issue["issue"] for issue in report["issues"])


def test_a_cache_cannot_resume_after_a_hold_or_cross_a_phase_change():
    summary, records = example()
    records[3]["predictions"]["right"]["inference_id"] = 1
    report = audit_native_records(summary, records)
    assert not report["verified"]
    assert any("after a coordination hold" in issue["issue"] for issue in report["issues"])
    summary, records = example()
    records[0]["phase"]["left"] = "approach_sleeve"
    report = audit_native_records(summary, records)
    assert any("crossed an instruction" in issue["issue"] for issue in report["issues"])


def test_new_inference_cannot_silently_mutate_cached_chunk_or_skip_identity():
    summary, records = example()
    records[1]["predictions"]["left"]["world_action_chunk"][2][0] = 0.4
    records[3]["predictions"]["right"]["inference_id"] = 3
    report = audit_native_records(summary, records)
    assert not report["verified"]
    assert any("mutated" in issue["issue"] for issue in report["issues"])
    assert any("skipped or reversed" in issue["issue"] for issue in report["issues"])


def test_previous_tiny_controller_and_zero_native_calls_are_rejected():
    summary, records = example()
    summary["models"]["left"].update(foundation_vla=False, model_id="relay-lab/cloth-left-v1")
    report = audit_native_records(summary, records)
    assert not report["verified"]
    assert any("not a foundation" in issue["issue"] for issue in report["issues"])
    summary, _ = example()
    summary["ticks"] = 0
    report = audit_native_records(summary, [])
    assert any("never executed" in issue["issue"] for issue in report["issues"])


def test_hold_must_preserve_measured_pose_and_commanded_gripper():
    summary, records = example()
    records[2]["actions"]["left"] = {"target": [0, 0, 0], "grip": False}
    report = audit_native_records(summary, records)
    assert any("hold moved" in issue["issue"] for issue in report["issues"])
    assert any("hold changed" in issue["issue"] for issue in report["issues"])


def test_native_normalization_is_checked_against_raw_model_output():
    summary, records = example()
    normalization = {"action": {"mean": [0] * 7, "std": [1] * 7}}
    for record in records:
        prediction = record["predictions"]["left"]
        if not prediction["coordinator_hold"]:
            chunk = np.asarray(prediction["world_action_chunk"], np.float32).tolist()
            prediction["world_action_chunk"] = chunk
            prediction["normalized_native_chunk"] = copy.deepcopy(chunk)
            record["actions"]["left"]["target"] = chunk[0][:3]
    assert audit_native_records(summary, records, smol_normalization=normalization)["verified"]
    records[0]["predictions"]["left"]["normalized_native_chunk"][0][0] += 0.1
    report = audit_native_records(summary, records, smol_normalization=normalization)
    assert any("native normalization" in issue["issue"] for issue in report["issues"])


def test_first_target_reuse_never_executes_untrained_future_tokens():
    summary, records = example()
    summary["models"]["left"].update(
        action_horizon=1, action_reuse="first_target", action_repeat_steps=8
    )
    for record in records:
        if not record["predictions"]["left"]["coordinator_hold"]:
            record["predictions"]["left"]["used_index"] = 0
    assert audit_native_records(summary, records)["verified"]
    records[1]["predictions"]["left"]["used_index"] = 1
    report = audit_native_records(summary, records)
    assert any("chunk index" in issue["issue"] for issue in report["issues"])


def test_two_smol_models_use_their_own_chunk_and_normalization_contracts():
    summary, records = smol_pair_example()
    normalizations = {
        "left": {"action": {"mean": [0] * 7, "std": [1] * 7}},
        "right": {"action": {"mean": [0.04, -0.03, 0.02, 0, 0, 0, -1], "std": [2] * 7}},
    }
    for record in records:
        for arm in ("left", "right"):
            prediction = record["predictions"][arm]
            if prediction["coordinator_hold"]:
                continue
            normalized = np.asarray(prediction["world_action_chunk"], np.float32)
            world = unnormalize(normalized, normalizations[arm]["action"])
            prediction["normalized_native_chunk"] = normalized.tolist()
            prediction["world_action_chunk"] = world.tolist()
            record["actions"][arm]["target"] = np.clip(
                world[0, :3], [-0.59, -0.49, 0.008], [0.59, 0.42, 0.52]
            ).tolist()
    report = audit_native_records(summary, records, smol_normalizations=normalizations)
    assert report["verified"]
    assert report["model_families"] == {"left": "smol", "right": "smol"}
    assert report["counts"]["right"] == {"inferences": 2, "native_commands": 4, "holds": 1}
    # The legacy argument still supplies only the left arm.
    assert audit_native_records(
        summary, records,
        smol_normalization=normalizations["left"],
        smol_normalizations={"right": normalizations["right"]},
    )["verified"]
    normalizations["right"] = normalizations["left"]
    report = audit_native_records(summary, records, smol_normalizations=normalizations)
    assert any(
        issue["agent"] == "right" and "native normalization" in issue["issue"]
        for issue in report["issues"]
    )


def test_right_yaw_contract_is_bound_even_though_rotation_is_not_actuated():
    summary, records = smol_pair_example()
    normalization = {
        "orientation_contract": RIGHT_YAW_CONTRACT,
        "action": {"mean": [0] * 7, "std": [1] * 7},
    }
    summary["models"]["right"]["orientation_contract"] = RIGHT_YAW_CONTRACT
    for record in records:
        prediction = record["predictions"]["right"]
        if prediction["coordinator_hold"]:
            continue
        chunk = np.asarray(prediction["world_action_chunk"], np.float32).tolist()
        prediction.update(
            world_action_chunk=chunk,
            normalized_native_chunk=copy.deepcopy(chunk),
            orientation_contract=RIGHT_YAW_CONTRACT,
        )
        record["actions"]["right"]["target"] = chunk[0][:3]
    assert audit_native_records(
        summary, records, smol_normalizations={"right": normalization}
    )["verified"]
    records[0]["predictions"]["right"]["orientation_contract"] = "world_xyz_euler_v1"
    report = audit_native_records(summary, records, smol_normalizations={"right": normalization})
    assert any("prediction orientation contract" in issue["issue"] for issue in report["issues"])


@pytest.mark.parametrize("arm", ["left", "right"])
def test_two_smol_reuse_expiry_integer_index_and_hold_rules_apply_to_each_arm(arm):
    summary, records = smol_pair_example()
    summary["models"][arm].update(
        action_horizon=1, action_reuse="first_target", action_repeat_steps=8
    )
    for record in records:
        if not record["predictions"][arm]["coordinator_hold"]:
            record["predictions"][arm]["used_index"] = 0
    assert audit_native_records(summary, records)["verified"]
    bad = copy.deepcopy(records)
    bad[1]["predictions"][arm]["used_index"] = False
    assert not audit_native_records(summary, bad)["verified"]
    bad = copy.deepcopy(records)
    bad[1]["predictions"][arm]["inference_age_ticks"] = 8
    assert not audit_native_records(summary, bad)["verified"]
    bad = copy.deepcopy(records)
    bad[3]["predictions"][arm]["inference_id"] = 1
    assert any(
        "after a coordination hold" in issue["issue"]
        for issue in audit_native_records(summary, bad)["issues"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": "unapproved/smol-like", "foundation_vla": True},
        {"revision": "unapproved-revision"},
        {"base_weight_sha256": "0" * 64},
        {"native_module": "replacement_controller"},
        {"agent": "left"},
    ],
)
def test_right_smol_model_pin_and_head_cannot_be_selected_by_a_flag(changes):
    summary, records = smol_pair_example()
    summary["models"]["right"].update(changes)
    assert not audit_native_records(summary, records)["verified"]


def test_model_family_must_match_the_native_chunk_and_xvla_is_right_only():
    summary, records = example()
    summary["models"]["right"] = copy.deepcopy(summary["models"]["left"])
    assert not audit_native_records(summary, records)["verified"]
    summary, records = example()
    summary["models"]["left"] = copy.deepcopy(summary["models"]["right"])
    report = audit_native_records(summary, records)
    assert any("Unsupported foundation model for left" in i["issue"] for i in report["issues"])


def write_full_audit_fixture(tmp_path, *, two_smol=False):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    summary, records = smol_pair_example() if two_smol else example()
    models = tmp_path / "models"
    models.mkdir()
    normalization = {
        "source": "training_split_only",
        "state": {"mean": [0] * 8, "std": [1] * 8},
        "action": {"mean": [0] * 7, "std": [1] * 7},
    }
    left_sha = save_native_adapter(
        models / "left.pt",
        "left",
        {"weight": np.zeros((32, 480), np.float32), "bias": np.zeros(32, np.float32), "fit": {}},
        normalization,
        {},
    )
    summary["models"]["left"]["checkpoint_sha256"] = left_sha
    right_normalization = copy.deepcopy(normalization)
    right_normalization["action"]["mean"][:3] = [0.04, -0.03, 0.02]
    if two_smol:
        summary["models"]["right"]["checkpoint_sha256"] = save_native_adapter(
            models / "right.pt", "right",
            {"weight": np.zeros((32, 480), np.float32), "bias": np.zeros(32, np.float32), "fit": {}},
            right_normalization, {},
        )
    else:
        right_meta = save_adapter(
            models / "xvla",
            np.zeros((1024, 20), np.float32),
            np.zeros(20, np.float32),
            dataset_sha256="a" * 64,
            training={},
        )
        (models / "right.pt").write_bytes((models / "xvla" / "adapter.json").read_bytes())
        summary["models"]["right"].update(
            adapter_sha256=right_meta["decoder_sha256"], adapter=right_meta
        )
    for record in records:
        for arm in ("left", "right") if two_smol else ("left",):
            p = record["predictions"][arm]
            if not p["coordinator_hold"]:
                normalized = np.asarray(p["world_action_chunk"], np.float32)
                stats = normalization if arm == "left" else right_normalization
                world = unnormalize(normalized, stats["action"])
                p["world_action_chunk"] = world.tolist()
                p["normalized_native_chunk"] = normalized.tolist()
                record["actions"][arm]["target"] = world[0, :3].tolist()
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "final.json").write_text("{}")
    with gzip.open(tmp_path / "events.jsonl.gz", "wt") as stream:
        stream.write("\n".join(json.dumps(record) for record in records))
    return left_sha


def test_full_audit_binds_native_adapter_snapshots_and_detects_sidecar_tampering(tmp_path):
    left_sha = write_full_audit_fixture(tmp_path)
    models = tmp_path / "models"
    report = audit_native_run(tmp_path)
    assert report["verified"]
    assert report["model_snapshot_sha256"]["left_adapter_sha256"] == left_sha
    (models / "xvla" / "native_decoder.safetensors").write_bytes(b"different weights")
    report = audit_native_run(tmp_path)
    assert not report["verified"]
    assert any("model snapshots" in issue["issue"] for issue in report["issues"])


def test_concurrent_source_replacement_cannot_change_which_bytes_are_attested(
    tmp_path, monkeypatch
):
    import relay_lab.cloth.native_audit as audit

    write_full_audit_fixture(tmp_path)
    original = (tmp_path / "final.json").read_bytes()
    original_auditor = audit.audit_native_records

    def replace_during_audit(*args, **kwargs):
        (tmp_path / "final.json").write_text('{"changed": true}')
        return original_auditor(*args, **kwargs)

    monkeypatch.setattr(audit, "audit_native_records", replace_during_audit)
    report = audit.audit_native_run(tmp_path)
    assert not report["verified"]
    assert report["source_sha256"]["final.json"] == hashlib.sha256(original).hexdigest()
    assert any("changed during validation" in item["issue"] for item in report["issues"])


def test_two_smol_snapshots_bind_separate_arm_weights_and_normalizations(tmp_path):
    left_sha = write_full_audit_fixture(tmp_path, two_smol=True)
    report = audit_native_run(tmp_path)
    assert report["verified"]
    assert report["model_snapshot_sha256"]["left_adapter_sha256"] == left_sha
    assert report["model_snapshot_sha256"]["right_adapter_sha256"] != left_sha
    assert report["model_families"] == {"left": "smol", "right": "smol"}
    assert not (tmp_path / "models/xvla").exists()
    # Matching a new hash does not authorize assigning the left adapter to right.
    (tmp_path / "models/right.pt").write_bytes((tmp_path / "models/left.pt").read_bytes())
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["models"]["right"]["checkpoint_sha256"] = left_sha
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    report = audit_native_run(tmp_path)
    assert not report["verified"]
    assert any(
        issue.get("agent") == "right" and "incompatible SmolVLA" in issue["issue"]
        for issue in report["issues"]
    )


def test_smol_snapshot_race_cannot_change_attested_weight_bytes(tmp_path, monkeypatch):
    import relay_lab.cloth.native_audit as audit

    write_full_audit_fixture(tmp_path, two_smol=True)
    original = (tmp_path / "models/right.pt").read_bytes()
    original_auditor = audit.audit_native_records

    def replace_during_audit(*args, **kwargs):
        (tmp_path / "models/right.pt").write_bytes(b"changed model")
        return original_auditor(*args, **kwargs)

    monkeypatch.setattr(audit, "audit_native_records", replace_during_audit)
    report = audit.audit_native_run(tmp_path)
    assert not report["verified"]
    assert report["model_snapshot_sha256"]["right_adapter_sha256"] == hashlib.sha256(original).hexdigest()
    assert any("snapshot changed during validation" in item["issue"] for item in report["issues"])


@pytest.mark.parametrize("two_smol", [False, True])
def test_snapshot_cannot_omit_its_right_adapter_digest(tmp_path, two_smol):
    write_full_audit_fixture(tmp_path, two_smol=two_smol)
    summary = json.loads((tmp_path / "summary.json").read_text())
    key = "checkpoint_sha256" if two_smol else "adapter_sha256"
    summary["models"]["right"][key] = None
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    report = audit_native_run(tmp_path)
    assert not report["verified"]
    assert any("explicit SHA-256 pin" in issue["issue"] for issue in report["issues"])
