"""Tampering must fail even when the reported terminal outcome stays unchanged."""

import gzip
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from relay_lab.cloth.communication import (
    Channel,
    LearnedFeatureCodec,
    observation_features,
)
from relay_lab.cloth.policies import ARCHITECTURES
from relay_lab.cloth.runtime import ClothConfig
from relay_lab.cloth.scene import shirt_mesh

SPEC = importlib.util.spec_from_file_location(
    "cloth_audit", Path(__file__).parents[1] / "scripts/audit_cloth_run.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as stream:
        json.dump(value, stream)


def rows(path):
    with gzip.open(path / "events.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream]


def write_rows(path, records):
    with gzip.open(path / "events.jsonl.gz", "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


@pytest.fixture
def evidence(tmp_path, request):
    """Tiny internally consistent failed observation trace, with no fabricated fold."""
    from dataclasses import asdict

    vertices, faces, landmarks = shirt_mesh()
    robots = {
        arm: {"ee": [sign * 0.38, -0.28, 0.28], "holding": None, "joint_positions": [0] * 6}
        for arm, sign in (("left", -1), ("right", 1))
    }
    observations = {
        arm: {
            "tick": 0,
            "agent": arm,
            "ee_position": robot["ee"],
            "grasped": False,
            "grasp_target": "none",
            "landmarks": {name: vertices[index].tolist() for name, index in landmarks[arm].items()},
            "obstacle": None,
        }
        for arm, robot in robots.items()
    }
    features = [
        observation_features(observations[arm]) for _ in range(32) for arm in ("left", "right")
    ]
    codec = LearnedFeatureCodec.fit(features, training={"split": "fixture"})
    codec.save(tmp_path / "codec.json")
    config = ClothConfig(
        seed=19,
        mode=getattr(request, "param", "structured_text"),
        scenario="clear",
        video=False,
        publish_every=1,
        codec=str(tmp_path / "codec.json"),
    )
    run_id = "cloth-audit-fixture"
    channel = Channel(run_id, mode=config.mode, codec=codec, network=config.network())
    final = {
        "tick": 3,
        "vertices": vertices.tolist(),
        "triangles": faces.tolist(),
        "robots": robots,
        "fixture_active": True,
        "physics_config": {"fixture": "explicit", "max_valid_cloth_speed_m_s": 5.0},
    }
    metrics = {
        **audit.terminal_geometry(final),
        "max_speed_m_s": 0.0,
        "peak_cloth_speed_m_s": 0.0,
        "physics_valid": True,
        "stable_ticks": 0,
        "success": False,
    }
    final["metrics"] = metrics
    actor_directory = tmp_path / "actors"
    actor_directory.mkdir()
    models = {}
    for arm in ("left", "right"):
        payload = f"test checkpoint bytes for {arm}".encode()
        (actor_directory / f"{arm}.pt").write_bytes(payload)
        models[arm] = {
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            "architecture": ARCHITECTURES[arm],
            "foundation_vla": False,
            "inference_calls": 3,
        }
    records = []
    phases = {arm: "approach_sleeve_above" for arm in robots}
    for tick in range(3):
        obs = {arm: {**value, "tick": tick} for arm, value in observations.items()}
        predictions = {
            arm: {
                "raw": [*robot["ee"], -4.0],
                "grip_probability": float(1 / (1 + np.exp(4))),
                "instruction": phases[arm],
                "coordinator_hold": False,
                "checkpoint_sha256": models[arm]["checkpoint_sha256"],
                "rgb_sha256": "a" * 64,
            }
            for arm, robot in robots.items()
        }
        for arm in robots:
            channel.publish(
                arm, {"features": observation_features(obs[arm]), "source_sha256": "a" * 64}, tick
            )
        channel.advance(tick, state={"tick": tick, "phases": phases})
        beliefs = {arm: channel.receive(arm, tick) for arm in robots}
        decisions = {
            arm: {
                "hold": False,
                "peer_packet_id": belief["packet_id"] if belief else None,
                "peer_observation_tick": belief["observation_tick"] if belief else None,
            }
            for arm, belief in beliefs.items()
        }
        record = {
            "tick": tick + 1,
            "decision_tick": tick,
            "time": (tick + 1) * 0.04,
            "observations": obs,
            "predictions": predictions,
            "decisions": decisions,
            "actions": {
                arm: {"target": robot["ee"], "grip": False} for arm, robot in robots.items()
            },
            "beliefs": beliefs,
            "robots": robots,
            "metrics": metrics,
            "phase": phases,
            "obstacle": {"active": False, "side": None, "position": [2.3, 0, 0.18]},
            "communication": {
                key: channel.counter[key]
                for key in (
                    "published",
                    "delivered",
                    "dropped",
                    "expired",
                    "out_of_order",
                    "packet_bytes",
                )
            },
        }
        records.append(record)
    summary = {
        "run_id": run_id,
        "mode": config.mode,
        "status": "exhausted",
        "success": False,
        "config": asdict(config),
        "ticks": 3,
        "metrics": metrics,
        "physics": final["physics_config"],
        "fixture_active": True,
        "models": models,
        "communication": channel.metrics,
        "phase": phases,
        "error": None,
    }
    write(tmp_path / "config.json", {**asdict(config), "run_id": run_id})
    write(tmp_path / "summary.json", summary)
    write(tmp_path / "final.json", final)
    write(tmp_path / "communication.json.gz", {"config": channel.config, "events": channel.events})
    write_rows(tmp_path, records)
    return tmp_path, codec, actor_directory


def run(evidence):
    path, codec, actors = evidence
    return audit.audit_run(path, codec=codec, actor_directory=actors)


def test_failed_run_is_valid_evidence_but_not_verified_success(evidence):
    result = run(evidence)
    assert result["evidence_valid"] and not result["success_verified"]
    assert result["ticks"] == 3 and result["neural_actions"] == 6


@pytest.mark.parametrize(
    "metric, value, issue",
    [
        ("peak_cloth_speed_m_s", 2.0, "invalid_peak_speed_counter"),
        ("peak_cloth_speed_m_s", 6.0, "physics_valid_flag_disagrees_with_speed"),
        ("max_speed_m_s", -0.1, "invalid_recorded_cloth_speed"),
    ],
)
def test_physical_validity_cannot_hide_unstable_speed_history(evidence, metric, value, issue):
    path, _, _ = evidence
    records = rows(path)
    records[0]["metrics"][metric] = value
    write_rows(path, records)
    result = run(evidence)
    assert not result["evidence_valid"]
    assert any(item["issue"] == issue for item in result["issues"])


def test_fixture_release_requires_a_single_recorded_causal_transition(evidence):
    path, _, _ = evidence
    records = rows(path)
    for record in records[:2]:
        record["events"] = [
            {"event": "vacuum_fixture_released", "tick": record["decision_tick"]}
        ]
    write_rows(path, records)
    result = run(evidence)
    assert not result["evidence_valid"]
    assert "invalid_fixture_release_history" in {item["issue"] for item in result["issues"]}


@pytest.mark.parametrize("evidence", ["embeddings", "hybrid", "none"], indirect=True)
def test_every_representation_is_audited_without_assuming_a_winner(evidence):
    result = run(evidence)
    assert result["evidence_valid"] and not result["success_verified"]


def test_terminal_geometry_distinguishes_a_flat_fold_from_a_collapsed_bundle():
    vertices, faces, _ = shirt_mesh()
    vertices[:, 0] = np.where(vertices[:, 0] < -0.24, -0.48 - vertices[:, 0], vertices[:, 0])
    vertices[:, 0] = np.where(vertices[:, 0] > 0.24, 0.48 - vertices[:, 0], vertices[:, 0])
    vertices[:, 1] = np.where(vertices[:, 1] < -0.02, -0.04 - vertices[:, 1], vertices[:, 1])
    result = audit.terminal_geometry({"vertices": vertices.tolist(), "triangles": faces.tolist()})
    assert result["fold_geometry_ok"] and max(result["hem_alignment_m"].values()) < 0.001
    vertices[:, :2] *= 0.01
    collapsed = audit.terminal_geometry(
        {"vertices": vertices.tolist(), "triangles": faces.tolist()}
    )
    assert not collapsed["fold_geometry_ok"]


@pytest.mark.parametrize(
    "mutation,issue",
    [
        (
            lambda row: row["actions"]["left"]["target"].__setitem__(0, 0.55),
            "applied_target_not_neural_or_hold_left",
        ),
        (
            lambda row: row["predictions"]["right"].__setitem__("checkpoint_sha256", "f" * 64),
            "prediction_checkpoint_right",
        ),
        (
            lambda row: row["beliefs"]["right"].__setitem__("age_ticks", 0),
            "belief_not_causally_delivered_right",
        ),
        (
            lambda row: row["beliefs"]["right"]["features"].__setitem__("ee_z", 0.99),
            "belief_not_causally_delivered_right",
        ),
        (
            lambda row: row["observations"]["left"].__setitem__("peer_ground_truth", {}),
            "observation_includes_unapproved_fields_left",
        ),
        (
            lambda row: row["decisions"]["left"].__setitem__("peer_packet_id", "foreign"),
            "decision_uses_wrong_packet_left",
        ),
        (
            lambda row: row["communication"].__setitem__("delivered", 999),
            "communication_counter_delivered",
        ),
    ],
)
def test_causal_action_and_observation_tampering_rejected(evidence, mutation, issue):
    path, _, _ = evidence
    records = rows(path)
    mutation(records[1])
    write_rows(path, records)
    result = run(evidence)
    assert not result["evidence_valid"]
    assert issue in {entry["issue"] for entry in result["issues"]}


def test_tampered_arrival_history_rejected(evidence):
    path, _, _ = evidence
    transport = audit.read_json(path / "communication.json.gz")
    delivered = next(event for event in transport["events"] if event["kind"] == "delivered")
    delivered["tick"] = 0
    write(path / "communication.json.gz", transport)
    assert not run(evidence)["evidence_valid"]


def test_false_success_and_fixture_retention_rejected(evidence):
    path, _, _ = evidence
    summary = audit.read_json(path / "summary.json")
    summary.update(status="succeeded", success=True, phase={"left": "settle", "right": "settle"})
    write(path / "summary.json", summary)
    result = run(evidence)
    assert not result["success_verified"]
    assert "success_without_physical_completion" in {entry["issue"] for entry in result["issues"]}


def test_unchanged_success_flag_does_not_hide_mutated_mesh(evidence):
    path, _, _ = evidence
    final = audit.read_json(path / "final.json")
    final["vertices"][0][2] = 2
    write(path / "final.json", final)
    result = run(evidence)
    assert not result["evidence_valid"]
    assert "recomputed_terminal_max_height_m" in {entry["issue"] for entry in result["issues"]}


def test_actual_actor_checkpoint_bytes_verified(evidence):
    _, _, actors = evidence
    (actors / "right.pt").write_bytes(b"changed checkpoint")
    assert not run(evidence)["evidence_valid"]


def test_immutable_run_snapshots_are_preferred_over_mutable_source_paths(evidence):
    path, codec, actors = evidence
    snapshot = path / "models"
    snapshot.mkdir()
    for arm in ("left", "right"):
        shutil.copyfile(actors / f"{arm}.pt", snapshot / f"{arm}.pt")
    codec.save(snapshot / "communication.json")
    (path / "codec.json").write_text("source checkpoint no longer exists")
    (actors / "left.pt").write_bytes(b"changed external checkpoint")
    assert audit.audit_run(path)["evidence_valid"]


def test_missing_final_physical_step_cannot_pass_audit(evidence):
    path, _, _ = evidence
    write_rows(path, rows(path)[:-1])
    result = run(evidence)
    assert not result["evidence_valid"]
    assert "terminal_tick_without_trace" in {entry["issue"] for entry in result["issues"]}


def test_optional_durable_sql_is_checked_against_all_recorded_batches(evidence):
    path, codec, actors = evidence
    transport = audit.read_json(path / "communication.json.gz")
    config = audit.read_json(path / "config.json")
    packets = [event["packet"] for event in transport["events"] if event["kind"] == "published"]

    class Client:
        corrupt = False

        def _sql(self, query):
            if "FROM cloth_channel" in query:
                return [{"config_json": json.dumps(transport["config"])}]
            if "FROM cloth_frame" in query:
                return [
                    {
                        "tick": tick,
                        "payload_json": json.dumps(
                            {
                                "packets": [p for p in packets if p["emitted_tick"] == tick],
                                "state": {
                                    "tick": tick,
                                    "phases": {
                                        arm: "changed" if self.corrupt else "approach_sleeve_above"
                                        for arm in ("left", "right")
                                    },
                                },
                            }
                        ),
                    }
                    for tick in range(3)
                ]
            assert "FROM cloth_packet" in query and config["run_id"] in query
            return [{"packet_json": json.dumps(packet)} for packet in packets]

    client = Client()
    result = audit.audit_run(path, codec=codec, actor_directory=actors, client=client)
    assert result["evidence_valid"] and result["durable_frames"] == 3
    client.corrupt = True
    assert not audit.audit_run(path, codec=codec, actor_directory=actors, client=client)[
        "evidence_valid"
    ]
