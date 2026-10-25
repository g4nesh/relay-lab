#!/usr/bin/env python3
"""Check recorded cloth causality, learned-action provenance and terminal geometry.

This is a consistency audit, not independent execution attestation. RGB hashes
are linked to publications but original pixels and complete simulator states are
not retained per tick. Neural predictions cannot be independently rerun here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from relay_lab.cloth.communication import (
    Channel,
    LearnedFeatureCodec,
    NetworkConfig,
    observation_features,
)
from relay_lab.cloth.policies import ARCHITECTURES, PHASES
from relay_lab.cloth.scene import shirt_mesh

AGENTS = ("left", "right")
HASH = re.compile(r"[a-f0-9]{64}\Z")
MAX_CLOTH_SPEED = 5.0


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def finite_vector(value, length):
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(item) in (int, float) and math.isfinite(item) for item in value)
    )


def near(a, b, tolerance=1e-6):
    try:
        left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return left.shape == right.shape and bool(np.allclose(left, right, rtol=0, atol=tolerance))
    except (ValueError, TypeError):
        return False


def terminal_geometry(final):
    """Recompute footprint and landmark criteria from the recorded physical mesh."""
    vertices = np.asarray(final["vertices"], dtype=float)
    rest, faces, landmarks = shirt_mesh()
    if vertices.shape != rest.shape or not np.isfinite(vertices).all():
        raise ValueError("Invalid final garment mesh")
    if not np.array_equal(np.asarray(final["triangles"]), faces):
        raise ValueError("Final garment topology differs from this benchmark")
    hem = {
        arm: float(
            np.linalg.norm(
                vertices[landmarks[arm]["hem"], :2] - vertices[landmarks[arm]["shoulder"], :2]
            )
        )
        for arm in AGENTS
    }
    sleeve = {
        arm: float(max(0, abs(vertices[landmarks[arm]["sleeve"], 0]) - 0.24)) for arm in AGENTS
    }
    image = Image.new("1", (512, 512))
    pixels = (vertices[:, :2] + 1) * 256
    drawer = ImageDraw.Draw(image)
    for face in faces:
        drawer.polygon([tuple(point) for point in pixels[face]], fill=1)
    occupied = np.asarray(image, dtype=bool)
    shoulders = np.asarray([vertices[landmarks[arm]["shoulder"]] for arm in AGENTS])
    center, shoulder_y = shoulders[:, 0].mean(), shoulders[:, 1].mean()
    low = np.clip(
        np.floor((np.array([center - 0.26, shoulder_y - 0.32]) + 1) * 256).astype(int), 0, 511
    )
    high = np.clip(
        np.ceil((np.array([center + 0.26, shoulder_y + 0.04]) + 1) * 256).astype(int), 0, 512
    )
    inside = int(occupied[low[1] : high[1], low[0] : high[0]].sum())
    footprint = {
        "projected_area_m2": float(occupied.sum()) / 256**2,
        "target_coverage": inside / 256**2 / (0.52 * 0.36),
        "inside_target_fraction": inside / max(1, int(occupied.sum())),
        "span_x_m": float(np.ptp(vertices[:, 0])),
        "span_y_m": float(np.ptp(vertices[:, 1])),
    }
    geometry_ok = (
        footprint["projected_area_m2"] >= 0.075
        and footprint["target_coverage"] >= 0.40
        and footprint["inside_target_fraction"] >= 0.75
        and 0.30 <= footprint["span_x_m"] <= 0.65
        and 0.20 <= footprint["span_y_m"] <= 0.45
    )
    return {
        "hem_alignment_m": hem,
        "sleeve_overhang_m": sleeve,
        "footprint": footprint,
        "fold_geometry_ok": bool(geometry_ok),
        "max_height_m": float(vertices[:, 2].max()),
        "on_table": bool(
            np.all(
                (vertices[:, 2] > -0.02)
                & (abs(vertices[:, 0]) < 0.96)
                & (abs(vertices[:, 1]) < 0.64)
            )
        ),
        "finite": True,
    }


def audit_run(directory, *, codec=None, actor_directory="auto", client=None):
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    config = read_json(directory / "config.json")
    final = read_json(directory / "final.json")
    transport = read_json(directory / "communication.json.gz")
    snapshot_codec = directory / "models" / "communication.json"
    codec = codec or LearnedFeatureCodec.load(
        snapshot_codec if snapshot_codec.is_file() else config["codec"]
    )
    if actor_directory == "auto":
        actor_directory = (
            directory / "models"
            if (directory / "models").is_dir()
            else Path("checkpoints/cloth-policies")
        )
    issues, warnings = [], []
    counts = {"ticks": 0, "neural_actions": 0, "packets": 0, "durable_frames": 0}
    physical_landmarks = shirt_mesh()[2]

    def check(condition, issue, tick=None):
        if not condition:
            issues.append({"issue": issue, **({"tick": tick} if tick is not None else {})})
        return bool(condition)

    run_id, mode = summary["run_id"], summary["mode"]
    check(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is not None, "invalid_run_id")
    check(config["run_id"] == run_id and config["mode"] == mode, "config_run_mode_mismatch")
    check(
        summary["config"] == {k: v for k, v in config.items() if k != "run_id"},
        "summary_config_mismatch",
    )
    check(mode == transport["config"]["mode"], "channel_mode_mismatch")
    network = NetworkConfig(**transport["config"]["network"])
    scenario = config["scenario"]
    expected_network = NetworkConfig(
        seed=config["seed"],
        min_delay_ticks=1 if scenario in ("clear", "obstacles") else 2,
        max_delay_ticks=1
        if scenario in ("clear", "obstacles")
        else 32
        if scenario == "blackout"
        else 18,
        drop_probability=0
        if scenario in ("clear", "obstacles")
        else 0.3
        if scenario == "blackout"
        else 0.15,
        ttl_ticks=24,
    )
    check(network == expected_network, "network_differs_from_declared_scenario")
    channel = Channel(run_id, mode=mode, codec=codec, network=network)
    check(channel.config == transport["config"], "channel_codec_configuration_mismatch")
    check(
        summary["communication"]["model"]["checkpoint_sha256"] == codec.sha256,
        "summary_codec_mismatch",
    )
    models = summary["models"]
    check(set(models) == set(AGENTS), "missing_robot_model")
    hashes = []
    for arm in AGENTS:
        model = models.get(arm, {})
        digest = model.get("checkpoint_sha256", "")
        check(bool(HASH.fullmatch(digest)), "invalid_actor_hash_" + arm)
        check(model.get("architecture") == ARCHITECTURES[arm], "actor_architecture_" + arm)
        check(model.get("foundation_vla") is False, "mislabelled_foundation_model_" + arm)
        hashes.append(digest)
        if actor_directory is not None:
            checkpoint = Path(actor_directory) / f"{arm}.pt"
            check(
                checkpoint.is_file()
                and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest,
                "actor_checkpoint_bytes_" + arm,
            )
    check(len(set(hashes)) == 2, "actors_share_checkpoint_identity")
    if actor_directory is None:
        warnings.append(
            "Actor identities are checked across evidence; checkpoint bytes were not provided."
        )

    durable_frames = {}
    if client is not None:
        rows = client._sql(f"SELECT * FROM cloth_channel WHERE run_id = '{run_id}'")
        check(
            len(rows) == 1 and json.loads(rows[0]["config_json"]) == channel.config,
            "durable_channel_configuration",
        )
        rows = client._sql(f"SELECT * FROM cloth_frame WHERE run_id = '{run_id}'")
        durable_frames = {row["tick"]: json.loads(row["payload_json"]) for row in rows}
        check(len(durable_frames) == len(rows), "duplicate_durable_frames")

    previous, prior_phase = None, {arm: None for arm in AGENTS}
    prior_stable, prior_peak_speed = 0, 0.0
    fixture_active = config.get("fixture")
    check(type(fixture_active) is bool, "invalid_initial_fixture_state")
    with gzip.open(directory / "events.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            tick, decision_tick = row["tick"], row["decision_tick"]
            check(
                type(tick) is int and tick == counts["ticks"] + 1 and decision_tick == tick - 1,
                "noncausal_physics_clock",
                tick,
            )
            check(near(row["time"], tick * 0.04), "simulation_time_mismatch", tick)
            counts["ticks"] += 1
            for event in row.get("events", []):
                if event.get("event") == "vacuum_fixture_released":
                    check(
                        fixture_active is True and event.get("tick") == decision_tick,
                        "invalid_fixture_release_history",
                        tick,
                    )
                    fixture_active = False
            for arm in AGENTS:
                obs, prediction, action = (
                    row["observations"][arm],
                    row["predictions"][arm],
                    row["actions"][arm],
                )
                check(
                    set(obs)
                    == {
                        "tick",
                        "agent",
                        "ee_position",
                        "grasped",
                        "grasp_target",
                        "landmarks",
                        "obstacle",
                    },
                    "observation_includes_unapproved_fields_" + arm,
                    tick,
                )
                check(
                    obs["agent"] == arm and obs["tick"] == decision_tick,
                    "observation_scope_clock_" + arm,
                    tick,
                )
                check(
                    set(obs["landmarks"]) <= {"hem", "sleeve", "shoulder"},
                    "private_landmark_schema_" + arm,
                    tick,
                )
                for point in obs["landmarks"].values():
                    check(
                        finite_vector(point, 3)
                        and (point[0] <= 0.05 if arm == "left" else point[0] >= -0.05),
                        "outside_local_visibility_" + arm,
                        tick,
                    )
                if previous:
                    check(
                        near(obs["ee_position"], previous["robots"][arm]["ee"]),
                        "own_position_disagrees_previous_step_" + arm,
                        tick,
                    )
                    check(
                        obs["grasped"] == (previous["robots"][arm]["holding"] is not None),
                        "own_grasp_disagrees_previous_step_" + arm,
                        tick,
                    )
                    held = previous["robots"][arm]["holding"]
                    own_landmarks = physical_landmarks[arm]
                    expected_target = (
                        "none"
                        if held is None
                        else next(
                            (name for name in ("hem", "sleeve") if own_landmarks[name] == held),
                            "unknown",
                        )
                    )
                    check(
                        obs.get("grasp_target") == expected_target,
                        "own_attachment_identity_" + arm,
                        tick,
                    )
                    obstacle = previous["obstacle"]
                    expected_obstacle = bool(obstacle["active"] and obstacle["side"] == arm)
                    check(
                        bool(obs["obstacle"]) == expected_obstacle,
                        "obstacle_visibility_leak_" + arm,
                        tick,
                    )
                    if expected_obstacle and obs["obstacle"]:
                        check(
                            near(obs["obstacle"]["position"], obstacle["position"]),
                            "obstacle_measurement_mismatch_" + arm,
                            tick,
                        )
                check(
                    prediction["checkpoint_sha256"] == models[arm]["checkpoint_sha256"],
                    "prediction_checkpoint_" + arm,
                    tick,
                )
                check(
                    bool(HASH.fullmatch(prediction.get("rgb_sha256", ""))),
                    "prediction_image_provenance_" + arm,
                    tick,
                )
                if decision_tick == 0 and "initial_rgb_sha256" in summary:
                    check(
                        prediction["rgb_sha256"] == summary["initial_rgb_sha256"].get(arm),
                        "initial_image_provenance_" + arm,
                        tick,
                    )
                raw = prediction["raw"]
                if check(finite_vector(raw, 4), "nonfinite_neural_output_" + arm, tick):
                    probability = float(1 / (1 + np.exp(-np.clip(raw[3], -40, 40))))
                    check(
                        near(prediction["grip_probability"], probability),
                        "neural_grip_decode_" + arm,
                        tick,
                    )
                    hold = prediction["coordinator_hold"]
                    check(
                        type(hold) is bool and row["decisions"][arm]["hold"] is hold,
                        "interlock_record_mismatch_" + arm,
                        tick,
                    )
                    expected_target = obs["ee_position"] if hold else raw[:3]
                    check(
                        near(action["target"], expected_target),
                        "applied_target_not_neural_or_hold_" + arm,
                        tick,
                    )
                    check(
                        action["grip"] is (bool(obs["grasped"]) if hold else probability >= 0.5),
                        "applied_grip_not_neural_or_hold_" + arm,
                        tick,
                    )
                phase = prediction["instruction"]
                valid_phase = check(phase in PHASES, "unknown_neural_instruction_" + arm, tick)
                if valid_phase and prior_phase[arm] is not None:
                    check(
                        PHASES.index(phase) - PHASES.index(prior_phase[arm]) in (0, 1),
                        "invalid_phase_progression_" + arm,
                        tick,
                    )
                prior_phase[arm] = phase if valid_phase else prior_phase[arm]
                counts["neural_actions"] += 1
                if decision_tick % config["publish_every"] == 0:
                    channel.publish(
                        arm,
                        {
                            "features": observation_features(obs),
                            "source_sha256": prediction["rgb_sha256"],
                        },
                        decision_tick,
                    )
            state = {
                "tick": decision_tick,
                "phases": {arm: row["predictions"][arm]["instruction"] for arm in AGENTS},
            }
            periodic = decision_tick % config["publish_every"] == 0
            published = list(channel.pending)
            channel.advance(decision_tick, state=state if periodic else None)
            if client is not None and periodic:
                expected = {"packets": published, "state": state}
                check(durable_frames.get(decision_tick) == expected, "durable_batch_mismatch", tick)
                counts["durable_frames"] += 1
            for arm in AGENTS:
                belief = channel.receive(arm, decision_tick)
                check(belief == row["beliefs"][arm], "belief_not_causally_delivered_" + arm, tick)
                decision = row["decisions"][arm]
                check(
                    decision.get("peer_packet_id") == (belief["packet_id"] if belief else None),
                    "decision_uses_wrong_packet_" + arm,
                    tick,
                )
                check(
                    decision.get("peer_observation_tick")
                    == (belief["observation_tick"] if belief else None),
                    "decision_observation_age_" + arm,
                    tick,
                )
            for key, value in row["communication"].items():
                check(channel.counter.get(key) == value, "communication_counter_" + key, tick)
            metrics = row["metrics"]
            speed, peak = metrics.get("max_speed_m_s"), metrics.get("peak_cloth_speed_m_s")
            finite_speeds = all(
                type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (speed, peak)
            )
            check(finite_speeds, "invalid_recorded_cloth_speed", tick)
            if finite_speeds:
                check(peak >= max(speed, prior_peak_speed), "invalid_peak_speed_counter", tick)
                check(
                    metrics.get("physics_valid")
                    is bool(metrics.get("finite") and peak <= MAX_CLOTH_SPEED),
                    "physics_valid_flag_disagrees_with_speed",
                    tick,
                )
                prior_peak_speed = peak
            stable = metrics.get("stable_ticks", 0)
            check(
                type(stable) is int and 0 <= stable <= prior_stable + 1,
                "invalid_stability_counter",
                tick,
            )
            if stable:
                check(
                    metrics["max_speed_m_s"] < 0.12
                    and fixture_active is False
                    and all(robot["holding"] is None for robot in row["robots"].values()),
                    "unsettled_stability_counter",
                    tick,
                )
            prior_stable, previous = stable, row

    counts["packets"] = channel.counter["published"]
    check(channel.events == transport["events"], "transport_event_history_mismatch")
    for key, value in channel.counter.items():
        check(summary["communication"].get(key) == value, "summary_counter_" + key)
    check(counts["ticks"] > 0, "missing_physical_trace")
    check(summary["ticks"] == final["tick"] == counts["ticks"], "terminal_tick_without_trace")
    check(summary["metrics"] == final["metrics"], "summary_final_metrics_mismatch")
    check(summary.get("fixture_active") is final.get("fixture_active"), "summary_fixture_mismatch")
    check(fixture_active is final.get("fixture_active"), "unrecorded_fixture_state_change")
    check(
        summary.get("physics") == final.get("physics_config"),
        "summary_physics_configuration_mismatch",
    )
    check(
        final.get("physics_config", {}).get("max_valid_cloth_speed_m_s") == MAX_CLOTH_SPEED,
        "unexpected_physics_speed_threshold",
    )
    if previous:
        check(previous["robots"] == final["robots"], "final_robots_without_trace")
        check(previous["metrics"] == final["metrics"], "final_metrics_without_trace")
    for arm in AGENTS:
        check(models[arm].get("inference_calls") == counts["ticks"], "actor_inference_count_" + arm)
    geometry = terminal_geometry(final)
    for key, value in geometry.items():
        recorded = final["metrics"].get(key)
        if isinstance(value, dict):
            check(
                isinstance(recorded, dict)
                and set(recorded) == set(value)
                and all(near(recorded[k], v) for k, v in value.items()),
                "recomputed_terminal_" + key,
            )
        else:
            check(near(recorded, value), "recomputed_terminal_" + key)
    metrics = final["metrics"]
    physical_success = bool(
        geometry["finite"]
        and metrics.get("physics_valid") is True
        and metrics.get("peak_cloth_speed_m_s", math.inf) <= MAX_CLOTH_SPEED
        and geometry["on_table"]
        and geometry["fold_geometry_ok"]
        and final.get("fixture_active") is False
        and max(geometry["hem_alignment_m"].values()) < 0.095
        and max(geometry["sleeve_overhang_m"].values()) < 0.06
        and geometry["max_height_m"] < 0.12
        and metrics["max_speed_m_s"] < 0.12
        and metrics["stable_ticks"] >= 15
        and all(robot["holding"] is None for robot in final["robots"].values())
    )
    check(metrics["success"] is physical_success, "physical_success_disagrees_with_criteria")
    check(
        summary["success"] is (summary["status"] == "succeeded"), "reported_success_status_mismatch"
    )
    if summary["status"] == "rejected":
        check(not physical_success, "rejected_status_but_physical_criteria_met")
    if summary["success"]:
        check(physical_success, "success_without_physical_completion")
        check(
            all(phase == "settle" for phase in summary["phase"].values()),
            "success_without_final_phase",
        )
        check(not summary.get("error"), "successful_run_contains_error")
    if client is not None:
        check(len(durable_frames) == counts["durable_frames"], "durable_unrecorded_frames")
        rows = client._sql(f"SELECT * FROM cloth_packet WHERE run_id = '{run_id}'")
        actual = {
            row["packet_id"]: row for row in [json.loads(record["packet_json"]) for record in rows]
        }
        expected = {
            event["packet"]["packet_id"]: event["packet"]
            for event in channel.events
            if event["kind"] == "published"
        }
        check(len(rows) == len(actual) and actual == expected, "durable_packet_table_mismatch")
    return {
        "schema": "relay.cloth.audit.v1",
        "run_id": run_id,
        "mode": mode,
        "reported_status": summary["status"],
        "evidence_valid": not issues,
        "success_verified": not issues and summary["success"] and physical_success,
        "physical_success": physical_success,
        **counts,
        "issues": issues,
        "warnings": warnings,
        "codec_sha256": codec.sha256,
        "actor_sha256": dict(zip(AGENTS, hashes)),
        "spacetime_checked": client is not None,
        "scope": "Recorded causal and geometric consistency; not independent execution attestation. RGB bytes and full per-tick physics states are unavailable; neural outputs and speed are recorded measurements.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument(
        "--actor-directory",
        type=Path,
        help="Override run/models; default uses immutable run snapshots",
    )
    parser.add_argument("--spacetime", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    try:
        client = None
        if args.spacetime:
            from relay_lab.spacetime import SpacetimeClient, SpacetimeConfig

            client = SpacetimeClient(SpacetimeConfig.from_env())
        result = audit_run(
            args.directory,
            codec=LearnedFeatureCodec.load(args.codec) if args.codec else None,
            actor_directory=args.actor_directory or "auto",
            client=client,
        )
    except Exception as error:  # noqa: BLE001 - do not expose credentials or source payloads
        result = {
            "evidence_valid": False,
            "success_verified": False,
            "audit_error": type(error).__name__,
        }
    print(json.dumps(result, separators=(",", ":")))
    return (
        0
        if result["evidence_valid"] and (not args.require_success or result["success_verified"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
