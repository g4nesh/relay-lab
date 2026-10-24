#!/usr/bin/env python3
"""Audit complete routing traces and optionally their read-only Spacetime commits.

Checks recorded causal provenance, neural action coverage, physical delivery,
and terminal route correctness. It does not attest model execution or reconstruct
unretained source pixels. A correctly recorded rejected run is valid evidence.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

from relay_lab.factory.communication import FIELDS, RECIPIENT, SCHEMA, SENDER, SPACE
from relay_lab.factory.contracts import STAGES
from relay_lab.factory.routing_codec import DEFAULT_CHECKPOINT, RoutingCodec
from relay_lab.factory.scene import ROBOT_SPECS, pallet_slot
from relay_lab.spacetime import SpacetimeClient, SpacetimeConfig, _identifier


def read_rows(path):
    path = Path(path)
    if not path.is_file() and path.with_suffix(path.suffix + ".gz").is_file():
        path = path.with_suffix(path.suffix + ".gz")
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class DurableEvidence:
    def __init__(self, client, run_id):
        self.client, self.run_id = client, _identifier(run_id, "factory run")
        self.rows = {}

    def select(self, table):
        return self.client._sql(f"SELECT * FROM {table} WHERE run_id = '{self.run_id}'")

    def batch(self, revision):
        if revision not in self.rows:
            start = revision // 10 * 10
            self.rows = {
                row["revision"]: row
                for row in self.client._sql(
                    "SELECT revision, prepare_json, result_json FROM factory_batch "
                    f"WHERE run_id = '{self.run_id}' AND revision >= {start} "
                    f"AND revision < {start + 10}"
                )
            }
        return self.rows.get(revision)


def _finite(values, size):
    return (
        isinstance(values, list)
        and len(values) == size
        and all(type(v) in (int, float) and math.isfinite(v) for v in values)
    )


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


def audit_run(directory, *, codec=None, client=None):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    codec = codec or RoutingCodec()
    run_id, status = summary["run_id"], summary["status"]
    channel = summary.get("communication") or {}
    mode = channel.get("mode")
    actor_hash = summary.get("policy", {}).get("checkpoint_sha256")
    codec_hash = codec.model_info["checkpoint_sha256"]
    period, ttl = channel.get("period_ticks"), channel.get("ttl_ticks")
    issues, warnings, counts = [], [], Counter()
    motor_counts = Counter({robot: 0 for robot in ROBOT_SPECS})
    commands, admitted, command_status, starts = {}, {}, {}, {}
    packets, latest, origins, decoded, decisions, delivery_events = {}, {}, {}, {}, [], {}
    place_actions = Counter()
    initial, previous, pending = None, None, {}
    durable = DurableEvidence(client, run_id) if client else None

    def check(value, code, tick=None):
        if not value:
            counts["errors"] += 1
            if len(issues) < 50:
                issues.append({"code": code, **({"tick": tick} if tick is not None else {})})
        return bool(value)

    def route_of(packet, tick):
        try:
            if mode == "embeddings":
                return codec.decode(packet["vector"])["route"]
            value = json.loads(packet["text"])
            if set(value) != {"route"} or value["route"] not in {"A", "B"}:
                raise ValueError("Invalid exact route text")
            return value["route"]
        except (KeyError, TypeError, ValueError, RuntimeError):
            check(False, "undecodable_packet", tick)
            return None

    check(summary.get("task") == "dispatch-routing", "not_routing_task")
    check(mode in ("embeddings", "text", "none"), "invalid_channel_mode")
    check(period == 5 and ttl == 15, "unexpected_channel_timing")
    check(_digest(actor_hash), "invalid_actor_checkpoint")
    check(channel.get("model", {}).get("checkpoint_sha256") == codec_hash, "codec_mismatch")
    check(channel.get("model", {}).get("space_id") == SPACE, "codec_space_mismatch")
    for row in read_rows(directory / "events.jsonl"):
        check(row.get("run_id") == run_id, "cross_run_record")
        kind = row.get("type")
        if kind == "created":
            check(initial is None and previous is None, "duplicate_creation")
            initial = previous = row["state"]
            check(previous.get("tick") == 0, "invalid_initial_tick")
            check(previous.get("task") == "dispatch-routing", "initial_task_mismatch")
            check(
                row.get("policy", {}).get("checkpoint_sha256") == actor_hash,
                "initial_actor_mismatch",
            )
            check(
                row.get("communication", {}).get("mode") == mode
                and row.get("communication", {}).get("model", {}).get("checkpoint_sha256")
                == codec_hash,
                "initial_channel_mismatch",
            )
        elif kind == "failure":
            counts["failure_records"] += 1
        elif kind == "route_decision":
            decision = {k: v for k, v in row.items() if k not in {"type", "run_id"}}
            tick, jid = decision["tick"], decision["job_id"]
            job = (previous or {}).get("jobs", {}).get(jid, {})
            check(previous is not None and tick == previous["tick"], "decision_clock", tick)
            check(
                job.get("stage") == "at_dispatch" and decision.get("robot_id") == RECIPIENT,
                "decision_scope",
                tick,
            )
            packet_id = decision.get("packet_id")
            packet = packets.get(packet_id)
            if packet_id is None:
                check(
                    not latest.get(jid) or latest[jid]["expires_tick"] <= tick,
                    "ignored_available_packet",
                    tick,
                )
                check(decision.get("route") == "A", "undeclared_no_packet_prior", tick)
                for key in (
                    "observation_tick",
                    "emitted_tick",
                    "source_frame_sha256",
                    "transport_age_ticks",
                    "observation_age_ticks",
                    "confidence",
                ):
                    check(decision.get(key) is None, "prior_claims_peer_evidence", tick)
            elif check(packet is not None, "uncommitted_decision_packet", tick):
                check(
                    mode != "none"
                    and packet["job_id"] == jid
                    and packet["recipient"] == RECIPIENT
                    and packet["emitted_tick"] <= tick < packet["expires_tick"]
                    and latest.get(jid, {}).get("packet_id") == packet_id,
                    "ineligible_decision_packet",
                    tick,
                )
                check(decision.get("route") == decoded.get(jid), "decision_decode_mismatch", tick)
                for key in ("observation_tick", "emitted_tick", "source_frame_sha256"):
                    check(decision.get(key) == packet[key], "decision_origin_mismatch", tick)
                check(
                    decision.get("transport_age_ticks") == tick - packet["emitted_tick"]
                    and decision.get("observation_age_ticks") == tick - packet["observation_tick"],
                    "decision_age_mismatch",
                    tick,
                )
            check(
                type(job.get("index")) is int
                and decision.get("route") in {"A", "B"}
                and decision.get("pallet_slot")
                == 2 * job.get("index", -10) + int(decision.get("route") == "B"),
                "decision_slot_mismatch",
                tick,
            )
            decisions.append(decision)
        elif kind == "execution_observed":
            tick = row["tick"]
            check(tick not in pending, "duplicate_physical_tick", tick)
            check(row.get("committed") is False, "physical_record_claims_commit", tick)
            check(row.get("revision") == tick - 1, "physical_revision_mismatch", tick)
            pending[tick] = row
        elif kind == "tick":
            tick, state = row["tick"], row["state"]
            counts["committed_ticks"] += 1
            check(
                previous is not None and tick == previous["tick"] + 1 and state["tick"] == tick,
                "noncontiguous_commit",
                tick,
            )
            observed = pending.pop(tick, None)
            if check(observed is not None, "missing_physical_observation", tick):
                for key in ("state", "outcomes", "events", "motor_actions"):
                    check(observed.get(key) == row.get(key), "physical_commit_" + key, tick)
            check(state.get("task") == "dispatch-routing", "changed_task", tick)
            check(
                state.get("motor_policy", {}).get("model", {}).get("checkpoint_sha256")
                == actor_hash,
                "motor_state_checkpoint",
                tick,
            )
            check(not row.get("embeddings"), "confounding_pose_channel", tick)
            check(set(state["jobs"]) == set((initial or {}).get("jobs", {})), "changed_jobs", tick)
            for message in row.get("messages", []):
                if message.get("kind") == "intent":
                    value = message["payload"]
                    cid = value["command_id"]
                    check(cid not in admitted, "duplicate_command_admission", tick)
                    admitted[cid] = {
                        "robot_id": message["sender"],
                        "job_id": value.get("job_id"),
                        "next_stage": value.get("subgoal"),
                    }
            for robot in state["robots"]:
                command = robot.get("command")
                if command:
                    cid = command["command_id"]
                    check(
                        command.get("robot_id") == robot["id"]
                        and cid in admitted
                        and admitted[cid]["robot_id"] == robot["id"],
                        "command_without_admission",
                        tick,
                    )
                    if cid in commands:
                        check(command == commands[cid], "active_command_changed", tick)
                    commands[cid] = command
            for event in row.get("events", []):
                if event.get("type") == "command_started":
                    cid = event["command_id"]
                    check(cid not in starts, "duplicate_command_start", tick)
                    starts[cid] = event["tick"]
                    command = commands.get(cid, {})
                    if command.get("target") == "pallet":
                        candidates = [
                            d
                            for d in decisions
                            if d["job_id"] == command.get("job_id") and d["tick"] == tick - 1
                        ]
                        if check(len(candidates) == 1, "missing_local_route_authorization", tick):
                            decision = candidates[0]
                            check(
                                command.get("robot_id") == RECIPIENT
                                and command.get("pallet_slot") == decision["pallet_slot"]
                                and command.get("route_packet_id") == decision["packet_id"],
                                "authorized_slot_or_packet_mismatch",
                                tick,
                            )
                elif event.get("type") == "delivery_verified":
                    cid = event["cargo_id"]
                    check(cid not in delivery_events, "duplicate_delivery_event", tick)
                    delivery_events[cid] = event
            outcomes = {o["command_id"]: o for o in row.get("outcomes", [])}
            check(len(outcomes) == len(row.get("outcomes", [])), "duplicate_outcome", tick)
            for cid in admitted:
                if cid not in command_status or command_status[cid] == "running":
                    check(cid in outcomes, "missing_command_outcome", tick)
            for cid, outcome in outcomes.items():
                command = admitted.get(cid, {})
                check(command.get("robot_id") == outcome["robot_id"], "outcome_scope", tick)
                check(
                    command_status.get(cid) not in {"completed", "failed"}, "outcome_replay", tick
                )
                check(
                    outcome["status"] in {"running", "completed", "failed"}, "outcome_status", tick
                )
                job = state["jobs"].get(command.get("job_id"))
                if job:
                    check(
                        outcome["observed"].get("stage") == job["stage"]
                        and outcome["observed"].get("custodian") == job.get("custodian"),
                        "outcome_observation_mismatch",
                        tick,
                    )
                    if outcome["status"] == "completed":
                        check(job["stage"] == command["next_stage"], "completion_stage", tick)
                command_status[cid] = outcome["status"]
            for jid, job in state["jobs"].items():
                old = (previous or {}).get("jobs", {}).get(jid, {})
                if old.get("stage") != job["stage"]:
                    evidence = [
                        o
                        for cid, o in outcomes.items()
                        if admitted.get(cid, {}).get("job_id") == jid and o["status"] == "completed"
                    ]
                    check(
                        old.get("stage") in STAGES[:-1]
                        and STAGES[STAGES.index(old["stage"]) + 1] == job["stage"]
                        and len(evidence) == 1,
                        "unsupported_job_transition",
                        tick,
                    )
            for action in row.get("motor_actions", []):
                rid, cid = action["robot_id"], action["command_id"]
                provenance = action.get("provenance", {})
                check(
                    rid in ROBOT_SPECS
                    and action["tick"] == tick - 1
                    and outcomes.get(cid, {}).get("robot_id") == rid,
                    "motor_scope_or_clock",
                    tick,
                )
                check(
                    provenance.get("model_id") == "relay-lab/factory-vla-v1"
                    and provenance.get("checkpoint_sha256") == actor_hash
                    and _finite(provenance.get("raw_output"), 7)
                    and _finite(action.get("raw_action", {}).get("delta"), 3)
                    and provenance.get("instruction") == action.get("instruction"),
                    "neural_motor_provenance",
                    tick,
                )
                check(
                    not action.get("peer_record_ids") and not provenance.get("peer_fusion"),
                    "confounding_motor_pose_fusion",
                    tick,
                )
                motor_counts[rid] += 1
                if rid == RECIPIENT and action.get("phase") == "approach_place":
                    command = commands.get(cid, {})
                    slot = command.get("pallet_slot")
                    expected = (
                        f"place {command.get('cargo_id')} on pallet-{slot + 1:03d}"
                        if type(slot) is int
                        else None
                    )
                    check(
                        expected is not None and action.get("instruction") == expected,
                        "motor_route_instruction_mismatch",
                        tick,
                    )
                    place_actions[command.get("job_id")] += 1
            seen = set()
            for packet in row.get("semantic_packets", []):
                counts["committed_packets"] += 1
                jid, pid = packet.get("job_id"), packet.get("packet_id")
                payload = "vector" if mode == "embeddings" else "text"
                check(
                    mode != "none"
                    and set(packet) == FIELDS | {payload}
                    and packet.get("schema") == SCHEMA
                    and packet.get("space_id") == SPACE
                    and packet.get("checkpoint_sha256") == codec_hash
                    and packet.get("mode") == mode
                    and packet.get("run_id") == run_id
                    and packet.get("sender") == SENDER
                    and packet.get("recipient") == RECIPIENT
                    and packet.get("topic") == "dispatch_route"
                    and jid in state["jobs"]
                    and jid not in seen
                    and state["jobs"][jid]["stage"] != "completed",
                    "packet_schema_or_scope",
                    tick,
                )
                check(
                    packet.get("emitted_tick") == tick
                    and packet.get("expires_tick") == tick + ttl
                    and type(packet.get("observation_tick")) is int
                    and 0 <= packet["observation_tick"] < tick
                    and pid == f"{SENDER}-{jid}-{tick}"
                    and pid not in packets
                    and _digest(packet.get("source_frame_sha256")),
                    "packet_clock_identity_or_frame",
                    tick,
                )
                if mode == "embeddings":
                    vector = packet.get("vector")
                    check(
                        _finite(vector, 32) and abs(sum(v * v for v in vector) - 1) < 0.002,
                        "invalid_latent_values",
                        tick,
                    )
                origin = {
                    k: v
                    for k, v in packet.items()
                    if k not in {"packet_id", "emitted_tick", "expires_tick"}
                }
                if jid in origins:
                    check(origin == origins[jid], "changed_observation_or_payload", tick)
                    check(tick % period == 0, "retransmission_cadence", tick)
                else:
                    command = commands.get(packet.get("source_command_id"), {})
                    check(
                        packet.get("observation_tick") == tick - 1
                        and starts.get(packet.get("source_command_id")) == tick - 1
                        and (previous or {}).get("jobs", {}).get(jid, {}).get("stage") == "queued"
                        and command.get("robot_id") == SENDER
                        and command.get("job_id") == jid
                        and command.get("capability") == "transfer",
                        "unauthorized_private_observation",
                        tick,
                    )
                    origins[jid], decoded[jid] = origin, route_of(packet, tick)
                seen.add(jid)
                packets[pid] = latest[jid] = packet
            latest = {
                jid: p for jid, p in latest.items() if state["jobs"][jid]["stage"] != "completed"
            }
            if durable:
                batch = durable.batch(tick - 1)
                if check(bool(batch and batch.get("result_json")), "missing_durable_commit", tick):
                    result = json.loads(batch["result_json"])
                    prepare = json.loads(batch["prepare_json"])
                    for key in ("state", "outcomes", "semantic_packets", "embeddings"):
                        check(result.get(key, []) == row.get(key, []), "durable_" + key, tick)
                    check(
                        prepare.get("messages", []) + result.get("messages", [])
                        == row.get("messages", []),
                        "durable_messages",
                        tick,
                    )
                    check(
                        {c["command_id"]: c["robot_id"] for c in prepare["commands"]}
                        == {cid: o["robot_id"] for cid, o in outcomes.items()},
                        "durable_authorized_commands",
                        tick,
                    )
                    for command in prepare["commands"]:
                        recorded = commands.get(command["command_id"], {})
                        if command.get("robot_id") == RECIPIENT:
                            payload = command.get("payload", {})
                            check(
                                all(
                                    payload.get(k) == recorded.get(k)
                                    for k in (
                                        "pallet_slot",
                                        "route_packet_id",
                                        "target",
                                        "cargo_id",
                                    )
                                ),
                                "durable_local_route_authorization",
                                tick,
                            )
                    counts["durable_commits_checked"] += 1
            previous = state

    final = previous or {}
    check(initial is not None and counts["committed_ticks"] > 0, "missing_committed_trace")
    for key in ("tick", "jobs", "cargo", "success", "physical_complete", "completed"):
        check(summary.get(key) == final.get(key), "summary_" + key)
    check(channel.get("published") == counts["committed_packets"], "summary_packet_count")
    check(channel.get("decisions") == decisions, "summary_decisions")
    check(summary.get("metrics", {}).get("commands") == len(admitted), "summary_command_count")
    check(
        summary.get("policy", {}).get("inference_calls") == sum(motor_counts.values()),
        "summary_motor_count",
    )
    observation_jobs = set()
    for observation in channel.get("observation_provenance", []):
        jid, cid = observation["job_id"], observation["source_command_id"]
        command = commands.get(cid, {})
        check(
            jid not in observation_jobs
            and jid in final.get("jobs", {})
            and command.get("robot_id") == SENDER
            and command.get("job_id") == jid
            and starts.get(cid) == observation["observation_tick"]
            and _digest(observation.get("source_frame_sha256")),
            "summary_observation_origin",
        )
        observation_jobs.add(jid)
        if mode != "none":
            check(
                all(origins.get(jid, {}).get(k) == v for k, v in observation.items()),
                "summary_observation_changed",
            )
    check(channel.get("local_observations") == len(observation_jobs), "summary_observation_count")
    completed = final.get("physical_complete") is True
    all_correct = False
    if completed:
        oracle = summary.get("routing_oracle", {})
        check(set(oracle) == set(final.get("jobs", {})), "missing_terminal_oracle")
        correct = []
        for jid, job in final.get("jobs", {}).items():
            evidence, slot = oracle.get(jid, {}), job.get("delivery_slot")
            selected = [d for d in decisions if d["job_id"] == jid]
            check(
                len(selected) == 1 and selected[0]["pallet_slot"] == slot,
                "final_slot_without_decision",
            )
            route = evidence.get("route")
            required = 2 * job["index"] + int(route == "B")
            expected_correct = route in {"A", "B"} and slot == required
            correct.append(expected_correct)
            check(
                route in {"A", "B"}
                and evidence.get("required_slot") == required
                and evidence.get("selected_slot") == slot
                and evidence.get("correct") is expected_correct
                and job.get("routing_correct") is expected_correct,
                "terminal_oracle_inconsistency",
            )
            valid_slot = type(slot) is int and slot in (2 * job["index"], 2 * job["index"] + 1)
            if check(valid_slot, "physical_slot_outside_job"):
                check(
                    job.get("stage") == "completed"
                    and job.get("location") == "pallet"
                    and job.get("delivered") is True
                    and job.get("processed") is True
                    and job.get("owner") is None
                    and job.get("speed", math.inf) < 0.08
                    and _finite(job.get("position"), 3)
                    and math.dist(job["position"], pallet_slot(slot)) < 0.065,
                    "unverified_physical_delivery",
                )
            event = delivery_events.get(job.get("cargo_id"), {})
            check(
                event.get("slot") == slot and event.get("routing_correct") is expected_correct,
                "terminal_delivery_event_mismatch",
            )
            check(place_actions[jid] > 0, "delivery_without_neural_place_actions")
        all_correct = bool(correct) and all(correct)
        check(all(motor_counts.values()), "missing_robot_neural_actions")
        check(all(v == "completed" for v in command_status.values()), "unfinished_terminal_command")
        check(not pending and not counts["failure_records"], "unconfirmed_or_failed_terminal_run")
        check(status == ("succeeded" if all_correct else "rejected"), "incorrect_terminal_status")
    else:
        check(not summary.get("routing_oracle"), "early_oracle_disclosure")
        check(
            status in {"failed", "cancelled", "exhausted", "running", "ready"},
            "unsupported_completion_status",
        )
        warnings.append("Incomplete run retained; terminal route correctness is not verified.")
    check(final.get("success") is (completed and all_correct), "success_disagrees_with_oracle")
    if durable:
        rows = durable.select("factory_run")
        if check(len(rows) == 1, "durable_run_missing_or_duplicate"):
            state = json.loads(rows[0]["state_json"])
            state.pop("abort_reason", None)
            check(state == final, "durable_final_state")
            check(
                rows[0]["tick"] == final.get("tick")
                and rows[0]["revision"] == counts["committed_ticks"],
                "durable_final_clock",
            )
            check(
                rows[0]["status"] == ("failed" if status == "cancelled" else status),
                "durable_final_status",
            )
        rows = durable.select("factory_semantic_channel")
        config = json.loads(rows[0]["config_json"]) if len(rows) == 1 else {}
        check(
            len(rows) == 1
            and config.get("mode") == mode
            and config.get("checkpoint_sha256") == codec_hash
            and config.get("space_id") == SPACE
            and config.get("period_ticks") == period
            and config.get("ttl_ticks") == ttl,
            "durable_channel_config",
        )
        retained = {
            p["job_id"]: p
            for p in (
                json.loads(r["packet_json"]) for r in durable.select("factory_semantic_packet")
            )
        }
        check(retained == latest, "durable_retained_packets")
    valid = not counts["errors"]
    return {
        "schema": "relay.factory.communication-audit.v1",
        "run_id": run_id,
        "mode": mode,
        "reported_status": status,
        "evidence_valid": valid,
        "success_verified": valid and completed and all_correct,
        "physical_completion_verified": valid and completed,
        "correct_routes": sum(
            j.get("routing_correct") is True for j in final.get("jobs", {}).values()
        ),
        "committed_ticks": counts["committed_ticks"],
        "committed_packets": counts["committed_packets"],
        "route_decisions": len(decisions),
        "commands": dict(Counter(command_status.values())),
        "neural_actions_by_robot": dict(motor_counts),
        "actor_checkpoint_sha256": actor_hash,
        "codec_checkpoint_sha256": codec_hash,
        "spacetime_checked": durable is not None,
        "durable_commits_checked": counts["durable_commits_checked"],
        "error_count": counts["errors"],
        "issues": issues,
        "warnings": warnings,
        "scope": "Full recorded causal/physical evidence consistency, not independent execution attestation; original RGB bytes are not retained in this trace.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--spacetime", action="store_true", help="Read-only durable SQL verification"
    )
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    try:
        codec = RoutingCodec(args.checkpoint)
        client = SpacetimeClient(SpacetimeConfig.from_env()) if args.spacetime else None
        result = audit_run(args.run_directory, codec=codec, client=client)
    except Exception as error:  # noqa: BLE001 - do not print bearer tokens or full log/SQL payloads
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
