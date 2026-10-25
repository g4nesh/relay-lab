#!/usr/bin/env python3
"""Audit saved factory VLA evidence; optionally compare durable rows read-only.

This checks recorded provenance and consistency, not hardware or model-execution
attestation. Failed runs remain failed. No inference, reducers, or SQL writes run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

from relay_lab.factory.contracts import STAGES
from relay_lab.factory.embeddings import SCHEMA, unit_vector
from relay_lab.factory.scene import ROBOT_SPECS
from relay_lab.spacetime import SpacetimeClient, SpacetimeConfig, _identifier


def _read_rows(path):
    if not path.is_file() and path.with_suffix(path.suffix + ".gz").is_file():
        path = path.with_suffix(path.suffix + ".gz")
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class DurableEvidence:
    """Small indexed batches avoid loading an entire run into one HTTP response."""

    def __init__(self, client, run_id):
        self.client, self.run_id = client, _identifier(run_id, "factory run")
        self.rows = {}

    def select(self, table, fields="*"):
        return self.client._sql(f"SELECT {fields} FROM {table} WHERE run_id = '{self.run_id}'")

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


def audit_run(directory, *, client=None):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    run_id, status = summary["run_id"], summary["status"]
    expected_hash = summary.get("policy", {}).get("checkpoint_sha256")
    issues, warnings, counts = [], [], Counter()
    actions_by_robot = Counter({robot: 0 for robot in ROBOT_SPECS})
    commands, command_status, vectors, latest = {}, {}, {}, {}
    previous, initial, pending_observed = None, None, {}
    durable = DurableEvidence(client, run_id) if client else None

    def check(condition, code, tick=None):
        if not condition:
            counts["errors"] += 1
            if len(issues) < 30:
                issues.append({"code": code, **({"tick": tick} if tick is not None else {})})
        return bool(condition)

    check(summary.get("policy", {}).get("policy") == "factory-vla", "not_factory_vla")
    check(
        isinstance(expected_hash, str) and bool(re.fullmatch("[a-f0-9]{64}", expected_hash)),
        "invalid_checkpoint_hash",
    )
    for row in _read_rows(directory / "events.jsonl"):
        check(row.get("run_id") == run_id, "cross_run_log_record")
        kind = row.get("type")
        if kind == "failure":
            counts["failure_records"] += 1
        if kind == "created":
            check(initial is None and previous is None, "duplicate_or_late_creation")
            initial = previous = row["state"]
            check(previous.get("tick") == 0, "invalid_initial_tick")
            check(
                row.get("policy", {}).get("checkpoint_sha256") == expected_hash,
                "created_checkpoint_mismatch",
            )
            check(
                {robot["id"] for robot in previous["robots"]} == set(ROBOT_SPECS),
                "factory_roster_mismatch",
            )
        elif kind == "execution_observed":
            tick = row["tick"]
            check(tick not in pending_observed, "duplicate_physical_tick", tick)
            check(row.get("revision") == tick - 1, "physical_revision_mismatch", tick)
            check(row.get("committed") is False, "physical_record_claims_commit", tick)
            pending_observed[tick] = row
        elif kind == "tick":
            state, tick = row["state"], row["tick"]
            counts["committed_ticks"] += 1
            check(previous is not None, "missing_creation", tick)
            check(
                state.get("tick") == tick and tick == (previous or {}).get("tick", -1) + 1,
                "noncontiguous_committed_tick",
                tick,
            )
            observed = pending_observed.pop(tick, None)
            check(observed is not None, "missing_physical_observation", tick)
            if observed:
                for field in ("state", "outcomes", "events", "motor_actions"):
                    check(
                        observed.get(field, []) == row.get(field, []),
                        "physical_commit_" + field,
                        tick,
                    )
            check(
                state.get("motor_policy", {}).get("model", {}).get("checkpoint_sha256")
                == expected_hash,
                "state_checkpoint_mismatch",
                tick,
            )
            for message in row.get("messages", []):
                if message["kind"] == "intent":
                    payload = message["payload"]
                    cid = payload["command_id"]
                    check(cid not in commands, "duplicate_command_admission", tick)
                    commands[cid] = {
                        "robot_id": message["sender"],
                        "job_id": payload.get("job_id"),
                        "next_stage": payload.get("subgoal"),
                    }
            outcomes = {outcome["command_id"]: outcome for outcome in row["outcomes"]}
            check(len(outcomes) == len(row["outcomes"]), "duplicate_command_outcome", tick)
            check(
                all(cid in outcomes for cid, value in command_status.items() if value == "running"),
                "missing_running_command_outcome",
                tick,
            )
            check(
                all(cid in outcomes for cid in commands if cid not in command_status),
                "admitted_command_without_outcome",
                tick,
            )
            for cid, outcome in outcomes.items():
                command = commands.get(cid, {})
                check(bool(command), "unknown_command_outcome", tick)
                check(
                    outcome["robot_id"] == command.get("robot_id"), "outcome_robot_mismatch", tick
                )
                check(
                    command_status.get(cid) not in ("completed", "failed"),
                    "terminal_command_replayed",
                    tick,
                )
                check(
                    outcome["status"] in ("running", "completed", "failed"),
                    "invalid_outcome_status",
                    tick,
                )
                job = state["jobs"].get(command.get("job_id"))
                if job:
                    check(
                        outcome["observed"].get("stage") == job["stage"]
                        and outcome["observed"].get("custodian") == job.get("custodian"),
                        "outcome_job_evidence_mismatch",
                        tick,
                    )
                    if outcome["status"] == "completed":
                        check(
                            job["stage"] == command["next_stage"],
                            "completed_command_stage_mismatch",
                            tick,
                        )
                command_status[cid] = outcome["status"]
            check(
                set(state["jobs"]) == set((previous or {}).get("jobs", {})),
                "job_inventory_changed",
                tick,
            )
            for jid, job in state["jobs"].items():
                old = (previous or {}).get("jobs", {}).get(jid, {})
                if old.get("stage") != job["stage"]:
                    associated = [
                        (cid, value)
                        for cid, value in outcomes.items()
                        if commands.get(cid, {}).get("job_id") == jid
                    ]
                    check(
                        old.get("stage") in STAGES[:-1]
                        and STAGES[STAGES.index(old["stage"]) + 1] == job["stage"]
                        and len(associated) == 1
                        and associated[0][1]["status"] == "completed"
                        and commands[associated[0][0]]["next_stage"] == job["stage"],
                        "unsupported_job_transition",
                        tick,
                    )
            for action in row.get("motor_actions", []):
                robot, provenance = action["robot_id"], action.get("provenance", {})
                check(robot in ROBOT_SPECS, "unknown_motor_robot", tick)
                check(action["tick"] == tick - 1, "motor_tick_mismatch", tick)
                check(
                    outcomes.get(action["command_id"], {}).get("robot_id") == robot,
                    "motor_command_scope_mismatch",
                    tick,
                )
                raw = provenance.get("raw_output", [])
                delta = action.get("raw_action", {}).get("delta", [])
                check(
                    provenance.get("checkpoint_sha256") == expected_hash
                    and provenance.get("model_id") == "relay-lab/factory-vla-v1"
                    and len(raw) == 7
                    and all(type(value) in (int, float) and math.isfinite(value) for value in raw),
                    "invalid_neural_action_provenance",
                    tick,
                )
                check(
                    len(delta) == 3
                    and all(
                        type(value) in (int, float) and math.isfinite(value) for value in delta
                    ),
                    "invalid_recorded_motor_action",
                    tick,
                )
                actions_by_robot[robot] += 1
                refs = action.get("peer_record_ids", [])
                fusion = provenance.get("peer_fusion") is True
                if fusion:
                    counts["peer_fusion_actions"] += 1
                    check(bool(refs), "fusion_without_vector_reference", tick)
                matched = False
                for ident in refs:
                    vector = vectors.get(ident)
                    eligible = vector is not None and (
                        vector["tick"] <= action["tick"] < vector["expires_tick"]
                        and vector["run_id"] == run_id
                        and vector["checkpoint_sha256"] == expected_hash
                        and vector["robot_id"] != robot
                        and latest.get(vector["robot_id"]) == ident
                    )
                    check(eligible, "ineligible_or_nonlatest_peer_vector", tick)
                    if eligible and vector["robot_id"] == provenance.get("target"):
                        matched = True
                if fusion:
                    check(matched, "fusion_target_peer_mismatch", tick)
            # Only now publish this tick's vectors to the audit's prior-commit view.
            seen_robots = set()
            for vector in row.get("embeddings", []):
                robot, ident = vector["robot_id"], vector["embedding_id"]
                check(
                    robot in ROBOT_SPECS
                    and robot not in seen_robots
                    and vector.get("schema") == SCHEMA
                    and vector.get("space_id") == "factory-vla-v1"
                    and vector.get("dimension") == 32
                    and vector.get("modality") == "vla-hidden"
                    and vector["run_id"] == run_id
                    and vector["checkpoint_sha256"] == expected_hash
                    and vector["tick"] == tick
                    and tick < vector["expires_tick"] <= tick + 64
                    and ident == f"{robot}-{tick}"
                    and ident not in vectors,
                    "invalid_committed_vector_provenance",
                    tick,
                )
                try:
                    unit_vector(vector.get("vector"))
                except (ValueError, RuntimeError):
                    check(False, "invalid_committed_vector_values", tick)
                seen_robots.add(robot)
                observed_robot = next((r for r in state["robots"] if r["id"] == robot), {})
                cid = vector.get("command_id")
                check(
                    (
                        cid is None
                        and vector.get("job_id") is None
                        and not observed_robot.get("active_command")
                    )
                    or (
                        cid in outcomes
                        and commands.get(cid, {}).get("robot_id") == robot
                        and commands[cid].get("job_id") == vector.get("job_id")
                        and observed_robot.get("active_command") in (None, cid)
                    ),
                    "committed_vector_command_scope_mismatch",
                    tick,
                )
                vectors[ident], latest[robot] = vector, ident
                counts["committed_vectors"] += 1
            if durable:
                batch = durable.batch(tick - 1)
                check(bool(batch and batch.get("result_json")), "missing_durable_commit", tick)
                if batch and batch.get("result_json"):
                    prepare, result = (
                        json.loads(batch["prepare_json"]),
                        json.loads(batch["result_json"]),
                    )
                    for field in ("state", "outcomes", "embeddings"):
                        check(
                            result.get(field, []) == row.get(field, []),
                            "durable_" + field + "_mismatch",
                            tick,
                        )
                    check(
                        prepare.get("messages", []) + result.get("messages", [])
                        == row.get("messages", []),
                        "durable_messages_mismatch",
                        tick,
                    )
                    check(
                        {c["command_id"]: c["robot_id"] for c in prepare["commands"]}
                        == {cid: o["robot_id"] for cid, o in outcomes.items()},
                        "durable_authorized_commands_mismatch",
                        tick,
                    )
                    counts["durable_commits_checked"] += 1
            previous = state
            vectors = {
                ident: value for ident, value in vectors.items() if value["expires_tick"] > tick
            }
            for robot in ROBOT_SPECS:
                retained = sorted(
                    (v for v in vectors.values() if v["robot_id"] == robot),
                    key=lambda value: value["tick"],
                    reverse=True,
                )
                for old in retained[16:]:
                    vectors.pop(old["embedding_id"])
    check(initial is not None, "missing_creation")
    final = previous or {}
    all_completed = bool(final.get("jobs")) and all(
        j["stage"] == "completed" for j in final["jobs"].values()
    )
    check(final.get("success") is all_completed, "final_success_stage_disagreement")
    check(summary.get("tick") == final.get("tick"), "summary_tick_mismatch")
    for field in ("jobs", "success", "completed"):
        check(summary.get(field) == final.get(field), "summary_" + field + "_mismatch")
    if status == "succeeded":
        check(all_completed and bool(counts["committed_ticks"]), "unsupported_summary_success")
        check(all(actions_by_robot.values()), "missing_robot_neural_actions")
        check(
            all(value == "completed" for value in command_status.values()),
            "unfinished_successful_command",
        )
        check(not pending_observed, "unconfirmed_execution_after_success")
        check(not counts["failure_records"], "failure_record_in_successful_run")
        check(
            summary.get("metrics", {}).get("commands") == len(commands),
            "summary_command_count_mismatch",
        )
        check(
            summary["policy"].get("inference_calls") == sum(actions_by_robot.values()),
            "summary_neural_action_count_mismatch",
        )
        check(
            summary["policy"].get("peer_fusion_calls") == counts["peer_fusion_actions"],
            "summary_peer_fusion_count_mismatch",
        )
    else:
        check(
            status in ("failed", "cancelled", "exhausted", "running", "ready"),
            "unknown_summary_status",
        )
        warnings.append(
            "Run did not report success; incomplete robot/command coverage is retained."
        )
    if pending_observed:
        warnings.append(
            "Physical observations without logged commit are excluded from success evidence."
        )
    if durable:
        rows = durable.select("factory_run")
        check(len(rows) == 1, "durable_run_missing_or_duplicate")
        if len(rows) == 1:
            db = rows[0]
            db_state = json.loads(db["state_json"])
            db_state.pop("abort_reason", None)
            check(db_state == final, "durable_final_state_mismatch")
            check(
                db["tick"] == final.get("tick") and db["revision"] == counts["committed_ticks"],
                "durable_final_revision_mismatch",
            )
            check(
                db["status"] == ("failed" if status == "cancelled" else status),
                "durable_final_status_mismatch",
            )
        db_commands = {r["command_id"]: r for r in durable.select("factory_command")}
        for cid, outcome in command_status.items():
            row = db_commands.get(cid, {})
            check(
                row.get("robot_id") == commands[cid]["robot_id"]
                and (
                    row.get("status") == outcome
                    or outcome == "running"
                    and status in ("failed", "cancelled")
                    and row.get("status") == "aborted"
                ),
                "durable_command_status_mismatch",
            )
        if status == "succeeded":
            check(set(db_commands) == set(commands), "durable_extra_commands_after_success")
        spaces = durable.select("factory_embedding_space")
        configured = json.loads(spaces[0]["config_json"]) if len(spaces) == 1 else {}
        expected_space = summary.get("embeddings", {}).get(
            "enabled", bool(counts["committed_vectors"])
        )
        check(
            len(spaces) == int(bool(expected_space))
            and (
                not spaces
                or configured.get("space_id") == "factory-vla-v1"
                and configured.get("checkpoint_sha256") == expected_hash
            ),
            "durable_embedding_space_mismatch",
        )
        stored_vectors = {
            row["embedding_id"]: json.loads(row["record_json"])
            for row in durable.select("factory_embedding", "embedding_id, record_json")
        }
        check(stored_vectors == vectors, "durable_retained_vectors_mismatch")
        counts["durable_embedding_rows_checked"] = len(stored_vectors)
    valid = not counts["errors"]
    return {
        "schema": "relay.factory.audit.v1",
        "run_id": run_id,
        "reported_status": status,
        "evidence_valid": valid,
        "success_verified": valid and status == "succeeded" and all_completed,
        "final_committed_tick": final.get("tick"),
        "checkpoint_sha256": expected_hash,
        "neural_actions_by_robot": dict(actions_by_robot),
        "committed_ticks": counts["committed_ticks"],
        "committed_vectors": counts["committed_vectors"],
        "peer_fusion_actions": counts["peer_fusion_actions"],
        "commands": dict(Counter(command_status.values())),
        "unconfirmed_observations": len(pending_observed),
        "spacetime_checked": durable is not None,
        "durable_commits_checked": counts["durable_commits_checked"],
        "durable_embedding_rows_checked": counts["durable_embedding_rows_checked"],
        "error_count": counts["errors"],
        "issues": issues,
        "warnings": warnings,
        "scope": "Recorded evidence consistency; no independent model-execution attestation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--spacetime",
        action="store_true",
        help="Read configured owner-authorized database; never mutate",
    )
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    try:
        client = SpacetimeClient(SpacetimeConfig.from_env()) if args.spacetime else None
        result = audit_run(args.run_directory, client=client)
    except Exception as error:  # noqa: BLE001 - never print bearer tokens or full SQL/log payloads
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
