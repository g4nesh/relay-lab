"""The saved-run audit must reject invented success and noncausal peer evidence."""

import importlib.util
import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from relay_lab.factory.contracts import STAGES
from relay_lab.factory.embeddings import SCHEMA
from relay_lab.factory.scene import ROBOT_SPECS

_SPEC = importlib.util.spec_from_file_location(
    "factory_audit", Path(__file__).parents[1] / "scripts/audit_factory_vla_run.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_run = _MODULE.audit_run
SHA, RUN = "a" * 64, "factory-audit-fixture"


def saved_run(tmp_path):
    policy = {"policy": "factory-vla", "checkpoint_sha256": SHA}
    initial = {
        "tick": 0,
        "success": False,
        "completed": 0,
        "robots": [{"id": robot, "active_command": None} for robot in ROBOT_SPECS],
        "jobs": {"job-001": {"stage": "queued", "custodian": None}},
        "motor_policy": {"model": policy},
    }
    rows = [{"type": "created", "run_id": RUN, "state": initial, "policy": policy}]
    state = deepcopy(initial)
    stages = ["queued", *STAGES[1:]]
    actors = [
        "humanoid-01",
        "humanoid-01",
        "amr-01",
        "arm-01",
        "arm-01",
        "arm-01",
        "amr-02",
        "gantry-01",
    ]
    for tick, (stage, robot) in enumerate(zip(stages, actors, strict=True), 1):
        cid = "cmd-1" if tick <= 2 else f"cmd-{tick - 1}"
        status = "running" if tick == 1 else "completed"
        state = deepcopy(state)
        state.update(tick=tick, success=stage == "completed", completed=int(stage == "completed"))
        state["jobs"]["job-001"]["stage"] = stage
        for item in state["robots"]:
            item["active_command"] = cid if tick == 1 and item["id"] == robot else None
        messages = (
            []
            if tick == 2
            else [
                {
                    "kind": "intent",
                    "sender": robot,
                    "payload": {
                        "command_id": cid,
                        "job_id": "job-001",
                        "subgoal": "loaded" if tick == 1 else stage,
                    },
                }
            ]
        )
        action = {
            "robot_id": robot,
            "command_id": cid,
            "tick": tick - 1,
            "peer_record_ids": ["amr-01-1"] if tick == 2 else [],
            "raw_action": {"delta": [0.01, 0, 0]},
            "provenance": {
                "model_id": "relay-lab/factory-vla-v1",
                "checkpoint_sha256": SHA,
                "raw_output": [0.1, 0, 0, 0, 0, 1, 0],
                "peer_fusion": tick == 2,
                "target": "amr-01",
            },
        }
        outcomes = [
            {
                "command_id": cid,
                "robot_id": robot,
                "status": status,
                "observed": {"stage": stage, "custodian": None},
            }
        ]
        records = [
            {
                "schema": SCHEMA,
                "embedding_id": f"{source}-{tick}",
                "run_id": RUN,
                "robot_id": source,
                "tick": tick,
                "expires_tick": tick + 15,
                "checkpoint_sha256": SHA,
                "space_id": "factory-vla-v1",
                "dimension": 32,
                "modality": "vla-hidden",
                "vector": [1.0] + [0.0] * 31,
                "command_id": cid if tick == 1 and source == robot else None,
                "job_id": "job-001" if tick == 1 and source == robot else None,
            }
            for source in ROBOT_SPECS
        ]
        record = {
            "type": "tick",
            "run_id": RUN,
            "tick": tick,
            "state": state,
            "outcomes": outcomes,
            "motor_actions": [action],
            "events": [],
            "messages": messages,
            "embeddings": records,
        }
        rows.append(
            {
                **deepcopy(record),
                "type": "execution_observed",
                "revision": tick - 1,
                "committed": False,
            }
        )
        rows.append(record)
    summary = {
        **deepcopy(state),
        "run_id": RUN,
        "status": "succeeded",
        "policy": {**policy, "inference_calls": 8, "peer_fusion_calls": 1},
        "metrics": {"commands": 7},
    }
    write_run(tmp_path, rows, summary)
    return rows, summary


def write_run(path, rows, summary):
    (path / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "summary.json").write_text(json.dumps(summary))


def codes(result):
    return {issue["code"] for issue in result["issues"]}


def test_success_requires_complete_neural_and_causal_evidence(tmp_path):
    saved_run(tmp_path)
    result = audit_run(tmp_path)
    assert result["evidence_valid"] and result["success_verified"]
    assert result["committed_ticks"] == 8 and result["peer_fusion_actions"] == 1
    assert all(result["neural_actions_by_robot"].values())


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("future", "ineligible_or_nonlatest_peer_vector"),
        ("expired", "ineligible_or_nonlatest_peer_vector"),
        ("wrong_target", "fusion_target_peer_mismatch"),
        ("wrong_hash", "invalid_neural_action_provenance"),
        ("wrong_run", "invalid_committed_vector_provenance"),
        ("nan", "invalid_committed_vector_values"),
        ("replayed", "terminal_command_replayed"),
    ],
)
def test_corrupted_evidence_cannot_pass(tmp_path, mutation, expected):
    rows, summary = saved_run(tmp_path)
    tick2 = next(row for row in rows if row["type"] == "tick" and row["tick"] == 2)
    peer = next(row for row in rows if row["type"] == "tick")["embeddings"][1]
    if mutation == "future":
        tick2["motor_actions"][0]["peer_record_ids"] = ["amr-01-2"]
    elif mutation == "expired":
        peer["expires_tick"] = 1
    elif mutation == "wrong_target":
        tick2["motor_actions"][0]["provenance"]["target"] = "amr-02"
    elif mutation == "wrong_hash":
        tick2["motor_actions"][0]["provenance"]["checkpoint_sha256"] = "b" * 64
    elif mutation == "wrong_run":
        peer["run_id"] = "other-run"
    elif mutation == "nan":
        peer["vector"][0] = float("nan")
    else:
        next(row for row in rows if row["type"] == "tick" and row["tick"] == 3)["outcomes"][0][
            "command_id"
        ] = "cmd-1"
    write_run(tmp_path, rows, summary)
    result = audit_run(tmp_path)
    assert not result["evidence_valid"] and not result["success_verified"]
    assert expected in codes(result)


def test_failed_run_excludes_unconfirmed_physics_without_inventing_success(tmp_path):
    rows, summary = saved_run(tmp_path)
    rows = rows[:4]  # initial, committed tick 1, then only physical tick 2
    final = rows[2]["state"]
    summary.update(deepcopy(final), status="failed")
    write_run(tmp_path, rows, summary)
    result = audit_run(tmp_path)
    assert result["evidence_valid"] and not result["success_verified"]
    assert result["final_committed_tick"] == 1 and result["unconfirmed_observations"] == 1
    summary["status"] = "succeeded"
    write_run(tmp_path, rows, summary)
    result = audit_run(tmp_path)
    assert "unsupported_summary_success" in codes(result)
    assert "unconfirmed_execution_after_success" in codes(result)


class ReadOnlyDatabase:
    def __init__(self, rows, summary):
        ticks = [row for row in rows if row["type"] == "tick"]
        self.tables = {
            "factory_run": [
                {
                    "state_json": json.dumps(ticks[-1]["state"]),
                    "tick": 8,
                    "revision": 8,
                    "status": "succeeded",
                }
            ],
            "factory_command": [],
        }
        commands = {}
        self.batches = []
        for row in ticks:
            for outcome in row["outcomes"]:
                commands[outcome["command_id"]] = {
                    key: outcome[key] for key in ("command_id", "robot_id", "status")
                }
            self.batches.append(
                {
                    "revision": row["tick"] - 1,
                    "prepare_json": json.dumps(
                        {"commands": row["outcomes"], "messages": row["messages"]}
                    ),
                    "result_json": json.dumps(
                        {
                            "state": row["state"],
                            "outcomes": row["outcomes"],
                            "messages": [],
                            "embeddings": row["embeddings"],
                        }
                    ),
                }
            )
        self.tables["factory_command"] = list(commands.values())
        self.tables["factory_embedding_space"] = [
            {"config_json": json.dumps({"space_id": "factory-vla-v1", "checkpoint_sha256": SHA})}
        ]
        self.tables["factory_embedding"] = [
            {"embedding_id": vector["embedding_id"], "record_json": json.dumps(vector)}
            for row in ticks
            for vector in row["embeddings"]
        ]
        self.queries = []

    def _sql(self, query):
        assert query.startswith("SELECT ") and RUN in query
        self.queries.append(query)
        if "FROM factory_batch " in query:
            start, stop = map(
                int, re.search(r"revision >= (\d+) AND revision < (\d+)", query).groups()
            )
            return [row for row in self.batches if start <= row["revision"] < stop]
        return self.tables[re.search(r"FROM (\w+)", query)[1]]


def test_durable_audit_reads_exact_commits_and_rejects_database_drift(tmp_path):
    rows, summary = saved_run(tmp_path)
    db = ReadOnlyDatabase(rows, summary)
    result = audit_run(tmp_path, client=db)
    assert result["success_verified"] and result["durable_commits_checked"] == 8
    assert len(db.queries) == 5  # batches, final state, commands, embedding space and retained rows
    result = json.loads(db.batches[3]["result_json"])
    result["embeddings"] = []
    db.batches[3]["result_json"] = json.dumps(result)
    result = audit_run(tmp_path, client=db)
    assert not result["success_verified"]
    assert "durable_embeddings_mismatch" in codes(result)


def test_missing_robot_actions_and_missing_commit_are_not_success(tmp_path):
    rows, summary = saved_run(tmp_path)
    for row in rows:
        if row.get("tick") == 8:
            row["motor_actions"] = []
    write_run(tmp_path, rows, summary)
    assert "missing_robot_neural_actions" in codes(audit_run(tmp_path))
    db = ReadOnlyDatabase(rows, summary)
    db.batches[1]["result_json"] = ""
    assert "missing_durable_commit" in codes(audit_run(tmp_path, client=db))


def test_durable_current_embedding_rows_must_match_committed_history(tmp_path):
    rows, summary = saved_run(tmp_path)
    db = ReadOnlyDatabase(rows, summary)
    db.tables["factory_embedding"].pop()
    result = audit_run(tmp_path, client=db)
    assert "durable_retained_vectors_mismatch" in codes(result)
    assert not result["success_verified"]
