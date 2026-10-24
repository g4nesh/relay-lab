"""Mutation tests for the evidence auditor; the real benchmark is audited separately.

The small protocol fixture is deliberately not a physical-simulation claim. Its
visual payload comes from the shipped held-out RGB crops and actual codec.
"""

import gzip
import importlib.util
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from relay_lab.factory.contracts import STAGES
from relay_lab.factory.routing_codec import DEFAULT_CHECKPOINT, RoutingCodec
from relay_lab.factory.scene import ROBOT_SPECS, pallet_slot

_SPEC = importlib.util.spec_from_file_location(
    "communication_audit", Path(__file__).parents[1] / "scripts/audit_factory_communication_run.py"
)
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)
RUN, SHA = "factory-routing-audit-test", "a" * 64


@pytest.fixture(scope="module")
def codec():
    pytest.importorskip("torch")
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("Train the inspection codec first")
    return RoutingCodec()


@pytest.fixture(scope="module")
def encoded(codec):
    with np.load(DEFAULT_CHECKPOINT.with_name("verification_rgb.npz")) as data:
        return {
            route: codec.encode(data["images"][np.flatnonzero(data["labels"] == i)[0]])
            for i, route in enumerate(("A", "B"))
        }


def write_run(path, rows, summary, *, compressed=False):
    data = "\n".join(json.dumps(r) for r in rows) + "\n"
    if compressed:
        with gzip.open(path / "events.jsonl.gz", "wt") as stream:
            stream.write(data)
    else:
        (path / "events.jsonl").write_text(data)
    (path / "summary.json").write_text(json.dumps(summary))


def saved_run(path, codec, encoded, *, mode="embeddings", truth="A", compressed=False):
    policy = {"policy": "factory-vla", "checkpoint_sha256": SHA, "inference_calls": 9}
    info = codec.model_info
    frame = encoded["A"]
    origin = {
        "job_id": "job-001",
        "observation_tick": 0,
        "source_command_id": "cmd-1",
        "source_frame_sha256": frame["source_frame_sha256"],
    }
    channel = {
        "mode": mode,
        "model": info,
        "period_ticks": 5,
        "ttl_ticks": 15,
        "published": 0,
        "local_observations": 0,
        "decisions": [],
        "observation_provenance": [],
    }
    job = {
        "id": "job-001",
        "job_id": "job-001",
        "cargo_id": "part-001",
        "index": 0,
        "stage": "queued",
        "custodian": None,
        "owner": None,
        "location": "receive",
        "position": [-8.0, 1.3, 1.02],
        "speed": 0.0,
        "processed": False,
        "delivered": False,
    }
    initial = {
        "task": "dispatch-routing",
        "tick": 0,
        "success": False,
        "physical_complete": False,
        "completed": 0,
        "jobs": {"job-001": job},
        "cargo": [deepcopy(job)],
        "robots": [{"id": robot, "command": None} for robot in ROBOT_SPECS],
        "motor_policy": {"model": policy},
    }
    rows = [
        {
            "type": "created",
            "run_id": RUN,
            "state": deepcopy(initial),
            "policy": policy,
            "communication": deepcopy(channel),
        }
    ]
    state, commands = deepcopy(initial), {}
    actors = [
        "humanoid-01",
        "humanoid-01",
        "amr-01",
        "arm-01",
        "arm-01",
        "arm-01",
        "amr-02",
        "gantry-01",
        "gantry-01",
    ]
    stages = ["queued", *STAGES[1:-1], "at_dispatch", "completed"]
    decision = None
    for tick, (robot, stage) in enumerate(zip(actors, stages, strict=True), 1):
        cid = "cmd-1" if tick <= 2 else "cmd-7" if tick >= 8 else f"cmd-{tick - 1}"
        new = cid not in commands
        if tick == 8:
            decision = {
                "job_id": "job-001",
                "robot_id": "gantry-01",
                "tick": 7,
                "route": "A",
                "pallet_slot": 0,
                "packet_id": "humanoid-01-job-001-5" if mode != "none" else None,
                "observation_tick": 0 if mode != "none" else None,
                "emitted_tick": 5 if mode != "none" else None,
                "source_frame_sha256": origin["source_frame_sha256"] if mode != "none" else None,
                "transport_age_ticks": 2 if mode != "none" else None,
                "observation_age_ticks": 7 if mode != "none" else None,
                "confidence": None,
                "source": "fixture",
            }
            rows.append({"type": "route_decision", "run_id": RUN, **decision})
        if new:
            commands[cid] = {
                "command_id": cid,
                "robot_id": robot,
                "job_id": "job-001",
                "capability": "transfer",
                "cargo_id": "part-001",
                "target": "pallet" if robot == "gantry-01" else "fixture",
            }
            if robot == "gantry-01":
                commands[cid].update(pallet_slot=0, route_packet_id=decision["packet_id"])
        command = commands[cid]
        state = deepcopy(state)
        state["tick"] = tick
        state["jobs"]["job-001"]["stage"] = stage
        events = [{"type": "command_started", "command_id": cid, "tick": tick - 1}] if new else []
        if tick == 9:
            state["jobs"]["job-001"].update(
                location="pallet",
                processed=True,
                delivered=True,
                delivery_slot=0,
                routing_correct=truth == "A",
                position=list(pallet_slot(0)),
            )
            state.update(physical_complete=True, success=truth == "A", completed=1)
            events.append(
                {
                    "type": "delivery_verified",
                    "command_id": cid,
                    "tick": tick - 1,
                    "cargo_id": "part-001",
                    "slot": 0,
                    "routing_correct": truth == "A",
                }
            )
        state["cargo"] = [deepcopy(state["jobs"]["job-001"])]
        for item in state["robots"]:
            item["command"] = deepcopy(command) if item["id"] == robot else None
        subgoal = "loaded" if tick <= 2 else "completed" if tick >= 8 else stage
        messages = (
            [
                {
                    "kind": "intent",
                    "sender": robot,
                    "payload": {"job_id": "job-001", "command_id": cid, "subgoal": subgoal},
                }
            ]
            if new
            else []
        )
        instruction = "place part-001 on pallet-001" if tick == 9 else "pick part-001"
        action = {
            "robot_id": robot,
            "command_id": cid,
            "tick": tick - 1,
            "phase": "approach_place" if tick == 9 else "approach_pick",
            "instruction": instruction,
            "raw_action": {"delta": [0.01, 0.0, 0.0]},
            "peer_record_ids": [],
            "provenance": {
                "model_id": "relay-lab/factory-vla-v1",
                "checkpoint_sha256": SHA,
                "instruction": instruction,
                "peer_fusion": False,
                "raw_output": [0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
        }
        outcome = {
            "command_id": cid,
            "robot_id": robot,
            "status": "running" if tick in (1, 8) else "completed",
            "observed": {"stage": stage, "custodian": None},
        }
        packets = []
        if tick in (1, 5) and mode != "none":
            packets = [
                {
                    "schema": "relay.factory.semantic.v1",
                    "mode": mode,
                    "run_id": RUN,
                    "space_id": "factory-route-v1",
                    "checkpoint_sha256": info["checkpoint_sha256"],
                    "sender": "humanoid-01",
                    "recipient": "gantry-01",
                    "topic": "dispatch_route",
                    "packet_id": f"humanoid-01-job-001-{tick}",
                    "emitted_tick": tick,
                    "expires_tick": tick + 15,
                    **origin,
                    **(
                        {"vector": frame["vector"]}
                        if mode == "embeddings"
                        else {"text": '{"route":"A"}'}
                    ),
                }
            ]
        record = {
            "tick": tick,
            "state": deepcopy(state),
            "outcomes": [outcome],
            "events": events,
            "messages": messages,
            "motor_actions": [action],
            "embeddings": [],
            "semantic_packets": packets,
        }
        rows.extend(
            [
                {
                    "type": "execution_observed",
                    "run_id": RUN,
                    "revision": tick - 1,
                    "committed": False,
                    **deepcopy(record),
                },
                {"type": "tick", "run_id": RUN, **record},
            ]
        )
    channel.update(
        published=2 if mode != "none" else 0,
        local_observations=1,
        observation_provenance=[origin],
        decisions=[decision],
    )
    summary = {
        **deepcopy(state),
        "run_id": RUN,
        "status": "succeeded" if truth == "A" else "rejected",
        "policy": policy,
        "communication": channel,
        "metrics": {"commands": 7},
        "routing_oracle": {
            "job-001": {
                "route": truth,
                "required_slot": int(truth == "B"),
                "selected_slot": 0,
                "correct": truth == "A",
            }
        },
    }
    write_run(path, rows, summary, compressed=compressed)
    return rows, summary


def codes(result):
    return {issue["code"] for issue in result["issues"]}


@pytest.mark.parametrize("mode", ["embeddings", "text", "none"])
def test_complete_committed_chain_and_gzip(codec, encoded, tmp_path, mode):
    saved_run(tmp_path, codec, encoded, mode=mode, compressed=True)
    result = _AUDIT.audit_run(tmp_path, codec=codec)
    assert result["evidence_valid"], result
    assert result["success_verified"] and result["committed_ticks"] == 9


def test_rejected_wrong_route_is_valid_failed_evidence(codec, encoded, tmp_path):
    saved_run(tmp_path, codec, encoded, mode="none", truth="B")
    result = _AUDIT.audit_run(tmp_path, codec=codec)
    assert result["evidence_valid"], result
    assert result["physical_completion_verified"] and not result["success_verified"]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("origin", "changed_observation_or_payload"),
        ("extra_truth", "packet_schema_or_scope"),
        ("decoder", "decision_decode_mismatch"),
        ("uncommitted", "uncommitted_decision_packet"),
        ("stale", "ineligible_decision_packet"),
        ("authorize", "authorized_slot_or_packet_mismatch"),
        ("motor", "motor_route_instruction_mismatch"),
        ("physical", "unverified_physical_delivery"),
        ("oracle", "terminal_oracle_inconsistency"),
        ("commit", "physical_commit_state"),
    ],
)
def test_causal_chain_mutations_rejected(codec, encoded, tmp_path, mutation, expected):
    rows, summary = saved_run(tmp_path, codec, encoded)
    ticks = {row["tick"]: row for row in rows if row["type"] == "tick"}
    decision = next(row for row in rows if row["type"] == "route_decision")
    if mutation == "origin":
        ticks[5]["semantic_packets"][0]["source_frame_sha256"] = "0" * 64
    elif mutation == "extra_truth":
        ticks[1]["semantic_packets"][0]["private_route"] = "A"
    elif mutation == "decoder":
        for tick in (1, 5):
            ticks[tick]["semantic_packets"][0]["vector"] = encoded["B"]["vector"]
    elif mutation == "uncommitted":
        decision["packet_id"] = "humanoid-01-job-001-10"
    elif mutation == "stale":
        ticks[5]["semantic_packets"][0]["expires_tick"] = 7
    elif mutation == "authorize":
        command = next(r["command"] for r in ticks[8]["state"]["robots"] if r["id"] == "gantry-01")
        command["pallet_slot"] = 1
    elif mutation == "motor":
        action = ticks[9]["motor_actions"][0]
        action["instruction"] = action["provenance"]["instruction"] = "place part-001 on pallet-002"
    elif mutation == "physical":
        ticks[9]["state"]["jobs"]["job-001"]["position"] = list(pallet_slot(1))
        summary["jobs"] = deepcopy(ticks[9]["state"]["jobs"])
    elif mutation == "oracle":
        summary["routing_oracle"]["job-001"]["route"] = "B"
    elif mutation == "commit":
        ticks[4]["state"]["cargo"] = []
    write_run(tmp_path, rows, summary)
    result = _AUDIT.audit_run(tmp_path, codec=codec)
    assert not result["evidence_valid"]
    assert expected in codes(result), result


class ReadOnlyDatabase:
    """Durable commit history can disagree while the final snapshot stays intact."""

    def __init__(self, rows, summary):
        self.queries = []
        ticks = [row for row in rows if row["type"] == "tick"]
        self.tables = {
            "factory_run": [
                {
                    "run_id": RUN,
                    "tick": 9,
                    "revision": 9,
                    "status": summary["status"],
                    "state_json": json.dumps(ticks[-1]["state"]),
                }
            ],
            "factory_semantic_channel": [
                {
                    "config_json": json.dumps(
                        {
                            "schema": "relay.factory.semantic.v1",
                            "space_id": "factory-route-v1",
                            "mode": summary["communication"]["mode"],
                            "period_ticks": 5,
                            "ttl_ticks": 15,
                            "checkpoint_sha256": summary["communication"]["model"][
                                "checkpoint_sha256"
                            ],
                        }
                    )
                }
            ],
            "factory_semantic_packet": [],
            "factory_batch": [],
        }
        for row in ticks:
            commands = []
            for robot in row["state"]["robots"]:
                command = robot.get("command")
                if command:
                    keys = {"command_id", "robot_id", "job_id", "capability"}
                    commands.append(
                        {
                            **{k: v for k, v in command.items() if k in keys},
                            "payload": {k: v for k, v in command.items() if k not in keys},
                        }
                    )
            self.tables["factory_batch"].append(
                {
                    "revision": row["tick"] - 1,
                    "prepare_json": json.dumps({"commands": commands, "messages": row["messages"]}),
                    "result_json": json.dumps(
                        {
                            "state": row["state"],
                            "outcomes": row["outcomes"],
                            "messages": [],
                            "embeddings": [],
                            "semantic_packets": row["semantic_packets"],
                        }
                    ),
                }
            )

    def _sql(self, sql):
        assert sql.startswith("SELECT ")
        self.queries.append(sql)
        table = re.search(r"FROM (\w+)", sql).group(1)
        rows = self.tables[table]
        if table == "factory_batch":
            low, high = map(int, re.search(r"revision >= (\d+) AND revision < (\d+)", sql).groups())
            rows = [r for r in rows if low <= r["revision"] < high]
        return deepcopy(rows)


def test_durable_audit_reads_every_commit_not_only_final_snapshot(codec, encoded, tmp_path):
    rows, summary = saved_run(tmp_path, codec, encoded)
    database = ReadOnlyDatabase(rows, summary)
    valid = _AUDIT.audit_run(tmp_path, codec=codec, client=database)
    assert valid["evidence_valid"], valid
    assert valid["durable_commits_checked"] == 9
    assert database.queries and all(q.startswith("SELECT ") for q in database.queries)
    batch = database.tables["factory_batch"][4]
    stored = json.loads(batch["result_json"])
    stored["semantic_packets"][0]["source_frame_sha256"] = "0" * 64
    batch["result_json"] = json.dumps(stored)
    invalid = _AUDIT.audit_run(tmp_path, codec=codec, client=database)
    assert not invalid["evidence_valid"]
    assert "durable_semantic_packets" in codes(invalid)
