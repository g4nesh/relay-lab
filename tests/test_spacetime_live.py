"""Opt-in checks against the real, published SpacetimeDB module.

Run with RELAY_SPACETIME_LIVE_URL, RELAY_SPACETIME_LIVE_DATABASE and
RELAY_SPACETIME_LIVE_TOKEN_FILE set. The file contains the local publisher's
token. These tests create uniquely named tasks; they never clear the database.
Raw HTTP deliberately bypasses the Python coordinator's input validation.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Database:
    def __init__(self, url, database, token):
        self.url = url.rstrip("/")
        self.database = database
        self.token = token
        self.created_tasks = []

    def request(self, path, body, *, content_type="application/json"):
        headers = {"Content-Type": content_type}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.url + path, data=body.encode(), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=20) as response:
                return response.status, response.read().decode()
        except HTTPError as error:
            return error.code, error.read().decode()

    def call(self, reducer, *arguments, accepted=True):
        status, body = self.request(
            f"/v1/database/{quote(self.database, safe='')}/call/{reducer}?confirmed=true",
            encode(list(arguments)),
        )
        if accepted:
            assert 200 <= status < 300, (reducer, status, body)
        else:
            assert status >= 400, (reducer, status, body)
        return status, body

    def rows(self, table, task_id):
        assert table in {"task", "assignment", "plan_frame", "action_frame", "feedback_frame"}
        assert all(character.isalnum() or character == "-" for character in task_id)
        status, body = self.request(
            f"/v1/database/{quote(self.database, safe='')}/sql",
            f"SELECT * FROM {table} WHERE task_id = '{task_id}'",
            content_type="text/plain",
        )
        assert status == 200, (status, body)
        result = json.loads(body)[0]
        names = []
        for field in result["schema"]["elements"]:
            name = field["name"]
            names.append(name["some"] if isinstance(name, dict) else name)
        return [dict(zip(names, row, strict=True)) for row in result["rows"]]


@pytest.fixture(scope="module")
def database():
    names = (
        "RELAY_SPACETIME_LIVE_URL",
        "RELAY_SPACETIME_LIVE_DATABASE",
        "RELAY_SPACETIME_LIVE_TOKEN_FILE",
    )
    values = [os.environ.get(name) for name in names]
    if not any(values):
        pytest.skip("Real SpacetimeDB checks are opt-in; set RELAY_SPACETIME_LIVE_* variables")
    if not all(values):
        pytest.fail("All three RELAY_SPACETIME_LIVE_* variables are required")
    token = Path(values[2]).read_text().strip()
    if token.startswith("{"):
        token = json.loads(token)["token"]
    assert token and "\n" not in token
    database = Database(values[0], values[1], token)
    yield database
    # Retain all evidence, but do not leave verification tasks advertised as
    # running after their tests have finished.
    for task_id, worker_id in database.created_tasks:
        if database.rows("task", task_id)[0]["status"] == "running":
            database.call("abort_task", task_id, worker_id, "Live verification finished")


def observation(agent, tick):
    return {
        "agent_id": agent,
        "tick": tick,
        "ee_position": [0.0, 0.0, 0.3],
        "gripper_opening": 0.08,
        "holding": None,
        "objects": {},
        "targets": {},
        "packed": [None, None, None],
        "order": ["coral", "mint", "gold"] if agent == "packer" else None,
    }


def plan(agent, tick=0):
    return {
        "instruction": {
            "kind": "wait",
            "text": "Wait at the current position.",
            "feedback": "",
            "replan": False,
        },
        "context": {"text": "", "critical": False},
        "observation": observation(agent, tick),
    }


def report(task_id, agent, tick=0):
    text = f"Tick {tick + 1}, {agent}: no movement or attachment observed."
    return {
        "report": {
            "schema": "relay-lab.action-feedback.v1",
            "task_id": task_id,
            "agent_id": agent,
            "before_tick": tick,
            "tick": tick + 1,
            "requested_delta": [0.0, 0.0, 0.0],
            "observed_delta": [0.0, 0.0, 0.0],
            "gripper_command": 0,
            "gripper_opening_before": 0.08,
            "gripper_opening_after": 0.08,
            "holding_before": None,
            "holding_after": None,
            "motion_limited": False,
            "local_events": [],
            "feedback": text,
            "critical": False,
        },
        "text": text,
        "critical": False,
        "execution_events": [],
        "observation": observation(agent, tick + 1),
    }


def outcome(tick=0):
    return {
        "tick": tick + 1,
        "packed": [None, None, None],
        "order": ["coral", "mint", "gold"],
        "success": False,
    }


def new_task(database, *, max_steps=3):
    task_id = f"test-{uuid4()}"
    worker_id = f"worker-{uuid4()}"
    spec = encode({"kind": "relay", "order": ["coral", "mint", "gold"]})
    database.call("create_task", task_id, "Verify a synchronized robot tick.", spec, max_steps)
    database.call("claim_task", task_id, worker_id)
    database.created_tasks.append((task_id, worker_id))
    return task_id, worker_id


def prepare(database, task_id, worker_id, *, tick=0):
    database.call(
        "plan_tick",
        task_id,
        worker_id,
        tick,
        encode(plan("picker", tick)),
        encode(plan("packer", tick)),
    )
    action = encode({"action": {"delta": [0.0, 0.0, 0.0], "gripper": 0}})
    database.call("prepare_tick", task_id, worker_id, tick, action, action)


def commit_arguments(task_id, worker_id, *, tick=0):
    return (
        task_id,
        worker_id,
        tick,
        encode(report(task_id, "picker", tick)),
        encode(report(task_id, "packer", tick)),
        encode(outcome(tick)),
    )


def snapshot(database, task_id):
    return {
        table: database.rows(table, task_id)
        for table in ("task", "assignment", "plan_frame", "action_frame", "feedback_frame")
    }


def test_concurrent_commit_is_one_durable_pair_and_one_tick(database):
    task_id, worker_id = new_task(database)
    prepare(database, task_id, worker_id)
    arguments = commit_arguments(task_id, worker_id)
    barrier = Barrier(4)

    def commit():
        barrier.wait(timeout=5)
        return database.call("commit_tick", *arguments)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: commit(), range(4)))
    rows = snapshot(database, task_id)
    assert len(rows["plan_frame"]) == len(rows["action_frame"]) == len(rows["feedback_frame"]) == 1
    assert len(rows["assignment"]) == 2
    assert rows["task"][0]["next_tick"] == 1
    for assignment in rows["assignment"]:
        assert assignment["last_feedback_tick"] == 0
        assert assignment["context_text"]


def test_changed_duplicate_and_stale_tick_leave_state_unchanged(database):
    task_id, worker_id = new_task(database)
    prepare(database, task_id, worker_id)
    arguments = commit_arguments(task_id, worker_id)
    database.call("commit_tick", *arguments)
    before = snapshot(database, task_id)
    changed = list(arguments)
    changed_report = report(task_id, "picker")
    changed_report["text"] = "A different report for the same operation."
    changed[3] = encode(changed_report)
    database.call("commit_tick", *changed, accepted=False)
    database.call(
        "plan_tick",
        task_id,
        worker_id,
        2,
        encode(plan("picker", 2)),
        encode(plan("packer", 2)),
        accepted=False,
    )
    database.call("claim_task", task_id, "another-worker", accepted=False)
    assert snapshot(database, task_id) == before


def test_plan_and_action_retries_cannot_replace_authorized_execution(database):
    task_id, worker_id = new_task(database)
    prepare(database, task_id, worker_id)
    before = snapshot(database, task_id)
    # Identical retries do not change timestamps or create another frame.
    prepare(database, task_id, worker_id)
    changed_plan = plan("picker")
    changed_plan["instruction"]["text"] = "Move to a different location."
    database.call(
        "plan_tick",
        task_id,
        worker_id,
        0,
        encode(changed_plan),
        encode(plan("packer")),
        accepted=False,
    )
    changed_action = encode({"action": {"delta": [0.01, 0.0, 0.0], "gripper": 0}})
    same_action = encode({"action": {"delta": [0.0, 0.0, 0.0], "gripper": 0}})
    database.call(
        "prepare_tick", task_id, worker_id, 0, changed_action, same_action, accepted=False
    )
    assert snapshot(database, task_id) == before


def test_missing_or_invalid_second_report_rolls_back_entire_tick(database):
    task_id, worker_id = new_task(database)
    prepare(database, task_id, worker_id)
    before = snapshot(database, task_id)
    arguments = list(commit_arguments(task_id, worker_id))
    arguments[4] = "{}"
    database.call("commit_tick", *arguments, accepted=False)
    assert snapshot(database, task_id) == before
    invalid = report(task_id, "packer")
    invalid["report"]["before_tick"] = 1
    arguments[4] = encode(invalid)
    database.call("commit_tick", *arguments, accepted=False)
    assert snapshot(database, task_id) == before


def test_observed_motion_must_match_authorized_action_and_actual_observation(database):
    task_id, worker_id = new_task(database)
    prepare(database, task_id, worker_id)
    before = snapshot(database, task_id)
    arguments = list(commit_arguments(task_id, worker_id))
    different_action = report(task_id, "picker")
    different_action["report"]["requested_delta"] = [0.01, 0.0, 0.0]
    arguments[3] = encode(different_action)
    database.call("commit_tick", *arguments, accepted=False)
    invented_motion = report(task_id, "picker")
    invented_motion["report"]["observed_delta"] = [0.01, 0.0, 0.0]
    arguments[3] = encode(invented_motion)
    database.call("commit_tick", *arguments, accepted=False)
    assert snapshot(database, task_id) == before


def test_execution_requires_prepared_actions_and_observed_success(database):
    task_id, worker_id = new_task(database)
    database.call("commit_tick", *commit_arguments(task_id, worker_id), accepted=False)
    prepare(database, task_id, worker_id)
    before = snapshot(database, task_id)
    arguments = list(commit_arguments(task_id, worker_id))
    fabricated_success = outcome()
    fabricated_success["success"] = True
    arguments[-1] = encode(fabricated_success)
    database.call("commit_tick", *arguments, accepted=False)
    assert snapshot(database, task_id) == before


def test_separate_client_reads_committed_feedback_and_terminal_fence(database):
    task_id, worker_id = new_task(database, max_steps=1)
    prepare(database, task_id, worker_id)
    arguments = commit_arguments(task_id, worker_id)
    database.call("commit_tick", *arguments)
    second_client = Database(database.url, database.database, database.token)
    before = snapshot(second_client, task_id)
    assert before["task"][0]["next_tick"] == 1
    assert len(before["feedback_frame"]) == 1
    database.call("commit_tick", *arguments)  # An exact retry remains safe after completion.
    database.call(
        "plan_tick",
        task_id,
        worker_id,
        1,
        encode(plan("picker", 1)),
        encode(plan("packer", 1)),
        accepted=False,
    )
    assert snapshot(second_client, task_id) == before


def test_unrelated_identity_cannot_write_or_read_private_task(database):
    task_id, worker_id = new_task(database)
    before = snapshot(database, task_id)
    anonymous = Database(database.url, database.database, "")
    status, body = anonymous.request("/v1/identity", "")
    assert status == 200, (status, body)
    stranger = Database(database.url, database.database, json.loads(body)["token"])
    stranger.call("claim_task", task_id, worker_id, accepted=False)
    stranger.call("abort_task", task_id, worker_id, "unauthorized abort", accepted=False)
    status, _ = stranger.request(
        f"/v1/database/{quote(database.database, safe='')}/sql",
        f"SELECT * FROM task WHERE task_id = '{task_id}'",
        content_type="text/plain",
    )
    assert status >= 400
    assert snapshot(database, task_id) == before
