"""Client fault handling plus opt-in tests against the real published module."""

import json
import os
from dataclasses import asdict, replace
from uuid import uuid4

import pytest

from relay_lab.action_feedback import ActionFeedback
from relay_lab.spacetime import (
    SpacetimeClient,
    SpacetimeConfig,
    SpacetimeConflict,
    SpacetimeError,
    SpacetimeRejected,
    SpacetimeUnavailable,
)
from relay_lab.types import Action, Instruction, Observation


def client(**kwargs):
    return SpacetimeClient(SpacetimeConfig(token="test-token", **kwargs))


def sql_result(rows):
    names = list(rows[0]) if rows else ["task_id"]
    return json.dumps(
        [
            {
                "schema": {"elements": [{"name": {"some": name}} for name in names]},
                "rows": [[row[name] for name in names] for row in rows],
            }
        ]
    ).encode()


@pytest.mark.parametrize(
    "server",
    [
        "http://example.com",
        "http://user:secret@localhost",
        "http://127.0.0.1/path",
        "http://127.0.0.1?secret=x",
        "http://127.0.0.1#secret",
        "http://localhost:wrong",
    ],
)
def test_config_does_not_send_credentials_to_insecure_or_ambiguous_origins(server):
    with pytest.raises(ValueError):
        client(server=server)


def test_token_is_private_and_not_in_config_repr(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_SPACETIME_TOKEN", raising=False)
    token = tmp_path / "token"
    token.write_text("secret-token")
    token.chmod(0o600)
    config = SpacetimeConfig.from_env(token_file=token)
    assert config.token == "secret-token"
    assert "secret-token" not in repr(config)
    assert "secret-token" not in repr(SpacetimeClient(config).backend_info)
    token.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        SpacetimeConfig.from_env(token_file=token)
    token.chmod(0o600)
    symlink = tmp_path / "linked-token"
    symlink.symlink_to(token)
    with pytest.raises(ValueError, match="private Spacetime token"):
        SpacetimeConfig.from_env(token_file=symlink)


def test_environment_token_overrides_file_without_creating_or_printing_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAY_SPACETIME_TOKEN", "environment-secret")
    config = SpacetimeConfig.from_env(token_file=tmp_path / "missing")
    assert config.token == "environment-secret"
    assert not (tmp_path / "missing").exists()


def test_sql_reads_named_sats_columns_and_rejects_injection(monkeypatch):
    backend = client()
    requests = []

    def request(endpoint, body, **kwargs):
        requests.append((endpoint, body))
        return sql_result([{"task_id": "safe-id", "phase": "ready", "next_tick": 4}])

    monkeypatch.setattr(backend, "_request", request)
    assert backend.get_task("safe-id")["next_tick"] == 4
    assert requests == [("/sql", b"SELECT * FROM task WHERE task_id = 'safe-id'")]
    with pytest.raises(ValueError):
        backend.get_task("x'; DELETE FROM task;--")
    assert len(requests) == 1


@pytest.mark.parametrize(
    "response",
    [b"{}", b"[]", b"invalid", b"[{}]", b'[{"schema":{"elements":[{"name":"a"}]},"rows":[[]]}]'],
)
def test_invalid_sql_response_cannot_grant_permission_to_act(response, monkeypatch):
    backend = client()
    monkeypatch.setattr(backend, "_request", lambda *args, **kwargs: response)
    with pytest.raises(SpacetimeError, match="Invalid Spacetime SQL response"):
        backend.get_task("task")


def test_lost_mutation_response_retries_exact_idempotent_body_and_requires_ack(monkeypatch):
    backend = client()
    stored = {}
    calls = []

    def request(endpoint, body, **kwargs):
        if endpoint == "/sql":
            return sql_result([stored])
        calls.append((endpoint, body))
        task_id, worker_id = json.loads(body)
        stored.update(task_id=task_id, worker_id=worker_id)
        if len(calls) == 1:
            raise SpacetimeUnavailable("Response lost after commit")
        return b""

    monkeypatch.setattr(backend, "_request", request)
    assert backend.claim_task("task", "worker") == stored
    assert calls == [calls[0], calls[0]]
    assert calls[0][0] == "/call/claim_task?confirmed=true"


def test_visible_state_without_durability_ack_still_stops_execution(monkeypatch):
    backend = client()
    calls = []

    def request(endpoint, body, **kwargs):
        if endpoint == "/sql":
            return sql_result([{"task_id": "task", "worker_id": "worker"}])
        calls.append(endpoint)
        raise SpacetimeUnavailable("All responses lost")

    monkeypatch.setattr(backend, "_request", request)
    with pytest.raises(SpacetimeUnavailable, match="commit status is uncertain"):
        backend.claim_task("task", "worker")
    assert len(calls) == 2


def test_conflicting_readback_after_timeout_is_never_overwritten(monkeypatch):
    backend = client()
    calls = []

    def request(endpoint, body, **kwargs):
        if endpoint == "/sql":
            return sql_result([{"task_id": "task", "worker_id": "other-worker"}])
        calls.append(endpoint)
        raise SpacetimeUnavailable("Response lost")

    monkeypatch.setattr(backend, "_request", request)
    with pytest.raises(SpacetimeConflict):
        backend.claim_task("task", "worker")
    assert len(calls) == 1


def test_definitive_rejection_is_not_retried(monkeypatch):
    backend = client()
    calls = []

    def request(endpoint, *args, **kwargs):
        calls.append(endpoint)
        raise SpacetimeRejected("Unauthorized")

    monkeypatch.setattr(backend, "_request", request)
    with pytest.raises(SpacetimeRejected):
        backend.claim_task("task", "worker")
    assert calls == ["/call/claim_task?confirmed=true"]


def test_assignment_context_is_separate_per_robot_and_requires_both(monkeypatch):
    backend = client()
    rows = [
        {
            "agent_id": "picker",
            "last_feedback_tick": 2,
            "last_feedback_json": json.dumps({"text": "picker observation", "critical": True}),
        },
        {
            "agent_id": "packer",
            "last_feedback_tick": 2,
            "last_feedback_json": json.dumps({"text": "packer observation", "critical": False}),
        },
    ]
    monkeypatch.setattr(backend, "_request", lambda *args, **kwargs: sql_result(rows))
    assert backend.task_context("task") == {
        "picker": {"text": "picker observation", "critical": True, "tick": 2},
        "packer": {"text": "packer observation", "critical": False, "tick": 2},
    }
    rows.pop()
    with pytest.raises(SpacetimeError, match="incomplete"):
        backend.task_context("task")


def test_nonfinite_payload_does_not_reach_server(monkeypatch):
    backend = client()
    monkeypatch.setattr(backend, "_request", lambda *args, **kwargs: pytest.fail("Network call"))
    with pytest.raises(ValueError):
        backend.prepare_tick("task", "worker", 0, {"action": float("nan")}, {})


def _frames(task_id, tick=0):
    plans, actions, reports = {}, {}, {}
    for agent in ("picker", "packer"):
        before = Observation(
            tick=tick,
            agent_id=agent,
            ee_position=(0.0, 0.0, 0.3),
            holding=None,
            objects={},
            targets={},
            packed=(None, None, None),
            order=("coral", "mint", "gold") if agent == "packer" else None,
        )
        after = replace(before, tick=tick + 1)
        instruction, action = Instruction("wait", "Wait for coordination."), Action()
        feedback = ActionFeedback(agent, task_id)
        report = feedback.update(before, after, instruction, action)
        before_data, after_data = asdict(before), asdict(after)
        for data in (before_data, after_data):
            del data["rgb"], data["wrist_rgb"]
        plans[agent] = {
            "instruction": asdict(instruction),
            "context": {"text": "", "critical": False},
            "observation": before_data,
        }
        actions[agent] = {"action": asdict(action)}
        reports[agent] = {
            "report": report.to_dict(),
            "text": feedback.context(),
            "critical": report.critical,
            "execution_events": [],
            "observation": after_data,
        }
    return plans, actions, reports


@pytest.mark.skipif(
    os.environ.get("RELAY_SPACETIME_INTEGRATION") != "1",
    reason="Requires real local SpacetimeDB and published relay-lab module",
)
def test_real_spacetime_transaction_contract_and_restart_safe_client():
    backend = SpacetimeClient(SpacetimeConfig.from_env())
    task_id, worker = f"contract-{uuid4()}", str(uuid4())
    spec = {"kind": "relay", "order": ["coral", "mint", "gold"]}
    backend.create_task(task_id, "Verify coordination transaction contract", spec, 1)
    backend.claim_task(task_id, worker)
    assert backend.task_context(task_id)["picker"] == {"text": "", "critical": False, "tick": -1}
    with pytest.raises(SpacetimeError):
        backend.claim_task(task_id, "different-worker")
    plans, actions, reports = _frames(task_id)
    with pytest.raises(SpacetimeError):
        backend.prepare_tick(task_id, worker, 0, actions["picker"], actions["packer"])
    backend.plan_tick(task_id, worker, 0, plans["picker"], plans["packer"])
    prepared = backend.prepare_tick(task_id, worker, 0, actions["picker"], actions["packer"])
    assert (
        backend.prepare_tick(task_id, worker, 0, actions["picker"], actions["packer"]) == prepared
    )
    assert backend.get_task(task_id)["next_tick"] == 0
    changed = {"action": {"delta": [0.01, 0.0, 0.0], "gripper": 0}}
    with pytest.raises(SpacetimeError):
        backend.prepare_tick(task_id, worker, 0, changed, actions["packer"])
    outcome = {"tick": 1, "packed": [None, None, None], "order": spec["order"], "success": False}
    fake_success = {**outcome, "success": True}
    with pytest.raises(SpacetimeError):
        backend.commit_tick(task_id, worker, 0, reports["picker"], reports["packer"], fake_success)
    assert backend.get_feedback(task_id, 0) is None
    assert backend.get_task(task_id)["phase"] == "executing"
    committed = backend.commit_tick(
        task_id, worker, 0, reports["picker"], reports["packer"], outcome
    )
    assert (
        backend.commit_tick(task_id, worker, 0, reports["picker"], reports["packer"], outcome)
        == committed
    )
    other_client = SpacetimeClient(SpacetimeConfig.from_env())
    state = other_client.get_task(task_id)
    assert (state["status"], state["phase"], state["next_tick"]) == ("exhausted", "terminal", 1)
    assert other_client.task_context(task_id)["picker"]["text"] == reports["picker"]["text"]
    assert other_client.get_feedback(task_id, 0) == committed
