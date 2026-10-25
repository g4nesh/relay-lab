"""Real auth/governance and task dispatch, with only Spacetime transport doubled."""

import sqlite3
import time

import pytest
from test_studio_permissions_http import Bridge
from test_studio_tasks import Journal

from relay_lab.studio.workers import WorkerStore


@pytest.fixture
def bridge(tmp_path):
    result = Bridge(tmp_path)
    result.save_and_release()
    result.task_journal = Journal()
    result.service.factory = result.task_journal.factory
    return result


def connection(bridge, agent="left"):
    status, _, value = bridge.request(
        "/api/workers",
        method="POST",
        actor=bridge.manager,
        value={"name": "Test runtime", "agent_id": agent, "execution_kind": "observation"},
    )
    assert status == 201, value
    return value


def task(bridge, target="left"):
    status, _, value = bridge.request(
        "/api/tasks",
        method="POST",
        actor=bridge.employee,
        value={
            "request_id": "create-http",
            "objective": "Inspect the local camera; hold if blocked.",
            "success_criteria": ["Observation recorded"],
            "target": target,
        },
    )
    assert status == 201, value
    return value


def agent_request(bridge, worker, path, value=None):
    return bridge.request(
        path,
        method="GET" if value is None else "POST",
        value=value,
        headers={"Authorization": "Bearer " + worker["token"]},
    )


def test_customer_objective_reaches_exact_agent_and_report_persists(bridge):
    worker, goal = connection(bridge), task(bridge)
    status, _, inbox = agent_request(bridge, worker, "/api/agent/tasks")
    assert status == 200 and inbox["tasks"][0]["objective"] == goal["objective"]
    status, _, goal = agent_request(
        bridge,
        worker,
        f"/api/agent/tasks/{goal['id']}/claim",
        {"request_id": "claim-http", "expected_revision": 0},
    )
    assert status == 200 and goal["assignments"]["left"]["execution_kind"] == "observation"
    status, _, result = agent_request(
        bridge,
        worker,
        f"/api/agent/tasks/{goal['id']}/report",
        {
            "request_id": "report-http",
            "expected_revision": goal["revision"],
            "status": "succeeded",
            "progress": 100,
            "summary": "Camera inspection complete",
            "evidence": ["Frame digest: abc"],
        },
    )
    assert status == 200 and result["status"] == "succeeded"
    assert result["success_basis"] == "agent_reported" and not result["verified_physical_outcome"]
    status, _, readback = bridge.request(f"/api/tasks/{goal['id']}", actor=bridge.employee)
    assert status == 200 and readback == result
    status, _, result = agent_request(bridge, worker, f"/api/agent/tasks/{goal['id']}")
    assert status == 200
    assert {r["request_id"] for r in result["request_receipts"]} == {"claim-http", "report-http"}


def test_employee_can_assign_but_cannot_create_or_revoke_runtime(bridge):
    worker = connection(bridge)
    assert bridge.request("/api/workers", actor=bridge.employee)[0] == 200
    assert (
        bridge.request(
            "/api/workers",
            method="POST",
            actor=bridge.employee,
            value={"name": "Denied", "agent_id": "left"},
        )[0]
        == 403
    )
    assert (
        bridge.request(
            f"/api/workers/{worker['worker']['id']}/revoke",
            method="POST",
            actor=bridge.employee,
            value={},
        )[0]
        == 403
    )
    task(bridge)


def test_browser_cookie_never_authenticates_agent_routes(bridge):
    assert bridge.request("/api/agent/me", actor=bridge.manager)[0] == 401
    assert (
        bridge.request("/api/agent/heartbeat", method="POST", actor=bridge.manager, value={})[0]
        == 401
    )


def test_worker_cannot_use_customer_or_administration_api(bridge):
    worker = connection(bridge)
    for path in ("/api/tasks", "/api/workers", "/api/team", "/api/bootstrap"):
        assert agent_request(bridge, worker, path)[0] == 401
    assert agent_request(bridge, worker, "/api/workspace", {})[0] == 401


def test_browser_mutations_require_csrf_but_agent_bearer_does_not(bridge):
    assert (
        bridge.request(
            "/api/tasks",
            method="POST",
            actor=bridge.employee,
            value={},
            headers={"X-Relay-Token": ""},
        )[0]
        == 403
    )
    worker = connection(bridge)
    status, _, data = agent_request(
        bridge, worker, "/api/agent/heartbeat", {"controller_label": "Controller"}
    )
    assert status == 200 and data["worker"]["status"] == "online"
    assert data["worker"]["controller_label"] == "Controller"
    assert "token" not in data["worker"]


def test_cross_origin_worker_request_is_rejected(bridge):
    worker = connection(bridge)
    assert (
        bridge.request(
            "/api/agent/me",
            headers={
                "Authorization": "Bearer " + worker["token"],
                "Origin": "https://attacker.invalid",
            },
        )[0]
        == 403
    )


def test_worker_scope_cannot_be_changed_in_payload(bridge):
    left, right, goal = connection(bridge), connection(bridge, "right"), task(bridge)
    assert agent_request(bridge, right, "/api/agent/tasks")[2]["tasks"] == []
    assert agent_request(bridge, right, f"/api/agent/tasks/{goal['id']}")[0] == 403
    assert (
        agent_request(
            bridge,
            right,
            f"/api/agent/tasks/{goal['id']}/claim",
            {
                "request_id": "forged",
                "expected_revision": 0,
                "agent_id": "left",
                "worker": left["worker"],
            },
        )[0]
        == 403
    )


def test_competing_runtime_cannot_read_claimed_assignment(bridge):
    one, two, goal = connection(bridge), connection(bridge), task(bridge)
    assert (
        agent_request(
            bridge,
            one,
            f"/api/agent/tasks/{goal['id']}/claim",
            {"request_id": "claim-one", "expected_revision": 0},
        )[0]
        == 200
    )
    assert agent_request(bridge, two, f"/api/agent/tasks/{goal['id']}")[0] == 403
    assert (
        agent_request(
            bridge,
            two,
            f"/api/agent/tasks/{goal['id']}/claim",
            {"request_id": "claim-two", "expected_revision": 1},
        )[0]
        == 409
    )


def test_stale_revision_conflicts_without_reexecuting_update(bridge):
    worker, goal = connection(bridge), task(bridge)
    value = {"request_id": "claim", "expected_revision": 0}
    assert agent_request(bridge, worker, f"/api/agent/tasks/{goal['id']}/claim", value)[0] == 200
    assert agent_request(bridge, worker, f"/api/agent/tasks/{goal['id']}/claim", value)[0] == 200
    assert bridge.task_journal.commits == 1
    assert (
        agent_request(
            bridge,
            worker,
            f"/api/agent/tasks/{goal['id']}/report",
            {"request_id": "stale", "expected_revision": 0},
        )[0]
        == 409
    )


def test_revocation_takes_effect_immediately_and_never_returns_token(bridge):
    worker = connection(bridge)
    public = bridge.request("/api/workers", actor=bridge.manager)[2]
    assert worker["token"] not in str(public)
    assert "token_hash" not in str(public)
    assert (
        bridge.request(
            f"/api/workers/{worker['worker']['id']}/revoke",
            method="POST",
            actor=bridge.manager,
            value={},
        )[0]
        == 200
    )
    assert agent_request(bridge, worker, "/api/agent/me")[0] == 401
    assert (
        bridge.request("/api/workers", actor=bridge.manager)[2]["workers"][0]["status"] == "revoked"
    )


def test_issuer_demotion_blocks_delegated_runtime(bridge):
    worker = connection(bridge)
    # Another manager performs the demotion; stale role dictionaries cannot preserve authority.
    invitation = bridge.access.invite(bridge.manager["user"], "manager2", "Manager2", "manager")
    other = bridge.access.accept_invite(invitation["code"], "another sufficiently long password")
    bridge.access.update_user(other["user"], bridge.manager["user"]["id"], "employee", True)
    assert agent_request(bridge, worker, "/api/agent/me")[0] == 403
    assert (
        bridge.request("/api/workers", actor=other)[2]["workers"][0]["status"] == "release_blocked"
    )


def test_retiring_pinned_release_blocks_claim_and_heartbeat(bridge):
    worker, goal = connection(bridge), task(bridge)
    model = worker["worker"]
    bridge.registry.transition(
        bridge.manager["user"], model["model_id"], "retired", model["release_revision"]
    )
    assert agent_request(bridge, worker, "/api/agent/heartbeat", {})[0] == 403
    assert (
        agent_request(
            bridge,
            worker,
            f"/api/agent/tasks/{goal['id']}/claim",
            {"request_id": "claim", "expected_revision": 0},
        )[0]
        == 403
    )
    assert bridge.request(f"/api/tasks/{goal['id']}", actor=bridge.employee)[0] == 200


def test_expiration_and_private_hashed_credential_storage(bridge):
    worker = connection(bridge)
    store = WorkerStore(bridge.service.storage, bridge.access, bridge.registry, bridge.service)
    assert store.path.stat().st_mode & 0o077 == 0
    assert worker["token"].encode() not in store.path.read_bytes()
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE runtime_worker SET expires_at=?", (time.time() - 1,))
    assert agent_request(bridge, worker, "/api/agent/me")[0] == 401
    assert (
        bridge.request("/api/workers", actor=bridge.employee)[2]["workers"][0]["status"]
        == "expired"
    )


def test_unreleased_or_hardware_runtime_is_not_registered(bridge):
    assert (
        bridge.request(
            "/api/workers",
            method="POST",
            actor=bridge.manager,
            value={"name": "Robot", "agent_id": "left", "execution_kind": "hardware"},
        )[0]
        == 400
    )
    model = bridge.registry.catalog(bridge.manager["user"])["models"][0]
    bridge.registry.transition(bridge.manager["user"], model["id"], "retired", model["revision"])
    assert (
        bridge.request(
            "/api/workers",
            method="POST",
            actor=bridge.manager,
            value={"name": "Robot", "agent_id": "left"},
        )[0]
        == 403
    )
