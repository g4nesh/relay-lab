"""Exact-spec releases are durable and enforced for managers and employees."""

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

from relay_lab.studio.governance import ModelRegistry, RegistryConflict
from relay_lab.studio.service import example_workspace

MANAGER = {"id": "owner", "username": "alex", "name": "Alex", "role": "manager", "active": True}
EMPLOYEE = {"id": "operator", "username": "sam", "name": "Sam", "role": "employee", "active": True}


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path)


def release(registry, workspace=None):
    workspace = workspace or example_workspace()
    records = registry.sync_workspace(MANAGER, workspace)["models"]
    for record in records:
        registry.transition(
            MANAGER, record["id"], "released", record["revision"], "Reviewed exact rig"
        )
    return registry.authorize_workspace(EMPLOYEE, workspace)


def test_old_workspace_does_not_implicitly_release(registry):
    workspace = example_workspace()
    assert registry.catalog(MANAGER) == {"models": []}
    assert registry.has_release(workspace) is False
    for actor in (MANAGER, EMPLOYEE):
        with pytest.raises(PermissionError, match="no current release"):
            registry.authorize_workspace(actor, workspace)
    records = registry.sync_workspace(MANAGER, workspace)["models"]
    assert len(records) == 2 and all(r["status"] == "draft" for r in records)
    with pytest.raises(PermissionError):
        registry.authorize_workspace(MANAGER, workspace)
    assert registry.sync_workspace(MANAGER, workspace)["models"] == records
    assert len(registry.audit_log(MANAGER)["events"]) == 2


def test_employee_can_inspect_but_cannot_mutate_or_read_audit(registry):
    records = registry.sync_workspace(MANAGER, example_workspace())["models"]
    assert len(registry.catalog(EMPLOYEE)["models"]) == 2
    with pytest.raises(PermissionError, match="Only managers"):
        registry.sync_workspace(EMPLOYEE, example_workspace())
    for status in ("released", "retired"):
        with pytest.raises(PermissionError, match="Only managers"):
            registry.transition(EMPLOYEE, records[0]["id"], status, 1)
    with pytest.raises(PermissionError):
        registry.audit_log(EMPLOYEE)


@pytest.mark.parametrize(
    "actor", [None, {}, {**MANAGER, "active": False}, {**MANAGER, "role": "admin"}]
)
def test_inactive_or_missing_identity_fails_closed(registry, actor):
    with pytest.raises(PermissionError):
        registry.catalog(actor)
    with pytest.raises(PermissionError):
        registry.authorize_workspace(actor, example_workspace())


def test_released_exact_spec_and_audit_survive_restart(tmp_path):
    first = ModelRegistry(tmp_path)
    snapshot = release(first)
    second = ModelRegistry(tmp_path)
    assert first.registry_id == second.registry_id == snapshot["registry_id"]
    assert second.authorize_snapshot(EMPLOYEE, snapshot) == snapshot
    assert second.has_release(example_workspace())
    events = second.audit_log(MANAGER)["events"]
    assert [e["action"] for e in events] == ["release", "release", "register", "register"]
    assert events[0]["actor_id"] == MANAGER["id"]
    assert events[0]["spec"] == snapshot["models"][1]["spec"]
    assert events[0]["previous_status"] == "draft"


def test_identical_specs_and_revisions_in_another_registry_cannot_authorize_session(tmp_path):
    first, other = ModelRegistry(tmp_path / "first"), ModelRegistry(tmp_path / "other")
    snapshot, other_snapshot = release(first), release(other)
    assert snapshot["models"] == other_snapshot["models"]
    assert snapshot["registry_id"] != other_snapshot["registry_id"]
    with pytest.raises(PermissionError):
        other.authorize_snapshot(EMPLOYEE, snapshot)
    with pytest.raises(PermissionError):
        first.authorize_snapshot(MANAGER, other_snapshot)
    assert other.authorize_snapshot(EMPLOYEE, other_snapshot) == other_snapshot


def test_recreated_registry_never_revives_old_session_even_in_old_process(registry):
    snapshot = release(registry)
    registry.path.unlink()
    replacement = ModelRegistry(registry.path.parent)
    current = release(replacement)
    assert current["models"] == snapshot["models"]
    assert current["registry_id"] != snapshot["registry_id"]
    with pytest.raises(PermissionError):
        replacement.authorize_snapshot(EMPLOYEE, snapshot)
    with pytest.raises(PermissionError, match="registry was replaced"):
        registry.authorize_snapshot(EMPLOYEE, snapshot)
    with pytest.raises(PermissionError, match="registry was replaced"):
        registry.authorize_workspace(MANAGER, example_workspace())
    assert not registry.has_release(example_workspace())
    assert replacement.authorize_snapshot(EMPLOYEE, current) == current


def test_legacy_snapshot_without_registry_identity_fails_closed(registry):
    snapshot = release(registry)
    snapshot.pop("registry_id")
    with pytest.raises(PermissionError, match="no valid model release snapshot"):
        registry.authorize_snapshot(EMPLOYEE, snapshot)


@pytest.mark.parametrize(
    "field", ["revision", "source", "rig", "action_contract", "name", "family"]
)
def test_source_and_rig_changes_require_new_release(registry, field):
    workspace = example_workspace()
    release(registry, workspace)
    altered = copy.deepcopy(workspace)
    agent = altered["agents"][0]
    if field == "revision":
        agent["source"]["revision"] = "new-weights"
    elif field == "source":
        agent["source"]["ref"] = "different/model"
    elif field == "rig":
        agent["rig"] = "https://new-robot.example/actuate"
    elif field == "action_contract":
        agent["action_contract"] = "Joint torques"
    elif field == "name":
        agent["name"] = "Different robot"
    else:
        agent["family"] = "custom"
    with pytest.raises(PermissionError):
        registry.authorize_workspace(EMPLOYEE, altered)
    new_records = registry.sync_workspace(MANAGER, altered)["models"]
    assert new_records[0]["status"] == "draft"
    assert new_records[1]["status"] == "released"
    assert len(registry.catalog(MANAGER)["models"]) == 3
    assert registry.has_release(workspace) and not registry.has_release(altered)


def test_normalization_has_stable_identity_and_objective_is_not_model_spec(registry):
    workspace = example_workspace()
    snapshot = release(registry, workspace)
    workspace["name"] = "Another team"
    workspace["objective"] = "Pick up a component"
    workspace["mode"] = "mediated"
    workspace["agents"][0]["name"] = "  Left arm  "
    assert registry.authorize_workspace(EMPLOYEE, workspace) == snapshot
    assert len(registry.sync_workspace(MANAGER, workspace)["models"]) == 2


def test_retirement_and_rerelease_invalidate_old_sessions(registry):
    snapshot = release(registry)
    model = snapshot["models"][0]
    retired = registry.transition(
        MANAGER, model["id"], "retired", model["revision"], "Safety review"
    )
    assert retired["revision"] == model["revision"] + 1
    for actor in (EMPLOYEE, MANAGER):
        with pytest.raises(PermissionError):
            registry.authorize_snapshot(actor, snapshot)
    registry.transition(MANAGER, model["id"], "released", retired["revision"])
    with pytest.raises(PermissionError):
        registry.authorize_snapshot(MANAGER, snapshot)
    current = registry.authorize_workspace(EMPLOYEE, example_workspace())
    assert current["models"][0]["revision"] == model["revision"] + 2
    assert registry.authorize_snapshot(EMPLOYEE, current) == current


def test_huggingface_release_requires_source_revision(registry):
    workspace = example_workspace()
    workspace["agents"][0]["source"]["revision"] = ""
    model = registry.sync_workspace(MANAGER, workspace)["models"][0]
    with pytest.raises(ValueError, match="Pin a Hugging Face"):
        registry.transition(MANAGER, model["id"], "released", 1)
    assert len(registry.audit_log(MANAGER)["events"]) == 2
    assert (
        next(r for r in registry.catalog(MANAGER)["models"] if r["id"] == model["id"])["revision"]
        == 1
    )


def test_two_concurrent_approvals_cannot_use_same_revision(registry):
    model = registry.sync_workspace(MANAGER, example_workspace())["models"][0]
    # Separate instances do not share the in-process lock; SQLite must still
    # serialize their revision checks and audit inserts in one transaction.
    instances = [registry, ModelRegistry(registry.path.parent)]

    def approve(index):
        try:
            instances[index].transition(MANAGER, model["id"], "released", 1)
            return "released"
        except RegistryConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(approve, range(2)))
    assert sorted(results) == ["conflict", "released"]
    assert len(registry.audit_log(MANAGER)["events"]) == 3


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "id", "revision", "spec"])
def test_missing_or_tampered_snapshot_cannot_resume(registry, tamper):
    snapshot = release(registry)
    if tamper == "missing":
        snapshot = None
    elif tamper == "duplicate":
        snapshot["models"][1] = copy.deepcopy(snapshot["models"][0])
    elif tamper == "id":
        snapshot["models"][0]["id"] = "mdl_not-released"
    elif tamper == "revision":
        snapshot["models"][0]["revision"] = True
    else:
        snapshot["models"][0]["spec"]["source"]["ref"] = "attacker/model"
    with pytest.raises(PermissionError):
        registry.authorize_snapshot(EMPLOYEE, snapshot)


def test_employee_request_is_durable_and_only_visible_to_requester_and_managers(registry):
    other = {**EMPLOYEE, "id": "another-employee", "username": "lee"}
    request = registry.submit_request(EMPLOYEE, "add", "Please add a wrist-camera policy")
    assert request["status"] == "pending" and request["model_id"] is None
    assert request["id"].startswith("req_")
    assert request["created_by"] == EMPLOYEE["id"] and request["resolved_by"] is None
    assert registry.requests(EMPLOYEE)["requests"] == [request]
    assert registry.requests(other)["requests"] == []
    assert registry.requests(MANAGER)["requests"] == [request]
    restarted = ModelRegistry(registry.path.parent)
    assert restarted.requests(EMPLOYEE)["requests"] == [request]
    assert restarted.catalog(MANAGER)["models"] == []


def test_request_resolution_never_implicitly_releases_or_retires_a_model(registry):
    model = registry.sync_workspace(MANAGER, example_workspace())["models"][0]
    request = registry.submit_request(EMPLOYEE, "release", "Please review this model", model["id"])
    with pytest.raises(PermissionError):
        registry.resolve_request(EMPLOYEE, request["id"], "resolved")
    resolved = registry.resolve_request(MANAGER, request["id"], "resolved", "Review completed")
    assert resolved["resolved_by"] == MANAGER["id"] and resolved["resolved_at"]
    assert resolved["reason"] == "Review completed"
    assert all(row["status"] == "draft" for row in registry.catalog(MANAGER)["models"])
    with pytest.raises(RegistryConflict):
        registry.resolve_request(MANAGER, request["id"], "declined")
    assert registry.requests(EMPLOYEE)["requests"] == [resolved]
    release(registry)
    retire = registry.submit_request(EMPLOYEE, "retire", "Retire after validation", model["id"])
    registry.resolve_request(MANAGER, retire["id"], "resolved")
    assert registry.has_release(example_workspace())
    actions = [event["action"] for event in registry.audit_log(MANAGER)["events"]]
    assert actions.count("request.submitted") == 2 and actions.count("request.resolved") == 2


def test_request_can_be_declined_with_recorded_manager_reason(registry):
    request = registry.submit_request(EMPLOYEE, "edit", "Change the action contract")
    rejected = registry.resolve_request(
        MANAGER, request["id"], "declined", "Contract is incompatible"
    )
    assert rejected["status"] == "declined" and rejected["reason"] == "Contract is incompatible"
    latest = registry.audit_log(MANAGER)["events"][0]
    assert latest["action"] == "request.declined" and latest["actor_id"] == MANAGER["id"]
    assert latest["target"] == request["id"]


@pytest.mark.parametrize(
    "kind,message,model_id",
    [
        ("deploy", "Please deploy this", None),
        ("add", "", None),
        ("add", "  ", None),
        ("add", "x" * 2001, None),
        ("edit", "Hidden\x00control", None),
        ("release", "Review", "mdl_missing"),
        ("release", "Review", 123),
    ],
)
def test_invalid_requests_do_not_create_queue_entries(registry, kind, message, model_id):
    with pytest.raises(ValueError):
        registry.submit_request(EMPLOYEE, kind, message, model_id)
    assert registry.requests(MANAGER)["requests"] == []
    assert registry.audit_log(MANAGER)["events"] == []


def test_concurrent_managers_can_resolve_request_only_once(registry):
    request = registry.submit_request(EMPLOYEE, "add", "Please add another model")
    instances = [registry, ModelRegistry(registry.path.parent)]

    def resolve(index):
        try:
            instances[index].resolve_request(MANAGER, request["id"], "resolved")
            return "resolved"
        except RegistryConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(resolve, range(2))) == ["conflict", "resolved"]
    assert len(registry.audit_log(MANAGER)["events"]) == 2
