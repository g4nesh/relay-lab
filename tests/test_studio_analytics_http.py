"""Analytics routes share real dashboard identity/CSRF and expose aggregates only."""

import json

import pytest
from test_studio_permissions_http import Bridge

from relay_lab.studio.analytics import EFFORT_DEFAULTS, AnalyticsService


@pytest.fixture
def bridge(tmp_path):
    return Bridge(tmp_path)


@pytest.mark.parametrize(
    "path,method,value",
    [
        ("/api/analytics", "GET", None),
        ("/api/analytics/estimate", "POST", EFFORT_DEFAULTS),
    ],
)
def test_anonymous_analytics_is_rejected_without_computation(bridge, path, method, value):
    assert bridge.request(path, method=method, value=value)[0] == 401
    assert bridge.journal.runs == {} and bridge.llm.calls == 0


@pytest.mark.parametrize("role", ["manager", "employee"])
def test_both_roles_can_read_aggregates_without_private_registry_fields(bridge, role):
    actor = getattr(bridge, role)
    workspace = bridge.save_and_release()
    status, _, result = bridge.request("/api/analytics", actor=actor)
    assert status == 200
    assert result["schema"] == "relay.studio.analytics.v1"
    assert result["governance"]["models"]["released"] == 2
    serialized = json.dumps(result)
    for private in (
        bridge.manager["token"],
        bridge.manager["csrf"],
        bridge.manager["user"]["id"],
        workspace["agents"][0]["source"]["ref"],
        "Reviewed exact configuration",
    ):
        assert private not in serialized
    assert bridge.llm.calls == 0 and bridge.journal.commits == 0


@pytest.mark.parametrize("role", ["manager", "employee"])
def test_effort_post_is_an_explicit_scenario_and_does_not_mutate(bridge, role):
    status, _, result = bridge.request(
        "/api/analytics/estimate", method="POST", value=EFFORT_DEFAULTS, actor=getattr(bridge, role)
    )
    assert status == 200
    assert result["measured"] is False
    assert result["manual_hours"] == 102 and result["relay_hours"] == 20
    assert bridge.registry.catalog(bridge.manager["user"])["models"] == []
    assert bridge.journal.runs == {} and bridge.llm.calls == 0


@pytest.mark.parametrize("csrf", ["", "forged"])
def test_effort_post_still_enforces_csrf(bridge, csrf):
    assert (
        bridge.request(
            "/api/analytics/estimate",
            method="POST",
            value=EFFORT_DEFAULTS,
            actor=bridge.employee,
            headers={"X-Relay-Token": csrf},
        )[0]
        == 403
    )


def test_revoked_account_cannot_read_cached_analytics(bridge, monkeypatch):
    analytics = AnalyticsService(bridge.service.root, bridge.service, bridge.registry, bridge.access)
    monkeypatch.setattr("relay_lab.studio.server.AnalyticsService", lambda *args: analytics)
    assert bridge.request("/api/analytics", actor=bridge.employee)[0] == 200
    assert analytics._cache is not None
    bridge.access.update_user(
        bridge.manager["user"], bridge.employee["user"]["id"], "employee", False
    )
    assert bridge.request("/api/analytics", actor=bridge.employee)[0] == 401
    assert (
        bridge.request(
            "/api/analytics/estimate", method="POST", value=EFFORT_DEFAULTS, actor=bridge.employee
        )[0]
        == 401
    )


@pytest.mark.parametrize(
    "value",
    [
        {},
        {**EFFORT_DEFAULTS, "robots": -1},
        {**EFFORT_DEFAULTS, "tasks": 1.5},
        {**EFFORT_DEFAULTS, "hourly_rate": True},
    ],
)
def test_invalid_scenarios_are_400_not_partial_estimates(bridge, value):
    assert (
        bridge.request(
            "/api/analytics/estimate", method="POST", value=value, actor=bridge.employee
        )[0]
        == 400
    )


def test_live_exchange_and_duplicate_have_one_aggregate_sample(bridge):
    workspace = bridge.save_and_release()
    status, _, session = bridge.request(
        "/api/sessions", method="POST", value=workspace, actor=bridge.employee
    )
    assert status == 201
    request = {
        "request_id": "one-observation",
        "source": "manual",
        "reports": {"left": "private-observation-alpha", "right": "private-observation-beta"},
    }
    route = f"/api/sessions/{session['session_id']}/exchange"
    assert bridge.request(route, method="POST", value=request, actor=bridge.employee)[0] == 200
    assert bridge.request(route, method="POST", value=request, actor=bridge.employee)[0] == 200
    status, _, result = bridge.request("/api/analytics", actor=bridge.employee)
    assert status == 200
    operations = result["operations"]
    assert operations["sessions"] == 1 and operations["exchanges"] == 1
    assert operations["coverage"]["timed_exchanges"] == 1
    assert operations["modes"][0]["elapsed_ms"]["n"] == 1
    assert operations["modes"][0]["verified_readbacks"] == 1
    assert bridge.journal.commits == 1 and bridge.llm.calls == 2
    serialized = json.dumps(result)
    assert "private-observation" not in serialized
    assert session["session_id"] not in serialized
