"""Local account switching preserves selected-role checks and hosted authentication."""

import json
from http.client import HTTPResponse
from types import SimpleNamespace

import pytest
from test_studio_permissions_http import Bridge, MemorySocket

from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.profiles import ProfileStore
from relay_lab.studio.server import StudioHandler


def test_fresh_profiles_open_without_password_and_persist(tmp_path):
    access = ProfileStore(tmp_path)
    profiles = access.profiles()["profiles"]
    assert [p["role"] for p in profiles] == ["manager", "employee"]
    assert not access.status()["setup_required"]
    assert not access.setup_path.exists()
    assert ProfileStore(tmp_path).profiles()["profiles"] == profiles
    credentials = access.select_profile(profiles[0]["id"], access.profiles()["nonce"])
    assert "models.release" in credentials["permissions"]
    assert access.authenticate(credentials["token"])["user"]["id"] == profiles[0]["id"]
    with pytest.raises(AuthenticationError):
        access.login("manager", "local-profile")


def test_switch_revokes_previous_browser_session_and_changes_permissions(tmp_path):
    access = ProfileStore(tmp_path)
    owner, employee = access.profiles()["profiles"]
    nonce = access.profiles()["nonce"]
    first = access.select_profile(owner["id"], nonce)
    second = access.select_profile(employee["id"], nonce, first["token"])
    assert second["permissions"] == ["sessions.run"]
    with pytest.raises(AuthenticationError):
        access.authenticate(first["token"])
    assert access.authenticate(second["token"])["user"]["id"] == employee["id"]


def test_bad_nonce_inactive_profile_and_client_role_cannot_grant_access(tmp_path):
    access = ProfileStore(tmp_path)
    owner, employee = access.profiles()["profiles"]
    with pytest.raises(PermissionError):
        access.select_profile(owner["id"], "incorrect")
    access.update_user(owner, employee["id"], "employee", False)
    assert len(access.profiles()["profiles"]) == 1
    with pytest.raises(AuthenticationError):
        access.select_profile(employee["id"], access.profiles()["nonce"])
    with pytest.raises(ValueError):
        access.select_profile({"id": owner["id"], "role": "manager"}, access.profiles()["nonce"])


def test_existing_accounts_releases_and_audit_identity_are_preserved(tmp_path):
    bridge = Bridge(tmp_path)
    bridge.save_and_release()
    before = bridge.registry.catalog(bridge.manager["user"])["models"]
    access = ProfileStore(bridge.service.storage)
    assert {p["id"] for p in access.profiles()["profiles"]} == {
        bridge.manager["user"]["id"],
        bridge.employee["user"]["id"],
    }
    chosen = access.select_profile(bridge.manager["user"]["id"], access.profiles()["nonce"])
    assert bridge.registry.catalog(chosen["user"])["models"] == before
    assert access.audit_log(chosen["user"])["events"][0]["action"] == "account.selected"


def test_http_profile_login_and_employee_authorization(tmp_path):
    bridge = Bridge(tmp_path)
    bridge.access = ProfileStore(bridge.service.storage)
    assert bridge.request("/api/auth/status")[2].get("account_switching", False) is True
    status, _, catalog = bridge.request("/api/auth/profiles")
    assert status == 200 and "password" not in str(catalog)
    employee = next(p for p in catalog["profiles"] if p["role"] == "employee")
    status, headers, data = bridge.request(
        "/api/auth/select-profile",
        method="POST",
        value={"user_id": employee["id"], "nonce": catalog["nonce"], "role": "manager"},
    )
    assert status == 200 and data["user"]["role"] == "employee"
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    status, _, boot = bridge.request("/api/bootstrap", headers={"Cookie": cookie})
    assert status == 200 and boot["permissions"] == ["sessions.run"]
    assert (
        bridge.request(
            "/api/workspace",
            method="POST",
            value={},
            headers={"Cookie": cookie, "X-Relay-Token": data["token"]},
        )[0]
        == 403
    )
    assert (
        bridge.request(
            "/api/auth/select-profile",
            method="POST",
            value={"user_id": employee["id"], "nonce": catalog["nonce"]},
            headers={"Origin": "https://wrong.invalid"},
        )[0]
        == 403
    )
    assert (
        bridge.request(
            "/api/auth/select-profile", method="POST", value={"user_id": employee["id"]}
        )[0]
        == 403
    )


def test_password_mode_does_not_expose_profile_routes(tmp_path):
    bridge = Bridge(tmp_path)
    assert not bridge.request("/api/auth/status")[2].get("account_switching", False)
    assert bridge.request("/api/auth/profiles")[0] == 404
    assert bridge.request("/api/auth/select-profile", method="POST", value={})[0] == 404


def test_hosted_handler_refuses_even_an_injected_profile_store(tmp_path):
    bridge = Bridge(tmp_path)
    access = ProfileStore(bridge.service.storage)
    for path in ("/api/auth/profiles", "/api/auth/status"):
        request = f"GET {path} HTTP/1.1\r\nHost: robots.example.com\r\nConnection: close\r\n\r\n"
        sock = MemorySocket(request.encode())
        StudioHandler(
            sock,
            ("127.0.0.1", 1),
            SimpleNamespace(server_port=8770),
            service=bridge.service,
            access=access,
            registry=bridge.registry,
            deployment=DeploymentConfig(public_origin="https://robots.example.com"),
        )
        response = HTTPResponse(MemorySocket(bytes(sock.sent)))
        response.begin()
        data = json.loads(response.read())
        if path.endswith("profiles"):
            assert response.status == 404
        else:
            assert response.status == 200 and not data.get("account_switching", False)
