import base64
import json
import zlib

import pytest

from relay_lab.spacetime import (
    SpacetimeConfig,
    SpacetimeConflict,
    SpacetimeError,
    SpacetimeUnavailable,
)
from relay_lab.studio.access import AccessStore, AuthenticationError
from relay_lab.studio.durable import DurableState, _FencedClient


class Journal:
    def __init__(self):
        self.state = None
        self.fail_commit = False
        self.lost_ack = False
        self.acquires = 0

    def _sql(self, query):
        return [dict(self.state)] if self.state else []

    def _request(self, endpoint, body, **kwargs):
        name = endpoint.split("/call/")[1].split("?")[0]
        args = json.loads(body)
        row = self.state
        if name == "studio_acquire":
            holder = args[0]
            if row and row["holder"] and row["holder"] != holder:
                raise SpacetimeConflict("busy")
            self.state = {
                **(row or {"revision": 0, "payload": "", "last_operation": ""}),
                "holder": holder,
            }
            self.acquires += 1
        elif name == "studio_release":
            if row and row["holder"] == args[0]:
                row["holder"] = ""
        elif name == "studio_finish_read":
            if row["last_operation"] == args[0]:
                return b""
            if row["holder"] != args[0] or row["revision"] != args[1]:
                raise SpacetimeConflict("stale")
            row.update(holder="", last_operation=args[0])
        elif name == "studio_commit_state":
            holder, revision, payload, operation = args
            if self.fail_commit:
                raise SpacetimeUnavailable("offline")
            if row["last_operation"] == operation:
                return b""
            if row["holder"] != holder or row["revision"] != revision:
                raise SpacetimeConflict("stale")
            row.update(
                revision=revision + 1,
                payload=payload,
                last_operation=operation,
                holder="",
            )
            if self.lost_ack:
                self.lost_ack = False
                raise SpacetimeUnavailable("lost acknowledgement")
        return b""


@pytest.fixture
def state(tmp_path):
    return DurableState(tmp_path, SpacetimeConfig(token="test"), client=Journal())


def test_account_session_and_setup_consumption_survive_empty_disk_restart(
    state, tmp_path
):
    with state.request():
        access = AccessStore(state.storage)
        code = access.setup_path.read_text().strip()
        credentials = access.setup(code, "manager", "Manager", "strong-test-password")
    fresh = DurableState(tmp_path / "fresh", state.config, client=state.client)
    with fresh.request():
        access = AccessStore(fresh.storage)
        assert access.status() == {"setup_required": False}
        assert not access.setup_path.exists()
        assert access.authenticate(credentials["token"])["user"]["role"] == "manager"
        assert (
            access.login("manager", "strong-test-password")["user"]["id"]
            == credentials["user"]["id"]
        )


def test_failed_commit_does_not_become_authoritative(state):
    with state.request():
        (state.storage / "workspace.json").write_text('{"old":true}')
    state.client.fail_commit = True
    with pytest.raises(SpacetimeUnavailable), state.request():
        (state.storage / "workspace.json").write_text('{"new":true}')
    state.client.fail_commit = False
    with state.request():
        assert json.loads((state.storage / "workspace.json").read_text()) == {
            "old": True
        }


def test_lost_commit_ack_is_retried_identically_without_duplicate_revision(state):
    state.client.lost_ack = True
    with state.request():
        (state.storage / "workspace.json").write_text("{}")
    assert state.client.state["revision"] == 1
    assert state.client.state["holder"] == ""


def test_unchanged_read_does_not_rewrite_checkpoint(state):
    with state.request():
        (state.storage / "workspace.json").write_text("{}")
    revision = state.revision
    with state.request():
        pass
    assert state.client.state["revision"] == revision


def test_superseded_request_cannot_commit_or_release_new_holder(state):
    with pytest.raises(SpacetimeConflict), state.request():
        (state.storage / "workspace.json").write_text("{}")
        state.client.state["holder"] = "another-instance"
    assert state.client.state["payload"] == ""
    assert state.client.state["holder"] == "another-instance"


def test_revocation_survives_restart(state, tmp_path):
    with state.request():
        access = AccessStore(state.storage)
        credentials = access.setup(
            access.setup_path.read_text().strip(),
            "manager",
            "Manager",
            "strong-test-password",
        )
    with state.request():
        access.logout(credentials["token"])
    fresh = DurableState(tmp_path / "fresh", state.config, client=state.client)
    with fresh.request(), pytest.raises(AuthenticationError):
        AccessStore(fresh.storage).authenticate(credentials["token"])


@pytest.mark.parametrize(
    "payload",
    [
        "not_base64",
        base64.b64encode(zlib.compress(b'{"version":2,"files":{}}')).decode(),
        base64.b64encode(
            zlib.compress(b'{"version":1,"files":{"../escape":"YQ=="}}')
        ).decode(),
        base64.b64encode(
            zlib.compress(b'{"version":1,"files":{"access.sqlite3":"YQ=="}}')
        ).decode(),
    ],
)
def test_invalid_remote_state_fails_closed_without_erasing_local_evidence(
    state, payload
):
    target = state.storage / "workspace.json"
    target.write_text("original")
    with pytest.raises(SpacetimeError):
        state._restore(payload)
    assert target.read_text() == "original"


def test_snapshot_excludes_owner_token_and_model_files(state):
    (state.storage / "owner-token").write_text("never-upload")
    (state.storage / "weights.bin").write_text("never-upload")
    value = json.loads(zlib.decompress(base64.b64decode(state._capture())))
    assert value["files"] == {}


def test_task_writes_require_request_lease(state):
    client = _FencedClient(state.config, state)
    with pytest.raises(SpacetimeConflict):
        client._request(
            "/call/create_factory_run?confirmed=true",
            b"[]",
            content_type="application/json",
        )


@pytest.mark.parametrize("name", [
    "studio_heartbeat", "studio_revoke_worker", "studio_sync_task",
    "studio_record_metric", "studio_configure_job", "studio_run_job",
])
def test_coordination_writes_require_current_request_lease(state, name):
    client = _FencedClient(state.config, state)
    with pytest.raises(SpacetimeConflict):
        client._request(f"/call/{name}?confirmed=true", b"[]", content_type="application/json")
