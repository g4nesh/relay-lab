"""A real task state machine drives the outbound SDK; no robot action is performed."""

from urllib.error import HTTPError

import pytest
from test_studio_tasks import EMPLOYEE, MANAGER, Journal, create, update

from relay_lab.spacetime import SpacetimeError
from relay_lab.studio import worker as sdk
from relay_lab.studio.governance import ModelRegistry
from relay_lab.studio.service import StudioService, example_workspace
from relay_lab.studio.tasks import TaskConflict, TaskService


class Client:
    def __init__(self, service, identity):
        self.service, self.identity = service, identity
        self.after_report = None
        self.before_report = None
        self.sent_reports = []

    def request(self, path, value=None):
        if path == "/api/agent/me":
            return {"worker": self.identity}
        if path == "/api/agent/heartbeat":
            return {"worker": self.identity}
        if path == "/api/agent/tasks":
            return self.service.inbox(self.identity)
        parts = path.split("/")
        task_id = parts[4]
        if len(parts) == 5:
            return self.service.worker_get(self.identity, task_id)
        try:
            if parts[-1] == "claim":
                return self.service.claim(self.identity, task_id, value)
            assert parts[-1] == "report"
            self.sent_reports.append(dict(value))
            if self.before_report:
                callback, self.before_report = self.before_report, None
                callback()
            result = self.service.report(self.identity, task_id, value)
            if self.after_report:
                callback, self.after_report = self.after_report, None
                callback()
            return result
        except TaskConflict:
            raise sdk.AgentHTTPError(409) from None
        except PermissionError:
            raise sdk.AgentHTTPError(403) from None
        except SpacetimeError:
            raise sdk.AgentHTTPError(503) from None


@pytest.fixture
def system(tmp_path):
    journal = Journal()
    studio = StudioService(tmp_path, experiments=False, coordinator_factory=journal.factory)
    studio.save(example_workspace())
    registry = ModelRegistry(studio.storage)
    for model in registry.sync_workspace(MANAGER, studio.workspace())["models"]:
        registry.transition(MANAGER, model["id"], "released", model["revision"])
    service = TaskService(studio, registry)
    release = registry.authorize_workspace(EMPLOYEE, studio.workspace())
    workers = {
        m["spec"]["id"]: {
            "id": "worker-" + ("1" if m["spec"]["id"] == "left" else "2") * 16,
            "agent_id": m["spec"]["id"],
            "model_id": m["id"],
            "release_revision": m["revision"],
            "registry_id": registry.registry_id,
            "execution_kind": "reference",
        }
        for m in release["models"]
    }
    yield service, journal, workers, tmp_path / "agent-state"
    studio.close()


def outcome(status="succeeded", progress=100, summary="Software test complete"):
    return {
        "status": status,
        "progress": progress,
        "summary": summary,
        "evidence": ["test://result"],
    }


def test_handler_receives_exact_objective_and_real_peer_feedback(system):
    service, _, workers, state_dir = system
    task = create(service)
    task = service.claim(workers["right"], task["id"], update(task, "peer-claim"))
    task = service.report(
        workers["right"],
        task["id"],
        update(task, "peer-obstacle", **outcome("blocked", 10, "Obstacle in right view")),
    )
    contexts = []

    def handler(context):
        contexts.append(context)
        assert context["should_stop"]() is False
        return outcome()

    runtime = sdk.AgentRuntime(Client(service, workers["left"]), handler, state_dir)
    try:
        runtime.run_once()
        assert len(contexts) == 1
        context = contexts[0]
        assert context["task"]["objective"] == task["objective"]
        assert context["task"]["success_criteria"] == task["success_criteria"]
        assert context["task"]["assignments"]["right"]["summary"] == "Obstacle in right view"
        assert context["step"] == 0
        assert runtime._load(task["id"])["phase"] == "done"
    finally:
        runtime.close()


def test_lost_report_http_ack_does_not_repeat_completed_handler_after_restart(system):
    service, _, workers, state_dir = system
    task = create(service, "left")
    calls = []
    client = Client(service, workers["left"])
    client.after_report = lambda: (_ for _ in ()).throw(sdk.AgentHTTPError(503))
    runtime = sdk.AgentRuntime(
        client, lambda context: calls.append(context["step"]) or outcome(), state_dir
    )
    try:
        with pytest.raises(sdk.AgentHTTPError):
            runtime.run_once()
        assert runtime._load(task["id"])["phase"] == "pending"
        assert service.get(EMPLOYEE, task["id"])["status"] == "succeeded"
    finally:
        runtime.close()
    restarted = sdk.AgentRuntime(
        client, lambda context: calls.append(context["step"]) or outcome(), state_dir
    )
    try:
        restarted.run_once()
        assert calls == [0]
        assert restarted._load(task["id"])["phase"] == "done"
    finally:
        restarted.close()


def test_peer_revision_conflict_rebases_cached_outcome_without_running_handler_twice(system):
    service, _, workers, state_dir = system
    task = create(service)
    task = service.claim(workers["right"], task["id"], update(task, "right-claim"))
    calls = []
    client = Client(service, workers["left"])

    def advance_peer():
        current = service.get(EMPLOYEE, task["id"])
        service.report(
            workers["right"],
            task["id"],
            update(
                current,
                "peer-advanced",
                **outcome("running", 35, "Peer changed while left was inferring"),
            ),
        )

    client.before_report = advance_peer
    runtime = sdk.AgentRuntime(
        client, lambda context: calls.append(context["step"]) or outcome(), state_dir
    )
    try:
        runtime.run_once()
        assert runtime._load(task["id"])["phase"] == "pending"
        runtime.run_once()
        assert calls == [0]
        assert len(client.sent_reports) == 2
        first, second = client.sent_reports
        assert first["summary"] == second["summary"] and first["evidence"] == second["evidence"]
        assert first["request_id"] != second["request_id"]
        assert second["expected_revision"] == first["expected_revision"] + 1
        assert service.get(EMPLOYEE, task["id"])["assignments"]["left"]["status"] == "succeeded"
    finally:
        runtime.close()


def test_unconfirmed_prepare_retries_exact_payload_and_never_reexecutes_handler(system):
    service, journal, workers, state_dir = system
    task = create(service, "left")
    calls = []

    def handler(context):
        calls.append(context["step"])
        journal.fail_after_prepare = True
        return outcome()

    runtime = sdk.AgentRuntime(Client(service, workers["left"]), handler, state_dir)
    try:
        with pytest.raises(sdk.AgentHTTPError):
            runtime.run_once()
        pending = runtime._load(task["id"])["payload"]
        runtime.run_once()
        assert calls == [0]
        receipt = service.worker_get(workers["left"], task["id"])["request_receipts"][-1]
        assert receipt["request_id"] == pending["request_id"]
    finally:
        runtime.close()


@pytest.mark.parametrize("phase", ["new", "executing"])
def test_missing_or_interrupted_journal_holds_even_after_peer_changes(system, phase):
    service, _, workers, state_dir = system
    task = create(service)
    task = service.claim(workers["left"], task["id"], update(task, "old-claim"))
    calls = []
    runtime = sdk.AgentRuntime(
        Client(service, workers["left"]), lambda context: calls.append(0) or outcome(), state_dir
    )
    try:
        if phase == "executing":
            runtime._save(task["id"], {"phase": "executing", "step": 0})
        runtime.run_once()
        assert calls == []
        current = service.get(EMPLOYEE, task["id"])
        assert current["assignments"]["left"]["status"] == "blocked"
        current = service.claim(workers["right"], task["id"], update(current, "peer-later"))
        runtime.run_once()
        assert calls == [], "A peer update must not authorize repeating an uncertain local action"
    finally:
        runtime.close()


def test_cancellation_before_or_during_handler_prevents_late_success(system):
    service, _, workers, state_dir = system
    task = create(service, "left")
    calls = []

    def handler(context):
        calls.append(0)
        current = service.get(EMPLOYEE, task["id"])
        service.cancel(EMPLOYEE, task["id"], update(current, "stop-during", reason="Stop test"))
        assert context["should_stop"]() is True
        return outcome()

    runtime = sdk.AgentRuntime(Client(service, workers["left"]), handler, state_dir)
    try:
        runtime.run_once()
        assert service.get(EMPLOYEE, task["id"])["status"] == "cancelled"
        runtime.run_once()
        assert calls == [0]
        second = create(service, "left", request_id="second")
        service.cancel(EMPLOYEE, second["id"], update(second, "stop-before"))
        runtime.step(second)  # Even a stale inbox entry requires a fresh read.
        assert calls == [0]
    finally:
        runtime.close()


def test_only_one_runtime_can_use_a_worker_journal(system):
    service, _, workers, state_dir = system
    client = Client(service, workers["left"])
    first = sdk.AgentRuntime(client, lambda context: outcome(), state_dir)
    try:
        with pytest.raises(RuntimeError, match="already has"):
            sdk.AgentRuntime(client, lambda context: outcome(), state_dir)
    finally:
        first.close()
    second = sdk.AgentRuntime(client, lambda context: outcome(), state_dir)
    second.close()


def test_expired_authorization_stops_controller_callback(system):
    service, _, workers, state_dir = system
    task = create(service, "left")
    client = Client(service, workers["left"])
    runtime = sdk.AgentRuntime(client, lambda context: outcome(), state_dir)
    try:
        original = client.request
        client.request = lambda *args, **kwargs: (_ for _ in ()).throw(sdk.AgentHTTPError(401))
        assert runtime.cancelled(task["id"])
        client.request = original
        runtime.stopped.set()
        with pytest.raises(RuntimeError, match="heartbeat"):
            runtime.run_once()
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://user:password@example.com",
        "https://example.com/path",
        "https://example.com?token=secret",
        "https://example.com#fragment",
    ],
)
def test_remote_agent_credentials_require_a_clean_https_origin(url):
    with pytest.raises(ValueError):
        sdk.AgentClient(url, "secret")


def test_token_file_must_be_private_owned_regular_and_not_symlink(tmp_path):
    token = "relay_worker_" + "a" * 32
    path = tmp_path / "token"
    path.write_text(token)
    path.chmod(0o600)
    assert sdk.read_token(path) == token
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        sdk.read_token(path)
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        sdk.read_token(link)
    path.write_text(token + "\nAuthorization: leaked")
    with pytest.raises(ValueError):
        sdk.read_token(path)


def test_http_redirect_is_reported_without_forwarding_agent_credentials(monkeypatch):
    # The client's opener must install the deny-redirect handler, and a redirect
    # response must be surfaced as an error rather than retried at Location.
    calls = []

    class RedirectingOrigin:
        def open(self, request, timeout):
            calls.append(request.full_url)
            assert request.headers["Authorization"] == "Bearer private-token"
            raise HTTPError(
                request.full_url, 302, "Redirect", {"Location": "https://other.test"}, None
            )

    def opener(*handlers):
        assert any(isinstance(handler, sdk.NoRedirect) for handler in handlers)
        handler = next(handler for handler in handlers if isinstance(handler, sdk.NoRedirect))
        assert (
            handler.redirect_request(None, None, 302, "Redirect", {}, "https://other.test") is None
        )
        return RedirectingOrigin()

    monkeypatch.setattr(sdk, "build_opener", opener)
    client = sdk.AgentClient("https://workspace.test", "private-token")
    with pytest.raises(sdk.AgentHTTPError) as error:
        client.request("/api/agent/me")
    assert error.value.status == 302
    assert calls == ["https://workspace.test/api/agent/me"]


def test_non_json_or_oversize_responses_never_reach_the_handler(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b"x" * limit

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(sdk, "build_opener", lambda *args: Opener())
    with pytest.raises(ValueError, match="size limit"):
        sdk.AgentClient("https://workspace.test", "secret").request("/api/agent/me")


@pytest.mark.parametrize(
    "identity",
    [
        {"id": "../../outside", "agent_id": "left"},
        {"id": "/tmp/outside", "agent_id": "left"},
        {"id": "worker-" + "1" * 16, "agent_id": "unknown"},
        {"id": "worker-" + "1" * 16, "agent_id": ["left"]},
        None,
    ],
)
def test_malformed_workspace_identity_cannot_create_a_local_journal(tmp_path, identity):
    class InvalidIdentityClient:
        def request(self, *args, **kwargs):
            return {"worker": identity}

    with pytest.raises(ValueError, match="invalid agent identity"):
        sdk.AgentRuntime(InvalidIdentityClient(), lambda context: outcome(), tmp_path / "state")
    assert not (tmp_path / "state").exists()
