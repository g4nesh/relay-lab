"""The standalone control plane never imports or executes research runtimes."""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_studio_permissions_http import Bridge
from test_studio_tasks import Journal
from test_studio_workers_http import agent_request, connection, task

from relay_lab.studio.analytics import AnalyticsService
from relay_lab.studio.asgi import create_app
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.service import CONTROL_PLANE_REASON, StudioService, example_workspace


@pytest.fixture
def bridge(tmp_path):
    result = Bridge(tmp_path)
    result.service.close()
    result.task_journal = Journal()
    result.service = StudioService(
        tmp_path, coordinator_factory=result.task_journal.factory, experiments=False
    )
    result.save_and_release()
    yield result
    result.service.close()


def test_isolated_import_and_startup_without_research_or_site_packages(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/relay_lab"
    package = tmp_path / "src/relay_lab"
    package.mkdir(parents=True)
    for name in ("__init__.py", "spacetime.py"):
        shutil.copy2(source / name, package / name)
    for directory, names in {
        "factory": "__init__ communication coordination embeddings spacetime",
        "studio": (
            "__init__ access analytics asgi deployment governance profiles server service "
            "tasks telemetry workers coordination realtime schedules"
        ),
    }.items():
        (package / directory).mkdir()
        for name in names.split():
            shutil.copy2(source / directory / f"{name}.py", package / directory / f"{name}.py")
    script = """
import asyncio, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
os.environ['RELAY_STUDIO_EXPERIMENTS'] = '0'
from relay_lab.studio.asgi import create_app
from relay_lab.studio.deployment import DeploymentConfig
app = create_app(config=DeploymentConfig(root=Path(sys.argv[2])))
service = app.runtime['service']
assert service.experiments_enabled is False
assert service.llm is None and service.artifacts is None
assert not any(name.startswith(('relay_lab.cloth', 'relay_lab.llm',
    'relay_lab.studio.artifacts', 'relay_lab.studio.simulations',
    'torch', 'mujoco', 'numpy', 'PIL', 'mlx', 'lerobot')) for name in sys.modules)
print(json.dumps(service.capabilities()))
asyncio.run(app.close())
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(package.parent), str(tmp_path / "state")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    assert json.loads(result.stdout)["experiments_enabled"] is False


@pytest.mark.parametrize("hosted", [False, True])
def test_app_selects_control_plane_for_hosted_or_explicit_local_mode(tmp_path, monkeypatch, hosted):
    monkeypatch.setenv("RELAY_STUDIO_EXPERIMENTS", "1" if hosted else "0")
    monkeypatch.setenv("RELAY_SPACETIME_TOKEN", "test-owner-token")
    config = DeploymentConfig(
        root=tmp_path,
        public_origin="https://robots.example.com" if hosted else "",
        bind="0.0.0.0" if hosted else "127.0.0.1",
    )
    app = create_app(config=config)
    try:
        assert app.runtime["service"].experiments_enabled is False
        assert app.runtime["service"].llm is None
    finally:
        asyncio.run(app.close())


def test_research_remains_explicit_service_default(tmp_path):
    service = StudioService(tmp_path)
    try:
        assert service.experiments_enabled is True
        assert service.llm is not None and service.artifacts is not None
    finally:
        service.close()


def test_capabilities_health_and_bootstrap_do_not_advertise_local_experiments(bridge):
    status, _, bootstrap = bridge.request("/api/bootstrap", actor=bridge.manager)
    assert status == 200 and bootstrap["example"] is None
    capability = bootstrap["capabilities"]
    assert capability["experiments_enabled"] is False
    assert capability["sample_available"] is capability["llm_available"] is False
    assert capability["adapter"]["profile"] == "control-plane"
    health = bridge.request("/api/health", actor=bridge.manager)[2]
    assert health["experiments_enabled"] is False
    assert health["llm"] == {
        "available": False, "loaded": False, "model_id": None, "revision": None
    }
    assert not hasattr(bridge.service, "simulations")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/samples/clear/left.png"),
        ("GET", "/api/sessions"),
        ("POST", "/api/sessions"),
        ("GET", "/api/sessions/studio-0123456789abcdef"),
        ("POST", "/api/sessions/studio-0123456789abcdef/exchange"),
        ("POST", "/api/sessions/studio-0123456789abcdef/resume"),
        ("POST", "/api/simulations"),
        ("GET", "/api/simulations/goal-0123456789abcdef"),
        ("GET", "/api/simulations/goal-0123456789abcdef/artifacts/overview.mp4"),
        ("POST", "/api/upload"),
    ],
)
def test_research_routes_fail_closed_before_optional_imports(bridge, method, path):
    status, _, value = bridge.request(path, method=method, actor=bridge.manager, value={})
    assert status == 403 and value == {"error": CONTROL_PLANE_REASON}
    assert bridge.task_journal.runs == {}
    assert not hasattr(bridge.service, "simulations")


def test_simulation_descriptor_requires_auth_and_never_starts_supervisor(bridge):
    assert bridge.request("/api/simulations")[0] == 401
    status, _, value = bridge.request("/api/simulations", actor=bridge.employee)
    assert status == 200
    assert value == {
        "available": False, "profile": "control-plane", "runs": [],
        "active": None, "reason": CONTROL_PLANE_REASON,
    }
    assert not hasattr(bridge.service, "simulations")


@pytest.mark.parametrize("name,args", [
    ("connect", (example_workspace(),)),
    ("sample", ("clear",)),
    ("sample_image", ("clear", "left")),
    ("exchange", ("studio-0123456789abcdef", {})),
    ("recover", ("studio-0123456789abcdef",)),
])
def test_direct_research_entry_points_are_guarded(bridge, name, args):
    with pytest.raises(PermissionError, match="Local research previews"):
        getattr(bridge.service, name)(*args)


def test_customer_task_claim_report_duplicate_and_revoke_survive_boundary(bridge):
    worker, goal = connection(bridge), task(bridge)
    status, _, inbox = agent_request(bridge, worker, "/api/agent/tasks")
    assert status == 200 and inbox["tasks"][0]["id"] == goal["id"]
    status, _, claimed = agent_request(
        bridge, worker, f"/api/agent/tasks/{goal['id']}/claim",
        {"request_id": "claim-control", "expected_revision": 0},
    )
    assert status == 200
    payload = {
        "request_id": "report-control", "expected_revision": claimed["revision"],
        "status": "succeeded", "progress": 100, "summary": "Inspection recorded",
        "evidence": ["Agent observation"],
    }
    path = f"/api/agent/tasks/{goal['id']}/report"
    status, _, finished = agent_request(bridge, worker, path, payload)
    assert status == 200 and finished["status"] == "succeeded"
    assert finished["success_basis"] == "agent_reported"
    assert not finished["verified_physical_outcome"]
    assert agent_request(bridge, worker, path, payload)[2] == finished
    assert bridge.request(f"/api/tasks/{goal['id']}", actor=bridge.employee)[2] == finished
    status, _, _ = bridge.request(
        f"/api/workers/{worker['worker']['id']}/revoke", method="POST",
        actor=bridge.manager, value={},
    )
    assert status == 200
    assert agent_request(bridge, worker, "/api/agent/tasks")[0] == 401


def test_live_analytics_kept_but_archived_research_never_read(bridge, monkeypatch):
    from relay_lab.studio import analytics

    archive = bridge.service.root / "research.json"
    archive.write_text('{"attempts": [{"result": {"exchange": 999}}]}')
    monkeypatch.setattr(analytics, "SOURCES", {"studio_archive": ("research.json", "archive")})
    def no_research_journal(*args):
        raise AssertionError("Customer analytics must not inspect database-wide research sessions")

    monkeypatch.setattr(bridge.service, "list_sessions", no_research_journal)
    monkeypatch.setattr(bridge.service, "history", no_research_journal)
    bridge.service.telemetry.record("studio-local", "both", {
        "exchange": 1, "mode": "mediated", "source": "manual", "bytes": 24,
        "elapsed_ms": 10, "database_ms": 3, "readback_verified": True,
    })
    service = AnalyticsService(
        bridge.service.root, bridge.service, bridge.registry, bridge.access
    )
    snapshot = service.snapshot(bridge.manager["user"])
    assert snapshot["experiments_enabled"] is False
    assert snapshot["sources"] == []
    assert snapshot["operations"]["exchanges"] == 1
    assert snapshot["operations"]["archive"]["exchanges"] == 0
    assert snapshot["operations"]["coverage"]["history_sessions_read"] == 0
    assert snapshot["operations"]["coverage"]["journal_only_exchanges"] == 0
    assert snapshot["governance"]["models"]["released"] == 2
    assert snapshot["headline"]["value"] == 1
