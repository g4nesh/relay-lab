"""The deployable snapshot contains its whole runtime and no research data."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_studio", ROOT / "scripts/export_studio.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_export_is_complete_deterministic_and_has_no_research_or_private_inputs(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = exporter.export_studio(first)
    assert exporter.export_studio(second) == manifest
    actual = {str(path.relative_to(first)) for path in first.rglob("*") if path.is_file()}
    assert actual == set(exporter.source_map()) | {"export-manifest.json"}
    for entry in manifest["files"]:
        assert hashlib.sha256((first / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        assert (first / entry["path"]).read_bytes() == (second / entry["path"]).read_bytes()
    assert not any(
        part in {".git", ".relay-lab", "checkpoints", "datasets", "examples", "cloth", "runs"}
        for name in actual
        for part in Path(name).parts
    )
    assert not any(
        name.endswith((".pt", ".mp4", ".npy", ".npz", ".safetensors")) for name in actual
    )
    metadata = tomllib.loads((first / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "relay-studio"
    assert set(metadata["project"]["dependencies"]) == {
        "uvicorn==0.53.0",
        "click==8.5.0",
        "h11==0.16.0",
    }
    assert "COPY pyproject.toml ./" in (first / "Dockerfile").read_text()
    assert "RELAY_STUDIO_EXPERIMENTS=0" in (first / "Dockerfile").read_text()
    assert (first / "subscriptions/src/bridge.cjs").is_file()
    assert (first / "spacetimedb/src/studio_tables.ts").is_file()
    assert (first / "src/relay_lab/studio/web/assets/relay-logo-exact.png").is_file()
    assert (first / "tests/test_realtime.py").is_file()
    assert (first / "src/relay_lab/studio/durable.py").is_file()
    assert (first / "subscriptions/src/module_bindings/studio_commit_state_reducer.ts").is_file()
    assert (first / "tests/test_durable.py").is_file()
    assert (first / "tests/test_studio_local_upload.py").is_file()
    assert (first / "deploy/read-setup-code.py").is_file()
    assert (first / "render.yaml").is_file()
    assert exporter.check_against(first) == {
        "missing": [],
        "changed": [],
        "unexpected": [],
        "untracked": [],
    }


def test_existing_destination_is_never_changed(tmp_path):
    target = tmp_path / "customer-app"
    target.mkdir()
    (target / "custom.py").write_text("preserve me")
    with pytest.raises(ValueError, match="already exists"):
        exporter.export_studio(target)
    assert list(target.iterdir()) == [target / "custom.py"]
    assert (target / "custom.py").read_text() == "preserve me"


def test_missing_or_symlinked_source_does_not_leave_partial_export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="Missing required"):
        exporter.export_studio(target, root=source)
    assert not target.exists()
    for relative in exporter.source_map().values():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    selected = source / "src/relay_lab/spacetime.py"
    selected.unlink()
    selected.symlink_to(ROOT / "src/relay_lab/spacetime.py")
    with pytest.raises(ValueError, match="symlinks"):
        exporter.export_studio(target, root=source)
    assert not target.exists()


def test_root_docker_context_includes_exact_deployable_source_allowlist():
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    included = {line[1:] for line in lines if line.startswith("!src/") and not line.endswith("/")}
    assert included == {path for path in exporter.source_map() if path.startswith("src/")}
    assert "!src/**" not in lines
    assert not any(line.startswith("!") and line.endswith("/") for line in lines)
    assert "COPY deploy/studio/pyproject.toml ./" in (ROOT / "Dockerfile").read_text()


def test_comparison_reports_changed_missing_and_unexpected_without_mutation(tmp_path):
    target = tmp_path / "app"
    exporter.export_studio(target)
    missing = target / "subscriptions/src/bridge.cjs"
    missing.unlink()
    changed = target / "README.md"
    changed.write_text("local change to retain")
    extra = target / "experiments/training.py"
    extra.parent.mkdir()
    extra.write_text("research does not belong here")
    before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert exporter.check_against(target) == {
        "missing": ["subscriptions/src/bridge.cjs"],
        "changed": ["README.md"],
        "unexpected": ["experiments/training.py"],
        "untracked": [],
    }
    assert before == {
        str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()
    }


def test_git_comparison_flags_tracked_research_but_ignores_untracked_state(tmp_path):
    target = tmp_path / "app"
    exporter.export_studio(target)
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "add", "."], check=True)
    private = target / "state"
    private.mkdir()
    # This untracked symlink must never be followed or included in the manifest.
    (private / "private-token").symlink_to(tmp_path / "unreadable-secret")
    assert not any(exporter.check_against(target).values())
    extra = target / "train_policy.py"
    extra.write_text("pass\n")
    subprocess.run(["git", "-C", str(target), "add", "train_policy.py"], check=True)
    assert exporter.check_against(target)["unexpected"] == ["train_policy.py"]
    subprocess.run(["git", "-C", str(target), "rm", "--cached", "-q", "README.md"], check=True)
    assert exporter.check_against(target)["untracked"] == ["README.md"]


def test_comparison_rejects_symlinked_expected_file_and_destination(tmp_path):
    target = tmp_path / "app"
    exporter.export_studio(target)
    doc = target / "README.md"
    doc.unlink()
    doc.symlink_to(ROOT / "deploy/studio/README.md")
    assert exporter.check_against(target)["changed"] == ["README.md"]
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="not a symlink"):
        exporter.check_against(alias)


def test_cli_comparison_exit_status_reflects_drift(tmp_path):
    target = tmp_path / "app"
    exporter.export_studio(target)
    command = [
        sys.executable,
        str(ROOT / "scripts/export_studio.py"),
        "--check-against",
        str(target),
    ]
    assert subprocess.run(command, capture_output=True, timeout=20, check=False).returncode == 0
    (target / "README.md").write_text("local change")
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 1
    assert "changed: README.md" in result.stdout
    assert "local change" not in result.stdout


@pytest.mark.parametrize("flag", ["", "0", "1"])
def test_export_starts_outside_checkout_without_any_installed_dependencies(tmp_path, flag):
    target = tmp_path / "app"
    exporter.export_studio(target)
    script = """
import asyncio, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
os.environ.pop('RELAY_STUDIO_EXPERIMENTS', None)
if sys.argv[3]:
    os.environ['RELAY_STUDIO_EXPERIMENTS'] = sys.argv[3]
from relay_lab.studio.asgi import create_app
from relay_lab.studio.deployment import DeploymentConfig
from relay_lab.studio.worker import AgentRuntime
app = create_app(config=DeploymentConfig(root=Path(sys.argv[2])))
assert app.runtime['service'].capabilities()['experiments_enabled'] is False
assert not any(n.startswith(('torch', 'numpy', 'mujoco', 'PIL', 'mlx', 'lerobot',
    'relay_lab.cloth', 'relay_lab.llm', 'relay_lab.studio.simulations')) for n in sys.modules)
print(json.dumps({'profile': 'control-plane', 'worker_import': AgentRuntime.__name__}))
asyncio.run(app.close())
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            str(target / "src"),
            str(tmp_path / "state"),
            flag,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(result.stdout)["worker_import"] == "AgentRuntime"
