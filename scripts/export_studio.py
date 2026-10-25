"""Export the deployable dashboard from an explicit, experiment-free file allowlist.

Run with Python 3.12+. The output is a new standalone project, never an in-place
rewrite. No Git history, workspace state, credentials, weights or datasets are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ("__init__", "spacetime")
COORDINATION = ("__init__", "communication", "coordination", "embeddings", "spacetime")
STUDIO = (
    "__init__",
    "access",
    "analytics",
    "asgi",
    "coordination",
    "deployment",
    "durable",
    "governance",
    "healthcheck",
    "profiles",
    "realtime",
    "server",
    "service",
    "tasks",
    "telemetry",
    "worker",
    "workers",
)
WEB = (
    "index.html",
    "demo.js",
    "demo.css",
    "studio.js",
    "studio.css",
    "tasks.js",
    "tasks.css",
    "analytics.js",
    "analytics.css",
    "profiles.js",
    "profiles.css",
    "coordination-design.css",
    "workspace-design.css",
    "hosted.css",
    "local.css",
    "assets/relay-logo-exact.png",
    "assets/fonts/LICENSE-OFL.txt",
    "assets/fonts/ibm-plex-mono-latin-400.woff2",
    "assets/fonts/ibm-plex-mono-latin-500.woff2",
    "assets/fonts/ibm-plex-sans-latin-400.woff2",
    "assets/fonts/ibm-plex-sans-latin-500.woff2",
    "assets/fonts/ibm-plex-sans-latin-600.woff2",
)


# Explicit generated inventory: schema changes must update and review this boundary.
SUBSCRIPTION_BINDINGS = (
    "abort_factory_run_reducer.ts",
    "abort_task_reducer.ts",
    "action_frame_table.ts",
    "assignment_table.ts",
    "claim_task_reducer.ts",
    "cloth_channel_table.ts",
    "cloth_frame_table.ts",
    "cloth_packet_table.ts",
    "commit_cloth_frame_reducer.ts",
    "commit_factory_batch_reducer.ts",
    "commit_tick_reducer.ts",
    "configure_factory_embeddings_reducer.ts",
    "configure_factory_semantic_channel_reducer.ts",
    "create_cloth_channel_reducer.ts",
    "create_factory_run_reducer.ts",
    "create_task_reducer.ts",
    "factory_batch_table.ts",
    "factory_command_table.ts",
    "factory_embedding_space_table.ts",
    "factory_embedding_table.ts",
    "factory_event_table.ts",
    "factory_resource_table.ts",
    "factory_robot_table.ts",
    "factory_run_table.ts",
    "factory_semantic_channel_table.ts",
    "factory_semantic_packet_table.ts",
    "feedback_frame_table.ts",
    "index.ts",
    "module_owner_table.ts",
    "plan_frame_table.ts",
    "plan_tick_reducer.ts",
    "prepare_factory_batch_reducer.ts",
    "prepare_tick_reducer.ts",
    "studio_abort_factory_run_reducer.ts",
    "studio_acquire_reducer.ts",
    "studio_attention_table.ts",
    "studio_commit_factory_batch_reducer.ts",
    "studio_commit_state_reducer.ts",
    "studio_configure_job_reducer.ts",
    "studio_create_factory_run_reducer.ts",
    "studio_execute_job_reducer.ts",
    "studio_expire_watchdog_reducer.ts",
    "studio_fenced_configure_job_reducer.ts",
    "studio_fenced_heartbeat_reducer.ts",
    "studio_fenced_record_metric_reducer.ts",
    "studio_fenced_revoke_worker_reducer.ts",
    "studio_fenced_run_job_reducer.ts",
    "studio_fenced_sync_task_reducer.ts",
    "studio_finish_read_reducer.ts",
    "studio_heartbeat_reducer.ts",
    "studio_job_table.ts",
    "studio_job_trigger_table.ts",
    "studio_lease_table.ts",
    "studio_metric_table.ts",
    "studio_prepare_factory_batch_reducer.ts",
    "studio_presence_table.ts",
    "studio_record_metric_reducer.ts",
    "studio_release_reducer.ts",
    "studio_revoke_worker_reducer.ts",
    "studio_rollup_table.ts",
    "studio_run_job_reducer.ts",
    "studio_state_table.ts",
    "studio_sync_task_reducer.ts",
    "studio_task_watch_table.ts",
    "studio_watchdog_table.ts",
    "studio_workspace_table.ts",
    "task_table.ts",
    "types.ts",
    "types/procedures.ts",
    "types/reducers.ts",
)

TESTS = (
    "test_access_regression",
    "test_durable",
    "test_durable_live",
    "test_governance_regression",
    "test_coordination",
    "test_coordination_live",
    "test_realtime",
    "test_studio_deployment",
    "test_studio_local_upload",
    "test_studio_tasks",
    "test_studio_worker",
    "test_worker_events",
)


def source_map():
    paths = [f"src/relay_lab/{name}.py" for name in CORE]
    paths += [f"src/relay_lab/factory/{name}.py" for name in COORDINATION]
    paths += [f"src/relay_lab/studio/{name}.py" for name in STUDIO]
    paths += [f"src/relay_lab/studio/web/{name}" for name in WEB]
    paths += [f"subscriptions/src/module_bindings/{name}" for name in SUBSCRIPTION_BINDINGS]
    paths += [f"tests/{name}.py" for name in TESTS]
    paths += [
        "deploy/compose.yaml",
        "deploy/Caddyfile",
        "deploy/.env.example",
        "deploy/init-state.py",
        "deploy/publish-spacetime.py",
        "deploy/read-setup-code.py",
        "render.yaml",
        "docs/render-free.md",
        "spacetimedb/src/index.ts",
        "spacetimedb/src/studio_policy.ts",
        "spacetimedb/src/studio_tables.ts",
        "spacetimedb/tests/studio.test.mjs",
        "subscriptions/README.md",
        "subscriptions/package.json",
        "subscriptions/package-lock.json",
        "subscriptions/tsconfig.json",
        "subscriptions/src/bridge.cjs",
        "subscriptions/src/invalidations.cjs",
        "subscriptions/tests/invalidations.test.cjs",
        "tests/web/demo.test.cjs",
        "docs/coordination.md",
        "docs/studio-sync.md",
        "spacetimedb/package.json",
        "spacetimedb/package-lock.json",
        "spacetimedb/tsconfig.json",
        "docs/studio-deployment.md",
    ]
    mapping = {name: name for name in paths}
    mapping.update(
        {
            "pyproject.toml": "deploy/studio/pyproject.toml",
            "README.md": "deploy/studio/README.md",
            ".gitignore": "deploy/studio/gitignore",
            ".dockerignore": ".dockerignore",
            "docs/agents.md": "deploy/studio/agents.md",
            "Dockerfile": "Dockerfile",
        }
    )
    return dict(sorted(mapping.items()))


def _read_source(root: Path, relative: str) -> bytes:
    path = root / relative
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise ValueError(f"Export sources must not be symlinks: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing required export source: {relative}")
    return path.read_bytes()


def snapshot(root: Path = ROOT) -> tuple[dict[str, bytes], dict]:
    """Read only allowlisted files and return deterministic standalone bytes."""
    root = root.resolve()
    files = {}
    entries = []
    for target, source in source_map().items():
        data = _read_source(root, source)
        if target == "Dockerfile":
            old = b"COPY deploy/studio/pyproject.toml ./"
            if data.count(old) != 1:
                raise ValueError("Root Dockerfile no longer matches standalone packaging")
            data = data.replace(old, b"COPY pyproject.toml ./")
        if target == ".dockerignore":
            data = data.replace(b"!deploy/studio/pyproject.toml", b"!pyproject.toml")
        files[target] = data
        entries.append(
            {
                "path": target,
                "source": source,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "format": 1,
        "profile": "control-plane",
        "files": entries,
        "note": "Source snapshot only. No customer state or experiment artifacts included.",
    }
    files["export-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return files, manifest


def export_studio(destination: Path, *, root: Path = ROOT) -> dict:
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output already exists; choose a new folder to preserve local changes")
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("Output parent must not be a symlink")
    files, manifest = snapshot(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".relay-studio-export-", dir=destination.parent))
    try:
        for target, data in files.items():
            path = staging / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        # The complete snapshot becomes visible at once. Never merge existing outputs.
        if destination.exists() or destination.is_symlink():
            raise ValueError("Output already exists; choose a new folder to preserve local changes")
        os.rename(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def check_against(destination: Path, *, root: Path = ROOT) -> dict[str, list[str]]:
    """Compare working files without writing; in Git, also enforce tracked inventory.

    Untracked customer state and build products are not read. Git-tracked files
    outside the allowlist are drift, even if the dashboard itself still imports.
    A plain export directory is checked against its complete file inventory.
    """
    destination = destination.absolute()
    if not destination.is_dir() or any(
        parent.is_symlink() for parent in (destination, *destination.parents)
    ):
        raise ValueError("Comparison target must be a real directory, not a symlink")
    expected, _ = snapshot(root)
    result: dict[str, list[str]] = {"missing": [], "changed": [], "unexpected": [], "untracked": []}
    for name, data in expected.items():
        path = destination / name
        if not path.exists() and not path.is_symlink():
            result["missing"].append(name)
            continue
        try:
            actual = _read_source(destination, name)
        except ValueError:
            result["changed"].append(name)
            continue
        if actual != data:
            result["changed"].append(name)
    git_root = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        git_root.returncode == 0
        and Path(git_root.stdout.strip()).resolve() == destination.resolve()
    ):
        tracked = subprocess.run(
            ["git", "-C", str(destination), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
        actual_names = {os.fsdecode(name) for name in tracked.stdout.split(b"\0") if name}
        result["untracked"] = sorted(expected.keys() - actual_names - set(result["missing"]))
    else:
        actual_names = {
            str(path.relative_to(destination))
            for path in destination.rglob("*")
            if path.is_file() or path.is_symlink()
        }
    result["unexpected"] = sorted(actual_names - expected.keys())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--output", type=Path, default=ROOT / "dist/relay-studio")
    operation.add_argument(
        "--check-against", type=Path, help="Read-only comparison with a standalone checkout"
    )
    args = parser.parse_args()
    try:
        if args.check_against is not None:
            drift = check_against(args.check_against)
            for kind, names in drift.items():
                for name in names:
                    print(f"{kind}: {name}")
            if any(drift.values()):
                parser.exit(1, "Standalone checkout differs from the canonical export.\n")
            print("Standalone checkout matches the canonical control-plane export.")
            return
        manifest = export_studio(args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Export refused: {error}\n")
    total = sum(item["bytes"] for item in manifest["files"])
    print(f"Exported {len(manifest['files'])} files ({total:,} bytes) to {args.output.absolute()}")
    print("Standalone project ready. Read README.md before deploying.")


if __name__ == "__main__":
    main()
