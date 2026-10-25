"""One trusted, local simulation process; scoped agents still use the public API.

This is a fixed demonstration recipe, never an arbitrary code/weights launcher.
Hosted workspaces continue to dispatch to customer-operated agent hosts.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from relay_lab.llm import MODEL_ID, MODEL_REVISION

from .simulation_recipe import (
    CHECKPOINT_PATHS,
    OBJECTIVE,
    RECIPE_ID,
    SUCCESS_CRITERIA,
    recipe_workspace,
)
from .tasks import TaskConflict, _request, _signature

ACTIVE = {"starting", "running", "verifying"}
ARTIFACTS = {
    "overview.mp4": ("overview.mp4", "video/mp4"),
    "left.mp4": ("left.mp4", "video/mp4"),
    "right.mp4": ("right.mp4", "video/mp4"),
    "poster.png": ("poster.png", "image/png"),
    "final.png": ("final.png", "image/png"),
    "summary.json": ("summary.json", "application/json"),
    "verification.json": ("verification/verification.json", "application/json"),
}
MAX_ARTIFACT_BYTES = 32 * 1024**2


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path, maximum=256 * 1024):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        return None
    try:
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else None
    except (ValueError, OSError):
        return None


class SimulationService:
    def __init__(self, service, access, registry, tasks, workers, *, hosted=False):
        self.studio, self.access, self.registry = service, access, registry
        self.tasks, self.workers, self.hosted = tasks, workers, hosted
        self.storage = service.storage / "simulations"
        self.storage.mkdir(mode=0o700, exist_ok=True)
        self.processes = {}
        self.lock = threading.RLock()
        self.closed = threading.Event()
        # A different server process may not resume an interrupted physics world.
        with registry.lock, self.lock:
            for manifest in self._manifests():
                if manifest["status"] in ACTIVE:
                    manifest.update(
                        status="interrupted", reason="Server restarted; create a new task."
                    )
                    _write(self._directory(manifest["goal_id"]) / "manifest.json", manifest)
                    self._cleanup(
                        manifest,
                        "Simulation server restarted; the interrupted world will not resume.",
                    )
                elif manifest.get("cleanup_pending"):
                    self._cleanup(manifest, manifest["cleanup_reason"])
        self.monitor = threading.Thread(target=self._watch, daemon=True, name="simulation-reaper")
        self.monitor.start()

    def _directory(self, goal_id):
        if not isinstance(goal_id, str) or not re.fullmatch(r"goal-[a-f0-9]{16}", goal_id):
            raise ValueError("Unknown simulation identifier")
        path = self.storage / goal_id
        if path.is_symlink():
            raise ValueError("Invalid simulation directory")
        return path

    def _manifests(self):
        values = []
        for path in sorted(self.storage.glob("goal-*/manifest.json"))[:100]:
            value = _read(path, 16 * 1024)
            if value and value.get("registry_id") == self.registry.registry_id:
                values.append(value)
        return sorted(values, key=lambda value: value["created_at"], reverse=True)

    def prerequisites(self):
        if self.hosted:
            return (
                False,
                "Run this MuJoCo recipe on a local simulation host. Hosted workspaces use connected agents.",
            )
        missing = [
            name
            for name in ("torch", "mujoco", "imageio", "PIL")
            if importlib.util.find_spec(name) is None
        ]
        for relative in (
            *CHECKPOINT_PATHS.values(),
            ".venv-llm/bin/python",
            "scripts/replay_cloth_actions.py",
        ):
            if not (self.studio.root / relative).is_file():
                missing.append(str(relative))
        cache = Path(os.environ.get("HF_HOME", self.studio.root / ".cache/huggingface"))
        snapshot = (
            cache
            / "hub"
            / ("models--" + MODEL_ID.replace("/", "--"))
            / "snapshots"
            / MODEL_REVISION
        )
        if not all(
            (snapshot / name).is_file()
            for name in (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        ):
            missing.append("the pinned Qwen model in the local Hugging Face cache")
        if missing:
            return False, "Install local simulation prerequisites: " + ", ".join(missing)
        return True, None

    def _release(self, actor):
        try:
            if self.studio.workspace()["agents"] != recipe_workspace()["agents"]:
                raise PermissionError(
                    "Prepare the simulation pair, then approve both model releases."
                )
            self.registry.authorize_workspace(actor, self.studio.workspace())
            return True, None
        except (PermissionError, ValueError) as error:
            return False, str(error)

    def _public(self, manifest):
        directory = self._directory(manifest["goal_id"])
        result = _read(directory / "state/result.json")
        progress = _read(directory / "state/progress.json", 32 * 1024)
        value = {key: manifest[key] for key in ("goal_id", "status", "created_at")}
        value.update(
            task_id=manifest["goal_id"],
            recipe_id=RECIPE_ID,
            progress=progress,
            result=None,
            artifacts={},
            cleanup_pending=bool(manifest.get("cleanup_pending")),
        )
        if manifest.get("reason"):
            value["reason"] = manifest["reason"]
        if manifest.get("cleanup_pending"):
            value["reason"] = (
                "Waiting for Spacetime to confirm runtime shutdown. New launches are held."
            )
        if result:
            if manifest["status"] in ACTIVE and result.get("status") == "verifying":
                value["status"] = "verifying"
            summary = result.get("summary") or {}
            value["result"] = {
                "status": manifest["status"]
                if manifest["status"] not in ACTIVE
                else result.get("status"),
                "success": manifest["status"] == "succeeded" and result.get("success") is True,
                "metrics": summary.get("metrics", {}),
                "verification": result.get("verification") or {},
                "controller_provenance": summary.get("models", {}),
                "summary": summary,
                "reason": result.get("error"),
                "verified_physical_outcome": False,
            }
        if manifest["status"] not in ACTIVE:
            for name in ARTIFACTS:
                try:
                    self._artifact_path(manifest["goal_id"], name)
                except ValueError:
                    continue
                value["artifacts"][name] = (
                    f"/api/simulations/{manifest['goal_id']}/artifacts/{name}"
                )
        return value

    def list(self, actor):
        actor = self.access.authorize_actor(actor)
        with self.registry.lock, self.lock:
            available, reason = self.prerequisites()
            ready, release_error = self._release(actor)
            runs = [self._public(item) for item in self._manifests()]
            if any(run["cleanup_pending"] for run in runs):
                available, reason = (
                    False,
                    "Waiting for Spacetime to confirm the previous runtime shutdown.",
                )
            return {
                "available": available,
                "reason": reason,
                "release_ready": ready,
                "release_error": release_error,
                "recipe": {
                    "id": RECIPE_ID,
                    "title": "Two-arm T-shirt fold",
                    "objective": OBJECTIVE,
                    "success_criteria": SUCCESS_CRITERIA,
                    "workspace": recipe_workspace(),
                    "provenance": "Original task-trained, state-assisted CNN and attention controllers. Qwen phase review; SpacetimeDB coordination. Measured simulation, not foundation-VLA or hardware validation.",
                },
                "active": next((run for run in runs if run["status"] in ACTIVE), None),
                "runs": runs,
            }

    def get(self, actor, goal_id):
        self.access.authorize_actor(actor)
        with self.registry.lock, self.lock:
            manifest = _read(self._directory(goal_id) / "manifest.json", 16 * 1024)
            if not manifest or manifest.get("registry_id") != self.registry.registry_id:
                raise ValueError("Simulation not found in this workspace")
            return self._public(manifest)

    def _preflight(self):
        coordinator = self.studio.factory()
        coordinator.client._sql("SELECT run_id FROM cloth_channel LIMIT 1")
        coordinator.client._sql("SELECT run_id FROM factory_run LIMIT 1")

    def start(self, actor, value, *, origin):
        actor = self.access.authorize_actor(actor, "models.manage")
        request_id = _request(value)
        if set(value) != {"request_id"}:
            raise ValueError("This recipe accepts only request_id; its models and task are fixed")
        if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}", origin):
            raise ValueError("Simulation requires the trusted loopback Studio origin")
        goal_id = "goal-" + _signature([self.registry.registry_id, actor["id"], request_id])[:16]
        with self.registry.lock, self.lock:
            directory = self._directory(goal_id)
            if (directory / "manifest.json").exists():
                return self.get(actor, goal_id)
            if self.closed.is_set():
                raise TaskConflict("Simulation service is shutting down")
            available, reason = self.prerequisites()
            if not available:
                raise PermissionError(reason)
            ready, release_error = self._release(actor)
            if not ready:
                raise PermissionError(release_error)
            if any(
                item["status"] in ACTIVE or item.get("cleanup_pending")
                for item in self._manifests()
            ):
                raise TaskConflict(
                    "A simulation is running or awaiting confirmed shutdown in this workspace"
                )
            with (self.storage / "engine.lock").open("a") as engine:
                try:
                    fcntl.flock(engine.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    raise TaskConflict(
                        "The previous simulation is still stopping; wait before starting another"
                    ) from None
                fcntl.flock(engine.fileno(), fcntl.LOCK_UN)
            self._preflight()
            task = self.tasks.create(
                actor,
                {
                    "request_id": request_id,
                    "title": "Two-arm T-shirt fold",
                    "objective": OBJECTIVE,
                    "success_criteria": SUCCESS_CRITERIA,
                    "target": "both",
                },
            )
            if task["status"] != "queued" or task["revision"] != 0:
                raise TaskConflict("An existing task cannot start a new physics world")
            directory.mkdir(mode=0o700, exist_ok=True)
            (directory / "state").mkdir(mode=0o700, exist_ok=True)
            manifest = {
                "goal_id": goal_id,
                "registry_id": self.registry.registry_id,
                "created_at": time.time(),
                "status": "starting",
                "workers": [],
            }
            _write(directory / "manifest.json", manifest)
            try:
                for arm in ("left", "right"):
                    credential = self.workers.create(
                        actor,
                        {
                            "name": f"Cloth simulation {arm}",
                            "agent_id": arm,
                            "execution_kind": "simulation",
                        },
                    )
                    manifest["workers"].append(credential["worker"]["id"])
                    _write(directory / "manifest.json", manifest)
                    descriptor = os.open(
                        directory / f"{arm}.token", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    with os.fdopen(descriptor, "w") as stream:
                        stream.write(credential["token"])
                python = self.studio.root / ".venv/bin/python"
                command = [
                    str(python) if python.is_file() else sys.executable,
                    "-m",
                    "relay_lab.studio.team_worker",
                    "--server",
                    origin,
                    "--task-id",
                    goal_id,
                    "--left-token-file",
                    str(directory / "left.token"),
                    "--right-token-file",
                    str(directory / "right.token"),
                    "--output",
                    str(directory / "rollout"),
                    "--root",
                    str(self.studio.root),
                    "--state-dir",
                    str(directory / "state"),
                    "--engine-lock-file",
                    str(self.storage / "engine.lock"),
                ]
                env = os.environ.copy()
                env.setdefault(
                    "RELAY_SPACETIME_TOKEN_FILE", str(Path(".relay-lab/spacetime-token").absolute())
                )
                env.setdefault("HF_HOME", str(self.studio.root / ".cache/huggingface"))
                env["HF_HUB_OFFLINE"] = "1"
                with (directory / "worker.log").open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=self.studio.root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                self.processes[goal_id] = process
                manifest.update(status="running")
                _write(directory / "manifest.json", manifest)
                self.access.audit(actor, "simulation.started", goal_id, {"recipe": RECIPE_ID})
            except Exception:
                manifest.update(
                    status="failed",
                    reason="Simulation launch failed; inspect the private worker log.",
                )
                _write(directory / "manifest.json", manifest)
                self._cleanup(
                    manifest,
                    "Simulation launch failed; no new world will be started for this task.",
                )
                raise
            return self._public(manifest)

    def _retire(self, manifest):
        # Narrow internal cleanup: only credentials created for this fixed run.
        # It must still work if the issuing manager was demoted during execution.
        with self.workers._db() as db:
            for worker_id in manifest["workers"]:
                db.execute("UPDATE runtime_worker SET active=0 WHERE id=?", (worker_id,))
        for arm in ("left", "right"):
            (self._directory(manifest["goal_id"]) / f"{arm}.token").unlink(missing_ok=True)

    def _cleanup(self, manifest, reason):
        # Persist the exact cleanup intent before stopping scoped credentials. The
        # private task transition remains possible after account/model revocation.
        manifest.setdefault("cleanup_reason", reason)
        manifest["cleanup_pending"] = True
        path = self._directory(manifest["goal_id"]) / "manifest.json"
        _write(path, manifest)
        try:
            self._retire(manifest)
            task = self.tasks.cancel_managed(
                manifest["goal_id"], manifest["workers"], manifest["cleanup_reason"]
            )
            if manifest["status"] == "succeeded" and task["status"] != "succeeded":
                manifest.update(
                    status="failed", reason="Both agent completion reports were not confirmed."
                )
            manifest["cleanup_pending"] = False
            manifest.pop("cleanup_error", None)
        except Exception:  # noqa: BLE001 - retain durable intent for the next backend readback
            manifest["cleanup_error"] = "Runtime shutdown is awaiting authoritative confirmation."
        _write(path, manifest)

    def _reap(self):
        for goal_id, process in list(self.processes.items()):
            if process.poll() is None:
                continue
            directory = self._directory(goal_id)
            manifest = _read(directory / "manifest.json")
            result = _read(directory / "state/result.json")
            state = (result or {}).get("status", "failed")
            manifest["status"] = (
                state if state in {"succeeded", "failed", "cancelled", "interrupted"} else "failed"
            )
            if manifest["status"] == "succeeded" and (
                process.returncode != 0
                or result.get("success") is not True
                or (result.get("verification") or {}).get("verified") is not True
                or (result.get("summary") or {}).get("success") is not True
            ):
                manifest["status"] = "failed"
            if not result:
                manifest["reason"] = (
                    "Worker exited without a completion receipt; inspect its private log."
                )
            _write(directory / "manifest.json", manifest)
            self._cleanup(
                manifest, "Simulation process exited; any unfinished assignment is stopped."
            )
            del self.processes[goal_id]

    def _watch(self):
        while not self.closed.wait(1):
            with self.registry.lock, self.lock:
                self._reap()
                for manifest in self._manifests():
                    if manifest.get("cleanup_pending"):
                        self._cleanup(manifest, manifest["cleanup_reason"])
                        break

    def _artifact_path(self, goal_id, name):
        if name not in ARTIFACTS:
            raise ValueError("Unknown simulation artifact")
        root = self._directory(goal_id) / "rollout"
        path = root / ARTIFACTS[name][0]
        if root.is_symlink() or any(
            parent.is_symlink() for parent in path.parents if parent.is_relative_to(root)
        ):
            raise ValueError("Invalid simulation artifact directory")
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Simulation artifact is not available")
        if not 0 < path.stat().st_size <= MAX_ARTIFACT_BYTES:
            raise ValueError("Simulation artifact exceeds the download bound")
        return path

    def artifact(self, actor, goal_id, name):
        run = self.get(actor, goal_id)
        if name not in run["artifacts"]:
            raise ValueError("Simulation artifact is not available yet")
        return self._artifact_path(goal_id, name), ARTIFACTS[name][1]

    def close(self):
        self.closed.set()
        with self.registry.lock, self.lock:
            self._reap()
            for goal_id in self.processes:
                manifest = _read(self._directory(goal_id) / "manifest.json")
                manifest.update(status="interrupted", reason="Studio stopped; create a new task.")
                _write(self._directory(goal_id) / "manifest.json", manifest)
                self._cleanup(manifest, "Studio stopped; the interrupted world will not resume.")
