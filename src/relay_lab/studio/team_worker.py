"""A reviewed local two-arm simulation, executed once with two scoped connections.

This controller runs one shared MuJoCo world. It does not dispatch arbitrary
registered weights, contact hardware, or claim that foundation VLAs fold cloth.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from .simulation_recipe import (
    CHECKPOINT_PATHS,
    CODEC_HASH,
    CODEC_PATH,
    OBJECTIVE,
    PINNED_HASHES,
    RECIPE_ID,
    SUCCESS_CRITERIA,
    recipe_config,
    recipe_workspace,
)
from .worker import AgentClient, AgentHTTPError, read_token, report_value

ARMS = ("left", "right")
TERMINAL = {"succeeded", "failed", "cancelled"}


class SimulationStopped(RuntimeError):
    """There is no longer fresh authorization for the shared simulation."""


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_replay(root, source, should_stop):
    """Re-execute recorded actions, without repeating policies or Qwen inference."""
    spec = importlib.util.spec_from_file_location(
        "relay_simulation_replay", Path(root) / "scripts" / "replay_cloth_actions.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("The reviewed action replay verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_world = module.ClothWorld

    class CancellableReplayWorld(original_world):
        def step(self, actions):
            if should_stop():
                raise SimulationStopped("Action replay cancelled before its next physics step")
            return super().step(actions)

    module.ClothWorld = CancellableReplayWorld
    return module.replay(Path(source), Path(source) / "verification", video=False)


class TeamRuntime:
    """One durable journal fences a pair, including ambiguous network outcomes."""

    def __init__(
        self, clients, task_id, output, root, state_dir, *,
        runner=None, verifier=None, poll_seconds=2.0, freshness_seconds=12.0,
        engine_lock_file=None,
    ):
        if not re.fullmatch(r"goal-[a-f0-9]{16}", task_id):
            raise ValueError("Invalid task identifier")
        if set(clients) != set(ARMS):
            raise ValueError("The shared world requires both scoped agent clients")
        self.clients, self.task_id = clients, task_id
        self.engine_lock = None
        if engine_lock_file is not None:
            fd = os.open(engine_lock_file, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            self.engine_lock = os.fdopen(fd, "a")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.engine_lock.close()
                raise RuntimeError("Another simulation owns this workspace engine") from None
        self.output, self.root = Path(output).absolute(), Path(root).absolute()
        self.directory = Path(state_dir).absolute()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.lock = (self.directory / "runtime.lock").open("a")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            if self.engine_lock:
                self.engine_lock.close()
            raise RuntimeError("This paired simulation already has a running process") from None
        self.runner, self.verifier = runner, verifier or verify_replay
        self.poll_seconds, self.freshness_seconds = poll_seconds, freshness_seconds
        self.stop_event, self.monitor_done = threading.Event(), threading.Event()
        self.monitor = None
        self.last_authority = 0.0
        self.stop_reason = None
        self.workers = {}
        path = self.directory / "journal.json"
        if path.exists():
            if path.is_symlink() or path.stat().st_size > 256 * 1024:
                self.close()
                raise ValueError("Invalid paired simulation journal")
            self.state = json.loads(path.read_text())
            if self.state.get("task_id") != task_id:
                self.close()
                raise ValueError("State directory belongs to another task")
        else:
            self.state = {"task_id": task_id, "phase": "new", "pending": None}

    def close(self):
        self.monitor_done.set()
        if self.monitor:
            self.monitor.join(timeout=1)
        if not self.lock.closed:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
        if self.engine_lock and not self.engine_lock.closed:
            fcntl.flock(self.engine_lock.fileno(), fcntl.LOCK_UN)
            self.engine_lock.close()

    def save(self):
        _write(self.directory / "journal.json", self.state)

    def result(self, status, **extra):
        value = {
            "task_id": self.task_id,
            "recipe_id": RECIPE_ID,
            "status": status,
            "success": status == "succeeded",
            "success_basis": "measured_simulation",
            "verified_physical_outcome": False,
            "progress": self.state.get("progress"),
            **extra,
        }
        _write(self.directory / "result.json", value)
        return value

    def current(self, arm):
        return self.clients[arm].request(f"/api/agent/tasks/{self.task_id}")

    def check_task(self, task):
        expected = {spec["id"]: spec for spec in recipe_workspace()["agents"]}
        if (
            task.get("id") != self.task_id
            or task.get("objective") != OBJECTIVE
            or task.get("success_criteria") != SUCCESS_CRITERIA
            or task.get("target") != "both"
            or set(task.get("assignments", {})) != set(ARMS)
            or task.get("readback_verified") is not True
        ):
            raise ValueError("Task does not match the reviewed two-arm simulation recipe")
        for arm in ARMS:
            assignment = task["assignments"][arm]
            worker = self.workers[arm]
            if (
                assignment.get("spec") != expected[arm]
                or assignment.get("model_id") != worker.get("model_id")
                or assignment.get("release_revision") != worker.get("release_revision")
                or assignment.get("worker_id") not in (None, worker["id"])
                or assignment.get("instruction") != OBJECTIVE
            ):
                raise ValueError("Assigned model, release, or runtime does not match the recipe")

    def _authority(self):
        for arm in ARMS:
            self.clients[arm].request(
                "/api/agent/heartbeat",
                {"controller_label": "Task-trained cloth controller · Qwen + Spacetime"},
            )
            task = self.current(arm)
            self.check_task(task)
            if task["status"] in TERMINAL:
                self.stop_reason = f"Workspace task is {task['status']}"
                raise SimulationStopped(self.stop_reason)
        self.last_authority = time.monotonic()

    def start_monitor(self):
        self._authority()

        def monitor():
            while not self.monitor_done.wait(self.poll_seconds):
                try:
                    self._authority()
                except Exception:  # noqa: BLE001 - authorization loss must stop the next action
                    self.stop_reason = self.stop_reason or "Workspace authorization is unavailable"
                    self.stop_event.set()
                    return

        self.monitor = threading.Thread(target=monitor, daemon=True)
        self.monitor.start()

    def should_stop(self):
        if time.monotonic() - self.last_authority > self.freshness_seconds:
            self.stop_reason = self.stop_reason or "Workspace authorization became stale"
            self.stop_event.set()
        return self.stop_event.is_set()

    def mutate(self, arm, operation, value):
        """Persist an exact proposal before sending; rebase only an uncommitted envelope."""
        pending = self.state.get("pending")
        if pending and (pending["arm"] != arm or pending["operation"] != operation):
            raise RuntimeError("An earlier report must be reconciled before another update")
        if not pending:
            task = self.current(arm)
            self.check_task(task)
            if task["status"] in TERMINAL:
                raise SimulationStopped(f"Workspace task is {task['status']}")
            pending = {
                "arm": arm,
                "operation": operation,
                "payload": {
                    **value, "request_id": uuid.uuid4().hex,
                    "expected_revision": task["revision"],
                },
            }
            self.state["pending"] = pending
            self.save()
        for attempt in range(6):
            try:
                task = self.clients[arm].request(
                    f"/api/agent/tasks/{self.task_id}/{operation}", pending["payload"]
                )
                self.state["pending"] = None
                self.save()
                return task
            except AgentHTTPError as exc:
                if exc.status in (401, 403, 404):
                    raise
                if exc.status == 409:
                    task = self.current(arm)
                    self.check_task(task)
                    receipt = next(
                        (r for r in task.get("request_receipts", [])
                         if r["request_id"] == pending["payload"]["request_id"]), None,
                    )
                    if receipt:
                        self.state["pending"] = None
                        self.save()
                        if receipt.get("cancelled_before_commit"):
                            raise SimulationStopped("Operator cancelled the pending report")
                        return task
                    if task["status"] in TERMINAL:
                        raise SimulationStopped(f"Workspace task is {task['status']}") from None
                    if not task.get("pending_update"):
                        pending["payload"].update(
                            request_id=uuid.uuid4().hex, expected_revision=task["revision"]
                        )
                        self.save()
                elif exc.status < 500:
                    raise
            except (OSError, TimeoutError):
                pass  # The exact request remains durable; its action is never repeated.
            if attempt < 5:
                self.monitor_done.wait(0.1 * (attempt + 1))
        raise RuntimeError("Workspace update could not be confirmed; simulation held")

    def progress(self, value):
        previous = self.state.get("progress") or {}
        self.state["progress"] = value
        self.save()
        _write(self.directory / "progress.json", value)
        self.result("running")
        if value.get("status") != "running":
            return  # Actual final geometry and independent replay decide success below.
        if self.should_stop():
            raise SimulationStopped(self.stop_reason or "Simulation stopped")
        for arm in ARMS:
            phase = value.get("phases", {}).get(arm, "starting")
            if previous.get("phases", {}).get(arm) == phase:
                continue
            index = value.get("phase_indices", {}).get(arm, 0)
            total = value.get("total_phases", 16)
            percentage = min(94, 94 * index / max(1, total))
            reason = value.get("hold_reasons", {}).get(arm) or "Local phase advanced"
            self.mutate(arm, "report", report_value({
                "status": "running", "progress": percentage,
                "summary": f"{phase.replace('_', ' ').capitalize()} · {reason}",
                "evidence": [f"MuJoCo tick {value.get('tick', 0)}; actual Qwen + SpacetimeDB"],
            }))

    def validate_files(self):
        for name, relative, digest in [
            *((arm, CHECKPOINT_PATHS[arm], PINNED_HASHES[arm]) for arm in ARMS),
            ("communication codec", CODEC_PATH, CODEC_HASH),
        ]:
            path = self.root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"The reviewed {name} checkpoint hash does not match")

    def run(self):
        if self.state["phase"] == "done":
            return json.loads((self.directory / "result.json").read_text())
        if self.state["phase"] in ("executing", "verifying", "held"):
            self.state["phase"] = "held"
            self.save()
            return self.result(
                "interrupted", error="Interrupted simulation cannot safely resume; create a new task"
            )
        try:
            for arm in ARMS:
                worker = self.clients[arm].request("/api/agent/me")["worker"]
                if (
                    worker.get("agent_id") != arm
                    or worker.get("execution_kind") != "simulation"
                    or not re.fullmatch(r"worker-[a-f0-9]{16}", worker.get("id", ""))
                ):
                    raise ValueError("A distinct scoped simulation credential is required per arm")
                self.workers[arm] = worker
            if self.workers["left"]["id"] == self.workers["right"]["id"]:
                raise ValueError("The two arms require distinct runtime identities")
            if self.state["phase"] == "reporting":
                return self.finish_reports()
            self.validate_files()
            for arm in ARMS:
                self.check_task(self.current(arm))
            self.state["phase"] = "claiming"
            self.save()
            if self.state.get("pending"):
                pending = self.state["pending"]
                self.mutate(pending["arm"], pending["operation"], {})
            for arm in ARMS:
                self.mutate(arm, "claim", {})
            self.start_monitor()
            if self.output.exists():
                raise ValueError("Rollout output already exists; refusing to repeat a world")
            self.state["phase"] = "executing"
            self.save()
            if self.runner is None:
                from relay_lab.cloth.runtime import ClothConfig, run_cloth

                self.runner = run_cloth
                config = ClothConfig(**{**recipe_config(), "codec": str(self.root / CODEC_PATH)})
            else:
                config = recipe_config()
            summary = self.runner(
                config, self.output,
                checkpoint_paths={arm: self.root / CHECKPOINT_PATHS[arm] for arm in ARMS},
                expected_checkpoint_hashes=PINNED_HASHES,
                should_stop=self.should_stop, on_progress=self.progress,
            )
            self.state["summary"] = summary
            self.save()
            if self.should_stop() or summary.get("status") == "cancelled":
                raise SimulationStopped(self.stop_reason or "Simulation cancelled")
            verification = {"verified": False}
            if summary.get("success") is True and summary.get("metrics", {}).get("success") is True:
                self.state["phase"] = "verifying"
                self.save()
                self.result("verifying", summary=summary)
                verification = self.verifier(self.root, self.output, self.should_stop)
            if self.should_stop():
                raise SimulationStopped(self.stop_reason or "Simulation cancelled")
            succeeded = (
                summary.get("status") == "succeeded"
                and summary.get("success") is True
                and summary.get("metrics", {}).get("success") is True
                and verification.get("verified") is True
            )
            self.state.update(
                phase="reporting", summary=summary, verification=verification,
                final_status="succeeded" if succeeded else "failed", reported=[],
            )
            self.save()
            # The last geometry is frozen; reporting may race with the monitor seeing
            # successful completion, so stop only the monitor, retaining its stop flag.
            self.monitor_done.set()
            return self.finish_reports()
        except Exception as exc:  # noqa: BLE001 - a failed run must leave inspectable evidence
            self.monitor_done.set()
            error = f"{type(exc).__name__}: {exc}"[:1000]
            stopped = isinstance(exc, SimulationStopped) or self.stop_event.is_set()
            status = "cancelled" if stopped else "failed"
            if self.state["phase"] == "reporting" and not stopped:
                # Keep the exact computed result and pending envelope for reconciliation.
                return self.result("report_pending", error=error,
                                   summary=self.state.get("summary"),
                                   verification=self.state.get("verification"))
            self.state.update(phase="held", error=error)
            self.save()
            if not stopped and not self.state.get("pending") and self.workers:
                try:
                    self.mutate("left", "report", report_value({
                        "status": "failed", "progress": 0,
                        "summary": "Simulation stopped before verified completion.",
                        "evidence": [error[:500]],
                    }))
                except Exception:  # noqa: BLE001 - never retry physics to repair reporting
                    self.state["failure_report_unconfirmed"] = True
                    self.save()
            return self.result(status, error=error, summary=self.state.get("summary"))

    def finish_reports(self):
        status = self.state["final_status"]
        summary = self.state["summary"]
        evidence = [
            f"Measured MuJoCo simulation, {summary.get('ticks', 0)} physics ticks.",
            "Original task-trained state-assisted cloth controllers; not foundation VLAs.",
            "Independent action replay verified identical final geometry and metrics."
            if self.state["verification"].get("verified") else "Independent verification did not pass.",
            "Simulation evidence only; no physical hardware outcome is attested.",
        ]
        for arm in ARMS:
            if arm in self.state["reported"]:
                continue
            task = self.current(arm)
            if task["status"] in TERMINAL:
                if task["status"] == status and (
                    status == "failed" or all(
                        item["status"] == "succeeded" for item in task["assignments"].values()
                    )
                ):
                    break
                raise SimulationStopped(f"Workspace task is {task['status']}")
            self.mutate(arm, "report", report_value({
                "status": status, "progress": 100 if status == "succeeded" else 95,
                "summary": "Cloth folded and measured; fresh-world action replay verified."
                if status == "succeeded" else "Cloth task did not satisfy all verification checks.",
                "evidence": evidence,
            }))
            self.state["reported"].append(arm)
            self.save()
        value = self.result(status, summary=summary, verification=self.state["verification"])
        self.state["phase"] = "done"
        self.save()
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "task-id", "left-token-file", "right-token-file", "output", "root", "state-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--engine-lock-file")
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    os.chdir(root)
    runtime = TeamRuntime(
        {arm: AgentClient(args.server, read_token(getattr(args, arm + "_token_file"))) for arm in ARMS},
        args.task_id, args.output, root, args.state_dir, engine_lock_file=args.engine_lock_file,
    )
    try:
        result = runtime.run()
        print(json.dumps({"task_id": args.task_id, "status": result["status"]}))
        return 0 if result["status"] == "succeeded" else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
