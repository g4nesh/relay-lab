"""Scheduled and repeating objectives, timed by SpacetimeDB scheduled reducers.

SpacetimeDB owns the clock: a timer row fires ``fire_task_schedule`` at each due
time, even while Studio is offline, and that reducer only enqueues a due
dispatch. Studio then re-checks the creator's account and the current model
releases, creates the objective through the normal ``TaskService`` path, and
records the outcome (created or blocked) back in SpacetimeDB. Each created
objective carries the schedule's attention deadline, which the coordination
watchdog tracks like any other task. No reducer here writes the objective
journal or bypasses governance.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time

from relay_lab.factory.coordination import canonical
from relay_lab.spacetime import SpacetimeError, SpacetimeRejected

from .access import AuthenticationError, AuthorizationError
from .governance import _actor, _text
from .tasks import TaskConflict, _request, _signature, _strings

REPEAT_MINUTES = (0, 5, 15, 30, 60, 360, 1440)
MAX_RUNS = 20
MAX_LEAD_SECONDS = 30 * 86400
DEFAULT_TIMEOUT_SECONDS = 1800
LIST_LIMIT = 500
PUMP_INTERVAL_SECONDS = 5.0
TIMER = "SpacetimeDB scheduled reducer"
UNAVAILABLE = (
    "Scheduling needs the updated SpacetimeDB module. Run `python3 scripts/spacetime.py publish`."
)
_KEY = re.compile(r"sched-[a-f0-9]{16}")


class ScheduleUnavailable(RuntimeError):
    """The connected database does not expose the schedule tables yet."""


def _micros(seconds):
    return round(seconds * 1_000_000)


def _seconds(micros):
    return micros / 1_000_000 if micros else None


def _objective(value):
    """The same objective fields and limits that ``TaskService.create`` enforces."""
    objective = value.get("objective")
    _text(objective, "Objective", 2000)
    criteria = _strings(value.get("success_criteria"), "Success criteria", required=True)
    title = _text(value.get("title") or objective.strip().splitlines()[0][:100], "Title", 100)
    target = value.get("target", "both")
    if target not in ("left", "right", "both"):
        raise ValueError("Target must be left, right, or both")
    timeout = value.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if type(timeout) is not int or not 60 <= timeout <= 86400:
        raise ValueError("Task attention deadline must be 60–86400 seconds")
    return {
        "title": title,
        "objective": objective,
        "success_criteria": criteria,
        "target": target,
        "timeout_seconds": timeout,
    }


class ScheduleService:
    def __init__(self, tasks, access, *, durable=None):
        self.tasks, self.access, self.durable = tasks, access, durable
        self.lock = threading.Lock()
        self.last_error = None
        self.stop = threading.Event()
        self.thread = None

    @property
    def worker_id(self):
        return self.tasks.worker_id

    def _client(self):
        return self.tasks.studio.factory().client

    def _rows(self, client, query):
        try:
            return client._sql(query)
        except SpacetimeRejected:
            # An older published module has no schedule tables; SQL rejects the query.
            raise ScheduleUnavailable(UNAVAILABLE) from None

    def _schedule(self, client, key):
        rows = self._rows(client, f"SELECT * FROM task_schedule WHERE schedule_key = '{key}'")
        return rows[0] if rows else None

    def _dispatch(self, client, key):
        rows = self._rows(client, f"SELECT * FROM task_dispatch WHERE dispatch_key = '{key}'")
        return rows[0] if rows else None

    def _pending(self, client):
        return self._rows(
            client,
            f"SELECT * FROM task_dispatch WHERE worker_id = '{self.worker_id}' "
            "AND status = 'pending' LIMIT 20",
        )

    @staticmethod
    def _public(row, runs):
        request = json.loads(row["request_json"])
        return {
            "key": row["schedule_key"],
            "title": request["title"],
            "objective": request["objective"],
            "target": request["target"],
            "timeout_seconds": request.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            "created_by": row["created_by"],
            "status": row["status"],
            "reason": row["reason"],
            "repeat_minutes": row["interval_micros"] // 60_000_000,
            "max_runs": row["max_runs"],
            "fired_runs": row["fired_runs"],
            "skipped_runs": row["skipped_runs"],
            "first_due_at": _seconds(row["first_due_micros"]),
            "next_due_at": _seconds(row["next_due_micros"]),
            "last_fired_at": _seconds(row["last_fired_micros"]),
            "created_at": _seconds(row["created_micros"]),
            "runs": [
                {
                    "run": run["run_number"],
                    "due_at": _seconds(run["due_micros"]),
                    "fired_at": _seconds(run["fired_micros"]),
                    "status": run["status"],
                    "task_id": run["task_id"] or None,
                    "reason": run["reason"],
                }
                for run in sorted(runs, key=lambda r: r["run_number"], reverse=True)
            ],
            "timer": TIMER,
        }

    # ---- reads ----
    def list(self, actor):
        _actor(actor)
        self.process_due(blocking=False)
        client = self._client()
        worker = self.worker_id
        try:
            schedules = self._rows(
                client,
                f"SELECT * FROM task_schedule WHERE worker_id = '{worker}' LIMIT {LIST_LIMIT}",
            )
            runs = self._rows(
                client,
                f"SELECT * FROM task_dispatch WHERE worker_id = '{worker}' LIMIT {LIST_LIMIT}",
            )
        except ScheduleUnavailable as error:
            return {"available": False, "reason": str(error), "schedules": []}
        grouped = {}
        for run in runs:
            grouped.setdefault(run["schedule_key"], []).append(run)
        public = [self._public(row, grouped.get(row["schedule_key"], [])) for row in schedules]
        public.sort(key=lambda s: (s["status"] != "active", -(s["created_at"] or 0)))
        return {
            "available": True,
            "backend": "SpacetimeDB scheduled reducers",
            "schedules": public,
            "repeat_minutes": list(REPEAT_MINUTES[1:]),
            "max_runs": MAX_RUNS,
            "pump_error": self.last_error,
        }

    # ---- writes ----
    def create(self, actor, value):
        actor = _actor(actor)
        request_id = _request(value)
        normalized = _objective(value)
        start_at = value.get("start_at")
        now = time.time()
        if (
            not isinstance(start_at, (int, float))
            or isinstance(start_at, bool)
            or not math.isfinite(start_at)
        ):
            raise ValueError("Choose a start time")
        repeat = value.get("repeat_minutes", 0)
        if type(repeat) is not int or repeat not in REPEAT_MINUTES:
            raise ValueError("Repeat must be every 5, 15, 30 or 60 minutes, 6 hours, or 24 hours")
        runs = value.get("max_runs", 1 if repeat == 0 else 3)
        if type(runs) is not int or (runs != 1 if repeat == 0 else not 2 <= runs <= MAX_RUNS):
            raise ValueError(
                "A one-time schedule runs once; a repeating schedule runs 2 to 20 times"
            )
        key = "sched-" + _signature([self.tasks.registry.registry_id, actor["id"], request_id])[:16]
        requested_start = _micros(start_at)
        # Keep the original intent even when an immediate start is clamped to now.
        request_json = canonical({**normalized, "requested_start_micros": requested_start})
        client = self._client()
        existing = self._schedule(client, key)
        if existing is not None:
            previous = json.loads(existing["request_json"])
            previous_start = previous.pop("requested_start_micros", existing["first_due_micros"])
            previous.setdefault("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
            if (
                canonical(previous) != canonical(normalized)
                or previous_start != requested_start
                or existing["interval_micros"] != repeat * 60_000_000
                or existing["max_runs"] != runs
            ):
                raise TaskConflict("This schedule request_id was reused for different content")
            # Reconcile a prior commit before checking current time or releases.
            # Its dispatch still rechecks current authority before creating a task.
            return self._public(existing, [])
        if not now - 60 <= start_at <= now + MAX_LEAD_SECONDS:
            raise ValueError("Start time must be between now and 30 days from now")
        studio = self.tasks.studio
        self.tasks.registry.authorize_workspace(
            actor, studio.validate(studio.workspace(), complete=True)
        )
        args = [
            key,
            self.worker_id,
            actor["id"],
            request_json,
            _micros(max(start_at, now)),
            repeat * 60_000_000,
            runs,
        ]
        row = client._mutate(
            "create_task_schedule",
            args,
            lambda: self._schedule(client, key),
            {"schedule_key": key, "worker_id": self.worker_id, "request_json": request_json},
        )
        return self._public(row, [])

    def _owned(self, actor, client, key, verb):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Unknown schedule")
        row = self._schedule(client, key)
        if row is not None and row["worker_id"] != self.worker_id:
            raise ValueError("Unknown schedule")
        if row is not None and actor["role"] != "manager" and row["created_by"] != actor["id"]:
            raise PermissionError(f"Only a manager or the schedule's creator can {verb} it")
        return row

    def cancel(self, actor, key, value):
        # API handlers hold registry.lock already. Keep the same order in the
        # background dispatcher so cancellation cannot race objective creation.
        with self.tasks.registry.lock, self.lock:
            return self._cancel(actor, key, value)

    def _cancel(self, actor, key, value):
        actor = _actor(actor)
        reason = _text((value or {}).get("reason") or "Cancelled in Studio", "Reason", 1000)
        client = self._client()
        row = self._owned(actor, client, key, "cancel")
        if row is None:
            raise ValueError("Unknown schedule")
        if row["status"] == "active":
            row = client._mutate(
                "cancel_task_schedule",
                [key, self.worker_id, reason],
                lambda: self._schedule(client, key),
                {"schedule_key": key, "status": "cancelled"},
            )
        elif row["status"] != "cancelled":
            raise TaskConflict("This schedule has already finished")
        runs = self._rows(client, f"SELECT * FROM task_dispatch WHERE schedule_key = '{key}'")
        return self._public(row, runs)

    def remove(self, actor, key):
        """Delete a finished schedule's records; objectives it created are kept."""
        actor = _actor(actor)
        client = self._client()
        row = self._owned(actor, client, key, "remove")
        if row is None:
            return {"key": key, "removed": True}
        if row["status"] == "active":
            raise TaskConflict("Cancel this schedule before removing it")
        runs = self._rows(client, f"SELECT * FROM task_dispatch WHERE schedule_key = '{key}'")
        if any(run["status"] == "pending" for run in runs):
            raise TaskConflict("A run is still waiting for Studio; try again in a few seconds")
        client._mutate(
            "remove_task_schedule",
            [key, self.worker_id],
            lambda: {"removed": self._schedule(client, key) is None},
            {"removed": True},
        )
        return {"key": key, "removed": True}

    # ---- dispatch ----
    def _resolve(self, client, dispatch, status, task_id, reason):
        client._mutate(
            "resolve_task_dispatch",
            [dispatch["dispatch_key"], self.worker_id, status, task_id, reason],
            lambda: self._dispatch(client, dispatch["dispatch_key"]),
            {"dispatch_key": dispatch["dispatch_key"], "status": status, "task_id": task_id},
        )

    def process_due(self, *, blocking=True):
        """Create objectives for runs SpacetimeDB has fired. Safe to call repeatedly."""
        if not self.tasks.registry.lock.acquire(blocking=blocking):
            return 0
        try:
            return self._process_due(blocking=blocking)
        finally:
            self.tasks.registry.lock.release()

    def _process_due(self, *, blocking):
        if not self.lock.acquire(blocking=blocking):
            return 0
        try:
            client = self._client()
            try:
                due = self._pending(client)
            except ScheduleUnavailable:
                return 0
            handled = 0
            for dispatch in sorted(due, key=lambda d: (d["due_micros"], d["dispatch_key"])):
                if self.stop.is_set():
                    break
                schedule = self._schedule(client, dispatch["schedule_key"])
                if schedule is None:
                    self._resolve(client, dispatch, "blocked", "", "Schedule record is missing")
                    continue
                request = json.loads(schedule["request_json"])
                # A previous pass may have committed the task but lost the
                # dispatch acknowledgement. Reconcile it before cancellation or
                # revocation can incorrectly report that no objective exists.
                task_id = "goal-" + _signature(
                    [self.tasks.registry.registry_id, schedule["created_by"], dispatch["dispatch_key"]]
                )[:16]
                committed = client._sql(
                    f"SELECT run_id, state_json FROM factory_run WHERE run_id = '{task_id}' "
                    f"AND worker_id = '{self.worker_id}'"
                )
                if committed:
                    state = json.loads(committed[0]["state_json"])
                    if (
                        state.get("task") != "studio-objective-v1"
                        or state.get("registry_id") != self.tasks.registry.registry_id
                        or state.get("create_signature") != _signature(_objective(request))
                    ):
                        self._resolve(
                            client, dispatch, "blocked", "",
                            "This dispatch identifier already belongs to different task content",
                        )
                        continue
                    self._resolve(client, dispatch, "created", task_id, "Objective created")
                    handled += 1
                    continue
                if schedule["status"] == "cancelled":
                    self._resolve(client, dispatch, "blocked", "", "Schedule was cancelled")
                    continue
                try:
                    actor = self.access.authorize_actor(
                        {"id": schedule["created_by"]}, "sessions.run"
                    )
                    # The dispatch key is the request_id, so a retried run maps to the same objective.
                    task = self.tasks.create(
                        actor,
                        {
                            "request_id": dispatch["dispatch_key"],
                            **{
                                k: request[k]
                                for k in ("title", "objective", "success_criteria", "target")
                            },
                            "timeout_seconds": request.get(
                                "timeout_seconds", DEFAULT_TIMEOUT_SECONDS
                            ),
                        },
                    )
                except (AuthenticationError, AuthorizationError):
                    self._resolve(
                        client,
                        dispatch,
                        "blocked",
                        "",
                        "The schedule's creator no longer has an active account that can run tasks",
                    )
                    continue
                except (PermissionError, ValueError) as error:
                    # Includes retired or unreleased models and full task history (TaskConflict).
                    self._resolve(client, dispatch, "blocked", "", (str(error) or "Blocked")[:500])
                    continue
                self._resolve(client, dispatch, "created", task["id"], "Objective created")
                handled += 1
            self.last_error = None
            return handled
        except SpacetimeError as error:
            # Leave pending runs in place; the next pass retries the same request_id.
            self.last_error = str(error)
            return 0
        finally:
            self.lock.release()

    # ---- background pass ----
    def run_once(self):
        """One background pass. Durable deployments restore Studio state first."""
        if self.durable is None or self.durable.holder:
            return self.process_due()
        try:
            # SQL reads are not fenced; take the request lease only when work is due.
            if not self._pending(self._client()):
                return 0
        except ScheduleUnavailable:
            return 0
        while not self.stop.is_set():
            if self.durable.lock.acquire(timeout=0.2):
                break
        else:
            return 0
        try:
            with self.durable.request():
                return self.process_due()
        finally:
            self.durable.lock.release()

    def _run(self):
        while not self.stop.wait(PUMP_INTERVAL_SECONDS):
            try:
                self.run_once()
            except Exception as error:  # noqa: BLE001 - keep the loop alive; surface in the UI
                self.last_error = f"{type(error).__name__}: {error}"[:300]

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="studio-schedules", daemon=True)
            self.thread.start()

    def request_stop(self):
        self.stop.set()

    def close(self):
        self.request_stop()
        if self.thread is not None:
            # A pass is bounded by Spacetime request timeouts and stops between runs.
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise RuntimeError("Schedule dispatcher did not finish shutdown")
