"""Scheduled objectives and deadlines, timed by SpacetimeDB scheduled reducers.

SpacetimeDB owns the clock: a timer row fires ``fire_task_schedule`` at each due
time, even while Studio is offline, and that reducer only enqueues a due
dispatch. Studio then re-checks the creator's account and the current model
releases, creates the objective through the normal ``TaskService`` path, and
records the outcome (created or blocked) back in SpacetimeDB. Deadlines work
the same way: ``check_task_deadline`` reads the objective journal at the
deadline and records whether it finished in time. No reducer here writes the
objective journal or bypasses governance.
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
DEADLINE_MINUTES = (5, 15, 30, 60, 240, 1440)
LIST_LIMIT = 500
PUMP_INTERVAL_SECONDS = 5.0
UNAVAILABLE = (
    "Scheduling needs the updated SpacetimeDB module. Run `python3 scripts/spacetime.py publish`."
)
_KEY = re.compile(r"sched-[a-f0-9]{16}")
_TASK = re.compile(r"goal-[a-f0-9]{16}")


class ScheduleUnavailable(RuntimeError):
    """The connected database does not expose the schedule tables yet."""


def _micros(seconds):
    return round(seconds * 1_000_000)


def _seconds(micros):
    return micros / 1_000_000 if micros else None


def deadline_minutes(value):
    """Validate an optional per-objective deadline before anything is created."""
    minutes = value.get("deadline_minutes") if isinstance(value, dict) else None
    if minutes is None:
        return None
    if type(minutes) is not int or minutes not in DEADLINE_MINUTES:
        raise ValueError("Deadline must be 5, 15, 30 or 60 minutes, 4 hours, or 24 hours")
    return minutes


def _objective(value):
    """The same objective fields and limits that ``TaskService.create`` enforces."""
    objective = value.get("objective")
    _text(objective, "Objective", 2000)
    criteria = _strings(value.get("success_criteria"), "Success criteria", required=True)
    title = _text(value.get("title") or objective.strip().splitlines()[0][:100], "Title", 100)
    target = value.get("target", "both")
    if target not in ("left", "right", "both"):
        raise ValueError("Target must be left, right, or both")
    return {"title": title, "objective": objective, "success_criteria": criteria, "target": target}


class ScheduleService:
    def __init__(self, tasks, access):
        self.tasks, self.access = tasks, access
        self.lock = threading.Lock()
        self.last_error = None

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

    def _deadline_row(self, client, task_id):
        rows = self._rows(client, f"SELECT * FROM task_deadline WHERE task_id = '{task_id}'")
        return rows[0] if rows else None

    @staticmethod
    def _public_deadline(row):
        return {
            "due_at": _seconds(row["deadline_micros"]),
            "status": row["status"],
            "detail": row["detail"],
            "checked_at": _seconds(row["checked_micros"]),
            "timer": "SpacetimeDB scheduled reducer",
        }

    @staticmethod
    def _public(row, runs):
        request = json.loads(row["request_json"])
        return {
            "key": row["schedule_key"],
            "title": request["title"],
            "objective": request["objective"],
            "target": request["target"],
            "deadline_minutes": request.get("deadline_minutes"),
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
            "timer": "SpacetimeDB scheduled reducer",
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
            deadlines = self._rows(
                client,
                f"SELECT * FROM task_deadline WHERE worker_id = '{worker}' LIMIT {LIST_LIMIT}",
            )
        except ScheduleUnavailable as error:
            return {"available": False, "reason": str(error), "schedules": [], "deadlines": {}}
        grouped = {}
        for run in runs:
            grouped.setdefault(run["schedule_key"], []).append(run)
        public = [self._public(row, grouped.get(row["schedule_key"], [])) for row in schedules]
        public.sort(key=lambda s: (s["status"] != "active", -(s["created_at"] or 0)))
        return {
            "available": True,
            "backend": "SpacetimeDB scheduled reducers",
            "schedules": public,
            "deadlines": {row["task_id"]: self._public_deadline(row) for row in deadlines},
            "repeat_minutes": list(REPEAT_MINUTES[1:]),
            "deadline_minutes": list(DEADLINE_MINUTES),
            "max_runs": MAX_RUNS,
            "pump_error": self.last_error,
        }

    # ---- writes ----
    def create(self, actor, value):
        actor = _actor(actor)
        request_id = _request(value)
        normalized = _objective(value)
        deadline = deadline_minutes(value)
        start_at = value.get("start_at")
        now = time.time()
        if (
            not isinstance(start_at, (int, float))
            or isinstance(start_at, bool)
            or not math.isfinite(start_at)
        ):
            raise ValueError("Choose a start time")
        if not now - 60 <= start_at <= now + MAX_LEAD_SECONDS:
            raise ValueError("Start time must be between now and 30 days from now")
        repeat = value.get("repeat_minutes", 0)
        if type(repeat) is not int or repeat not in REPEAT_MINUTES:
            raise ValueError("Repeat must be every 5, 15, 30 or 60 minutes, 6 hours, or 24 hours")
        runs = value.get("max_runs", 1 if repeat == 0 else 3)
        if type(runs) is not int or (runs != 1 if repeat == 0 else not 2 <= runs <= MAX_RUNS):
            raise ValueError(
                "A one-time schedule runs once; a repeating schedule runs 2 to 20 times"
            )
        # Fail early with the same release check every dispatch repeats.
        studio = self.tasks.studio
        self.tasks.registry.authorize_workspace(
            actor, studio.validate(studio.workspace(), complete=True)
        )
        key = "sched-" + _signature([self.tasks.registry.registry_id, actor["id"], request_id])[:16]
        request_json = canonical({**normalized, "deadline_minutes": deadline})
        client = self._client()
        args = [
            key,
            self.worker_id,
            actor["id"],
            request_json,
            _micros(max(start_at, now)),
            repeat * 60_000_000,
            runs,
        ]
        existing = self._schedule(client, key)
        if existing is not None and existing["request_json"] != request_json:
            raise TaskConflict("This schedule request_id was reused for different content")
        if existing is None:
            # A retried request keeps its original start time; only new keys are sent.
            row = client._mutate(
                "create_task_schedule",
                args,
                lambda: self._schedule(client, key),
                {"schedule_key": key, "worker_id": self.worker_id, "request_json": request_json},
            )
        else:
            row = existing
        return self._public(row, [])

    def cancel(self, actor, key, value):
        actor = _actor(actor)
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Unknown schedule")
        reason = _text((value or {}).get("reason") or "Cancelled in Studio", "Reason", 1000)
        client = self._client()
        row = self._schedule(client, key)
        if row is None or row["worker_id"] != self.worker_id:
            raise ValueError("Unknown schedule")
        if actor["role"] != "manager" and row["created_by"] != actor["id"]:
            raise PermissionError("Only a manager or the schedule's creator can cancel it")
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
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Unknown schedule")
        client = self._client()
        row = self._schedule(client, key)
        if row is None:
            return {"key": key, "removed": True}
        if row["worker_id"] != self.worker_id:
            raise ValueError("Unknown schedule")
        if actor["role"] != "manager" and row["created_by"] != actor["id"]:
            raise PermissionError("Only a manager or the schedule's creator can remove it")
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

    def set_deadline(self, task, minutes):
        """Arm the deadline watchdog for an objective; deterministic for retries."""
        task_id = task["id"]
        if not _TASK.fullmatch(task_id) or minutes not in DEADLINE_MINUTES:
            raise ValueError("Invalid objective deadline")
        client = self._client()
        due = _micros(task["created_at"] + minutes * 60)
        row = client._mutate(
            "set_task_deadline",
            [task_id, self.worker_id, due],
            lambda: self._deadline_row(client, task_id),
            {"task_id": task_id, "deadline_micros": due},
        )
        return self._public_deadline(row)

    def clear_deadline(self, actor, task_id):
        """Disarm and delete an objective's deadline watchdog (manager only)."""
        _actor(actor, manager=True)
        if not isinstance(task_id, str) or not _TASK.fullmatch(task_id):
            raise ValueError("Unknown objective")
        client = self._client()
        client._mutate(
            "clear_task_deadline",
            [task_id, self.worker_id],
            lambda: {"cleared": self._deadline_row(client, task_id) is None},
            {"cleared": True},
        )
        return {"task_id": task_id, "cleared": True}

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
        if not self.lock.acquire(blocking=blocking):
            return 0
        try:
            client = self._client()
            try:
                due = self._rows(
                    client,
                    f"SELECT * FROM task_dispatch WHERE worker_id = '{self.worker_id}' "
                    f"AND status = 'pending' LIMIT 20",
                )
            except ScheduleUnavailable:
                return 0
            handled = 0
            for dispatch in sorted(due, key=lambda d: (d["due_micros"], d["dispatch_key"])):
                schedule = self._schedule(client, dispatch["schedule_key"])
                if schedule is None:
                    self._resolve(client, dispatch, "blocked", "", "Schedule record is missing")
                    continue
                request = json.loads(schedule["request_json"])
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
                if request.get("deadline_minutes"):
                    try:
                        self.set_deadline(task, request["deadline_minutes"])
                    except (SpacetimeError, ValueError):
                        pass  # The objective exists; a missed watchdog is not a failed run.
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


class SchedulePump(threading.Thread):
    """Background loop so fired runs become objectives even with no browser open."""

    def __init__(self, service, interval=PUMP_INTERVAL_SECONDS):
        super().__init__(name="relay-schedule-pump", daemon=True)
        self.service, self.interval = service, interval
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.wait(self.interval):
            try:
                self.service.process_due()
            except Exception as error:  # noqa: BLE001 - keep the loop alive; surface in the UI
                self.service.last_error = f"{type(error).__name__}: {error}"[:300]

    def stop(self):
        self.stopped.set()
