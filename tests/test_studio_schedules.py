"""Scheduled objectives: SpacetimeDB fires runs; Studio re-checks governance."""

import hashlib
import json
import re
import threading
import time
from contextlib import contextmanager

import pytest

from relay_lab.spacetime import SpacetimeRejected, SpacetimeUnavailable
from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.schedules import ScheduleService
from relay_lab.studio.tasks import TaskConflict

MANAGER = {"id": "owner", "role": "manager", "active": True}
EMPLOYEE = {"id": "operator", "role": "employee", "active": True}
OTHER = {"id": "someone-else", "role": "employee", "active": True}


class FakeDatabase:
    """The schedule tables and the owner reducers Studio calls (scheduler fires via ``fire``)."""

    def __init__(self):
        self.schedules, self.dispatch, self.calls = {}, {}, []
        self.reject_sql = False

    def _sql(self, query):
        if self.reject_sql:
            raise SpacetimeRejected("Spacetime request rejected (HTTP 400)")
        table = re.search(r"FROM (\w+)", query)[1]
        rows = {
            "task_schedule": self.schedules,
            "task_dispatch": self.dispatch,
        }[table].values()
        for column, value in re.findall(r"(\w+) = '([^']+)'", query):
            rows = [row for row in rows if row[column] == value]
        return [dict(row) for row in rows]

    def _mutate(self, reducer, args, readback, expected):
        self.calls.append(reducer)
        getattr(self, reducer)(*args)
        row = readback()
        assert row is not None and all(row.get(k) == v for k, v in expected.items())
        return row

    def create_task_schedule(self, key, worker, creator, request_json, first, interval, runs):
        assert key not in self.schedules
        self.schedules[key] = {
            "schedule_key": key,
            "worker_id": worker,
            "created_by": creator,
            "request_json": request_json,
            "first_due_micros": first,
            "interval_micros": interval,
            "max_runs": runs,
            "fired_runs": 0,
            "skipped_runs": 0,
            "next_due_micros": first,
            "last_fired_micros": 0,
            "status": "active",
            "reason": "",
            "created_micros": first,
            "updated_micros": first,
        }

    def fire(self, key):
        schedule = self.schedules[key]
        run = schedule["fired_runs"] + 1
        pending = any(
            d["status"] == "pending" and d["schedule_key"] == key for d in self.dispatch.values()
        )
        self.dispatch[f"{key}-r{run}"] = {
            "dispatch_key": f"{key}-r{run}",
            "schedule_key": key,
            "worker_id": schedule["worker_id"],
            "run_number": run,
            "due_micros": schedule["next_due_micros"],
            "fired_micros": 1,
            "status": "skipped" if pending else "pending",
            "task_id": "",
            "reason": "",
            "updated_micros": 1,
        }
        finished = schedule["interval_micros"] == 0 or run >= schedule["max_runs"]
        schedule.update(
            fired_runs=run,
            status="completed" if finished else "active",
            next_due_micros=0
            if finished
            else schedule["next_due_micros"] + schedule["interval_micros"],
        )

    def cancel_task_schedule(self, key, worker, reason):
        self.schedules[key].update(status="cancelled", reason=reason, next_due_micros=0)

    def resolve_task_dispatch(self, key, worker, status, task_id, reason):
        assert self.dispatch[key]["status"] in ("pending", status)
        self.dispatch[key].update(status=status, task_id=task_id, reason=reason)

    def remove_task_schedule(self, key, worker):
        self.schedules.pop(key, None)
        for dispatch_key in [k for k, d in self.dispatch.items() if d["schedule_key"] == key]:
            del self.dispatch[dispatch_key]


class FakeTasks:
    worker_id = "studio-goals-abc123abc123"

    def __init__(self, database):
        self.database, self.created, self.release_error, self.create_error = (
            database,
            {},
            None,
            None,
        )
        self.registry = self
        self.registry_id = "registry-1"
        self.studio = self

    # registry / studio surface used by ScheduleService
    def authorize_workspace(self, actor, workspace):
        if self.release_error:
            raise PermissionError(self.release_error)
        return {"models": []}

    def validate(self, workspace, complete=False):
        return workspace

    def workspace(self):
        return {"agents": []}

    def factory(self):
        database = self.database

        class Coordinator:
            client = database

        return Coordinator()

    def create(self, actor, value):
        if self.create_error:
            raise self.create_error
        self.authorize_workspace(actor, None)
        task_id = "goal-" + hashlib.sha256(value["request_id"].encode()).hexdigest()[:16]
        self.created.setdefault(
            task_id, {"id": task_id, "created_at": 1_800_000_000.0, "actor": actor["id"], **value}
        )
        return self.created[task_id]


class FakeAccess:
    def __init__(self):
        self.inactive = set()

    def authorize_actor(self, actor, permission=None):
        if actor["id"] in self.inactive:
            raise AuthenticationError("Sign in to continue")
        return {"id": actor["id"], "role": "manager", "active": True}


@pytest.fixture
def env():
    database = FakeDatabase()
    tasks = FakeTasks(database)
    access = FakeAccess()
    return database, tasks, access, ScheduleService(tasks, access)


def request(**extra):
    value = {
        "request_id": "req-1",
        "objective": "Fold the shirt.",
        "success_criteria": ["Shirt folded"],
        "target": "both",
        "start_at": time.time() + 60,
    }
    value.update(extra)
    return value


def test_create_validates_timing_and_objective(env):
    database, _, _, service = env
    bad = [
        {"start_at": time.time() - 3600},
        {"start_at": time.time() + 31 * 86400},
        {"start_at": float("nan")},
        {"start_at": True},
        {"repeat_minutes": 7},
        {"repeat_minutes": 5, "max_runs": 1},
        {"repeat_minutes": 5, "max_runs": 21},
        {"repeat_minutes": 0, "max_runs": 2},
        {"timeout_seconds": 30},
        {"timeout_seconds": 1800.0},
        {"success_criteria": []},
        {"target": "middle"},
    ]
    for extra in bad:
        with pytest.raises(ValueError):
            service.create(MANAGER, request(**extra))
    assert database.calls == []


def test_create_checks_releases_before_scheduling(env):
    database, tasks, _, service = env
    tasks.release_error = "This exact model configuration has no current release"
    with pytest.raises(PermissionError, match="no current release"):
        service.create(MANAGER, request())
    assert database.calls == []


def test_create_is_idempotent_and_rejects_reused_ids(env):
    database, _, _, service = env
    first = service.create(MANAGER, request(repeat_minutes=60, max_runs=3, timeout_seconds=900))
    assert first["status"] == "active" and first["repeat_minutes"] == 60 and first["max_runs"] == 3
    assert first["timeout_seconds"] == 900 and first["key"].startswith("sched-")
    again = service.create(MANAGER, request(repeat_minutes=60, max_runs=3, timeout_seconds=900))
    assert again["key"] == first["key"] and database.calls.count("create_task_schedule") == 1
    with pytest.raises(TaskConflict):
        service.create(MANAGER, request(objective="Something else."))


def test_fired_run_creates_objective_with_its_attention_deadline(env):
    database, tasks, _, service = env
    schedule = service.create(EMPLOYEE, request(timeout_seconds=300))
    database.fire(schedule["key"])
    assert service.process_due() == 1
    run = database.dispatch[f"{schedule['key']}-r1"]
    assert run["status"] == "created" and run["task_id"] in tasks.created
    task = tasks.created[run["task_id"]]
    # The dispatch key is the request_id, so a retried run maps to the same objective.
    assert task["request_id"] == run["dispatch_key"] and task["actor"] == "operator"
    # The coordination watchdog tracks this deadline like any other task's.
    assert task["timeout_seconds"] == 300
    assert service.process_due() == 0  # nothing left to do


def test_schedules_without_a_deadline_use_the_task_default(env):
    database, tasks, _, service = env
    schedule = service.create(MANAGER, request())
    assert schedule["timeout_seconds"] == 1800
    database.fire(schedule["key"])
    service.process_due()
    assert next(iter(tasks.created.values()))["timeout_seconds"] == 1800


def test_run_is_blocked_when_releases_change_or_account_is_inactive(env):
    database, tasks, access, service = env
    first = service.create(MANAGER, request(request_id="a"))
    second = service.create(MANAGER, request(request_id="b"))
    tasks.release_error = "This exact model configuration has no current release"
    database.fire(first["key"])
    service.process_due()
    blocked = database.dispatch[f"{first['key']}-r1"]
    assert blocked["status"] == "blocked" and "no current release" in blocked["reason"]
    tasks.release_error = None
    access.inactive.add("owner")
    database.fire(second["key"])
    service.process_due()
    blocked = database.dispatch[f"{second['key']}-r1"]
    assert blocked["status"] == "blocked" and "active account" in blocked["reason"]
    assert tasks.created == {}


def test_database_outage_leaves_run_pending_for_retry(env):
    database, tasks, _, service = env
    schedule = service.create(MANAGER, request())
    database.fire(schedule["key"])
    tasks.create_error = SpacetimeUnavailable(
        "Spacetime request did not receive a confirmed response"
    )
    assert service.process_due() == 0
    assert database.dispatch[f"{schedule['key']}-r1"]["status"] == "pending"
    assert "confirmed response" in service.last_error
    tasks.create_error = None
    assert service.process_due() == 1 and service.last_error is None
    assert database.dispatch[f"{schedule['key']}-r1"]["status"] == "created"


def test_cancel_and_remove_permissions_and_states(env):
    database, _, _, service = env
    schedule = service.create(EMPLOYEE, request(repeat_minutes=5, max_runs=2))
    with pytest.raises(PermissionError):
        service.cancel(OTHER, schedule["key"], {})
    with pytest.raises(TaskConflict, match="Cancel"):
        service.remove(EMPLOYEE, schedule["key"])
    database.fire(schedule["key"])  # a run is waiting for Studio
    cancelled = service.cancel(EMPLOYEE, schedule["key"], {"reason": "No longer needed"})
    assert cancelled["status"] == "cancelled" and cancelled["reason"] == "No longer needed"
    assert service.cancel(MANAGER, schedule["key"], {})["status"] == "cancelled"  # idempotent
    with pytest.raises(TaskConflict, match="waiting"):
        service.remove(EMPLOYEE, schedule["key"])
    service.process_due()
    assert service.remove(MANAGER, schedule["key"]) == {"key": schedule["key"], "removed": True}
    assert schedule["key"] not in database.schedules and database.dispatch == {}
    assert service.remove(MANAGER, schedule["key"])["removed"] is True  # idempotent


def test_completed_schedule_cannot_be_cancelled(env):
    database, _, _, service = env
    schedule = service.create(MANAGER, request())
    database.fire(schedule["key"])
    service.process_due()
    with pytest.raises(TaskConflict, match="finished"):
        service.cancel(MANAGER, schedule["key"], {})


def test_list_reports_runs_and_degrades_without_module(env):
    database, _, _, service = env
    schedule = service.create(MANAGER, request(repeat_minutes=15, max_runs=2, timeout_seconds=3600))
    database.fire(schedule["key"])
    listed = service.list(MANAGER)  # also processes the due run
    assert listed["available"] is True and listed["backend"] == "SpacetimeDB scheduled reducers"
    item = listed["schedules"][0]
    assert item["fired_runs"] == 1 and item["runs"][0]["status"] == "created"
    assert item["timeout_seconds"] == 3600
    assert (
        json.loads(database.schedules[schedule["key"]]["request_json"])["timeout_seconds"] == 3600
    )
    database.reject_sql = True
    degraded = service.list(MANAGER)
    assert degraded["available"] is False and "publish" in degraded["reason"]
    assert service.process_due() == 0  # an unpublished module is not an error


class FakeDurable:
    """Durable deployments restore Studio state inside a fenced request."""

    def __init__(self):
        self.lock, self.holder, self.requests = threading.RLock(), None, 0

    @contextmanager
    def request(self):
        with self.lock:
            self.holder, self.requests = "req-1", self.requests + 1
            try:
                yield
            finally:
                self.holder = None


def test_durable_background_pass_takes_a_request_only_when_runs_are_due(env):
    database, tasks, access, _ = env
    durable = FakeDurable()
    service = ScheduleService(tasks, access, durable=durable)
    assert service.run_once() == 0 and durable.requests == 0
    schedule = service.create(MANAGER, request())
    database.fire(schedule["key"])
    assert service.run_once() == 1 and durable.requests == 1
    assert database.dispatch[f"{schedule['key']}-r1"]["status"] == "created"


def test_background_dispatcher_starts_and_stops(env, monkeypatch):
    database, tasks, _, service = env
    monkeypatch.setattr("relay_lab.studio.schedules.PUMP_INTERVAL_SECONDS", 0.01)
    schedule = service.create(MANAGER, request())
    database.fire(schedule["key"])
    service.start()
    deadline = time.time() + 2
    while not tasks.created and time.time() < deadline:
        time.sleep(0.01)
    service.close()
    assert tasks.created and not service.thread.is_alive()
