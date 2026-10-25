"""Scheduled objectives: SpacetimeDB fires runs; Studio re-checks governance."""

import hashlib
import json
import re
import time

import pytest

from relay_lab.spacetime import SpacetimeRejected, SpacetimeUnavailable
from relay_lab.studio.access import AuthenticationError
from relay_lab.studio.schedules import ScheduleService, deadline_minutes
from relay_lab.studio.tasks import TaskConflict

MANAGER = {"id": "owner", "role": "manager", "active": True}
EMPLOYEE = {"id": "operator", "role": "employee", "active": True}
OTHER = {"id": "someone-else", "role": "employee", "active": True}


class FakeDatabase:
    """The schedule tables and the owner reducers Studio calls (scheduler fires via ``fire``)."""

    def __init__(self):
        self.schedules, self.dispatch, self.deadlines, self.calls = {}, {}, {}, []
        self.reject_sql = False

    def _sql(self, query):
        if self.reject_sql:
            raise SpacetimeRejected("Spacetime request rejected (HTTP 400)")
        table = re.search(r"FROM (\w+)", query)[1]
        rows = {
            "task_schedule": self.schedules,
            "task_dispatch": self.dispatch,
            "task_deadline": self.deadlines,
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

    def set_task_deadline(self, task_id, worker, due):
        prior = self.deadlines.get(task_id)
        assert prior is None or prior["deadline_micros"] == due
        self.deadlines[task_id] = {
            "task_id": task_id,
            "worker_id": worker,
            "deadline_micros": due,
            "status": "watching",
            "checked_micros": 0,
            "detail": "",
        }

    def clear_task_deadline(self, task_id, worker):
        self.deadlines.pop(task_id, None)


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
        {"deadline_minutes": 3},
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
    first = service.create(MANAGER, request(repeat_minutes=60, max_runs=3, deadline_minutes=15))
    assert first["status"] == "active" and first["repeat_minutes"] == 60 and first["max_runs"] == 3
    assert first["deadline_minutes"] == 15 and first["key"].startswith("sched-")
    again = service.create(MANAGER, request(repeat_minutes=60, max_runs=3, deadline_minutes=15))
    assert again["key"] == first["key"] and database.calls.count("create_task_schedule") == 1
    with pytest.raises(TaskConflict):
        service.create(MANAGER, request(objective="Something else."))


def test_fired_run_creates_objective_and_arms_deadline(env):
    database, tasks, _, service = env
    schedule = service.create(EMPLOYEE, request(deadline_minutes=5))
    database.fire(schedule["key"])
    assert service.process_due() == 1
    run = database.dispatch[f"{schedule['key']}-r1"]
    assert run["status"] == "created" and run["task_id"] in tasks.created
    task = tasks.created[run["task_id"]]
    # The dispatch key is the request_id, so a retried run maps to the same objective.
    assert task["request_id"] == run["dispatch_key"] and task["actor"] == "operator"
    deadline = database.deadlines[run["task_id"]]
    assert deadline["deadline_micros"] == round((task["created_at"] + 300) * 1_000_000)
    assert service.process_due() == 0  # nothing left to do


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


def test_list_reports_runs_deadlines_and_degrades_without_module(env):
    database, _, _, service = env
    schedule = service.create(MANAGER, request(repeat_minutes=15, max_runs=2, deadline_minutes=60))
    database.fire(schedule["key"])
    listed = service.list(MANAGER)  # also processes the due run
    assert listed["available"] is True and listed["backend"] == "SpacetimeDB scheduled reducers"
    item = listed["schedules"][0]
    assert item["fired_runs"] == 1 and item["runs"][0]["status"] == "created"
    assert item["runs"][0]["task_id"] in listed["deadlines"]
    assert json.loads(database.schedules[schedule["key"]]["request_json"])["deadline_minutes"] == 60
    database.reject_sql = True
    degraded = service.list(MANAGER)
    assert degraded["available"] is False and "publish" in degraded["reason"]


def test_clear_deadline_requires_a_manager(env):
    database, _, _, service = env
    database.set_task_deadline("goal-" + "a" * 16, FakeTasks.worker_id, 1)
    with pytest.raises(PermissionError):
        service.clear_deadline(EMPLOYEE, "goal-" + "a" * 16)
    assert service.clear_deadline(MANAGER, "goal-" + "a" * 16)["cleared"] is True
    assert database.deadlines == {}


def test_deadline_minutes_validation():
    assert deadline_minutes({}) is None
    assert deadline_minutes({"deadline_minutes": 15}) == 15
    for value in (0, 7, "15", 15.0, True, 10_000):
        with pytest.raises(ValueError):
            deadline_minutes({"deadline_minutes": value})
