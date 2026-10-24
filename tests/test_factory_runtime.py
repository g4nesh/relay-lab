"""The coordinator barrier controls whether the physical world may advance."""

import json

import pytest

from relay_lab.factory.coordination import CoordinationError, MemoryCoordinator
from relay_lab.factory.runtime import FactoryRun
from relay_lab.factory.world import FactoryWorld


class CountingWorld(FactoryWorld):
    def __init__(self):
        super().__init__(batch_size=1)
        self.physical_steps = 0

    def step(self, commands=None):
        self.physical_steps += 1
        return super().step(commands)


class RejectPrepare(MemoryCoordinator):
    def prepare(self, revision, commands, messages=None):
        raise CoordinationError("Reservation rejected before physical execution")


class RejectCommit(MemoryCoordinator):
    def commit(self, revision, state, outcomes, messages=None):
        raise CoordinationError("Observed result commit was not confirmed")


def make_run(tmp_path, *, world=None, coordinator=None):
    return FactoryRun(
        orders=1,
        planner="rules",
        coordination="memory",
        output=tmp_path / "run",
        world=world or CountingWorld(),
        coordinator=coordinator or MemoryCoordinator(),
    )


def test_prepare_rejection_prevents_any_physical_step(tmp_path):
    world = CountingWorld()
    run = make_run(tmp_path, world=world, coordinator=RejectPrepare())
    try:
        with pytest.raises(CoordinationError, match="Reservation rejected"):
            run.tick()
        assert world.physical_steps == 0
        assert world.tick == 0
        assert run.status == "failed"
        with pytest.raises(RuntimeError, match="Worker is stopped"):
            run.tick()
        assert world.physical_steps == 0
    finally:
        run.close()


def test_unconfirmed_commit_poisons_worker_and_never_reexecutes_motion(tmp_path):
    world = CountingWorld()
    run = make_run(tmp_path, world=world, coordinator=RejectCommit())
    try:
        with pytest.raises(CoordinationError, match="not confirmed"):
            run.tick()
        assert world.physical_steps == 1
        assert world.tick == 1
        assert run.status == "failed"
        # The UI remains on the last committed state, while the physical fact
        # is durably distinguishable as unconfirmed execution for diagnosis.
        assert run.view()["tick"] == 0
        records = [
            json.loads(line) for line in (run.output / "events.jsonl").read_text().splitlines()
        ]
        observed = [record for record in records if record["type"] == "execution_observed"]
        assert len(observed) == 1
        assert observed[0]["tick"] == observed[0]["state"]["tick"] == 1
        assert observed[0]["revision"] == 0
        assert observed[0]["committed"] is False
        assert observed[0]["outcomes"]
        assert observed[0]["events"]
        assert not [record for record in records if record["type"] == "tick"]
        with pytest.raises(RuntimeError, match="Worker is stopped"):
            run.tick()
        assert world.physical_steps == 1
        assert world.tick == 1
    finally:
        run.close()


def test_failed_observed_skill_stops_runner_instead_of_replanning(tmp_path):
    class FailedSkill(CountingWorld):
        def _start(self, robot):
            raise ValueError("Injected docking fault")

    world = FailedSkill()
    run = make_run(tmp_path, world=world)
    try:
        with pytest.raises(RuntimeError, match="motion skill failed"):
            run.tick()
        assert world.physical_steps == 1
        assert run.failed_commands == 1
        assert run.status == "failed"
        assert any("failed" in feedback["text"] for feedback in run.feedback)
        with pytest.raises(RuntimeError, match="Worker is stopped"):
            run.tick()
        assert world.physical_steps == 1
    finally:
        run.close()


def test_rules_reference_finishes_all_observed_stages_through_memory_barrier(tmp_path):
    run = make_run(tmp_path)
    observed_stages = set()
    try:
        for _ in range(1600):
            run.tick()
            observed_stages.add(run.state["jobs"]["job-001"]["stage"])
            if run.status != "running":
                break
        assert run.status == "succeeded"
        assert observed_stages == {
            "queued",
            "loaded",
            "at_cell",
            "in_fixture",
            "processed",
            "outbound_loaded",
            "at_dispatch",
            "completed",
        }
        assert run.world.success
        assert run.completed_commands == 9
        assert run.failed_commands == 0
        assert run.authority["tick"] == run.world.tick
        assert run.authority["phase"] == "terminal"
        assert all(not lease["holder"] for lease in run.authority["resources"].values())
        assert run.planner.info["mode"] == "rules"
        assert run.planner.info["learned_motor_control"] is False
        steps = run.world.physical_steps
        run.tick()
        assert run.world.physical_steps == steps
    finally:
        run.close()


def test_missing_backend_records_failure_without_advancing_world(tmp_path):
    class RejectCreate(MemoryCoordinator):
        def create(self, *args, **kwargs):
            raise CoordinationError("Database unavailable")

    world = CountingWorld()
    run = make_run(tmp_path, world=world, coordinator=RejectCreate())
    try:
        with pytest.raises(CoordinationError, match="Database unavailable"):
            run.tick()
        assert run.status == "failed"
        assert world.physical_steps == 0
        assert not run.view()["coordination"]["connected"]
    finally:
        run.close()
