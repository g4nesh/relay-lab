"""Behavioral checks for the original factory's physical/coordination boundary."""

import math

import numpy as np
import pytest

from relay_lab.factory.scene import ROBOT_SPECS, cargo_start, pallet_slot, scene_spec
from relay_lab.factory.world import FactoryWorld


def execute(world, robot_id, kind, **kwargs):
    command = {
        "command_id": f"test-{len(world._commands)}",
        "robot_id": robot_id,
        "kind": kind,
        **kwargs,
    }
    observed = []
    for _ in range(1500):
        frame = world.step([command])
        observed.extend(frame["events"])
        results = [
            outcome
            for outcome in frame["outcomes"]
            if outcome["command_id"] == command["command_id"] and outcome["status"] != "running"
        ]
        if results:
            return results[0], observed
    pytest.fail("command did not finish")


def run_workflow(world, max_ticks=6000):
    active = {}
    sequence = 0
    events = []
    failures = []
    previous = {item["id"]: np.array(item["position"]) for item in world.snapshot()["cargo"]}
    max_cargo_step = 0
    for _ in range(max_ticks):
        resources = {resource for command in active.values() for resource in command["resources"]}
        for candidate in world.candidates():
            if resources.intersection(candidate["resources"]):
                continue
            sequence += 1
            # Match the committed backend envelope, including repeated commands.
            command = {
                "command_id": f"command-{sequence}",
                "robot_id": candidate["robot_id"],
                "capability": candidate["capability"],
                "job_id": candidate["job_id"],
                "resources": candidate["resources"],
                "lease_ticks": 1200,
                "payload": {
                    key: candidate[key]
                    for key in (
                        "kind",
                        "source",
                        "target",
                        "cargo_id",
                        "expected_stage",
                        "next_stage",
                    )
                },
            }
            active[command["command_id"]] = command
            resources.update(command["resources"])
        frame = world.step(list(active.values()))
        events.extend(frame["events"])
        for cargo in frame["snapshot"]["cargo"]:
            position = np.array(cargo["position"])
            max_cargo_step = max(
                max_cargo_step, float(np.linalg.norm(position - previous[cargo["id"]]))
            )
            previous[cargo["id"]] = position
        for outcome in frame["outcomes"]:
            if outcome["status"] != "running":
                active.pop(outcome["command_id"], None)
                if outcome["status"] == "failed":
                    failures.append(outcome)
        if world.success:
            break
    return events, failures, max_cargo_step


@pytest.mark.parametrize("batch_size", [1, 3, 6])
def test_complete_factory_workflow_has_observed_custody_and_no_teleport(batch_size):
    world = FactoryWorld(batch_size=batch_size)
    events, failures, max_step = run_workflow(world)
    assert world.success
    assert not failures
    assert max_step <= 1.15 * world.dt + 0.001
    assert len([event for event in events if event["type"] == "delivery_verified"]) == batch_size
    assert len([event for event in events if event["type"] == "process_verified"]) == batch_size
    assert len([event for event in events if event["type"] == "custody_acquired"]) == 4 * batch_size
    assert all(
        item["stage"] == "completed" and item["owner"] is None for item in world.cargo.values()
    )
    assert {event["robot_id"] for event in events if event["type"] == "command_completed"} == set(
        ROBOT_SPECS
    )
    assert all(
        np.linalg.norm(world._cargo_position(cargo_id) - np.array(pallet_slot(item["index"])))
        < 0.065
        for cargo_id, item in world.cargo.items()
    )


def test_navigation_limits_speed_acceleration_and_turn_rate_from_actual_poses():
    world = FactoryWorld(batch_size=1)
    command = {
        "command_id": "navigate",
        "robot_id": "amr-01",
        "kind": "navigate",
        "target_id": "process-in",
    }
    previous_position = world.robots["amr-01"].position.copy()
    previous_velocity = np.zeros(3)
    previous_yaw = world.robots["amr-01"].yaw
    for _ in range(500):
        result = world.step([command])
        robot = world.robots["amr-01"]
        velocity = (robot.position - previous_position) / world.dt
        assert np.linalg.norm(velocity) <= 1.15 + 1e-8
        assert np.linalg.norm(velocity - previous_velocity) / world.dt <= 0.65 + 1e-8
        angle = (robot.yaw - previous_yaw + math.pi) % (2 * math.pi) - math.pi
        assert abs(angle) / world.dt <= 1.2 + 1e-8
        previous_position, previous_velocity, previous_yaw = (
            robot.position.copy(),
            velocity,
            robot.yaw,
        )
        if result["outcomes"][0]["status"] == "completed":
            break
    else:
        pytest.fail("navigation did not finish")


def test_repeated_command_id_is_idempotent_and_changed_payload_is_rejected():
    world = FactoryWorld(batch_size=1)
    command = {"command_id": "same", "robot_id": "amr-01", "kind": "navigate", "target_id": "load"}
    starts = []
    for _ in range(150):
        frame = world.step([command])
        starts.extend(event for event in frame["events"] if event["type"] == "command_started")
    assert len(starts) == 1
    assert world.robots["amr-01"].command is None
    frame = world.step([{**command, "target_id": "process-in"}])
    assert any(
        event["type"] == "command_rejected" and "changed payload" in event["reason"]
        for event in frame["events"]
    )
    assert np.allclose(world.robots["amr-01"].position, [-6, 2, 0])


def test_remote_handoff_is_rejected_without_changing_cargo_owner():
    world = FactoryWorld(batch_size=1)
    result, _ = execute(world, "humanoid-01", "transfer", cargo_id="part-001", target_id="amr-01")
    assert result["status"] == "failed"
    assert "not at load" in result["detail"]
    assert world.cargo["part-001"]["owner"] is None
    assert world.cargo["part-001"]["stage"] == "queued"


def test_motion_lane_is_exclusive_even_without_backend():
    world = FactoryWorld(batch_size=1)
    commands = [
        {"command_id": "a", "robot_id": "amr-01", "kind": "navigate", "target_id": "load"},
        {"command_id": "b", "robot_id": "amr-02", "kind": "navigate", "target_id": "process-out"},
    ]
    start = world.robots["amr-02"].position.copy()
    frame = world.step(commands)
    assert world._lane_owner == "amr-01"
    assert world.robots["amr-02"].phase == "waiting_for_aisle"
    assert np.array_equal(world.robots["amr-02"].position, start)
    assert len([event for event in frame["events"] if event["type"] == "resource_acquired"]) == 1


def test_collision_interlock_stops_an_occupied_route_and_times_out():
    world = FactoryWorld(batch_size=1)
    world.BLOCKED_TIMEOUT = 0.3
    # A stopped carrier is a dynamic obstruction that cannot be passed through.
    world.robots["amr-02"].position[:] = [-6, 0, 0]
    result, events = execute(world, "amr-01", "navigate", target_id="load")
    assert result["status"] == "failed"
    assert "amr-02" in result["detail"]
    assert any(event["type"] == "collision_blocked" for event in events)
    assert world.robots["amr-01"].position[1] < -1.1


def test_stale_job_stage_fails_without_motion():
    world = FactoryWorld(batch_size=1)
    start = world.robots["humanoid-01"].ee.copy()
    result, _ = execute(
        world, "humanoid-01", "pick", cargo_id="part-001", expected_stage="processed"
    )
    assert result["status"] == "failed"
    assert "stage changed" in result["detail"]
    assert np.array_equal(world.robots["humanoid-01"].ee, start)


def test_processing_and_dispatch_cannot_be_claimed_by_text():
    world = FactoryWorld(batch_size=1)
    result, _ = execute(world, "arm-01", "process", cargo_id="part-001", next_stage="completed")
    assert result["status"] == "failed"
    assert not world.cargo["part-001"]["processed"]
    world.cargo["part-001"]["delivered"] = True
    assert not world.success  # Output still has to be observed on the pallet.


def test_arbitrary_next_stage_never_overrides_observed_macro_result():
    world = FactoryWorld(batch_size=1)
    result, _ = execute(world, "amr-01", "navigate", target_id="load")
    assert result["status"] == "completed"
    result, _ = execute(
        world,
        "humanoid-01",
        "transfer",
        cargo_id="part-001",
        target_id="amr-01",
        next_stage="completed",
    )
    assert result["status"] == "completed"
    assert result["observed"]["stage"] == "loaded"
    assert not world.success


def test_failed_grasp_keeps_cargo_free_and_reports_measured_failure():
    world = FactoryWorld(batch_size=1)
    command = {
        "command_id": "pick",
        "robot_id": "humanoid-01",
        "kind": "pick",
        "cargo_id": "part-001",
    }
    world.step([command])
    # External disturbance after trajectory commitment: the arm must not attach
    # the object at a distance or rewrite its pose to manufacture success.
    address = world._qpos["part-001"]
    world.data.qpos[address] -= 0.25
    for _ in range(150):
        frame = world.step([command])
        if frame["outcomes"][0]["status"] != "running":
            break
    assert frame["outcomes"][0]["status"] == "failed"
    assert world.cargo["part-001"]["owner"] is None


def test_payloads_have_real_gravity_and_contact():
    world = FactoryWorld(batch_size=1)
    address = world._qpos["part-001"]
    world.data.qpos[address : address + 3] = [3.5, -4, 1.2]
    for _ in range(40):
        world.step()
    assert 0.115 < world._cargo_position("part-001")[2] < 0.125
    assert not world.success


def test_snapshot_and_scene_are_independent_readonly_views():
    world = FactoryWorld(batch_size=1)
    snapshot = world.snapshot()
    snapshot["robots"][0]["position"][0] = 999
    snapshot["cargo"][0]["owner"] = "invented"
    assert world.robots["humanoid-01"].position[0] == -7
    assert world.cargo["part-001"]["owner"] is None
    scene = scene_spec()
    scene["stations"][0]["position"][0] = 999
    assert scene_spec()["stations"][0]["position"][0] == -8
    assert scene["physics"]["learned_control"] is False


@pytest.mark.parametrize("bad", [0, 7])
def test_batch_bounds_are_explicit(bad):
    with pytest.raises(ValueError):
        FactoryWorld(batch_size=bad)


def test_deterministic_initial_contact_world():
    first, second = FactoryWorld(seed=17, batch_size=6), FactoryWorld(seed=17, batch_size=6)
    assert first.snapshot() == second.snapshot()
    assert all(
        np.linalg.norm(np.array(cargo_start(i)) - np.array(cargo_start(j))) >= 0.55
        for i in range(6)
        for j in range(i)
    )
