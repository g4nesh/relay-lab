"""Private-observation and physical routing contract tests, without a learned oracle."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from relay_lab.factory.scene import pallet_slot
from relay_lab.factory.vision import instruction_for, local_marker_positions, marker_positions
from relay_lab.factory.world import FactoryWorld


def run_routing(world, choices):
    active, observed = {}, []
    sequence = 0
    for _ in range(6000):
        used = {resource for command in active.values() for resource in command["resources"]}
        for candidate in world.candidates():
            if used.intersection(candidate["resources"]):
                continue
            sequence += 1
            payload = {
                key: candidate[key]
                for key in ("kind", "source", "target", "cargo_id", "expected_stage", "next_stage")
            }
            if candidate["target"] == "pallet":
                payload["pallet_slot"] = choices[candidate["job_id"]]
            command = {
                "command_id": f"routing-{sequence}",
                "robot_id": candidate["robot_id"],
                "capability": candidate["capability"],
                "job_id": candidate["job_id"],
                "resources": candidate["resources"],
                "payload": payload,
            }
            active[command["command_id"]] = command
            used.update(command["resources"])
        result = world.step(list(active.values()))
        for outcome in result["outcomes"]:
            if outcome["status"] != "running":
                assert outcome["status"] == "completed", outcome
                observed.append(outcome)
                active.pop(outcome["command_id"], None)
        if world.physical_complete:
            return observed
    pytest.fail("Routing work did not physically complete")


@pytest.mark.parametrize("wrong_first", [False, True])
def test_routing_scores_measured_deliveries_without_aborting_later_orders(wrong_first):
    world = FactoryWorld(task="dispatch-routing", batch_size=3)
    # Explicit evaluator fixture. These labels never inform the scheduler below.
    world._private_routes = {"part-001": "A", "part-002": "B", "part-003": "A"}
    chosen = {"job-001": int(wrong_first), "job-002": 3, "job-003": 4}
    with pytest.raises(RuntimeError, match="sealed"):
        world.routing_oracle()
    outcomes = run_routing(world, chosen)
    state = world.snapshot()
    assert state["physical_complete"]
    assert state["completed"] == 3
    assert state["success"] is (not wrong_first)
    assert all(job["stage"] == "completed" for job in state["jobs"].values())
    assert sum(job["routing_correct"] for job in state["jobs"].values()) == 3 - wrong_first
    delivered = [outcome for outcome in outcomes if "routing_correct" in outcome["observed"]]
    assert len(delivered) == 3
    assert delivered[0]["observed"]["routing_correct"] is (not wrong_first)
    for item in world.cargo.values():
        slot = chosen[item["job_id"]]
        assert item["delivery_slot"] == slot
        assert np.linalg.norm(world._cargo_position(item["id"]) - pallet_slot(slot)) < 0.065
    assert world.routing_oracle()["job-001"] == {
        "route": "A",
        "required_slot": 0,
        "selected_slot": int(wrong_first),
        "correct": not wrong_first,
    }


def test_private_cue_changes_do_not_change_public_state_or_available_actions():
    first = FactoryWorld(task="dispatch-routing", batch_size=3, seed=17, pose_jitter=0.02)
    second = FactoryWorld(task="dispatch-routing", batch_size=3, seed=17, pose_jitter=0.02)
    second._private_routes = {
        key: "B" if value == "A" else "A" for key, value in first._private_routes.items()
    }
    assert first.snapshot() == second.snapshot()
    assert first.candidates() == second.candidates()
    assert first.scene() == second.scene()
    assert marker_positions(first.snapshot()) == marker_positions(second.snapshot())
    assert "_private_routes" not in repr(first.snapshot())


def test_cue_camera_is_receiving_humanoid_only_and_returns_pixels(monkeypatch):
    rendered = []

    def render(route, seed):
        rendered.append((route, seed))
        return np.full((64, 64, 3), 123, dtype=np.uint8)

    monkeypatch.setitem(
        sys.modules,
        "relay_lab.factory.routing_codec",
        SimpleNamespace(render_inspection_cue=render),
    )
    world = FactoryWorld(task="dispatch-routing", batch_size=1)
    with pytest.raises(PermissionError):
        world.inspect_private_cue("gantry-01", "job-001")
    assert rendered == []
    rgb = world.inspect_private_cue("humanoid-01", "job-001")
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert len(rendered) == 1
    world.cargo["part-001"]["owner"] = "humanoid-01"
    with pytest.raises(ValueError, match="only while"):
        world.inspect_private_cue("humanoid-01", "job-001")
    assert len(rendered) == 1


@pytest.mark.parametrize("slot", [None, True, -1, 2, 3, 6, "0"])
def test_dispatch_never_invents_a_route_or_accepts_another_jobs_reserved_slot(slot):
    world = FactoryWorld(task="dispatch-routing", batch_size=2)
    world.cargo["part-001"]["processed"] = True
    command = {"target_id": "pallet", "cargo_id": "part-001"}
    if slot is not None:
        command["pallet_slot"] = slot
    with pytest.raises(ValueError, match="explicit slot"):
        world._placement_target(world.robots["gantry-01"], command)


def test_gantry_motor_instruction_uses_committed_local_slot_choice():
    world = FactoryWorld(task="dispatch-routing", batch_size=1)
    world.cargo["part-001"]["processed"] = True
    command = {"target_id": "pallet", "cargo_id": "part-001", "pallet_slot": 1}
    assert np.array_equal(
        world._placement_target(world.robots["gantry-01"], command), pallet_slot(1)
    )
    assert instruction_for({}, command, "approach_place") == "place part-001 on pallet-002"
    assert world.candidates() == deepcopy(world.candidates())


def test_local_dynamic_visibility_preserves_static_navigation_beacons():
    world = FactoryWorld(task="dispatch-routing", batch_size=3)
    snapshot = world.snapshot()
    humanoid = local_marker_positions(snapshot, "humanoid-01")
    arm = local_marker_positions(snapshot, "arm-01")
    gantry = local_marker_positions(snapshot, "gantry-01")
    mobile = local_marker_positions(snapshot, "amr-01")
    assert "part-001" in humanoid
    assert "part-001" not in arm and "part-001" not in gantry
    assert "gantry-01" not in humanoid
    assert "arm-01" not in mobile and "amr-02" not in mobile
    assert {"load", "process-in", "process-out", "dispatch"} <= mobile.keys()
    assert "part-001" in marker_positions(snapshot)  # Standard sensor path stays unchanged.


def test_dispatch_task_has_explicit_capacity_and_independent_seeded_cues():
    with pytest.raises(ValueError, match="maximum three"):
        FactoryWorld(task="dispatch-routing", batch_size=4)
    same = FactoryWorld(task="dispatch-routing", seed=33, batch_size=3)
    jittered = FactoryWorld(task="dispatch-routing", seed=33, batch_size=3, pose_jitter=0.02)
    assert same._private_routes == jittered._private_routes
    # Per-job labels are independent; no balanced batch lets an observer infer
    # a final hidden label from the previous delivery outcomes.
    labels = [
        FactoryWorld(task="dispatch-routing", seed=s, batch_size=1)._private_routes["part-001"]
        for s in range(16)
    ]
    assert set(labels) == {"A", "B"}
