"""Controller information boundaries and honest comparison accounting."""

import pytest

from relay_lab.cloth.benchmark import paired_checks, summarize, validate_evaluation_seeds
from relay_lab.cloth.runtime import ClothConfig, LocalPhaseController, PhaseJudge


def observation(**changes):
    return {
        "agent": "left",
        "tick": 10,
        "ee_position": [0.0, 0.0, 0.2],
        "grasped": True,
        "grasp_target": "hem",
        "obstacle": None,
        **changes,
    }


def belief(**features):
    return {
        "packet_id": "right-4",
        "observation_tick": 4,
        "age_ticks": 6,
        "features": {"obstacle_active": 0.0, "grasped": 1.0, "grasp_hem": 1.0, **features},
    }


def test_private_peer_obstacle_changes_action_only_through_received_report():
    control = LocalPhaseController("left", ("carry_hem", "settle"))
    own = observation()
    assert not control.decision(own, belief())["hold"]
    decision = control.decision(own, belief(obstacle_active=1.0))
    assert decision["hold"]
    assert decision["reason"] == "peer obstacle report"
    assert decision["peer_packet_id"] == "right-4"


def test_stale_absent_peer_while_carrying_stops_without_fabricating_peer_state():
    control = LocalPhaseController("left", ("carry_hem",))
    assert control.decision(observation(), None)["hold"]
    assert not control.decision(observation(), None, communication=False)["hold"]


def test_local_obstacle_and_explicit_none_still_stop_local_robot():
    control = LocalPhaseController("left", ("carry_hem",))
    assert control.decision(observation(obstacle={"active": True}), None, communication=False)[
        "hold"
    ]


def test_partner_grasp_required_before_shared_lift():
    control = LocalPhaseController("left", ("lift_hem",))
    assert control.decision(observation(), belief(grasped=0.0, grasp_hem=0.0))["hold"]
    assert not control.decision(observation(), belief(grasped=1.0, grasp_hem=1.0))["hold"]
    assert control.decision(observation(), belief(grasped=1.0, grasp_hem=0.0, grasp_sleeve=1.0))[
        "hold"
    ]


@pytest.mark.parametrize("seed", [0, 17, 99, 10000, 10004, True])
def test_benchmark_rejects_training_and_development_seeds(seed):
    with pytest.raises(ValueError, match="Evaluation seeds"):
        validate_evaluation_seeds([seed])


def test_phase_duration_does_not_override_failed_physical_target():
    control = LocalPhaseController("left", ("carry_hem", "settle"))
    control.active_ticks = 179
    control.last_decision = {"hold": False}
    with pytest.raises(RuntimeError, match="could not complete"):
        control.update(observation(), observation(), {"target": [1, 1, 1]}, belief(), PhaseJudge())
    assert control.phase == "carry_hem"


def test_wrong_local_grasp_does_not_complete_hem_grasp_phase():
    control = LocalPhaseController("left", ("grasp_hem", "lift_hem"))
    control.active_ticks = 30
    control.last_decision = {"hold": False}
    result = control.update(
        observation(),
        observation(grasp_target="sleeve"),
        {"target": [0, 0, 0.2]},
        belief(),
        PhaseJudge(),
    )
    assert result is None
    assert control.phase == "grasp_hem"


def test_hold_does_not_advance_local_phase_clock():
    control = LocalPhaseController("left", ("carry_hem", "settle"))
    control.active_ticks = 39
    control.last_decision = {"hold": True}
    assert (
        control.update(
            observation(), observation(), {"target": [0, 0, 0.2]}, belief(), PhaseJudge()
        )
        is None
    )
    assert control.active_ticks == 39


def test_judge_refusal_is_rate_limited_and_cannot_bypass_phase_timeout():
    class RefusingJudge:
        calls = 0

        def advance(self, *args):
            self.calls += 1
            return False

    judge = RefusingJudge()
    control = LocalPhaseController("left", ("carry_hem", "settle"))
    control.active_ticks = 44
    control.last_decision = {"hold": False}
    action = {"target": [0, 0, 0.2]}
    control.update(observation(), observation(tick=10), action, belief(), judge)
    control.update(observation(), observation(tick=11), action, belief(), judge)
    assert judge.calls == 1
    control.update(observation(), observation(tick=35), action, belief(), judge)
    assert judge.calls == 2
    control.active_ticks = 179
    with pytest.raises(RuntimeError, match="180 active ticks"):
        control.update(observation(), observation(tick=60), action, belief(), judge)


@pytest.mark.parametrize(
    "values", [{"seed": -1}, {"max_ticks": True}, {"mode": "fake"}, {"scenario": "fake"}]
)
def test_config_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        ClothConfig(**values)


def test_network_impairment_is_matched_across_representations():
    configs = [ClothConfig(seed=12, mode=m) for m in ("embeddings", "structured_text", "hybrid")]
    schedules = [[c.network().schedule("left", t) for t in range(100)] for c in configs]
    assert schedules[0] == schedules[1] == schedules[2]
    assert len({arrival - t for t, (arrival, _) in enumerate(schedules[0])}) > 1


def test_failed_runs_are_in_denominator_and_null_hashes_do_not_prove_parity():
    rows = [
        {
            "seed": 101,
            "scenario": "delayed",
            "mode": mode,
            "success": False,
            "status": "failed",
            "ticks": 0,
        }
        for mode in ("embeddings", "structured_text", "hybrid")
    ]
    assert all(row["runs"] == 1 and row["success_rate"] == 0 for row in summarize(rows))
    pairing = paired_checks(rows)[0]
    assert pairing["complete"]
    assert not pairing["same_initial_physics"]
    assert not pairing["same_initial_rgb"]
    assert not pairing["same_motor_models"]
    assert not pairing["same_codec"]
