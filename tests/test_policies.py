import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from relay_lab.policies import HttpVLAPolicy, RuleOrchestrator, ScriptedPolicy
from relay_lab.types import MAX_DELTA, MIN_HEIGHT, SAFE_HEIGHT, Instruction, Message, Observation


def observation(agent="picker", **changes):
    base = Observation(
        tick=0,
        agent_id=agent,
        ee_position=(-0.54, 0.0, SAFE_HEIGHT) if agent == "picker" else (0.48, 0.0, SAFE_HEIGHT),
        holding=None,
        objects={"coral": (-0.4, -0.1, 0.035), "mint": (-0.4, 0.1, 0.035)},
        targets={"relay": (0.0, 0.0, 0.035), "slot_0": (0.4, 0.1, 0.035)},
        order=None if agent == "picker" else ("mint", "coral", "gold"),
        packed=() if agent == "picker" else (None, None, None),
    )
    return replace(base, **changes)


def test_picker_uses_only_delivered_requests_and_holds_its_own_context():
    picker = RuleOrchestrator("picker")
    independent = RuleOrchestrator("picker")
    first = observation()
    assert picker.decide(first, []).instruction.object_id == "coral"
    request = Message("packer", "picker", "request:mint", 0, 1)
    assert picker.decide(first, [request]).instruction.object_id == "coral"
    next_tick = replace(first, tick=1)
    assert picker.decide(next_tick, [request]).instruction.object_id == "mint"
    assert picker.decide(replace(first, tick=2), []).instruction.object_id == "mint"
    assert independent.decide(next_tick, []).instruction.object_id == "coral"
    picker.reset()
    assert picker.decide(next_tick, []).instruction.object_id == "coral"


def test_packer_requests_next_slot_and_retransmits_without_spamming():
    packer = RuleOrchestrator("packer", request_interval=30)
    obs = observation("packer", objects={})
    assert packer.decide(obs, []).message == "request:mint"
    assert packer.decide(replace(obs, tick=1), []).message is None
    assert packer.decide(replace(obs, tick=30), []).message == "request:mint"
    next_slot = replace(obs, tick=31, packed=("mint", None, None))
    assert packer.decide(next_slot, []).message == "request:coral"


def test_relay_blocks_picker_and_wrong_cube_blocks_packer():
    obs = observation(objects={"coral": (0.0, 0.0, 0.035), "mint": (-0.4, 0.1, 0.035)})
    assert RuleOrchestrator("picker").decide(obs, []).instruction.kind == "wait"
    wrong = observation("packer", objects={"coral": (0.0, 0.0, 0.035)})
    assert RuleOrchestrator("packer").decide(wrong, []).instruction.kind == "wait"
    right = replace(wrong, objects={"mint": (0.0, 0.0, 0.035)})
    assert RuleOrchestrator("packer").decide(right, []).instruction.object_id == "mint"


def test_fetch_handshake_requires_verified_ready_feedback_before_collection():
    packer = RuleOrchestrator("packer", require_transfer_ready=True)
    at_relay = observation("packer", objects={"mint": (0.0, 0.0, 0.035)})
    assert packer.decide(at_relay, []).instruction.kind == "wait"
    ready = {
        "agent_id": "picker",
        "kind": "transfer_ready",
        "object_id": "mint",
        "target_id": "relay",
        "evidence": {"picker_clear": True},
    }
    assert packer.decide(at_relay, [], [{**ready, "agent_id": "packer"}]).instruction.kind == "wait"
    assert packer.decide(at_relay, [], [{**ready, "evidence": {}}]).instruction.kind == "wait"
    assert packer.decide(at_relay, [], [ready]).instruction.kind == "pick"
    packer.decide(replace(at_relay, holding="mint"), [])
    assert packer.decide(at_relay, []).instruction.kind == "wait"


def test_picker_regrasp_and_release_requires_a_fresh_verified_handoff():
    packer = RuleOrchestrator("packer", require_transfer_ready=True)
    ready = {
        "agent_id": "picker",
        "kind": "transfer_ready",
        "object_id": "mint",
        "target_id": "relay",
        "evidence": {"picker_clear": True},
    }
    at_relay = observation("packer", tick=10, objects={"mint": (0.0, 0.0, 0.035)})
    assert packer.decide(at_relay, [], [ready]).instruction.kind == "pick"

    # A peer-held cube disappears from the packer's local observation. The
    # holding event also invalidates the old readiness for any later release.
    holding = {"agent_id": "picker", "kind": "holding", "object_id": "mint"}
    held_by_picker = replace(at_relay, tick=11, objects={})
    assert packer.decide(held_by_picker, [], [holding]).instruction.kind == "wait"
    released_again = replace(at_relay, tick=12)
    assert packer.decide(released_again, []).instruction.kind == "wait"
    assert packer.decide(replace(at_relay, tick=13), []).instruction.kind == "wait"
    assert packer.decide(replace(at_relay, tick=14), [], [ready]).instruction.kind == "pick"


def test_controller_lifts_travels_lowers_and_closes_with_bounded_actions():
    policy = ScriptedPolicy("picker")
    obs = observation(ee_position=(-0.54, 0.0, 0.035))
    pick = Instruction("pick", "Pick mint.", "mint")
    first = policy.act(obs, pick)
    assert first.delta[0] == first.delta[1] == 0
    assert first.delta[2] > 0
    for _ in range(100):
        action = policy.act(obs, pick)
        assert np.linalg.norm(action.delta) <= MAX_DELTA + 1e-9
        if action.gripper == -1:
            break
        if abs(action.delta[0]) + abs(action.delta[1]) > 0:
            assert abs(obs.ee_position[2] - SAFE_HEIGHT) < 0.004
        obs = replace(obs, ee_position=tuple(np.array(obs.ee_position) + action.delta))
    else:
        pytest.fail("Controller never reached its grasp waypoint")
    assert np.linalg.norm(np.array(obs.ee_position) - obs.objects["mint"]) < 0.004


def test_changed_instruction_lifts_again_and_wait_retreats():
    policy = ScriptedPolicy("picker")
    low = observation(ee_position=(-0.4, 0.1, 0.035))
    policy.act(low, Instruction("pick", "Pick mint.", "mint"))
    held = replace(low, holding="mint")
    action = policy.act(held, Instruction("place", "Relay mint.", "mint", "relay"))
    assert action.delta[0] == action.delta[1] == 0
    assert action.delta[2] > 0
    action = policy.act(low, Instruction("wait", "Return home."))
    assert action.delta[2] > 0
    assert action.gripper == 0


def test_settled_cube_below_ee_workspace_still_reaches_grasp():
    policy = ScriptedPolicy("picker")
    obs = observation(objects={"mint": (-0.4, 0.1, 0.029986)})
    for _ in range(100):
        action = policy.act(obs, Instruction("pick", "Pick mint.", "mint"))
        if action.gripper == -1:
            break
        position = np.array(obs.ee_position) + action.delta
        position[2] = max(MIN_HEIGHT, position[2])
        obs = replace(obs, ee_position=tuple(position))
    else:
        pytest.fail("Controller tried to reach an object below the EE workspace")
    assert abs(obs.ee_position[2] - MIN_HEIGHT) < 0.003


class Response:
    def __init__(self, data):
        self.body = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _):
        return self.body


def test_http_contract_encodes_local_rgb_and_validates_action():
    policy = HttpVLAPolicy("http://localhost:8000/act", "picker")
    obs = observation(rgb=np.zeros((3, 4, 3), dtype=np.uint8))
    with patch(
        "relay_lab.policies.urlopen",
        return_value=Response(
            {
                "action": {"delta": [0.01, 0, 0], "gripper": -1},
            }
        ),
    ) as request:
        action = policy.act(obs, Instruction("pick", "Pick mint.", "mint"))
    assert action.delta == (0.01, 0.0, 0.0)
    payload = json.loads(request.call_args.args[0].data)
    assert payload["agent_id"] == "picker"
    assert payload["session_id"] == policy.session_id
    assert HttpVLAPolicy("http://localhost:8000/act", "picker").session_id != policy.session_id
    previous_session = policy.session_id
    policy.reset()
    assert policy.session_id != previous_session
    assert payload["control_dt"] == 0.1
    assert payload["observation"]["order"] is None
    assert payload["observation"]["objects"] == {
        key: list(value) for key, value in obs.objects.items()
    }
    assert payload["rgb_png_base64"].startswith("iVBORw0KGgo")
    for delta, gripper in (([0.04, 0, 0], 0), ([float("nan"), 0, 0], 0), ([0, 0, 0], True)):
        with (
            patch(
                "relay_lab.policies.urlopen",
                return_value=Response(
                    {
                        "action": {"delta": delta, "gripper": gripper},
                    }
                ),
            ),
            pytest.raises(RuntimeError, match="invalid action"),
        ):
            policy.act(obs, Instruction("wait", "Wait."))
