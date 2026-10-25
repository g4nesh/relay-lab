"""Bounded two-foundation-model integration through the actual cloth journal.

This does not use the task-trained phase controller or teacher. Both pretrained
policies act from their own observation at one common physics tick. Delivered
peer measurements enter each model's language prompt. It is an integration
trial, not a controlled communication comparison or a folding-success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

from .communication import Channel, LearnedFeatureCodec, NetworkConfig, observation_features
from .pretrained import SmolVLAClothProbe, XVLAClothProbe
from .world import ClothWorld

AGENTS = ("left", "right")


def compact_context(observation, belief):
    """Accept only the local sensor record and a delivered/decoded peer packet."""
    peer = None
    if belief is not None:
        if belief["recipient"] != observation["agent"]:
            raise ValueError("A robot cannot consume another recipient's report")
        features = belief["features"]
        peer = {
            "grasp": "hem"
            if features.get("grasp_hem", 0) >= 0.5
            else "sleeve"
            if features.get("grasp_sleeve", 0) >= 0.5
            else "none",
            "obstacle": features.get("obstacle_active", 0) >= 0.5,
            "tool_xyz_m": [round(features[f"ee_{axis}"], 2) for axis in "xyz"],
            "age_ticks": belief["age_ticks"],
            "packet_id": belief["packet_id"],
            "precision": belief["precision"],
        }
    return {
        "agent": observation["agent"],
        "own_grasp": observation.get("grasp_target", "none"),
        "own_obstacle": bool(observation.get("obstacle")),
        "peer": peer,
    }


def instruction_from_context(context):
    """Render received facts as language, without generating Cartesian targets."""
    peer = context["peer"]
    own = " Own obstacle present." if context["own_obstacle"] else ""
    if peer is None:
        return "Fold the T-shirt with your arm. Peer state is unknown." + own
    xyz = ",".join(f"{v:.2f}" for v in peer["tool_xyz_m"])
    return (
        f"Fold the T-shirt. Peer grip {peer['grasp']}; obstacle "
        f"{'present' if peer['obstacle'] else 'absent'}; tool {xyz} m; "
        f"report age {peer['age_ticks']} ticks. Coordinate lifting.{own}"
    )


class PromptMediator:
    """Optional Qwen call on categorical changes; fresh facts accompany each call.

    Histories are separate and receive only one robot's own local state and its
    delivered packet. A generation failure aborts; it never selects a teacher.
    """

    def __init__(self, mode="template"):
        from ..llm import LocalLLM

        if mode not in ("template", "qwen"):
            raise ValueError("unknown pretrained-team mediator")
        self.mode = mode
        self.llm = LocalLLM(max_tokens=64) if mode == "qwen" else None
        self.signatures, self.advice, self.traces = {}, {}, []

    def instruction(self, observation, belief):
        context = compact_context(observation, belief)
        literal = instruction_from_context(context)
        if self.llm is None:
            return literal, context
        peer = context["peer"]
        signature = (
            context["own_grasp"],
            context["own_obstacle"],
            None if peer is None else (peer["grasp"], peer["obstacle"]),
        )
        agent = observation["agent"]
        if self.signatures.get(agent) != signature:
            messages = [
                {
                    "role": "system",
                    "content": "Coordinate one robot arm folding a T-shirt. You receive only "
                    "its local status and a delivered peer report. Give one short instruction "
                    "of at most 20 words. Do not invent positions or claim success. Treat "
                    "missing reports as unknown. No JSON, coordinates, or explanations. /no_think",
                },
                {"role": "user", "content": json.dumps(context)},
            ]
            generated = self.llm.generate(messages)
            advice = generated.text.strip()
            if not advice or len(advice) > 400:
                raise ValueError("Qwen returned an invalid bounded instruction; no fallback")
            self.advice[agent], self.signatures[agent] = advice, signature
            self.traces.append({"input": context, "generation": generated.to_dict()})
        return literal + " " + self.advice[agent], context

    def close(self):
        if self.llm:
            self.llm.close()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(
    output: Path,
    *,
    calls=12,
    action_ticks=4,
    seed=77,
    mode="hybrid",
    coordination="spacetime",
    mediator="template",
    codec_path=Path("checkpoints/cloth-policies/communication.json"),
):
    import imageio.v2 as imageio
    import imageio.v3 as iio

    if type(calls) is not int or not 1 <= calls <= 32:
        raise ValueError("The bounded integration trial requires 1–32 model calls per arm")
    if type(action_ticks) is not int or not 1 <= action_ticks <= 8:
        raise ValueError("action_ticks must be 1–8")
    if coordination not in ("spacetime", "memory"):
        raise ValueError("unknown coordination backend")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = "cloth-pretrained-" + uuid.uuid4().hex[:12]
    report = {
        "schema": "relay.cloth.pretrained-team.v1",
        "run_id": run_id,
        "status": "starting",
        "seed": seed,
        "requested_calls_per_arm": calls,
        "action_ticks": action_ticks,
        "communication_mode": mode,
        "coordination": coordination,
        "mediator": mediator,
        "full_task_evaluation": False,
        "controlled_representation_comparison": False,
        "scripted_target_fallback": False,
        "action_mapping": {
            "maximum_cartesian_delta_m": 0.018,
            "workspace_lower_m": [-0.59, -0.49, 0.008],
            "workspace_upper_m": [0.59, 0.42, 0.52],
            "rotation": "Both adapters discard predicted rotation; tools remain downward.",
            "execution": "Both actions share the same pre-step observation tick; predictions run sequentially on CPU.",
        },
        "limitations": [
            "SmolVLA LIBERO and X-VLA Soft-FOLD are unadapted to this workcell and camera.",
            "This short trial checks real inference, shared-state transport, and physical execution; it cannot estimate folding success rates.",
            "The codec encodes measured local features, not foundation-model hidden states.",
            "Language prompts contain delivered peer facts; this does not prove the policies use those facts effectively.",
            "Simulation time pauses during inference and synchronous Spacetime commits.",
            "Idealized cloth vertex grasps and the vacuum fixture are shared simulator assumptions.",
        ],
        "records": [],
    }
    path = output / "report.json"
    _write(path, report)
    world = channel = prompts = writer = None
    policies = {}
    started = time.perf_counter()
    try:
        codec = LearnedFeatureCodec.load(codec_path)
        codec.save(output / "communication-codec.json")
        backend = None
        if coordination == "spacetime":
            from .spacetime import SpacetimeJournal

            backend = SpacetimeJournal(worker_id="cloth-pretrained-worker")
        channel = Channel(
            run_id,
            mode=mode,
            codec=codec,
            network=NetworkConfig(
                seed=seed, min_delay_ticks=1, max_delay_ticks=3, drop_probability=0, ttl_ticks=24
            ),
            backend=backend,
        )
        world = ClothWorld(
            seed=seed,
            disturbances=True,
            obstacle_schedule=((12, 24, "left"), (32, 40, "right")),
        )
        report["initial_snapshot"] = world.snapshot()
        iio.imwrite(output / "initial.png", world.render("overview", width=960, height=720))
        policies["left"] = SmolVLAClothProbe("left", "cpu", seed)
        policies["right"] = XVLAClothProbe("right", "cpu", seed)
        report["models"] = {agent: policy.model_info for agent, policy in policies.items()}
        prompts = PromptMediator(mediator)
        writer = imageio.get_writer(
            str(output / "overview.mp4"), fps=1 / world.CONTROL_DT, codec="libx264", quality=8
        )
        for index in range(calls):
            tick = world.tick
            observations = {agent: world.observe(agent, rgb=True) for agent in AGENTS}
            hashes = {
                agent: hashlib.sha256(observations[agent]["rgb"].tobytes()).hexdigest()
                for agent in AGENTS
            }
            for agent in AGENTS:
                if index == 0:
                    iio.imwrite(output / f"{agent}-input.png", observations[agent]["rgb"])
                channel.publish(
                    agent,
                    {
                        "features": observation_features(observations[agent]),
                        "source_sha256": hashes[agent],
                    },
                    tick,
                )
            previous = report["records"][-1] if report["records"] else None
            channel.advance(
                tick,
                state={
                    "tick": tick,
                    "trial_call": index,
                    "previous_execution": {
                        "measured_robots": previous["measured_robots"],
                        "bounded_actions": {
                            agent: previous["decisions"][agent]["bounded_action"]
                            for agent in AGENTS
                        },
                    }
                    if previous
                    else None,
                },
            )
            beliefs = {agent: channel.receive(agent, tick) for agent in AGENTS}
            actions, decisions = {}, {}
            for agent in AGENTS:
                instruction, context = prompts.instruction(observations[agent], beliefs[agent])
                inference_start = time.perf_counter()
                actions[agent] = policies[agent].act(observations[agent], instruction)
                decisions[agent] = {
                    "instruction": instruction,
                    "mediator_input": context,
                    "belief": beliefs[agent],
                    "local_rgb_sha256": hashes[agent],
                    "local_observation": {
                        key: value for key, value in observations[agent].items() if key != "rgb"
                    },
                    "model_seconds": time.perf_counter() - inference_start,
                    "prediction": policies[agent].last_prediction,
                    "bounded_action": actions[agent],
                }
            for _ in range(action_ticks):
                snapshot = world.step(actions)
                channel.advance(world.tick)
                writer.append_data(world.render("overview", width=960, height=720))
            report["records"].append(
                {
                    "call": index,
                    "observation_tick": tick,
                    "after_tick": world.tick,
                    "decisions": decisions,
                    "measured_robots": snapshot["robots"],
                    "metrics": snapshot["metrics"],
                }
            )
            report["models"] = {agent: policy.model_info for agent, policy in policies.items()}
            _write(path, report)
            print(json.dumps({"trial": run_id, "call": index + 1, "tick": world.tick}), flush=True)
        # Remove the explicit table fixture before reporting final physical state;
        # no extra geometric controller or settling actions are inserted.
        world.release_fixture()
        report["final_snapshot"] = world.snapshot()
        report["metrics"] = world.metrics()
        report["status"] = "joint_inference_transport_and_execution_confirmed"
        report["physical_success_at_budget"] = bool(report["metrics"]["success"])
        channel.advance(
            world.tick,
            state={
                "status": report["status"],
                "physical_success_at_budget": report["physical_success_at_budget"],
                "calls_per_arm": calls,
                "metrics": report["metrics"],
            },
        )
        iio.imwrite(output / "final.png", world.render("overview", width=960, height=720))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        cleanup_errors = []

        def cleanup(name, operation):
            try:
                operation()
            except Exception as error:  # noqa: BLE001 - preserve partial execution evidence
                cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")

        if writer:
            cleanup("video", writer.close)
        if prompts:
            cleanup("mediator", prompts.close)
            report["mediator_traces"] = prompts.traces
        if channel:
            report["communication"] = channel.metrics
            _write(
                output / "communication.json", {"config": channel.config, "events": channel.events}
            )
        if world:
            cleanup("world", world.close)
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        report["models"] = {agent: policy.model_info for agent, policy in policies.items()}
        report["elapsed_seconds"] = time.perf_counter() - started
        _write(path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=12)
    parser.add_argument("--action-ticks", type=int, default=4)
    parser.add_argument(
        "--mode", choices=("embeddings", "structured_text", "hybrid"), default="hybrid"
    )
    parser.add_argument("--coordination", choices=("spacetime", "memory"), default="spacetime")
    parser.add_argument("--mediator", choices=("template", "qwen"), default="template")
    args = parser.parse_args()
    run(
        args.output,
        calls=args.calls,
        action_ticks=args.action_ticks,
        mode=args.mode,
        coordination=args.coordination,
        mediator=args.mediator,
    )


if __name__ == "__main__":
    main()
