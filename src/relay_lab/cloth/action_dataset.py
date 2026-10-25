"""Original, time-aligned action demonstrations for adapting native VLA heads.

The teacher can inspect local cloth landmarks. Learner records deliberately do
not contain them: only one arm's camera, measured tool pose, gripper command and
phase instruction are inputs. Labels are the actual commands submitted before
the next MuJoCo integration step, not targets reconstructed from later states.
This module is an offline data tool; it must never be imported as a fallback
controller by a VLA execution adapter.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .policies import PHASES

SCHEMA = "relay.cloth.native-action-demonstrations.v1"
CORRECTION_SCHEMA = "relay.cloth.native-action-corrections.v1"
RECOVERY_SCHEMA = "relay.cloth.native-action-recovery.v1"
AGENTS = ("left", "right")
RECOVERY_PHASES = frozenset(
    ("approach_sleeve_above", "approach_sleeve", "approach_hem_above", "approach_hem")
)
ARRAY_KEYS = frozenset(
    (
        "rgb",
        "ee_position",
        "ee_rotation",
        "grasped",
        "gripper_closed",
        "phase",
        "hold",
        "target",
        "grip",
        "tick",
        "post_ee_position",
        "executed_target",
    )
)
PHASE_WORDS = {
    "approach_sleeve_above": "approach the sleeve from above",
    "approach_sleeve": "lower the gripper to the sleeve tab",
    "grasp_sleeve": "close the gripper on the sleeve tab",
    "lift_sleeve": "lift the grasped sleeve",
    "carry_sleeve": "carry the sleeve inward across the torso",
    "fold_sleeve": "lower the sleeve onto the torso",
    "release_sleeve": "release the folded sleeve",
    "retreat_sleeve": "raise the open gripper away from the sleeve",
    "approach_hem_above": "approach the hem tab from above",
    "approach_hem": "lower the gripper to the hem tab",
    "grasp_hem": "close the gripper on the hem tab",
    "lift_hem": "lift the grasped hem with the other arm",
    "carry_hem": "carry the hem toward the shoulders with the other arm",
    "fold_hem": "lower the hem onto the torso with the other arm",
    "release_hem": "release the folded hem",
    "settle": "move the open gripper to its home position",
}


def phase_instruction(agent: str, phase: str) -> str:
    if agent not in AGENTS or phase not in PHASE_WORDS:
        raise ValueError("unknown cloth agent or phase")
    return f"Fold the T-shirt. {agent.capitalize()} arm: {PHASE_WORDS[phase]}."


def local_proprio(world, agent: str) -> dict:
    """Read only this arm's measured tool pose and commanded gripper state.

    The returned fields can augment an observation without adding object or peer
    measurements. This shared transform keeps collection and inference frames
    identical. The clamp attachment flag remains separate from finger closure.
    """
    if agent not in AGENTS:
        raise ValueError("unknown cloth agent")
    rotation = world.data.site_xmat[world._sites[agent]].reshape(3, 3).copy()
    rotation_xyz(rotation)
    return {
        "ee_position": world.ee(agent).tolist(),
        "ee_rotation": rotation.tolist(),
        "gripper_closed": bool(world._grip[agent]),
        "grasped": world.holding[agent] is not None,
    }


def _finite(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"invalid {name}")
    return result


def rotation_xyz(rotation):
    """Measured local-to-world matrix to fixed-axis XYZ Euler angles, radians.

    This uses the standard R = Rz(z) @ Ry(y) @ Rx(x) convention. The fixed-tool
    simulator does not execute predicted rotations; recording these explicitly
    avoids calling an invented downward pose measured proprioception.
    """
    matrix = _finite(rotation, (3, 3), "tool rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-4
    ):
        raise ValueError("tool rotation must be a proper rotation matrix")
    cy = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    x = (
        np.arctan2(matrix[2, 1], matrix[2, 2])
        if cy > 1e-6
        else np.arctan2(-matrix[1, 2], matrix[1, 1])
    )
    return np.asarray(
        [
            x,
            np.arctan2(-matrix[2, 0], cy),
            np.arctan2(matrix[1, 0], matrix[0, 0]) if cy > 1e-6 else 0,
        ],
        dtype=np.float32,
    )


def smol_state(position, rotation, gripper_closed):
    """8D native input: own xyz, XYZ Euler, signed finger offsets in metres."""
    aperture = 0.012 if gripper_closed else 0.022
    return np.concatenate(
        (_finite(position, (3,), "tool position"), rotation_xyz(rotation), [aperture, -aperture])
    ).astype(np.float32)


def smol_action(target, rotation, grip):
    """7D label: absolute table xyz, measured XYZ Euler, +1 closed / -1 open."""
    return np.concatenate(
        (_finite(target, (3,), "target"), rotation_xyz(rotation), [1.0 if grip else -1.0])
    ).astype(np.float32)


def xvla_state(position, rotation, gripper_closed, agent):
    """20D native pose layout with unknown peer block zero, never ground truth."""
    if agent not in AGENTS:
        raise ValueError("unknown cloth agent")
    # Validate before using the same interleaved column layout as Soft-FOLD.
    rotation_xyz(rotation)
    own = np.concatenate(
        (
            _finite(position, (3,), "tool position"),
            np.asarray(rotation, dtype=np.float32)[:, :2].reshape(6),
            [float(bool(gripper_closed))],
        )
    )
    result = np.zeros(20, dtype=np.float32)
    offset = 0 if agent == "left" else 10
    result[offset : offset + 10] = own
    return result


def xvla_action(target, rotation, grip, agent):
    """Absolute pose label; mask identifies only actually executed xyz/grip."""
    result = xvla_state(target, rotation, grip, agent)
    mask = np.zeros(20, dtype=bool)
    offset = 0 if agent == "left" else 10
    mask[offset : offset + 3] = True
    mask[offset + 9] = True
    return result, mask


def future_chunk(arrays, index, length, *, family, agent):
    """Consecutive future commands within one phase/hold segment only.

    A changed instruction requires a new model call. Padding repeats the last
    valid command and is explicitly masked; it never borrows the next phase's
    label or another episode. ``index`` denotes the pre-step observation.
    """
    size = len(arrays["tick"])
    if type(index) is not int or not 0 <= index < size or type(length) is not int or length < 1:
        raise ValueError("invalid action chunk index or length")
    if family not in ("smolvla", "xvla") or agent not in AGENTS:
        raise ValueError("unknown native VLA family or agent")
    end = index + 1
    while end < min(index + length, size):
        if (
            arrays["tick"][end] != arrays["tick"][end - 1] + 1
            or arrays["phase"][end] != arrays["phase"][index]
            or arrays["hold"][end] != arrays["hold"][index]
        ):
            break
        end += 1
    chunks = []
    for i in range(index, end):
        if family == "smolvla":
            item = smol_action(arrays["target"][i], arrays["ee_rotation"][i], arrays["grip"][i])
        else:
            item, _ = xvla_action(
                arrays["target"][i], arrays["ee_rotation"][i], arrays["grip"][i], agent
            )
        chunks.append(item)
    valid = len(chunks)
    chunks.extend([chunks[-1]] * (length - valid))
    return {
        "actions": np.stack(chunks),
        "action_is_pad": np.arange(length) >= valid,
        "indices": np.minimum(np.arange(index, index + length), end - 1),
    }


def _json(path, value):
    # The manifest is a readiness marker for independent training processes.
    # Publish it atomically only after the complete document has been flushed.
    temporary = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError(path)
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_seed(seed):
    if type(seed) is not int or not 10000 <= seed < 2**32:
        raise ValueError("training seeds must be integers >=10000; evaluation seeds are forbidden")


def load_episode(path, *, require_success=True):
    """Read immutable, hash-verified training-only arrays without pickle."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("split") != "train":
        raise ValueError("not a native action training episode")
    _validate_seed(manifest["seed"])
    if require_success and manifest.get("success") is not True:
        raise ValueError("teacher episode did not complete a measured fold")
    agents = {}
    for agent in AGENTS:
        archive = path / f"{agent}.npz"
        if _sha(archive) != manifest["files"][archive.name]:
            raise ValueError(f"{agent} demonstration digest changed")
        with np.load(archive, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if set(arrays) != ARRAY_KEYS:
            raise ValueError("unexpected learner fields (possibly privileged state)")
        n = manifest["ticks"]
        if any(len(array) != n for array in arrays.values()) or not np.array_equal(
            arrays["tick"], np.arange(n)
        ):
            raise ValueError("demonstrations are not consecutive aligned pre-step records")
        shapes = {
            "rgb": (n, manifest["image_size"], manifest["image_size"], 3),
            "ee_position": (n, 3),
            "ee_rotation": (n, 3, 3),
            "target": (n, 3),
            "post_ee_position": (n, 3),
            "executed_target": (n, 3),
        }
        shapes.update(
            {key: (n,) for key in ("grasped", "gripper_closed", "phase", "hold", "grip", "tick")}
        )
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"invalid {agent} {key} shape")
        for key, array in arrays.items():
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite {agent} {key}")
        if (
            arrays["rgb"].dtype != np.uint8
            or arrays["phase"].dtype.kind not in "iu"
            or arrays["tick"].dtype.kind not in "iu"
            or any(
                arrays[key].dtype != np.bool_
                for key in ("grasped", "gripper_closed", "hold", "grip")
            )
            or not np.isin(arrays["phase"], np.arange(len(PHASES))).all()
        ):
            raise ValueError("invalid RGB or phase vocabulary")
        agents[agent] = arrays
    return manifest, agents


def collect_episode(output, *, seed=10020, max_ticks=1800, image_size=128, scenario="delayed"):
    """Collect a new full teacher rollout; never copy a held-out evaluation run.

    Collection uses existing original state-assisted policies solely as teaching
    controllers. Their hashes and all actual commands are recorded. An invalid
    fold remains explicitly failed and is rejected by ``load_episode`` by default.
    """
    _validate_seed(seed)
    if (
        type(max_ticks) is not int
        or max_ticks < 1
        or type(image_size) is not int
        or not (16 <= image_size <= 512)
    ):
        raise ValueError("invalid episode length or image size")
    if scenario not in ("clear", "delayed"):
        raise ValueError("collection supports clear or delayed scenarios")
    from .communication import Channel, LearnedFeatureCodec, observation_features
    from .policies import ClothPolicy
    from .runtime import ClothConfig, LocalPhaseController, PhaseJudge
    from .world import ClothWorld

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config = ClothConfig(
        seed=seed,
        mode="structured_text",
        scenario=scenario,
        video=False,
        planner="rules",
        coordination="memory",
        max_ticks=max_ticks,
    )
    rows = {agent: {key: [] for key in ARRAY_KEYS} for agent in AGENTS}
    world = None
    status, error, models, metrics, physics = "failed", None, {}, {}, {}
    phase_events = []
    terminal_settle_start = None
    try:
        world = ClothWorld(seed=seed, disturbances=scenario != "clear", fixture=True)
        policies = {agent: ClothPolicy(agent) for agent in AGENTS}
        # These tiny offline teachers stay on CPU. One BLAS worker avoids
        # oversubscribing the host while collecting independent MuJoCo worlds.
        policies["left"].torch.set_num_threads(1)
        controllers = {agent: LocalPhaseController(agent, PHASES) for agent in AGENTS}
        judge = PhaseJudge("rules")
        channel = Channel(
            f"original-native-training-{seed}",
            mode="structured_text",
            codec=LearnedFeatureCodec.load(config.codec),
            network=config.network(),
        )
        initial = world.snapshot()
        _json(output / "initial.json", initial)
        while world.tick < max_ticks:
            tick = world.tick
            observations = {agent: world.observe(agent) for agent in AGENTS}
            for agent in AGENTS:
                observations[agent]["rgb"] = world.render(
                    agent, width=image_size, height=image_size
                )
            if tick % config.publish_every == 0:
                for agent in AGENTS:
                    channel.publish(
                        agent,
                        {
                            "features": observation_features(observations[agent]),
                            "source_sha256": hashlib.sha256(
                                observations[agent]["rgb"].tobytes()
                            ).hexdigest(),
                        },
                        tick,
                    )
            channel.advance(tick)
            beliefs = {agent: channel.receive(agent, tick) for agent in AGENTS}
            actions = {}
            for agent in AGENTS:
                controller, observation = controllers[agent], observations[agent]
                decision = controller.decision(observation, beliefs[agent], communication=True)
                action = policies[agent].act(
                    observation,
                    controller.phase,
                    clearance=decision["clearance"],
                    hold=decision["hold"] or controller.done,
                )
                if policies[agent].last_prediction["coordinator_hold"]:
                    decision["hold"] = True
                actions[agent] = action
                proprio = local_proprio(world, agent)
                values = {
                    "rgb": observation["rgb"].copy(),
                    "ee_position": np.asarray(proprio["ee_position"], dtype=np.float32),
                    "ee_rotation": np.asarray(proprio["ee_rotation"], dtype=np.float32),
                    "grasped": proprio["grasped"],
                    "gripper_closed": proprio["gripper_closed"],
                    "phase": PHASES.index(controller.phase),
                    "hold": bool(decision["hold"]),
                    "target": np.asarray(action["target"], dtype=np.float32),
                    "grip": bool(action["grip"]),
                    "tick": tick,
                }
                for key, value in values.items():
                    rows[agent][key].append(value)
            if world.fixture_active and all(c.phase == "settle" for c in controllers.values()):
                world.release_fixture()
            world.step(actions)
            for agent in AGENTS:
                rows[agent]["post_ee_position"].append(world.ee(agent).astype(np.float32))
                rows[agent]["executed_target"].append(
                    world._targets[agent].astype(np.float32).copy()
                )
                event = controllers[agent].update(
                    observations[agent], world.observe(agent), actions[agent], beliefs[agent], judge
                )
                if event:
                    phase_events.append(event)
                    print(json.dumps({"seed": seed, **event}), flush=True)
            if all(controller.done for controller in controllers.values()):
                if world.metrics()["success"]:
                    status = "succeeded"
                    break
                if terminal_settle_start is None:
                    terminal_settle_start = world.tick
                elif world.tick - terminal_settle_start >= 100:
                    status = "rejected"
                    break
        else:
            status = "exhausted"
        models = {agent: policy.model_info for agent, policy in policies.items()}
    except Exception as exc:  # noqa: BLE001 - preserve failed teacher evidence
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if world is not None:
            final = world.snapshot()
            metrics, physics = final["metrics"], final["physics_config"]
            _json(output / "final.json", final)
            world.close()
    files = {}
    for agent in AGENTS:
        arrays = {key: np.asarray(values) for key, values in rows[agent].items()}
        np.savez_compressed(output / f"{agent}.npz", **arrays)
        files[f"{agent}.npz"] = _sha(output / f"{agent}.npz")
    manifest = {
        "schema": SCHEMA,
        "split": "train",
        "seed": seed,
        "scenario": scenario,
        "ticks": len(rows["left"]["tick"]),
        "image_size": image_size,
        "status": status,
        "success": status == "succeeded",
        "error": error,
        "metrics": metrics,
        "physics": physics,
        "wall_seconds": perf_counter() - started,
        "teacher": {
            "kind": "original task-trained state-assisted offline label generator",
            "models": models,
            "privileged_inputs": ["own local cloth landmarks"],
            "planner": "rules",
            "coordination": "in-memory structured text",
            "device": "cpu",
            "torch_threads": 1,
        },
        "learner_inputs": [
            "own local RGB",
            "own measured EE xyz/rotation",
            "own gripper command",
            "phase instruction",
        ],
        "forbidden_learner_inputs": [
            "cloth landmarks",
            "mesh",
            "peer pose or camera",
            "teacher hidden features",
            "future observations",
        ],
        "alignment": "Row t is captured before action t advances physics to t+1; 25 Hz.",
        "action_frame": "MuJoCo world/table frame; x across table, y toward shoulders, z up; metres.",
        "action_target": "Unclipped teacher command to world.step; executed_target records the subsequent slew-limited actuator setpoint.",
        "orientation": "Measured local-to-world tool matrix from MuJoCo site_xmat; native predicted rotations are not executed by this fixed-orientation rig.",
        "phases": list(PHASES),
        "instructions": {
            agent: [phase_instruction(agent, phase) for phase in PHASES] for agent in AGENTS
        },
        "phase_events": phase_events,
        "terminal_settle_ticks": world.tick - terminal_settle_start
        if world is not None and terminal_settle_start is not None
        else 0,
        "files": files,
        "initial_sha256": _sha(output / "initial.json")
        if (output / "initial.json").exists()
        else None,
        "final_sha256": _sha(output / "final.json") if (output / "final.json").exists() else None,
        "limitations": [
            "Demonstrations from one small simulated workcell; no real garment validation.",
            "Teacher success does not establish native VLA adaptation success.",
            "A geometric phase scaffold and idealized clamps remain explicit assumptions.",
        ],
    }
    _json(output / "manifest.json", manifest)
    return manifest


def extend_settling(source, output, *, max_extra_ticks=100):
    """Reproduce a rejected training episode exactly, then let its cloth settle.

    The source stays immutable and rejected. Its actual commands are reexecuted
    in a fresh world; cached RGB is reused only after initial/final mesh, every
    recorded arm pose, and final metrics agree exactly. Only new stationary
    open-gripper actions and freshly rendered local observations are appended.
    """
    from .world import ClothWorld

    if type(max_extra_ticks) is not int or not 1 <= max_extra_ticks <= 100:
        raise ValueError("terminal settling is bounded to 100 actual steps")
    source, output = Path(source), Path(output)
    original, arrays = load_episode(source, require_success=False)
    if original["success"] or not original.get("metrics", {}).get("physics_valid"):
        raise ValueError("only a physically valid rejected training episode can be extended")
    if any(int(arrays[a]["phase"][-1]) != len(PHASES) - 1 for a in AGENTS):
        raise ValueError("teacher must have reached the terminal phase on both arms")
    expected_initial = json.loads((source / "initial.json").read_text())
    expected_final = json.loads((source / "final.json").read_text())
    if (
        _sha(source / "initial.json") != original["initial_sha256"]
        or _sha(source / "final.json") != original["final_sha256"]
    ):
        raise ValueError("source physics evidence digest changed")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    world = ClothWorld(
        seed=original["seed"], disturbances=original["scenario"] != "clear", fixture=True
    )
    verification = {
        "source_manifest_sha256": _sha(source / "manifest.json"),
        "source": str(source),
        "reexecutes_teacher_models": False,
        "verified": False,
        "steps": 0,
    }
    extra = {agent: {key: [] for key in ARRAY_KEYS} for agent in AGENTS}
    final = None
    try:
        if not np.array_equal(world.vertices, np.asarray(expected_initial["vertices"])):
            raise ValueError("replay initial cloth mesh changed")
        for tick in range(original["ticks"]):
            if world.fixture_active and all(
                int(arrays[a]["phase"][tick]) == len(PHASES) - 1 for a in AGENTS
            ):
                world.release_fixture()
            actions = {
                agent: {
                    "target": arrays[agent]["target"][tick].tolist(),
                    "grip": bool(arrays[agent]["grip"][tick]),
                }
                for agent in AGENTS
            }
            world.step(actions)
            for agent in AGENTS:
                if not np.array_equal(
                    world.ee(agent).astype(np.float32), arrays[agent]["post_ee_position"][tick]
                ):
                    raise ValueError(f"recorded {agent} pose changed during replay at {tick}")
            verification["steps"] += 1
        reconstructed = world.snapshot()
        verification.update(
            {
                "final_vertices_bitwise_equal": np.array_equal(
                    world.vertices, np.asarray(expected_final["vertices"])
                ),
                "same_metrics": reconstructed["metrics"] == expected_final["metrics"],
                "same_robots": reconstructed["robots"] == expected_final["robots"],
                "same_fixture": reconstructed["fixture_active"] == expected_final["fixture_active"],
            }
        )
        if not all(
            verification[key]
            for key in (
                "final_vertices_bitwise_equal",
                "same_metrics",
                "same_robots",
                "same_fixture",
            )
        ):
            raise ValueError("reexecuted training prefix did not match original physics exactly")
        verification["verified"] = True
        print(
            json.dumps(
                {"seed": original["seed"], "prefix_replay_verified": True, "ticks": world.tick}
            ),
            flush=True,
        )
        for _ in range(max_extra_ticks):
            if world.metrics()["success"]:
                break
            actions = {}
            for agent in AGENTS:
                proprio = local_proprio(world, agent)
                action = {
                    "target": np.asarray(proprio["ee_position"], dtype=np.float32).tolist(),
                    "grip": False,
                }
                values = {
                    "rgb": world.render(
                        agent, width=original["image_size"], height=original["image_size"]
                    ),
                    "ee_position": np.asarray(proprio["ee_position"], dtype=np.float32),
                    "ee_rotation": np.asarray(proprio["ee_rotation"], dtype=np.float32),
                    "grasped": proprio["grasped"],
                    "gripper_closed": proprio["gripper_closed"],
                    "phase": len(PHASES) - 1,
                    "hold": True,
                    "target": np.asarray(action["target"], dtype=np.float32),
                    "grip": False,
                    "tick": world.tick,
                }
                for key, value in values.items():
                    extra[agent][key].append(value)
                actions[agent] = action
            world.step(actions)
            for agent in AGENTS:
                extra[agent]["post_ee_position"].append(world.ee(agent).astype(np.float32))
                extra[agent]["executed_target"].append(
                    world._targets[agent].astype(np.float32).copy()
                )
        final = world.snapshot()
    except Exception as exc:
        verification["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        world.close()
        _json(output / "prefix-verification.json", verification)
    files = {}
    for agent in AGENTS:
        combined = {
            key: np.concatenate([values, np.asarray(extra[agent][key], dtype=values.dtype)])
            if extra[agent][key]
            else values
            for key, values in arrays[agent].items()
        }
        np.savez_compressed(output / f"{agent}.npz", **combined)
        files[f"{agent}.npz"] = _sha(output / f"{agent}.npz")
    (output / "initial.json").write_bytes((source / "initial.json").read_bytes())
    _json(output / "final.json", final)
    manifest = {
        **original,
        "ticks": final["tick"],
        "success": final["metrics"]["success"],
        "status": "succeeded" if final["metrics"]["success"] else "rejected",
        "metrics": final["metrics"],
        "error": None,
        "files": files,
        "final_sha256": _sha(output / "final.json"),
        "terminal_settle_ticks": final["tick"] - original["ticks"],
        "continuation": {
            **verification,
            "additional_actions": "stationary open grippers",
            "wall_seconds": perf_counter() - started,
        },
        "wall_seconds": original["wall_seconds"] + perf_counter() - started,
    }
    _json(output / "manifest.json", manifest)
    return manifest


def _correction_native_fields(model, agent):
    """Select record fields by the publisher identity, never by arm or flags."""
    from ..smolvla import DEFAULT_MODEL_ID
    from .pretrained import XVLA_ID

    if model.get("model_id") == DEFAULT_MODEL_ID:
        return "smol", "native_action_executed", "inference_age_ticks"
    if agent == "right" and model.get("model_id") == XVLA_ID:
        return "xvla", "native_action_head_executed", "action_age_ticks"
    raise ValueError("unsupported source model identity for native corrections")


def collect_corrections(source, output, *, agent="left", should_cancel=None):
    """Label observed native-policy mistakes without changing their execution.

    This is a one-step counterfactual correction set, not a successful rollout
    and not a source of future action chunks. Only recorded native actions drive
    the fresh simulator. An offline teacher is queried at actual fresh native
    inference states, before replaying that state's original native command.
    """
    from PIL import Image

    from .policies import ClothPolicy
    from .world import ClothWorld

    if agent not in AGENTS:
        raise ValueError("correction collection requires a left or right arm")

    def cancelled():
        return should_cancel is not None and should_cancel()

    if cancelled():
        raise RuntimeError("correction collection cancelled before creation")
    source, output = Path(source), Path(output)
    summary = json.loads((source / "summary.json").read_text())
    config = summary["config"]
    seed = config["seed"]
    if type(seed) is not int or not (0 <= seed < 100 or 10000 <= seed < 2**32):
        raise ValueError("counterfactual correction data may not use benchmark evaluation seeds")
    if summary.get("models", {}).get(agent, {}).get("foundation_vla") is not True:
        raise ValueError("corrections require a recorded foundation-VLA execution")
    family, native_flag, age_key = _correction_native_fields(summary["models"][agent], agent)
    source_files = {
        name: _sha(source / name)
        for name in ("summary.json", "config.json", "events.jsonl.gz", "final.json")
    }
    # Earlier left-only development runs predate paired native snapshots. The
    # right-arm extension requires the complete paired native-output audit in
    # addition to the independent exact physics and RGB replay below.
    native_audit = None
    if agent == "right":
        from .native_audit import audit_native_run

        native_audit = audit_native_run(source)
        if native_audit.get("verified") is not True or any(
            native_audit.get("source_sha256", {}).get(name) != source_files[name]
            for name in ("summary.json", "events.jsonl.gz", "final.json")
        ):
            raise ValueError("right-arm corrections require a verified, source-bound native audit")
        if native_audit.get("model_families", {}).get(agent) != family:
            raise ValueError("native correction model identity differs from its source audit")
    expected_final = json.loads((source / "final.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    if native_audit is not None:
        _json(output / "native-audit.json", native_audit)
    world = ClothWorld(
        seed=seed, disturbances=config["scenario"] != "clear", fixture=config["fixture"]
    )
    teacher = ClothPolicy(agent)
    teacher.torch.set_num_threads(1)
    rows, images = [], {}
    phase = PHASES[0]
    started = perf_counter()
    proof = {
        "verified": False,
        "steps": 0,
        "source_files": source_files,
        "teacher_actions_executed": 0,
        "native_rgb_frames_verified": 0,
        "native_audit_verified": native_audit is not None,
    }
    try:
        if (
            hashlib.sha256(world.vertices.astype(np.float64).tobytes()).hexdigest()
            != summary["initial_physics_sha256"]
        ):
            raise ValueError("counterfactual replay initial cloth mesh changed")
        with gzip.open(source / "events.jsonl.gz", "rt") as stream:
            for line in stream:
                if cancelled():
                    raise RuntimeError("correction collection cancelled during replay")
                record = json.loads(line)
                if record["decision_tick"] != world.tick or record["tick"] != world.tick + 1:
                    raise ValueError("counterfactual replay source has missing physics ticks")
                prediction = record["predictions"][agent]
                if prediction.get("coordinator_hold") is not True and (
                    prediction.get(native_flag) is not True
                    or type(prediction.get(age_key)) is not int
                    or prediction[age_key] < 0
                ):
                    raise ValueError("native prediction fields differ from the source model family")
                sample = (
                    prediction.get(native_flag) is True
                    and prediction.get("coordinator_hold") is False
                    and prediction.get(age_key) == 0
                )
                if sample:
                    observation = world.observe(agent, rgb=True)
                    rgb_hash = hashlib.sha256(observation["rgb"].tobytes()).hexdigest()
                    if rgb_hash != prediction.get("rgb_sha256"):
                        raise ValueError("replayed local camera differs from actual native input")
                    if prediction.get("model_language_prompt", "").strip() != phase_instruction(
                        agent, phase
                    ):
                        raise ValueError("native input phase does not match pre-step instruction")
                    proprio = local_proprio(world, agent)
                    recorded_observation = record.get("observations", {}).get(agent, {})
                    for key, value in proprio.items():
                        if key in recorded_observation and not np.array_equal(
                            np.asarray(value), np.asarray(recorded_observation[key])
                        ):
                            raise ValueError(f"replayed own proprioception changed: {key}")
                    label = teacher.act(
                        observation,
                        phase,
                        clearance=record["decisions"][agent]["clearance"],
                        hold=False,
                    )
                    if teacher.last_prediction.get("coordinator_hold"):
                        raise ValueError("offline teacher cannot label a missing local observation")
                    filename = f"images/{world.tick:06d}.png"
                    Image.fromarray(observation["rgb"]).save(output / filename)
                    images[filename] = _sha(output / filename)
                    rows.append(
                        {
                            "agent": agent,
                            "tick": world.tick,
                            "phase": PHASES.index(phase),
                            "instruction": phase_instruction(agent, phase),
                            "rgb_file": filename,
                            "rgb_sha256": rgb_hash,
                            **proprio,
                            "target": label["target"],
                            "grip": label["grip"],
                            "source_native_inference_id": prediction["inference_id"],
                        }
                    )
                    proof["native_rgb_frames_verified"] += 1
                if cancelled():
                    raise RuntimeError("correction collection cancelled before physics step")
                for event in record["events"]:
                    if event.get("event") == "vacuum_fixture_released":
                        if event["tick"] != world.tick or not world.fixture_active:
                            raise ValueError("counterfactual replay fixture timing changed")
                        world.release_fixture()
                # The counterfactual label is deliberately never used here.
                snapshot = world.step(record["actions"])
                if (
                    snapshot["robots"] != record["robots"]
                    or snapshot["metrics"] != record["metrics"]
                ):
                    raise ValueError(f"counterfactual replay physics changed at tick {world.tick}")
                next_phase = record["phase"][agent]
                if next_phase not in PHASES or PHASES.index(next_phase) not in (
                    PHASES.index(phase),
                    PHASES.index(phase) + 1,
                ):
                    raise ValueError("counterfactual replay source changed phase sequence")
                phase = next_phase
                proof["steps"] += 1
        final = world.snapshot()
        proof.update(
            {
                "final_vertices_bitwise_equal": np.array_equal(
                    world.vertices, np.asarray(expected_final["vertices"])
                ),
                "same_final_metrics": final["metrics"] == expected_final["metrics"],
                "same_final_robots": final["robots"] == expected_final["robots"],
                "same_tick_count": world.tick == summary["ticks"] == expected_final["tick"],
                "same_source_artifacts": all(
                    _sha(source / name) == digest for name, digest in source_files.items()
                ),
            }
        )
        if not all(
            proof[key]
            for key in (
                "final_vertices_bitwise_equal",
                "same_final_metrics",
                "same_final_robots",
                "same_tick_count",
                "same_source_artifacts",
            )
        ):
            raise ValueError("counterfactual replay did not exactly reproduce source execution")
        if not rows or len(rows) != summary["models"][agent]["inference_calls"]:
            raise ValueError(
                "counterfactual labels do not cover exactly the native inference states"
            )
        proof["verified"] = True
    except Exception as exc:
        proof["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        world.close()
        _json(output / "verification.json", proof)
    _json(output / "corrections.json", rows)
    manifest = {
        "schema": CORRECTION_SCHEMA,
        "kind": "offline_counterfactual_one_step_corrections",
        "split": "development_corrections",
        "seed": seed,
        "agent": agent,
        "samples": len(rows),
        "source": str(source),
        "source_status": summary["status"],
        "source_success": summary["success"],
        "source_steps": summary["ticks"],
        "replay_verified": True,
        "native_audit_verified": native_audit is not None,
        "source_files": source_files,
        "source_controller": summary["models"][agent],
        "teacher": {
            "model": teacher.model_info,
            "actions_executed": 0,
            "device": "cpu",
            "purpose": "counterfactual label only; never applied to replay physics",
        },
        "future_action_chunks_available": False,
        "learner_inputs": [
            "own local RGB",
            "own measured pose",
            "own gripper command",
            "phase language",
        ],
        "label": "One counterfactual absolute Cartesian target and gripper command at the same pre-step native input.",
        "wall_seconds": perf_counter() - started,
        "files": {
            **images,
            **({"native-audit.json": _sha(output / "native-audit.json")} if native_audit else {}),
            "corrections.json": _sha(output / "corrections.json"),
            "verification.json": _sha(output / "verification.json"),
        },
        "limitations": [
            "Counterfactual teacher actions were not executed; their outcomes are unmeasured.",
            "This is development correction data, never a successful folding demonstration.",
            "One-step labels cannot be repeated and presented as measured future action chunks.",
        ],
    }
    _json(output / "manifest.json", manifest)
    return manifest


def load_corrections(path):
    """Load hash-verified single-action corrections; never synthesize futures."""
    from PIL import Image

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("schema") != CORRECTION_SCHEMA
        or manifest.get("split") != "development_corrections"
        or manifest.get("replay_verified") is not True
        or manifest.get("future_action_chunks_available") is not False
        or manifest.get("agent") not in AGENTS
    ):
        raise ValueError("not a verified one-step native correction dataset")
    seed = manifest.get("seed")
    agent = manifest["agent"]
    if type(seed) is not int or not (0 <= seed < 100 or 10000 <= seed < 2**32):
        raise ValueError("correction dataset may not contain benchmark evaluation seeds")
    if not {"corrections.json", "verification.json"} <= set(manifest["files"]):
        raise ValueError("correction verification and labels must be hash-bound")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or _sha(path / relative) != digest:
            raise ValueError("counterfactual correction artifact digest changed")
    proof = json.loads((path / "verification.json").read_text())
    if proof.get("verified") is not True or proof.get("teacher_actions_executed") != 0:
        raise ValueError("counterfactual replay verification does not establish label isolation")
    if agent == "right" or manifest.get("native_audit_verified"):
        if (
            manifest.get("native_audit_verified") is not True
            or proof.get("native_audit_verified") is not True
            or "native-audit.json" not in manifest["files"]
        ):
            raise ValueError("right-arm corrections require a hash-bound native audit")
        native_audit = json.loads((path / "native-audit.json").read_text())
        if native_audit.get("verified") is not True or any(
            native_audit.get("source_sha256", {}).get(name) != manifest["source_files"].get(name)
            for name in ("summary.json", "events.jsonl.gz", "final.json")
        ):
            raise ValueError("native correction audit does not match source artifacts")
        if agent == "right":
            controller = manifest.get("source_controller", {})
            if controller.get("foundation_vla") is not True:
                raise ValueError("native correction source controller is not a foundation VLA")
            family, _, _ = _correction_native_fields(controller, agent)
            audited_families = native_audit.get("model_families")
            # Historical paired audits supported only left Smol/right X-VLA.
            # They cannot establish provenance for a newer right Smol model.
            audited_family = "xvla" if audited_families is None else audited_families.get(agent)
            if audited_family != family:
                raise ValueError("native correction model identity differs from its source audit")
    rows = json.loads((path / "corrections.json").read_text())
    if len(rows) != manifest["samples"] or not rows:
        raise ValueError("invalid correction sample count")
    keys = (
        "rgb",
        "ee_position",
        "ee_rotation",
        "grasped",
        "gripper_closed",
        "phase",
        "target",
        "grip",
        "tick",
    )
    arrays = {key: [] for key in keys}
    previous = -1
    for row in rows:
        if (
            row.get("agent") != agent
            or type(row.get("tick")) is not int
            or row["tick"] <= previous
            or type(row.get("phase")) is not int
            or not 0 <= row["phase"] < len(PHASES)
            or row.get("instruction") != phase_instruction(agent, PHASES[row["phase"]])
            or any(type(row.get(key)) is not bool for key in ("grasped", "gripper_closed", "grip"))
        ):
            raise ValueError("invalid correction observation identity or order")
        previous = row["tick"]
        rgb_file = row["rgb_file"]
        if rgb_file not in manifest["files"]:
            raise ValueError("correction RGB is not hash-bound")
        with Image.open(path / rgb_file) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
            raise ValueError("decoded correction RGB differs from recorded native input")
        arrays["rgb"].append(rgb)
        smol_state(row["ee_position"], row["ee_rotation"], row["gripper_closed"])
        smol_action(row["target"], row["ee_rotation"], row["grip"])
        for key in keys[1:]:
            arrays[key].append(row[key])
    return manifest, {agent: {key: np.asarray(value) for key, value in arrays.items()}}


def recovery_behavior(label, *, agent, phase, active_ticks, grasped, hold, seed):
    """Bounded exploration of a free tool; never alter an attached cloth grasp."""
    target = _finite(label["target"], (3,), "recovery teacher target").astype(float)
    offset = np.zeros(3)
    if (
        phase in RECOVERY_PHASES
        and not grasped
        and not hold
        and not label["grip"]
        and (8 <= active_ticks < 16 or 24 <= active_ticks < 32)
    ):
        # Fixed seed/arm/phase signs keep collection reproducible without drawing
        # exploration noise from a native model's random stream.
        direction = 1 if (seed + PHASES.index(phase) + (agent == "right")) % 2 else -1
        if active_ticks >= 24:
            direction *= -1
        offset = np.asarray([0.020 * direction, 0.015 * -direction, 0.008])
    return {"target": (target + offset).tolist(), "grip": bool(label["grip"])}, offset


def collect_recovery(
    output, *, seed, max_ticks=1800, image_size=128, sample_every=4, should_cancel=None
):
    """Collect measured recovery states with one-step labels and separate behavior.

    Only free-tool approach phases receive bounded exploration. Each sampled
    camera/pose precedes its teacher label and actual physics step. Fresh 32px
    cameras suffice for the tiny offline teacher between samples; sampled learner
    images are fresh 128px cameras. Every actual action is retained in a separate
    audit log, including perturbations. This is never a future-action dataset.
    """
    from PIL import Image

    from .communication import Channel, LearnedFeatureCodec, observation_features
    from .policies import ClothPolicy
    from .runtime import ClothConfig, LocalPhaseController, PhaseJudge
    from .world import ClothWorld

    _validate_seed(seed)
    if (
        type(max_ticks) is not int
        or max_ticks < 1
        or type(image_size) is not int
        or not 32 <= image_size <= 512
        or type(sample_every) is not int
        or not 1 <= sample_every <= 16
    ):
        raise ValueError("invalid recovery collection dimensions or cadence")

    def cancelled():
        return should_cancel is not None and should_cancel()

    if cancelled():
        raise RuntimeError("recovery collection cancelled before creation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    config = ClothConfig(
        seed=seed,
        mode="structured_text",
        scenario="delayed",
        video=False,
        planner="rules",
        coordination="memory",
        max_ticks=max_ticks,
    )
    rows, files, phase_events = {agent: [] for agent in AGENTS}, {}, []
    previous_phase = dict.fromkeys(AGENTS)
    status, error, world, final = "failed", None, None, None
    models, terminal_start, started = {}, None, perf_counter()
    perturb_steps = dict.fromkeys(AGENTS, 0)
    steps = 0
    try:
        world = ClothWorld(seed=seed, disturbances=True, fixture=True)
        policies = {agent: ClothPolicy(agent) for agent in AGENTS}
        policies["left"].torch.set_num_threads(1)
        controllers = {agent: LocalPhaseController(agent, PHASES) for agent in AGENTS}
        judge = PhaseJudge("rules")
        channel = Channel(
            f"original-native-recovery-{seed}",
            mode="structured_text",
            codec=LearnedFeatureCodec.load(config.codec),
            network=config.network(),
        )
        _json(output / "initial.json", world.snapshot())
        with gzip.open(output / "behavior.jsonl.gz", "wt") as log:
            while world.tick < max_ticks:
                if cancelled():
                    status = "cancelled"
                    break
                tick = world.tick
                observations, sampled = {}, {}
                for agent in AGENTS:
                    phase = controllers[agent].phase
                    sampled[agent] = (phase in RECOVERY_PHASES or phase.startswith("grasp_")) and (
                        tick % sample_every == 0 or phase != previous_phase[agent]
                    )
                    observations[agent] = world.observe(agent)
                    size = image_size if sampled[agent] else 32
                    observations[agent]["rgb"] = world.render(agent, width=size, height=size)
                if tick % config.publish_every == 0:
                    for agent in AGENTS:
                        channel.publish(
                            agent,
                            {
                                "features": observation_features(observations[agent]),
                                "source_sha256": hashlib.sha256(
                                    observations[agent]["rgb"].tobytes()
                                ).hexdigest(),
                            },
                            tick,
                        )
                channel.advance(tick)
                beliefs = {agent: channel.receive(agent, tick) for agent in AGENTS}
                actions, labels, details, pending = {}, {}, {}, {}
                for agent in AGENTS:
                    controller, observation = controllers[agent], observations[agent]
                    phase = controller.phase
                    decision = controller.decision(observation, beliefs[agent], communication=True)
                    label = policies[agent].act(
                        observation,
                        phase,
                        clearance=decision["clearance"],
                        hold=decision["hold"] or controller.done,
                    )
                    hold = bool(policies[agent].last_prediction["coordinator_hold"])
                    if hold:
                        decision["hold"] = True
                    action, offset = recovery_behavior(
                        label,
                        agent=agent,
                        phase=phase,
                        active_ticks=controller.active_ticks,
                        grasped=observation["grasped"],
                        hold=hold,
                        seed=seed,
                    )
                    explored = bool(np.any(offset))
                    if explored:
                        perturb_steps[agent] += 1
                    actions[agent], labels[agent] = action, label
                    details[agent] = {
                        "phase": phase,
                        "behavior": "bounded_exploration" if explored else "teacher",
                        "offset_metres": offset.tolist(),
                        "pre_grasped": bool(observation["grasped"]),
                        "hold": hold,
                        "teacher_rgb_size": int(observation["rgb"].shape[0]),
                    }
                    if sampled[agent] and not hold:
                        rgb = observation["rgb"]
                        filename = f"images/{agent}-{tick:06d}.png"
                        Image.fromarray(rgb).save(output / filename)
                        files[filename] = _sha(output / filename)
                        pending[agent] = {
                            "agent": agent,
                            "tick": tick,
                            "phase": PHASES.index(phase),
                            "instruction": phase_instruction(agent, phase),
                            "rgb_file": filename,
                            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                            **local_proprio(world, agent),
                            "target": label["target"],
                            "grip": bool(label["grip"]),
                            "behavior_action": action,
                            "behavior": details[agent]["behavior"],
                            "teacher_label_executed": not explored,
                        }
                    previous_phase[agent] = phase
                # A cancellation during rendering/teacher inference cannot execute
                # a prepared exploration command or leave an unexecuted row ready.
                if cancelled():
                    status = "cancelled"
                    break
                if world.fixture_active and all(c.phase == "settle" for c in controllers.values()):
                    world.release_fixture()
                snapshot = world.step(actions)
                steps += 1
                log.write(
                    json.dumps(
                        {
                            "decision_tick": tick,
                            "tick": world.tick,
                            "actions": actions,
                            "teacher_labels": labels,
                            "details": details,
                            "robots": snapshot["robots"],
                            "metrics": snapshot["metrics"],
                            "fixture_active": world.fixture_active,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                for agent, row in pending.items():
                    rows[agent].append(row)
                for agent in AGENTS:
                    event = controllers[agent].update(
                        observations[agent],
                        world.observe(agent),
                        actions[agent],
                        beliefs[agent],
                        judge,
                    )
                    if event:
                        phase_events.append(event)
                        print(json.dumps({"seed": seed, **event}), flush=True)
                if all(controller.done for controller in controllers.values()):
                    if world.metrics()["success"]:
                        status = "succeeded"
                        break
                    if terminal_start is None:
                        terminal_start = world.tick
                    elif world.tick - terminal_start >= 100:
                        status = "rejected"
                        break
            else:
                status = "exhausted"
        models = {agent: policy.model_info for agent, policy in policies.items()}
    except Exception as exc:  # noqa: BLE001 - retain failed physical evidence
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if world is not None:
            final = world.snapshot()
            _json(output / "final.json", final)
            world.close()
    for agent in AGENTS:
        _json(output / f"{agent}-samples.json", rows[agent])
    for name in (
        "initial.json",
        "final.json",
        "behavior.jsonl.gz",
        "left-samples.json",
        "right-samples.json",
    ):
        if (output / name).exists():
            files[name] = _sha(output / name)
    manifest = {
        "schema": RECOVERY_SCHEMA,
        "kind": "original_bounded_exploration_one_step_recovery",
        "split": "train_recovery",
        "seed": seed,
        "agents": list(AGENTS),
        "samples": {agent: len(rows[agent]) for agent in AGENTS},
        "ticks": steps,
        "status": status,
        "error": error,
        "success": bool(final and final["metrics"]["success"] and status == "succeeded"),
        "metrics": final["metrics"] if final else {},
        "physics": final["physics_config"] if final else {},
        "image_size": image_size,
        "sample_every": sample_every,
        "perturbation": {
            "maximum_axis_offset_metres": [0.020, 0.015, 0.008],
            "phases": sorted(RECOVERY_PHASES),
            "steps": perturb_steps,
            "only_ungrasped_open_nonheld_tools": True,
        },
        "teacher": {
            "kind": "offline original task-trained CPU teacher",
            "models": models,
            "torch_threads": 1,
            "sampled_rgb_size": image_size,
            "unsampled_rgb_size": 32,
            "privileged_inputs": ["local cloth landmarks"],
        },
        "future_action_chunks_available": False,
        "learner_inputs": [
            "own local RGB",
            "own measured pose",
            "own gripper command",
            "phase language",
        ],
        "label": "One pre-step teacher Cartesian target/grip; actual behavior is recorded separately.",
        "alignment": "Sampled state and one-step label precede the recorded actual action and physics step.",
        "phase_events": phase_events,
        "files": files,
        "wall_seconds": perf_counter() - started,
        "limitations": [
            "Exploration actions are not teacher labels and must not be used as desirable targets.",
            "Counterfactual labels on exploration steps have no measured subsequent outcome.",
            "Physical success is measured separately; one-step labels do not establish a native policy succeeds.",
            "No future chunks are available, including after a successful physical trajectory.",
        ],
    }
    _json(output / "manifest.json", manifest)
    return manifest


def load_recovery(path):
    """Return only own learner inputs/one-step labels, excluding behavior/audit state."""
    from PIL import Image

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    _validate_seed(manifest.get("seed"))
    if (
        manifest.get("schema") != RECOVERY_SCHEMA
        or manifest.get("split") != "train_recovery"
        or manifest.get("agents") != list(AGENTS)
        or manifest.get("future_action_chunks_available") is not False
        or manifest.get("status") not in ("succeeded", "rejected", "exhausted")
        or manifest.get("error") is not None
    ):
        raise ValueError("not a complete one-step recovery dataset")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or _sha(path / relative) != digest:
            raise ValueError("recovery artifact digest changed")
    required = {
        "behavior.jsonl.gz",
        "initial.json",
        "final.json",
        "left-samples.json",
        "right-samples.json",
    }
    if not required <= set(manifest["files"]):
        raise ValueError("recovery evidence is not hash-bound")
    with gzip.open(path / "behavior.jsonl.gz", "rt") as stream:
        behavior = [json.loads(line) for line in stream]
    if len(behavior) != manifest["ticks"]:
        raise ValueError("recovery behavior has missing physics steps")
    for index, record in enumerate(behavior):
        if record["decision_tick"] != index or record["tick"] != index + 1:
            raise ValueError("recovery behavior has missing physics steps")
        for agent in AGENTS:
            details = record["details"][agent]
            offset = _finite(details["offset_metres"], (3,), "exploration offset")
            label, action = record["teacher_labels"][agent], record["actions"][agent]
            target = _finite(label["target"], (3,), "teacher target")
            executed = _finite(action["target"], (3,), "behavior target")
            if (
                type(label["grip"]) is not bool
                or action["grip"] != label["grip"]
                or not np.allclose(executed, target + offset, atol=1e-7, rtol=0)
                or details["behavior"] != ("bounded_exploration" if np.any(offset) else "teacher")
            ):
                raise ValueError("recovery behavior differs from its recorded teacher and offset")
            if np.any(np.abs(offset) > np.asarray([0.020, 0.015, 0.008]) + 1e-8):
                raise ValueError("recovery exploration exceeded declared bound")
            if np.any(offset) and (
                details["pre_grasped"]
                or details["hold"]
                or details["phase"] not in RECOVERY_PHASES
                or record["actions"][agent]["grip"]
            ):
                raise ValueError("recovery exploration changed an attached or held tool")
    final = json.loads((path / "final.json").read_text())
    if (
        not behavior
        or final["tick"] != manifest["ticks"]
        or final["robots"] != behavior[-1]["robots"]
        or final["metrics"] != behavior[-1]["metrics"]
        or final["metrics"] != manifest["metrics"]
        or manifest["success"]
        != (manifest["status"] == "succeeded" and final["metrics"]["success"])
    ):
        raise ValueError("recovery physical outcome does not match evidence")
    keys = (
        "rgb",
        "ee_position",
        "ee_rotation",
        "grasped",
        "gripper_closed",
        "phase",
        "target",
        "grip",
        "tick",
    )
    result = {}
    for agent in AGENTS:
        rows = json.loads((path / f"{agent}-samples.json").read_text())
        if not rows or len(rows) != manifest["samples"][agent]:
            raise ValueError("invalid recovery sample count")
        arrays, previous = {key: [] for key in keys}, -1
        for row in rows:
            tick, phase = row.get("tick"), row.get("phase")
            if (
                row.get("agent") != agent
                or type(tick) is not int
                or not previous < tick < len(behavior)
                or type(phase) is not int
                or not 0 <= phase < len(PHASES)
                or row.get("instruction") != phase_instruction(agent, PHASES[phase])
                or any(
                    type(row.get(key)) is not bool for key in ("grasped", "gripper_closed", "grip")
                )
            ):
                raise ValueError("invalid recovery identity or temporal order")
            previous = tick
            record = behavior[tick]
            if (
                row["behavior_action"] != record["actions"][agent]
                or {"target": row["target"], "grip": row["grip"]} != record["teacher_labels"][agent]
                or row["teacher_label_executed"] != (row["behavior"] == "teacher")
                or row["behavior"] != record["details"][agent]["behavior"]
                or PHASES[phase] != record["details"][agent]["phase"]
                or row["grasped"] != record["details"][agent]["pre_grasped"]
            ):
                raise ValueError("recovery teacher label was confused with actual behavior")
            if row["rgb_file"] not in manifest["files"]:
                raise ValueError("recovery camera is not hash-bound")
            with Image.open(path / row["rgb_file"]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if (
                rgb.shape != (manifest["image_size"], manifest["image_size"], 3)
                or hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]
            ):
                raise ValueError("recovery camera differs from recorded pre-step input")
            smol_state(row["ee_position"], row["ee_rotation"], row["gripper_closed"])
            smol_action(row["target"], row["ee_rotation"], row["grip"])
            arrays["rgb"].append(rgb)
            for key in keys[1:]:
                arrays[key].append(row[key])
        result[agent] = {key: np.asarray(value) for key, value in arrays.items()}
    return manifest, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=10020)
    parser.add_argument("--max-ticks", type=int, default=1800)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--scenario", choices=("clear", "delayed"), default="delayed")
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="collect bounded one-step recovery labels for both arms",
    )
    parser.add_argument(
        "--extend-settling",
        type=Path,
        help="reexecute and append up to 100 stationary steps to a training episode",
    )
    parser.add_argument(
        "--corrections-from",
        type=Path,
        help="label one-step corrections during exact native-action replay",
    )
    parser.add_argument("--correction-agent", choices=AGENTS, default="left")
    args = parser.parse_args()
    if (
        sum(bool(value) for value in (args.extend_settling, args.corrections_from, args.recovery))
        > 1
    ):
        parser.error("choose one collection operation")
    if args.recovery:
        result = collect_recovery(
            args.output, seed=args.seed, max_ticks=args.max_ticks, image_size=args.image_size
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("schema", "seed", "samples", "success", "status", "error")
                }
            )
        )
        raise SystemExit(0 if result["status"] in ("succeeded", "rejected", "exhausted") else 1)
    if args.corrections_from:
        result = collect_corrections(
            args.corrections_from, args.output, agent=args.correction_agent
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("schema", "seed", "samples", "source_success", "replay_verified")
                }
            )
        )
        raise SystemExit(0)
    result = (
        extend_settling(args.extend_settling, args.output)
        if args.extend_settling
        else collect_episode(
            args.output,
            seed=args.seed,
            max_ticks=args.max_ticks,
            image_size=args.image_size,
            scenario=args.scenario,
        )
    )
    print(json.dumps({k: result[k] for k in ("seed", "status", "ticks", "metrics", "error")}))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
