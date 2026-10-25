"""Measured cloth rollouts; policies see only their local sensor and delivered mail.

The phase scaffold, guarded progression, and idealized point grasps are explicit
experimental assumptions. A terminal phase is never sufficient for task success.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .communication import Channel, LearnedFeatureCodec, NetworkConfig, observation_features

AGENTS = ("left", "right")
DEFAULT_CODEC = Path("checkpoints/cloth-policies/communication.json")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass(frozen=True)
class ClothConfig:
    seed: int = 101
    mode: str = "hybrid"
    scenario: str = "delayed"
    coordination: str = "memory"
    planner: str = "rules"
    max_ticks: int = 1800
    publish_every: int = 4
    video_every: int = 4
    video: bool = True
    fixture: bool = True
    codec: str = str(DEFAULT_CODEC)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if self.mode not in ("embeddings", "structured_text", "hybrid", "none"):
            raise ValueError("unknown communication representation")
        if self.scenario not in ("clear", "obstacles", "delayed", "blackout"):
            raise ValueError("unknown disturbance scenario")
        if self.coordination not in ("memory", "spacetime") or self.planner not in ("rules", "llm"):
            raise ValueError("unknown coordinator or planner")
        if type(self.fixture) is not bool:
            raise ValueError("fixture must be a boolean")
        for name in ("max_ticks", "publish_every", "video_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    def network(self):
        if self.scenario in ("clear", "obstacles"):
            return NetworkConfig(
                seed=self.seed,
                min_delay_ticks=1,
                max_delay_ticks=1,
                drop_probability=0.0,
                ttl_ticks=24,
            )
        return NetworkConfig(
            seed=self.seed,
            min_delay_ticks=2,
            max_delay_ticks=32 if self.scenario == "blackout" else 18,
            drop_probability=0.3 if self.scenario == "blackout" else 0.15,
            ttl_ticks=24,
        )


class PhaseJudge:
    """Optional actual Qwen review at eligible local phase boundaries.

    Each request contains one robot's local status and already delivered peer facts;
    it cannot consult the spectator world or the other robot's current observation.
    """

    def __init__(self, mode="rules"):
        from ..llm import LocalLLM

        self.mode = mode
        self.llm = LocalLLM(max_tokens=100) if mode == "llm" else None
        self.calls = 0
        self.traces = []

    def advance(self, agent, phase, next_phase, observation, belief):
        if self.llm is None:
            return True
        context = {
            "robot": agent,
            "phase": phase,
            "next_phase": next_phase,
            "local_grasped": observation["grasped"],
            "local_grasp_target": observation["grasp_target"],
            "local_obstacle": observation.get("obstacle"),
            "delivered_peer_report": belief,
            "local_phase_target_reached": True,
        }
        messages = [
            {
                "role": "system",
                "content": "You review one arm's next step in a cloth-folding "
                "sequence. The local controller has verified it reached its current target. "
                "Approve progression unless the supplied observations indicate an active obstacle. "
                "You receive only local information and delivered peer reports; unknown peer facts "
                "are not proof of failure. Never invent measurements or claim task success. "
                'Return only JSON {"advance":true,"reason":"brief explanation"}. /no_think',
            },
            {"role": "user", "content": json.dumps(context)},
        ]
        for attempt in range(2):
            generation = self.llm.generate(messages)
            self.calls += 1
            self.traces.append({"input": context, "generation": generation.to_dict()})
            try:
                result = json.loads(generation.text)
                if (
                    set(result) != {"advance", "reason"}
                    or type(result["advance"]) is not bool
                    or not isinstance(result["reason"], str)
                ):
                    raise ValueError("invalid phase review")
                return result["advance"]
            except (ValueError, TypeError):
                if attempt:
                    raise RuntimeError("Qwen phase review failed; no reference fallback") from None
                messages.append({"role": "user", "content": "Return the exact JSON schema only."})
        raise AssertionError("unreachable")

    def close(self):
        if self.llm:
            self.llm.close()


class LocalPhaseController:
    """Same explicit progression rules for every representation and robot.

    Only own observations and received peer reports enter this controller. Temporal
    limits bound failed phases; they do not award completion or fabricate geometry.
    """

    def __init__(self, agent, phases):
        self.agent, self.phases = agent, tuple(phases)
        self.index, self.active_ticks, self.wall_ticks = 0, 0, 0
        self.done = False
        self.waiting_ticks = 0
        self.phase_timeouts = []
        self.last_decision = {}
        self.next_review_tick = 0

    @property
    def phase(self):
        return self.phases[self.index]

    def decision(self, observation, belief, *, communication=True):
        features = belief["features"] if belief else {}
        obstacle = bool(observation.get("obstacle"))
        peer_obstacle = features.get("obstacle_active", 0.0) >= 0.5
        carrying = bool(observation["grasped"])
        lost_peer = communication and carrying and belief is None
        # Local hazards must be communicated before they can stop the other arm.
        hold = obstacle or peer_obstacle or lost_peer
        reason = (
            "local obstacle"
            if obstacle
            else "peer obstacle report"
            if peer_obstacle
            else "await fresh peer report"
            if lost_peer
            else "execute local subgoal"
        )
        # Synchronize the shared hem lift using the received grasp measurement.
        if (
            communication
            and self.phase == "lift_hem"
            and self.active_ticks < 2
            and features.get("grasp_hem", 0.0) < 0.5
        ):
            hold, reason = True, "await peer hem grasp"
        # Keep the two grasp points together along a shared fold trajectory.
        # Only received geometry enters this check; in a hybrid packet these
        # coordinates still come from its learned representation.
        if communication and carrying and features.get("grasp_hem", 0.0) >= 0.5 and not hold:
            own = observation["ee_position"]
            if self.phase == "lift_hem" and own[2] - features.get("ee_z", own[2]) > 0.065:
                hold, reason = True, "synchronize lift with peer height"
            elif self.phase == "carry_hem" and own[1] - features.get("ee_y", own[1]) > 0.09:
                hold, reason = True, "synchronize fold with peer progress"
            elif self.phase == "fold_hem" and features.get("ee_z", own[2]) - own[2] > 0.065:
                hold, reason = True, "synchronize lowering with peer height"
        self.last_decision = {
            "hold": hold,
            "reason": reason,
            "clearance": 0.23,
            "peer_packet_id": belief["packet_id"] if belief else None,
            "peer_observation_tick": belief["observation_tick"] if belief else None,
            "peer_age_ticks": belief["age_ticks"] if belief else None,
            "peer_obstacle_prediction": float(features.get("obstacle_active", 0.0)),
        }
        if hold:
            self.waiting_ticks += 1
        return self.last_decision

    def update(self, before, after, action, belief, judge):
        if self.done:
            return None
        self.wall_ticks += 1
        if self.last_decision.get("hold"):
            return None
        self.active_ticks += 1
        error = float(np.linalg.norm(np.asarray(after["ee_position"]) - action["target"]))
        phase = self.phase
        minimum = 90 if phase == "fold_hem" else 45
        if phase == "settle":
            minimum = 160
        reached = error < 0.035
        if phase.startswith("grasp_"):
            reached = bool(after["grasped"]) and after.get("grasp_target") == phase.removeprefix(
                "grasp_"
            )
        elif phase.startswith("release_") or phase == "settle":
            reached = reached and not after["grasped"]
        if self.active_ticks >= 180:
            self.phase_timeouts.append({"phase": phase, "tick": after["tick"], "error_m": error})
            raise RuntimeError(f"{self.agent} could not complete {phase} within 180 active ticks")
        if not (self.active_ticks >= minimum and reached):
            return None
        if after["tick"] < self.next_review_tick:
            return None
        next_phase = self.phases[min(self.index + 1, len(self.phases) - 1)]
        if not judge.advance(self.agent, phase, next_phase, after, belief):
            self.next_review_tick = after["tick"] + 25
            return None
        if self.index == len(self.phases) - 1:
            self.done = True
        else:
            self.index += 1
        self.active_ticks = self.wall_ticks = 0
        self.next_review_tick = 0
        return {
            "event": "phase_completed",
            "agent": self.agent,
            "phase": phase,
            "next_phase": self.phase,
            "tick": after["tick"],
            "measured_error_m": error,
        }


def run_cloth(config: ClothConfig, output: str | Path):
    from .policies import ClothPolicy
    from .world import ClothWorld

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = "cloth-" + uuid.uuid4().hex[:16]
    write_json(output / "config.json", {**asdict(config), "run_id": run_id})
    world = judge = channel = None
    writers, frames, events = {}, [], []
    error, status = None, "starting"
    policies, controllers, initial, initial_hash, first_rgb_hash = {}, {}, None, None, {}
    started = time.perf_counter()
    try:
        codec = LearnedFeatureCodec.load(config.codec)
        world = ClothWorld(
            seed=config.seed, disturbances=config.scenario != "clear", fixture=config.fixture
        )
        initial = world.snapshot()
        initial_hash = hashlib.sha256(
            np.asarray(initial["vertices"], dtype=np.float64).tobytes()
        ).hexdigest()
        policies = {a: ClothPolicy(a) for a in AGENTS}
        # Keep the exact evaluated weights with their evidence, even after refits.
        (output / "models").mkdir()
        for a, policy in policies.items():
            shutil.copyfile(policy.path, output / "models" / f"{a}.pt")
        codec.save(output / "models" / "communication.json")
        controllers = {a: LocalPhaseController(a, world.PHASES) for a in AGENTS}
        judge = PhaseJudge(config.planner)
        backend = None
        if config.coordination == "spacetime":
            from .spacetime import SpacetimeJournal

            backend = SpacetimeJournal()
        channel = Channel(
            run_id, mode=config.mode, codec=codec, network=config.network(), backend=backend
        )
        if config.video:
            import imageio.v2 as imageio
            from PIL import Image

            for camera in ("overview", "left", "right"):
                writers[camera] = imageio.get_writer(
                    str(output / f"{camera}.mp4"),
                    fps=1 / (world.CONTROL_DT * config.video_every),
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )
            Image.fromarray(world.render(width=1280, height=800)).save(output / "poster.png")
        status = "running"
        previous_events = 0
        with gzip.open(output / "events.jsonl.gz", "wt") as log:
            while world.tick < config.max_ticks:
                tick = world.tick
                observations = {a: world.observe(a, rgb=True) for a in AGENTS}
                if tick == 0:
                    first_rgb_hash = {
                        a: hashlib.sha256(observations[a]["rgb"].tobytes()).hexdigest()
                        for a in AGENTS
                    }
                if tick % config.publish_every == 0:
                    for a in AGENTS:
                        channel.publish(
                            a,
                            {
                                "features": observation_features(observations[a]),
                                "source_sha256": hashlib.sha256(
                                    observations[a]["rgb"].tobytes()
                                ).hexdigest(),
                            },
                            tick,
                        )
                channel.advance(
                    tick,
                    state={"tick": tick, "phases": {a: controllers[a].phase for a in AGENTS}}
                    if tick % config.publish_every == 0
                    else None,
                )
                beliefs = {a: channel.receive(a, tick) for a in AGENTS}
                decisions, actions = {}, {}
                for a in AGENTS:
                    decisions[a] = controllers[a].decision(
                        observations[a], beliefs[a], communication=config.mode != "none"
                    )
                    actions[a] = policies[a].act(
                        observations[a],
                        controllers[a].phase,
                        clearance=decisions[a]["clearance"],
                        hold=decisions[a]["hold"] or controllers[a].done,
                    )
                    if policies[a].last_prediction["coordinator_hold"] and not decisions[a]["hold"]:
                        decisions[a]["hold"] = True
                        decisions[a]["reason"] = (
                            "local sequence completed"
                            if controllers[a].done
                            else "required local landmark unavailable"
                        )
                if world.fixture_active and all(c.phase == "settle" for c in controllers.values()):
                    world.release_fixture()
                snapshot = world.step(actions)
                outcome_events = world.events[previous_events:]
                previous_events = len(world.events)
                transition_error = None
                for a in AGENTS:
                    try:
                        transition = controllers[a].update(
                            observations[a], world.observe(a), actions[a], beliefs[a], judge
                        )
                        if transition:
                            outcome_events.append(transition)
                    except Exception as exc:  # noqa: BLE001 - log executed tick, then re-raise
                        transition_error = exc
                        outcome_events.append(
                            {
                                "event": "phase_error",
                                "agent": a,
                                "tick": world.tick,
                                "error": str(exc),
                            }
                        )
                        break
                events.extend(outcome_events)
                record = {
                    "tick": world.tick,
                    "decision_tick": tick,
                    "time": world.tick * world.CONTROL_DT,
                    "video_time": len(frames) * config.video_every * world.CONTROL_DT,
                    "phase": {a: controllers[a].phase for a in AGENTS},
                    "robots": snapshot["robots"],
                    "metrics": snapshot["metrics"],
                    "decisions": decisions,
                    "actions": actions,
                    "predictions": {a: policies[a].last_prediction for a in AGENTS},
                    "observations": {
                        a: {k: v for k, v in observations[a].items() if k != "rgb"} for a in AGENTS
                    },
                    "beliefs": beliefs,
                    "obstacle": snapshot["obstacle"],
                    "events": outcome_events,
                    "communication": {
                        k: channel.counter[k]
                        for k in (
                            "published",
                            "delivered",
                            "dropped",
                            "expired",
                            "out_of_order",
                            "packet_bytes",
                        )
                    },
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
                if tick % config.video_every == 0:
                    frames.append(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("observations", "predictions", "actions")
                        }
                    )
                if config.video and tick % config.video_every == 0:
                    writers["overview"].append_data(world.render(width=1280, height=800))
                    for a in AGENTS:
                        writers[a].append_data(world.render(a, width=384, height=384))
                if transition_error:
                    raise transition_error
                if all(c.done for c in controllers.values()):
                    status = "succeeded" if world.metrics()["success"] else "rejected"
                    break
            else:
                status = "exhausted"
    except Exception as exc:  # noqa: BLE001 - record every failed benchmark condition
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:

        def cleanup_failure(stage, exc):
            nonlocal status, error
            status = "failed"
            detail = f"{stage}: {type(exc).__name__}: {exc}"
            error = f"{error}; {detail}" if error else detail

        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - retain episode on artifact failure
                cleanup_failure("video close", exc)
        final = {}
        if world:
            try:
                final = world.snapshot()
                write_json(output / "final.json", final)
                if config.video:
                    from PIL import Image

                    Image.fromarray(world.render(width=1280, height=800)).save(output / "final.png")
            except Exception as exc:  # noqa: BLE001 - preserve summary if rendering fails
                cleanup_failure("final evidence", exc)
            finally:
                try:
                    world.close()
                except Exception as exc:  # noqa: BLE001 - bounded cleanup reporting
                    cleanup_failure("world close", exc)
        if judge:
            try:
                judge.close()
            except Exception as exc:  # noqa: BLE001 - bounded cleanup reporting
                cleanup_failure("planner close", exc)
        metrics = final.get("metrics", {})
        summary = {
            "schema": "relay.cloth.run.v1",
            "run_id": run_id,
            "config": asdict(config),
            "mode": config.mode,
            "scenario": config.scenario,
            "seed": config.seed,
            "status": status,
            "success": status == "succeeded",
            "error": error,
            "ticks": world.tick if world else 0,
            "wall_seconds": time.perf_counter() - started,
            "initial_physics_sha256": initial_hash,
            "initial_rgb_sha256": first_rgb_hash,
            "metrics": metrics,
            "physics": final.get("physics_config", {}),
            "fixture_active": final.get("fixture_active"),
            "models": {a: p.model_info for a, p in policies.items()},
            "communication": channel.metrics if channel else {},
            "planner": {"mode": config.planner, "calls": judge.calls if judge else 0},
            "wait_ticks": {a: c.waiting_ticks for a, c in controllers.items()},
            "phase": {a: c.phase for a, c in controllers.items()},
            "limitations": [
                "Single-layer T-shirt-shaped flex sheet; material parameters are not calibrated to real fabric.",
                "Distance-gated physical vertex clamps approximate grasping; not frictional finger grasp validation.",
                "An explicit vacuum-table fixture stabilizes the upper garment during folding; released before final scoring.",
                "Task-trained policies use local landmark state as well as RGB; not foundation-model control.",
                "Shared learned representations encode measured local features, not foundation-VLA hidden states.",
                "Synthetic message delays advance with simulation; wall-clock database/inference calls pause the worker.",
            ],
        }
        write_json(output / "summary.json", summary)
        write_json(output / "trace.json", {"dt": 0.04, "frames": frames, "events": events})
        write_json(output / "planner.json", judge.traces if judge else [])
        if channel:
            with gzip.open(output / "communication.json.gz", "wt") as stream:
                json.dump({"config": channel.config, "events": channel.events}, stream)
    return summary
