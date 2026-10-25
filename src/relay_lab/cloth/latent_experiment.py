"""Recorded-scene channel stress test and bounded live learned-head execution.

The stress test changes private observations on held-out frozen physical states;
it does not integrate those predicted actions. The live command runs fresh
backbone inference and real MuJoCo actions, without a geometric target fallback.
Neither command is a full-fold benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .communication import AGENTS, NetworkConfig, canonical, observation_features
from .latent_adapter import MODES, LearnedLatentPolicy
from .latent_channel import LatentChannel, feature_source_hash
from .latent_training import implementation_hashes, load_pairs, sha, write_json


def journal(backend):
    if backend == "memory":
        return None
    if backend != "spacetime":
        raise ValueError("Unknown learned-message backend")
    from .spacetime import SpacetimeJournal

    return SpacetimeJournal(worker_id="cloth-latent-worker")


def _score(records):
    result = {}
    for arm in AGENTS:
        decisions = [r["decisions"][arm] for r in records]
        hazards = [d for d in decisions if d["expected_hold"]]
        clear = [d for d in decisions if not d["expected_hold"]]
        result[arm] = {
            "decisions": len(decisions),
            "hold_accuracy": sum(d["actual_hold"] == d["expected_hold"] for d in decisions)
            / len(decisions),
            "hazard_hold_rate": sum(d["actual_hold"] for d in hazards) / len(hazards),
            "clear_go_rate": sum(not d["actual_hold"] for d in clear) / len(clear),
            "unsafe_go_count": sum(not d["actual_hold"] for d in hazards),
            "false_hold_count": sum(d["actual_hold"] for d in clear),
            "missing_message_holds": sum(
                d["prediction"]["missing_message_hold"] for d in decisions
            ),
        }
    return result


def compare(dataset, cache, checkpoints, output, *, backend="spacetime"):
    dataset, cache, checkpoints, output = map(Path, (dataset, cache, checkpoints, output))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    _, sources, _ = load_pairs(dataset, cache)
    rows = json.loads((dataset / "rows.json").read_text())
    features = {}
    for arm in AGENTS:
        stored = np.load(cache / f"{arm}.npz", allow_pickle=False)
        features.update(
            {int(i): vector for i, vector in zip(stored["row_indices"], stored["features"])}
        )
    snapshots = {}
    for index, row in enumerate(rows):
        if row["split"] == "test":
            snapshots.setdefault(row["snapshot_id"], {}).setdefault(row["counterfactual"], {})[
                row["agent"]
            ] = (index, row)
    # All modes see the identical sequence, fixed before any model runs. Each
    # hazard is private; clear/left/right labels do not enter policy inputs.
    sequence = []
    for snapshot in sorted(snapshots):
        for condition in ("clear", "left_obstacle", "right_obstacle"):
            for _ in range(4):
                sequence.append(snapshots[snapshot][condition])
    scenarios = {
        "one_tick": NetworkConfig(
            seed=104, min_delay_ticks=1, max_delay_ticks=1, drop_probability=0, ttl_ticks=24
        ),
        "delayed_dropped": NetworkConfig(
            seed=104, min_delay_ticks=2, max_delay_ticks=18, drop_probability=0.15, ttl_ticks=24
        ),
    }
    manifest = {
        "kind": "held-out-frozen-scene-observation-sequence",
        "implementation_sha256": implementation_hashes(),
        "backend": backend,
        "actions_integrated": False,
        "full_task_success_evaluation": False,
        "model_inference": "Previously computed actual frozen backbone features, verified cache; learned heads execute now.",
        "observation_schedule": "New private condition every 16 diagnostic ticks; decision/publication every 4 ticks.",
        "sequence_sha256": hashlib.sha256(
            canonical([p["left"][1]["id"] for p in sequence]).encode()
        ).hexdigest(),
        "dataset_sha256": sha(dataset / "rows.json"),
        "attempts": [
            {"mode": mode, "scenario": name, "status": "pending"}
            for name in scenarios
            for mode in MODES
        ],
    }
    write_json(output / "manifest.json", manifest)
    for attempt in manifest["attempts"]:
        mode, scenario = attempt["mode"], attempt["scenario"]
        destination = output / f"{scenario}-{mode}"
        destination.mkdir()
        started = time.perf_counter()
        channel = None
        records = []
        try:
            policy = LearnedLatentPolicy(checkpoints / f"{mode}.pt")
            channel = LatentChannel(
                "latent-" + uuid.uuid4().hex[:16],
                policy=policy,
                network=scenarios[scenario],
                backend=journal(backend),
            )
            for index, pair in enumerate(sequence):
                tick = index * 4
                for arm in AGENTS:
                    row_index, row = pair[arm]
                    channel.publish(
                        arm,
                        features[row_index],
                        observation_features(row["observation"]),
                        tick,
                        extractor_id=sources[arm]["model"]["feature_space_id"],
                        source_sha256=hashlib.sha256(
                            canonical(
                                {
                                    "row": row["id"],
                                    "rgb": row["rgb_sha256"],
                                    "feature": hashlib.sha256(
                                        features[row_index].astype("<f4").tobytes()
                                    ).hexdigest(),
                                }
                            ).encode()
                        ).hexdigest(),
                    )
                channel.advance(tick, state={"tick": tick, "diagnostic_index": index})
                decisions = {}
                for arm in AGENTS:
                    row_index, row = pair[arm]
                    belief = channel.receive(arm, tick)
                    action, prediction = policy.act(
                        row["observation"],
                        row["phase"],
                        features[row_index],
                        belief,
                        extractor_id=sources[arm]["model"]["feature_space_id"],
                    )
                    decisions[arm] = {
                        "row_id": row["id"],
                        "belief": belief,
                        "action": action,
                        "prediction": prediction,
                        "expected_hold": row["labels"]["coordinated_hold"],
                        "actual_hold": prediction["learned_hold"]
                        or prediction["missing_message_hold"],
                    }
                records.append({"tick": tick, "decisions": decisions})
                # Actual scheduled arrivals occur on the diagnostic clock even
                # between decisions, as in a process awaiting its next action.
                for substep in range(1, 4):
                    channel.advance(tick + substep)
            # Independent DB query verifies the entire durable journal, not only
            # the immediate ACK path. Memory remains explicitly nondurable.
            history = channel.backend.history() if backend == "spacetime" else None
            published = [e["packet"] for e in channel.events if e["kind"] == "published"]
            if history is not None and canonical(
                [p for frame in history for p in frame["packets"]]
            ) != canonical(published):
                raise RuntimeError(
                    "Independent Spacetime readback disagrees with sent learned packets"
                )
            attempt.update(
                status="completed",
                metrics=_score(records),
                model=policy.model_info,
                communication=channel.metrics,
                independent_db_frames=len(history) if history is not None else None,
            )
        except Exception as error:  # noqa: BLE001 - retain every predeclared failed attempt
            attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            attempt["elapsed_seconds"] = time.perf_counter() - started
            write_json(destination / "decisions.json", records)
            if channel:
                write_json(
                    destination / "communication.json",
                    {"config": channel.config, "events": channel.events},
                )
            write_json(output / "manifest.json", manifest)
        print(
            canonical(
                {
                    k: v
                    for k, v in attempt.items()
                    if k in ("mode", "scenario", "status", "metrics", "error")
                }
            ),
            flush=True,
        )
    return manifest


def live(checkpoint, output, *, backend="spacetime", calls=12, seed=105):
    import imageio.v2 as imageio

    from .features import FrozenVLAFeatureExtractor
    from .latent_dataset import private_rgb
    from .world import ClothWorld

    if type(calls) is not int or not 2 <= calls <= 32:
        raise ValueError("Bounded live trial requires 2–32 calls")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world = writer = channel = None
    extractors, records = {}, []
    report = {
        "kind": "live-frozen-foundation-features-and-learned-action-heads",
        "implementation_sha256": implementation_hashes(),
        "full_task_success_evaluation": False,
        "native_action_heads_used": False,
        "teacher_targets_used": False,
        "calls_per_arm": calls,
        "seed": seed,
        "phase": "approach_sleeve_above",
        "status": "starting",
        "checkpoint_path": str(checkpoint),
        "environment_cleanup": "The fixture is released after the bounded budget, outside learned control; no settling or completion is inferred.",
    }
    write_json(output / "report.json", report)
    started = time.perf_counter()
    try:
        policy = LearnedLatentPolicy(checkpoint)
        report["model"] = policy.model_info
        for arm, family in (("left", "smolvla"), ("right", "xvla")):
            extractors[arm] = FrozenVLAFeatureExtractor(family, arm)
            if extractors[arm].space_id != policy.extractor_ids[arm]:
                raise ValueError("Live feature extractor differs from adapter training")
        world = ClothWorld(
            seed=seed, disturbances=True, obstacle_schedule=((8, 24, "left"), (32, 44, "right"))
        )
        report["initial"] = world.snapshot()
        channel = LatentChannel(
            "latent-live-" + uuid.uuid4().hex[:12],
            policy=policy,
            network=NetworkConfig(
                seed=seed, min_delay_ticks=1, max_delay_ticks=3, drop_probability=0, ttl_ticks=24
            ),
            backend=journal(backend),
        )
        writer = imageio.get_writer(
            str(output / "overview.mp4"), fps=25, codec="libx264", quality=7
        )
        for call in range(calls):
            tick = world.tick
            observations = {
                arm: {**world.observe(arm), "rgb": private_rgb(world, arm, 128)} for arm in AGENTS
            }
            features = {
                arm: extractors[arm].extract(
                    observations[arm], "Fold the T-shirt. approach sleeve above."
                )
                for arm in AGENTS
            }
            for arm in AGENTS:
                channel.publish(
                    arm,
                    features[arm].vector,
                    observation_features(observations[arm]),
                    tick,
                    extractor_id=features[arm].metadata["feature_space_id"],
                    source_sha256=feature_source_hash(features[arm]),
                )
            channel.advance(tick, state={"tick": tick, "call": call})
            actions, decisions = {}, {}
            for arm in AGENTS:
                belief = channel.receive(arm, tick)
                actions[arm], prediction = policy.act(
                    observations[arm],
                    "approach_sleeve_above",
                    features[arm].vector,
                    belief,
                    extractor_id=features[arm].metadata["feature_space_id"],
                )
                decisions[arm] = {
                    "feature": features[arm].metadata,
                    "belief": belief,
                    "action": actions[arm],
                    "prediction": prediction,
                    "local_observation": {k: v for k, v in observations[arm].items() if k != "rgb"},
                }
            record = {
                "observation_tick": tick,
                "after_tick": tick,
                "executed_ticks": 0,
                "decisions": decisions,
                "execution_status": "pending",
            }
            records.append(record)
            for _ in range(4):
                snapshot = world.step(actions)
                record.update(
                    after_tick=world.tick,
                    executed_ticks=record["executed_ticks"] + 1,
                    measured_robots=snapshot["robots"],
                    metrics=snapshot["metrics"],
                    execution_status="partial",
                )
                channel.advance(world.tick)
                writer.append_data(world.render("overview", width=960, height=720))
            record["execution_status"] = "complete"
            print(canonical({"live_call": call + 1, "tick": world.tick}), flush=True)
        world.release_fixture()
        report.update(
            status="completed",
            final=world.snapshot(),
            metrics=world.metrics(),
            communication=channel.metrics,
        )
        history = channel.backend.history() if backend == "spacetime" else None
        if history is not None:
            published = [e["packet"] for e in channel.events if e["kind"] == "published"]
            if canonical([p for frame in history for p in frame["packets"]]) != canonical(
                published
            ):
                raise RuntimeError("Live database packet audit mismatch")
            report["independent_db_frames"] = len(history)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        cleanup_errors = []
        for name, resource in (("video", writer), ("world", world)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:  # noqa: BLE001 - preserve partial physical evidence
                    cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        if channel:
            report["communication"] = channel.metrics
            write_json(
                output / "communication.json", {"config": channel.config, "events": channel.events}
            )
        report["records"] = records
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stress = commands.add_parser("compare")
    for name in ("dataset", "cache", "checkpoints", "output"):
        stress.add_argument("--" + name, type=Path, required=True)
    stress.add_argument("--backend", choices=("memory", "spacetime"), default="spacetime")
    probe = commands.add_parser("live")
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--backend", choices=("memory", "spacetime"), default="spacetime")
    probe.add_argument("--calls", type=int, default=12)
    args = vars(parser.parse_args())
    command = args.pop("command")
    compare(**args) if command == "compare" else live(**args)


if __name__ == "__main__":
    main()
