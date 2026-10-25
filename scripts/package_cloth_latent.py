"""Publish inspectable learned-message evidence to the local cloth replay folder."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from relay_lab.cloth.latent_adapter import MODES
from relay_lab.cloth.latent_training import load_pairs, sha, write_json


def mean_arm(report, key):
    values = [arm[key] for arm in report["arms"].values()]
    return float(np.mean(values)) if all(v is not None for v in values) else None


def live_hold_audit(trial):
    hazard_decisions, missed, clear_decisions, false_holds = 0, 0, 0, 0
    for record in trial.get("records", []):
        decisions = record["decisions"]
        # This is a post-hoc evaluator only. The two private observations were
        # never joined before inference or supplied to either action head.
        hazard = any(d["local_observation"]["obstacle"] for d in decisions.values())
        for decision in decisions.values():
            prediction = decision["prediction"]
            hold = prediction["learned_hold"] or prediction["missing_message_hold"]
            hazard_decisions += int(hazard)
            clear_decisions += int(not hazard)
            missed += int(hazard and not hold)
            false_holds += int(not hazard and hold)
    return {
        "kind": "posthoc-live-hold-diagnostic",
        "hazard_decisions": hazard_decisions,
        "missed_hazard_holds": missed,
        "clear_decisions": clear_decisions,
        "unnecessary_holds": false_holds,
        "criterion": "Same teaching criterion: if either local observation reports an obstacle, both arms should hold.",
    }


def package(dataset, cache, checkpoints, comparison, live, output):
    dataset, cache, checkpoints, comparison, live, output = map(
        Path, (dataset, cache, checkpoints, comparison, live, output)
    )
    load_pairs(dataset, cache)  # Verify original aligned sources before publishing.
    evaluation = json.loads((checkpoints / "evaluation.json").read_text())
    transport = json.loads((comparison / "manifest.json").read_text())
    trial = json.loads((live / "report.json").read_text())
    live_audit = live_hold_audit(trial)
    dataset_sha256 = sha(dataset / "rows.json")
    training = evaluation.get("training")
    if not isinstance(training, dict) or training.get("dataset_rows_sha256") != dataset_sha256:
        raise ValueError("Adapter evaluation used another original dataset")
    cache_sha256 = {arm: sha(cache / f"{arm}.npz") for arm in ("left", "right")}
    if training.get("feature_cache_sha256") != cache_sha256:
        raise ValueError("Adapter evaluation used different frozen feature caches")
    if transport["dataset_sha256"] != dataset_sha256:
        raise ValueError("Transport comparison used another original dataset")
    for mode in MODES:
        if sha(checkpoints / f"{mode}.pt") != evaluation["modes"][mode]["checkpoint_sha256"]:
            raise ValueError("Adapter differs from evaluated checkpoint")
    for attempt in transport["attempts"]:
        if (
            attempt["status"] == "completed"
            and attempt["model"]["checkpoint_sha256"]
            != evaluation["modes"][attempt["mode"]]["checkpoint_sha256"]
        ):
            raise ValueError("Transport used a different learned adapter")
    if "model" in trial and trial["model"]["checkpoint_sha256"] not in {
        row["checkpoint_sha256"] for row in evaluation["modes"].values()
    }:
        raise ValueError("Live execution used an unevaluated adapter")
    output.mkdir(parents=True, exist_ok=False)
    for name, source, files in (
        ("dataset", dataset, ["metadata.json", "rows.json", "rgb.npz"]),
        (
            "features",
            cache,
            [f"{arm}.{ext}" for arm in ("left", "right") for ext in ("json", "npz")],
        ),
        ("models", checkpoints, ["evaluation.json", *(f"{mode}.pt" for mode in MODES)]),
        ("live", live, ["report.json", "communication.json", "overview.mp4"]),
    ):
        directory = output / name
        directory.mkdir()
        for filename in files:
            source_file = source / filename
            if source_file.is_file():
                shutil.copy2(source_file, directory / filename)
            elif name != "live":
                raise FileNotFoundError(source_file)
    shutil.copytree(comparison, output / "transport")
    # The replay server advertises Content-Encoding:gzip for these retained logs.
    for directory in (output / "transport").iterdir():
        if directory.is_dir():
            for path in directory.glob("*.json"):
                compressed = path.with_suffix(".json.gz")
                compressed.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
                path.unlink()
    static_rows = []
    for mode in MODES:
        metrics = evaluation["modes"][mode]
        static_rows.append(
            {
                "mode": mode,
                **{
                    key: mean_arm(metrics["test"], key)
                    for key in ("hold_accuracy", "hidden_peer_accuracy", "mean_action_error_mm")
                },
                "zero_peer_accuracy": mean_arm(metrics["test_zero_peer"], "hold_accuracy"),
                "shuffled_peer_accuracy": mean_arm(metrics["test_shuffled_peer"], "hold_accuracy"),
            }
        )
    transport_rows = []
    original_rows = {row["id"]: row for row in json.loads((dataset / "rows.json").read_text())}
    for attempt in transport["attempts"]:
        row = {
            "mode": attempt["mode"],
            "scenario": attempt["scenario"],
            "status": attempt["status"],
        }
        if attempt["status"] == "completed":
            row.update(
                {
                    key: float(np.mean([arm[key] for arm in attempt["metrics"].values()]))
                    for key in ("hold_accuracy", "hazard_hold_rate", "clear_go_rate")
                }
            )
            channel = attempt["communication"]
            decisions = json.loads(
                (
                    comparison / f"{attempt['scenario']}-{attempt['mode']}" / "decisions.json"
                ).read_text()
            )
            private = [
                decision
                for record in decisions
                for decision in record["decisions"].values()
                if not original_rows[decision["row_id"]]["labels"]["local_obstacle"]
            ]
            row["posthoc_hidden_peer_accuracy"] = sum(
                d["actual_hold"] == d["expected_hold"] for d in private
            ) / len(private)
            row["posthoc_hidden_peer_count"] = len(private)
            row["mean_packet_bytes"] = (
                channel["packet_bytes"] / channel["published"] if channel["published"] else 0
            )
        transport_rows.append(row)
    previous, grasps = {"left": False, "right": False}, 0
    for record in trial.get("records", []):
        for arm, state in record.get("measured_robots", {}).items():
            holding = state["holding"] is not None
            grasps += int(holding and not previous[arm])
            previous[arm] = holding
    results = {
        "kind": "trained-mixed-vla-message-diagnostic",
        "heading": "Learned messages",
        "static": static_rows,
        "transport": transport_rows,
        "notes": [
            f"The short live trial missed {live_audit['missed_hazard_holds']} of {live_audit['hazard_decisions']} required obstacle holds. Inference, message delivery and physical actuation ran, but the fresh-scene diagnostic does not establish reliable live coordination.",
            "Two original training episodes, one development episode, and one held-out episode. The held-out set has 20 physical states with three private-obstacle conditions; adjacent and counterfactual samples are correlated.",
            "Hidden-peer accuracy uses only locally clear views. Clear and peer-obstructed inputs are identical for the receiving arm, making the labels balanced and requiring peer information to distinguish them.",
            "Both actual frozen backbones feed learned 64D encoders. State-assisted replacement heads were trained jointly; the native VLA action heads were not used.",
            "The transport comparison replays changing observations on frozen physical states. It does not integrate actions. Delays/drops use the same seeded schedule across formats, with exact SpacetimeDB readback.",
            "These adapters were trained on fresh messages only. The unused age input is initialized to zero; missing/expired messages trigger an explicit hold. Neither mechanism establishes learned stale-message recovery.",
            "Structured text carries exact simulator measurements, whereas latent messages carry learned visual/language information. This is a system comparison with different information sources, not a matched-content bandwidth claim.",
        ],
        "live": {
            "status": "execution_completed_coordination_failed"
            if trial["status"] == "completed" and live_audit["missed_hazard_holds"]
            else trial["status"],
            "hold_audit": live_audit,
            "calls_per_arm": len(trial.get("records", [])),
            "ticks": trial.get("final", {}).get("tick"),
            "grasp_count": grasps,
            "full_task_success": trial.get("metrics", {}).get("success", False),
            "video": "learned-latent/live/overview.mp4"
            if (output / "live/overview.mp4").is_file()
            else None,
        },
        "sources": {
            "training_and_static_results": "learned-latent/models/evaluation.json",
            "transport_results": "learned-latent/transport/manifest.json",
            "live_trial": "learned-latent/live/report.json",
            "original_dataset": "learned-latent/dataset/metadata.json",
            "CoLA": "https://arxiv.org/html/2610.15337v1",
            "VLA_Fabric": "https://arxiv.org/html/2610.23018v1",
        },
    }
    write_json(output / "results.json", results)
    write_json(
        output / "files.json",
        {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "cache", "checkpoints", "comparison", "live", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    package(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
