#!/usr/bin/env python3
"""Freeze an 8D local-feature codec from original training measurements only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from relay_lab.cloth.communication import (
    FEATURES,
    LearnedFeatureCodec,
    canonical,
    migrate_training_features,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(canonical(value) + "\n")


def fit_codec(calibration, collection, trajectory, output, *, dimensions=8):
    calibration, collection, trajectory, output = map(
        Path, (calibration, collection, trajectory, output)
    )
    original = json.loads(calibration.read_text())
    collection_info = json.loads(collection.read_text())
    rollout = json.loads(trajectory.read_text())
    seeds = collection_info.get("training_seeds", [])
    if (
        not seeds
        or any(type(seed) is not int or seed < 10000 for seed in seeds)
        or collection_info.get("evaluation_seeds_used") is not False
        or rollout.get("split") != "training"
        or type(rollout.get("seed")) is not int
        or rollout["seed"] < 10000
        or not rollout.get("rows")
    ):
        raise ValueError("Codec fitting requires explicitly documented training-only seed sources")
    calibration_rows, migration = migrate_training_features(original)
    trajectory_rows = []
    for row in rollout["rows"]:
        if row.get("agent") not in ("left", "right") or type(row.get("tick")) is not int:
            raise ValueError("Training trajectory lacks local observation provenance")
        features = row["features"]
        if set(features) != set(FEATURES):
            raise ValueError(
                "Recollect trajectory with measured grasp identity; do not infer from phase"
            )
        if (
            features["grasp_sleeve"] not in (0, 1)
            or features["grasp_hem"] not in (0, 1)
            or features["grasp_sleeve"] + features["grasp_hem"] != features["grasped"]
        ):
            raise ValueError("Trajectory has ambiguous measured attachment identity")
        trajectory_rows.append(features)
    if not any(row["grasp_hem"] == 1 for row in trajectory_rows) or not any(
        row["grasp_sleeve"] == 1 for row in trajectory_rows
    ):
        raise ValueError(
            "Training trajectory must contain measured sleeve and hem grasp observations"
        )
    rows = calibration_rows + trajectory_rows
    training_seeds = sorted(set(seeds + [rollout["seed"]]))
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / f"communication-trajectory-{rollout['seed']}.json"
    source_path.write_bytes(trajectory.read_bytes())
    inputs = {
        "schema": "relay.cloth.codec-training.v2",
        "split": "training",
        "training_seeds": training_seeds,
        "features": list(FEATURES),
        "rows": rows,
        "migration": migration,
        "sources": [
            {
                "name": calibration.name,
                "sha256": sha256(calibration),
                "retained_rows": len(calibration_rows),
            },
            {
                "name": source_path.name,
                "sha256": sha256(source_path),
                "retained_rows": len(trajectory_rows),
            },
            {"name": collection.name, "sha256": sha256(collection)},
        ],
    }
    input_path = output / "communication-training-v2.json"
    write_json(input_path, inputs)
    np.save(
        output / "communication-training-v2.npy",
        np.asarray([[row[key] for key in FEATURES] for row in rows], dtype=np.float64),
    )
    metadata = {
        "source": "original local MuJoCo measurements, calibration and complete training trajectory",
        "training_seeds": training_seeds,
        "split": "training",
        "evaluation_seeds_used": False,
        "combined_input_sha256": sha256(input_path),
        "migration": migration,
        "representation": "standardized PCA of measured local features; not a VLA hidden state",
    }
    codec = LearnedFeatureCodec.fit(rows, dimensions=dimensions, training=metadata)
    checkpoint = output / "communication.json"
    if checkpoint.is_file():
        previous = json.loads(checkpoint.read_text())
        old_sha = hashlib.sha256(canonical(previous).encode()).hexdigest()
        if old_sha != codec.sha256:
            archive = output / "archive"
            archive.mkdir(exist_ok=True)
            (archive / f"communication-{old_sha}.json").write_bytes(checkpoint.read_bytes())
    codec.save(checkpoint)
    reconstruction = codec.evaluate(rows)
    reconstruction["split"] = "training reconstruction only; no held-out performance claim"
    report = {
        "schema": "relay.cloth.codec-fit.v2",
        "model": codec.model_info,
        "combined_input_sha256": sha256(input_path),
        "source_dimensions": len(FEATURES),
        "latent_dimensions": dimensions,
        "training_seeds": training_seeds,
        "calibration_retained": len(calibration_rows),
        "trajectory_rows": len(trajectory_rows),
        "migration": migration,
        "training_reconstruction": reconstruction,
        "heldout_evaluation": "Evaluate recorded test observations after freezing; never refit on them.",
    }
    write_json(output / "communication-fit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("checkpoints/cloth-policies/communication-training.json"),
    )
    parser.add_argument(
        "--collection", type=Path, default=Path("checkpoints/cloth-policies/collection.json")
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/cloth-policies"))
    args = parser.parse_args()
    report = fit_codec(args.calibration, args.collection, args.trajectory, args.output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "training_reconstruction"}
        )
    )


if __name__ == "__main__":
    main()
