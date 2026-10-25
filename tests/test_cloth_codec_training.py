"""Training provenance and safe grasp-identity migration remain separate from evaluation."""

import importlib.util
import json
from pathlib import Path

import pytest

from relay_lab.cloth.communication import LearnedFeatureCodec, observation_features

SPEC = importlib.util.spec_from_file_location(
    "train_cloth_codec", Path(__file__).parents[1] / "scripts/train_cloth_codec.py"
)
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


@pytest.fixture
def sources(tmp_path):
    rows = []
    for tick in range(64):
        target = ("none", "sleeve", "hem")[tick % 3]
        observation = {
            "ee_position": [0.2 + tick / 1000, 0, 0.2],
            "grasped": target != "none",
            "grasp_target": target,
            "obstacle": None,
            "landmarks": {
                name: [0.2, index * 0.1, 0.008]
                for index, name in enumerate(("hem", "sleeve", "shoulder"))
            },
        }
        rows.append(
            {
                "agent": "left" if tick % 2 else "right",
                "tick": tick,
                "features": observation_features(observation),
            }
        )
    negative = {
        key: value
        for key, value in rows[0]["features"].items()
        if key not in {"grasp_sleeve", "grasp_hem"}
    }
    calibration, collection, trajectory = [
        tmp_path / f"{name}.json" for name in ("calibration", "collection", "trajectory")
    ]
    calibration.write_text(json.dumps([negative, {**negative, "grasped": 1.0}]))
    collection.write_text(json.dumps({"training_seeds": [10000], "evaluation_seeds_used": False}))
    trajectory.write_text(json.dumps({"seed": 10004, "split": "training", "rows": rows}))
    return calibration, collection, trajectory, tmp_path / "output"


def test_exact_training_inputs_migration_and_frozen_codec_are_reproducible(sources):
    first = training.fit_codec(*sources)
    assert first["source_dimensions"] == 23 and first["latent_dimensions"] == 8
    assert first["migration"]["discarded_unknown_attachment"] == 1
    assert first["calibration_retained"] == 1 and first["trajectory_rows"] == 64
    assert first["training_seeds"] == [10000, 10004]
    checkpoint = LearnedFeatureCodec.load(sources[-1] / "communication.json")
    assert checkpoint.sha256 == first["model"]["checkpoint_sha256"]
    inputs = json.loads((sources[-1] / "communication-training-v2.json").read_text())
    assert len(inputs["rows"]) == 65
    assert "training reconstruction only" in first["training_reconstruction"]["split"]
    second = training.fit_codec(*sources)
    assert first == second


@pytest.mark.parametrize(
    "mutation",
    [
        {"seed": 101},
        {"split": "evaluation"},
        {"rows": []},
    ],
)
def test_evaluation_sources_never_enter_fitting(sources, mutation):
    trajectory = sources[2]
    data = json.loads(trajectory.read_text())
    trajectory.write_text(json.dumps({**data, **mutation}))
    with pytest.raises(ValueError):
        training.fit_codec(*sources)
    assert not (sources[-1] / "communication.json").exists()


def test_positive_grasp_trajectory_requires_actual_attachment_identity(sources):
    trajectory = sources[2]
    data = json.loads(trajectory.read_text())
    data["rows"][1]["features"].pop("grasp_sleeve")
    trajectory.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Recollect"):
        training.fit_codec(*sources)


def test_replaced_codec_is_archived_by_its_previous_identity(sources):
    first = training.fit_codec(*sources)
    trajectory = sources[2]
    data = json.loads(trajectory.read_text())
    data["rows"][0]["features"]["ee_x"] += 0.01
    trajectory.write_text(json.dumps(data))
    second = training.fit_codec(*sources)
    old = first["model"]["checkpoint_sha256"]
    assert old != second["model"]["checkpoint_sha256"]
    assert (
        LearnedFeatureCodec.load(sources[-1] / "archive" / f"communication-{old}.json").sha256
        == old
    )
