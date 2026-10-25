"""Packaging must bind evaluated adapters to the supplied dataset and caches."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def package_fixture(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "package_cloth_latent.py"
    spec = importlib.util.spec_from_file_location("cloth_latent_package_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Alignment and cache shape are covered by loader tests. This focused check
    # uses small byte fixtures and never loads a foundation model or checkpoint.
    monkeypatch.setattr(module, "load_pairs", lambda *_: None)
    paths = {
        name: tmp_path / name
        for name in ("dataset", "cache", "checkpoints", "comparison", "live", "output")
    }
    for name, directory in paths.items():
        if name != "output":
            directory.mkdir()

    def write(directory, name, value):
        (directory / name).write_text(json.dumps(value))

    write(paths["dataset"], "rows.json", [])
    write(paths["dataset"], "metadata.json", {"fixture": True})
    (paths["dataset"] / "rgb.npz").write_bytes(b"test-only-rgb")
    for arm in ("left", "right"):
        (paths["cache"] / f"{arm}.npz").write_bytes(f"test-only-{arm}-features".encode())
        write(paths["cache"], f"{arm}.json", {"fixture": True})
    training = {
        "dataset_rows_sha256": module.sha(paths["dataset"] / "rows.json"),
        "feature_cache_sha256": {
            arm: module.sha(paths["cache"] / f"{arm}.npz") for arm in ("left", "right")
        },
    }
    metrics = {
        "arms": {
            arm: {"hold_accuracy": 0.5, "hidden_peer_accuracy": 0.5, "mean_action_error_mm": 1.0}
            for arm in ("left", "right")
        }
    }
    modes = {}
    for mode in module.MODES:
        checkpoint = paths["checkpoints"] / f"{mode}.pt"
        checkpoint.write_bytes(f"test-only-{mode}-checkpoint".encode())
        modes[mode] = {
            "checkpoint_sha256": module.sha(checkpoint),
            "test": metrics,
            "test_zero_peer": metrics,
            "test_shuffled_peer": metrics,
        }
    write(paths["checkpoints"], "evaluation.json", {"training": training, "modes": modes})
    write(
        paths["comparison"],
        "manifest.json",
        {"dataset_sha256": training["dataset_rows_sha256"], "attempts": []},
    )
    write(
        paths["live"],
        "report.json",
        {"status": "completed", "records": [], "metrics": {"success": False}},
    )
    return module, paths


@pytest.mark.parametrize(
    "directory,filename,error",
    [
        ("dataset", "rows.json", "another original dataset"),
        ("cache", "left.npz", "different frozen feature caches"),
        ("cache", "right.npz", "different frozen feature caches"),
    ],
)
def test_mismatched_training_sources_are_rejected_before_output(
    package_fixture, directory, filename, error
):
    module, paths = package_fixture
    source = paths[directory] / filename
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match=error):
        module.package(**paths)
    assert not paths["output"].exists()


def test_matching_training_sources_can_be_packaged(package_fixture):
    module, paths = package_fixture
    results = module.package(**paths)
    assert [row["mode"] for row in results["static"]] == list(module.MODES)
    hashes = json.loads((paths["output"] / "files.json").read_text())
    assert hashes["dataset/rows.json"] == module.sha(paths["dataset"] / "rows.json")
    assert hashes["features/left.npz"] == module.sha(paths["cache"] / "left.npz")
    assert hashes["features/right.npz"] == module.sha(paths["cache"] / "right.npz")
