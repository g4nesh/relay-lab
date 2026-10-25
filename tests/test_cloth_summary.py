"""Reporting cannot quietly drop failed initialization attempts."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cloth_summary", Path(__file__).parents[1] / "scripts/summarize_cloth_experiment.py"
)
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def failed_experiment(path):
    modes = ("embeddings", "structured_text", "hybrid")
    rows = [
        {
            "mode": mode,
            "seed": 101,
            "scenario": "delayed",
            "success": False,
            "status": "failed",
            "error": "Database unavailable during setup",
            "ticks": 0,
        }
        for mode in modes
    ]
    comparison = {
        "recorded_runs": 3,
        "planned_runs": 3,
        "protocol_sha256": "test-protocol",
        "conclusion": "No successful conditions",
        "planner": "rules",
        "coordination": "spacetime",
        "seeds": [101],
        "scenarios": ["delayed"],
        "episodes": rows,
    }
    (path / "comparison.json").write_text(json.dumps(comparison))
    (path / "manifest.json").write_text(json.dumps({"runs": [{"id": m} for m in modes]}))
    return comparison


def test_setup_failures_remain_in_denominators_without_fake_measurements(tmp_path):
    failed_experiment(tmp_path)
    report = reporting.summarize(tmp_path)
    for mode in report["mode_summary"]:
        assert mode["successes"] == 0 and mode["runs"] == 1
        assert mode["mean_worst_hem_error_m"] is None
        assert mode["mean_packet_bytes"] is None
        assert mode["missing_geometry_runs"] == mode["missing_traffic_runs"] == 1
    assert report["codec_heldout_reconstruction"] is None
    assert not report["all_evidence_audited"]
    assert all(row["error"] == "Database unavailable during setup" for row in report["episodes"])
    assert "unavailable" in (tmp_path / "RESULTS.md").read_text()


def test_partial_experiment_cannot_be_declared_complete(tmp_path):
    comparison = failed_experiment(tmp_path)
    comparison["planned_runs"] = 6
    (tmp_path / "comparison.json").write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="all predeclared conditions"):
        reporting.summarize(tmp_path)


def test_success_with_missing_models_is_corrupt_evidence(tmp_path):
    comparison = failed_experiment(tmp_path)
    comparison["episodes"][0].update(success=True, status="succeeded")
    (tmp_path / "comparison.json").write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="Missing frozen codec"):
        reporting.summarize(tmp_path)
