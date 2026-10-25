"""Benchmark setup failures stay in the denominator and cannot imply pairing."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "communication_benchmark",
    Path(__file__).parents[1] / "scripts/benchmark_factory_communication.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_constructor_failures_are_recorded_and_remaining_modes_are_attempted(tmp_path, monkeypatch):
    attempts = []

    def unavailable(**kwargs):
        attempts.append((kwargs["seed"], kwargs["communication"]))
        raise FileNotFoundError("Original checkpoint is missing")

    monkeypatch.setattr(_MODULE, "FactoryRun", unavailable)
    output = tmp_path / "experiment"
    monkeypatch.setattr("sys.argv", ["benchmark", "--output", str(output), "--seeds", "101", "102"])
    _MODULE.main()
    assert attempts == [
        (101, "embeddings"),
        (101, "text"),
        (101, "none"),
        (102, "text"),
        (102, "none"),
        (102, "embeddings"),
    ]
    result = json.loads((output / "results.json").read_text())
    assert len(result["runs"]) == 6
    for row in result["runs"]:
        assert row["status"] == "setup_failed" and not row["success"]
        assert row["orders"] == 3 and row["completed"] == row["correct_routes"] == 0
        assert row["actor_sha256"] is None
        assert row["communication"]["model"]["checkpoint_sha256"] is None
        assert row["error"] == "FileNotFoundError: Original checkpoint is missing"
        assert json.loads((Path(row["directory"]) / "setup_failure.json").read_text()) == row
    for mode in _MODULE.MODES:
        assert result["aggregate"][mode]["runs"] == 2
        assert result["aggregate"][mode]["total_routes"] == 6
        assert result["aggregate"][mode]["setup_failures"] == 2
    for pair in result["paired_checks"]:
        assert not pair["identical_actor_weights"]
        assert not pair["identical_codec_weights"]
        assert not pair["identical_initial_physics"]
        assert not pair["identical_private_sensor_inputs"]
        assert not pair["embedding_text_decisions_equal"]


def test_successful_trial_counts_unique_source_payload_and_packet_overhead_separately(
    tmp_path, monkeypatch
):
    class FinishedRun:
        def __init__(self, **kwargs):
            self.state = {"robots": [], "cargo": [], "jobs": {}, "tick": 0}
            self.status = "succeeded"
            self.output = kwargs["output"]
            self.output.mkdir()
            self.routing_exchange = SimpleNamespace(memories={"job-1": {"text": '{"route":"B"}'}})

        def close(self):
            summary = {
                "run_id": "fixture",
                "status": self.status,
                "success": True,
                "jobs": {"job-1": {"routing_correct": True}},
                "completed": 1,
                "tick": 4,
                "metrics": {"wall_seconds": 1.2, "failed_commands": 0},
                "policy": {"checkpoint_sha256": "a" * 64},
                "motor_policy": {"metrics": {}},
                "communication": {
                    "published": 3,
                    "model": {"checkpoint_sha256": "b" * 64},
                    "metrics": {"packet_payload_write_bytes": 1900, "transport_bytes": 2100},
                    "encode_ms": [0.4],
                    "decode_ms": [0.2, 0.1],
                },
            }
            (self.output / "summary.json").write_text(json.dumps(summary))

    monkeypatch.setattr(_MODULE, "FactoryRun", FinishedRun)
    args = SimpleNamespace(orders=1, planner="rules", coordination="memory", max_ticks=5000)
    row = _MODULE.run_trial(args, 101, "text", tmp_path / "run")
    assert row["source_payload_bytes"] == len(b'{"route":"B"}')
    assert row["source_payload_count"] == 1
    result = _MODULE.aggregate([row])["text"]
    assert result["packet_envelope_write_bytes"] == 1900
    assert result["refresh_transport_body_bytes"] == 2100
    assert result["source_payload_bytes"] == 13
    assert (
        result["semantic_decode_parse_ms_total"] == result["semantic_decode_parse_ms_per_attempt"]
    )
    assert (
        "sender neural decoding plus receiver JSON parsing" in result["semantic_decode_parse_scope"]
    )
    legacy = deepcopy(row)
    del legacy["source_payload_bytes"], legacy["source_payload_count"]
    assert _MODULE.aggregate([legacy])["text"]["source_payload_bytes"] is None


def test_real_matched_hashes_and_complete_records_remain_pairable():
    rows = [
        {
            "seed": 101,
            "mode": mode,
            "orders": 1,
            "status": "succeeded",
            "success": True,
            "actor_sha256": "a" * 64,
            "initial_physical_state_sha256": "b" * 64,
            "communication": {
                "model": {"checkpoint_sha256": "c" * 64},
                "published": int(mode != "none"),
                "observation_provenance": [{"job_id": "job-1", "source_frame_sha256": "d" * 64}],
                "decisions": [{"job_id": "job-1", "route": "A"}],
            },
        }
        for mode in _MODULE.MODES
    ]
    pair = _MODULE.paired_checks(rows)[0]
    assert pair["identical_actor_weights"] and pair["identical_codec_weights"]
    assert pair["identical_initial_physics"] and pair["identical_private_sensor_inputs"]
    assert pair["embedding_text_decisions_equal"]
    rows[1]["communication"]["model"]["checkpoint_sha256"] = None
    assert not _MODULE.paired_checks(rows)[0]["identical_codec_weights"]
