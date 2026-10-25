"""Re-execute recorded actions in MuJoCo; verify physics and optionally render at 25 Hz.

This does not re-run policies, Qwen, or transport. The original controller and
communication evidence remains in the source run, separate from this replay.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from relay_lab.cloth.world import ClothWorld


def replay(source: Path, output: Path, *, video=True):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Replay exactly the bytes whose digests are reported, even if a source
    # path is replaced while the independent world is running.
    inputs = {
        name: (source / name).read_bytes()
        for name in ("summary.json", "final.json", "events.jsonl.gz")
    }
    summary = json.loads(inputs["summary.json"])
    expected = json.loads(inputs["final.json"])
    report = {
        "kind": "independent_mujoco_action_reexecution",
        "source_run_id": summary["run_id"],
        "source_artifact_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in inputs.items()
        },
        "reexecutes_neural_models": False,
        "reexecutes_communication": False,
        "verified": False,
        "steps": 0,
        "maximum_robot_state_error": 0.0,
        "video_fps": 25 if video else None,
    }
    world = writer = None
    try:
        config = summary["config"]
        world = ClothWorld(
            seed=config["seed"],
            disturbances=config["scenario"] != "clear",
            fixture=config["fixture"],
        )
        if video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(output / "overview.mp4"), fps=25, codec="libx264", quality=8
            )
        with gzip.GzipFile(fileobj=io.BytesIO(inputs["events.jsonl.gz"]), mode="rb") as stream:
            for line in stream:
                record = json.loads(line)
                if world.tick != record["decision_tick"]:
                    raise ValueError("Recorded actions must cover every consecutive physics tick")
                for event in record["events"]:
                    if event["event"] == "vacuum_fixture_released":
                        if event["tick"] != world.tick or not world.fixture_active:
                            raise ValueError("Fixture release does not match pre-step state")
                        world.release_fixture()
                snapshot = world.step(record["actions"])
                for agent in ("left", "right"):
                    actual, saved = snapshot["robots"][agent], record["robots"][agent]
                    if actual["holding"] != saved["holding"]:
                        raise ValueError(f"Grasp identity differs at tick {world.tick}")
                    for key in ("ee", "joint_positions"):
                        error = float(np.max(np.abs(np.array(actual[key]) - saved[key])))
                        report["maximum_robot_state_error"] = max(
                            report["maximum_robot_state_error"], error
                        )
                        if error > 1e-9:
                            raise ValueError(f"Physics differs at tick {world.tick}: {agent} {key}")
                report["steps"] += 1
                if writer:
                    writer.append_data(world.render(width=1280, height=800))
        final = world.snapshot()
        # A failed trajectory can replay exactly. Expose measured geometry so
        # callers cannot confuse reproducibility with successful folding.
        report["final_metrics"] = final["metrics"]
        report["final_vertices_bitwise_equal"] = bool(
            np.array_equal(np.array(final["vertices"]), np.array(expected["vertices"]))
        )
        report["same_final_metrics"] = final["metrics"] == expected["metrics"]
        report["same_fixture_state"] = final["fixture_active"] == expected["fixture_active"]
        report["same_tick_count"] = world.tick == expected["tick"] == summary["ticks"]
        report["verified"] = all(
            report[k]
            for k in (
                "final_vertices_bitwise_equal", "same_final_metrics",
                "same_fixture_state", "same_tick_count",
            )
        )
        if not report["verified"]:
            raise ValueError("Final recorded and re-executed physics differ")
    except Exception as exc:
        report["verified"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for name, resource in (("video", writer), ("physics", world)):
            if resource:
                try:
                    resource.close()
                except Exception as exc:  # noqa: BLE001 - preserve evidence on cleanup failure
                    report["verified"] = False
                    report.setdefault("cleanup_errors", []).append(f"{name}: {exc}")
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if report.get("cleanup_errors"):
        raise RuntimeError("Action re-execution cleanup failed; inspect verification.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    print(json.dumps(replay(args.source, args.output, video=not args.no_video), indent=2))
