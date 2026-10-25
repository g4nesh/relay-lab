"""Post-hoc measurements only: never refit policies or the communication codec."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from relay_lab.cloth.communication import FEATURES, LearnedFeatureCodec, observation_features


def summarize(directory):
    directory = Path(directory)
    comparison = json.loads((directory / "comparison.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if comparison["recorded_runs"] != comparison["planned_runs"]:
        raise ValueError("Wait for all predeclared conditions, including failed attempts")
    rows, text_observations, codec = [], [], None
    for index, item in enumerate(manifest["runs"]):
        root = directory / item["id"]
        summary_path = root / "summary.json"
        summary = (
            json.loads(summary_path.read_text())
            if summary_path.exists()
            else comparison["episodes"][index]
        )
        codec_path = root / "models/communication.json"
        if codec_path.exists():
            local_codec = LearnedFeatureCodec.load(codec_path)
            if codec is not None and codec.sha256 != local_codec.sha256:
                raise ValueError("Cannot pool different frozen codecs")
            codec = local_codec
        elif summary["status"] != "failed":
            raise ValueError(f"Missing frozen codec for recorded episode {item['id']}")
        holds = Counter()
        event_path = root / "events.jsonl.gz"
        if event_path.exists():
            with gzip.open(event_path, "rt") as stream:
                for line in stream:
                    record = json.loads(line)
                    holds.update(d["reason"] for d in record["decisions"].values() if d["hold"])
                    if summary["mode"] == "structured_text" and record["decision_tick"] % 4 == 0:
                        text_observations.extend(
                            observation_features(o) for o in record["observations"].values()
                        )
        elif summary["status"] != "failed":
            raise ValueError(f"Missing action evidence for recorded episode {item['id']}")
        communication = summary.get("communication", {})
        packet_bytes, published = communication.get("packet_bytes"), communication.get("published")
        errors = summary.get("metrics", {}).get("hem_alignment_m", {})
        audit_path = root / "audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        rows.append(
            {
                "id": item["id"],
                "mode": summary["mode"],
                "scenario": summary["scenario"],
                "seed": summary["seed"],
                "success": summary["success"],
                "status": summary["status"],
                "error": summary.get("error"),
                "ticks": summary["ticks"],
                "sim_seconds": summary["ticks"] * 0.04,
                "worst_hem_error_m": max(errors.values()) if errors else None,
                "packet_bytes": packet_bytes,
                "packets": published,
                "mean_packet_bytes": packet_bytes / published if published else None,
                "encode_ms": communication.get("encode_ms", {}).get("mean"),
                "decode_ms": communication.get("decode_ms", {}).get("mean"),
                "database_commit_readback_ms": communication.get("durable_commit_ms", {}).get("mean"),
                "qwen_calls": summary.get("planner", {}).get("calls", 0),
                "hold_reasons": dict(holds),
                "evidence_audited": audit.get("evidence_valid") is True,
                "spacetime_audited": audit.get("spacetime_checked") is True,
            }
        )
    modes = []
    for mode in ("embeddings", "structured_text", "hybrid"):
        selected = [r for r in rows if r["mode"] == mode]
        modes.append(
            {
                "mode": mode,
                "successes": sum(r["success"] for r in selected),
                "runs": len(selected),
                **{
                    f"mean_{key}": (
                        mean(r[key] for r in selected if r[key] is not None)
                        if any(r[key] is not None for r in selected)
                        else None
                    )
                    for key in (
                        "sim_seconds",
                        "worst_hem_error_m",
                        "packet_bytes",
                        "mean_packet_bytes",
                        "encode_ms",
                        "decode_ms",
                        "database_commit_readback_ms",
                    )
                },
                "missing_geometry_runs": sum(r["worst_hem_error_m"] is None for r in selected),
                "missing_traffic_runs": sum(r["packet_bytes"] is None for r in selected),
            }
        )
    reconstruction = codec.evaluate(text_observations) if codec and text_observations else None
    if reconstruction:
        reconstruction["sampling"] = (
            "Both local observations every four ticks in held-out structured-text episodes; one shared evaluation distribution, no refitting."
        )
        truth, reconstructed = [], []
        for features in text_observations:
            truth.append(features)
            reconstructed.append(codec.decode(codec.encode(features)))
        reconstruction["binary_accuracy"] = {
            key: mean((p[key] >= 0.5) == (q[key] >= 0.5) for p, q in zip(truth, reconstructed))
            for key in FEATURES
            if key in ("grasped", "grasp_sleeve", "grasp_hem", "obstacle_active")
        }
    result = {
        "protocol_sha256": comparison["protocol_sha256"],
        "conclusion": comparison["conclusion"],
        "mode_summary": modes,
        "episodes": rows,
        "codec_heldout_reconstruction": reconstruction,
        "all_evidence_audited": all(r["evidence_audited"] for r in rows),
        "all_spacetime_audited": all(r["spacetime_audited"] for r in rows),
        "interpretation": [
            "Descriptive paired pilot; few seeds, one garment/fixture and one fixed intrusion schedule.",
            "Simulated duration reflects controller/communication behavior, not real-time inference speed.",
            "Payload bytes are serialized JSON including envelope; exclude HTTP and database readback traffic.",
            "Local wall-clock timings include shared-machine contention; not an isolated latency contest.",
            "Setup/runtime failures stay in success denominators; unavailable measurements are null, not zero, and missing counts are reported.",
            "State-assisted task policies and numeric PCA; no foundation-VLA or real-textile transfer conclusion.",
        ],
    }
    (directory / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Recorded cloth communication comparison",
        "",
        comparison["conclusion"],
        "",
        (
            f"Planner: **{comparison['planner']}**. Backend: **{comparison['coordination']}**. "
            f"Seeds: {comparison['seeds']}. Conditions: {comparison['scenarios']}."
        ),
        "",
        "| Mode | Completed | Mean simulated time | Mean worst-arm hem error | Mean packet size |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in modes:
        error = row["mean_worst_hem_error_m"]
        size = row["mean_mean_packet_bytes"]
        error_label = f"{error * 100:.2f} cm" if error is not None else "unavailable"
        size_label = f"{size:.0f} bytes" if size is not None else "unavailable"
        lines.append(
            f"| {row['mode']} | {row['successes']}/{row['runs']} | "
            f"{row['mean_sim_seconds']:.2f} s | {error_label} | {size_label} |"
        )
    lines += [
        "",
        (
            "All attempted episodes remain in the denominators. Check `comparison.json` "
            "for pairing hashes and `analysis.json` for per-episode holds, codec errors and timings."
        ),
        "",
        (
            "Each episode contains an audit of causal message delivery, model identity, raw actions, "
            "recorded dynamics, durable Spacetime history and final geometry. An audit is consistency "
            "checking, not independent neural/physics execution attestation."
        ),
        "",
        *[f"- {note}" for note in result["interpretation"]],
        "",
    ]
    (directory / "RESULTS.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = summarize(args.directory)
    print(json.dumps(report["mode_summary"], indent=2))
