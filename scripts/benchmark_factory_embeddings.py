#!/usr/bin/env python3
"""Synthetic transport benchmark, not a learned-policy quality measurement.

Populates the bounded 5×16 mailbox then measures 100 forced refreshes separately
from 500 latest-peer local cosine retrievals. No simulated production is claimed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from uuid import uuid4

from relay_lab.factory.coordination import MemoryCoordinator
from relay_lab.factory.embeddings import SCHEMA
from relay_lab.factory.spacetime import SpacetimeCoordinator


def benchmark(kind, iterations):
    backend = SpacetimeCoordinator() if kind == "spacetime" else MemoryCoordinator()
    ids = ["humanoid-01", "amr-01", "amr-02", "arm-01", "gantry-01"]
    run_id = "embedding-bench-" + uuid4().hex[:16]
    state = {
        "tick": 0,
        "success": False,
        "jobs": {"job-1": {"job_id": "job-1", "stage": "queued", "custodian": None}},
        "robots": [{"id": robot, "active_command": None} for robot in ids],
    }
    backend.create(
        run_id,
        "benchmark-worker",
        [{"robot_id": robot, "capabilities": ["wait"]} for robot in ids],
        list(state["jobs"].values()),
        [],
        100,
        state,
    )
    backend.configure_embedding_space("synthetic-benchmark-v1", "a" * 64)
    randomizer = random.Random(137)
    vectors = {}
    for tick in range(1, 21):
        records = []
        for robot in ids:
            raw = [randomizer.uniform(-1, 1) for _ in range(32)]
            norm = math.sqrt(sum(x * x for x in raw))
            vectors[robot] = [x / norm for x in raw]
            records.append(
                {
                    "schema": SCHEMA,
                    "embedding_id": f"{robot}-{tick}",
                    "run_id": run_id,
                    "robot_id": robot,
                    "tick": tick,
                    "command_id": None,
                    "job_id": None,
                    "modality": "vla-hidden",
                    "space_id": "synthetic-benchmark-v1",
                    "checkpoint_sha256": "a" * 64,
                    "dimension": 32,
                    "vector": vectors[robot],
                    "expires_tick": tick + 64,
                    "decoder": {
                        "text": "Synthetic test vector; no model prediction",
                        "confidence": 0.0,
                        "version": "fixture-v1",
                    },
                }
            )
        backend.prepare(tick - 1, [])
        state["tick"] = tick
        backend.commit(tick - 1, state, [], embeddings=records)
    start = time.perf_counter()
    for _ in range(iterations):
        backend.refresh_embeddings(20, force=True)
        for robot in ids:
            matches = backend.retrieve_embeddings(robot, vectors[robot], 20)
            assert len(matches) == 4 and all(row["tick"] == 20 for row in matches)
    elapsed = time.perf_counter() - start
    result = {
        "backend": kind,
        "run_id": run_id,
        "synthetic_vectors": True,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "metrics": backend.embedding_metrics,
    }
    backend.abort("Synthetic embedding benchmark finished; no production task was executed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/factory-embedding-benchmark.json")
    )
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000:
        parser.error("iterations must be between 1 and 1000")
    results = [benchmark(kind, args.iterations) for kind in ("memory", "spacetime")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "purpose": "Synthetic bounded embedding transport and retrieval measurement",
                "results": results,
            },
            indent=2,
        )
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
