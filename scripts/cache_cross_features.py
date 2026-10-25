"""Encode each arm's camera with the *other* VLA family (X-VLA on left, SmolVLA on right).

The packaged caches hold SmolVLA features of the left camera and X-VLA features of
the right camera. This adds the swapped pairing for the same rows so that the same
view is encoded by both model families. Run from the repository root (the pinned
model caches live in .cache/huggingface/hub); writes <output>/<family>_<arm>.npz.

    PYTHONPATH=<worktree>/src .venv/bin/python <worktree>/scripts/cache_cross_features.py \
        --dataset <worktree>/examples/cloth-coordination/learned-latent/dataset --output runs/cross-features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from relay_lab.cloth.features import FrozenVLAFeatureExtractor, feature_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.dataset / "rows.json").read_text())
    images = np.load(args.dataset / "rgb.npz", allow_pickle=False)["rgb"]
    for family, arm in (("xvla", "left"), ("smolvla", "right")):
        path = args.output / f"{family}_{arm}.npz"
        if path.exists():
            print("exists", path)
            continue
        extractor = FrozenVLAFeatureExtractor(family, arm)
        memo, indices, vectors = {}, [], []
        started = time.perf_counter()
        for index, row in enumerate(rows):
            if row["agent"] != arm:
                continue
            rgb = images[row["rgb_index"]]
            key = (hashlib.sha256(rgb.tobytes()).hexdigest(), row["instruction"])
            if key not in memo:
                memo[key] = extractor.extract({"agent": arm, "rgb": rgb}, row["instruction"]).vector
            indices.append(index)
            vectors.append(memo[key])
        np.savez_compressed(
            path,
            row_indices=np.asarray(indices),
            features=np.asarray(vectors, np.float32),
            feature_space_id=feature_contract(family)["feature_space_id"],
            rows_sha256=hashlib.sha256((args.dataset / "rows.json").read_bytes()).hexdigest(),
        )
        print(
            family,
            arm,
            len(indices),
            "rows",
            len(memo),
            "forward passes",
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        del extractor


if __name__ == "__main__":
    main()
