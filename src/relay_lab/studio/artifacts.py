"""Verified, repository-owned CoLA-inspired assets; never arbitrary user weights.

The manifest is an integrity inventory, not a publisher signature. A changed file
invalidates the runtime cache before its bytes can be used again.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import threading
from pathlib import Path

BUNDLE = "examples/cloth-coordination/learned-latent"
FILES = (
    "dataset/metadata.json",
    "dataset/rows.json",
    "dataset/rgb.npz",
    "features/left.json",
    "features/right.json",
    "features/left.npz",
    "features/right.npz",
    "models/embeddings.pt",
    "models/hybrid.pt",
    "models/evaluation.json",
)
LIMITATIONS = [
    "CoLA-inspired implementation, not a reproduction of the published diffusion head.",
    "The trained replacement action head uses local measured state and a supplied phase.",
    "Captured probes reuse frozen VLA features; only adapter inference runs now.",
    "Qwen instructions are delivered separately and are not consumed by the learned head.",
    "Fresh-message diagnostic training does not establish delay robustness or complete folding.",
    "The recorded live controller trial missed 12 of 14 required holds, with no grasp or fold.",
]


class ArtifactRegistry:
    def __init__(self, root):
        self.path = Path(root) / BUNDLE
        self.lock = threading.RLock()
        self._fingerprint = None
        self._metadata = None
        self._policies = {}
        self._sample = None

    def _stamp(self):
        return tuple(
            (name, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
            for name in ("files.json", *FILES)
            for p in (self.path / name,)
        )

    def verify(self):
        with self.lock:
            stamp = self._stamp()
            if stamp == self._fingerprint:
                return copy.deepcopy(self._metadata)
            self._policies.clear()
            self._sample = None
            self._fingerprint = None
            manifest = json.loads((self.path / "files.json").read_text())
            for name in FILES:
                actual = hashlib.sha256((self.path / name).read_bytes()).hexdigest()
                if manifest.get(name) != actual:
                    raise ValueError(f"Bundled artifact integrity mismatch: {name}")
            evaluation = json.loads((self.path / "models/evaluation.json").read_text())
            sources = evaluation["source_models"]
            training = evaluation["training"]
            if training["dataset_rows_sha256"] != manifest["dataset/rows.json"]:
                raise ValueError("Adapter training dataset differs from the bundled observations")
            for arm in ("left", "right"):
                if training["feature_cache_sha256"][arm] != manifest[f"features/{arm}.npz"]:
                    raise ValueError(
                        "Adapter training feature cache differs from the bundled features"
                    )
            for mode in ("embeddings", "hybrid"):
                if evaluation["modes"][mode]["checkpoint_sha256"] != manifest[f"models/{mode}.pt"]:
                    raise ValueError(
                        "Evaluation checkpoint identity differs from the bundled adapter"
                    )
            extractors = {
                arm: {
                    k: source[k]
                    for k in (
                        "family",
                        "model_id",
                        "revision",
                        "width",
                        "feature_space_id",
                        "layer",
                        "pooling",
                    )
                }
                for arm, source in sources.items()
            }
            metadata = {
                "available": importlib.util.find_spec("torch") is not None,
                "verified": True,
                "error": None,
                "method": "CoLA-inspired",
                "bundle": BUNDLE,
                "dimension": 64,
                "receiver_dimension": 32,
                "source_dimensions": {a: s["width"] for a, s in extractors.items()},
                "extractors": extractors,
                "checkpoints": {
                    mode: {
                        "checkpoint_sha256": manifest[f"models/{filename}.pt"],
                        "path": f"{BUNDLE}/models/{filename}.pt",
                        "native_action_heads_used": False,
                        "loaded": False,
                    }
                    for mode, filename in (("latent", "embeddings"), ("hybrid", "hybrid"))
                },
                "training": {
                    k: training[k]
                    for k in (
                        "dataset_rows_sha256",
                        "feature_cache_sha256",
                        "episodes",
                        "seed",
                        "max_epochs",
                        "selection",
                        "objective",
                        "supervision",
                        "delay_training",
                    )
                },
                "limitations": LIMITATIONS,
            }
            if not metadata["available"]:
                metadata["error"] = (
                    "Install the learned extra to load trained adapters (PyTorch missing)"
                )
            if self._stamp() != stamp:
                raise ValueError("Bundled artifacts changed during verification; retry")
            self._metadata, self._fingerprint = metadata, stamp
            return copy.deepcopy(metadata)

    def describe(self):
        try:
            with self.lock:
                metadata = self.verify()
                for mode, policy in self._policies.items():
                    metadata["checkpoints"][mode].update(policy.model_info, loaded=True)
                return metadata
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {
                "available": False,
                "verified": False,
                "method": "CoLA-inspired",
                "bundle": BUNDLE,
                "dimension": 64,
                "receiver_dimension": 32,
                "source_dimensions": {"left": 960, "right": 1024},
                "extractors": {},
                "checkpoints": {},
                "limitations": LIMITATIONS,
                "error": f"{type(error).__name__}: {error}",
            }

    def policy(self, mode):
        from relay_lab.cloth.features import feature_contract
        from relay_lab.cloth.latent_adapter import LearnedLatentPolicy

        if mode not in ("latent", "hybrid"):
            raise ValueError("No trained adapter for this communication mode")
        with self.lock:
            metadata = self.verify()
            if mode not in self._policies:
                filename = "hybrid" if mode == "hybrid" else "embeddings"
                policy = LearnedLatentPolicy(self.path / f"models/{filename}.pt")
                if policy.mode != filename:
                    raise ValueError("Checkpoint mode differs from selected adapter")
                if policy.sha256 != metadata["checkpoints"][mode]["checkpoint_sha256"]:
                    raise ValueError("Checkpoint changed after artifact verification")
                for arm, family in (("left", "smolvla"), ("right", "xvla")):
                    expected = feature_contract(family)["feature_space_id"]
                    if policy.extractor_ids[arm] != expected:
                        raise ValueError(
                            "Checkpoint extractor differs from current feature contract"
                        )
                expected_training = json.loads((self.path / "models/evaluation.json").read_text())[
                    "training"
                ]
                if any(policy.training.get(k) != v for k, v in expected_training.items()):
                    raise ValueError("Checkpoint training provenance differs from evaluation")
                selected = json.loads((self.path / "models/evaluation.json").read_text())["modes"][
                    filename
                ]["selected_epoch"]
                if policy.training["selected_epoch"] != selected:
                    raise ValueError("Checkpoint selection differs from evaluation")
                self._policies[mode] = policy
            return self._policies[mode]

    def samples(self):
        import numpy as np

        from relay_lab.cloth.latent_training import load_pairs

        with self.lock:
            self.verify()
            if self._sample is None:
                dataset, cache = self.path / "dataset", self.path / "features"
                _, sources, _ = load_pairs(dataset, cache)
                rows = json.loads((dataset / "rows.json").read_text())
                vectors = {}
                for arm in ("left", "right"):
                    with np.load(cache / f"{arm}.npz", allow_pickle=False) as data:
                        vectors.update(
                            {
                                int(i): v.copy()
                                for i, v in zip(
                                    data["row_indices"],
                                    data["features"],
                                    strict=True,
                                )
                            }
                        )
                with np.load(dataset / "rgb.npz", allow_pickle=False) as data:
                    rgb = {k: v.copy() for k, v in data.items()}
                self._sample = rows, sources, vectors, rgb
            return self._sample
