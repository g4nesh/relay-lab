"""Cache real frozen features; fit and evaluate small learned peer adapters.

The cache is observation-only. Labels are used exclusively by the training or
evaluation stages. Entire episodes, including counterfactual variants, stay in
one split. The held-out test set is evaluated only after dev checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .communication import AGENTS, FEATURES, canonical, observation_features
from .latent_adapter import FORMAT, MODES, WIDTHS, DualLatentAdapter, bounded_label, local_input


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hashes():
    directory = Path(__file__).parent
    return {
        name: sha(directory / name)
        for name in (
            "features.py",
            "pretrained.py",
            "latent_adapter.py",
            "latent_dataset.py",
            "latent_training.py",
            "latent_channel.py",
            "latent_experiment.py",
        )
    }


def cache_features(dataset, output, *, family):
    from .features import FrozenVLAFeatureExtractor

    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arm = {"smolvla": "left", "xvla": "right"}[family]
    path = output / f"{arm}.npz"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Feature cache already exists; never replace training provenance")
    rows = json.loads((dataset / "rows.json").read_text())
    images = np.load(dataset / "rgb.npz", allow_pickle=False)["rgb"]
    extractor = FrozenVLAFeatureExtractor(family, arm)
    implementation = implementation_hashes()
    indices, vectors, audit, memo = [], [], [], {}
    started = time.perf_counter()
    for index, row in enumerate(rows):
        if row["agent"] != arm:
            continue
        rgb = images[row["rgb_index"]]
        key = (hashlib.sha256(rgb.tobytes()).hexdigest(), row["instruction"])
        if key not in memo:
            result = extractor.extract({"agent": arm, "rgb": rgb}, row["instruction"])
            memo[key] = result
            print(canonical({"family": family, "extractions": len(memo), "row": index}), flush=True)
        result = memo[key]
        indices.append(index)
        vectors.append(result.vector)
        audit.append({"row_index": index, "row_id": row["id"], **result.metadata})
    np.savez_compressed(path, row_indices=np.asarray(indices), features=np.asarray(vectors))
    write_json(
        path.with_suffix(".json"),
        {
            "kind": "actual-frozen-vla-feature-cache",
            "implementation_sha256": implementation,
            "model": extractor.model_info,
            "rows_sha256": sha(dataset / "rows.json"),
            "rgb_sha256": sha(dataset / "rgb.npz"),
            "cache_sha256": sha(path),
            "rows": len(indices),
            "unique_inferences": len(memo),
            "elapsed_seconds": time.perf_counter() - started,
            "audit": audit,
            "deduplication": "Exact RGB bytes and instruction within the same pinned extractor only.",
        },
    )
    del extractor
    gc.collect()


def load_pairs(dataset, cache):
    from .features import feature_contract
    from .latent_dataset import array_hash, canonical_hash, validate_episodes

    dataset, cache = Path(dataset), Path(cache)
    rows = json.loads((dataset / "rows.json").read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Require nonempty aligned observation rows")
    with np.load(dataset / "rgb.npz", allow_pickle=False) as source:
        images = source["rgb"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Invalid original RGB source")
    groups, episodes, feature_rows, sources = {}, {}, {}, {}
    identifiers, episode_seeds, seed_episodes, snapshots = set(), {}, {}, {}
    for row in rows:
        identifier, arm, seed = row["id"], row["agent"], row["seed"]
        if identifier in identifiers or arm not in AGENTS:
            raise ValueError("Duplicate observation id or unknown arm")
        identifiers.add(identifier)
        split, episode = row["split"], row["episode_id"]
        if episode in episodes and episodes[episode] != split:
            raise ValueError("An episode leaked across training/evaluation splits")
        if seed in seed_episodes and seed_episodes[seed] != episode:
            raise ValueError("One episode seed was renamed across observation groups")
        if episode in episode_seeds and episode_seeds[episode] != seed:
            raise ValueError("One episode id contains multiple seeds")
        episodes[episode], episode_seeds[episode], seed_episodes[seed] = split, seed, episode
        signature = (episode, split, seed, row["phase"], row["physical_state_sha256"])
        if row["snapshot_id"] in snapshots and snapshots[row["snapshot_id"]] != signature:
            raise ValueError("Snapshot counterfactuals do not share one physical state and split")
        snapshots[row["snapshot_id"]] = signature
        index = row["rgb_index"]
        if type(index) is not int or not 0 <= index < len(images):
            raise ValueError("Invalid RGB-to-observation alignment")
        if (
            row["rgb_sha256"] != array_hash(images[index])
            or row["observation_sha256"] != canonical_hash(row["observation"])
            or row["observation"].get("agent") != arm
        ):
            raise ValueError("Original local observation or RGB evidence changed")
    validate_episodes(tuple((split, episode_seeds[episode]) for episode, split in episodes.items()))
    rows_sha, rgb_sha = sha(dataset / "rows.json"), sha(dataset / "rgb.npz")
    for arm in AGENTS:
        path = cache / f"{arm}.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["rows_sha256"] != rows_sha or meta["cache_sha256"] != sha(path):
            raise ValueError("Feature cache or aligned row manifest changed")
        if meta["rgb_sha256"] != rgb_sha:
            raise ValueError("Feature cache RGB source changed")
        contract = feature_contract("smolvla" if arm == "left" else "xvla")
        if (
            meta.get("kind") != "actual-frozen-vla-feature-cache"
            or any(meta["model"].get(k) != v for k, v in contract.items())
            or meta["model"].get("agent") != arm
            or meta["model"].get("backbone_frozen") is not True
            or meta["model"].get("native_action_head_executed") is not False
            or meta["model"].get("observation_numeric_state_used") is not False
        ):
            raise ValueError("Frozen feature cache has incompatible extractor provenance")
        with np.load(path, allow_pickle=False) as stored:
            indices, features = stored["row_indices"], stored["features"]
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("Feature row indices must be a one-dimensional integer array")
        if features.shape != (len(indices), WIDTHS[arm]) or features.dtype != np.float32:
            raise ValueError("Incorrect native feature cache width")
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite cached backbone features")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("Duplicate feature row")
        audit = meta.get("audit")
        if (
            not isinstance(audit, list)
            or len(audit) != len(indices)
            or meta["rows"] != len(indices)
        ):
            raise ValueError("Missing aligned feature extraction audit")
        for i, vector, evidence in zip(indices, features, audit, strict=True):
            i = int(i)
            if not 0 <= i < len(rows) or rows[i]["agent"] != arm or i in feature_rows:
                raise ValueError("Feature-to-observation alignment mismatch")
            rgb = images[rows[i]["rgb_index"]]
            if (
                type(evidence.get("row_index")) is not int
                or evidence["row_index"] != i
                or evidence.get("row_id") != rows[i]["id"]
                or any(evidence.get(k) != v for k, v in contract.items())
                or evidence.get("agent") != arm
                or evidence.get("vector_sha256")
                != hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest()
                or evidence.get("rgb_sha256") != hashlib.sha256(rgb.tobytes()).hexdigest()
                or evidence.get("instruction_sha256")
                != hashlib.sha256(rows[i]["instruction"].encode()).hexdigest()
                or evidence.get("tokenization", {}).get("full_prompt_roundtrip_verified")
                is not True
            ):
                raise ValueError("Feature extraction audit does not match the original observation")
            feature_rows[i] = vector
        sources[arm] = meta
    if set(feature_rows) != set(range(len(rows))):
        raise ValueError("Missing frozen feature rows")
    for i, row in enumerate(rows):
        split, episode = row["split"], row["episode_id"]
        pair = groups.setdefault(row["pair_id"], {})
        if row["agent"] in pair:
            raise ValueError("Duplicate arm in aligned observation pair")
        pair[row["agent"]] = (i, row)
    pairs = []
    for pair in groups.values():
        if set(pair) != set(AGENTS):
            raise ValueError("Incomplete robot pair")
        a, b = (pair[k][1] for k in AGENTS)
        if any(
            a[k] != b[k]
            for k in (
                "split",
                "snapshot_id",
                "episode_id",
                "phase",
                "seed",
                "counterfactual",
                "physical_state_sha256",
            )
        ):
            raise ValueError("Unaligned private robot observations")
        pairs.append(pair)
    data = {}
    for split in ("train", "dev", "test"):
        selected = [p for p in pairs if p["left"][1]["split"] == split]
        if not selected:
            raise ValueError("Require separate nonempty train/dev/test episodes")
        batch = {
            k: {} for k in ("features", "states", "facts", "nominal", "hold", "actual", "grasp")
        }
        for arm in AGENTS:
            for key, values in (
                ("features", [feature_rows[p[arm][0]] for p in selected]),
                (
                    "states",
                    [local_input(p[arm][1]["observation"], p[arm][1]["phase"]) for p in selected],
                ),
                (
                    "facts",
                    [
                        [observation_features(p[arm][1]["observation"])[k] for k in FEATURES]
                        for p in selected
                    ],
                ),
                (
                    "nominal",
                    [
                        bounded_label(
                            p[arm][1]["observation"], p[arm][1]["labels"]["nominal_action"]
                        )
                        for p in selected
                    ],
                ),
                (
                    "actual",
                    [
                        bounded_label(p[arm][1]["observation"], p[arm][1]["action"])
                        for p in selected
                    ],
                ),
                ("hold", [float(p[arm][1]["labels"]["coordinated_hold"]) for p in selected]),
                ("grasp", [float(p[arm][1]["observation"]["grasped"]) for p in selected]),
            ):
                batch[key][arm] = torch.tensor(np.asarray(values), dtype=torch.float32)
        batch["size"], batch["pair_ids"] = (
            len(selected),
            [p["left"][1]["pair_id"] for p in selected],
        )
        data[split] = batch
    return data, sources, episodes


def predictions(model, batch, intervention=None):
    return model(
        batch["features"],
        batch["states"],
        batch["facts"],
        torch.ones(batch["size"]),
        torch.zeros(batch["size"]),
        intervention=intervention,
    )


def loss_for(output, batch):
    losses = []
    for arm in AGENTS:
        raw, nominal = output[arm], batch["nominal"][arm]
        losses.append(
            F.mse_loss(raw[:, :3].tanh(), nominal[:, :3])
            + 0.3 * F.binary_cross_entropy_with_logits(raw[:, 3], nominal[:, 3])
            + F.binary_cross_entropy_with_logits(raw[:, 4], batch["hold"][arm])
        )
    return sum(losses) / len(losses)


def evaluate(model, batch, intervention=None):
    with torch.inference_mode():
        output = predictions(model, batch, intervention)
        report = {"pairs": batch["size"], "loss": float(loss_for(output, batch)), "arms": {}}
        for arm in AGENTS:
            raw = output[arm]
            hold, expected = raw[:, 4] >= 0, batch["hold"][arm].bool()
            delta = raw[:, :3].tanh()
            delta = delta / delta.norm(dim=1, keepdim=True).clamp(min=1)
            delta[hold] = 0
            grip = torch.where(hold, batch["grasp"][arm].bool(), raw[:, 3] >= 0)
            actual = batch["actual"][arm]
            private = batch["facts"][arm][:, FEATURES.index("obstacle_active")] < 0.5
            motion = batch["nominal"][arm][:, :3].norm(dim=1) > 0.1
            errors = (delta - actual[:, :3]).norm(dim=1) * 18
            report["arms"][arm] = {
                "hold_accuracy": float((hold == expected).float().mean()),
                "hidden_peer_accuracy": float((hold[private] == expected[private]).float().mean())
                if bool(private.any())
                else None,
                "hidden_peer_count": int(private.sum()),
                "clear_count": int((~expected).sum()),
                "hazard_count": int(expected.sum()),
                "clear_go_rate": float((~hold[~expected]).float().mean())
                if bool((~expected).any())
                else None,
                "hazard_hold_rate": float(hold[expected].float().mean())
                if bool(expected.any())
                else None,
                "unsafe_go_count": int((~hold & expected).sum()),
                "false_hold_count": int((hold & ~expected).sum()),
                "mean_action_error_mm": float(errors.mean()),
                "nontrivial_motion_count": int(motion.sum()),
                "nontrivial_motion_error_mm": float(errors[motion].mean())
                if bool(motion.any())
                else None,
                "gripper_accuracy": float((grip == actual[:, 3].bool()).float().mean()),
                "hold_probabilities": raw[:, 4].sigmoid().tolist(),
            }
    return report


def train(dataset, cache, output, *, epochs=400, seed=20261024):
    if type(epochs) is not int or not 1 <= epochs <= 2000:
        raise ValueError("Require 1–2000 training epochs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    data, sources, episodes = load_pairs(dataset, cache)
    training = {
        "implementation_sha256": implementation_hashes(),
        "dataset_rows_sha256": sha(Path(dataset) / "rows.json"),
        "episodes": episodes,
        "feature_cache_sha256": {a: s["cache_sha256"] for a, s in sources.items()},
        "seed": seed,
        "max_epochs": epochs,
        "selection": "minimum whole-development-episode loss",
        "objective": "nominal Cartesian delta MSE + 0.3 gripper BCE + coordinated-hold BCE",
        "supervision": "original geometric teacher; nominal motion and explicit centralized hazard-hold labels",
        "native_action_heads_used": False,
        "full_task_success_evaluation": False,
        "delay_training": False,
        "age_policy": "expiration/invalid-message conservative hold in runtime; no learned delay robustness claim",
    }
    report = {
        "kind": "mixed-vla-frozen-feature-one-step-diagnostic",
        "training": training,
        "source_models": {a: s["model"] for a, s in sources.items()},
        "modes": {},
    }
    for mode in MODES:
        started = time.perf_counter()
        torch.manual_seed(seed)
        model = DualLatentAdapter(mode)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        best, best_epoch, best_state = float("inf"), 0, None
        curve = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(predictions(model, data["train"]), data["train"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Adapter training became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                dev = float(loss_for(predictions(model, data["dev"]), data["dev"]))
            if dev < best:
                best, best_epoch = dev, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch % 25 == 0 or epoch == epochs - 1:
                curve.append({"epoch": epoch, "train_loss": float(loss.detach()), "dev_loss": dev})
        model.load_state_dict(best_state)
        path = output / f"{mode}.pt"
        torch.save(
            {
                "format": FORMAT,
                "mode": mode,
                "widths": WIDTHS,
                "state_dict": best_state,
                "training": {**training, "selected_epoch": best_epoch},
                "extractor_ids": {a: s["model"]["feature_space_id"] for a, s in sources.items()},
            },
            path,
        )
        result = {
            "checkpoint_sha256": sha(path),
            "selected_epoch": best_epoch,
            "train": evaluate(model, data["train"]),
            "dev": evaluate(model, data["dev"]),
            "test": evaluate(model, data["test"]),
            "test_zero_peer": evaluate(model, data["test"], "zero"),
            "test_shuffled_peer": evaluate(model, data["test"], "shuffle"),
            "elapsed_seconds": time.perf_counter() - started,
            "curve": curve,
        }
        report["modes"][mode] = result
        write_json(output / "evaluation.json", report)
        print(canonical({"mode": mode, "test": result["test"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--dataset", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--family", choices=("smolvla", "xvla"), required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--cache", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    if args.command == "cache":
        cache_features(args.dataset, args.output, family=args.family)
    else:
        train(args.dataset, args.cache, args.output, epochs=args.epochs)


if __name__ == "__main__":
    main()
