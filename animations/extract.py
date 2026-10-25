"""Export real tensors from the packaged learned-latent experiment for the animations.

Every number the scenes draw comes from this file's output. It loads the packaged
dataset and frozen feature caches through ``latent_training.load_pairs`` (which
re-verifies every hash and alignment), runs the trained ``embeddings.pt``
checkpoint, and records intermediate activations. Nothing is retrained.

Run from the repository root with the project environment:

    PYTHONPATH=src .venv/bin/python animations/extract.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from relay_lab.cloth.communication import AGENTS, FEATURES, SAFETY_FIELDS
from relay_lab.cloth.latent_adapter import DIMENSION, LOCAL_DIM, MODES, WIDTHS, DualLatentAdapter
from relay_lab.cloth.latent_training import load_pairs, predictions
from relay_lab.cloth.policies import PHASES

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "examples/cloth-coordination/learned-latent"
OUT = Path(__file__).resolve().parent / "data"
FAMILY = {"left": "SmolVLA", "right": "X-VLA"}


def pca2(x, reference=None):
    """Project rows onto the top two principal axes of ``reference`` (default ``x``)."""
    reference = x if reference is None else reference
    mean = reference.mean(0)
    _, s, vt = np.linalg.svd(reference - mean, full_matrices=False)
    explained = (s**2 / (s**2).sum())[:2]
    return (x - mean) @ vt[:2].T, explained


def load_model(mode):
    payload = torch.load(EVIDENCE / f"models/{mode}.pt", map_location="cpu", weights_only=True)
    model = DualLatentAdapter(payload["mode"]).eval().requires_grad_(False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def main():
    OUT.mkdir(exist_ok=True)
    data, sources, _ = load_pairs(EVIDENCE / "dataset", EVIDENCE / "features")
    rows = json.loads((EVIDENCE / "dataset/rows.json").read_text())
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["agent"]] = row
    evaluation = json.loads((EVIDENCE / "models/evaluation.json").read_text())
    model, payload = load_model("embeddings")

    export = {
        "source": {
            "checkpoint": "examples/cloth-coordination/learned-latent/models/embeddings.pt",
            "selected_epoch": payload["training"]["selected_epoch"],
            "objective": payload["training"]["objective"],
            "extractors": {
                arm: {
                    k: sources[arm]["model"][k]
                    for k in ("family", "model_id", "revision", "width", "layer", "pooling")
                }
                for arm in AGENTS
            },
        },
        "dims": {
            "widths": WIDTHS,
            "message": DIMENSION,
            "local_projection": 32,
            "receiver_context": 32,
            "local_state": LOCAL_DIM,
            "features": len(FEATURES),
            "phases": len(PHASES),
            "safety_fields": len(SAFETY_FIELDS),
            "head_input": 32 + LOCAL_DIM + 32 + len(FEATURES) + 2,
            "head_output": 5,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameters_by_module": {
                name: sum(p.numel() for p in getattr(model, name).parameters())
                for name in ("senders", "local_encoders", "receivers", "heads")
            },
        },
        "layers": {
            "sender": [str(m) for m in model.senders["left"]],
            "receiver": [str(m) for m in model.receivers["left"]],
            "local_encoder": [str(m) for m in model.local_encoders["left"]],
            "head": [str(m) for m in model.heads["left"]],
        },
        "splits": {},
        "results": {},
        "curves": {m: evaluation["modes"][m]["curve"] for m in MODES},
    }

    for split in ("train", "dev", "test"):
        batch = data[split]
        meta = [by_pair[p] for p in batch["pair_ids"]]
        with torch.inference_mode():
            messages = {a: model.encode(a, batch["features"][a]) for a in AGENTS}
            normed = {a: model.senders[a][0](batch["features"][a]) for a in AGENTS}
            contexts = {}
            for arm, other in (("left", "right"), ("right", "left")):
                contexts[arm] = model.receivers[arm](messages[other])
            outputs = {
                key: predictions(model, batch, None if key == "delivered" else key)
                for key in ("delivered", "zero", "shuffle")
            }
        entry = {
            "pair_ids": batch["pair_ids"],
            "counterfactual": [m["left"]["counterfactual"] for m in meta],
            "phase": [m["left"]["phase"] for m in meta],
            "snapshot": [m["left"]["snapshot_id"] for m in meta],
            "arms": {},
        }
        for arm in AGENTS:
            entry["arms"][arm] = {
                "rgb_index": [m[arm]["rgb_index"] for m in meta],
                "local_obstacle": [bool(m[arm]["labels"]["local_obstacle"]) for m in meta],
                "peer_obstacle": [bool(m[arm]["labels"]["peer_obstacle"]) for m in meta],
                "hold_label": [bool(m[arm]["labels"]["coordinated_hold"]) for m in meta],
                "hold_prob": {
                    k: torch.sigmoid(v[arm][:, 4]).tolist() for k, v in outputs.items()
                },
            }
        if split == "test":
            for arm in AGENTS:
                raw = batch["features"][arm].numpy()
                msg = messages[arm].numpy()
                ctx = contexts[arm].numpy()
                raw2, raw_var = pca2(raw)
                msg2, msg_var = pca2(msg)
                ctx2, ctx_var = pca2(ctx)
                entry["arms"][arm].update(
                    {
                        "raw_pca": raw2.tolist(),
                        "raw_pca_explained": raw_var.tolist(),
                        "message_pca": msg2.tolist(),
                        "message_pca_explained": msg_var.tolist(),
                        "context_pca": ctx2.tolist(),
                        "context_pca_explained": ctx_var.tolist(),
                        "raw_stats": {
                            "mean": float(raw.mean()),
                            "std": float(raw.std()),
                            "absmax": float(np.abs(raw).max()),
                        },
                        "layernorm_stats": {
                            "mean": float(normed[arm].numpy().mean()),
                            "std": float(normed[arm].numpy().std()),
                        },
                        "message_absmax": float(np.abs(msg).max()),
                    }
                )
            # Exact numeric sample for the "one real message" shot.
            index = next(
                i
                for i, (c, p) in enumerate(zip(entry["counterfactual"], entry["phase"]))
                if c == "left_obstacle" and p == "carry_sleeve"
            )
            clear = next(
                i
                for i, s in enumerate(entry["snapshot"])
                if s == entry["snapshot"][index] and entry["counterfactual"][i] == "clear"
            )
            images = np.load(EVIDENCE / "dataset/rgb.npz", allow_pickle=False)["rgb"]
            sample = {"index": index, "clear_index": clear, "snapshot": entry["snapshot"][index]}
            for arm in AGENTS:
                for tag, i in (("obstacle", index), ("clear", clear)):
                    rgb = images[entry["arms"][arm]["rgb_index"][i]]
                    name = f"rgb_{arm}_{tag}.png"
                    Image.fromarray(rgb).resize((512, 512), Image.NEAREST).save(OUT / name)
                    sample[f"{arm}_{tag}_png"] = name
                sample[f"{arm}_raw_head"] = batch["features"][arm][index, :24].tolist()
                sample[f"{arm}_message"] = messages[arm][index].tolist()
                sample[f"{arm}_message_clear"] = messages[arm][clear].tolist()
                sample[f"{arm}_context"] = contexts[arm][index].tolist()
            with torch.inference_mode():
                for arm in AGENTS:
                    x = batch["features"][arm][index : index + 1]
                    stages = {"raw": x}
                    for name, layer in zip(
                        ("layernorm", "linear1", "gelu", "linear2", "tanh"), model.senders[arm]
                    ):
                        x = layer(x)
                        stages[name] = x
                    sample[f"{arm}_stages"] = {k: v[0].tolist() for k, v in stages.items()}
                # "Does normalizing align the spaces?" Per-vector standardization
                # (LayerNorm without its learned affine) of both test feature sets.
                norm = {
                    a: torch.nn.functional.layer_norm(
                        batch["features"][a], (WIDTHS[a],)
                    ).numpy()
                    for a in AGENTS
                }
                bins = np.linspace(-4, 4, 41)
                export["normalization"] = {
                    "bins": bins.tolist(),
                    "hist": {
                        a: np.histogram(norm[a].ravel(), bins=bins, density=True)[0].tolist()
                        for a in AGENTS
                    },
                    "stats": {
                        a: {"mean": float(norm[a].mean()), "std": float(norm[a].std())}
                        for a in AGENTS
                    },
                    "samples": int(batch["size"]),
                }
            left_px = images[entry["arms"]["right"]["rgb_index"][index]]
            right_px = images[entry["arms"]["right"]["rgb_index"][clear]]
            sample["receiver_pixels_identical"] = bool(np.array_equal(left_px, right_px))
            export["sample"] = sample
        export["splits"][split] = entry

    for mode in MODES:
        result = evaluation["modes"][mode]
        export["results"][mode] = {
            key: {
                "hidden_peer_accuracy": float(
                    np.mean([result[key]["arms"][a]["hidden_peer_accuracy"] for a in AGENTS])
                ),
                "hold_accuracy": float(
                    np.mean([result[key]["arms"][a]["hold_accuracy"] for a in AGENTS])
                ),
                "action_error_mm": float(
                    np.mean([result[key]["arms"][a]["mean_action_error_mm"] for a in AGENTS])
                ),
                "pairs": result[key]["pairs"],
            }
            for key in ("test", "test_zero_peer", "test_shuffled_peer")
        }

    (OUT / "latent.json").write_text(json.dumps(export, indent=1) + "\n")
    print(json.dumps({k: export[k] for k in ("dims",)}, indent=1))
    print("sample", {k: v for k, v in export["sample"].items() if not isinstance(v, list)})
    for arm in AGENTS:
        a = export["splits"]["test"]["arms"][arm]
        print(arm, a["raw_stats"], a["layernorm_stats"], "msg absmax", a["message_absmax"])
        print("  explained raw/msg/ctx", a["raw_pca_explained"], a["message_pca_explained"], a["context_pca_explained"])
    print(json.dumps(export["results"], indent=1))


if __name__ == "__main__":
    main()
