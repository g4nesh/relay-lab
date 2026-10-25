"""Pairwise private codes vs one standardized message space, on the held-out episode.

Architectures (both reuse ``DualLatentAdapter`` in ``embeddings`` mode):

* ``pairwise``: the packaged design. Each receiver has its own decoder, so each
  direction may invent a private code.
* ``shared``: one receiver module used by every arm, plus one shared linear
  "anchor" decoder that must read each sender's *own* measured facts (23 local
  features, z-scored) out of its message during training. Every sender is
  therefore pushed into the same semantic layout. The anchor is training-time
  supervision only; nothing extra is transmitted.
* ``shared_receiver_only`` and ``anchor_only``: ablations with one ingredient each.

Tests, all on the held-out episode (seed 104), 5 training seeds each:

1. ``task``: hidden-peer hold accuracy with the normal pairing.
2. ``swap``: a receiver hears the *other* family's sender describe the mirrored
   situation (same physical state, obstacle on the other side). Interchangeable
   messages keep the decision correct; private codes do not.
3. ``plug-in``: a new model joins. Only its encoder is trained (everything else
   frozen), against ONE arm. ``plugin_trained_arm`` checks that arm; the key test,
   ``plugin_other_arm``, checks the *other* arm zero-shot (mirrored states).
   New models: X-VLA on the left camera, and SmolVLA on the right camera, from
   ``cache_cross_features.py``.

    PYTHONPATH=src .venv/bin/python scripts/shared_latent_experiment.py \
        --cross <repo>/runs/cross-features --output runs/shared-latent
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from relay_lab.cloth.communication import AGENTS, FEATURES
from relay_lab.cloth.latent_adapter import DIMENSION, DualLatentAdapter
from relay_lab.cloth.latent_training import load_pairs, loss_for, predictions

EVIDENCE = Path("examples/cloth-coordination/learned-latent")
EPOCHS, SEEDS, ANCHOR_WEIGHT = 400, (0, 1, 2, 3, 4), 0.5
OTHER = {"left": "right", "right": "left"}


def sender(width):
    return nn.Sequential(
        nn.LayerNorm(width), nn.Linear(width, 96), nn.GELU(), nn.Linear(96, DIMENSION), nn.Tanh()
    )


class Variant(DualLatentAdapter):
    """``shared_receiver`` ties one receiver across arms; ``anchor`` adds the shared fact decoder."""

    def __init__(self, shared_receiver, anchor):
        super().__init__("embeddings")
        if shared_receiver:
            self.receivers["right"] = self.receivers["left"]
        if anchor:
            self.anchor = nn.Linear(DIMENSION, len(FEATURES))


KINDS = {
    "pairwise": (False, False),
    "shared_receiver_only": (True, False),
    "anchor_only": (False, True),
    "shared": (True, True),
}


def load(cross_dir):
    data, _, _ = load_pairs(EVIDENCE / "dataset", EVIDENCE / "features")
    rows = json.loads((EVIDENCE / "dataset/rows.json").read_text())
    cross = {}
    for family, arm in (("xvla", "left"), ("smolvla", "right")):
        stored = np.load(Path(cross_dir) / f"{family}_{arm}.npz")
        cross[arm] = dict(zip(stored["row_indices"].tolist(), stored["features"], strict=True))
    index = {(r["pair_id"], r["agent"]): i for i, r in enumerate(rows)}
    meta = {r["pair_id"]: r for r in rows if r["agent"] == "left"}
    for batch in data.values():
        batch["cross"] = {
            a: torch.tensor(np.stack([cross[a][index[p, a]] for p in batch["pair_ids"]]))
            for a in AGENTS
        }
        batch["cf"] = np.array([meta[p]["counterfactual"] for p in batch["pair_ids"]])
        batch["snap"] = np.array([meta[p]["snapshot_id"] for p in batch["pair_ids"]])
    facts = torch.cat([data["train"]["facts"][a] for a in AGENTS])
    mean, std = facts.mean(0), facts.std(0).clamp(min=1e-3)
    for batch in data.values():
        batch["zfacts"] = {a: (batch["facts"][a] - mean) / std for a in AGENTS}
    return data


def arm_loss(raw, batch, arm, idx=slice(None)):
    nominal = batch["nominal"][arm][idx]
    return (
        F.mse_loss(raw[:, :3].tanh(), nominal[:, :3])
        + 0.3 * F.binary_cross_entropy_with_logits(raw[:, 3], nominal[:, 3])
        + F.binary_cross_entropy_with_logits(raw[:, 4], batch["hold"][arm][idx])
    )


def anchor_loss(model, messages, batch):
    if not hasattr(model, "anchor"):
        return torch.zeros(())
    return sum(F.mse_loss(model.anchor(messages[a]), batch["zfacts"][a]) for a in messages) / len(
        messages
    )


def fit(parameters, objective, dev_objective, state_of, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5)
        optimizer.step()
        with torch.no_grad():
            dev = float(dev_objective())
        if dev < best:
            best, best_state = dev, copy.deepcopy(state_of())
    return best_state


def train_base(kind, data, seed):
    torch.manual_seed(seed)
    model = Variant(*KINDS[kind])

    def total(batch):
        model.train()
        messages = {a: model.encode(a, batch["features"][a]) for a in AGENTS}
        return loss_for(predictions(model, batch), batch) + ANCHOR_WEIGHT * anchor_loss(
            model, messages, batch
        )

    def dev():
        model.eval()
        return loss_for(predictions(model, data["dev"]), data["dev"])

    model.load_state_dict(
        fit(list(model.parameters()), lambda: total(data["train"]), dev, model.state_dict)
    )
    return model.eval().requires_grad_(False)


def decide(model, batch, receiver, idx, messages):
    raw = model.decode_action(
        receiver,
        batch["features"][receiver][idx],
        batch["states"][receiver][idx],
        messages,
        batch["facts"][OTHER[receiver]][idx] * 0,
        torch.ones(len(idx)),
        torch.zeros(len(idx)),
    )
    return float(((raw[:, 4] >= 0) == batch["hold"][receiver][idx].bool()).float().mean())


def clear_view(batch, receiver):
    """Rows where the receiver's own view is clear: only a message can tell it to hold."""
    return np.flatnonzero(batch["cf"] != f"{receiver}_obstacle")


def mirrored(batch, receiver, about_own_camera):
    """Messages about the receiver's own camera at the mirrored state of each row."""
    rows, cf, snap = clear_view(batch, receiver), batch["cf"], batch["snap"]
    picks = []
    for i in rows:
        if cf[i] == f"{OTHER[receiver]}_obstacle":
            picks.append(int(np.flatnonzero((snap == snap[i]) & (cf == f"{receiver}_obstacle"))[0]))
        else:
            picks.append(int(i))
    return torch.tensor(rows), about_own_camera[torch.tensor(picks)]


def evaluate_base(model, batch):
    with torch.no_grad():
        messages = {a: model.encode(a, batch["features"][a]) for a in AGENTS}
        out = {"task": {}, "swap": {}}
        for r in AGENTS:
            rows = torch.tensor(clear_view(batch, r))
            out["task"][r] = decide(model, batch, r, rows, messages[OTHER[r]][rows])
            idx, swapped = mirrored(batch, r, messages[r])
            out["swap"][r] = decide(model, batch, r, idx, swapped)
        d = {
            a: (
                messages[a][torch.tensor(batch["cf"] == f"{a}_obstacle")].mean(0)
                - messages[a][torch.tensor(batch["cf"] != f"{a}_obstacle")].mean(0)
            )
            for a in AGENTS
        }
        out["obstacle_direction_cosine"] = float(F.cosine_similarity(d["left"], d["right"], dim=0))
    return out


def plug_in(model, data, camera, seed):
    """Train only a new encoder for ``camera`` (cross family) against the frozen system.

    The arm that hears this camera in normal operation is OTHER[camera]; that is the
    only arm whose loss trains the encoder.
    """
    torch.manual_seed(1000 + seed)
    width = data["train"]["cross"][camera].shape[1]
    encoder, listener = sender(width), OTHER[camera]

    def objective(batch, train=True):
        encoder.train(train)
        message = encoder(batch["cross"][camera])
        raw = model.decode_action(
            listener,
            batch["features"][listener],
            batch["states"][listener],
            message,
            batch["facts"][camera] * 0,
            torch.ones(batch["size"]),
            torch.zeros(batch["size"]),
        )
        loss = arm_loss(raw, batch, listener)
        if train:
            loss = loss + ANCHOR_WEIGHT * anchor_loss(model, {camera: message}, batch)
        return loss

    encoder.load_state_dict(
        fit(
            list(encoder.parameters()),
            lambda: objective(data["train"]),
            lambda: objective(data["dev"], train=False),
            encoder.state_dict,
        )
    )
    encoder.eval()
    test = data["test"]
    with torch.no_grad():
        message = encoder(test["cross"][camera])
        rows = torch.tensor(clear_view(test, listener))
        trained = decide(model, test, listener, rows, message[rows])
        idx, swapped = mirrored(test, camera, message)
        other = decide(model, test, camera, idx, swapped)
        # Diagnostic: the existing sender of the same family, applied to the other
        # camera with no training (X-VLA's sender was trained on the right camera).
        existing = model.encode(OTHER[camera], test["cross"][camera])
        zero_shot = decide(model, test, listener, rows, existing[rows])
    return {
        "plugin_trained_arm": trained,
        "plugin_other_arm": other,
        "existing_sender_zero_shot": zero_shot,
    }


def summarize(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    data = load(args.cross)
    report = {"seeds": list(SEEDS), "epochs": EPOCHS, "anchor_weight": ANCHOR_WEIGHT, "runs": {}}
    for kind in KINDS:
        runs = []
        for seed in SEEDS:
            model = train_base(kind, data, seed)
            result = evaluate_base(model, data["test"])
            result["plugin"] = {
                "xvla_on_left_camera": plug_in(model, data, "left", seed),
                "smolvla_on_right_camera": plug_in(model, data, "right", seed),
            }
            runs.append(result)
            print(kind, seed, json.dumps(result), flush=True)
        report["runs"][kind] = runs
    summary = {}
    for kind, runs in report["runs"].items():
        summary[kind] = {
            "task": summarize([np.mean(list(r["task"].values())) for r in runs]),
            "swap": summarize([np.mean(list(r["swap"].values())) for r in runs]),
            "obstacle_direction_cosine": summarize([r["obstacle_direction_cosine"] for r in runs]),
        }
        for key in ("plugin_trained_arm", "plugin_other_arm", "existing_sender_zero_shot"):
            summary[kind][key] = summarize(
                [np.mean([p[key] for p in r["plugin"].values()]) for r in runs]
            )
    report["summary"] = summary
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
