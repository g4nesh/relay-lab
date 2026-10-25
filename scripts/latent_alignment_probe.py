"""Do the two senders of a trained cross-VLA checkpoint share message semantics?

Two checks on the held-out episode, using the packaged checkpoint unchanged:

1. Direction alignment: cosine between each sender's "I see an obstacle" direction
   (mean message with its own obstacle minus mean without) in the 64-D space.
2. Swapped decoding: give a receiver the *other* sender's message for the matched
   physical state (e.g. the left arm hears SmolVLA saying "obstacle" instead of
   X-VLA saying "obstacle"). A standardized space would keep the decision correct.

    PYTHONPATH=src .venv/bin/python scripts/latent_alignment_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from relay_lab.cloth.communication import AGENTS
from relay_lab.cloth.latent_adapter import DualLatentAdapter
from relay_lab.cloth.latent_training import load_pairs

EVIDENCE = Path("examples/cloth-coordination/learned-latent")


def obstacle_direction(messages, own_obstacle):
    return messages[own_obstacle].mean(0) - messages[~own_obstacle].mean(0)


def swapped_accuracy(model, batch, meta, receiver):
    """Receiver hears its *own* family's sender describing the mirrored situation."""
    sender = "left" if receiver == "right" else "right"
    cf = np.array([m["counterfactual"] for m in meta])
    snap = np.array([m["snapshot_id"] for m in meta])
    own_side = f"{receiver}_obstacle"
    peer_side = f"{sender}_obstacle"
    with torch.inference_mode():
        messages = {a: model.encode(a, batch["features"][a]) for a in AGENTS}
    rows, normal, swapped = [], [], []
    for i in range(batch["size"]):
        if cf[i] == own_side:  # receiver sees the obstacle itself; not a peer-information case
            continue
        rows.append(i)
        normal.append(messages[sender][i])
        if cf[i] == peer_side:
            # Same physical state, but the obstacle on the receiver's own side, so the
            # receiver's own-family sender is the one reporting "I see an obstacle".
            j = int(np.flatnonzero((snap == snap[i]) & (cf == own_side))[0])
        else:
            j = i
        swapped.append(messages[receiver][j])
    idx = torch.tensor(rows)
    labels = batch["hold"][receiver][idx].bool()
    out = {}
    for name, msgs in (("delivered", normal), ("swapped_sender", swapped)):
        with torch.inference_mode():
            raw = model.decode_action(
                receiver,
                batch["features"][receiver][idx],
                batch["states"][receiver][idx],
                torch.stack(msgs),
                batch["facts"][sender][idx],
                torch.ones(len(rows)),
                torch.zeros(len(rows)),
            )
        out[name] = float(((raw[:, 4] >= 0) == labels).float().mean())
    out["decisions"] = len(rows)
    return out


def main(checkpoint=EVIDENCE / "models/embeddings.pt"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DualLatentAdapter(payload["mode"]).eval().requires_grad_(False)
    model.load_state_dict(payload["state_dict"], strict=True)
    data, _, _ = load_pairs(EVIDENCE / "dataset", EVIDENCE / "features")
    rows = json.loads((EVIDENCE / "dataset/rows.json").read_text())
    by_pair = {r["pair_id"]: r for r in rows if r["agent"] == "left"}
    batch = data["test"]
    meta = [by_pair[p] for p in batch["pair_ids"]]
    cf = np.array([m["counterfactual"] for m in meta])
    with torch.inference_mode():
        msg = {a: model.encode(a, batch["features"][a]).numpy() for a in AGENTS}
    d = {a: obstacle_direction(msg[a], cf == f"{a}_obstacle") for a in AGENTS}
    cosine = float(d["left"] @ d["right"] / np.linalg.norm(d["left"]) / np.linalg.norm(d["right"]))
    report = {
        "checkpoint": str(checkpoint),
        "obstacle_direction_cosine": cosine,
        "receivers": {r: swapped_accuracy(model, batch, meta, r) for r in AGENTS},
    }
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
