# Is the learned message space standardized across models?

**Short answer:** not in the packaged checkpoint. It is when two changes are made
in training, on this small held-out test.

## The packaged checkpoint uses private codes

`scripts/latent_alignment_probe.py` tests `learned-latent/models/embeddings.pt`
unchanged ([result](packaged-checkpoint-probe.json)):

- SmolVLA's "I see an obstacle" direction and X-VLA's are orthogonal in the 64-D
  space (cosine −0.03).
- If a receiver hears the other family's sender describe the mirrored situation
  (same physical state, obstacle on the other side), hold accuracy falls from
  100% to 47.5% (left arm) and 10% (right arm).

Each direction learned its own code. The 64 numbers are a shared *shape*, not a
shared *meaning*. That matches the original design note: a common width is an
interface, not alignment.

## A standardized variant

`scripts/shared_latent_experiment.py` trains four designs with the same data,
optimizer, 400 epochs and dev-loss selection, over 5 seeds each:

| Design | Change from the packaged design |
| --- | --- |
| pairwise | none (each receiver may learn a private code) |
| shared receiver only | one receiver module used by both arms |
| anchor only | one linear decoder, shared by all senders, must read each sender's *own* 23 measured local facts out of its message (training-time loss only) |
| shared | both |

To get same-camera features from both families, `scripts/cache_cross_features.py`
ran X-VLA on the left camera and SmolVLA on the right camera: 160 real forward
passes each ([features](features/)).

### Results (held-out episode 104, mean over 5 seeds, minimum in brackets)

| Design | Normal task | Swap senders | Plug-in: trained arm | **Plug-in: other arm, zero-shot** | Obstacle-direction cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| pairwise | 100% | 51.0% (47.5) | 100% | **48.0%** (45.0) | 0.05 |
| shared receiver only | 100% | 51.0% (18.8) | 100% | **49.5%** (11.2) | 0.01 |
| anchor only | 100% | 74.3% (65.0) | 100% | **73.8%** (65.0) | 0.41 |
| shared | 100% | **98.5%** (92.5) | 100% | **96.8%** (92.5) | 0.47 |

All accuracies are hold decisions where the receiver's own view is clear, so only
the message can tell it to stop (40 per arm per test).

- **Swap senders:** any receiver understands any sender only in the `shared` design.
- **Plug-in:** a new model (X-VLA on the left camera, or SmolVLA on the right
  camera) joins by training *only its encoder*, against one arm, with everything
  else frozen. In `shared`, the *other* arm understands it with no training
  (96.8%). In `pairwise`, it is at chance (48.0%). That is the property that
  scales: with N models, a standardized space needs one encoder per new model
  instead of re-integrating every pair.
- **Both changes are needed.** The shared receiver alone does nothing; the shared
  fact anchor alone gets about three quarters of the way.
- Even the existing X-VLA sender, trained only on the right camera, works on
  left-camera features with no training in `shared` (96.5%; pairwise 52.8%).

[Full per-seed results](report.json).

## Limits

- **Small test:** one held-out episode, 20 physical states and one hazard
  concept (hold vs go). Decisions from the same state are correlated.
- **One-step diagnostic:** the same as the packaged experiment, not a closed-loop
  fold. The delay and live-execution limits of the
  [learned-latent experiment](../learned-latent/README.md) still apply.
- **The anchor is the mechanism.** The space is standardized by grounding every
  sender's message in a common vocabulary of measured local facts during
  training. A new model needs those measurements from its own robot to be trained
  in. Nothing extra is transmitted at runtime.
- **Two VLA families, two cameras.** "Every model" is not tested; a third family
  and other tasks are the next step.
- **The swap test uses mirrored states:** a receiver hears a message about the
  same physical state with the obstacle on the other side. It tests whether the
  meaning of "obstacle reported" is shared, not a physical scenario that occurs
  in this cell.

## Reproduce

```bash
# from the repository root; needs the pinned model caches in .cache/huggingface/hub
PYTHONPATH=src .venv/bin/python scripts/cache_cross_features.py \
  --dataset examples/cloth-coordination/learned-latent/dataset --output runs/cross-features
PYTHONPATH=src .venv/bin/python scripts/latent_alignment_probe.py
PYTHONPATH=src .venv/bin/python scripts/shared_latent_experiment.py \
  --cross runs/cross-features --output runs/shared-latent
```

The packaged features in [features/](features/) can be passed as `--cross` to
skip extraction.
