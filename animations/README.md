# Animations: learned messages between SmolVLA and X-VLA

Manim scenes explaining the [learned-latent experiment](../examples/cloth-coordination/learned-latent/README.md):
frozen SmolVLA (960-D) and X-VLA (1024-D) features, learned 64-D messages,
directed receivers, and the replacement action heads. The implementation is
CoLA-inspired; it is not CoLA's diffusion policy or VLA-Fabric, and the native
VLA action heads are not used.

| Scene | What it shows |
| --- | --- |
| `PrivateViews` | The two real 128×128 camera frames for held-out state `test-104-s005`; the right camera's frame is byte-identical with and without the hidden obstacle. |
| `TwoSpaces` | Feature widths and extraction layers, PCA of all 60 held-out feature vectors per model, and per-vector LayerNorm statistics: normalizing gives matching statistics, not shared coordinates. |
| `Sender` | `senders[arm]` (LayerNorm → Linear → GELU → Linear → tanh) with the actual activations for the sample, the clear-scene message for the same state, and PCA of all held-out messages. |
| `Receiver` | The directed channel through SpacetimeDB, `receivers[arm]`, the validity mask `c = R(m·v)·v`, and PCA of the receivers' 32-D contexts. |
| `ActionHead` | The 128-D head input (32 + 39 + 32 + 23 + 1 + 1), the MLP, and the five outputs. |
| `Training` | The joint objective, the cross-arm gradient path, and the recorded train/dev loss curves (selected epoch 329). |
| `Results` | The 80 hidden-peer held-out decisions with delivered, zeroed and shuffled messages, then the four interfaces' accuracy and action error. |
| `Closing` | Summary and open problems. |
| `RelayLatentFull` | All of the above in sequence. |

[`short_scene.py`](short_scene.py) has `RelayLatentShort`, a 28-second
plain-language cut of the same data for demos: the two camera views, a
"translator" on each side passing a 64-number note, and the held-out result
(50% without the note, 100% with it, on decisions where the robot cannot see
the hazard itself).

Rendered 1080p60 videos are in [`renders/`](renders/).

## Data provenance

Every number and image comes from [`data/latent.json`](data/latent.json) and the
PNGs beside it, which [`extract.py`](extract.py) produces from the packaged
evidence without retraining:

- it loads the dataset and frozen feature caches through
  `relay_lab.cloth.latent_training.load_pairs`, which re-verifies every hash and
  alignment;
- it runs `examples/cloth-coordination/learned-latent/models/embeddings.pt` and
  records sender activations, messages, receiver contexts and hold probabilities;
- PCA projections are computed on the held-out split only;
- accuracy and action-error figures are read from the packaged `evaluation.json`.
  The scenes assert that hold accuracy recomputed from the exported probabilities
  matches it.

Limits shown in the videos match the experiment README: one held-out episode
(120 arm decisions from 20 physical states), a one-step diagnostic rather than a
completed fold, structured text reading exact simulator facts while messages come
from camera and language features, 66.3% hold accuracy under delayed and dropped
messages, and zero grasps in the first live trial.

## Regenerate

```bash
# 1. Export data (project environment, from the repository root)
PYTHONPATH=src .venv/bin/python animations/extract.py

# 2. One-time: Manim from conda-forge (pycairo has no macOS wheels on PyPI)
micromamba create -p animations/.env -c conda-forge python=3.12 "manim=0.19" "pillow=10.4" numpy

# 3. Render (from animations/)
.env/bin/manim -qh latent_scenes.py RelayLatentFull
```

Pillow is pinned to 10.4: Manim 0.19's image compositing raises "Invalid mode
for data type" on Pillow 13.
The scenes use only `Text`/`MarkupText` (STIX Two Text and Menlo), so no LaTeX
installation is needed.
