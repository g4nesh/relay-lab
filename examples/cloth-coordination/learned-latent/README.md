# Learned messages between SmolVLA and X-VLA

This experiment implements a **CoLA-inspired learned channel** between genuine
frozen features from two different VLA families. It is distinct from the older
eight-dimensional PCA channel. The native action heads are replaced by small,
state-assisted Cartesian heads. It is not a reproduction of CoLA's diffusion
controller or VLA-Fabric's internal action-generation interactions.

```text
SmolVLA RGB/language features (960D) → learned encoder → 64D message
                                                       ↓ SpacetimeDB
X-VLA local features (1024D) + learned receiver → learned action head
```

The reverse direction has its own learned sender and receiver. Both foundation
backbones stay frozen; 324,874 adapter/head parameters train jointly. No original
paper demonstration dataset is used. Qwen and the previous complete-fold
baseline remain available separately; Qwen supplies no hidden peer-information
channel in this controlled experiment.

## Fresh private-observation test

Two original episodes train the model, one selects checkpoints, and a fourth is
held out. Each episode supplies 20 MuJoCo states with clear, left-obstructed and
right-obstructed counterfactuals. Local pixels and measurements are **exactly
identical** between clear and hidden-peer-obstacle cases for the receiving arm.
The sender must supply the distinguishing information.

| Interface | Hidden-peer accuracy | All hold decisions | Mean Cartesian action error |
| --- | ---: | ---: | ---: |
| No communication, separately trained | 50.0% | 66.7% | 4.68 mm |
| Structured text | 100.0% | 100.0% | 1.09 mm |
| Learned embeddings | 100.0% | 100.0% | 0.89 mm |
| Hybrid | 100.0% | 100.0% | 0.96 mm |

Zeroing the trained embedding payload reduces hidden-peer accuracy to 50.0%.
Shuffling its payload reduces it to 25.0%. These interventions preserve the
trained receiver; the no-communication baseline is trained independently.

**This is one held-out episode, not a broad success-rate estimate.** Its 120 arm
decisions share only 20 physical states. The balanced hidden-peer subset contains
80 arm decisions. Action error is a one-step teacher-imitation measurement, not
a completed fold. Structured text has exact simulator measurements while latent
messages have visual/language features, so this is a system comparison with
different information sources. [Raw training and evaluation](models/evaluation.json).

## Delays expose a failure

The same held-out scene sequence changes private obstacle conditions every 16
diagnostic ticks. All formats use the same seed, publication times, drop schedule
and expiry. SpacetimeDB commits are independently read back. These are changing
observations on frozen states; the predicted actions are **not integrated**.

| Interface | One-tick hold accuracy | Delayed/dropped hold accuracy | Delayed mean JSON packet |
| --- | ---: | ---: | ---: |
| No communication | 66.7% | 66.7% | No packets |
| Structured text | 87.3% | 66.3% | 1,112.6 B |
| Learned embeddings | 87.3% | 66.3% | 1,666.9 B |
| Hybrid | 87.3% | 66.3% | 1,916.5 B |

The delayed condition uses 2–18 tick delays, 15% configured drops, and a 24-tick
TTL. A post-hoc breakdown on locally clear observations gives **49.375%**
hidden-peer accuracy for all three communication formats, versus 50% without
communication. Thus fresh-message usefulness does not carry over to this rapid
condition-change schedule. Text is the smallest of these actual JSON interfaces;
the 64D channel has no measured bandwidth advantage here.

The three formats have identical hold decisions in this sequence. They receive
287 delivered packets each, reject 112 out-of-order packets, and retain all
attempts. Synchronous durable commits average approximately 4–6 ms on this
machine; that excludes backbone inference and is not a real-time control claim.
Training used fresh observations only. The age-input weights start at zero;
expiry and missing-message holds are explicit guards, not learned delay recovery.
[All eight attempts](transport/manifest.json).

## Live execution check: coordination still fails

The embedding checkpoint also ran in a fresh MuJoCo scene with **24 actual
backbone forward passes, 24 delivered learned messages, 12 independently read
back database frames, and 48 physics ticks**. Both arms physically moved using
the replacement heads. No geometric target fallback was invoked.

The short trial missed **12 of 14 required obstacle holds**, made **zero observed
grasps**, and completed **no fold**. This is a successful integration check and a
failed behavioral check. The live arm/obstacle poses and historical peer inputs
are not the same as the fresh counterfactual training samples; this single trial
does not isolate which factor caused each error.

The 1.92 simulated seconds took 61.0 wall seconds including model loading,
rendering and transport. Mean feature extraction was 1.13 s for SmolVLA and
0.66 s for X-VLA. Physics pauses during inference. The fixture was released by
explicit environment cleanup after the action budget, not by a learned decision.
[Raw live record](live/report.json) · [Recording](live/overview.mp4).

The next training data needs complete action transitions, varied obstacle poses,
and actual historical/missing peer messages. A complete held-out fold remains
required before making a manipulation-success claim. See the
[paper comparison and implementation contract](../../../docs/latent-coordination.md).

## Reproduce and inspect

The original aligned images/labels, frozen feature caches, all four trained
checkpoints, raw decision/channel traces, and source/file hashes are included
in this directory. Dataset collection has 480 rows; each backbone cache contains
240 rows from 160 actual forward passes after exact-input deduplication.

```bash
.venv/bin/python -m relay_lab.cloth.latent_training train \
  --dataset examples/cloth-coordination/learned-latent/dataset \
  --cache examples/cloth-coordination/learned-latent/features \
  --output runs/retrained-latent-adapters --epochs 400
.venv/bin/relay-lab cloth
```

Open `http://127.0.0.1:8769/latent` for the evidence page. See the root README for
fresh collection and extraction commands. `files.json` hashes the packaged
binary and JSON evidence; the report prose and preview image are not inputs to
training or evaluation.
