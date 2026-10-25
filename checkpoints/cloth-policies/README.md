# Original cloth workcell controllers

These two small, independently trained policies are **state-assisted task controllers**, not released foundation VLAs. Their inputs are each robot's local RGB, explicitly measured local cloth landmarks and visibility masks, its own end-effector/grasp state, a phase instruction and a coordinator-specified clearance. They never receive the world object, the other arm's observation, hidden cloth vertices or an evaluation success label.

The left model uses spatial convolutions with phase-conditioned affine experts. The right model uses image patch attention with different learned weights and phase-conditioned affine experts. The expert weights are fitted from original local-state demonstrations; the visual residuals are trained separately. Precise geometric control is largely explained by the explicit local measurements. RGB contexts are randomized during training, so **this experiment does not establish that pixels are essential to these controllers**. Calling these general-purpose visual robot models would be incorrect.

The 80 original MuJoCo RGB contexts per arm come from training seeds 10000–10003, including changing physical obstacles and physically verified initial-contact grasps. Each policy is fit to 18,000 synthetic local-state examples and checked against 3,000 held-out state examples. The latter establish imitation accuracy, not whole-task success. Full physical rollouts provide the task-level test. No DuoMind or RoboPoly demonstrations are used.

The task rollout seed partition is fixed: **development 0–99, evaluation 100–9999,
training 10000 and above**. Supervised numeric state generation uses separate
streams 40000/40001 and minibatch streams 42000/42001; its isolated regression
validation uses 50000/50001. These regression checks are not held-out physical
task rollouts. Additional codec training trajectories, when present, use training
seeds such as 10004 and are documented in that codec's metadata.

The current frozen models use the `reachable-arc-v2` control recipe: a reachable
hem lift, a lower carry arc, and measured local shoulder coordinates for placement.
Their feature basis is included in the checkpoint and checked at inference.
`pre-arc/` preserves earlier development weights for inspection; they are not the
weights used in the final matched comparison. Runtime code supports their original
feature basis, so a saved historical model is not silently reinterpreted.

`evaluation.json` records architectures, parameter counts, training seeds, held-out errors and checkpoint hashes. `communication-training.npy` contains the 160 original 21-dimensional local observations collected with those RGB contexts, available for fitting the communication codec. `communication-training.json` contains the same observations with named fields. `collection.json` records their provenance. The code does not fit communication representations on evaluation episodes.

Reproduce from the repository root:

```bash
uv run --extra learned python -m relay_lab.cloth.train_policies all --seeds 4 --steps 300
```

The exact compact RGB contexts used for the frozen checkpoint are also included
in `training/`. Retrain from these with `train --data checkpoints/cloth-policies/training`
to preserve the original image bytes while the workcell's visual materials evolve.
`data-manifest.json` records the original source file hashes.

Grasps in the environment are explicitly idealized, proximity-checked point constraints. A separately logged coordinator hold can override a learned target when communication or local conditions are unsafe. The raw network output is still recorded; there is no hidden reference-controller fallback.

Released model transfer is a separate experiment: `relay_lab.cloth.pretrained` has pinned adapters for `HuggingFaceVLA/smolvla_libero` and `lerobot/xvla-folding`. Their native camera and embodiment distributions differ from this original workcell. Merely loading those checkpoints is not evidence of successful cloth folding.

The released-model integration probe is reproducible with:

```bash
uv run --extra vla python -m relay_lab.cloth.transfer_probe smolvla --calls 12
uv run --extra vla python scripts/download_cloth_xvla.py
uv run --extra vla python -m relay_lab.cloth.transfer_probe xvla --calls 4
```

The second command resumes downloaded public weight ranges and verifies the full
publisher SHA-256 before installing them. The adapter uses the original Soft-FOLD
domain ID 5, interleaved 6D rotations and official ImageNet normalization; missing
peer state is explicitly unknown/zero, and only the local arm's action block is
executed. Neither integration command constitutes a cooperative folding benchmark.

A separate bounded two-model integration trial connects both released policies
to the same MuJoCo world and the actual Spacetime journal:

```bash
uv run --extra vla python -m relay_lab.cloth.pretrained_team \
  --output runs/cloth-pretrained-team --calls 12 --mode hybrid \
  --coordination spacetime --mediator qwen
```

Both models receive their own local image and proprioception at the same physics
tick. Decoded, delivered peer measurements become language context; Qwen is
called when local or reported grasp/obstacle categories change. Raw action
chunks, applied Cartesian bounds, instructions, message identities and physical
outcomes are retained. The default `--mediator template` formats facts without
an LLM. This short transfer trial is separate from the controlled communication
comparison and provides no evidence of cooperative folding until its recorded
physical outcome actually establishes it.
