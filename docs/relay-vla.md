# Compact Relay VLA

The working controller is a **new, task-specific learned policy**, separate from
SmolVLA. It has 36,532 parameters and runs on the CPU. SmolVLA's unadapted LIBERO
checkpoint remains an experimental comparison: this change does not establish
that the pretrained SmolVLA model can grasp in Relay Lab.

The compact policy receives the robot's **local front RGB image, end-effector
position, and instruction text**. It has no access to simulator object poses,
holding flags, target dictionaries, packing order, or task success. Each
translation and gripper command comes from its neural action head. The ordinary
motion bound is applied to its output, exactly as for other policies. No scripted
controller or waypoint fallback runs during evaluation.

## Model and scope

A small pixel classifier recognizes parcel colours. Confidence-weighted image
centroids pass through a camera calibration fitted to original rendered examples.
A learned bag-of-words language router selects a visually estimated parcel or a
calibrated target embedding. A three-layer neural controller maps that goal and
current proprioception to a translation and three gripper logits.

The targets are measured during dataset generation, then saved as calibration
parameters; they are not queried from the simulator at inference time. Picking
assumes parcels rest on this workcell's table. This is a deliberately narrow
vision/language/action model, **not** an open-vocabulary robot foundation model.
Moving cameras or packing targets, changing parcel appearance, or adding tasks
requires new training and evaluation.

The visual goal is retained for the current instruction so the robot's own
fingers do not obscure the object during descent. A changed instruction or an
explicit recovery replan refreshes the RGB estimate. The motor network still
runs every tick. Logs record whether vision refreshed and which observation tick
supplied the cached goal. Mediator feedback becomes the next validated instruction;
old feedback text is not mixed into the small model's language tokens.

## Training our own policy

All training data is generated locally. No RoboPoly, DuoMind, LIBERO, or other
external demonstrations are used. MuJoCo segmentation and object coordinates are
supervision **only during dataset generation and calibration**. A reactive expert
produces synthetic action labels across randomly perturbed states, including
near-grasp and near-release states. That teacher exists only in the training
module and is not imported by the runtime policy.

From the project directory:

```bash
uv sync --extra learned
uv run --extra learned python -m relay_lab.train_relay_vla collect --scenes 400 --seed 2000
uv run --extra learned python -m relay_lab.train_relay_vla train --seed 17 --pixel-steps 1500 --control-steps 10000
uv run --extra learned python -m relay_lab.train_relay_vla evaluate --seeds 7 42 101
```

Default outputs:

- `datasets/relay-vla/original.npz`: 400 randomized, original camera examples.
- `checkpoints/relay-vla/policy.pt`: learned pixel, calibration, language, target,
  and action parameters.
- `checkpoints/relay-vla/policy.json`: training settings and measurements.
- `runs/relay-vla-evaluation/`: complete action traces and summary.

Training does not require a GPU. The measured collection took 2.61 seconds and
training took 20.54 seconds on this M3 Pro. These are one local measurement, not
a hardware-independent speed claim. The reference dataset uses scene seeds
2000–2399 and the evaluated task seeds below are disjoint.

## Verified controller results

These controller tests used the existing **rule planner**, so they establish
learned motor execution, not LLM-planning quality. The integrated LLM/Spacetime
rollout is a separate validation recorded by the main runner.

| Held-out seed | Task outcome | Simulator steps |
| --- | --- | --- |
| 7 | 3/3 parcels in the requested order | 261 |
| 42 | 3/3 parcels in the requested order | 264 |
| 101 | 3/3 parcels in the requested order | 265 |

Across these three runs, the network performed 1,580 robot action inferences in
0.296 seconds total, excluding simulation, rendering, database operations, and
planning. Goal vision is cached as described above; this is not a claim of a
new image encoding every tick. Simulator grasp radius and task success criteria
were unchanged.

The reference checkpoint is 152,589 bytes. Its SHA-256 is
`7b13dbba5c0e64b491d5a8f3e63984b0e2d458cb2ea08714eb91a1f5c5b19c38`.
The original dataset SHA-256 is
`4f1d3619a4b97291eef5c2fdbdff84f7f679ad47a0ec59aaa8b683eb2b25de51`.
The runtime includes the loaded checkpoint hash in each run's model metadata.

Three successful seeds are useful engineering evidence, not a broad robotics
benchmark. The local pixel classifier and fixed workcell calibration are narrow;
there is no claim of transfer to physical robots or an unfamiliar simulator.
