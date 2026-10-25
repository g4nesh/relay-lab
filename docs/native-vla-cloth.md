# Native VLA cloth control

The foundation-control runtime supports **two independently adapted SmolVLA
instances**, and an experimental SmolVLA-left/X-VLA-right pair. The two-Smol
configuration shares an architecture, with separately identified arm weights and
normalization. Each instance executes its released vision/language backbone and
native action network, with supervised adaptation of existing native modules to
Relay's simulated workcell. This is task adaptation, not zero-shot cloth folding.

**Development result, 2026-10-25:** the packaged left v4 and right v3 SmolVLA
instances completed one dashboard-launched fold with Qwen and SpacetimeDB in
1,164 physics ticks. The native command audit passed for both arms; a separate
MuJoCo command replay reproduced the final cloth vertices bitwise and all
measurements exactly. Seed **10007 was used in training**. This is a successful
development demonstration, not a held-out success rate. Separate-scene hem-grasp
predictions remain inaccurate. X-VLA's final-block, pointwise-MLP and multimodal projection
trials failed validation. Its dense single-scene decoder fit achieved 2.27 mm
training p95, but 658 mm on omitted frames of that scene; it is unsuitable for
deployment. Neither training loss nor a
partial rollout is a release or task-success result.

The original CNN and attention controllers remain available for historical
experiments and offline demonstration collection. They are not a fallback in
`relay_lab.cloth.native_runtime`.

## What controls each movement

| Component | Responsibility |
| --- | --- |
| SmolVLA, each arm | Own RGB and measured pose plus an instruction → native flow action chunk. With `--horizon 1 --repeat 4`, execute only token 0 and retain its world target for at most four physics ticks. |
| Experimental X-VLA right arm | Own RGB and measured pose plus an instruction → native action decoder. With the same runtime options, retain only token 0's world target for at most four physics ticks. |
| Local phase guards | Check actual grasp identity, enforce obstacle/stale-message holds, and bound each phase. They cannot manufacture successful geometry. |
| Qwen | Review eligible phase transitions using local status and delivered peer information. |
| SpacetimeDB | Persist task assignments, worker reports and the simulated communication channel. |
| MuJoCo | Apply bounded Cartesian commands through IK, integrate cloth dynamics, and measure the result. |

Target reuse does not integrate an action repeatedly or execute untrained future
tokens. `used_index` stays `0` while `inference_age_ticks` advances from `0` to
`3`; a refresh produces a new inference identity. Instruction changes and
coordination holds invalidate retained actions. A hold commands the measured
current tool position and commanded gripper state and is explicitly marked as a
coordination action, not a fresh VLA prediction. Smol's token-0-only adapters
reject a requested execution horizon above their supervised horizon.

Each foundation policy receives an allowlisted observation: its own camera,
measured tool position and rotation, commanded gripper state, local grasp state,
and simulation tick. Cloth landmark coordinates and the other arm's current
observation are excluded. The coordination guards still use explicit local
measurements. This experiment does not establish autonomous visual task planning.

## Native network adaptation

- SmolVLA: [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero), revision
  `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Supported adapters update the existing
  output projection (v1), native state/action/time projections (v2), and the
  original final expert feed-forward/norm modules (v3). The current v3 candidate
  retains the v2 input-projection adaptation and additionally trains layer 31's
  post-attention norm and MLP, the expert's final norm, and `action_out_proj`.
  This is the architecture used by the packaged left v4 and right v3 adaptations.
  The VLM, expert attention, and earlier expert layers remain frozen. Its final
  training stage updates 1,859,552 native parameters. Frozen attention-residual
  caches accelerate this pointwise training path. The cached features come from
  the native model's measured inputs; teacher labels are stored separately as
  training targets. State and action normalization remains fixed
  to training seed 10004.
  The right-arm path derives its own statistics from the right arm of that
  training episode. Its versioned yaw reference is centered on π to avoid
  discontinuities at ±π in the mirrored rig. The original measured rotation
  matrix remains in the dataset; the reference is applied once to model state
  and supervised rotation labels. Predicted XYZ and gripper commands receive no
  mirror transform or teacher correction. Rotation columns are not actuated.
  Cross-arm initialization records its source artifact and inherited training
  seeds; an initializer is rejected for deployment until fitting is complete.
- X-VLA: [`lerobot/xvla-folding`](https://huggingface.co/lerobot/xvla-folding), revision
  `c87d6e8fcf47fb19d6aac8b67d35dd20d74bbc8d`. Supported scopes include the original
  domain-5 right-arm XYZ/gripper decoder columns, native domain-5 action encoder
  and soft prompts, and the original final action-transformer block. Florence-2
  remains frozen; the final-block variant also preserves the preceding 23
  transformer blocks. The narrower pointwise variant trains block 23's second
  norm and MLP, the original final norm and native decoder columns, with the
  final attention frozen. Those pointwise trials failed validation and are not
  released. The multimodal variant instead adapts the original `vlm_proj`,
  domain-5 soft prompts, final norm and selected decoder columns, propagating
  gradients through the 24 unchanged action-transformer blocks. Its 32-observation
  gate failed at 284.7 mm p95, so the larger fit was not started. Decoder fitting
  preserves other columns; deeper shared
  block adaptation can also change unexecuted predictions. Frozen Florence and
  native-prefix caches accelerate training without supplying teacher features.
- Both use one native denoising step and deterministic noise, identically during
  feature collection and deployment. This differs from the publishers' default
  inference configurations. X-VLA's noise backend must match the training cache.
- Both cached training paths are checked numerically against full native model
  inference. Deployment runs each full visual/language backbone again; it does
  not retrieve training features or target actions.

Both linked foundation model cards specify Apache-2.0. Distributed adapted
weights must retain the applicable upstream license and attribution; the local
adaptation does not remove those obligations.

The action frame is absolute Cartesian XYZ in the MuJoCo table frame, in metres.
Native rotations are retained in the evidence but not executed by this
fixed-orientation rig. The only target modification at deployment is clipping
to the work envelope; the simulator separately enforces actuator speed limits.

The runtime checks exact cached foundation-weight hashes and adapter bytes.
The dashboard candidates are `checkpoints/cloth-native/left.pt` (v4, SHA-256
`ad47dc42e1caf24477ab6a8551156ce6d08c3f8782c29ee7cb81e25006674371`)
and `right.pt` (v3, SHA-256
`c72f368f1e25365659017649ff5afbadc0aa08caf8004669a3ae83b8b19f3e02`).
Their adjacent training reports contain provenance and post-install equivalence
checks, whose maximum normalized-action differences were `1.67e-6` and `2.15e-6`.
This proves the cached training computation matches the native model
at the checked inputs; it does not prove a fold or generalization.

## Original data and evaluation boundaries

`python -m relay_lab.cloth.action_dataset` collects time-aligned local RGB,
measured proprioception and the actual commands submitted before each physics
step. The offline teacher may use landmarks, but those are not learner inputs.
Future action chunks stop at phase and coordination-hold boundaries. Collection
requires seeds at least 10000, and failed demonstrations are rejected by default.

Development corrections are a separate dataset type. They replay failed native
commands to recover the actual local observations, then ask the offline teacher
for one corrective label. The teacher does not actuate that replay. These labels
supervise only the first action; they are never presented as successful episodes
or padded into fictitious future trajectories.

Recovery trajectories are also a separate one-step dataset type. Controlled
exploration creates off-trajectory observations, and the offline teacher supplies
the corrective label. The source trajectory's actual outcome is retained even
when it fails; no future corrective outcomes are invented.

The left v3 warm start used this split:

| Role | Sources | Boundary |
| --- | --- | --- |
| Successful original training episodes | `10004-settled`, `10005`, `10007` | Ten active observations per phase per episode; 480 frames total. Seed 10005 was previously used for validation and was explicitly reclassified as training. |
| Development corrections | Failed native development seed `0` | 28 recorded local input states with one-step counterfactual labels; the teacher never actuated the replay. |
| Recovery training | `10009`, `10010` | 144 one-step observations. Recovery 10009's final physical fold failed; 10010's succeeded. Both retain their recorded status. |
| Validation/tuning | `10008` | 160 separate observations; excluded from fitting and normalization. This episode has been inspected during development and is not an untouched final test. |
| Excluded failed demonstrations | Original unsettled `10004`, `10006` | Preserved as failed sources. The separate `10004-settled` recording adds actual stationary physics steps and verifies its original prefix. |
| Reserved native evaluation | `103` | Excluded from native fitting; no complete two-VLA fold at this seed is claimed. |

For this candidate, training target-distance p95 is **18.4 mm** over 652 frames.
Validation p95 is **89.1 mm** overall, **22.9 mm** for sleeve grasp and **101.1 mm**
for hem grasp. Binary gripper prediction is 100% on these sampled observations;
that is not measured cloth attachment. These are one-step predictions on recorded
observations, not closed-loop success rates. The datasets, model snapshots and
reports are local artifacts rather than bundled third-party demonstrations.

The packaged **left v4** adds 167 counterfactual one-step labels from a failed
native-left/teacher-right development run at seed 10007. Exact replay verified
every input and executed no teacher corrections. Sleeve execution had shifted
the hem about 112 mm from the original demonstration. The v4 fit uses 819 rows;
training p95 is 20.7 mm, while separate-scene validation p95 remains 93.2 mm.
The **right v3** uses its own arm observations from the three original episodes
and two recovery sources (624 rows), with training p95 24.0 mm and validation
p95 65.0 mm. Its warm start is explicitly inherited from the left adaptation.
Unused recorded frames in scene 10007 give p95 errors of 68.5 mm (left) and
73.6 mm (right). These diagnostic errors prevent a robustness claim.

The mixed collector in `relay_lab.cloth.mixed_dataset` separately records an
actual native left controller beside the offline teacher on the right. Successful
imitation eligibility requires the source and fresh replay to agree on successful
geometry. Failed sources remain inspectable and are never silently promoted into
successful demonstrations. The correction dataset above has its own explicit
counterfactual contract.

Keep training and validation episodes separate. A low prediction error on a
demonstration is not a successful fold. Freeze the adapters before a final
closed-loop evaluation, and retain unsuccessful development runs alongside it.

## Training entry points

Run from the repository root with the MuJoCo workcell described in
[`cloth-coordination.md`](cloth-coordination.md), completed local datasets, and the pinned
foundation checkpoints already cached. Use new output paths; fitting commands
refuse to overwrite existing adapters. Run the large model jobs sequentially on
the development machine's shared GPU.

The command below repeats the recorded **final Smol v3 stage**, given its existing
v2 warm start and source recordings. It is not a from-scratch reproduction of
the preceding adaptation stages. The warm-start file
`smol-left-gradient-refit-h1.pt` has SHA-256
`172afb39ec8697804d7c38cc21f81365397981ce9a3bae2bc29d9dcc222cf1b6`.
Residual caches are bound to source hashes, normalization, the foundation pin and
parameters before the cache boundary. New head/final-MLP training can reuse them
only when those upstream parameters match.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m relay_lab.cloth.smol_native_ffn \
  --adapter .relay-lab/vla-adaptation/models/smol-left-gradient-refit-h1.pt \
  --train-episode .relay-lab/vla-adaptation/data/seed-10004-settled \
  --train-episode .relay-lab/vla-adaptation/data/seed-10005 \
  --train-episode .relay-lab/vla-adaptation/data/seed-10007 \
  --train-episode .relay-lab/vla-adaptation/data/dagger-smol-dev0 \
  --train-episode .relay-lab/vla-adaptation/data/recovery-10009 \
  --train-episode .relay-lab/vla-adaptation/data/recovery-10010 \
  --validation-episode .relay-lab/vla-adaptation/data/seed-10008 \
  --cache-root .relay-lab/vla-adaptation/smol-final-cache \
  --output .relay-lab/vla-adaptation/models/smol-left-final-v3-reproduction.pt \
  --samples-per-phase 10 --steps 3000 --batch-size 64 \
  --learning-rate 0.00003 --device mps
```

For X-VLA, these existing CLI entry points cache real Florence outputs, cache the
original final-transformer input, and fit that original block. The paths below
show a separate experiment with one training episode and one validation episode;
they do not describe a released successful controller. Add repeated `--cache`
arguments for additional training sources. Do not use a validation source to
create a training cache argument.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m relay_lab.cloth.xvla_training cache \
  --episode .relay-lab/vla-adaptation/data/seed-10007 \
  --output .relay-lab/vla-adaptation/xvla-docs-cache/train --stride 8

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m relay_lab.cloth.xvla_training cache \
  --episode .relay-lab/vla-adaptation/data/seed-10008 \
  --output .relay-lab/vla-adaptation/xvla-docs-cache/validation --stride 8

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m relay_lab.cloth.xvla_final_block cache \
  --florence-cache .relay-lab/vla-adaptation/xvla-docs-cache/train \
  --output .relay-lab/vla-adaptation/xvla-docs-prefix/train

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m relay_lab.cloth.xvla_final_block cache \
  --florence-cache .relay-lab/vla-adaptation/xvla-docs-cache/validation \
  --output .relay-lab/vla-adaptation/xvla-docs-prefix/validation

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m relay_lab.cloth.xvla_final_block train \
  --cache .relay-lab/vla-adaptation/xvla-docs-prefix/train \
  --validation-cache .relay-lab/vla-adaptation/xvla-docs-prefix/validation \
  --equivalence-cache .relay-lab/vla-adaptation/xvla-docs-cache/train \
  --equivalence-episode .relay-lab/vla-adaptation/data/seed-10007 \
  --output .relay-lab/vla-adaptation/xvla-docs-adapter \
  --steps 5000 --batch-size 4 --learning-rate 0.0001
```

The pointwise X-VLA experiment uses the same paired native-prefix and Florence
caches. It requires a strict adapter package carrying the original final-block
contract, such as the output above. This command is an experimental training
entry point, not a claim that the resulting policy completes a fold:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m relay_lab.cloth.xvla_pointwise \
  --adapter .relay-lab/vla-adaptation/xvla-docs-adapter \
  --cache .relay-lab/vla-adaptation/xvla-docs-prefix/train \
  --florence-cache .relay-lab/vla-adaptation/xvla-docs-cache/train \
  --validation-cache .relay-lab/vla-adaptation/xvla-docs-prefix/validation \
  --validation-florence-cache .relay-lab/vla-adaptation/xvla-docs-cache/validation \
  --equivalence-cache .relay-lab/vla-adaptation/xvla-docs-cache/train \
  --equivalence-episode .relay-lab/vla-adaptation/data/seed-10007 \
  --output .relay-lab/vla-adaptation/xvla-docs-pointwise-adapter \
  --steps 3000 --learning-rate 0.0001
```

The smaller native-input adaptation entry points remain available through
`python -m relay_lab.cloth.smol_native_training --help` and
`python -m relay_lab.cloth.xvla_training train --help`. Cached numerical
equivalence, held-out prediction reporting and a saved adapter are prerequisites
for a closed-loop experiment, not substitutes for one.

## Run a local development episode

With the pinned SmolVLA base model cached and both fitted arm adapters available,
on the Apple Silicon
development host (18 GB unified memory):

```sh
.venv/bin/python -m relay_lab.cloth.native_runtime \
  --smol-adapter checkpoints/cloth-native/left.pt \
  --right-smol-adapter checkpoints/cloth-native/right.pt \
  --output /absolute/path/to/new-rollout \
  --device mps --seed 10007 --horizon 1 --repeat 4 \
  --planner llm --coordination spacetime
```

The output directory must not already exist. No model or teacher fallback is
used on missing artifacts, invalid predictions or runtime errors. Cancellation
is checked before each arm's model loading and inference and before physics
advances, including after model and Qwen calls.
Completed phases may require up to 100 additional stationary checks for actual
cloth settling; the success thresholds and total episode budget do not change.

The right SmolVLA file must be a fitted
right-arm adapter; relabelling a left-arm artifact or passing an untrained
initializer is rejected. For the experimental heterogeneous pair, replace
`--right-smol-adapter` with `--xvla-adapter /absolute/path/to/xvla-adapter-directory`
and cache its separately pinned foundation model. The two right-controller options
are mutually exclusive; no successful X-VLA cloth controller is packaged.

## Recorded development result

The compact, source-hash-bound record is
[`demo-result.json`](../checkpoints/cloth-native/demo-result.json).
Dashboard task `goal-b91c739583c58561` records the original video, per-arm cameras,
native command audit and independent physics replay.

- 231 left and 241 right native model inferences; 1,814 native movement commands.
  Another 514 commands were explicit coordination holds, separately identified.
- 32 Qwen reviews and 291 durable Spacetime frame commits. The simulated channel
  published 582 packets, delivered 343 and dropped 84; out-of-order delivery
  remained enabled.
- Final hem alignment: 69.2 mm left and 48.3 mm right, both below the unchanged
  95 mm threshold. Both grippers and the vacuum fixture were released.
- Final footprint: 0.126 m², 98.8% within the target; 36 stable ticks. Fresh replay
  covered all 1,164 steps with zero robot-state error and identical cloth geometry.
- The live rollout took 25.1 minutes on the local M3 Pro, representing 46.56 seconds
  of simulated time. This is not real-time robot control.

The right arm released its sleeve early, but the eventual fold passed the existing
geometry checks. Retain that limitation: one successful final outcome does not
establish reliable grasp retention or generalization. Complete raw run artifacts
remain in the local dashboard workspace; the bundled JSON is a compact evidence
record, not a replacement for those recordings.

## Evidence contracts

Every physics tick records the chosen native action, its raw chunk, inference
identity and age, local input image digest, instruction, delivered messages,
coordination holds and measured outcomes. The exact adapter snapshots accompany
the run. `python -m relay_lab.cloth.native_audit RUN_DIRECTORY` checks those
records against native outputs and the pinned artifacts.

This audit verifies recorded command consistency; it does not independently
rerun the neural models. `scripts/replay_cloth_actions.py` separately re-executes
the recorded commands in a fresh MuJoCo world and compares the final vertices
and measurements. A dashboard success requires the measured folding conditions
and verified action replay, rather than merely reaching the final phase.

The scene uses a single-layer T-shirt-shaped flex sheet, idealized point clamps
and an explicit vacuum fixture released before scoring. Results describe this
simulation, not real garments or hardware.
