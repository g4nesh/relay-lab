# Two-arm cloth coordination

This workcell is an original MuJoCo experiment inspired by the coordination
question in [DuoMind](https://arxiv.org/html/2610.02161v1). It does not reproduce
RoboPoly, reuse its demonstrations, or claim to reproduce the paper's results.
Its task is to fold the sleeves of a T-shirt silhouette inward, then bring the
hem toward the shoulders using two independent robot controllers.

## Physics and appearance

The garment is a single-layer triangular flex mesh with gravity, bending,
inextensibility constraints, self-contact and table/obstacle contact. Its 271
vertices and 460 triangles evolve through MuJoCo dynamics. No rollout code
repositions cloth vertices to manufacture a fold. The two original articulated
arms have six revolute joints each, inverse kinematics, position actuators and
bounded end-effector target motion. Materials, cotton texture, seam markings,
worktable, sensors and room geometry are original procedural assets.

This remains a simulation approximation:

- The T-shaped sheet is not a sewn front/back garment, and fabric constants have
  not been calibrated against physical measurements.
- Marked grasp tabs attach through nearby, distance-gated point constraints.
  This does not validate frictional finger pinching or cloth-layer selection.
- Cloth collides with itself, the table and the moving obstruction; arm shell
  contacts are excluded from the idealized grasp model.
- An explicit vacuum-table fixture holds the upper garment during folding. Its
  physical constraints and visible pads are recorded. It must be released
  before task success is possible.
- MuJoCo raster rendering is a view of the actual simulation. It is not a
  photorealism or real-world transfer claim.

MuJoCo 3.17 or later is required for this flex/bending configuration. The project
lockfile retains the tested version. See `world.snapshot().physics_config` for
the physical assumptions recorded with each experiment.

## Observation boundaries

Each robot receives its own narrow overhead RGB image, own end-effector/grasp
state, and visible local sleeve/hem/shoulder coordinates. The coordinate sensor
is ideal MuJoCo metrology restricted to its workspace half with a small central
overlap; it is not a learned vision tracker or a raycast occlusion model. A
landmark that crosses outside the local workspace is omitted, with an explicit
visibility mask. A moving obstruction is reported only to the arm on its side.

Neither policy receives the world object, the other arm's live observation,
the spectator camera, all cloth vertices, or a success label. The local phase
controller sees only its own observation and packets actually delivered through
the communication layer. Spectator state and evaluation geometry are logged
separately for review. These are trusted-worker application boundaries, not
security isolation against arbitrary Python or database-owner access.

The obstacle enters and leaves on a predeclared simulation-time schedule, which
is independent of the model's actions. An arm can act on a peer's hazard only
after the corresponding message arrives. Shared hem lifting and carrying also
use received peer gripper position to limit progress differences. The intrusions
are physically collidable and trigger workspace safety interlocks; they are not
guaranteed to intersect every episode's particular cloth trajectory. Measured
attachment identity distinguishes a sleeve grasp from a hem grasp.

## Controllers and Qwen

The left controller uses convolutional RGB features; the right uses image-patch
attention. They have independent weights and checkpoints, with learned
phase-conditioned affine experts for precise small-data control. Their original
training and held-out imitation accuracy are documented in
[the checkpoint directory](../checkpoints/cloth-policies/README.md).

These are compact **state-assisted task controllers**, not general-purpose
foundation VLAs. They support the workcell's explicit phase vocabulary. The
local landmark measurements explain much of their precision; the experiment
does not establish that RGB is necessary. High-level phase scaffolding, IK,
motion limits, and safety holds remain separate. Every hold records both the raw
neural output and the applied interlock. No reference-controller fallback runs
inside a measured learned rollout.

The `rules` condition uses identical deterministic phase guards for all modes.
The `llm` condition adds actual local Qwen review at eligible phase boundaries.
Each Qwen request contains one arm's local status and already-delivered peer
reports. Schema failures stop the run; they do not switch to rules. Phase
completion is checked from actual local motion and grasp state. The final task
outcome is always a physical measurement, never an LLM declaration.

Released model adapters and bounded transfer probes are separate from this
controlled experiment. A pretrained checkpoint's action convention and camera
distribution do not automatically match this workcell. A short probe establishes
that inference and the action interface execute, not that the full task works.

## Communication experiment

All modes observe the same 23-feature local sensor schema. The feature codec is
standardized PCA trained only on original training observations; it produces an
eight-dimensional learned numerical representation. It is **not a foundation
VLA hidden state, nor an image encoder**.

The frozen codec uses 632 original training observations: 144 known no-grasp
calibration states and 488 measured states from a full physical training fold
(seed 10004). Sixteen legacy positive-grasp records without attachment identity
were excluded. Attachment identity is never inferred from an intended phase.
See [codec provenance](../checkpoints/cloth-policies/COMMUNICATION.md).

| Mode | Content delivered to the other arm |
| --- | --- |
| `embeddings` | Learned bottleneck, decoded at the receiver |
| `structured_text` | Every measured feature in full-precision canonical JSON |
| `hybrid` | Bottleneck plus exact obstacle, grasp and visibility fields |
| `none` | Optional diagnostic with no private packets |

Structured text is deliberately a strong baseline. Hybrid geometry still comes
from the learned representation and is used in shared-motion synchronization.
There is no assumption that either learned format should outperform exact state.
The representation's held-out reconstruction error must be considered alongside
whole-task outcomes.

Packets carry run/robot identity, original observation time, planned delivery
time, expiry, source hash and codec checkpoint hash. A seeded hash of sender and
observation tick schedules delays and drops identically across modes. Newer
observations cannot be overwritten by a reordered older packet. Expired reports
cannot enter the next action. Simulation continues during the synthetic delay;
the message queue is released by simulation time rather than wall-clock sleeps.

Spacetime stores private immutable channel configuration, frame batches and
packets. The worker requires exact authoritative readback before a packet becomes
eligible for delivery. Retrieval uses indexed identities, not nearest-neighbor
vector search. Database and inference calls remain synchronous wall-clock
barriers in this prototype; that limitation is distinct from simulated network
delay. There is no claim of real-time distributed robot-process synchronization.

## Scoring and comparisons

Success requires physical hem alignment, sleeves contained within the folded
shape, sufficient projected cloth area, target footprint coverage, bounded
width/length/height, released grasps and fixture, and settled low cloth velocity.
Corner coincidence in a collapsed bundle does not count. Finishing the phase
list can produce a `rejected` result; time limits and runtime errors remain
visible failures. A peak substep cloth speed above the predeclared 5 m/s validity
bound also invalidates the episode. This numerical sanity check is not material
calibration. Ordinary phases receive at least 45 active ticks, hem placement
receives 90, and final settling receives 160. Each phase has a 180-active-tick
timeout and the episode is bounded at 1,800 total ticks. Holds pause the active
phase clock; all durations and bounds are identical across communication modes.

`cloth-compare` writes its protocol before the first condition, rotates mode
order, and retains all attempts. It reports success, alignment, bytes, simulated
ticks, packet age, delays, drops and measured encode/decode/database timings.
Initial cloth/camera hashes and checkpoint hashes are checked within each paired
seed/scenario. Wall runtime includes rendering and local contention; it is not
an isolated network-latency ranking.

Physical rollout seeds 0–99 are reserved for development, 100–9999 for evaluation,
and 10000+ for training. The comparison command rejects seeds outside the
evaluation partition. Each run snapshots its exact actor and codec files in
`models/`, so later training cannot change the evidence being audited.

The `clear` scenario checks controller competence. `obstacles` adds changing
physical obstructions with one-tick communication. `delayed` preserves those
obstructions and adds 2–18 tick delay and 15% drops. The optional `blackout`
condition reaches 32 ticks and 30% drops. At 25 Hz, 18 ticks is 720 ms. Every
mode uses the same action budget and 24-tick report lifetime.

Controlled fixed-guard results and actual Qwen-mediated runs must be reported
separately. If every representation fails because the motor policy cannot fold,
the comparison cannot establish a successful communication method. Even a clean
small-sample advantage is specific to these sensors, models, fixtures and tasks.

The vacuum fixture is released by a shared workcell interlock after both local
controllers enter their final settling phase. That control-plane signal is the
same in every mode and is separate from the private observation channel.

## Reproduce

For a fresh checkout, first follow the repository README's **Run locally**
instructions to install/start SpacetimeDB and prepare `.venv-llm`. The recorded
replay itself needs neither the database nor model inference.

```bash
uv sync --extra learned
python3 scripts/spacetime.py publish
uv run --extra learned relay-lab cloth-run --communication structured_text \
  --scenario delayed --coordination spacetime --planner llm --output runs/cloth-task
uv run --extra learned relay-lab cloth-compare --seeds 101 102 \
  --scenarios clear delayed --coordination spacetime --output runs/cloth-comparison
uv run --extra learned relay-lab cloth --directory runs/cloth-comparison --port 8769
```

The read-only replay shows actual recorded MuJoCo videos, each local camera,
the changing obstacle, phase decisions, delivered evidence ages and measured
results. `events.jsonl.gz` retains every action and prediction;
`communication.json.gz` retains packet provenance and delivery events.

To independently inspect a run and regenerate the post-hoc comparison:

```bash
uv run python scripts/audit_cloth_run.py RUN --spacetime
uv run python scripts/summarize_cloth_experiment.py runs/cloth-comparison
```

The post-hoc analysis evaluates the frozen codec on the structured-text runs'
local observations, giving all representations one shared held-out input
distribution. It does not refit anything. Per-feature coordinate errors, grasp
classification, actual payload sizes, database timings and recorded hold reasons
help explain task outcomes. Comparison success rates retain every attempt;
pairing failures prevent a winner claim.

For independent physics re-execution, replay the saved applied actions:

```bash
uv run python scripts/replay_cloth_actions.py RUN runs/cloth-physics-replay
```

This creates a fresh MuJoCo world, checks both robots' positions, joint angles
and grasp identities at every tick, and compares the final cloth vertices
bitwise with the original. It also checks fixture release, final metrics and
the complete tick count. The optional 25 Hz video is a render of those
re-executed physical steps. This check does not re-run the neural policies,
Qwen or the communication channel. `--no-video` retains the physics check
without rendering.

For the actual released-model pair, see the [pretrained integration evidence](../examples/cloth-coordination/pretrained/README.md)
and [remaining calibration and adaptation work](../checkpoints/cloth-policies/foundation-adaptation.md).
Those trials are deliberately separate from the successful task-controller
comparison.
