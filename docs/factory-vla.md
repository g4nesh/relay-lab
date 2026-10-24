# Factory VLA and learned peer communication

The factory has five independent robot policy contexts sharing one compact,
original task-specific neural network. Qwen remains the event-driven macro
planner. SpacetimeDB authorizes every physical tick and commits measured outcomes.
Neither is replaced by the action policy or by embedding similarity.

## Run

Use the Python, Qwen and Spacetime setup from the root README, including
`uv sync --python 3.13 --extra learned`. Start Spacetime and publish the additive
schema before running:

```bash
uv run --extra learned relay-lab factory-run \
  --orders 3 --seed 7 --policy factory-vla --embeddings on \
  --output runs/factory-vla-demo
uv run --extra learned relay-lab factory \
  --replay runs/factory-vla-demo/replay.json
```

Open <http://127.0.0.1:8767/?replay=1> for the recorded full task or
<http://127.0.0.1:8767> to start another live mission. Replay uses only committed
frames and includes stage transitions and terminal outcomes. The full
`events.jsonl` remains the audit trail; `execution_observed` rows explicitly mark
physical execution before database confirmation.

## What is learned

The inference boundary contains two RGB inspection views, a constrained text
instruction, local proprioception and optional compatible peer embeddings.
Instructions select navigation, pick, place, process or wait. The RGB views are
real MuJoCo renders of a simplified sensor scene with physical colored fiducials
on cargo, docks and robot decks. They are not the browser's PBR renderer.

The network learns per-pixel tag classification, two-view localization,
camera calibration, language routing and a motor MLP. Its outputs are Cartesian
increments, yaw increments and gripper logits. A normalized learned encoder
compresses local tool/deck position, visually estimated goal, intent and body
type into 32 values. A learned decoder reconstructs those quantities.

The policy does not receive simulator target positions, waypoint arrays,
the analytic training teacher or a target-tracking fallback. It retains a
visually inferred stationary goal during each command. Provenance records the
camera tick separately from the current action tick. The simulator still owns
collision/workspace/speed limits, physical grasp predicates and outcome checks.
Only predicted gripper outputs can trigger a grasp or release.

This is a calibrated imitation-learning VLA, not SmolVLA, a fine-tuned foundation
model, open-vocabulary understanding or a sim-to-real result. The fixed camera,
fiducial palette, layout and language vocabulary are deliberate constraints.
The humanoid does not walk or balance; articulated links use kinematic IK and
grasp attachments are idealized after measured proximity checks.

## How embeddings affect actions

Every five simulation ticks and at terminal transitions, all five policies encode
their fresh local state. The worker commits those vectors in the same transaction
as observed outcomes, then reads them back from Spacetime before peer delivery.
The latest eligible record from each peer is selected before cosine ranking;
an older, more similar pose cannot displace a newer pose.

When a manipulator places onto a carrier, the matching carrier's decoded learned
deck pose enters a trained fusion layer with the visual goal. That fused goal
feeds the motor MLP. Thus vectors can change real motor outputs. Other peer
vectors remain available for inspection and decoded planner context; the current
motor model does not consume arbitrary peer features for every skill.

All five policies use a common encoder. Unrelated pretrained models cannot
exchange these vectors directly: they would need trained adapters into an agreed
space and a new compatibility version.

Qwen receives the latent decoder's bounded text predictions, alongside verified
outcomes. Its prompt distinguishes predictions from evidence. Decoder confidence
is an aggregate held-out reconstruction statistic, not a calibrated per-message
probability or a claim that an action succeeded.

## Transport and retrieval guarantees

Each record carries run, robot, command and job IDs, encoder checkpoint SHA-256,
space/schema version, source tick, expiry, dimension, normalized vector and
decoder provenance. The backend rejects incompatible spaces/checkpoints,
nonfinite or nonunit vectors, changed IDs, wrong command scope and stale ticks.
The space is immutable within a run. The checkpoint also binds the sensor
geometry, tag palette and mounting offsets; incompatible sensors fail before
motor execution. Expired records are pruned atomically and
storage is bounded to 16 records per robot. Runtime TTL is 15 ticks; records are
never relabelled as current after a failed refresh.

Spacetime uses indexed run/robot reads. One batched database refresh supplies a
bounded cache shared by peer queries, followed by exact client-side cosine
ranking over at most four latest peers. This is not native approximate-nearest-
neighbor search: Spacetime's [documented indexes](https://spacetimedb.com/docs/tables/indexes/)
are B-tree/direct indexes. Database-refresh and local-ranking latency are measured
separately, along with bytes and cache age. A database error stops execution;
there is no unconfirmed local-vector fallback.

Use `--embeddings off` to disable vector publication and consumption while keeping
the same learned weights, physical task and ordinary text protocol. For an
isolated communication comparison, fix seed, workload, checkpoint, pose jitter
and planner. Qwen runs can choose different macro actions, so a paired rules-
planner run is useful for isolating the vector input. Improved task performance
must be measured; a working communication channel alone does not establish it.

## Reproduce training

All data is original. No RoboPoly demonstrations, DuoMind checkpoint or downloaded
training corpus is used. Visual training scenes randomize tags across the five
camera viewpoints. Analytic teachers generate state/action pairs only inside the
training module; the runtime does not import that module.

```bash
uv run --extra learned python -m relay_lab.factory.train_vla collect --scenes 800
uv run --extra learned python -m relay_lab.factory.train_vla train \
  --pixel-steps 3000 --control-steps 24000 --codec-steps 5000
uv run --extra learned python -m relay_lab.factory.train_vla refine --steps 6000 --seed 919
uv run --extra learned python -m relay_lab.factory.train_vla evaluate
```

The checkpoint metadata records data and weight hashes, architecture and training
settings, including motor refinement and a later physical-tag calibration. The
commands above train a fresh compatible model from the current sensor data;
they do not promise a byte-identical historical checkpoint. A mount-only
recalibration is available as `recalibrate-offsets` and preserves the motor and
codec weights while fitting tag offsets from paired simulator labels. `evaluation.json` records held-out vision, motor imitation and latent
reconstruction error. Unit tests verify missing-image failure, language grounding,
weight dependence, embedding normalization and a causal peer-vector effect on
motor predictions. These checks complement full physical task runs; supervised
error alone is not proof of task completion.

The original tabletop policy, dataset and evaluation remain separate.

## Verified task evidence

The [included run](../examples/factory-vla/README.md) completes three workpieces
with Qwen, SpacetimeDB and all five learned motor contexts: 4,390 neural actions,
338 peer-fusion actions and 3,257 independently database-checked commits. A
separate six-workpiece rules/Memory run completes in 6,505 ticks. The paired
three-workpiece embedding comparison completes in both modes, with sharing on
taking 11 additional ticks; no performance benefit is established.

Actual run vector refresh latency was 5.46 ms p50 / 7.63 ms p95; bounded client
selection was 0.044 / 0.056 ms. These measurements include a local machine with
other work active and should not be treated as an isolated throughput benchmark.

Recheck a retained run's recorded evidence without executing models:

```bash
uv run --extra learned python scripts/audit_factory_vla_run.py \
  runs/factory-vla-demo --spacetime --require-success
```

Omit `--spacetime` for a local trace-only check. The audit reports failures and
unconfirmed observations honestly; it cannot independently attest model execution.
