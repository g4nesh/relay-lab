# Partial observations and useful robot communication

`dispatch-routing` adds a decision that cannot be recovered from the gantry's
local image. A receiving inspection station displays an independently sampled
A/B indicator for each part. Only the humanoid loader can read its 64×64 MuJoCo
camera. Each job owns a pair of pallet slots; the gantry must choose the indicated
slot after loading, transport, processing and a second transport.

The original five robot motor contexts and frozen Factory VLA checkpoint still
execute the complete physical workflow. The private indicator uses a separate,
original 20,907-parameter visual communication head. Its learned normalized 32D
representation carries the observation to the gantry. The receiver decodes a
slot instruction, which the existing RGB/language motor actor executes. This is
a task-specific communication adapter, **not** a foundation-model hidden-state
protocol or end-to-end training of peer latents into the motor network.

## Information available to each component

| Component | Available information |
| --- | --- |
| Humanoid | Local receiving scene, own proprioception, private inspection RGB |
| Assembly arm | Nearby cargo/carriers and assembly workspace |
| Drive units | Nearby dynamic markers, own state, shared static dock map |
| Gantry | Local dispatch scene, own state, received job-scoped packet if enabled |
| Qwen | Public job stage/custody, legal skill candidates, verified public feedback |
| Evaluator | Hidden indicator truth and measured final positions |

Dynamic markers outside each robot's workspace are removed from its rendered
sensor scene. Fixed navigation beacons remain public. The private indicator is a
separate inspection fixture, not a globally visible object color. Its random
stream is independent of physical pose randomization and the labels are sampled
independently across jobs. Object IDs, candidate descriptions and job ordering
do not encode the answer. Changing the private label leaves public state,
candidate actions and motor-camera marker positions unchanged.

Qwen chooses the generic `transfer to pallet` skill. Only the gantry's local
route adapter reads the communication packet and adds the selected slot before
command authorization. The evaluator never chooses a destination for the robot.
Wrong routes do not stop later jobs: all physical work can finish while the run
ends `rejected`. Physical completion and correct routing are separate metrics.

These are application-level observation boundaries within a trusted simulation
worker, not process isolation or protection from a malicious agent with Python
or database-owner access.

## Fair channel comparison

All three modes use the same local RGB observations, motor checkpoint, visual
encoder, task seeds, stage rules, safety limits and execution budget.

| Mode | What crosses the private-information channel |
| --- | --- |
| `embeddings` | 32 learned values; decode the route at the gantry |
| `text` | The same encoder/decoder prediction as exact `{"route":"A"}` JSON; parse at the gantry |
| `none` | Nothing; the gantry uses the declared, seed-independent A prior |

The text baseline is intentionally strong. It does not pay for an extra LLM
generation or discard useful information. Both communication modes share the
same cadence, recipients, original sensor evidence and delivery eligibility.
No-communication still retains public control-plane traffic: reservations,
handoff ACKs, custody and stage updates. It disables the private semantic
channel. The old pose-embedding channel is disabled for **all** routing modes.

For this binary decision, text should be compact and competitive. A useful
embedding channel means it improves over missing information; it does not imply
that vectors outperform an exact symbolic report. The old fully observed pose
fusion experiment remains a separate baseline with no demonstrated benefit.

## Spacetime channel

The schema adds an immutable per-run semantic configuration and one latest
packet per unfinished job, bounded to three records. The same transaction
commits motor outcomes and semantic packets. Delivery uses confirmed readback;
there is no sender-memory fallback if the database fails.

Each packet contains its run, job, sender, recipient, versioned latent space,
checkpoint hash, source command and RGB hash. `observation_tick` preserves the
original inspection time. `emitted_tick` identifies a retransmission and expires
after 15 ticks. A sender re-emits its immutable belief every five ticks until
that job completes. A fresh packet is **not** a claim that the old cue was seen
again. This is valid because the cue is immutable for that job.

These lifetimes use simulation ticks. The simulation waits at its authorization
and commit barriers; it does not keep moving during a slow model or network
call. This experiment measures information sharing under partial observation,
not robustness to real-time asynchronous motion or changing private cues.

The first packet must refer to that job's authorized humanoid transfer from
receiving. Retransmissions cannot change the evidence or payload. Completed
jobs delete retained rows; expired packets cannot be delivered. Recipient/job
checks prevent one job's cue from selecting another job's slot. Text packets
cannot smuggle vectors, and vector packets cannot smuggle a decoded label.

Retrieval is an indexed run/job read and an exact bounded client lookup. A
known job should use a key, not nearest-neighbor similarity over unrelated
jobs. This is **not** native Spacetime vector search. Recorded metrics separate
database refresh from local lookup time; the dashboard separates transport age
from original evidence age. JSON float vectors also carry substantially more
bytes than a binary fact.

## Run and reproduce

Use the root README's local Qwen, Python and Spacetime setup, then publish the
additive schema. Existing tables and runs are preserved.

```bash
python3 scripts/spacetime.py publish

uv run --extra learned relay-lab factory-run \
  --task dispatch-routing --communication embeddings \
  --seed 101 --orders 3 --max-ticks 5000 \
  --planner llm --coordination spacetime --output runs/private-dispatch

uv run --extra learned relay-lab factory \
  --replay runs/private-dispatch/replay.json
```

Use `--communication text` or `none` for the other channels. The dashboard also
exposes task and channel selectors. This task supports one to three jobs because
each job reserves two of the six existing slots.

The primary experiment uses 12 held-out task seeds, three jobs each and a fixed
5,000-tick budget. Its deterministic public reference scheduler and in-memory
equivalent of the Spacetime transaction barrier isolate representation changes
from LLM scheduling variability. All motor actions still come from the learned
VLA. A separate matched three-seed integration suite uses actual Qwen and live
Spacetime. The report keeps these suites separate.

```bash
uv run --extra learned python scripts/benchmark_factory_communication.py \
  --output runs/communication-heldout

uv run --extra learned python scripts/benchmark_factory_communication.py \
  --coordination spacetime --planner llm --seeds 101 102 103 \
  --output runs/communication-integration
```

The protocol is saved before execution. Every mode runs every specified seed;
mode order rotates across seeds. Errors remain in the denominator. Paired
checks compare initial physical states, private-image hashes and checkpoints.
All full tick traces and replays are retained losslessly as gzip files. The
summary includes a terminal-only oracle for independently scoring the chosen
slots. No oracle is an agent input.

See [recorded results](../examples/factory-communication/README.md) for measured
outcomes. Confidence intervals use complete tasks, not correlated workpieces.
Wall times include rendering, logging and other local workloads; they are not
isolated model or network latency measurements.

The trained visual head, provenance and held-out image fixtures are in
`checkpoints/factory-routing/`. Training generates original MuJoCo imagery; it
uses no RoboPoly demonstrations. See that directory's README for reproduction.
