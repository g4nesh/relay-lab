# Relay Works: the factory extension

A 24 × 14 m pilot factory with five robots, four embodiment types, a production
queue, shared traffic resources, and a real local language-model mediator.
The original two-robot learned-VLA experiment remains available with `relay-lab run`.

## Start

Complete the Python, local Qwen, and SpacetimeDB setup in the repository README.
Keep `python3 scripts/spacetime.py start` running. Publish the additive factory
schema once, then start the dashboard:

```bash
python3 scripts/spacetime.py publish
uv run --extra learned relay-lab factory
```

Open <http://127.0.0.1:8767>. Choose one, three, or six workpieces, then start a
mission. The default planner is the pinned, locally running Qwen3-1.7B MLX model.
Its first call may take longer while weights load. The browser displays live
state; orbit, zoom, select robots, inspect messages, or change camera views.
Pause takes effect at the next committed simulation boundary. Reset waits for an
in-flight model call or database operation to finish before discarding the world.
Simulation speed changes pacing, not dynamics or decision budgets. The default
run budget is 10,000 ticks to accommodate six learned-policy workpieces;
`factory-run --max-ticks` overrides it. Individual skill timeouts remain bounded.

For a headless run with complete state, decision, and message logs:

```bash
uv run --extra learned relay-lab factory-run --orders 3 --seed 17 --output runs/factory-check
# Explicit reference modes for testing without model or database dependencies:
uv run relay-lab factory-run --policy model-based --planner rules --coordination memory --orders 6
```

The learned mode randomizes initial cargo positions by up to 2 cm per axis using
the recorded seed. `--pose-jitter 0` disables that perturbation. Factory layout and
precision fixtures remain fixed. `--embeddings off` runs the same learned policy
without peer-vector exchange; it does not disable ordinary coordination messages.

Each finished or stopped run exports `replay.json` from confirmed commits. View it
with `uv run --extra learned relay-lab factory --replay runs/factory-check/replay.json`,
then open <http://127.0.0.1:8767/?replay=1>. Recorded playback is explicitly labelled
and supports seeking, pausing and speed changes.

The bundled Three.js renderer needs no CDN. After editing the UI source, rebuild:

```bash
npm ci --prefix src/relay_lab/factory/web
npm run build --prefix src/relay_lab/factory/web
```

## Production workflow

| Robot | Job | Learned motor outputs |
|---|---|---|
| H01 humanoid loader | Receive material and load the inbound carrier | End-effector XYZ and gripper, fixed stance |
| M01 inbound carrier | Travel between load dock and assembly input | Holonomic XY and yaw, planar base |
| A01 articulated arm | Unload, process on a fixture, load outbound carrier | End-effector XYZ and gripper |
| M02 outbound carrier | Travel between assembly output and dispatch | Independent context, same learned mobile policy |
| G01 overhead grappler | Unload and palletize processed workpieces | Bridge, trolley, hoist and gripper |

Each workpiece progresses through:

`queued → loaded → at_cell → in_fixture → processed → outbound_loaded → at_dispatch → completed`

The mediator sees measured job state, legal affordances, prioritized peer messages,
previous outcomes and decoded peer embeddings. Decodes are labelled model
predictions and cannot establish successful execution. It chooses one or more compatible macro skills and explains
the next subgoal. It is called when new legal work becomes available; an accepted
motion command remains unchanged until completion. A strict validator rejects
invented IDs or conflicting selections. One model repair is allowed; another
invalid response stops the run. There is no silent rules fallback.

## Coordination and synchronization

SpacetimeDB is the authoritative command and observation ledger. A trusted local
worker owns MuJoCo and uses this transaction sequence for **every tick**:

1. Prepare a batch against the current revision. Check capabilities, job stage,
   exclusive robot/station/cargo/aisle leases, and handoff acknowledgements.
2. Read back the authorized batch. Run exactly one simulation step.
3. Commit the observed state, every command's running/completed/failed outcome,
   and any fresh learned embeddings. Advance the revision, prune expired vectors
   and release terminal leases atomically.

A failed prepare performs no physics. The worker logs a clearly marked, unconfirmed
physical observation before asking the database to commit it. A failed observation
commit poisons the worker: it cannot take another physical step, and the dashboard
retains the last confirmed state while the execution evidence stays in the log. Exact database retries are
idempotent; retrying a database request never repeats a physical action. Changed
or terminal command IDs, stale revisions, omitted running commands, unauthorized
stage jumps, and conflicting resource owners are rejected.

Messages have a typed envelope: sender, recipient, sequence, causal parent,
priority, confidence, expiration, and structured payload. Intents carry expected
subgoals; outcomes carry measured evidence. Handoffs use a directed request and
reverse-direction acknowledgement scoped to the workpiece. The acknowledgement
is emitted by the trusted simulator worker only when the world reports the peer
stationary, docked, and ready. It is not an independent robot's cryptographic
attestation. New peer-custody transfers must reference that acknowledgement.
Queues are bounded, stale messages expire, and aging raises waiting-message
priority. Full messages remain in the audit trail.

The database checks evidence consistency and the process graph; MuJoCo and the
trusted worker establish the underlying physical observations. It cannot detect
a malicious worker fabricating both state and evidence. There is no automatic
worker takeover or simulator recovery after a crash. The shared aisle is
conservatively exclusive, so throughput remains limited compared with a real
multi-lane plant. Message priority aging does not guarantee fair resource scheduling;
resource admission remains planner-controlled. Audit tables currently retain all
records, so this is a pilot workload rather than an unbounded deployment. Abort
releases backend leases and marks the run failed; if the
database is unavailable, the local worker still stops and records an unconfirmed
abort for inspection.

## What is physically simulated

Robot base/end-effector poses evolve continuously with speed and acceleration
limits. Learned mobile robots have holonomic translation, bounded yaw and swept collision
checks. The reference controller uses turn-then-translate paths. Grasping requires
measured proximity and an available destination. Released payloads have MuJoCo
rigid-body gravity and contacts; a free payload must settle in the correct target
before the stage can complete. Cargo custody follows measured grasp/release
transitions, not model text. Processing is a timed simulated operation.

The default **Factory VLA** is a compact original imitation model with learned
RGB localization, instruction grounding, motor/gripper output and a shared 32D
latent codec. All five robot contexts use its weights. There is no motion teacher,
waypoint generator or simulator target-coordinate input on the learned inference
path. Simulator limits may clip an action or stop a collision; they do not steer
toward a goal. `--policy model-based` explicitly selects the previous reference.

See [Factory VLA and embedding exchange](factory-vla.md) for training, input
boundaries, reproducibility and validation evidence. The model uses calibrated
inspection cameras and colored fiducials. It is not a pretrained foundation VLA
or general visual reasoning system. Its original data and weights are independent
of the existing tabletop VLA and of DuoMind.

The factory still has no learned balance, humanoid walking, force-controlled
articulated robot dynamics, production-grade navigation, or real manufacturing
process. Grasp attachments are idealized after proximity checks; robot links and
IK are kinematic. The PBR browser view visualizes this simplified simulation.
The policy's actual MuJoCo inspection-camera input is available in Robot details;
these cameras use a simplified geometric sensor scene and physical tags, not the
browser's decorative materials. Neither view is a photorealistic digital twin.

## Relationship to the paper

[DuoMind](https://arxiv.org/html/2610.02161v1) motivates sharing local intentions,
expected subgoals, beliefs and uncertainty between robot controllers. Relay Works
uses an original factory layout, robot models and workflow, and adds explicit
resource leases, custody protocols and transactional execution. It does not use
RoboPoly, its demonstrations, DuoMind weights, or copied simulator assets. The
paper's reported results are not results of this project. RoboPoly evaluation can
be added later as a separate adapter without redefining its success criteria.
