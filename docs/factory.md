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
uv run relay-lab factory
```

Open <http://127.0.0.1:8767>. Choose one, three, or six workpieces, then start a
mission. The default planner is the pinned, locally running Qwen3-1.7B MLX model.
Its first call may take longer while weights load. The browser displays live
state; orbit, zoom, select robots, inspect messages, or change camera views.
Pause takes effect at the next committed simulation boundary. Reset waits for an
in-flight model call or database operation to finish before discarding the world.
Simulation speed changes pacing, not dynamics or decision budgets.

For a headless run with complete state, decision, and message logs:

```bash
uv run relay-lab factory-run --orders 3 --seed 17 --output runs/factory-check
# Explicit reference modes for testing without model or database dependencies:
uv run relay-lab factory-run --planner rules --coordination memory --orders 6
```

The seed is recorded for reproducibility; this pilot uses fixed precision fixtures
and does not yet randomize the factory layout or object poses.

The bundled Three.js renderer needs no CDN. After editing the UI source, rebuild:

```bash
npm ci --prefix src/relay_lab/factory/web
npm run build --prefix src/relay_lab/factory/web
```

## Production workflow

| Robot | Job | Control adapter |
|---|---|---|
| H01 humanoid loader | Receive material and load the inbound carrier | Fixed stance, upper-body analytic IK |
| M01 inbound carrier | Travel between load dock and assembly input | Bounded acceleration, turning and swept collision checks |
| A01 articulated arm | Unload, process on a fixture, load outbound carrier | Analytic arm IK and Cartesian trajectories |
| M02 outbound carrier | Travel between assembly output and dispatch | Same mobile controller, independent state and commands |
| G01 overhead grappler | Unload and palletize processed workpieces | Bounded bridge, trolley and hoist trajectories |

Each workpiece progresses through:

`queued → loaded → at_cell → in_fixture → processed → outbound_loaded → at_dispatch → completed`

The mediator sees measured job state, legal affordances, prioritized peer messages,
and previous outcomes. It chooses one or more compatible macro skills and explains
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
3. Commit the observed state and every command's running/completed/failed outcome.
   Advance the revision and release terminal leases atomically.

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
limits. Mobile robots turn in place and check swept collisions. Grasping requires
measured proximity and an available destination. Released payloads have MuJoCo
rigid-body gravity and contacts; a free payload must settle in the correct target
before the stage can complete. Cargo custody follows measured grasp/release
transitions, not model text. Processing is a timed simulated operation.

**These new embodiments use model-based controllers, not newly trained VLAs.**
The existing small learned policy is calibrated to the original tabletop cell;
reusing it unchanged on humanoids or mobile robots would be misleading. This
extension has no learned balance, humanoid walking, force-controlled articulated
robot dynamics, production-grade navigation, or real manufacturing process.
Grasp attachments are idealized after proximity checks, and robot links are
kinematic. The PBR view uses original geometry and realistic material cues; it is
a visualization of this simplified simulation, not a photorealistic digital twin.

A future VLA adapter should retain the same command/outcome contract but replace
an embodiment's inner controller with task-adapted, observation-conditioned
motor outputs. Train and validate each new embodiment before making learned
control or sim-to-real claims.

## Relationship to the paper

[DuoMind](https://arxiv.org/html/2610.02161v1) motivates sharing local intentions,
expected subgoals, beliefs and uncertainty between robot controllers. Relay Works
uses an original factory layout, robot models and workflow, and adds explicit
resource leases, custody protocols and transactional execution. It does not use
RoboPoly, its demonstrations, DuoMind weights, or copied simulator assets. The
paper's reported results are not results of this project. RoboPoly evaluation can
be added later as a separate adapter without redefining its success criteria.
