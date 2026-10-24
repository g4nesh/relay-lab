# SpacetimeDB coordination

SpacetimeDB is the durable coordination backend for local CLI runs. A local
Qwen LLM assigns semantic subgoals and a deterministic judge turns observed
simulation effects into bounded text. MuJoCo and learned action policies run
locally. The database does not execute model inference or physics. Fetch has
been removed from the active runtime.

## Local setup

Run these commands from the repository root. Python 3.12+, Node.js 22+ and npm
are required. The provided installer is verified on Apple Silicon; other
platforms need the matching official CLI placed in the path printed by it.
The setup pins the server, CLI, and TypeScript SDK to **2.10.0**.
No hosted database, account, or paid API is required.

```bash
python3 scripts/spacetime.py install
python3 scripts/spacetime.py start
```

Keep that server running. In another terminal:

```bash
python3 scripts/spacetime.py publish
python3 scripts/spacetime.py status
uv sync --python 3.13 --extra learned

# Learned actions + local Qwen mediator + real database transactions.
uv run --extra learned relay-lab run --policy relay-vla --planner llm

# Explicit infrastructure reference without neural models.
uv run relay-lab run --policy scripted --planner rules --no-video
```

The server uses `127.0.0.1:3000` and database `relay-lab`. Database files and
server keys stay under `.relay-lab/spacetime/`. The owner JWT is stored in
`.relay-lab/spacetime-token` with mode 0600. These paths are ignored by Git.
Do not delete this directory to restart the server. Stop it with Ctrl-C and
run `start` again against the same directory.

The Python client supports `RELAY_SPACETIME_SERVER`, `RELAY_SPACETIME_DATABASE`,
and `RELAY_SPACETIME_TOKEN_FILE`; `RELAY_SPACETIME_TOKEN` is also supported for
managed process environments. Keep tokens out of command arguments and logs.
Non-loopback servers require HTTPS. Remote deployment is not configured here.

## Execution contract

Each rollout creates a unique task ID and process-specific worker ID. The
database records the seeded or explicitly requested packing order and two
assignments. A worker claims the task once. Another worker cannot take over it.

| Reducer | Required state | Result |
| --- | --- | --- |
| `create_task` | New task ID | Task and both robot assignments exist atomically |
| `claim_task` | Queued, unclaimed task | Immutable worker claim; task running |
| `plan_tick` | Current tick, ready phase | Both instructions, local observations and contexts saved |
| `prepare_tick` | Current tick, planned phase | Both model actions saved; phase executing |
| `commit_tick` | Current tick, executing phase | Both observed reports and outcome saved; next tick or terminal state |
| `abort_task` | Running task, matching worker | Terminal failure; history retained |

The worker takes both observations from the same pre-step scene. Before
planning it reads each robot's preceding committed feedback and checks that
its active interpreter agrees. The mediator receives both robots' verified execution events. It then commits and reads back both instructions,
runs both policies, commits and reads back both proposed actions, and executes
one simultaneous simulation step. Both resulting reports commit in one reducer
transaction. Only then are progress notifications emitted and another tick
allowed. A database failure stops execution; there is no memory fallback.

Each reducer is authenticated to the module's publishing identity. Tables are
private. The local simulator is a trusted worker using that identity; this is
not a multi-tenant API or independent database login for each robot. The model
input boundary for each action policy excludes the other robot's private
observation. The central mediator sees both structured observations and order. Camera
pixels stay local; JSON database rows retain structured observations and model
prediction metadata for audit.

Reducers reject a mismatched worker or tick, a skipped phase, conflicting
duplicate payloads, and reports that disagree with their authorized action or
before/after observations. Success is derived from observed tray slots matching
the assigned order, and must agree with the packer's observation. This validates
worker-reported evidence; it is not independent sensing of a malicious worker.

The client requests `confirmed=true` for durable reducer acknowledgements and
reads back authoritative rows. A lost response can only trigger an identical
idempotent database request. It never triggers a second physical step. The
starter synchronizes through HTTP reducers and SQL reads. It does **not** yet
provide a live WebSocket dashboard or separately scheduled robot processes.

## Inspect and recover

```bash
uv run --extra learned relay-lab task-status TASK_ID
```

`coordination.json` identifies the database, task and worker. `summary.json`
contains the final durable task row. The database also keeps `assignment`,
`plan_frame`, `action_frame`, and `feedback_frame` records keyed by task/tick.
The usual replay, video and JSONL files remain available for inspection.

Restarting SpacetimeDB preserves coordination history. **Restarting a rollout
does not resume MuJoCo.** The database is not a simulator checkpoint. A crash
after an action is saved may occur before or after physics moves; replaying
that action would be unsafe. Tasks therefore have no automatic worker takeover
or action replay. A caught error attempts to mark the task failed; if the
database is unavailable, the last phase remains visible as unresolved. Inspect
the record and start a new task for a fresh simulation. No success is inferred
from a stale or incomplete task.

`--coordination memory` explicitly selects the standalone reference path.
The `compare` command and Python `RunConfig` defaults retain that offline
mode for existing experiments; the public CLI `run` defaults to SpacetimeDB. `--feedback off` omits outcome text and judge interruption flags from model
inputs. Structured observations and execution events remain available for
legal-action validation and verified handoffs; this is a text-feedback
ablation, not a no-communication mode. Database evidence and barriers remain.

## Validation

The ordinary test suite includes client failure/retry handling and runner
ordering checks. Real database tests are opt-in and never use fake server
responses as evidence of backend integration:

```bash
RELAY_SPACETIME_INTEGRATION=1 uv run --extra learned pytest tests/test_spacetime.py
RELAY_SPACETIME_LIVE_URL=http://127.0.0.1:3000 \
RELAY_SPACETIME_LIVE_DATABASE=relay-lab \
RELAY_SPACETIME_LIVE_TOKEN_FILE="$PWD/.relay-lab/spacetime-token" \
  uv run --extra learned pytest tests/test_spacetime_live.py
```

The live tests send concurrent duplicate commits, conflicting writes,
out-of-order phases, invalid evidence, and unauthorized identities directly to
the published reducers. Scripted success verifies infrastructure; learned-policy competence requires
separate held-out rollouts.

## Historical backend validation

The earlier integration verified durable storage through a server restart and
module republish: all 330 plan, action and feedback rows survived unchanged.
Those results remain under `runs/spacetime-validation/`. Its Fetch/ACP results
are historical; the public gateway is stopped and is not part of this runtime.
Current model/mediator outcomes are saved per run, rather than inferred from
that scripted infrastructure check.

## Official references

- [TypeScript modules](https://spacetimedb.com/docs/quickstarts/typescript/)
- [HTTP database API](https://spacetimedb.com/docs/http/database/)
- [HTTP authentication](https://spacetimedb.com/docs/http/authorization/)
- [Transactions and architecture](https://spacetimedb.com/docs/intro/key-architecture/)
