# Relay Lab coordination module

This SpacetimeDB module is the synchronization authority for the local two-robot
simulator. It stores assignments, instructions, proposed actions, measured
outcomes and the text supplied to the next model invocation. The Fetch gateway
remains the ACP/ASI:One entry point; the model and MuJoCo execute in the local
Python process.

The server and module SDK are pinned to **2.10.0**. From the repository root:

```sh
python3 scripts/spacetime.py install
python3 scripts/spacetime.py start
```

Keep the server running, and in another terminal:

```sh
python3 scripts/spacetime.py publish
python3 scripts/spacetime.py status
```

The installer is verified on Apple Silicon and uses a pinned, checksum-verified
official bootstrap. It installs only inside `.relay-lab/spacetime/`, makes no
shell changes, and installs module packages with `npm ci`. On another platform,
install the official 2.10.0 CLI binaries at the path identified by the installer.
The server binds to `127.0.0.1:3000`, stores durable data in
`.relay-lab/spacetime/data/`, and uses a 1 GiB page-pool limit. Publishing saves
the local development owner token in `.relay-lab/spacetime-token` with mode
0600. The helper neither displays that token nor puts it in process arguments.
All runtime files are git-ignored. Publishing preserves existing data;
restarting uses the same directory and does not republish or clear it.

## State transitions

```text
create_task → queued / ready (two robot assignments)
claim_task  → running / ready (one immutable worker)
plan_tick  → running / planned (both instructions + current observations)
prepare_tick → running / executing (both model action proposals)
commit_tick  → running / ready (both measured reports, next_tick += 1)
            ↘ succeeded / terminal, if observed slots equal assigned order
            ↘ exhausted / terminal, if the step budget is spent
abort_task  → failed / terminal
```

Both robots form one batch because this simulator advances them simultaneously.
`plan_tick`, `prepare_tick`, and `commit_tick` each run as one database
transaction. A malformed second robot report prevents either report, either
robot's context, or the task tick from changing. A next-tick plan cannot precede
the previous observed commit.

The worker must read the committed plan before inference and the committed
action proposals before stepping the simulator. It reads the latest committed
feedback for future planning. Reducer calls use the HTTP API's durable
`confirmed=true` mode. Exact retries are idempotent; a changed JSON payload at
the same task and tick is rejected. The Python client emits canonical JSON so
byte equality is unambiguous.

## Ownership and failure semantics

The publisher's identity is captured in `init`. Every reducer checks this
identity, and every table is private. The local runtime uses the publisher's
token. This is a single-operator development backend, not a multi-tenant
authorization system. The database owner is trusted to provide simulator
observations; consistency checks do not independently verify physical reality.

A task's worker ID cannot change. There is deliberately no lease takeover or
automatic crash recovery: after `prepare_tick`, the process may have executed
the physical step even if it never committed feedback. Such a task remains
`executing` until explicitly aborted and a fresh task is created. Never replay
an action from an executing row after a process restart. This version does not
checkpoint MuJoCo, policy queues or the interpreter's in-memory history.

The database does validate that reported requests match authorized actions,
reported motion matches before/after observations, attachments and gripper
openings agree, and reported completion agrees with the packer's observed tray
slots and the originally assigned order. Model-generated text or action requests
alone cannot mark a task successful.

## HTTP reducer contract

POST `/v1/database/{database}/call/{reducer}?confirmed=true` with a JSON array
of positional arguments and `Authorization: Bearer <publisher-token>`.

| Reducer | Arguments in order |
| --- | --- |
| `create_task` | `task_id, intent, spec_json, max_steps` |
| `claim_task` | `task_id, worker_id` |
| `plan_tick` | `task_id, worker_id, tick, picker_json, packer_json` |
| `prepare_tick` | `task_id, worker_id, tick, picker_json, packer_json` |
| `commit_tick` | `task_id, worker_id, tick, picker_report_json, packer_report_json, outcome_json` |
| `abort_task` | `task_id, worker_id, reason` |

IDs contain 1–128 ASCII letters, digits, underscores or hyphens. JSON arguments
are strings, limited to 65,536 characters. `spec_json` contains `kind: "relay"`
and the concrete seeded `order` permutation of coral, mint and gold.

Each plan contains an `instruction`, `{text, critical}` feedback `context`, and
an `observation` without images. Instruction `feedback` and `replan` must match
the stored context. Each prepared action contains `action: {delta, gripper}`;
optional prediction metadata can be retained alongside it. Requests remain
unclipped in the record so that the interpreter can describe differences
between requested and executed motion.

Each result contains the existing `relay-lab.action-feedback.v1` `report`,
bounded `text`, a `critical` flag, `execution_events`, and the observed
post-step `observation`. For transition `tick`, the report's `before_tick` is
`tick` and its observation/report `tick` is `tick + 1`. The global outcome
contains `tick`, `packed`, `order`, and `success`.

The private tables are `task`, `assignment`, `plan_frame`, `action_frame`,
`feedback_frame`, and `module_owner`. Frame primary keys are
`{task_id}:{tick}`. The task's `next_tick` is the next transition to execute;
`last_feedback_tick` is the last committed transition index, initially `-1`.
Assignments expose `context_text` and `context_critical` alongside their latest
complete feedback JSON. The transition index differs from the post-step
observation tick by one.

## Official API references

- [Reducer transactions](https://spacetimedb.com/docs/functions/reducers/)
- [Owner identity in the init lifecycle](https://spacetimedb.com/docs/functions/reducers/lifecycle/)
- [HTTP calls, durable confirmation and SQL reads](https://spacetimedb.com/docs/http/database/)
- [HTTP authorization](https://spacetimedb.com/docs/http/authorization/)

The live integration tests in `tests/test_spacetime_live.py` bypass Python
validation and exercise the published reducers, including authorization,
concurrent duplicate delivery, invalid result rollback and terminal fencing.
