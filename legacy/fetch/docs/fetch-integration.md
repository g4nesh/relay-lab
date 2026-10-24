# Fetch integration

Relay Lab connects an ACP team agent, two local Fetch robot agents, a local
Fetch feedback coordinator, and the MuJoCo/VLA execution loop. The coordinator
turns action proposals plus measured execution into text that is included in
future model inputs. It retains separate histories for the two robots. The
action model remains on the operator's machine; this new feedback path does not
establish competence on the custom task.

SpacetimeDB now stores the task, both robot assignments, plans, action batches,
and observed feedback. CLI runs and the public gateway use it by default.
Start and publish the local database using [the backend setup](spacetime.md)
before running the commands below. Fetch remains the ACP/message/interpreter
layer. The gateway checks the durable terminal result before reporting an
outcome, and observed progress is emitted only after its database commit.

## Challenge requirements

| Requirement | Relay Lab implementation and evidence |
| --- | --- |
| Registered on Agentverse | The stable team identity is registered through Agentverse Mailbox; the [profile screenshot](../runs/fetch-integration/coordinator-profile.png) shows **Active** and **ASI Available**. |
| Agent Chat Protocol | Standard ACP v0.3 messages, immediate acknowledgement, original session preservation, progress, and terminal outcome; verified by SDK and public chat round trips. |
| Discoverable and directly usable in ASI:One | The public [Relay Lab profile](https://asi1.ai/) opens a direct chat accepting `pack mint, gold, coral`. The [published guide](../runs/fetch-integration/coordinator-guide.png) shows the updated commands and starter prompts. The local Mac and gateway must stay online to execute it. |

This addresses the challenge's registration, protocol, and direct-use
requirements. It does not assert marketplace ranking or completion of optional
profile and analytics setup. The protocol follows the
[official ACP guide](https://docs.asi1.ai/documentation/tutorials/agent-chat-protocol).

## Install and run

```bash
uv sync --python 3.13 --extra vla --extra fetch

# Real model, real Fetch SDK peer messages, local recording.
uv run --extra vla --extra fetch relay-lab run \
  --policy smolvla --transport fetch --device mps --max-steps 60 \
  --order gold coral mint

# Same task without recursive text feedback for comparison.
uv run --extra vla --extra fetch relay-lab run \
  --policy smolvla --transport fetch --device mps --max-steps 60 \
  --order gold coral mint --feedback off

# Successful reference for checking the protocol and simulator.
uv run --extra vla --extra fetch relay-lab run \
  --policy scripted --transport fetch --seed 7 --no-video

# ACP team gateway, with an Agentverse mailbox.
uv run --extra vla --extra fetch relay-lab fetch-agent --device mps

# Local ACP gateway without Agentverse registration.
uv run --extra vla --extra fetch relay-lab fetch-agent --local --device mps
```

Keep both extras on `uv run` so uv retains both SDK and model dependencies.
The pinned packages are `uagents==0.26.1` and `uagents-core==0.4.12`.

The chat commands are deliberately constrained:

- `help`: describe the available simulation and limitations.
- `capabilities`: list the implemented task adapter and accepted intent grammar.
- `pack coral, mint, gold`: set the parcel order for tray slots 1, 2, and 3.
- `Please pack gold then mint then coral`: use an alternate order.
- `task: pack mint, coral, gold`: use an explicit task label.
- `run relay`: run the seeded three-parcel task.
- `run relay seed 7`: select a reproducible task seed.
- `status`: report the requesting chat session's current job.

ASI:One may prepend the selected agent's address to the text. The gateway
accepts exactly one leading `@<its-own-address>` followed by whitespace before
these commands. Other addresses, repeated mentions, and free-form instructions
remain unsupported; the same input size and execution limits apply. The pack
grammar requires all three colors exactly once and rejects extra instructions.
The request log stores a canonical task goal and validated parameters, and the
gateway rejects a simulator summary whose order differs from the requested one.

`TaskSpec`, `TaskAdapter`, and `RelayTaskAdapter` separate the relay environment,
task planner, and role-specific goals from the feedback loop. This is an
extension point, not open-ended task understanding: another task still needs
parameter validation, a compatible execution environment, and final outcome
checks. Only the relay task family is implemented.

The operator selects `--policy`, `--max-steps` (at most 600), `--timeout` (at
most 3600 seconds), and `--device` when starting the gateway. Remote text cannot
choose arbitrary commands, filesystem paths, code, or policies. Only one run
is active at a time. `scripted` is explicitly labeled in responses; there is no
automatic fallback from SmolVLA.

## Execution and communication

The gateway implements the standard `AgentChatProtocol` with `ChatMessage`,
`ChatAcknowledgement`, and `EndSessionContent`. It immediately acknowledges each
request, preserves the originating envelope session for background progress,
deduplicates message IDs within sender/session scope, and ends the session only
with a terminal response. A subprocess keeps neural inference and rendering off
the chat event loop.

Within that subprocess, the picker and packer have separate uAgents identities
and typed message handlers. Requests and observed feedback pass through the
SDK's local dispatcher and acknowledgement handlers. This is real local SDK
communication, not a claim that every robot message crosses Agentverse or a
signed network envelope. Missing acknowledgements stop the rollout; messages
never silently bypass Fetch. Simulation delivery still respects delayed ticks.

With feedback enabled, each robot also exchanges typed orchestration requests
and replies with a third local uAgent, the feedback coordinator. Before a model
decision it asks for its own bounded history plus newly delivered peer evidence.
After execution it submits the action and local before/after observations; the
coordinator translates these into text such as requested motion versus measured
motion and close command versus actual held object. The response becomes input
to the next decision. Requests and replies are correlated to task, robot,
session, sequence, and tick; a missing or invalid reply stops the rollout.

The per-robot task planner chooses the current local instruction using relay
rules. The VLA predicts motion and grip actions from images, proprioception,
that instruction, and the feedback text. Routine new feedback enters the next
action chunk. Critical evidence can invalidate the queued chunk and trigger
early replanning. This is inference-time feedback, not online model training.
The model weights are unchanged, and no second language model invents action
outcomes. `--feedback off` disables model feedback without changing the policy;
Spacetime mode still interprets and saves evidence for its tick barrier.
It disables both appended language history and critical-event replanning, so an
on/off comparison measures that combined mechanism. SmolVLA uses the same
512-token language budget in both modes; oversized prompts fail explicitly
instead of dropping evidence through truncation. Prediction records retain the
actual normalized prompt and token count.

Local and Fetch modes share the action-effect digest, but only Fetch mode
currently delivers peer execution reports and requires verified handoff events.
A local-versus-Fetch comparison is consequently not a pure transport ablation.

Each robot's feedback tracker consumes its own observations before and after
a step, its instruction, and relevant simulator events. It emits `started`,
`holding`, `transfer_ready`, `completed`, or `blocked`, with task, instruction,
event IDs and evidence. `completed` always means a **local instruction**.
Only the simulator's final success check establishes episode completion.

A relay handoff requires an observed release, stationary parcel at the relay,
and a clear picker. The packer must receive that verified readiness event before
it collects the requested color. A wrong-color grasp is reported as a mismatch;
a raw action vector or open-gripper command never proves placement. The local
reference transport remains available for earlier experiments.

## Identity and Agentverse setup

A private project identity is generated in `.relay-lab/identity.json`, with file
permissions 0600 and separate derived team/robot identities. This folder is
ignored by Git. Keep it to retain the same public addresses across restarts.
Never paste its contents or Agentverse API keys into chat or commit them.

Start the gateway and open the Inspector URL printed in its console. The
[official mailbox workflow](https://uagents.fetch.ai/docs/agentverse/mailbox)
uses **Connect → Mailbox** to associate the running agent with the signed-in
Agentverse account. Mailbox delivery needs no public tunnel. The inspector API
binds to `127.0.0.1`; the mailbox client makes outbound connections.
If Inspector cannot detect the running gateway, check the browser's Local
Network Access permission for Agentverse, as described in that official guide.

If the browser inspector cannot reach localhost, create a short-lived API key
with Mailbox write access in Agentverse, then use the SDK's same connection API.
Agentverse's **Write** selection resolves to **read and write** access in its
final consent dialog; review that exact scope and keep every other service set
to None. The one-hour expiry is sufficient for setup.

```bash
uv run --extra vla --extra fetch relay-lab fetch-connect
```

The terminal prompts for the key without echoing it. An automation may instead
use `--token-file .relay-lab/agentverse-token`; keep that file private and remove
it after setup. The agent's signed identity handles subsequent mailbox access;
the setup key is not needed by the running gateway. Never put a key directly in
command-line arguments or an example file.

During setup on October 23, 2026, Agentverse's API-key consent flow redirected
to a retired `/profile/create-api-key` callback and did not create a key. The
already approved Mailbox consent was completed through Fetch's documented
`/v1/tokens` exchange, preserving the `av:mailbox` scope and one-hour expiry.
The [official SDK implementation](https://github.com/fetchai/agent-launch-toolkit/blob/main/packages/sdk/src/wallet-auth.ts)
documents that exchange. This was a workaround for the website callback failure;
normal setup should use Inspector or the API-key UI. No authorization code or
token is needed in this repository, and no additional credential CLI is required.

The gateway publishes its ACP manifest and uses the off-chain Almanac API. It
does not fund a wallet or submit ledger transactions. Signing into Agentverse,
creating a mailbox, and making the agent discoverable are distinct steps; a
local SDK test alone does not prove Agentverse/ASI:One connectivity. No ASI model
API key is needed because this starter uses a constrained command parser.

Once connected, open the agent's profile and choose **Chat with Agent** to reach
ASI:One, then send `run relay`. The machine and gateway must remain running to
execute requests. See the [official ACP guide](https://docs.asi1.ai/documentation/tutorials/agent-chat-protocol).

## Evidence

Every simulation run saves its normal video/replay and logs, plus:

| File | Contents |
| --- | --- |
| `execution.jsonl` | Verified per-robot instruction progress and supporting evidence |
| `transport.json` | SDK version, public addresses, session, ACK/delivery counters, network mode |
| `transport.jsonl` | Observed event and typed peer delivery receipt |
| `steps.jsonl` | Own and received execution events alongside instructions/actions |
| `action-feedback.jsonl` | Proposed actions, measured effects, text returned to the model, critical flags |
| `coordination.json` | Spacetime server/database provenance, task and worker IDs; no credentials |

The Fetch transport summary also reports orchestration request/reply counts.
The exact model language prompt and feedback are recorded in the step log so
the feedback path can be inspected separately from task success.

Gateway runs use a unique task folder under `runs/fetch/`, with adjacent request
metadata and subprocess console logs. Public replies include policy and actual
outcome, without exposing local paths or private identity material.

Earlier transport/account validation, before recursive action feedback:

- Fetch scripted reference: seed 7, **329 steps, 3/3 packed**, three verified
  handoffs, 45 SDK acknowledgement round trips (12 requests and 33 feedback events).
- Fetch SmolVLA probe: **60 steps, 0/3 packed**; actual MPS neural inference and
  Fetch messages ran together. This was an integration probe, not a task-success
  evaluation.
- Real ACP integration test: SDK chat request → real simulator subprocess →
  robot Fetch messages → observed progress → ACP result, **3/3 packed** using
  the explicitly selected scripted reference.

Account connection completed on October 23, 2026:

- The team gateway's Agentverse mailbox was successfully registered to the
  signed-in account using `fetch-connect --token-file`.
- Public team address:
  `<YOUR_REGISTERED_AGENT_ADDRESS>`.
- The setup token had Mailbox scope only and a verified lifetime of exactly
  3600 seconds. Temporary local token and exchange files were deleted after
  successful registration; the stable project identity remains for mailbox use.

Live Agentverse round trip confirmed on October 23, 2026:

- Sent `run relay` through the account's Agentverse **Testing → Manual Test**
  chat. The cloud mailbox delivered the ACP request to the local gateway, which
  launched task `98f3e19c-3b98-4a79-ac1a-de7c4d53d73f`.
- The actual SmolVLA MPS run executed **600 steps in 151.214 seconds**, with
  **243 model inference calls and 0/3 parcels packed**. The robot transport
  recorded **38 SDK acknowledgements**: 20 requests and 18 observed-feedback
  messages. These robot exchanges used the local SDK dispatcher.
- Observed progress and the final **TASK NOT COMPLETED** result returned to
  the same Agentverse chat session, `9e6b721b-ab4c-4f45-ad3d-bc90e19bd80d`.
  This verifies the cloud chat, mailbox, local execution, and reply path; it
  does not establish action-policy task success.
- Artifacts: [chat result screenshot](../runs/fetch-integration/agentverse-live-result.png),
  [rollout replay](../runs/fetch/98f3e19c-3b98-4a79-ac1a-de7c4d53d73f/replay.html),
  and [measured summary](../runs/fetch/98f3e19c-3b98-4a79-ac1a-de7c4d53d73f/summary.json).

Live ASI:One round trip also confirmed on October 23, 2026:

- Account signup and onboarding completed. The public
  [Relay Lab profile](https://asi1.ai/)
  is active, with the project README and `run relay`, `run relay seed 7`, and
  `help` starter prompts available.
- ASI:One's automatically prepended agent mention exposed a strict-parser
  incompatibility. After the narrow own-address prefix fix, the same chat
  successfully launched task `a0d52507-0d37-44f3-b408-a5e52b1209c9`.
- The actual SmolVLA MPS run executed **600 steps in 155.292 seconds**, with
  **243 model inference calls, 38 robot SDK acknowledgements, and 0/3 parcels
  packed**. Acceptance, observed progress, and the final **TASK NOT COMPLETED**
  response returned to ASI:One session `d1868362-180d-474d-a0d4-ca2323712680`.
- Artifacts: [ASI:One result screenshot](../runs/fetch-integration/asi-live-result.png),
  [SDK envelope history with 11 records](../runs/fetch-integration/asi-acp.json),
  [rollout replay](../runs/fetch/a0d52507-0d37-44f3-b408-a5e52b1209c9/replay.html),
  and [measured summary](../runs/fetch/a0d52507-0d37-44f3-b408-a5e52b1209c9/summary.json).
  The saved SDK history includes message payloads and session metadata; it is
  not a signed-packet dump and does not retain cryptographic signature fields.

The account, mailbox, and ASI:One execution path are verified. The machine and
gateway must remain running for new chat requests to execute. These successful
integration checks do not change the observed action-policy failure on this task.

The test suite covers explicit order
requests, request/result consistency, observation-grounded text, separate robot
histories, missing coordinator replies, and real ACP → simulator → result
round trips for both seeded and user-selected orders using the scripted
reference. These tests establish the integration contract, not learned task
competence.

The recursive-feedback public run completed on October 24, 2026:
`5edcb2e3-edf3-4510-a67d-a5d499eeee90`, order mint → gold → coral,
600 steps, 1,088 MPS inference calls, 2,400 coordinator request/reply pairs,
and **0/3 packed** in 788.17 seconds. Its final result returned to ASI:One;
see [the result](../runs/fetch-integration/asi-recursive-result.png) and
[replay](../runs/fetch/5edcb2e3-edf3-4510-a67d-a5d499eeee90/replay.html).
Frequent critical feedback caused much more inference than the earlier run;
this is not evidence of improved model competence.

Spacetime validation is recorded separately in [backend evidence](spacetime.md).

The orchestration logic remains rule-based. Our next learning step is adapting
the action policy using demonstrations generated in this project's own scene.
