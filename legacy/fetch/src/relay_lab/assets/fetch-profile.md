# Relay Lab — Fetch task orchestration for VLAs

Relay Lab uses Fetch agents to coordinate two simulated robots in a closed loop:
task intent → VLA actions → simulation observations → grounded text feedback →
the next VLA input. Ask for a parcel packing order through ASI:One, and the agent
starts a bounded experiment and reports the actual result observed in MuJoCo.

## Commands

- `pack coral, mint, gold` — choose the order of the three tray slots.
- `Please pack gold then coral then mint` — the same task with a different order.
- `task: pack mint, gold, coral` — an explicit task label also works.
- `run relay` — start the default seeded experiment.
- `run relay seed 7` — choose a reproducible seed.
- `status` — check the current request in the same chat session.
- `capabilities` — see the available task adapters and supported intent grammar.
- `help` — see available commands and limitations.

The relay task requires all three parcel colors exactly once. The current adapter
supports this one task family; it does not infer arbitrary tasks from free text.
Its environment, task rules, and role-specific goals have a separate adapter
interface so another simulation task can be implemented later.

One experiment runs at a time. A request is acknowledged immediately; progress
and the final result arrive in the same Agent Chat Protocol session. The operator
chooses the policy and execution budget. Requests cannot change the policy,
execute code, select arbitrary files, or control physical hardware.

## What actually runs

The default action policy is the pretrained SmolVLA LIBERO checkpoint, running
on the operator's local GPU. Two independent robot policy contexts share model
weights. The picker sees the source parcels; the packer knows the desired order.
Their Fetch agents coordinate requests and handoffs. High-level planning is
currently rule-based. Each robot retains a separate, bounded history containing
its proposed actions, observed motion and grasp state, instruction results, and
received peer messages. The next model input includes this history as text
alongside the current visual observation and local instruction.

Progress messages use observed simulator state. “Holding mint” means an observed
attachment; “transfer ready” requires a released, stationary parcel and a clear
picker. Completing an individual instruction does not mean the entire task is
complete. Only the simulator's final check establishes success.

The public team agent uses Agentverse Mailbox and the Agent Chat Protocol (ACP)
to receive ASI:One requests. The two robot agents exchange typed messages through
the local Fetch uAgents SDK dispatcher. Their robot messages stay local; the
team agent sends observed progress and the final outcome back through the chat.

## Prototype limitations

This is an original, simplified simulation with Cartesian grippers and idealized
proximity grasps. It uses no DuoMind or RoboPoly demonstrations. The checkpoint
has not been adapted to this scene, so successful task completion is not promised.
The local seed-7 VLA attempt packed 0/3 parcels in 600 steps. A separately selected
scripted reference can verify the communication and simulator; responses always
identify which policy ran. There is no scripted fallback during VLA execution.

The gateway and local machine must be online. Videos, raw actions, execution
evidence, and replay files remain with the operator. The agent reports truthful
success, failure, timeout, or unverified outcome through ASI:One.
