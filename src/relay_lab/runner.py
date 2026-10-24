"""Run the same observe -> communicate -> act loop for both robots."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, fields, replace
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import imageio.v2 as imageio
from PIL import Image

from .communication import LocalMessageBus
from .feedback import ExecutionFeedback
from .orchestration import FeedbackOrchestrator, LocalFeedbackCoordinator
from .policies import HttpVLAPolicy, ScriptedPolicy
from .replay import write_replay
from .tasks import TaskSpec, get_task_adapter
from .types import AGENTS, Action, Instruction, Observation


@dataclass(frozen=True)
class RunConfig:
    seed: int = 7
    max_steps: int = 600
    communication: bool = True
    message_latency: int = 1
    policy: str = "scripted"
    video: bool = True
    picker_url: str | None = None
    packer_url: str | None = None
    device: str = "auto"
    action_horizon: int = 5
    denoising_steps: int = 10
    transport: str = "local"
    task_id: str | None = None
    order: tuple[str, ...] | None = None
    feedback: bool = True
    coordination: str = "memory"
    planner: str = "rules"
    checkpoint: str | None = None
    llm_python: str | None = None

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.message_latency < 1:
            raise ValueError("message_latency must be at least one tick")
        if self.policy not in ("scripted", "http", "smolvla", "relay-vla"):
            raise ValueError("policy must be scripted, http, smolvla, or relay-vla")
        if self.policy == "http" and not (self.picker_url and self.packer_url):
            raise ValueError("HTTP policies need both --picker-url and --packer-url")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be auto, cpu, mps, or cuda")
        if not 1 <= self.action_horizon <= 50:
            raise ValueError("action_horizon must be between 1 and 50")
        if not 1 <= self.denoising_steps <= 100:
            raise ValueError("denoising_steps must be between 1 and 100")
        if self.transport != "local":
            raise ValueError("Fetch was removed; use the local mediator with SpacetimeDB")
        if self.planner not in ("rules", "llm"):
            raise ValueError("planner must be rules or llm")
        if self.planner == "llm" and not self.communication:
            raise ValueError("The shared LLM mediator requires communication enabled")
        if self.coordination not in ("memory", "spacetime"):
            raise ValueError("coordination must be memory or spacetime")
        if self.task_id is not None and (not self.task_id.strip() or len(self.task_id) > 128):
            raise ValueError("task_id must contain 1 to 128 characters")
        TaskSpec(order=self.order)


def observation_record(observation: Observation) -> dict:
    """Don't copy or dump large RGB arrays into each JSONL row."""
    return {
        field.name: getattr(observation, field.name)
        for field in fields(observation)
        if field.name not in ("rgb", "wrist_rgb")
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new --output path.")
    path.mkdir(parents=True, exist_ok=True)


def run_episode(config: RunConfig, output: Path) -> dict:
    """Separate agent contexts, next-tick mail, then a simultaneous environment step.

    Policy inputs are local observations. The central mediator receives the
    team's verified symbolic observations, including the packer's task order.
    The replay's tick labels name the decision; its scene shows the resulting state.
    """
    prepare_directory(output)
    task_id = config.task_id or str(uuid4())
    write_json(output / "config.json", {**asdict(config), "task_id": task_id})
    task_adapter = get_task_adapter(TaskSpec(order=config.order))
    env = task_adapter.create_environment(config.seed)
    bus = LocalMessageBus(enabled=config.communication, latency_ticks=config.message_latency)
    orchestrators = task_adapter.create_planners()
    feedback = {agent: ExecutionFeedback(agent, task_id) for agent in AGENTS}
    endpoints = {"picker": config.picker_url, "packer": config.packer_url}
    policies = {}
    backend = None
    records = []
    initial_state = env.state()
    writer = None
    started = time.perf_counter()
    counts = {"attempted": 0, "sent": 0, "delivered": 0, "dropped": 0}
    feedback_counts = {}
    database = None
    worker_id = str(uuid4())
    coordinator = None
    claimed = False
    llm = None
    mediator = None
    execution_events = {agent: [] for agent in AGENTS}
    last_mediator_call = None

    try:
        if config.coordination == "spacetime":
            from .spacetime import SpacetimeClient, SpacetimeConfig, SpacetimeError

            database = SpacetimeClient(SpacetimeConfig.from_env())
            database.create_task(
                task_id,
                task_adapter.spec.goal,
                {"kind": task_adapter.spec.kind, "order": list(env.order)},
                config.max_steps,
            )
            database.claim_task(task_id, worker_id)
            claimed = True
            write_json(
                output / "coordination.json",
                {**database.backend_info, "task_id": task_id, "worker_id": worker_id},
            )
        if config.feedback or database is not None or config.planner == "llm":
            coordinator = LocalFeedbackCoordinator(task_id)
        if config.feedback or config.planner == "llm":
            if database is not None:
                from .spacetime_feedback import SpacetimeFeedbackCoordinator

                coordinator = SpacetimeFeedbackCoordinator(
                    task_id, worker_id, database, coordinator
                )
            if config.planner == "rules":
                orchestrators = {
                    agent: FeedbackOrchestrator(agent, orchestrators[agent], coordinator)
                    for agent in AGENTS
                }
        if config.planner == "llm":
            from .llm import LocalLLM
            from .mediator import LLMMediator

            print("Loading the local Qwen mediator…", flush=True)
            llm = LocalLLM(**({"python": config.llm_python} if config.llm_python else {}))
            mediator = LLMMediator(llm)
        if config.policy == "smolvla":
            from .smolvla import SmolVLABackend, SmolVLAPolicy

            print("Loading the pinned SmolVLA LIBERO checkpoint…", flush=True)
            backend = SmolVLABackend(
                device=config.device,
                seed=config.seed,
                denoising_steps=config.denoising_steps,
            )
            policies = {
                agent: SmolVLAPolicy(agent, backend, action_horizon=config.action_horizon)
                for agent in AGENTS
            }
            write_json(output / "model.json", backend.model_info)
        elif config.policy == "relay-vla":
            from .relay_vla import RelayVLABackend, RelayVLAPolicy

            backend = RelayVLABackend(checkpoint=config.checkpoint, device=config.device)
            policies = {agent: RelayVLAPolicy(agent, backend) for agent in AGENTS}
            write_json(output / "model.json", backend.model_info)
        else:
            policies = {
                agent: ScriptedPolicy(agent)
                if config.policy == "scripted"
                else HttpVLAPolicy(endpoint=endpoints[agent], agent_id=agent)
                for agent in AGENTS
            }
        needs_rgb = any(policy.needs_rgb for policy in policies.values())
        if config.video:
            writer = imageio.get_writer(
                str(output / "rollout.mp4"),
                fps=10,
                codec="libx264",
                quality=7,
                macro_block_size=1,
            )
            writer.append_data(env.render())
        if config.video or needs_rgb:
            for agent in AGENTS:
                for camera, label in ((agent, "view"), (f"{agent}_wrist", "wrist")):
                    Image.fromarray(env.render(camera=camera, width=256, height=256)).save(
                        output / f"{agent}-{label}.png"
                    )

        with (
            (output / "steps.jsonl").open("w", encoding="utf-8") as step_log,
            (output / "messages.jsonl").open("w", encoding="utf-8") as mail_log,
            (output / "execution.jsonl").open("w", encoding="utf-8", buffering=1) as execution_log,
            (output / "mediator.jsonl").open("w", encoding="utf-8", buffering=1) as mediator_log,
            (output / "action-feedback.jsonl").open(
                "w", encoding="utf-8", buffering=1
            ) as action_log,
        ):

            def log_mail(event: dict) -> None:
                mail_log.write(json.dumps(event, allow_nan=False) + "\n")

            for _ in range(config.max_steps):
                tick = env.tick
                if database is not None:
                    task = database.get_task(task_id)
                    if not task or (
                        task["worker_id"] != worker_id
                        or task["status"] != "running"
                        or task["phase"] != "ready"
                        or task["next_tick"] != tick
                    ):
                        raise RuntimeError(
                            "Spacetime task does not authorize the next simulation tick"
                        )
                # Take BOTH observations before any decisions or world changes.
                observations = {
                    agent: env.observe(agent, rgb=policies[agent].needs_rgb) for agent in AGENTS
                }
                inboxes = {agent: bus.receive(agent, tick) for agent in AGENTS}
                peer_events = {agent: [] for agent in AGENTS}
                for inbox in inboxes.values():
                    for message in inbox:
                        counts["delivered"] += 1
                        log_mail({"event": "delivered", "tick": tick, **asdict(message)})
                if mediator is not None:
                    contexts = {agent: coordinator.context(agent, [], tick) for agent in AGENTS}
                    model_contexts = (
                        contexts
                        if config.feedback
                        else {agent: {"text": "", "critical": False} for agent in AGENTS}
                    )
                    decisions = mediator.decide(observations, execution_events, model_contexts)
                    decisions = {
                        agent: replace(
                            decisions[agent],
                            instruction=replace(
                                decisions[agent].instruction,
                                feedback=contexts[agent]["text"] if config.feedback else "",
                                replan=(
                                    decisions[agent].instruction.replan
                                    or (config.feedback and contexts[agent]["critical"])
                                ),
                            ),
                        )
                        for agent in AGENTS
                    }
                    if (
                        mediator.last_record
                        and mediator.last_record["call_id"] != last_mediator_call
                    ):
                        last_mediator_call = mediator.last_record["call_id"]
                        mediator_log.write(json.dumps(mediator.last_record, allow_nan=False) + "\n")
                else:
                    decisions = {
                        agent: orchestrators[agent].decide(
                            observations[agent], inboxes[agent], peer_events[agent]
                        )
                        for agent in AGENTS
                    }
                if database is not None:
                    plans = {
                        agent: {
                            "instruction": asdict(decisions[agent].instruction),
                            "context": {
                                "text": decisions[agent].instruction.feedback,
                                "critical": decisions[agent].instruction.replan,
                            },
                            "observation": observation_record(observations[agent]),
                            "mediator": mediator.last_record if mediator is not None else None,
                        }
                        for agent in AGENTS
                    }
                    stored = database.plan_tick(
                        task_id, worker_id, tick, plans["picker"], plans["packer"]
                    )
                    # Models consume the committed instructions, including the
                    # feedback read from the previous committed transition.
                    decisions = {
                        agent: replace(
                            decisions[agent],
                            instruction=Instruction(
                                **json.loads(stored[f"{agent}_json"])["instruction"]
                            ),
                        )
                        for agent in AGENTS
                    }
                sent = {}
                for agent in AGENTS:
                    text = decisions[agent].message
                    sent[agent] = None
                    if text:
                        counts["attempted"] += 1
                        sent[agent] = bus.send(agent, text, tick)
                        event = "sent" if sent[agent] else "dropped"
                        counts[event] += 1
                        payload = (
                            asdict(sent[agent])
                            if sent[agent]
                            else {
                                "sender": agent,
                                "text": text,
                                "sent_tick": tick,
                            }
                        )
                        log_mail({"event": event, "tick": tick, **payload})
                actions = {
                    agent: policies[agent].act(observations[agent], decisions[agent].instruction)
                    for agent in AGENTS
                }
                if database is not None:
                    proposals = {
                        agent: {
                            "action": asdict(actions[agent]),
                            "prediction": getattr(policies[agent], "last_prediction", None),
                        }
                        for agent in AGENTS
                    }
                    stored = database.prepare_tick(
                        task_id, worker_id, tick, proposals["picker"], proposals["packer"]
                    )
                    actions = {
                        agent: Action(**json.loads(stored[f"{agent}_json"])["action"])
                        for agent in AGENTS
                    }
                # Never retry this physical step. Database retries above only
                # resend an identical reducer, not simulator execution.
                result = env.step(actions)
                post_state = env.state()
                execution_events = {}
                action_feedback = {}
                after_observations = {}
                for agent in AGENTS:
                    after = env.observe(agent)
                    after_observations[agent] = after
                    execution_events[agent] = [
                        event.to_dict()
                        for event in feedback[agent].update(
                            observations[agent],
                            after,
                            decisions[agent].instruction,
                            result["events"],
                            actions[agent],
                        )
                    ]
                    if coordinator is not None:
                        action_feedback[agent] = coordinator.digest(
                            agent,
                            observations[agent],
                            after,
                            decisions[agent].instruction,
                            actions[agent],
                            execution_events[agent],
                        )
                if database is not None:
                    reports = {
                        agent: {
                            **action_feedback[agent],
                            "execution_events": execution_events[agent],
                            "observation": observation_record(after_observations[agent]),
                        }
                        for agent in AGENTS
                    }
                    database.commit_tick(
                        task_id,
                        worker_id,
                        tick,
                        reports["picker"],
                        reports["packer"],
                        {
                            "tick": env.tick,
                            "packed": list(env.packed),
                            "order": list(env.order),
                            "success": bool(env.success),
                        },
                    )
                # External progress and peer reports describe committed state.
                # If the DB is unavailable, no future instruction is issued.
                for agent in AGENTS:
                    if agent in action_feedback:
                        action_log.write(json.dumps(action_feedback[agent], allow_nan=False) + "\n")
                    for event in execution_events[agent]:
                        execution_log.write(json.dumps(event, allow_nan=False) + "\n")
                        feedback_counts[event["kind"]] = feedback_counts.get(event["kind"], 0) + 1
                executed_deltas = {
                    agent: [
                        after - before
                        for after, before in zip(
                            post_state["robots"][agent]["position"], observations[agent].ee_position
                        )
                    ]
                    for agent in AGENTS
                }
                record = {
                    "tick": tick,
                    "agents": {
                        agent: {
                            "observation": observation_record(observations[agent]),
                            "instruction": asdict(decisions[agent].instruction),
                            "action": asdict(actions[agent]),
                            "executed_delta": executed_deltas[agent],
                            "workspace_clipped": any(
                                abs(requested - actual) > 1e-8
                                for requested, actual in zip(
                                    actions[agent].delta, executed_deltas[agent]
                                )
                            ),
                            "inbox": [asdict(message) for message in inboxes[agent]],
                            "peer_execution_events": peer_events[agent],
                            "execution_events": execution_events[agent],
                            "outgoing": asdict(sent[agent]) if sent[agent] else None,
                            "attempted_message": decisions[agent].message,
                            "policy": config.policy,
                            "prediction": getattr(policies[agent], "last_prediction", None),
                            "action_feedback": action_feedback.get(agent),
                        }
                        for agent in AGENTS
                    },
                    "events": result["events"],
                    "state": post_state,
                    "mediator_call_id": last_mediator_call,
                    "mediator": (
                        {
                            key: mediator.last_record.get(key)
                            for key in ("call_id", "tick", "trigger", "reason")
                        }
                        if mediator is not None and mediator.last_record is not None
                        else None
                    ),
                }
                records.append(record)
                step_log.write(json.dumps(record, allow_nan=False) + "\n")
                if writer:
                    writer.append_data(env.render())
                if env.success:
                    break
                if config.policy in ("smolvla", "relay-vla") and env.tick % 25 == 0:
                    print(
                        f"{config.policy} · step {env.tick}/{config.max_steps} · packed {list(env.packed)}",
                        flush=True,
                    )

        if writer:
            writer.close()
            writer = None
        summary = {
            **asdict(config),
            "task_id": task_id,
            "success": bool(env.success),
            "termination": "success" if env.success else "step_limit",
            "steps": env.tick,
            "order": list(env.order),
            "packed": list(env.packed),
            "messages": counts,
            "execution_events": feedback_counts,
            "transport_info": {"kind": "local", "external_services": False},
            "duration_seconds": round(time.perf_counter() - started, 3),
            "initial_state": initial_state,
            "control_dt": 0.1,
            "model": backend.model_info if backend else None,
            "task": {"kind": task_adapter.spec.kind, "goal": task_adapter.spec.goal},
            "coordination_info": (
                {**database.backend_info, "task": database.get_task(task_id)}
                if database is not None
                else {"backend": "memory"}
            ),
            "orchestrator": (
                "Local Qwen LLM mediator"
                if mediator is not None
                else "Rule-based relay task planner"
            ),
            "mediator": mediator.stats if mediator is not None else None,
            "judge": "Deterministic simulator evidence; no judge-model inference",
            "versions": {
                "relay-lab": version("relay-lab"),
                "mujoco": version("mujoco"),
                "python": platform.python_version(),
            },
            "limitations": [
                f"Task planner: {config.planner}; action-policy mode: {config.policy}.",
                "Kinematic Cartesian grippers; grasp attachment uses an idealized proximity rule.",
                "Task planners receive structured simulator observations; learned actions use local RGB.",
                "Private-order diagnostic task; not a RoboPoly evaluation or a novelty claim.",
            ],
        }
        if config.policy == "smolvla":
            summary["limitations"].extend(
                [
                    "Pretrained on LIBERO, not adapted to Relay Lab's scene or embodiment.",
                    "VLA sees RGB, proprioception, local instruction and optional grounded execution text; task planners still use local object poses.",
                    "Fixed orientation: predicted rotation is logged but not applied; translation is bounded.",
                ]
            )
        if backend:
            write_json(output / "model.json", summary["model"])
        write_json(output / "summary.json", summary)
        write_replay(
            output / "replay.html",
            summary,
            records,
            video_name="rollout.mp4" if config.video else None,
        )
        return summary
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if mediator is not None and mediator.last_record is not None:
            try:
                write_json(output / "mediator-last-record.json", mediator.last_record)
            except OSError:
                pass  # Preserve the original failure if the output volume is unavailable.
        if database is not None and claimed:
            # A failed worker is never reassigned automatically. A durable
            # action proposal is not a checkpoint of MuJoCo's physical state.
            try:
                database.abort_task(task_id, worker_id, f"Worker stopped: {type(exc).__name__}")
            except SpacetimeError as coordination_error:
                failure["coordination_abort"] = type(coordination_error).__name__
        write_json(output / "error.json", failure)
        raise
    finally:
        try:
            if writer:
                writer.close()
        finally:
            try:
                env.close()
            finally:
                try:
                    if llm is not None:
                        llm.close()
                finally:
                    if hasattr(bus, "close"):
                        bus.close()
