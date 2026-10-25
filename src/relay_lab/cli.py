"""Small command line entry point. Start with: relay-lab run --seed 7."""

import argparse
from datetime import datetime
from pathlib import Path

from .runner import RunConfig, prepare_directory, run_episode, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Relay Lab: robot teams, local model coordination, and shared simulation state."
    )
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Record one rollout with video and an HTML replay")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--comm", choices=("on", "off"), default="on")
    run.add_argument("--max-steps", type=int, default=600)
    run.add_argument("--message-latency", type=int, default=1)
    run.add_argument(
        "--policy", choices=("relay-vla", "smolvla", "scripted", "http"), default="relay-vla"
    )
    run.add_argument("--planner", choices=("llm", "rules"), default="llm")
    run.add_argument("--checkpoint", help="Task-adapted Relay VLA checkpoint path")
    run.add_argument("--llm-python", help="Python interpreter for the isolated local MLX runtime")
    run.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    run.add_argument(
        "--action-horizon",
        type=int,
        default=5,
        help="Number of model-predicted actions to execute before replanning (1–50)",
    )
    run.add_argument("--denoising-steps", type=int, default=10)
    run.add_argument(
        "--coordination",
        choices=("spacetime", "memory"),
        default="spacetime",
        help="Shared durable task backend; memory is the standalone reference mode",
    )
    run.add_argument("--task-id", help="Optional unique task identifier in SpacetimeDB")
    run.add_argument(
        "--order",
        nargs=3,
        choices=("coral", "mint", "gold"),
        help="Packing order; each color must appear exactly once",
    )
    run.add_argument(
        "--feedback",
        choices=("on", "off"),
        default="on",
        help="Feed grounded outcome text into the mediator (and compatible action models)",
    )
    run.add_argument("--picker-url")
    run.add_argument("--packer-url")
    run.add_argument(
        "--no-video",
        action="store_true",
        help="Skip recording; scripted runs need no OpenGL (learned policies still render RGB)",
    )
    run.add_argument("--output", type=Path)
    compare = commands.add_parser(
        "compare", help="Paired scripted-reference comm on/off runs with identical seeds"
    )
    compare.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    compare.add_argument("--max-steps", type=int, default=600)
    compare.add_argument("--message-latency", type=int, default=1)
    compare.add_argument("--output", type=Path)
    serve = commands.add_parser("serve", help="Serve local replays with seekable video")
    serve.add_argument("--directory", type=Path, default=Path("runs"))
    serve.add_argument("--port", type=int, default=8766)
    status = commands.add_parser("task-status", help="Read a task's authoritative Spacetime state")
    status.add_argument("task_id")
    factory = commands.add_parser("factory", help="Open the live five-robot factory dashboard")
    factory.add_argument("--port", type=int, default=8767)
    factory.add_argument("--coordination", choices=("spacetime", "memory"), default="spacetime")
    factory.add_argument("--replay", type=Path, help="Completed factory replay to make viewable")
    factory.add_argument("--comparison", type=Path, help="Measured communication benchmark results")
    factory_run = commands.add_parser("factory-run", help="Run and record a full factory workflow")
    factory_run.add_argument("--seed", type=int, default=7)
    factory_run.add_argument("--orders", type=int, default=3)
    factory_run.add_argument("--planner", choices=("llm", "rules"), default="llm")
    factory_run.add_argument("--coordination", choices=("spacetime", "memory"), default="spacetime")
    factory_run.add_argument("--max-ticks", type=int, default=10000)
    factory_run.add_argument("--output", type=Path)
    factory_run.add_argument(
        "--policy", choices=("factory-vla", "model-based"), default="factory-vla"
    )
    factory_run.add_argument("--checkpoint", type=Path)
    factory_run.add_argument("--embeddings", choices=("on", "off"), default="on")
    factory_run.add_argument("--pose-jitter", type=float, default=0.02)
    factory_run.add_argument("--task", choices=("standard", "dispatch-routing"), default="standard")
    factory_run.add_argument(
        "--communication", choices=("embeddings", "text", "none"), default="embeddings"
    )
    cloth = commands.add_parser("cloth", help="Open a measured two-arm cloth replay")
    cloth.add_argument("--directory", type=Path, default=Path("examples/cloth-coordination"))
    cloth.add_argument("--port", type=int, default=8769)
    cloth_run = commands.add_parser("cloth-run", help="Run the two-arm deformable T-shirt task")
    cloth_run.add_argument("--seed", type=int, default=101)
    cloth_run.add_argument(
        "--communication",
        choices=("embeddings", "structured_text", "hybrid", "none"),
        default="structured_text",
    )
    cloth_run.add_argument(
        "--scenario", choices=("clear", "obstacles", "delayed", "blackout"), default="delayed"
    )
    cloth_run.add_argument("--coordination", choices=("memory", "spacetime"), default="spacetime")
    cloth_run.add_argument("--planner", choices=("rules", "llm"), default="llm")
    cloth_run.add_argument("--max-ticks", type=int, default=1800)
    cloth_run.add_argument("--no-video", action="store_true")
    cloth_run.add_argument("--output", type=Path)
    cloth_compare = commands.add_parser(
        "cloth-compare", help="Matched cloth communication experiments"
    )
    cloth_compare.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104])
    cloth_compare.add_argument(
        "--scenarios",
        nargs="+",
        choices=("clear", "obstacles", "delayed", "blackout"),
        default=["clear", "obstacles", "delayed"],
    )
    cloth_compare.add_argument("--coordination", choices=("memory", "spacetime"), default="memory")
    cloth_compare.add_argument("--planner", choices=("rules", "llm"), default="rules")
    cloth_compare.add_argument("--max-ticks", type=int, default=1800)
    cloth_compare.add_argument("--no-video", action="store_true")
    cloth_compare.add_argument("--output", type=Path)
    return result


def default_output(label: str) -> Path:
    return Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{label}"


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        if args.command == "cloth":
            from .cloth.server import serve_cloth

            serve_cloth(args.directory, args.port)
            return
        if args.command == "cloth-run":
            from .cloth.runtime import ClothConfig, run_cloth

            summary = run_cloth(
                ClothConfig(
                    seed=args.seed,
                    mode=args.communication,
                    scenario=args.scenario,
                    coordination=args.coordination,
                    planner=args.planner,
                    max_ticks=args.max_ticks,
                    video=not args.no_video,
                ),
                args.output or default_output("cloth"),
            )
            print(f"{summary['status'].upper()} · {summary['ticks']} ticks · {summary['metrics']}")
            if summary["error"]:
                print(summary["error"])
            return
        if args.command == "cloth-compare":
            from .cloth.benchmark import benchmark

            benchmark(
                output=args.output or default_output("cloth-comparison"),
                seeds=args.seeds,
                scenarios=args.scenarios,
                coordination=args.coordination,
                planner=args.planner,
                max_ticks=args.max_ticks,
                video=not args.no_video,
            )
            return
        if args.command == "factory":
            from .factory.server import serve_factory

            serve_factory(
                args.port, args.coordination, replay=args.replay, comparison=args.comparison
            )
            return
        if args.command == "factory-run":
            from .factory.runtime import run_factory

            summary = run_factory(
                seed=args.seed,
                orders=args.orders,
                planner=args.planner,
                coordination=args.coordination,
                max_ticks=args.max_ticks,
                output=args.output,
                policy=args.policy,
                checkpoint=args.checkpoint,
                embeddings=args.embeddings == "on",
                pose_jitter=args.pose_jitter,
                task=args.task,
                communication=args.communication,
            )
            print(
                f"{summary['status'].upper()} · {summary['mission']['completed']}/{args.orders} workpieces · {summary['tick']} ticks"
            )
            print(f"Logs: {summary['output']}")
            return
        if args.command == "serve":
            from .server import serve

            serve(args.directory, args.port)
            return
        if args.command == "task-status":
            import json

            from .spacetime import SpacetimeClient, SpacetimeConfig

            task = SpacetimeClient(SpacetimeConfig.from_env()).get_task(args.task_id)
            if task is None:
                raise ValueError("Task not found in SpacetimeDB")
            print(json.dumps(task, indent=2))
            return
        output = (args.output or default_output(args.command)).resolve()
        if args.command == "run":
            config = RunConfig(
                seed=args.seed,
                communication=args.comm == "on",
                max_steps=args.max_steps,
                message_latency=args.message_latency,
                policy=args.policy,
                video=not args.no_video,
                picker_url=args.picker_url,
                packer_url=args.packer_url,
                device=args.device,
                action_horizon=args.action_horizon,
                denoising_steps=args.denoising_steps,
                task_id=args.task_id,
                order=tuple(args.order) if args.order else None,
                feedback=args.feedback == "on",
                coordination=args.coordination,
                planner=args.planner,
                checkpoint=args.checkpoint,
                llm_python=args.llm_python,
            )
            summary = run_episode(config, output)
            print(f"{'SUCCESS' if summary['success'] else 'STEP LIMIT'} · {summary['steps']} steps")
            print(f"Order: {' → '.join(summary['order'])}")
            print(f"Replay: {output / 'replay.html'}")
            print(f"Logs:   {output}")
            return
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Comparison seeds must be unique")
        # Validate before creating any output folders.
        for seed in args.seeds:
            RunConfig(seed=seed, max_steps=args.max_steps, message_latency=args.message_latency)
        prepare_directory(output)
        episodes = []
        for seed in args.seeds:
            for enabled in (True, False):
                label = "on" if enabled else "off"
                summary = run_episode(
                    RunConfig(
                        seed=seed,
                        communication=enabled,
                        max_steps=args.max_steps,
                        message_latency=args.message_latency,
                        video=False,
                    ),
                    output / f"seed-{seed}-comm-{label}",
                )
                episodes.append(
                    {
                        key: summary[key]
                        for key in (
                            "seed",
                            "communication",
                            "success",
                            "steps",
                            "order",
                            "packed",
                            "messages",
                        )
                    }
                )
                print(
                    f"Seed {seed:3d} · comm {label:3s} · "
                    f"{'success' if summary['success'] else 'step limit'} · {summary['steps']} steps"
                )
        rates = {
            label: sum(e["success"] for e in episodes if e["communication"] == enabled)
            / len(args.seeds)
            for label, enabled in (("on", True), ("off", False))
        }
        write_json(
            output / "comparison.json",
            {
                "seeds": args.seeds,
                "max_steps": args.max_steps,
                "message_latency": args.message_latency,
                "success_rate": rates,
                "episodes": episodes,
                "interpretation": "Scripted private-order diagnostic only; not evidence of VLA quality "
                "or general multi-robot communication performance.",
            },
        )
        print(f"Success rate · on: {rates['on']:.0%} · off: {rates['off']:.0%}")
        print(f"Results: {output / 'comparison.json'}")
    except (ValueError, OSError, RuntimeError) as exc:
        argument_parser.exit(1, f"relay-lab: {exc}\n")


if __name__ == "__main__":
    main()
