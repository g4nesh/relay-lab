"""Record bounded, actual released-model inference and physical execution.

This is an integration probe, not a full-task zero-shot success benchmark. Each
model runs separately to fit local memory, with the untested other arm holding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .pretrained import SmolVLAClothProbe, XVLAClothProbe
from .world import ClothWorld


def run(model: str, output: Path, calls: int = 4, device: str = "cpu", seed: int = 77):
    import imageio.v3 as iio

    output.mkdir(parents=True, exist_ok=True)
    agent = "left" if model == "smolvla" else "right"
    report = {
        "model": model,
        "agent": agent,
        "seed": seed,
        "probe": "bounded unadapted model-to-simulator integration",
        "requested_calls": calls,
        "full_task_evaluation": False,
        "scripted_fallback": False,
        "other_arm": "holds its initial pose",
        "status": "starting",
    }
    path = output / f"{model}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    started = perf_counter()
    try:
        with ClothWorld(seed=seed, disturbances=False) as world:
            initial = world.observe(agent, rgb=True)
            iio.imwrite(output / f"{model}-input.png", initial["rgb"])
            policy = (
                SmolVLAClothProbe(agent, device, seed)
                if model == "smolvla"
                else XVLAClothProbe(agent, device, seed)
            )
            report["model_info"] = policy.model_info
            report["actions"] = []
            for index in range(calls):
                observation = initial if index == 0 else world.observe(agent, rgb=True)
                t = perf_counter()
                action = policy.act(
                    observation,
                    "Grasp the bottom corner of the blue T-shirt and fold it toward its shoulder.",
                )
                world.step({agent: action})
                report["actions"].append(
                    {
                        "call": index,
                        "tick": world.tick,
                        "seconds": perf_counter() - t,
                        "action": action,
                        "prediction": policy.last_prediction,
                        "measured_ee": world.ee(agent).tolist(),
                        "grasped": world.holding[agent] is not None,
                    }
                )
                report["model_info"] = policy.model_info
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "stage": "pretrained_probe",
                            "model": model,
                            "call": index + 1,
                            "action": action,
                            "seconds": round(perf_counter() - t, 3),
                        }
                    ),
                    flush=True,
                )
            report.update(
                {
                    "status": "inference_and_execution_confirmed",
                    "metrics": world.metrics(),
                    "snapshot": world.snapshot(),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            iio.imwrite(
                output / f"{model}-final.png", world.render("overview", width=960, height=720)
            )
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": perf_counter() - started,
            }
        )
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("smolvla", "xvla"))
    parser.add_argument("--output", type=Path, default=Path("runs/cloth-pretrained-probes"))
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("calls must be positive")
    run(args.model, args.output, args.calls, args.device)


if __name__ == "__main__":
    main()
