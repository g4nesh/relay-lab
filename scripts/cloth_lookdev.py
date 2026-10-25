"""Render look-development stills of the cloth workcell at key moments of a fold.

Drives both arms with the geometric teaching targets (``reference_target``) so
the frames show a representative fold; this is a rendering aid, not a policy
evaluation. Writes PNGs for the presentation and arm cameras.

    PYTHONPATH=src .venv/bin/python scripts/cloth_lookdev.py --output runs/lookdev
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from relay_lab.cloth.world import AGENTS, ClothWorld

MOMENTS = {
    "carry_sleeve": "sleeve-lift",
    "release_sleeve": "sleeves-folded",
    "carry_hem": "hem-lift",
    "settle": "folded",
}


def drive(world, phase, max_ticks=45, on_tick=None):
    for _ in range(max_ticks):
        actions = {a: world.reference_target(a, phase) for a in AGENTS}
        world.step(actions)
        if on_tick:
            on_tick()
        if all(
            np.linalg.norm(world.ee(a) - np.asarray(actions[a]["target"])) < 0.012 for a in AGENTS
        ):
            break


def save(world, output, tag, size, arm_size):
    started = time.perf_counter()
    for camera in ("overview", "top"):
        Image.fromarray(world.render(camera, width=size[0], height=size[1])).save(
            output / f"{tag}-{camera}.png"
        )
    for a in AGENTS:
        Image.fromarray(world.render(a, width=arm_size, height=arm_size)).save(
            output / f"{tag}-{a}.png"
        )
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--video", action="store_true", help="also write overview.mp4 (every 2nd tick)"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        import imageio.v2 as iio

        writer = iio.get_writer(
            args.output / "overview.mp4", fps=12.5, quality=9, macro_block_size=16
        )

    def record():
        if writer is not None and world.tick % 2 == 0:
            writer.append_data(world.render("overview", width=args.width, height=args.height))

    with ClothWorld(seed=args.seed, obstacle_schedule=((0, 10_000, "left"),)) as world:
        world.step({})
        seconds = [save(world, args.output, "00-start", (args.width, args.height), 384)]
        for index, phase in enumerate(ClothWorld.PHASES, start=1):
            if phase == "approach_hem_above" and world.fixture_active:
                world.release_fixture()
            drive(world, phase, on_tick=record)
            if phase in MOMENTS:
                tag = f"{index:02d}-{MOMENTS[phase]}"
                seconds.append(save(world, args.output, tag, (args.width, args.height), 384))
        print(f"tick {world.tick}; mean render time per moment {np.mean(seconds):.2f} s")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
