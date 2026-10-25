"""Run the frozen foundation backbones with adapted *native* action projections.

This entry point never loads ClothPolicy. The common runtime still supplies
explicit phase/obstacle guards, Qwen review, message transport and physics.
Adapter bytes are snapshotted before model loading, including X-VLA's sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .runtime import ClothConfig, run_cloth


def _validate_options(checkpoint_paths, device, action_horizon, action_repeat):
    if set(checkpoint_paths) != {"left", "right"}:
        raise ValueError("Both native adapter paths are required")
    if device not in {"cpu", "mps"}:
        raise ValueError("Unsupported local native VLA device")
    if type(action_horizon) is not int or not 1 <= action_horizon <= 50:
        raise ValueError("Invalid native action refresh horizon")
    if (
        type(action_repeat) is not int or not 1 <= action_repeat <= 50
        or (action_repeat > 1 and action_horizon != 1)
    ):
        raise ValueError("Native target repetition requires horizon 1 and a bounded repeat count")


def run_smol_pair(
    config,
    output,
    *,
    checkpoint_paths,
    expected_checkpoint_hashes=None,
    device="mps",
    action_horizon=1,
    action_repeat=4,
    should_stop=None,
    on_progress=None,
):
    """Two independent SmolVLA instances with separately bound arm adaptations.

    Both execute the same pinned foundation architecture. Each adapter must
    identify its own arm; neither is relabelled or loaded as the other arm.
    The common runtime verifies and snapshots approved bytes before any world
    or model creation. A malformed adapter fails without a controller fallback.
    """
    _validate_options(checkpoint_paths, device, action_horizon, action_repeat)
    if should_stop is not None and should_stop():
        return run_cloth(config, output, should_stop=lambda: True, on_progress=on_progress)
    bound_hashes = dict(expected_checkpoint_hashes) if expected_checkpoint_hashes is not None else {
        arm: hashlib.sha256(Path(checkpoint_paths[arm]).read_bytes()).hexdigest()
        for arm in ("left", "right")
    }

    def factory(arm, snapshot):
        from .smol_adaptation import SmolNativePolicy

        return SmolNativePolicy(
            arm, snapshot, device=device, action_horizon=action_horizon,
            action_repeat=action_repeat,
            expected_sha256=bound_hashes[arm],
        )

    return run_cloth(
        config, output, checkpoint_paths=checkpoint_paths,
        expected_checkpoint_hashes=bound_hashes,
        policy_factory=factory, should_stop=should_stop, on_progress=on_progress,
    )


def run_native_cloth(
    config,
    output,
    *,
    checkpoint_paths,
    expected_checkpoint_hashes=None,
    device="mps",
    action_horizon=1,
    action_repeat=4,
    should_stop=None,
    on_progress=None,
):
    """Left checkpoint is a Smol adapter; right is X-VLA's bound JSON manifest.

    Hashing the right manifest binds its contract, data provenance, inference
    configuration and decoder hash to one manager-approved model revision.
    Only these two built-in model families can be constructed here.
    """
    _validate_options(checkpoint_paths, device, action_horizon, action_repeat)
    if should_stop is not None and should_stop():
        # Preserve the common runtime's cancellation evidence without loading
        # tensors, allocating a world, or touching an adapter sidecar.
        return run_cloth(config, output, should_stop=lambda: True, on_progress=on_progress)
    right_path = Path(checkpoint_paths["right"])
    left_hash = hashlib.sha256(Path(checkpoint_paths["left"]).read_bytes()).hexdigest()
    manifest_bytes = right_path.read_bytes()
    metadata = json.loads(manifest_bytes)
    if expected_checkpoint_hashes is not None and hashlib.sha256(manifest_bytes).hexdigest() != expected_checkpoint_hashes.get("right"):
        raise ValueError("Approved X-VLA manifest SHA-256 does not match its bytes")
    weights = (right_path.parent / "native_decoder.safetensors").read_bytes()
    if hashlib.sha256(weights).hexdigest() != metadata.get("decoder_sha256"):
        raise ValueError("X-VLA decoder does not match the approved manifest")
    bound_hashes = expected_checkpoint_hashes if expected_checkpoint_hashes is not None else {
        "left": left_hash,
        "right": hashlib.sha256(manifest_bytes).hexdigest(),
    }

    def factory(arm, snapshot):
        if arm == "left":
            from .smol_adaptation import SmolNativePolicy

            return SmolNativePolicy(
                arm, snapshot, device=device, action_horizon=action_horizon,
                expected_sha256=left_hash, action_repeat=action_repeat,
            )
        from .xvla_adaptation import NativeXVLA

        # The common runtime captured both source files before creating a world.
        # Reject even a valid-but-different manifest swapped in between reads.
        if snapshot.read_bytes() != manifest_bytes:
            raise ValueError("X-VLA manifest changed during snapshot creation")
        package = snapshot.parent / "xvla"
        package.mkdir()
        (package / "adapter.json").write_bytes(manifest_bytes)
        (package / "native_decoder.safetensors").write_bytes(weights)
        return NativeXVLA(
            adapter=package, expected_sha256=metadata["decoder_sha256"],
            device=device, action_horizon=max(action_horizon, action_repeat),
        )

    return run_cloth(
        config, output, checkpoint_paths=checkpoint_paths,
        expected_checkpoint_hashes=bound_hashes,
        policy_factory=factory, should_stop=should_stop, on_progress=on_progress,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smol-adapter", type=Path, required=True)
    right = parser.add_mutually_exclusive_group(required=True)
    right.add_argument("--xvla-adapter", type=Path, help="Directory containing adapter.json")
    right.add_argument("--right-smol-adapter", type=Path, help="Separately adapted right-arm SmolVLA weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=4, help="Hold native token 0 for this many ticks; requires --horizon 1")
    parser.add_argument("--planner", choices=("rules", "llm"), default="llm")
    parser.add_argument("--coordination", choices=("memory", "spacetime"), default="spacetime")
    parser.add_argument("--max-ticks", type=int, default=1800)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    config = ClothConfig(
        seed=args.seed, mode="structured_text", scenario="delayed",
        coordination=args.coordination, planner=args.planner,
        max_ticks=args.max_ticks, video=not args.no_video,
    )
    runner = run_smol_pair if args.right_smol_adapter else run_native_cloth
    result = runner(
        config, args.output,
        checkpoint_paths={
            "left": args.smol_adapter,
            "right": args.right_smol_adapter or args.xvla_adapter / "adapter.json",
        },
        device=args.device, action_horizon=args.horizon, action_repeat=args.repeat,
        on_progress=lambda value: print(json.dumps(value, allow_nan=False), flush=True),
    )
    print(json.dumps(result, allow_nan=False), flush=True)
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
