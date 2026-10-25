"""The single reviewed local simulation recipe; safe to import without ML packages."""

RECIPE_ID = "two-smolvla-cloth-v1"
# Native adapter provenance can exceed 256 KiB; keep receipt/journal readers
# aligned while retaining a finite bound for local persisted worker state.
MAX_STATE_BYTES = 2 * 1024**2
PROVENANCE = (
    "Two independent SmolVLA instances, with separate arm adaptations of their native action "
    "networks. Qwen phase review; SpacetimeDB coordination. Training-scene development "
    "demonstration (seed 10007), not a held-out benchmark or hardware validation."
)
OBJECTIVE = (
    "Fold the T-shirt with both simulated arms. Share local observations, wait for "
    "fresh peer reports, and hold when either arm observes an obstacle. Release "
    "the garment and vacuum fixture, then verify the final cloth geometry."
)
SUCCESS_CRITERIA = [
    "Both grippers and the vacuum fixture are released; cloth is stable for at least 15 ticks.",
    "Hem error is below 0.095 m, sleeve overhang below 0.06 m, and height below 0.12 m.",
    "Footprint coverage is at least 0.40, containment at least 0.75, and area at least 0.075 m².",
    "Fresh-world action replay reproduces the final cloth vertices and measured metrics.",
    "Both pinned SmolVLA native action networks execute; recorded commands pass the native audit.",
]
CHECKPOINT_PATHS = {
    arm: f"checkpoints/cloth-native/{arm}.pt" for arm in ("left", "right")
}
PINNED_HASHES = {
    "left": "ad47dc42e1caf24477ab6a8551156ce6d08c3f8782c29ee7cb81e25006674371",
    "right": "c72f368f1e25365659017649ff5afbadc0aa08caf8004669a3ae83b8b19f3e02",
}
CODEC_PATH = "checkpoints/cloth-policies/communication.json"
CODEC_HASH = "082ef11b9753f48b58bd1ecc21b92e76667c7481e7b36ceea013bcda37781dce"


def recipe_workspace():
    """Drafts require the normal explicit manager release before they can execute."""
    return {
        "version": 1,
        "name": "Native SmolVLA cloth workcell",
        "objective": OBJECTIVE,
        "mode": "mediated",
        "direction": "both",
        "agents": [
            {
                "id": arm,
                "name": f"SmolVLA · {arm} arm",
                "robot": "arm",
                "family": "smolvla",
                "source": {
                    "kind": "local",
                    "ref": CHECKPOINT_PATHS[arm],
                    "revision": PINNED_HASHES[arm],
                },
                "rig": "",
                "action_contract": (
                    "Pinned pretrained SmolVLA with arm-specific native network adaptation; "
                    "own RGB + measured pose + phase to absolute XYZ (m) + grasp. MuJoCo simulation only."
                ),
            }
            for arm in ("left", "right")
        ],
    }


def recipe_config():
    return {
        "seed": 10007,
        "mode": "structured_text",
        "scenario": "delayed",
        "coordination": "spacetime",
        "planner": "llm",
        "max_ticks": 1800,
        "video": True,
        "codec": CODEC_PATH,
    }


def native_model_assets(families=("smolvla", "xvla")):
    """Pinned offline inputs, without importing Torch or constructing a model."""
    from relay_lab.cloth.pretrained import BART_ID, BART_REVISION, XVLA_ID, XVLA_REVISION
    from relay_lab.smolvla import (
        DEFAULT_MODEL_ID,
        DEFAULT_REVISION,
        VLM_MODEL_ID,
        VLM_REVISION,
    )

    if not set(families) <= {"smolvla", "xvla"}:
        raise ValueError("Unknown native model family")
    assets = (
        (DEFAULT_MODEL_ID, DEFAULT_REVISION, (
            "config.json", "model.safetensors", "policy_preprocessor.json",
            "policy_postprocessor.json", "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
        )),
        (VLM_MODEL_ID, VLM_REVISION, (
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "preprocessor_config.json", "processor_config.json",
        )),
        (XVLA_ID, XVLA_REVISION, (
            "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
        )),
        (BART_ID, BART_REVISION, ("config.json", "tokenizer.json", "tokenizer_config.json")),
    )
    return tuple(
        asset for family, asset in zip(("smolvla", "smolvla", "xvla", "xvla"), assets)
        if family in families
    )
