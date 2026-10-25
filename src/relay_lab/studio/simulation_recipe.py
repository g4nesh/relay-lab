"""The single reviewed local simulation recipe; safe to import without ML packages."""

RECIPE_ID = "two-arm-cloth"
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
]
CHECKPOINT_PATHS = {
    arm: f"checkpoints/cloth-policies/{arm}.pt" for arm in ("left", "right")
}
PINNED_HASHES = {
    "left": "33d38794b1aea9f89bf622262d5482c405609a537350cc982b2c7c230737d709",
    "right": "1dfee3e300a46e9007bed395b4b2bac0102a1d37ec230cc0bd28d41bfc58555d",
}
CODEC_PATH = "checkpoints/cloth-policies/communication.json"
CODEC_HASH = "082ef11b9753f48b58bd1ecc21b92e76667c7481e7b36ceea013bcda37781dce"


def recipe_workspace():
    """Drafts require the normal explicit manager release before they can execute."""
    return {
        "version": 1,
        "name": "Two-arm cloth simulation",
        "objective": OBJECTIVE,
        "mode": "mediated",
        "direction": "both",
        "agents": [
            {
                "id": arm,
                "name": f"{arm.title()} cloth controller",
                "robot": "arm",
                "family": "custom",
                "source": {
                    "kind": "local",
                    "ref": CHECKPOINT_PATHS[arm],
                    "revision": PINNED_HASHES[arm],
                },
                "rig": "",
                "action_contract": (
                    "Task-trained state-assisted cloth controller; local RGB + landmarks + phase "
                    "to Cartesian target xyz (m) + grasp. MuJoCo simulation only."
                ),
            }
            for arm in ("left", "right")
        ],
    }


def recipe_config():
    return {
        "seed": 103,
        "mode": "structured_text",
        "scenario": "delayed",
        "coordination": "spacetime",
        "planner": "llm",
        "max_ticks": 1800,
        "video": True,
        "codec": CODEC_PATH,
    }
