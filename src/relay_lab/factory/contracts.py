"""Shared factory vocabulary. World observations, never model prose, establish outcomes."""
from __future__ import annotations

ROBOT_IDS = ("humanoid-01", "amr-01", "amr-02", "arm-01", "gantry-01")
ROBOT_TYPES = {
    "humanoid-01": "humanoid",
    "amr-01": "amr",
    "amr-02": "amr",
    "arm-01": "arm",
    "gantry-01": "gantry",
}
STAGES = (
    "queued", "loaded", "at_cell", "in_fixture", "processed",
    "outbound_loaded", "at_dispatch", "completed",
)
CAPABILITIES = {
    "humanoid-01": ("transfer", "wait"),
    "amr-01": ("navigate", "wait"),
    "amr-02": ("navigate", "wait"),
    "arm-01": ("transfer", "process", "wait"),
    "gantry-01": ("transfer", "wait"),
}
MESSAGE_KINDS = ("intent", "belief", "request", "ack", "outcome", "blocked")
DT = 0.1
