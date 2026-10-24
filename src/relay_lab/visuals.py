"""Keep the original Cartesian controls attached to their visual gantry hardware.

Every affected body/geom is collision-free. This changes only how the same
end-effector poses are drawn, not action integration or task physics.
"""

from .types import AGENTS

RAIL_HEIGHT = 0.7


def sync_gantry(model, data) -> None:
    """Update visual-only bridges, carriages, and telescoping Z columns.

    Call before ``mj_forward`` when preparing a camera render. The motion model
    remains the existing Cartesian end effector; these are visual attachments.
    """
    for agent in AGENTS:
        ee = data.mocap_pos[int(model.body(f"{agent}_gripper").mocapid[0])]
        for part, position in (
            ("bridge", (ee[0], 0, RAIL_HEIGHT)),
            ("carriage", (ee[0], ee[1], RAIL_HEIGHT)),
        ):
            data.mocap_pos[int(model.body(f"{agent}_{part}").mocapid[0])] = position
        bottom, top = float(ee[2]) + 0.12, RAIL_HEIGHT - 0.035
        column = int(model.body(f"{agent}_column").mocapid[0])
        data.mocap_pos[column] = (ee[0], ee[1], (bottom + top) / 2)
        model.geom(f"{agent}_column_geom").size[2] = max(0.001, (top - bottom) / 2)
