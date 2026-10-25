"""Original metric factory layout shared by simulation and the renderer.

All poses use metres and Z-up. Robot shapes are original, simplified industrial
forms, not replicas of a manufacturer's hardware. Kinematic embodiments are
explicitly described in the scene contract; only released payloads have MuJoCo
rigid-body contact dynamics in this extension.
"""

from __future__ import annotations

from copy import deepcopy

CARGO_SIZE = (0.42, 0.32, 0.24)
CARGO_HALF_HEIGHT = CARGO_SIZE[2] / 2
FLOOR_SIZE = (24.0, 14.0)

# Positions are support/cargo-centre poses. Docks include their mobile deck's
# 0.50 m support height, so objects stay at the same height through transfer.
STATIONS = {
    "receive": {
        "id": "receive",
        "kind": "receiving",
        "label": "Inbound inspection",
        "position": [-8.0, 2.0, 1.02],
        "size": [1.3, 2.4, 0.9],
    },
    "load": {
        "id": "load",
        "kind": "dock",
        "label": "Load dock · A",
        "position": [-6.0, 2.0, 0.62],
        "size": [1.25, 1.3, 0.04],
    },
    "process-in": {
        "id": "process-in",
        "kind": "dock",
        "label": "Machine input",
        "position": [-1.8, 2.0, 0.62],
        "size": [1.25, 1.3, 0.04],
    },
    "fixture": {
        "id": "fixture",
        "kind": "fixture",
        "label": "Precision assembly",
        "position": [0.0, 2.0, 1.02],
        "size": [1.2, 1.15, 0.9],
    },
    "process-out": {
        "id": "process-out",
        "kind": "dock",
        "label": "Machine output",
        "position": [1.8, 2.0, 0.62],
        "size": [1.25, 1.3, 0.04],
    },
    "dispatch": {
        "id": "dispatch",
        "kind": "dock",
        "label": "Dispatch dock · B",
        "position": [6.0, 2.0, 0.62],
        "size": [1.25, 1.3, 0.04],
    },
    "pallet": {
        "id": "pallet",
        "kind": "pallet",
        "label": "Outbound pallet",
        "position": [7.8, 2.0, 0.32],
        "size": [1.6, 2.4, 0.2],
    },
}

ROBOT_SPECS = {
    "humanoid-01": {
        "id": "humanoid-01",
        "kind": "humanoid",
        "label": "H01 · Loader",
        "position": [-7.0, 2.0, 0.0],
        "yaw": 0.0,
        "ee_position": [-7.0, 2.0, 1.45],
        "max_speed": 0.7,
        "max_acceleration": 1.2,
        "reach": 1.7,
        "radius": 0.34,
        "home": "humanoid-home",
        "capabilities": ["pick", "place", "transfer"],
        "controller": "bounded Cartesian manipulation + analytic upper-body IK; fixed stance",
    },
    "amr-01": {
        "id": "amr-01",
        "kind": "amr",
        "label": "M01 · Inbound carrier",
        "position": [-6.0, -1.5, 0.0],
        "yaw": 1.57079632679,
        "ee_position": [-6.0, -1.5, 0.62],
        "max_speed": 1.15,
        "max_acceleration": 0.65,
        "reach": 0.0,
        "radius": 0.55,
        "home": "amr-01-home",
        "capabilities": ["navigate"],
        "controller": "acceleration-limited navigation with swept-circle collision checks",
    },
    "arm-01": {
        "id": "arm-01",
        "kind": "arm",
        "label": "A01 · Assembly arm",
        "position": [0.0, 3.35, 0.0],
        "yaw": -1.57079632679,
        "ee_position": [0.0, 2.8, 2.0],
        "max_speed": 0.85,
        "max_acceleration": 1.5,
        "reach": 2.7,
        "radius": 0.4,
        "home": "arm-home",
        "capabilities": ["pick", "place", "transfer", "process"],
        "controller": "analytic two-link arm IK + bounded Cartesian trajectory",
    },
    "amr-02": {
        "id": "amr-02",
        "kind": "amr",
        "label": "M02 · Outbound carrier",
        "position": [6.0, -1.5, 0.0],
        "yaw": 1.57079632679,
        "ee_position": [6.0, -1.5, 0.62],
        "max_speed": 1.15,
        "max_acceleration": 0.65,
        "reach": 0.0,
        "radius": 0.55,
        "home": "amr-02-home",
        "capabilities": ["navigate"],
        "controller": "acceleration-limited navigation with swept-circle collision checks",
    },
    "gantry-01": {
        "id": "gantry-01",
        "kind": "gantry",
        "label": "G01 · Pallet grappler",
        "position": [6.9, 2.0, 3.8],
        "yaw": 0.0,
        "ee_position": [6.9, 2.0, 2.7],
        "max_speed": 0.95,
        "max_acceleration": 1.25,
        "reach": 3.6,
        "radius": 0.25,
        "home": "gantry-home",
        "capabilities": ["pick", "place", "transfer"],
        "controller": "bounded XYZ bridge and hoist trajectory; idealized grapple",
    },
}

OBSTACLES = [
    {
        "id": "receiving-table",
        "kind": "table",
        "center": [-8.0, 2.0],
        "half_size": [0.65, 1.2],
        "height": 0.9,
    },
    {
        "id": "assembly-table",
        "kind": "machine",
        "center": [0.0, 2.0],
        "half_size": [0.6, 0.575],
        "height": 0.9,
    },
    {
        "id": "assembly-base",
        "kind": "pedestal",
        "center": [0.0, 3.35],
        "half_size": [0.5, 0.5],
        "height": 0.55,
    },
    {
        "id": "outbound-pallet",
        "kind": "pallet",
        "center": [7.8, 2.0],
        "half_size": [0.8, 1.2],
        "height": 0.2,
    },
    {
        "id": "rack-west",
        "kind": "rack",
        "center": [-7.5, 5.7],
        "half_size": [2.4, 0.65],
        "height": 3.7,
    },
    {
        "id": "utilities",
        "kind": "cabinet",
        "center": [0.0, 5.7],
        "half_size": [2.2, 0.6],
        "height": 2.0,
    },
    {
        "id": "rack-east",
        "kind": "rack",
        "center": [7.5, 5.7],
        "half_size": [2.4, 0.65],
        "height": 3.7,
    },
]

NAVIGATION = {
    "spine_y": -1.5,
    "lane_width": 1.8,
    "docks": ["load", "process-in", "process-out", "dispatch"],
    "routes": [
        {"id": "aisle-main", "points": [[-9.5, -1.5, 0.015], [9.5, -1.5, 0.015]], "exclusive": True}
    ],
}


def scene_spec() -> dict:
    """Return an independent JSON-serializable scene description."""
    return deepcopy(
        {
            "name": "Relay Works · Pilot Factory",
            "units": "metres",
            "up_axis": "z",
            "floor_size": list(FLOOR_SIZE),
            "cargo_size": list(CARGO_SIZE),
            "stations": list(STATIONS.values()),
            "robots": list(ROBOT_SPECS.values()),
            "obstacles": OBSTACLES,
            "navigation": NAVIGATION,
            "materials": {
                "floor": "#d4d9dc",
                "shell": "#eef1f2",
                "steel": "#697985",
                "graphite": "#24323d",
                "inbound": "#047f83",
                "outbound": "#cb771a",
            },
            "physics": {
                "engine": "MuJoCo",
                "released_cargo": "rigid-body gravity and contact",
                "robots": "model-based kinematic embodiments",
                "grasp": "idealized attachment after measured proximity",
                "humanoid": "fixed stance; upper-body IK, no balance or locomotion policy",
                "learned_control": False,
            },
        }
    )


def cargo_start(index: int) -> tuple[float, float, float]:
    """Six inbound fixtures spaced to prevent payload overlap."""
    if not 0 <= index < 6:
        raise ValueError("factory pilot accepts one to six workpieces")
    return (-8.0 + (-0.28 if index >= 3 else 0.28), 1.3 + 0.7 * (index % 3), 1.02)


def pallet_slot(index: int) -> tuple[float, float, float]:
    if not 0 <= index < 6:
        raise ValueError("pallet has six non-overlapping slots")
    return (7.8 + (-0.28 if index >= 3 else 0.28), 1.3 + 0.7 * (index % 3), 0.32)


def mujoco_xml(cargo_ids: list[str]) -> str:
    """Create the contact world. Robot deck beds follow bounded AMR motion.

    Manipulator links are visualized from the same IK joints in the client.
    They deliberately have no unvalidated contact/force control in this pilot.
    """
    import html

    static = []
    for obstacle in OBSTACLES:
        x, y = obstacle["center"]
        sx, sy = obstacle["half_size"]
        h = obstacle["height"]
        static.append(
            f'<geom name="{obstacle["id"]}" type="box" pos="{x} {y} {h / 2}" size="{sx} {sy} {h / 2}" rgba="0.7 0.73 0.75 1"/>'
        )
    decks = []
    for robot in ROBOT_SPECS.values():
        if robot["kind"] == "amr":
            x, y, _ = robot["position"]
            decks.append(
                f'<body name="{robot["id"]}-deck" mocap="true" pos="{x} {y} 0.46"><geom type="box" size="0.46 0.38 0.04" friction="1.2 0.01 0.001"/></body>'
            )
    payloads = []
    for index, cargo_id in enumerate(cargo_ids):
        safe_id = html.escape(cargo_id, quote=True)
        x, y, z = cargo_start(index)
        payloads.append(
            f'<body name="{safe_id}" pos="{x} {y} {z}"><freejoint name="{safe_id}-joint"/><geom name="{safe_id}-geom" type="box" size="0.21 0.16 0.12" mass="2.5" friction="1.2 0.01 0.001" rgba="0.15 0.55 0.60 1"/></body>'
        )
    return f"""<mujoco model="Relay Works contact world"><option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/><default><geom solref="0.01 1" solimp="0.95 0.99 0.001"/></default><worldbody><geom name="floor" type="plane" size="12 7 0.1"/><light pos="0 0 10"/>{"".join(static)}{"".join(decks)}{"".join(payloads)}</worldbody></mujoco>"""
