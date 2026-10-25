"""Original, procedural dual-arm textile workcell; no third-party meshes.

The garment is a single-layer triangulated T-shirt silhouette. It is not a sewn
front/back garment and its material constants are illustrative, not calibrated.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

TABLE_Z = 0.0
AGENTS = ("left", "right")


def shirt_mesh():
    """Triangulate a T silhouette including a shallow open neckline."""
    xs = np.linspace(-0.44, 0.44, 23)
    ys = np.linspace(-0.32, 0.32, 17)
    points, faces, lookup = [], [], {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            in_body = abs(x) <= 0.2401
            in_sleeve = 0.08 - 1e-5 <= y <= 0.28 + 1e-5
            neck = y > 0.24 + 1e-5 and abs(x) < 0.08001
            if (in_body or in_sleeve) and not neck:
                lookup[i, j] = len(points)
                points.append((x, y, 0.008))
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            p = [
                lookup.get((i, j)),
                lookup.get((i + 1, j)),
                lookup.get((i, j + 1)),
                lookup.get((i + 1, j + 1)),
            ]
            if all(k is not None for k in p):
                a, b, c, d = p
                faces.extend(((a, b, d), (a, d, c)) if (i + j) % 2 else ((a, b, c), (b, d, c)))
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    landmarks = {}
    for agent, sign in (("left", -1), ("right", 1)):
        landmarks[agent] = {}
        for name, xy in {
            "sleeve": (sign * 0.44, 0.16),
            "hem": (sign * 0.24, -0.32),
            "shoulder": (sign * 0.24, 0.28),
        }.items():
            landmarks[agent][name] = int(np.argmin(np.linalg.norm(points[:, :2] - xy, axis=1)))
    return points, faces, landmarks


@lru_cache(maxsize=1)
def cotton_texture() -> bytes:
    """Original woven cotton and seam texture generated deterministically."""
    rng = np.random.default_rng(1987)
    yy, xx = np.indices((1024, 1024))
    weave = 1.8 * ((xx % 4 < 2).astype(float) - 0.5) + 1.5 * ((yy % 4 < 2).astype(float) - 0.5)
    noise = rng.normal(0, 1.2, (1024, 1024))
    rgb = np.clip(
        np.array([45.0, 84.0, 119.0])[None, None, :] + (weave + noise)[..., None], 0, 255
    ).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    vertices, triangles, _ = shirt_mesh()
    edges = {}
    for a, b, c in triangles:
        for i, j in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(i), int(j))))
            edges[edge] = edges.get(edge, 0) + 1
    uv = np.column_stack(((vertices[:, 0] + 0.44) / 0.88, (vertices[:, 1] + 0.32) / 0.64)) * 1023
    for (a, b), count in edges.items():
        if count != 1:
            continue
        start, end = uv[a], uv[b]
        draw.line([tuple(start), tuple(end)], fill=(57, 96, 130), width=9)
        length = float(np.linalg.norm(end - start))
        for distance in np.arange(0, length, 9):
            p = start + (end - start) * distance / max(length, 1)
            q = start + (end - start) * min(length, distance + 3) / max(length, 1)
            draw.line([tuple(p), tuple(q)], fill=(88, 119, 146), width=1)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _arm(name: str, sign: int) -> str:
    """Six revolute joints, dynamic motors, original capsule/shell geometry."""
    base_x = sign * 0.76
    yaw = 0 if sign == -1 else 180
    accent = "teal" if sign == -1 else "amber"
    return f'''
    <body name="{name}_base" pos="{base_x} 0.10 0.025" euler="0 0 {yaw}">
      <geom type="cylinder" size="0.10 0.025" material="steel"/>
      <geom type="cylinder" size="0.076 0.055" pos="0 0 .07" material="shell"/>
      <geom class="visual" type="cylinder" size="0.078 .006" pos="0 0 .12" material="{accent}"/>
      <body name="{name}_yaw" pos="0 0 .14">
        <joint name="{name}_j0" axis="0 0 1" range="-160 160"/>
        <geom type="cylinder" size=".063 .045" material="graphite"/>
        <body name="{name}_upper" pos="0 0 .06">
          <joint name="{name}_j1" axis="0 1 0" range="-150 100"/>
          <geom type="cylinder" size=".066 .065" euler="90 0 0" material="shell"/>
          <geom class="visual" type="cylinder" size=".044 .067" euler="90 0 0" material="{accent}"/>
          <geom type="capsule" fromto=".025 0 0 .42 0 0" size=".045" material="shell"/>
          <geom class="visual" type="box" size=".14 .038 .008" pos=".23 0 .043" material="aluminum"/>
          <geom class="visual" type="capsule" fromto=".09 -.046 0 .35 -.046 0" size=".007" material="graphite"/>
          <body name="{name}_forearm" pos=".45 0 0">
            <joint name="{name}_j2" axis="0 1 0" range="0 160"/>
            <geom type="cylinder" size=".058 .057" euler="90 0 0" material="shell"/>
            <geom class="visual" type="cylinder" size=".038 .06" euler="90 0 0" material="{accent}"/>
            <geom type="capsule" fromto=".02 0 0 .37 0 0" size=".033" material="shell"/>
            <geom class="visual" type="capsule" fromto=".07 -.036 0 .33 -.036 0" size=".005" material="graphite"/>
            <body name="{name}_pitch" pos=".40 0 0">
              <joint name="{name}_j3" axis="0 1 0" range="-160 160"/>
              <geom type="cylinder" size=".042 .042" euler="90 0 0" material="graphite"/>
              <body name="{name}_roll">
                <joint name="{name}_j4" axis="1 0 0" range="-180 180"/>
                <geom type="cylinder" size=".036 .026" pos="0 0 -.028" material="shell"/>
                <body name="{name}_tool" pos="0 0 -.065">
                  <joint name="{name}_j5" axis="0 0 1" range="-180 180"/>
                  <geom type="box" size=".042 .028 .021" pos="0 0 .005" material="graphite"/>
                  <geom class="visual" type="box" size=".043 .029 .007" pos="0 0 .019" material="{accent}"/>
                  <geom name="{name}_finger_l" type="box" size=".007 .014 .022" pos="-.020 0 -.027" material="steel"/>
                  <geom name="{name}_finger_r" type="box" size=".007 .014 .022" pos=".020 0 -.027" material="steel"/>
                  <geom class="visual" type="box" size=".009 .017 .006" pos="-.020 0 -.048" material="rubber"/>
                  <geom class="visual" type="box" size=".009 .017 .006" pos=".020 0 -.048" material="rubber"/>
                  <site name="{name}_ee" pos="0 0 -.052" size=".004" rgba="0 0 0 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>'''


def build_scene() -> tuple[str, np.ndarray, np.ndarray, dict]:
    points, faces, landmarks = shirt_mesh()
    point_s = " ".join(f"{v:.6f}" for v in points.ravel())
    face_s = " ".join(map(str, faces.ravel()))
    uv = np.column_stack(((points[:, 0] + 0.44) / 0.88, (points[:, 1] + 0.32) / 0.64))
    uv_s = " ".join(f"{v:.6f}" for v in uv.ravel())
    arms = _arm("left", -1) + _arm("right", 1)
    actuators = "".join(
        f'<position name="{a}_motor_{j}" joint="{a}_j{j}" kp="450" kv="35" forcerange="-100 100"/>'
        for a in AGENTS
        for j in range(6)
    )
    markers = "".join(
        f'<body name="marker_{a}_{n}" mocap="true"><geom type="sphere" size=".006" rgba="{color}" contype="0" conaffinity="0"/></body>'
        for a in AGENTS
        for n, color in (
            ("sleeve", ".95 .62 .14 1"),
            ("hem", ".15 .78 .69 1"),
            ("shoulder", ".92 .94 .98 1"),
        )
    )
    fixture_points = [(x, y) for x in (-0.20, 0.0, 0.20) for y in (-0.04, 0.24)]
    fixture_indices = [
        int(np.argmin(np.linalg.norm(points[:, :2] - p, axis=1))) for p in fixture_points
    ]
    fixture_equalities = "".join(
        f'<connect name="vacuum_{j}" body1="shirt_{i}" anchor="0 0 0" active="false" solref=".008 1" solimp=".98 .999 .0001"/>'
        for j, i in enumerate(fixture_indices)
    )
    fixture_visuals = "".join(
        f'<geom name="vacuum_pad_{j}" class="visual" type="cylinder" size=".014 .0005" pos="{x} {y} .0006" material="graphite"/><geom class="visual" type="cylinder" size=".008 .0006" pos="{x} {y} .0007" material="teal"/>'
        for j, (x, y) in enumerate(fixture_points)
    )
    xml = f'''<mujoco model="Relay Lab Textile Cell">
      <compiler angle="degree" autolimits="true"/>
      <option timestep=".004" gravity="0 0 -9.81" integrator="discrete" solver="Newton" iterations="30" tolerance="1e-7" jacobian="sparse"/>
      <size memory="256M"/><statistic center="0 0 .1" extent="2.2"/>
      <visual><global offwidth="1600" offheight="1200"/><quality shadowsize="8192" offsamples="8" numslices="32" numstacks="16"/><headlight ambient=".30 .32 .34" diffuse=".14 .14 .14" specular=".08 .08 .08"/><map znear=".005" shadowclip="1.4"/><rgba haze=".88 .9 .93 1"/></visual>
      <default><joint damping="2" armature=".04"/><geom friction=".65 .008 .001" solref=".012 1" solimp=".95 .99 .001" contype="2" conaffinity="1" density="700"/><default class="visual"><geom contype="0" conaffinity="0"/></default><default class="cloth_node"><joint damping="0" armature="0"/></default></default>
      <asset>
        <texture name="sky" type="skybox" builtin="gradient" rgb1=".65 .74 .81" rgb2=".94 .96 .98" width="256" height="1536"/>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1=".28 .32 .36" rgb2=".30 .34 .38" width="512" height="512"/>
        <texture name="weave" type="2d" file="cotton.png"/>
        <material name="floor" texture="floor_tex" texrepeat="4 4" reflectance=".08" specular=".2" shininess=".3"/>
        <material name="cloth" texture="weave" texrepeat="1 1" specular=".08" shininess=".12" rgba="1 1 1 1"/>
        <material name="table" rgba=".60 .65 .66 1" specular=".16" shininess=".4"/>
        <material name="shell" rgba=".91 .94 .95 1" specular=".45" shininess=".65"/>
        <material name="aluminum" rgba=".57 .63 .68 1" specular=".8" shininess=".8"/>
        <material name="steel" rgba=".25 .30 .35 1" specular=".65" shininess=".65"/>
        <material name="graphite" rgba=".055 .075 .10 1" specular=".3" shininess=".4"/>
        <material name="rubber" rgba=".025 .03 .04 1" specular=".05"/>
        <material name="teal" rgba=".03 .57 .60 1" specular=".35" shininess=".5"/>
        <material name="amber" rgba=".98 .49 .10 1" specular=".35" shininess=".5"/>
      </asset>
      <worldbody>
        <light pos="-1 -1 3" dir=".2 .3 -1" directional="true" diffuse=".29 .28 .26" specular=".18 .18 .18"/>
        <light pos="2 1 2.8" dir="-.4 -.2 -1" directional="true" diffuse=".23 .27 .31" castshadow="false"/>
        <light pos="0 2 2" dir="0 -.4 -1" directional="true" diffuse=".15 .18 .22" castshadow="false"/>
        <camera name="overview" pos="0 -1.85 1.40" xyaxes="1 0 0 0 .584 .812" fovy="48"/>
        <camera name="top" pos="0 0 1.65" xyaxes="1 0 0 0 1 0" fovy="49"/>
        <camera name="left" pos="-.33 0 .93" xyaxes="1 0 0 0 1 0" fovy="43"/>
        <camera name="right" pos=".33 0 .93" xyaxes="1 0 0 0 1 0" fovy="43"/>
        <geom type="plane" size="20 20 .1" pos="0 0 -.84" material="floor" contype="1" conaffinity="3"/>
        <geom name="table" type="box" size=".98 .66 .035" pos="0 0 -.035" material="table" contype="1" conaffinity="3" friction=".65 .005 .001"/>
        <geom class="visual" type="box" size=".99 .67 .012" pos="0 0 -.082" material="graphite"/>
        <geom class="visual" type="box" size=".94 .62 .07" pos="0 0 -.14" material="shell"/>
        <geom class="visual" type="box" size=".27 .004 .011" pos="-.56 -.626 -.14" material="teal"/>
        <geom class="visual" type="box" size=".27 .004 .011" pos=".56 -.626 -.14" material="amber"/>
        <geom class="visual" type="box" size=".04 .04 .30" pos="-.84 -.50 -.51" material="aluminum"/>
        <geom class="visual" type="box" size=".04 .04 .30" pos=".84 -.50 -.51" material="aluminum"/>
        <geom class="visual" type="box" size=".04 .04 .30" pos="-.84 .50 -.51" material="aluminum"/>
        <geom class="visual" type="box" size=".04 .04 .30" pos=".84 .50 -.51" material="aluminum"/>
        <geom class="visual" type="box" size=".84 .024 .024" pos="0 .50 -.65" material="steel"/>
        <geom class="visual" type="box" size=".05 2 1.5" pos="-2.8 1 .66" material="shell"/>
        <geom class="visual" type="box" size="3 .05 1.5" pos="0 2.8 .66" material="shell"/>
        <geom class="visual" type="box" size=".62 .055 .42" pos=".8 2.73 .8" rgba=".12 .2 .26 1"/>
        <geom class="visual" type="box" size=".59 .058 .39" pos=".8 2.72 .8" rgba=".25 .39 .47 1"/>
        <geom class="visual" type="box" size=".013 .06 .4" pos=".8 2.71 .8" material="aluminum"/>
        <geom class="visual" type="box" size=".62 .06 .013" pos=".8 2.71 .8" material="aluminum"/>
        <geom class="visual" type="box" size=".17 .07 .12" pos="-.68 .69 .14" material="graphite"/>
        <geom class="visual" type="box" size=".145 .003 .09" pos="-.68 .617 .15" rgba=".04 .12 .16 1"/>
        <geom class="visual" type="box" size=".11 .004 .004" pos="-.69 .612 .20" material="teal"/>
        <geom class="visual" type="box" size=".075 .004 .004" pos="-.725 .612 .17" rgba=".45 .7 .74 1"/>
        <geom class="visual" type="box" size=".09 .004 .004" pos="-.71 .612 .14" rgba=".45 .7 .74 1"/>
        <geom class="visual" type="box" size=".02 .015 .32" pos=".66 .62 .32" material="aluminum"/>
        <geom class="visual" type="box" size=".06 .04 .03" pos=".66 .59 .67" material="graphite"/>
        <geom class="visual" type="cylinder" size=".017 .006" pos=".66 .545 .67" euler="90 0 0" material="steel"/>
        {arms}
        {fixture_visuals}
        <body name="obstacle" mocap="true" pos="2.3 0 .18"><geom name="obstacle_geom" type="cylinder" size=".065 .075" material="amber" contype="1" conaffinity="3"/><geom class="visual" type="cylinder" size=".066 .008" pos="0 0 .055" material="graphite"/></body>
        {markers}
        <body name="garment" childclass="cloth_node"><flexcomp name="shirt" type="direct" dim="2" point="{point_s}" element="{face_s}" texcoord="{uv_s}" radius=".0018" mass=".18" material="cloth" flatskin="false">
          <edge equality="true" solref=".008 1" solimp=".95 .99 .0001" damping=".0008"/>
          <elasticity young="50000" poisson=".3" thickness=".0004" elastic2d="bend" damping=".01"/>
          <contact selfcollide="auto" contype="4" conaffinity="1" condim="3" friction=".7 .005 .0001" solref=".006 1" solimp=".95 .99 .0001"/>
        </flexcomp></body>
      </worldbody>
      <equality>
        {fixture_equalities}
        <connect name="grasp_left" body1="left_tool" body2="shirt_0" anchor="0 0 0" active="false" solref=".006 1" solimp=".99 .999 .0001"/>
        <connect name="grasp_right" body1="right_tool" body2="shirt_0" anchor="0 0 0" active="false" solref=".006 1" solimp=".99 .999 .0001"/>
      </equality>
      <actuator>{actuators}</actuator>
    </mujoco>'''
    return xml, points, faces, landmarks
