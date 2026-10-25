"""Presentation-only look for the evaluator cameras (``overview`` and ``top``).

The arm cameras feed policies and cached VLA features, so their pixels must not
change. Everything here is applied only while rendering a presentation camera
and restored afterwards: studio lights replace the base lights, studio materials
temporarily overwrite the base material rows, and the frame is supersampled,
shaded with depth-based ambient occlusion, tone mapped and downsampled.
No physical quantity (geometry, contact, mass, solver setting) is touched.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .scene import shirt_mesh

PRESENTATION_CAMERAS = ("overview", "top")
SUPERSAMPLE = 2
STUDIO_GEOM_GROUP = 3

# Base material -> studio material defined in the scene XML.
STUDIO_MATERIALS = {
    "cloth": "cloth_studio",
    "floor": "floor_studio",
    "table": "table_studio",
    "wall": "wall_studio",
    "cabinet": "cabinet_studio",
    "shell": "shell_studio",
    "aluminum": "aluminum_studio",
    "steel": "steel_studio",
    "graphite": "graphite_studio",
    "rubber": "rubber_studio",
    "teal": "teal_studio",
    "amber": "amber_studio",
}
_MATERIAL_FIELDS = (
    "mat_texid",
    "mat_texuniform",
    "mat_texrepeat",
    "mat_emission",
    "mat_specular",
    "mat_shininess",
    "mat_reflectance",
    "mat_metallic",
    "mat_roughness",
    "mat_rgba",
)
HEADLIGHT = {
    "ambient": (0.085, 0.085, 0.09),
    "diffuse": (0.0, 0.0, 0.0),
    "specular": (0.0, 0.0, 0.0),
}
FOG = {"start": 1.6, "end": 6.0, "rgba": (0.63, 0.62, 0.60, 1.0)}
# Clip planes (fractions of the model extent) tight enough for precise depth-based occlusion.
CLIP = {"znear": 0.15, "zfar": 5.0}


# ------------------------------------------------------------------ textures --
def _png(rgb):
    output = BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output, format="PNG")
    return output.getvalue()


def _fbm(rng, height, width, octaves=6, base=4):
    """Fractal value noise in [-1, 1] built from bicubically upsampled grids."""
    total = np.zeros((height, width))
    amplitude, weight = 1.0, 0.0
    for octave in range(octaves):
        cells = base * 2**octave
        grid = rng.uniform(-1, 1, (cells + 1, cells + 1)).astype(np.float32)
        layer = Image.fromarray(grid, mode="F").resize((width, height), Image.BICUBIC)
        total += amplitude * np.asarray(layer)
        weight += amplitude
        amplitude *= 0.55
    return total / weight


@lru_cache(maxsize=1)
def jersey_texture() -> bytes:
    """Heathered navy cotton jersey with knit wales, hems, collar rib and seams.

    The UV layout is the garment's planar footprint (0.88 m by 0.64 m), so the
    image keeps that aspect ratio to avoid stretching the knit.
    """
    width, height = 2048, 1490
    rng = np.random.default_rng(2026)
    yy, xx = np.indices((height, width)).astype(np.float32)
    # Knit: vertical wales of interlocking V loops (about 1.7 mm apart).
    pitch = 4.0
    phase = (xx % pitch) / pitch
    loop = np.abs(
        ((yy + np.where(phase < 0.5, phase, 1 - phase) * pitch * 1.6) % pitch) / pitch - 0.5
    )
    knit = 0.55 * (np.cos(2 * np.pi * phase) * 0.5 + 0.5) + 0.45 * (1 - 2 * loop)
    heather = rng.random((height, width)) < 0.035
    yarn = Image.fromarray(rng.normal(0, 1, (height // 2, width // 2)).astype(np.float32), mode="F")
    yarn = np.asarray(yarn.resize((width, height), Image.BILINEAR))
    dye = _fbm(rng, height, width, octaves=4, base=3)
    base = np.array([39.0, 60.0, 96.0])
    shade = 1 + 0.10 * (knit - 0.5) + 0.05 * yarn + 0.035 * dye
    rgb = base[None, None, :] * shade[..., None]
    rgb[heather] = rgb[heather] * 0.55 + np.array([118.0, 134.0, 160.0]) * 0.45

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    vertices, triangles, _ = shirt_mesh()
    # Same planar UV convention as the mesh texcoords and cotton_texture().
    uv = np.column_stack(
        (
            (vertices[:, 0] + 0.44) / 0.88 * (width - 1),
            (vertices[:, 1] + 0.32) / 0.64 * (height - 1),
        )
    )
    owners = {}
    for a, b, c in triangles:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            owners.setdefault(tuple(sorted((int(i), int(j)))), []).append(int(k))
    boundary = [(e, ks[0]) for e, ks in owners.items() if len(ks) == 1]

    def inset(p, q, inward, distance):
        return p + inward * distance, q + inward * distance

    for (a, b), opposite in boundary:
        p, q = uv[a], uv[b]
        d = (q - p) / max(np.linalg.norm(q - p), 1e-9)
        inward = np.array([-d[1], d[0]])
        if np.dot(inward, uv[opposite] - p) < 0:
            inward = -inward
        mid = (vertices[a] + vertices[b]) / 2
        neck = mid[1] > 0.235 and abs(mid[0]) < 0.085
        band = 34 if neck else 26
        # Folded hem/rib band: slightly darker, then two lines of lighter stitches.
        p1, q1 = inset(p, q, inward, band)
        draw.polygon(
            [tuple(p), tuple(q), tuple(q1), tuple(p1)], fill=(18, 28, 48, 46 if neck else 30)
        )
        if neck:
            for k in range(4, band, 5):
                pk, qk = inset(p, q, inward, k)
                draw.line([tuple(pk), tuple(qk)], fill=(14, 22, 40, 60), width=2)
        for distance in (band - 6, band + 2):
            ps, qs = inset(p, q, inward, distance)
            length = float(np.linalg.norm(qs - ps))
            for t in np.arange(0, length, 9):
                s0 = ps + (qs - ps) * t / max(length, 1)
                s1 = ps + (qs - ps) * min(length, t + 5) / max(length, 1)
                draw.line([tuple(s0), tuple(s1)], fill=(96, 112, 140, 150), width=2)
        draw.line([tuple(p), tuple(q)], fill=(16, 24, 42, 170), width=5)
    return _png(np.asarray(image))


@lru_cache(maxsize=1)
def concrete_texture() -> bytes:
    """Sealed concrete floor: fractal mottling, aggregate flecks, faint trowel arcs."""
    size = 1024
    rng = np.random.default_rng(77)
    mottle = _fbm(rng, size, size, octaves=7, base=3)
    fine = rng.normal(0, 1, (size, size))
    yy, xx = np.indices((size, size))
    arcs = np.sin(np.hypot(xx - 300, yy + 900) / 23.0) * np.sin(np.hypot(xx + 700, yy - 200) / 31.0)
    flecks = rng.random((size, size)) < 0.004
    value = 0.50 + 0.065 * mottle + 0.018 * fine + 0.012 * arcs
    value[flecks] -= 0.12
    rgb = np.stack([value * 1.00, value * 0.975, value * 0.94], -1) * 255
    return _png(rgb)


@lru_cache(maxsize=1)
def benchtop_texture() -> bytes:
    """Matte high-pressure-laminate lab bench: fine speckle on warm light grey."""
    size = 1024
    rng = np.random.default_rng(13)
    value = (
        0.70
        + 0.012 * _fbm(rng, size, size, octaves=5, base=6)
        + 0.02 * rng.normal(0, 1, (size, size))
    )
    speck = rng.random((size, size))
    value[speck < 0.010] -= 0.10
    value[speck > 0.994] += 0.06
    rgb = np.stack([value * 1.0, value * 0.99, value * 0.965], -1) * 255
    return _png(rgb)


def studio_assets():
    return {
        "jersey.png": jersey_texture(),
        "concrete.png": concrete_texture(),
        "benchtop.png": benchtop_texture(),
    }


# ------------------------------------------------------------ model overlay --
class StudioOverlay:
    """Resolve and apply the studio overrides for one compiled model."""

    def __init__(self, model):
        self.model = model
        self.pairs = [
            (model.material(base).id, model.material(studio).id)
            for base, studio in STUDIO_MATERIALS.items()
        ]
        self.studio_lights = np.array(
            [model.light(i).name.startswith("studio_") for i in range(model.nlight)]
        )

    @contextmanager
    def applied(self):
        m = self.model
        saved = {f: getattr(m, f).copy() for f in (*_MATERIAL_FIELDS, "light_active")}
        head = {k: getattr(m.vis.headlight, k).copy() for k in HEADLIGHT}
        fog = (float(m.vis.map.fogstart), float(m.vis.map.fogend), m.vis.rgba.fog.copy())
        clip = (float(m.vis.map.znear), float(m.vis.map.zfar))
        try:
            for base, studio in self.pairs:
                for field in _MATERIAL_FIELDS:
                    getattr(m, field)[base] = getattr(m, field)[studio]
            m.light_active[:] = self.studio_lights
            for key, value in HEADLIGHT.items():
                getattr(m.vis.headlight, key)[:] = value
            m.vis.map.fogstart, m.vis.map.fogend = FOG["start"], FOG["end"]
            m.vis.rgba.fog[:] = FOG["rgba"]
            m.vis.map.znear, m.vis.map.zfar = CLIP["znear"], CLIP["zfar"]
            yield
        finally:
            for field, value in saved.items():
                getattr(m, field)[:] = value
            for key, value in head.items():
                getattr(m.vis.headlight, key)[:] = value
            m.vis.map.fogstart, m.vis.map.fogend = fog[0], fog[1]
            m.vis.rgba.fog[:] = fog[2]
            m.vis.map.znear, m.vis.map.zfar = clip


def studio_scene_option():
    option = mujoco.MjvOption()
    option.geomgroup[STUDIO_GEOM_GROUP] = 1
    return option


def configure_studio_scene(scene):
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
    scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
    scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = True
    scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True


# ------------------------------------------------------------ post-process --
def _box_blur(image, radius, axis):
    pad = [(0, 0)] * image.ndim
    pad[axis] = (radius + 1, radius)
    summed = np.cumsum(np.pad(image, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = image.shape[axis]
    upper = np.take(summed, np.arange(2 * radius + 1, n + 2 * radius + 1), axis=axis)
    lower = np.take(summed, np.arange(0, n), axis=axis)
    return ((upper - lower) / (2 * radius + 1)).astype(np.float32)


def gaussian_blur(image, sigma):
    """Three box passes per axis approximate a Gaussian of the given sigma (pixels)."""
    radius = max(1, round((float(np.sqrt(4 * sigma * sigma + 1)) - 1) / 2))
    out = image.astype(np.float32)
    for axis in (0, 1):
        for _ in range(3):
            out = _box_blur(out, radius, axis)
    return out


# (radius in output pixels, depth reach in metres, weight)
_AO_BANDS = ((2, 0.006, 0.30), (6, 0.02, 0.40), (16, 0.06, 0.45))
_AO_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def ambient_occlusion(depth, scale):
    """Crease occlusion from symmetric depth pairs.

    For each pixel and direction, compare its depth with the mean of the two
    neighbours at +/- r. A pixel deeper than that mean sits in a concavity (fold,
    contact, corner). Planes give zero by symmetry. Any pair that straddles a
    depth discontinuity (a silhouette against the far wall) is dropped, so
    there are no halos.
    """
    depth = depth.astype(np.float32)
    reach_max = max(int(r * scale) for r, _, _ in _AO_BANDS)
    # Odd reflection extends planes linearly, so frame borders read as flat.
    padded = np.pad(depth, reach_max, mode="reflect", reflect_type="odd")
    h, w = depth.shape
    occlusion = np.zeros_like(depth)
    for radius, reach, weight in _AO_BANDS:
        r = int(radius * scale)
        band = np.zeros_like(depth)
        for dy, dx in _AO_DIRECTIONS:
            a = padded[
                reach_max + dy * r : reach_max + dy * r + h,
                reach_max + dx * r : reach_max + dx * r + w,
            ]
            b = padded[
                reach_max - dy * r : reach_max - dy * r + h,
                reach_max - dx * r : reach_max - dx * r + w,
            ]
            valid = (np.abs(depth - a) < 4 * reach) & (np.abs(depth - b) < 4 * reach)
            band += np.where(valid, np.clip((depth - (a + b) / 2) / reach, 0, 1), 0)
        occlusion += weight * band / len(_AO_DIRECTIONS)
    return np.clip(gaussian_blur(occlusion, 1.2 * scale), 0, 0.6)


def _to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def _filmic(x):
    """ACES fitted tone curve (Narkowicz 2015)."""
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def grade(rgb, depth, output_size, *, exposure=1.3):
    """Occlusion, filmic tone mapping, gentle split toning and vignette, then downsample."""
    scale = rgb.shape[1] / output_size[0]
    linear = _to_linear(rgb.astype(np.float32) / 255)
    linear *= (1 - ambient_occlusion(depth.astype(np.float32), scale))[..., None]
    mapped = _filmic(linear * exposure)
    luminance = mapped @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    warm, cool = np.array([1.02, 1.0, 0.975]), np.array([0.985, 1.0, 1.025])
    t = np.clip(luminance, 0, 1)[..., None]
    mapped = mapped * (cool * (1 - t) + warm * t)
    h, w = luminance.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
    mapped *= (1 - 0.22 * radius**2.4)[..., None]
    srgb = (np.clip(_to_srgb(np.clip(mapped, 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(srgb).resize(output_size, Image.LANCZOS))
