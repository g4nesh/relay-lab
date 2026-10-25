import numpy as np
import pytest

pytest.importorskip("mujoco")

from relay_lab.cloth import presentation
from relay_lab.cloth.world import AGENTS, ClothWorld


@pytest.fixture(scope="module")
def world():
    with ClothWorld(seed=2) as w:
        yield w


def test_studio_lights_are_inactive_for_policy_cameras(world):
    names = [world.model.light(i).name for i in range(world.model.nlight)]
    studio = [n.startswith("studio_") for n in names]
    assert any(studio)
    assert world.model.light_active.tolist() == [not s for s in studio]


def test_presentation_render_restores_model_and_arm_pixels(world):
    fields = ("mat_rgba", "mat_texid", "mat_specular", "light_active")
    before = {f: getattr(world.model, f).copy() for f in fields}
    clip = (world.model.vis.map.znear, world.model.vis.map.zfar)
    arm = {a: world.render(a, 128, 128) for a in AGENTS}
    frame = world.render("overview", 160, 100)
    assert frame.shape == (100, 160, 3) and frame.dtype == np.uint8
    for f in fields:
        assert np.array_equal(getattr(world.model, f), before[f]), f
    assert (world.model.vis.map.znear, world.model.vis.map.zfar) == clip
    for a in AGENTS:
        assert np.array_equal(world.render(a, 128, 128), arm[a])


def test_occlusion_ignores_planes_and_silhouettes():
    yy, xx = np.indices((64, 64)).astype(np.float32)
    plane = 1.5 + 0.01 * xx + 0.004 * yy
    assert presentation.ambient_occlusion(plane, 1).max() < 1e-4
    step = np.where(xx < 32, 1.5, 4.0).astype(np.float32)
    assert presentation.ambient_occlusion(step, 1).max() < 1e-4
    valley = 1.6 - 0.004 * np.abs(xx - 32)  # deepest at the centre column
    assert presentation.ambient_occlusion(valley.astype(np.float32), 1)[:, 32].mean() > 0.05
