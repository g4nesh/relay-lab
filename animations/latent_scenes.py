"""Manim scenes for Relay Lab's learned SmolVLA <-> X-VLA message channel.

Every value drawn here is read from ``data/latent.json``, which ``extract.py``
produces by running the packaged ``embeddings.pt`` checkpoint on the packaged
held-out episode. Architecture labels mirror ``relay_lab.cloth.latent_adapter``.

Render (from this directory, with the conda-forge Manim environment):

    .env/bin/manim -qh latent_scenes.py RelayLatentFull
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Arrow,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    Group,
    ImageMobject,
    LaggedStart,
    Line,
    MarkupText,
    Rectangle,
    RoundedRectangle,
    Scene,
    ShowPassingFlash,
    Square,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    VMobject,
    Write,
    config,
    interpolate_color,
)
from manim.utils.color import ManimColor
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data/latent.json").read_text())
DIMS = DATA["dims"]
TEST = DATA["splits"]["test"]
SAMPLE = DATA["sample"]
RESULTS = DATA["results"]
NORM = DATA["normalization"]
CURVES = DATA["curves"]
SOURCE = DATA["source"]

# Palette
BG = "#0C0E13"
FG = "#E9EDF2"
MUTED = "#8E98A6"
FAINT = "#2A303A"
SMOL = "#5CC8E8"  # left arm, SmolVLA
XVLA = "#B48CF2"  # right arm, X-VLA
MSG = "#FFD166"  # 64-D message
CTX = "#7BE0A8"  # receiver context
GRAD = "#FF7A59"  # gradients / failures
GOOD = "#7BE0A8"
BAD = "#FF6B6B"
COND = {"clear": "#9AA3AD", "left_obstacle": "#F2994A", "right_obstacle": "#E05A8A"}
COND_LABEL = {
    "clear": "clear",
    "left_obstacle": "obstacle by left arm",
    "right_obstacle": "obstacle by right arm",
}
NEG, MID, POS = ManimColor("#3E7BDB"), ManimColor("#1A1E26"), ManimColor("#F0B43C")

SERIF = "STIX Two Text"
MONO = "Menlo"

config.background_color = BG
config.frame_rate = 60


# ---------------------------------------------------------------- helpers ----
def T(text, size=28, color=FG, font=SERIF, **kw):
    return Text(text, font=font, font_size=size, color=color, **kw)


def M(markup, size=28, color=FG, font=SERIF):
    return MarkupText(markup, font=font, font_size=size, color=color)


def code(text, size=22, color=MUTED):
    return Text(text, font=MONO, font_size=size, color=color)


def title(text, sub=None):
    head = T(text, 42).to_edge(UP, buff=0.42)
    if sub is None:
        return VGroup(head)
    return VGroup(head, T(sub, 22, MUTED).next_to(head, DOWN, buff=0.16))


def diverging(value, vmax):
    x = float(np.clip(value / vmax, -1, 1))
    return interpolate_color(MID, POS, x) if x >= 0 else interpolate_color(MID, NEG, -x)


def strip_image(values, vmax, height, width=0.32):
    """A real vector as a 1-pixel-wide image: one row per component."""
    v = np.clip(np.asarray(values, float) / vmax, -1, 1)
    pos, neg, mid = (np.array(c.to_rgb()) for c in (POS, NEG, MID))
    rgb = np.where(v[:, None] >= 0, mid + (pos - mid) * v[:, None], mid + (neg - mid) * -v[:, None])
    pixels = (np.repeat(rgb[:, None, :], 1, axis=1) * 255).round().astype(np.uint8)
    image = ImageMobject(pixels)
    image.set_resampling_algorithm(0)  # nearest
    image.stretch_to_fit_height(height).stretch_to_fit_width(width)
    frame = Rectangle(width=width, height=height, stroke_color=FAINT, stroke_width=1.5)
    frame.move_to(image)
    return Group(image, frame)


def grid8(values, cell=0.3, gap=0.035):
    """The 64-D message as an 8x8 grid, row-major, coloured by value in [-1, 1]."""
    cells = VGroup()
    for i, value in enumerate(values):
        r, c = divmod(i, 8)
        sq = Square(cell, stroke_width=0, fill_color=diverging(value, 1.0), fill_opacity=1)
        sq.move_to(np.array([c * (cell + gap), -r * (cell + gap), 0]))
        cells.add(sq)
    return cells.move_to(ORIGIN)


def recolor_grid(grid, values):
    return AnimationGroup(
        *(
            sq.animate.set_fill(diverging(v, 1.0), opacity=1)
            for sq, v in zip(grid, values, strict=True)
        )
    )


def block(lines, width, height, color, size=22):
    box = RoundedRectangle(
        corner_radius=0.14,
        width=width,
        height=height,
        stroke_color=color,
        stroke_width=2.5,
        fill_color=color,
        fill_opacity=0.07,
    )
    text = VGroup(*(code(line, size, FG) for line in lines)).arrange(DOWN, buff=0.12)
    text.move_to(box)
    return VGroup(box, text)


def scatter_panel(points, conditions, width, height, label, explained):
    """PCA scatter with equal aspect. Clear = dot; obstacles = rings so overlaps show."""
    pts = np.asarray(points, float)
    frame = Rectangle(width=width, height=height, stroke_color=FAINT, stroke_width=1.5)
    span = max(np.abs(pts[:, 0]).max() / (width / 2), np.abs(pts[:, 1]).max() / (height / 2))
    scale = 0.86 / span
    cross = VGroup(
        Line(LEFT * width / 2, RIGHT * width / 2, stroke_color=FAINT, stroke_width=1),
        Line(DOWN * height / 2, UP * height / 2, stroke_color=FAINT, stroke_width=1),
    )
    marks = VGroup()
    order = {"clear": 0, "left_obstacle": 1, "right_obstacle": 2}
    for p, c in sorted(zip(pts, conditions), key=lambda pc: order[pc[1]]):
        position = np.array([p[0] * scale, p[1] * scale, 0])
        if c == "clear":
            mark = Dot(position, radius=0.05, color=COND[c])
        else:
            radius = 0.1 if c == "left_obstacle" else 0.15
            mark = Circle(radius=radius, stroke_color=COND[c], stroke_width=2.2).move_to(position)
        marks.add(mark)
    x_label = T(f"PC1 · {explained[0] * 100:.1f}% of variance", 18, MUTED)
    x_label.next_to(frame, DOWN, buff=0.12)
    y_label = T(f"PC2 · {explained[1] * 100:.1f}%", 18, MUTED).rotate(np.pi / 2)
    y_label.next_to(frame, LEFT, buff=0.12)
    head = T(label, 24).next_to(frame, UP, buff=0.14)
    return VGroup(frame, cross, marks, x_label, y_label, head)


def legend(keys=("clear", "left_obstacle", "right_obstacle"), size=20):
    items = VGroup()
    for k in keys:
        if k == "clear":
            mark = Dot(radius=0.05, color=COND[k])
        else:
            mark = Circle(radius=0.1 if k == "left_obstacle" else 0.13, stroke_color=COND[k], stroke_width=2.2)
        items.add(VGroup(mark, T(COND_LABEL[k], size, MUTED)).arrange(RIGHT, buff=0.14))
    return items.arrange(RIGHT, buff=0.6)


def snapshot_label():
    snap = SAMPLE["snapshot"]  # e.g. test-104-s005
    _, seed, state = snap.split("-")
    return seed, state


def clear_scene(scene, run_time=0.8):
    if scene.mobjects:
        scene.play(*(FadeOut(m) for m in scene.mobjects), run_time=run_time)


# ----------------------------------------------------------------- scenes ----
def private_views(s: Scene):
    seed, state = snapshot_label()
    head = title("Two arms, two private cameras", f"held-out episode {seed} · state {state} · obstacle beside the left arm")
    s.play(Write(head[0]), FadeIn(head[1], shift=DOWN * 0.1), run_time=1.6)

    size = 3.5
    left = ImageMobject(str(HERE / "data" / SAMPLE["left_obstacle_png"])).set(height=size)
    right = ImageMobject(str(HERE / "data" / SAMPLE["right_obstacle_png"])).set(height=size)
    left.move_to(LEFT * 3.1 + DOWN * 0.35)
    right.move_to(RIGHT * 3.1 + DOWN * 0.35)
    left_frame = SurroundingRectangle(left, buff=0, color=SMOL, stroke_width=3)
    right_frame = SurroundingRectangle(right, buff=0, color=XVLA, stroke_width=3)
    left_name = M(f'left arm · <span foreground="{SMOL}">SmolVLA</span>', 26).next_to(left, DOWN, buff=0.22)
    right_name = M(f'right arm · <span foreground="{XVLA}">X-VLA</span>', 26).next_to(right, DOWN, buff=0.22)
    s.play(
        FadeIn(left, shift=RIGHT * 0.3),
        Create(left_frame),
        FadeIn(right, shift=LEFT * 0.3),
        Create(right_frame),
        run_time=1.4,
    )
    s.play(FadeIn(left_name, shift=UP * 0.1), FadeIn(right_name, shift=UP * 0.1), run_time=0.8)

    # Locate the orange obstacle from the real pixels.
    rgb = np.asarray(Image.open(HERE / "data" / SAMPLE["left_obstacle_png"]).convert("RGB")).astype(int)
    mask = (rgb[..., 0] > 200) & (rgb[..., 1] > 120) & (rgb[..., 1] < 200) & (rgb[..., 2] < 110)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cx = left.get_left()[0] + xs.mean() / w * size
    cy = left.get_top()[1] - ys.mean() / h * size
    radius = max(np.ptp(xs), np.ptp(ys)) / w * size * 0.62
    ring = Circle(radius=radius, stroke_color=COND["left_obstacle"], stroke_width=4).move_to([cx, cy, 0])
    note = T("only the left camera sees it", 24, COND["left_obstacle"])
    note.next_to(left_frame, UP, buff=0.18).align_to(left_frame, LEFT)
    s.play(Create(ring), FadeIn(note, shift=DOWN * 0.1), run_time=1.0)
    s.wait(1.2)

    # The receiver's view is byte-identical with and without the hidden obstacle.
    assert SAMPLE["receiver_pixels_identical"]
    clear_img = ImageMobject(str(HERE / "data" / SAMPLE["right_clear_png"])).set(height=2.5)
    obstacle_img = right.copy()
    s.play(
        FadeOut(left), FadeOut(left_frame), FadeOut(left_name), FadeOut(ring), FadeOut(note),
        FadeOut(right_name), FadeOut(right_frame),
        obstacle_img.animate.set(height=2.5).move_to(RIGHT * 2.2 + DOWN * 0.2),
        FadeOut(right),
        run_time=1.2,
    )
    clear_img.move_to(LEFT * 2.2 + DOWN * 0.2)
    frames = VGroup(
        SurroundingRectangle(clear_img, buff=0, color=XVLA, stroke_width=2.5),
        SurroundingRectangle(obstacle_img, buff=0, color=XVLA, stroke_width=2.5),
    )
    eq = T("=", 64).move_to(DOWN * 0.2)
    caps = VGroup(
        T("right camera · clear scene", 22, MUTED).next_to(clear_img, DOWN, buff=0.2),
        T("right camera · obstacle by left arm", 22, MUTED).next_to(obstacle_img, DOWN, buff=0.2),
    )
    s.play(FadeIn(clear_img, shift=RIGHT * 0.2), Create(frames), FadeIn(caps), run_time=1.0)
    s.play(Write(eq), run_time=0.6)
    verdict = M(
        "byte-for-byte identical pixels and local measurements",
        26,
    ).next_to(caps, DOWN, buff=0.45)
    s.play(FadeIn(verdict, shift=UP * 0.1), run_time=0.8)
    s.wait(1.0)

    holds = [TEST["arms"][a]["hold_label"][SAMPLE["index"]] for a in ("left", "right")]
    assert all(holds), "sample must require a coordinated hold"
    need = M(
        f'Both arms must <span foreground="{COND["left_obstacle"]}">hold</span>. '
        f"The right arm can only know if the left arm tells it.",
        28,
    ).to_edge(DOWN, buff=0.5)
    s.play(FadeOut(verdict), FadeIn(need, shift=UP * 0.15), run_time=1.0)
    s.wait(2.4)


def two_spaces(s: Scene):
    head = title("Two models, two unrelated feature spaces")
    s.play(Write(head), run_time=1.2)

    rows = []
    for arm, color, name, details in (
        ("left", SMOL, "SmolVLA", ("final normalized prefix hidden states", "mean of valid image + language tokens")),
        ("right", XVLA, "X-VLA", ("Florence-2 encoder final hidden states", "mean of image + valid language tokens")),
    ):
        width = DIMS["widths"][arm]
        raw = SAMPLE[f"{arm}_stages"]["raw"]
        strip = strip_image(raw, 2.5, height=0.0031 * width, width=0.42)
        sym = M(f'<i>f</i><sub>{"L" if arm == "left" else "R"}</sub> ∈ ℝ<sup>{width}</sup>', 34, color)
        info = VGroup(
            sym,
            T(name, 24, color),
            *(T(d, 19, MUTED) for d in details),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        rows.append((strip, info))
    (ls, li), (rs, ri) = rows
    ls.move_to(LEFT * 5.7 + DOWN * 0.3)
    li.next_to(ls, RIGHT, buff=0.4).shift(UP * 0.3)
    rs.move_to(RIGHT * 0.9 + DOWN * 0.3)
    ri.next_to(rs, RIGHT, buff=0.4).shift(UP * 0.3)
    tick_l = T("960", 20, MUTED).next_to(ls, DOWN, buff=0.12)
    tick_r = T("1024", 20, MUTED).next_to(rs, DOWN, buff=0.12)
    s.play(GrowFromEdge(ls, DOWN), GrowFromEdge(rs, DOWN), run_time=1.4)
    s.play(FadeIn(li, shift=LEFT * 0.2), FadeIn(ri, shift=LEFT * 0.2), FadeIn(tick_l), FadeIn(tick_r), run_time=1.2)
    cap = T("one pooled vector per camera frame, from the frozen backbone · action heads never run", 21, MUTED)
    cap.to_edge(DOWN, buff=0.45)
    s.play(FadeIn(cap), run_time=0.8)
    s.wait(1.6)
    s.play(*(FadeOut(m) for m in (ls, rs, li, ri, tick_l, tick_r, cap)), run_time=0.8)

    # Real held-out features, projected to their own top principal components.
    cf = TEST["counterfactual"]
    panels = VGroup(
        scatter_panel(TEST["arms"]["left"]["raw_pca"], cf, 5.0, 3.6, "SmolVLA features (960-D)", TEST["arms"]["left"]["raw_pca_explained"]),
        scatter_panel(TEST["arms"]["right"]["raw_pca"], cf, 5.0, 3.6, "X-VLA features (1024-D)", TEST["arms"]["right"]["raw_pca_explained"]),
    ).arrange(RIGHT, buff=1.4).shift(DOWN * 0.15)
    for p in panels:
        s.play(Create(p[0]), Create(p[1]), FadeIn(p[3]), FadeIn(p[4]), FadeIn(p[5]), run_time=0.6)
        s.play(LaggedStart(*(FadeIn(m, scale=0.5) for m in p[2]), lag_ratio=0.02), run_time=1.4)
    key = legend().to_edge(DOWN, buff=0.35)
    s.play(FadeIn(key), run_time=0.6)
    note = M(
        f'Each model only encodes what <i>its</i> camera saw: '
        f'<span foreground="{COND["right_obstacle"]}">right-side</span> obstacles land exactly on '
        f'<span foreground="{COND["clear"]}">clear</span> points for SmolVLA, and vice versa.',
        21,
    ).next_to(panels, DOWN, buff=0.5)
    s.play(FadeIn(note, shift=UP * 0.1), run_time=1.0)
    s.wait(3.0)
    s.play(FadeOut(panels), FadeOut(key), FadeOut(note), run_time=0.8)

    # Normalization gives both the same statistics, not the same coordinates.
    sub = T("Normalizing makes them look alike…", 30).next_to(head, DOWN, buff=0.35)
    s.play(FadeIn(sub), run_time=0.8)
    bins = np.asarray(NORM["bins"])
    width, height = 8.0, 3.0
    origin = np.array([-width / 2, -2.3, 0])
    peak = max(max(NORM["hist"]["left"]), max(NORM["hist"]["right"]))
    axis = Line(origin, origin + RIGHT * width, stroke_color=MUTED, stroke_width=1.5)
    ticks = VGroup()
    for v in (-4, -2, 0, 2, 4):
        x = origin[0] + (v - bins[0]) / (bins[-1] - bins[0]) * width
        ticks.add(T(f"{v:+d}" if v else "0", 18, MUTED).move_to([x, origin[1] - 0.25, 0]))
    curves = VGroup()
    for arm, color in (("left", SMOL), ("right", XVLA)):
        pts = []
        for i, d in enumerate(NORM["hist"][arm]):
            x0 = origin[0] + i / (len(bins) - 1) * width
            x1 = origin[0] + (i + 1) / (len(bins) - 1) * width
            y = origin[1] + d / peak * height
            pts += [[x0, y, 0], [x1, y, 0]]
        pts = [[origin[0], origin[1], 0], *pts, [origin[0] + width, origin[1], 0]]
        curve = VMobject(stroke_color=color, stroke_width=3, fill_color=color, fill_opacity=0.12)
        curve.set_points_as_corners(pts)
        curves.add(curve)
    s.play(Create(axis), FadeIn(ticks), run_time=0.6)
    s.play(Create(curves[0]), Create(curves[1]), run_time=1.6)
    stats = VGroup(
        M(
            f'<span foreground="{SMOL}">SmolVLA</span>  mean {NORM["stats"]["left"]["mean"]:+.4f}   std {NORM["stats"]["left"]["std"]:.4f}',
            22,
        ),
        M(
            f'<span foreground="{XVLA}">X-VLA</span>      mean {NORM["stats"]["right"]["mean"]:+.4f}   std {NORM["stats"]["right"]["std"]:.4f}',
            22,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    stats.next_to(axis, UP, buff=height + 0.15).align_to(axis, RIGHT).shift(RIGHT * 1.3)
    hist_cap = T(f"per-vector LayerNorm of all {NORM['samples']} held-out features, pooled histogram", 18, MUTED)
    hist_cap.next_to(ticks, DOWN, buff=0.15)
    s.play(FadeIn(stats), FadeIn(hist_cap), run_time=0.8)
    s.wait(1.6)
    point = M(
        "…but dimension <i>k</i> of one space has no counterpart in the other.\n"
        f'960 ≠ 1024 coordinates, trained by different models. A <span foreground="{MSG}">learned translation</span> is required.',
        24,
    )
    point.move_to(sub)
    s.play(FadeOut(sub), FadeIn(point, shift=UP * 0.1), run_time=1.0)
    s.wait(3.0)


def sender(s: Scene):
    head = title("The sender: a learned encoder per model", "senders[arm] = LayerNorm → Linear → GELU → Linear → tanh")
    s.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)

    def row(arm, color, y, scale=1.0):
        stages = SAMPLE[f"{arm}_stages"]
        width = DIMS["widths"][arm]
        raw = strip_image(stages["raw"], 2.5, height=2.6 * scale, width=0.36 * scale)
        hidden = strip_image(stages["gelu"], 1.5, height=2.6 * scale * 96 / width * 3.2, width=0.36 * scale)
        box1 = block([f"LayerNorm({width})", f"Linear({width} → 96)", "GELU"], 3.1 * scale, 1.5 * scale, color, int(21 * scale))
        box2 = block(["Linear(96 → 64)", "tanh"], 2.5 * scale, 1.1 * scale, color, int(21 * scale))
        grid = grid8(stages["tanh"], cell=0.27 * scale)
        parts = Group(raw, box1, hidden, box2, grid).arrange(RIGHT, buff=0.55 * scale)
        parts.move_to([0.35, y, 0])
        arrows = VGroup(
            *(
                Arrow(a.get_right(), b.get_left(), buff=0.08, stroke_width=2.5, color=MUTED, max_tip_length_to_length_ratio=0.25)
                for a, b in zip(parts[:-1], parts[1:])
            )
        )
        labels = VGroup(
            M(f"ℝ<sup>{width}</sup>", int(22 * scale), color).next_to(raw, DOWN, buff=0.12),
            M("ℝ<sup>96</sup>", int(22 * scale), MUTED).next_to(hidden, DOWN, buff=0.12),
            M("<i>m</i> ∈ [−1, 1]<sup>64</sup>", int(22 * scale), MSG).next_to(grid, DOWN, buff=0.14),
        )
        name = T("SmolVLA" if arm == "left" else "X-VLA", int(24 * scale), color).next_to(raw, UP, buff=0.14)
        return parts, arrows, labels, name

    p, a, lab, name = row("left", SMOL, 0.35)
    raw, box1, hidden, box2, grid = p
    s.play(GrowFromEdge(raw, DOWN), FadeIn(name), FadeIn(lab[0]), run_time=1.0)
    s.play(GrowArrow(a[0]), FadeIn(box1, shift=RIGHT * 0.1), run_time=0.8)
    s.play(GrowArrow(a[1]), GrowFromEdge(hidden, DOWN), FadeIn(lab[1]), run_time=0.8)
    s.play(GrowArrow(a[2]), FadeIn(box2, shift=RIGHT * 0.1), run_time=0.8)
    s.play(GrowArrow(a[3]), LaggedStart(*(FadeIn(c, scale=0.4) for c in grid), lag_ratio=0.012), FadeIn(lab[2]), run_time=1.4)
    flow = VGroup(*(ar.copy().set_color(MSG).set_stroke(width=6) for ar in a))
    s.play(LaggedStart(*(ShowPassingFlash(f, time_width=0.6) for f in flow), lag_ratio=0.35), run_time=1.4)
    caption = T(
        f"actual values for held-out state {snapshot_label()[1]}: left camera sees the obstacle",
        20,
        MUTED,
    ).to_edge(DOWN, buff=0.5)
    s.play(FadeIn(caption), run_time=0.6)
    s.wait(1.0)

    # Same physical state, obstacle removed from the scene: the message changes.
    clear_caption = T("same state, clear scene: the message changes", 20, MUTED).move_to(caption)
    s.play(recolor_grid(grid, SAMPLE["left_message_clear"]), Transform(caption, clear_caption), run_time=1.4)
    s.wait(0.8)
    back = T(f"obstacle visible again · {DIMS['parameters_by_module']['senders'] // 2:,} sender weights per arm, trained", 20, MUTED).move_to(caption)
    s.play(recolor_grid(grid, SAMPLE["left_stages"]["tanh"]), Transform(caption, back), run_time=1.4)
    s.wait(1.2)

    top = Group(p, a, lab, name)
    s.play(top.animate.scale(0.72).move_to(UP * 1.25 + RIGHT * 0.2), FadeOut(caption), run_time=1.0)
    p2, a2, lab2, name2 = row("right", XVLA, -1.95, scale=0.72)
    s.play(FadeIn(Group(p2, a2, lab2, name2), shift=UP * 0.2), run_time=1.2)
    same = M(
        f'separate weights for each model, same <span foreground="{MSG}">64-number</span> interface',
        24,
    ).to_edge(DOWN, buff=0.32)
    s.play(FadeIn(same), run_time=0.8)
    s.wait(2.0)
    clear_scene(s)

    # Real held-out messages from both senders.
    head2 = title("What the 64 numbers learned to say")
    cf = TEST["counterfactual"]
    panels = VGroup(
        scatter_panel(TEST["arms"]["left"]["message_pca"], cf, 5.0, 3.6, "SmolVLA messages", TEST["arms"]["left"]["message_pca_explained"]),
        scatter_panel(TEST["arms"]["right"]["message_pca"], cf, 5.0, 3.6, "X-VLA messages", TEST["arms"]["right"]["message_pca_explained"]),
    ).arrange(RIGHT, buff=1.4).shift(DOWN * 0.15)
    s.play(Write(head2), run_time=1.0)
    for pnl in panels:
        s.play(Create(pnl[0]), Create(pnl[1]), FadeIn(pnl[3]), FadeIn(pnl[4]), FadeIn(pnl[5]), run_time=0.6)
        s.play(LaggedStart(*(FadeIn(m, scale=0.5) for m in pnl[2]), lag_ratio=0.02), run_time=1.3)
    key = legend().to_edge(DOWN, buff=0.35)
    note = M(
        "The first principal axis of each sender's messages splits on the one thing only it can see.",
        22,
    ).next_to(panels, DOWN, buff=0.5)
    s.play(FadeIn(key), FadeIn(note, shift=UP * 0.1), run_time=1.0)
    s.wait(3.2)


def receiver(s: Scene):
    head = title("Directed receivers translate the peer's message")
    s.play(Write(head), run_time=1.2)

    # Channel: each arm's message crosses SpacetimeDB to the other arm's receiver.
    band = Rectangle(width=2.6, height=4.6, stroke_color=FAINT, stroke_width=1.5, fill_color=FAINT, fill_opacity=0.25)
    band.move_to(DOWN * 0.4)
    band_label = VGroup(T("SpacetimeDB", 24), code("relay.cloth.learned-latent.v1", 15)).arrange(DOWN, buff=0.1)
    band_label.next_to(band, UP, buff=0.14)
    ls = block(["senders['left']"], 3.0, 0.8, SMOL, 20).move_to(LEFT * 5.0 + UP * 0.6)
    rr = block(["receivers['right']"], 3.0, 0.8, XVLA, 20).move_to(RIGHT * 5.0 + UP * 0.6)
    rs = block(["senders['right']"], 3.0, 0.8, XVLA, 20).move_to(RIGHT * 5.0 + DOWN * 1.6)
    lr = block(["receivers['left']"], 3.0, 0.8, SMOL, 20).move_to(LEFT * 5.0 + DOWN * 1.6)
    top_arrow = Arrow(ls.get_right(), rr.get_left(), buff=0.1, color=MSG, stroke_width=3)
    bot_arrow = Arrow(rs.get_left(), lr.get_right(), buff=0.1, color=MSG, stroke_width=3)
    m1 = M("<i>m</i><sub>L</sub> ∈ ℝ<sup>64</sup>", 22, MSG).next_to(top_arrow, UP, buff=0.1)
    m2 = M("<i>m</i><sub>R</sub> ∈ ℝ<sup>64</sup>", 22, MSG).next_to(bot_arrow, DOWN, buff=0.1)
    s.play(FadeIn(band), FadeIn(band_label), run_time=0.8)
    s.play(FadeIn(ls), FadeIn(rs), run_time=0.8)
    s.play(GrowArrow(top_arrow), GrowArrow(bot_arrow), FadeIn(m1), FadeIn(m2), run_time=1.2)
    s.play(FadeIn(rr), FadeIn(lr), run_time=0.8)
    notes = VGroup(
        T("decodes SmolVLA messages for X-VLA", 18, MUTED).next_to(rr, DOWN, buff=0.12),
        T("decodes X-VLA messages for SmolVLA", 18, MUTED).next_to(lr, UP, buff=0.12),
    )
    s.play(FadeIn(notes), run_time=0.8)
    for _ in range(2):
        s.play(
            ShowPassingFlash(top_arrow.copy().set_stroke(width=7, color=MSG), time_width=0.5),
            ShowPassingFlash(bot_arrow.copy().set_stroke(width=7, color=MSG), time_width=0.5),
            run_time=1.0,
        )
    s.wait(0.6)
    clear_scene(s)

    # Receiver layers and the validity mask.
    head2 = title("Inside one receiver", "receivers[arm] = Linear(64 → 64) → GELU → Linear(64 → 32) → tanh")
    s.play(Write(head2[0]), FadeIn(head2[1]), run_time=1.2)
    msg = SAMPLE["left_message"]
    grid = grid8(msg, cell=0.27).move_to(LEFT * 4.6 + UP * 0.2)
    rx = block(["Linear(64 → 64)", "GELU", "Linear(64 → 32)", "tanh"], 3.0, 2.0, XVLA, 21).move_to(UP * 0.2)
    ctx_values = SAMPLE["right_context"]
    ctx = strip_image(ctx_values, 1.0, height=2.6, width=0.42).move_to(RIGHT * 4.3 + UP * 0.2)
    arrows = VGroup(
        Arrow(grid.get_right(), rx.get_left(), buff=0.15, color=MUTED, stroke_width=2.5),
        Arrow(rx.get_right(), ctx.get_left(), buff=0.15, color=MUTED, stroke_width=2.5),
    )
    lab = VGroup(
        M("<i>m</i><sub>L</sub> (from SmolVLA)", 22, MSG).next_to(grid, DOWN, buff=0.2),
        M("<i>c</i> ∈ ℝ<sup>32</sup> (for X-VLA's head)", 22, CTX).next_to(ctx, DOWN, buff=0.2),
    )
    s.play(LaggedStart(*(FadeIn(c, scale=0.4) for c in grid), lag_ratio=0.01), FadeIn(lab[0]), run_time=1.0)
    s.play(GrowArrow(arrows[0]), FadeIn(rx), run_time=0.8)
    s.play(GrowArrow(arrows[1]), GrowFromEdge(ctx, DOWN), FadeIn(lab[1]), run_time=0.9)
    formula = M(
        '<i>c</i> = <i>R</i>(<i>m</i> · <i>v</i>) · <i>v</i>      <i>v</i> = 1 if a valid packet arrived, else 0',
        26,
    ).to_edge(DOWN, buff=1.05)
    why = T("a missing message gives exactly zero context, so decoder biases cannot invent a peer", 20, MUTED)
    why.next_to(formula, DOWN, buff=0.18)
    s.play(Write(formula), run_time=1.4)
    s.play(FadeIn(why), run_time=0.7)
    zero_ctx = strip_image(np.zeros(32), 1.0, height=2.6, width=0.42).move_to(ctx)
    zero_grid = [0.0] * 64
    s.play(recolor_grid(grid, zero_grid), Transform(ctx[0], zero_ctx[0]), run_time=1.2)
    s.wait(0.8)
    ctx_back = strip_image(ctx_values, 1.0, height=2.6, width=0.42).move_to(ctx)
    s.play(recolor_grid(grid, msg), Transform(ctx[0], ctx_back[0]), run_time=1.2)
    s.wait(1.0)
    clear_scene(s)

    # The receivers' contexts, on held-out data, separate the PEER's obstacle.
    head3 = title("After translation, each arm 'knows' its peer's view")
    cf = TEST["counterfactual"]
    panels = VGroup(
        scatter_panel(TEST["arms"]["left"]["context_pca"], cf, 5.0, 3.6, "SmolVLA arm's context (from X-VLA)", TEST["arms"]["left"]["context_pca_explained"]),
        scatter_panel(TEST["arms"]["right"]["context_pca"], cf, 5.0, 3.6, "X-VLA arm's context (from SmolVLA)", TEST["arms"]["right"]["context_pca_explained"]),
    ).arrange(RIGHT, buff=1.4).shift(DOWN * 0.15)
    s.play(Write(head3), run_time=1.0)
    for pnl in panels:
        s.play(Create(pnl[0]), Create(pnl[1]), FadeIn(pnl[3]), FadeIn(pnl[4]), FadeIn(pnl[5]), run_time=0.6)
        s.play(LaggedStart(*(FadeIn(m, scale=0.5) for m in pnl[2]), lag_ratio=0.02), run_time=1.3)
    key = legend().to_edge(DOWN, buff=0.35)
    note = M(
        "The split flips sides: each receiver separates the obstacle it <i>cannot</i> see.",
        22,
    ).next_to(panels, DOWN, buff=0.5)
    s.play(FadeIn(key), FadeIn(note, shift=UP * 0.1), run_time=1.0)
    s.wait(3.2)


def head_scene(s: Scene):
    head = title("The replacement action head", "the native VLA action heads are not used")
    s.play(Write(head[0]), FadeIn(head[1]), run_time=1.2)
    n_feat, n_state = DIMS["features"], DIMS["local_state"]
    segments = [
        ("own features", "LayerNorm → Linear → tanh", 32, SMOL),
        ("local state", f"{n_feat} measurements + {DIMS['phases']} phase one-hot", n_state, FG),
        ("peer context c", "from the receiver", 32, CTX),
        ("peer facts", "×0 in embeddings mode", n_feat, MUTED),
        ("v", "", 1, MSG),
        ("age", "", 1, MSG),
    ]
    total = sum(seg[2] for seg in segments)
    assert total == DIMS["head_input"]
    unit = 12.0 / total
    bar = VGroup()
    x = -6.0
    for name, detail, width, color in segments:
        rect = Rectangle(width=width * unit, height=0.55, stroke_color=BG, stroke_width=2, fill_color=color, fill_opacity=0.85 if name != "peer facts" else 0.18)
        rect.move_to([x + width * unit / 2, 1.0, 0])
        bar.add(rect)
        x += width * unit
    s.play(LaggedStart(*(GrowFromEdge(r, LEFT) for r in bar), lag_ratio=0.15), run_time=1.6)
    labels = VGroup()
    for (name, detail, width, color), rect in zip(segments, bar):
        if width == 1:
            continue
        top = T(f"{name} · {width}", 20, color if color != MUTED else MUTED).next_to(rect, UP, buff=0.14)
        lines = [top]
        if detail:
            lines.append(T(detail, 15, MUTED).next_to(top, UP, buff=0.08))
        labels.add(VGroup(*lines))
    tiny = VGroup(bar[4], bar[5])
    tiny_label = VGroup(T("v · 1", 17, MSG), T("age · 1", 17, MSG)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
    tiny_label.next_to(tiny, DOWN, buff=0.16).align_to(tiny, RIGHT)
    s.play(LaggedStart(*(FadeIn(l, shift=DOWN * 0.1) for l in labels), lag_ratio=0.15), FadeIn(tiny_label), run_time=1.6)
    total_label = M(f"concatenated input ∈ ℝ<sup>{total}</sup>", 24).next_to(bar, DOWN, buff=0.75)
    s.play(FadeIn(total_label), run_time=0.6)

    mlp = block([f"Linear({total} → 96) → GELU", "Linear(96 → 64) → GELU", "Linear(64 → 5)"], 4.6, 1.5, FG, 20)
    mlp.next_to(total_label, DOWN, buff=0.35)
    arrow = Arrow(total_label.get_bottom(), mlp.get_top(), buff=0.06, color=MUTED, stroke_width=2.5)
    s.play(GrowArrow(arrow), FadeIn(mlp, shift=DOWN * 0.1), run_time=0.8)
    outs = VGroup()
    for name, color in (("Δx", FG), ("Δy", FG), ("Δz", FG), ("grip", MUTED), ("hold", COND["left_obstacle"])):
        cell = RoundedRectangle(corner_radius=0.08, width=0.95, height=0.55, stroke_color=color, stroke_width=2)
        outs.add(VGroup(cell, T(name, 20, color).move_to(cell)))
    outs.arrange(RIGHT, buff=0.12).next_to(mlp, RIGHT, buff=0.7)
    out_arrow = Arrow(mlp.get_right(), outs.get_left(), buff=0.08, color=MUTED, stroke_width=2.5)
    s.play(GrowArrow(out_arrow), LaggedStart(*(FadeIn(o, shift=RIGHT * 0.1) for o in outs), lag_ratio=0.1), run_time=1.0)
    rule = VGroup(
        T("hold if hold logit ≥ 0", 19, MUTED),
        T("Δ = tanh(·) × 18 mm, norm-clipped to 18 mm", 19, MUTED),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(outs, DOWN, buff=0.25).align_to(outs, LEFT)
    s.play(FadeIn(rule), run_time=0.8)
    s.wait(3.0)


def training(s: Scene):
    head = title("Trained jointly: one arm's loss teaches the other arm's sender")
    s.play(Write(head), run_time=1.4)
    loss = M(
        "<i>L</i> = ‖tanh(Δ̂) − Δ*‖² + 0.3 · BCE(grip) + BCE(hold)      averaged over both arms",
        26,
    ).next_to(head, DOWN, buff=0.35)
    s.play(FadeIn(loss, shift=DOWN * 0.1), run_time=1.0)

    # Gradient path, left side of the frame.
    smol = block(["SmolVLA", "frozen"], 1.9, 0.9, MUTED, 18).move_to(LEFT * 5.6 + UP * 0.5)
    snd = block(["senders['left']"], 2.4, 0.7, SMOL, 17).move_to(LEFT * 5.35 + DOWN * 0.9)
    rcv = block(["receivers['right']"], 2.6, 0.7, XVLA, 17).move_to(LEFT * 2.3 + DOWN * 0.9)
    hd = block(["heads['right']"], 2.2, 0.7, XVLA, 17).move_to(LEFT * 2.3 + UP * 0.5)
    lossbox = block(["right-arm loss"], 2.2, 0.7, GRAD, 17).move_to(LEFT * 2.3 + UP * 1.75)
    fwd = VGroup(
        Arrow(smol.get_bottom(), snd.get_top(), buff=0.06, color=MUTED, stroke_width=2),
        Arrow(snd.get_right(), rcv.get_left(), buff=0.06, color=MSG, stroke_width=2.5),
        Arrow(rcv.get_top(), hd.get_bottom(), buff=0.06, color=MUTED, stroke_width=2),
        Arrow(hd.get_top(), lossbox.get_bottom(), buff=0.06, color=MUTED, stroke_width=2),
    )
    cache_note = T("features cached once, one backbone at a time", 16, MUTED).next_to(smol, UP, buff=0.12)
    s.play(FadeIn(smol), FadeIn(cache_note), run_time=0.6)
    s.play(LaggedStart(*(FadeIn(b) for b in (snd, rcv, hd, lossbox)), lag_ratio=0.2), LaggedStart(*(GrowArrow(a) for a in fwd), lag_ratio=0.2), run_time=1.6)
    grad_path = VMobject(stroke_color=GRAD, stroke_width=6)
    grad_path.set_points_as_corners(
        [lossbox.get_bottom(), hd.get_top(), hd.get_bottom(), rcv.get_top(), rcv.get_left(), snd.get_right()]
    )
    stop = T("no gradient into the frozen backbone", 16, GRAD).next_to(smol, LEFT, buff=0.1).shift(DOWN * 0.0)
    stop.next_to(snd, DOWN, buff=0.15)
    for _ in range(2):
        s.play(ShowPassingFlash(grad_path.copy(), time_width=0.45), run_time=1.3)
    s.play(FadeIn(stop), run_time=0.6)

    # Real loss curves from evaluation.json.
    ox, oy, W, H = 0.6, -2.6, 5.6, 3.6
    max_epoch = max(p["epoch"] for p in CURVES["embeddings"])
    max_loss = max(p["train_loss"] for p in CURVES["embeddings"]) * 1.05

    def to_xy(epoch, value):
        return [ox + epoch / max_epoch * W, oy + value / max_loss * H, 0]

    axes = VGroup(
        Line([ox, oy, 0], [ox + W, oy, 0], stroke_color=MUTED, stroke_width=1.5),
        Line([ox, oy, 0], [ox, oy + H, 0], stroke_color=MUTED, stroke_width=1.5),
    )
    xt = VGroup(*(T(str(e), 16, MUTED).move_to([to_xy(e, 0)[0], oy - 0.22, 0]) for e in (0, 100, 200, 300, 400) if e <= max_epoch + 1))
    yt = VGroup(*(T(f"{v:.1f}", 16, MUTED).move_to([ox - 0.3, to_xy(0, v)[1], 0]) for v in (0.0, 0.5, 1.0) if v <= max_loss))
    xl = T("epoch", 17, MUTED).next_to(axes[0], DOWN, buff=0.38)
    yl = T("loss", 17, MUTED).rotate(np.pi / 2).next_to(axes[1], LEFT, buff=0.55)
    s.play(Create(axes), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), run_time=0.8)

    def polyline(mode, key, color, width=3, dashed=False):
        pts = [to_xy(p["epoch"], p[key]) for p in CURVES[mode]]
        line = VMobject(stroke_color=color, stroke_width=width)
        line.set_points_smoothly(pts) if not dashed else line.set_points_as_corners(pts)
        return line

    emb_train = polyline("embeddings", "train_loss", MSG, 2.5)
    emb_dev = polyline("embeddings", "dev_loss", MSG, 4)
    none_dev = polyline("none", "dev_loss", MUTED, 3)
    s.play(Create(none_dev), run_time=1.4)
    none_label = T("no communication · dev", 17, MUTED).next_to(to_xy(max_epoch, CURVES["none"][-1]["dev_loss"]), UP, buff=0.12).shift(LEFT * 1.2)
    s.play(FadeIn(none_label), run_time=0.5)
    s.play(Create(emb_dev), Create(emb_train.set_stroke(opacity=0.45)), run_time=1.8)
    emb_label = T("learned messages · dev", 17, MSG).next_to(to_xy(max_epoch, CURVES["embeddings"][-1]["dev_loss"]), UP, buff=0.12).shift(LEFT * 1.2)
    s.play(FadeIn(emb_label), run_time=0.5)
    sel = SOURCE["selected_epoch"]
    vline = DashedLine(to_xy(sel, 0), to_xy(sel, max_loss * 0.9), stroke_color=FG, stroke_width=1.5, dash_length=0.08)
    sel_label = T(f"selected epoch {sel} · lowest dev loss", 16, FG).next_to(vline, UP, buff=0.08)
    s.play(Create(vline), FadeIn(sel_label), run_time=0.8)
    opt = T("AdamW · lr 0.001 · weight decay 0.01 · clip 5 · 400 epochs · CPU", 17, MUTED).to_edge(DOWN, buff=0.28)
    s.play(FadeIn(opt), run_time=0.6)
    s.wait(3.0)


def results(s: Scene):
    head = title("Does the receiver use the message?", "80 held-out decisions where the receiver's own view is clear")
    s.play(Write(head[0]), FadeIn(head[1]), run_time=1.2)

    # Hidden-peer subset: receiver sees no local obstacle; half need a hold.
    decisions = []
    for arm in ("left", "right"):
        a = TEST["arms"][arm]
        for i in range(len(a["hold_label"])):
            if not a["local_obstacle"][i]:
                decisions.append((arm, i, a["hold_label"][i]))
    W = 10.0
    x0 = -W / 2
    axis = Line([x0, -2.0, 0], [x0 + W, -2.0, 0], stroke_color=MUTED, stroke_width=1.5)
    ticks = VGroup(*(T(t, 18, MUTED).move_to([x0 + v * W, -2.3, 0]) for v, t in ((0, "0"), (0.5, "0.5"), (1, "1"))))
    xl = T("predicted hold probability", 18, MUTED).next_to(axis, DOWN, buff=0.5)
    thresh = DashedLine([0, -2.0, 0], [0, 1.75, 0], stroke_color=MUTED, stroke_width=1.5)
    rows = VGroup(
        T("peer sees an obstacle · must hold", 20, COND["left_obstacle"]),
        T("scene is clear · must go", 20, COND["clear"]),
    )
    rows[0].move_to([x0, 1.62, 0], aligned_edge=LEFT)
    rows[1].move_to([x0, -0.12, 0], aligned_edge=LEFT)
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.45, 0.45, len(decisions))

    def dots(key):
        group = VGroup()
        for (arm, i, label), j in zip(decisions, jitter):
            prob = TEST["arms"][arm]["hold_prob"][key][i]
            y = (0.95 if label else -0.85) + j
            ok = (prob >= 0.5) == label
            d = Dot([x0 + prob * W, y, 0], radius=0.065, color=GOOD if ok else BAD)
            d.set_stroke(BG, 1)
            group.add(d)
        acc = np.mean([(TEST["arms"][a]["hold_prob"][key][i] >= 0.5) == lab for a, i, lab in decisions])
        return group, acc

    group, acc = dots("delivered")
    assert abs(acc - RESULTS["embeddings"]["test"]["hidden_peer_accuracy"]) < 1e-9
    s.play(Create(axis), FadeIn(ticks), FadeIn(xl), Create(thresh), FadeIn(rows), run_time=1.0)
    s.play(LaggedStart(*(FadeIn(d, scale=0.3) for d in group), lag_ratio=0.01), run_time=1.6)

    def acc_text(name, value):
        return M(f'{name}:  <span foreground="{GOOD if value == 1 else BAD}">{value * 100:.0f}%</span> correct', 26).to_corner(RIGHT + UP, buff=0.5).shift(DOWN * 1.15)

    label = acc_text("message delivered", acc)
    s.play(FadeIn(label), run_time=0.6)
    s.wait(1.6)
    for key, name, result_key in (
        ("zero", "message zeroed", "test_zero_peer"),
        ("shuffle", "messages shuffled", "test_shuffled_peer"),
    ):
        new_group, new_acc = dots(key)
        assert abs(new_acc - RESULTS["embeddings"][result_key]["hidden_peer_accuracy"]) < 1e-9
        new_label = acc_text(name, new_acc)
        s.play(Transform(group, new_group), Transform(label, new_label), run_time=1.6)
        s.wait(1.6)
    same = T("same trained checkpoint; only the delivered message changes", 19, MUTED).to_edge(DOWN, buff=0.25)
    s.play(FadeIn(same), run_time=0.6)
    s.wait(1.4)
    clear_scene(s)

    # Four separately trained interfaces on the same held-out decisions.
    head2 = title("Four interfaces, same held-out episode")
    s.play(Write(head2), run_time=1.0)
    modes = (
        ("none", "no communication", MUTED),
        ("structured_text", "structured text", FG),
        ("embeddings", "learned 64-D messages", MSG),
        ("hybrid", "hybrid", CTX),
    )
    base_y, H, bw = -2.2, 3.6, 1.5
    axis = Line([-4.6, base_y, 0], [4.6, base_y, 0], stroke_color=MUTED, stroke_width=1.5)
    s.play(Create(axis), run_time=0.5)
    bars, texts = VGroup(), VGroup()
    for k, (mode, name, color) in enumerate(modes):
        r = RESULTS[mode]["test"]
        x = -3.45 + k * 2.3
        bar = Rectangle(width=bw, height=max(r["hidden_peer_accuracy"] * H, 0.01), stroke_width=0, fill_color=color, fill_opacity=0.85)
        bar.move_to([x, base_y, 0], aligned_edge=DOWN)
        pct = T(f"{r['hidden_peer_accuracy'] * 100:.0f}%", 26, color).next_to(bar, UP, buff=0.12)
        nm = T(name, 19, color).next_to([x, base_y, 0], DOWN, buff=0.22)
        err = T(f"{r['action_error_mm']:.2f} mm action error", 16, MUTED).next_to(nm, DOWN, buff=0.1)
        bars.add(bar)
        texts.add(VGroup(pct, nm, err))
    ylab = T("hidden-peer hold accuracy", 18, MUTED).rotate(np.pi / 2).move_to([-5.3, base_y + H / 2, 0])
    s.play(FadeIn(ylab), LaggedStart(*(GrowFromEdge(b, DOWN) for b in bars), lag_ratio=0.2), run_time=1.6)
    s.play(LaggedStart(*(FadeIn(t, shift=UP * 0.1) for t in texts), lag_ratio=0.15), run_time=1.2)
    s.wait(2.0)
    caveat = VGroup(
        T(
            f"One held-out episode: {RESULTS['embeddings']['test']['pairs'] * 2} arm decisions from 20 physical states. "
            "One-step diagnostic, not a completed fold.",
            17,
            MUTED,
        ),
        T("Structured text reads exact simulator facts; the 64-D messages come from camera + language features.", 17, MUTED),
    ).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.15)
    s.play(texts.animate.shift(UP * 0.0), FadeIn(caveat), run_time=0.8)
    s.wait(3.4)


def closing(s: Scene):
    lines = VGroup(
        T("Frozen SmolVLA (960-D) and X-VLA (1024-D) features", 30),
        M(f'→ learned 64-D <span foreground="{MSG}">messages</span> through SpacetimeDB', 30),
        M(f'→ directed <span foreground="{CTX}">receivers</span> → replacement action heads', 30),
        T(f"{DIMS['parameters']:,} trained parameters · backbones untouched", 24, MUTED),
    ).arrange(DOWN, buff=0.3)
    s.play(LaggedStart(*(FadeIn(l, shift=UP * 0.15) for l in lines), lag_ratio=0.3), run_time=2.4)
    s.wait(1.6)
    limits = T(
        "Open problems: delayed messages (66.3% hold accuracy) and a live fold (0 grasps in the first trial).",
        20,
        MUTED,
    ).to_edge(DOWN, buff=0.6)
    s.play(FadeIn(limits), run_time=0.8)
    s.wait(3.0)


# One Scene class per section, plus the full sequence.
SECTIONS = (private_views, two_spaces, sender, receiver, head_scene, training, results, closing)


class PrivateViews(Scene):
    def construct(self):
        private_views(self)


class TwoSpaces(Scene):
    def construct(self):
        two_spaces(self)


class Sender(Scene):
    def construct(self):
        sender(self)


class Receiver(Scene):
    def construct(self):
        receiver(self)


class ActionHead(Scene):
    def construct(self):
        head_scene(self)


class Training(Scene):
    def construct(self):
        training(self)


class Results(Scene):
    def construct(self):
        results(self)


class Closing(Scene):
    def construct(self):
        closing(self)


class RelayLatentFull(Scene):
    def construct(self):
        for section in SECTIONS:
            section(self)
            clear_scene(self)
