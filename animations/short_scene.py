"""A ~40 second, plain-language cut of the learned-message animations.

Same real data as ``latent_scenes.py`` (camera frames, the sender's actual
message, held-out results), with almost no technical terms.

    .env/bin/manim -qh short_scene.py RelayLatentShort
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    FadeIn,
    Group,
    GrowArrow,
    GrowFromEdge,
    ImageMobject,
    Indicate,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    ShowPassingFlash,
    SurroundingRectangle,
    VGroup,
)
from PIL import Image

from latent_scenes import (
    COND,
    HERE,
    MSG,
    MUTED,
    RESULTS,
    SAMPLE,
    SMOL,
    TEST,
    XVLA,
    M,
    T,
    block,
    clear_scene,
    grid8,
    strip_image,
    strut_text,
    swap,
)


def caption(text, size=34):
    return M(text, size).to_edge(DOWN, buff=0.55)


class RelayLatentShort(Scene):
    def construct(self):
        s = self
        # 1. Two robots, one hazard only the left robot can see.
        size = 3.3
        left = ImageMobject(str(HERE / "data" / SAMPLE["left_obstacle_png"])).set(height=size)
        right = ImageMobject(str(HERE / "data" / SAMPLE["right_obstacle_png"])).set(height=size)
        left.move_to(LEFT * 3.2 + UP * 0.45)
        right.move_to(RIGHT * 3.2 + UP * 0.45)
        lf = SurroundingRectangle(left, buff=0, color=SMOL, stroke_width=3)
        rf = SurroundingRectangle(right, buff=0, color=XVLA, stroke_width=3)
        ln = T("left robot", 26, SMOL).next_to(left, UP, buff=0.18)
        rn = T("right robot", 26, XVLA).next_to(right, UP, buff=0.18)
        s.play(FadeIn(left), FadeIn(right), Create(lf), Create(rf), FadeIn(ln), FadeIn(rn), run_time=1.2)

        rgb = np.asarray(Image.open(HERE / "data" / SAMPLE["left_obstacle_png"]).convert("RGB")).astype(int)
        mask = (rgb[..., 0] > 200) & (rgb[..., 1] > 120) & (rgb[..., 1] < 200) & (rgb[..., 2] < 110)
        ys, xs = np.nonzero(mask)
        h, w = mask.shape
        center = [left.get_left()[0] + xs.mean() / w * size, left.get_top()[1] - ys.mean() / h * size, 0]
        ring = Circle(radius=np.ptp(xs) / w * size * 0.62, stroke_color=COND["left_obstacle"], stroke_width=5)
        ring.move_to(center)
        cap = caption(f'The left robot sees a <span foreground="{COND["left_obstacle"]}">hazard</span>.')
        s.play(Create(ring), FadeIn(cap, shift=UP * 0.1), run_time=1.0)
        s.wait(1.4)
        cap = swap(s, cap, caption("The right robot can’t."), 0.7, Indicate(rf, color=XVLA, scale_factor=1.03))
        s.wait(1.4)

        # 2. They run different AI models, which don't share a language.
        small = 1.7
        s.play(
            ring.animate.set_opacity(0),
            Group(left, lf).animate.set(height=small).move_to(LEFT * 5.3 + UP * 2.4),
            Group(right, rf).animate.set(height=small).move_to(RIGHT * 5.3 + UP * 2.4),
            ln.animate.set_opacity(0),
            rn.animate.set_opacity(0),
            run_time=1.0,
        )
        s.remove(ring, ln, rn)
        ls = strip_image(SAMPLE["left_stages"]["raw"], 2.5, height=2.9, width=0.4).move_to(LEFT * 5.3 + DOWN * 0.55)
        rs = strip_image(SAMPLE["right_stages"]["raw"], 2.5, height=3.1, width=0.4).move_to(RIGHT * 5.3 + DOWN * 0.55)
        lm = T("SmolVLA", 22, SMOL).next_to(ls, DOWN, buff=0.15)
        rm = T("X-VLA", 22, XVLA).next_to(rs, DOWN, buff=0.15)
        cap = swap(
            s,
            cap,
            caption("Each runs a different AI. They don’t share a language."),
            0.8,
            GrowFromEdge(ls, UP),
            GrowFromEdge(rs, UP),
            FadeIn(lm),
            FadeIn(rm),
        )
        s.wait(2.4)

        # 3. A trained translator on each side passes a short note.
        send = block(["translator"], 2.2, 0.8, SMOL, 22).move_to(LEFT * 2.75 + DOWN * 0.55)
        note = grid8(SAMPLE["left_message"], cell=0.22).move_to(DOWN * 0.55)
        read = block(["translator"], 2.2, 0.8, XVLA, 22).move_to(RIGHT * 2.75 + DOWN * 0.55)
        arrows = VGroup(
            Arrow(ls.get_right(), send.get_left(), buff=0.12, color=MUTED, stroke_width=3),
            Arrow(send.get_right(), note.get_left(), buff=0.12, color=MSG, stroke_width=3),
            Arrow(note.get_right(), read.get_left(), buff=0.12, color=MSG, stroke_width=3),
            Arrow(read.get_right(), rs.get_left(), buff=0.12, color=MUTED, stroke_width=3),
        )
        note_label = T("a 64-number note", 22, MSG).next_to(note, UP, buff=0.18)
        cap = swap(s, cap, caption("We trained a small translator on each side."), 0.8, FadeIn(send), FadeIn(read))
        s.play(
            GrowArrow(arrows[0]),
            GrowArrow(arrows[1]),
            LaggedStart(*(FadeIn(c, scale=0.4) for c in note), lag_ratio=0.01),
            FadeIn(note_label),
            run_time=1.2,
        )
        s.play(GrowArrow(arrows[2]), GrowArrow(arrows[3]), run_time=0.8)
        stop = VGroup(
            RoundedRectangle(
                corner_radius=0.12,
                width=1.6,
                height=0.7,
                stroke_color=COND["left_obstacle"],
                stroke_width=3,
                fill_color=COND["left_obstacle"],
                fill_opacity=0.15,
            ),
            T("STOP", 28, COND["left_obstacle"]),
        )
        stop[1].move_to(stop[0])
        stop.next_to(Group(right, rf), LEFT, buff=0.3)
        # The real decision for this exact sample: the right arm must hold, and does.
        assert TEST["arms"]["right"]["hold_label"][SAMPLE["index"]]
        assert TEST["arms"]["right"]["hold_prob"]["delivered"][SAMPLE["index"]] >= 0.5
        flow = VGroup(*(a.copy().set_stroke(width=7, color=MSG) for a in arrows))
        cap = swap(s, cap, caption("Now the right robot knows to stop."), 0.8)
        s.play(LaggedStart(*(ShowPassingFlash(f, time_width=0.6) for f in flow), lag_ratio=0.3), run_time=1.3)
        s.play(FadeIn(stop, scale=1.2), run_time=0.6)
        s.wait(1.8)
        clear_scene(s, 0.7)

        # 4. Does it work? Held-out results.
        without = RESULTS["none"]["test"]["hidden_peer_accuracy"]
        with_note = RESULTS["embeddings"]["test"]["hidden_peer_accuracy"]
        question = T("When a robot can’t see the hazard itself, does it make the right call?", 30)
        question.to_edge(UP, buff=0.8)
        base_y, H, bw = -1.9, 3.4, 2.0
        axis = Line([-3.2, base_y, 0], [3.2, base_y, 0], stroke_color=MUTED, stroke_width=1.5)
        bars, labels = VGroup(), VGroup()
        for x, value, name, color in ((-1.6, without, "no note", MUTED), (1.6, with_note, "with the note", MSG)):
            bar = Rectangle(width=bw, height=value * H, stroke_width=0, fill_color=color, fill_opacity=0.9)
            bar.move_to([x, base_y, 0], aligned_edge=DOWN)
            pct = T(f"{value * 100:.0f}%", 40, color).next_to(bar, UP, buff=0.15)
            nm = strut_text(name, 26, color).next_to([x, base_y, 0], DOWN, buff=0.25)
            bars.add(bar)
            labels.add(VGroup(pct, nm))
        s.play(FadeIn(question), Create(axis), run_time=0.9)
        s.play(GrowFromEdge(bars[0], DOWN), FadeIn(labels[0]), run_time=0.9)
        guess = T("a coin flip", 22, MUTED).next_to(labels[0][0], UP, buff=0.1)
        s.play(FadeIn(guess), run_time=0.5)
        s.play(GrowFromEdge(bars[1], DOWN), FadeIn(labels[1]), run_time=0.9)
        s.wait(2.2)
        clear_scene(s, 0.7)

        # 5. Close.
        name = T("Relay Lab", 56)
        line = T("Robots on different AI models, sharing what they see.", 30, MUTED).next_to(name, DOWN, buff=0.3)
        fine = T("simulation · one held-out test episode · early results", 20, MUTED).to_edge(DOWN, buff=0.5)
        s.play(FadeIn(name, shift=UP * 0.15), run_time=0.8)
        s.play(FadeIn(line), FadeIn(fine), run_time=0.8)
        s.wait(2.2)
