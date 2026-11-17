"""Illustrated environments: sky (sun/moon/clouds), sea, beach, hills, trees, grass, ice,
houses.

Same contract as the other generators: a pure function of (clip-local time, params, frame
size), so chunks render independently and identically. Anything that does not move is drawn
once per generator instance and reused; the animated parts (clouds, waves, surf, grass,
glints, window flicker) are cheap per-frame passes on top.

Style: flat, layered "paper cut-out" illustration with atmospheric perspective (farther
layers paler, smaller and closer to the horizon), matching mountains/river.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from storyreel.render import draw as d
from storyreel.render.draw import Planes
from storyreel.render.fxspec import SCENERY_DEFAULTS  # noqa: F401  (re-export)


def _f(p: dict[str, Any], k: str) -> float:
    return float(p[k])


class _Scene:
    def __init__(self, p: dict[str, Any], w: int, h: int) -> None:
        self.p, self.w, self.h = p, w, h
        self.rng = np.random.default_rng(int(p.get("seed", 1)))
        self._static: tuple[np.ndarray, np.ndarray | None] | None = None

    def frame(self, t: float) -> Planes:
        raise NotImplementedError

    def static(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self._static is None:
            rgb, a = self.build()
            rgb.setflags(write=False)
            if a is not None:
                a.setflags(write=False)
            self._static = (rgb, a)
        return self._static

    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError


# -- sky ------------------------------------------------------------------------------------
def _cloud_sprite(rng: np.random.Generator, cw: int, color: np.ndarray) -> tuple[np.ndarray, ...]:
    ch = max(8, int(cw * 0.42))
    cv = d.canvas(cw, ch)
    base_y = ch * 0.8
    puffs = int(rng.integers(5, 9))
    shade = d.col(color, 0.94)
    light = d.col(color)
    for i in range(puffs):  # lower, greyer puffs first, brighter tops over them
        x = cw * (0.15 + 0.7 * (i + rng.random()) / puffs)
        r = cw * rng.uniform(0.1, 0.19) * (1.15 - abs(x / cw - 0.5))
        d.circle(cv, x, base_y - r * 0.6, r, shade)
    for i in range(puffs):
        x = cw * (0.15 + 0.7 * (i + rng.random()) / puffs)
        r = cw * rng.uniform(0.08, 0.16) * (1.15 - abs(x / cw - 0.5))
        d.circle(cv, x - r * 0.15, base_y - r * 0.95, r, light)
    d.ellipse(cv, cw * 0.5, base_y, cw * 0.4, ch * 0.12, shade)
    rgb, a = d.planes(cv)
    a = np.asarray(cv2.GaussianBlur(a, (0, 0), max(1.0, cw / 160)), np.float32)
    return rgb, a


class Sky(_Scene):
    def __init__(self, p: dict[str, Any], w: int, h: int) -> None:
        super().__init__(p, w, h)
        n = int(p["clouds"])
        color = d.rgb(p["cloud_color"])
        self.clouds = []
        for _ in range(n):
            cw = int(w * self.rng.uniform(0.14, 0.3))
            spr = _cloud_sprite(self.rng, cw, color)
            depth = self.rng.uniform(0.4, 1.0)
            y = h * (0.04 + 0.34 * self.rng.random() * (1.1 - depth))
            self.clouds.append((spr, self.rng.random(), y, depth))

    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        p, w, h = self.p, self.w, self.h
        k = np.linspace(0, 1, h, dtype=np.float32)[:, None, None] ** 0.9
        sky = d.rgb(p["top"]) * (1 - k) + d.rgb(p["bottom"]) * k
        rgb = np.ascontiguousarray(np.broadcast_to(sky, (h, w, 3)), np.float32)
        body = str(p["body"])
        if body != "none":
            cx, cy = _f(p, "body_x") * w, _f(p, "body_y") * h
            r = _f(p, "body_size") * h
            bc = d.rgb(p["body_color"])
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dist = np.hypot(xx - cx, yy - cy) / r
            glow = _f(p, "glow") * np.exp(-((np.maximum(dist - 1, 0) / 2.2) ** 1.3))
            d.over(rgb, np.broadcast_to(bc, rgb.shape), (glow * 0.85).astype(np.float32))
            disc = np.clip(1.0 - (dist - 1.0) * r, 0, 1).astype(np.float32)
            if body == "moon":  # soft maria on the moon's face
                rng = np.random.default_rng(int(p["seed"]) + 1)
                spots = np.zeros_like(disc)
                for _ in range(6):
                    sx, sy = cx + rng.uniform(-0.5, 0.5) * r, cy + rng.uniform(-0.5, 0.5) * r
                    rs = r * rng.uniform(0.12, 0.3)
                    spots = np.maximum(spots, np.clip(1 - np.hypot(xx - sx, yy - sy) / rs, 0, 1))
                face = bc[None, None, :] * (1 - 0.12 * spots[..., None])
                d.over(rgb, np.broadcast_to(face, rgb.shape), disc)
            else:
                d.over(rgb, np.broadcast_to(bc, rgb.shape), disc)
        n_stars = int(p["stars"])
        if n_stars:
            rng = np.random.default_rng(int(p["seed"]) + 7)
            star_x = (rng.random(n_stars) * w).astype(int)
            star_y = (rng.random(n_stars) ** 1.6 * h * 0.75).astype(int)
            b = (0.35 + 0.65 * rng.random(n_stars) ** 3).astype(np.float32)
            rgb[star_y, star_x] = np.maximum(rgb[star_y, star_x], b[:, None])
            big = b > 0.85
            for x_, y_ in zip(star_x[big], star_y[big], strict=True):
                cv2.circle(rgb, (int(x_), int(y_)), 1, (0.95, 0.95, 1.0), -1, cv2.LINE_AA)
        return rgb, None

    def frame(self, t: float) -> Planes:
        base, _ = self.static()
        if not self.clouds:
            return base, None
        rgb = base.copy()
        opacity = _f(self.p, "cloud_opacity")
        for (srgb, sa), u, y, depth in self.clouds:
            cw = sa.shape[1]
            span = self.w + cw
            x = (u * span + _f(self.p, "cloud_speed") * depth * t) % span - cw
            d.paste(rgb, None, srgb, sa * opacity, int(x), int(y))
        return rgb, None


# -- sea ------------------------------------------------------------------------------------
class Sea(_Scene):
    def __init__(self, p: dict[str, Any], w: int, h: int) -> None:
        super().__init__(p, w, h)
        n = int(p["waves"])
        self.u = self.rng.random((n, 4))
        self.g = self.rng.random((int(160 * _f(p, "glitter")), 3))

    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        p, w, h = self.p, self.w, self.h
        r0 = int(_f(p, "horizon") * h)
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        s = np.linspace(0, 1, h - r0, dtype=np.float32)[:, None] ** 0.7
        rgb[r0:] = (d.rgb(p["far"]) * (1 - s) + d.rgb(p["near"]) * s)[:, None, :]
        a[r0:] = 1.0
        a[r0] = 0.5  # soft horizon line
        return rgb, a

    def frame(self, t: float) -> Planes:
        p, w, h = self.p, self.w, self.h
        base, a = self.static()
        r0 = int(_f(p, "horizon") * h)
        rows = h - r0
        sp = _f(p, "speed")
        z = np.linspace(0.0, 1.0, rows, dtype=np.float32) ** 1.6  # 0 horizon, 1 near
        # swells rolling toward the viewer: brightness bands, denser near the horizon
        band = 1.0 + 0.07 * np.sin(1.0 / (z + 0.08) * 6.0 - t * 1.3 * sp)
        rgb = base.copy()
        rgb[r0:] *= band[:, None, None].astype(np.float32)
        crest = np.zeros((rows, w), np.float32)
        u = self.u
        depth = u[:, 1] ** 1.7
        y = depth * rows
        length = 6 + 90 * depth * (0.5 + u[:, 2])
        x = (u[:, 0] * (w + 200) + sp * (8 + 20 * depth) * t) % (w + 200) - 100
        lift = np.sin(t * (0.6 + u[:, 3]) * sp + u[:, 3] * 6.28)
        alpha = np.clip(lift, 0, 1) * (0.3 + 0.6 * depth)
        for b in range(3):
            sel = np.minimum((alpha * 3).astype(int), 2) == b
            if sel.any():
                segs = [
                    np.array(
                        [[xi - li / 2, yi + li * 0.03], [xi, yi], [xi + li / 2, yi + li * 0.04]]
                    )
                    for xi, yi, li in zip(x[sel], y[sel], length[sel], strict=True)
                ]
                cv2.polylines(
                    crest,
                    [np.round(s_ * 16).astype(np.int32) for s_ in segs],
                    False,
                    0.25 + 0.3 * b,
                    max(1, int(h / 540)),
                    cv2.LINE_AA,
                    4,
                )
        g = self.g
        if len(g):  # sun glitter: a column of twinkling sparks under the sun
            gx = _f(p, "glitter_x") * w + (g[:, 0] - 0.5) * w * 0.18 * (0.3 + g[:, 1])
            gy = g[:, 1] ** 1.5 * rows
            tw = np.clip(np.sin(t * 5 * (0.5 + g[:, 2]) + g[:, 2] * 40), 0, 1) ** 4
            for xi, yi, ti, gi in zip(gx, gy, tw, g[:, 1], strict=True):
                if ti > 0.05:
                    rr = 1 + 3 * gi
                    cv2.line(crest, (int(xi - rr * 2), int(yi)), (int(xi + rr * 2), int(yi)),
                             float(ti), max(1, int(rr / 2)), cv2.LINE_AA)  # fmt: skip
        k = np.ascontiguousarray(crest)[..., None]
        rgb[r0:] = rgb[r0:] * (1 - k) + k
        return rgb, a


# -- beach ----------------------------------------------------------------------------------
class Beach(_Scene):
    def __init__(self, p: dict[str, Any], w: int, h: int) -> None:
        super().__init__(p, w, h)
        xs = np.arange(w, dtype=np.float32)
        ph = self.rng.uniform(0, 6.28, 3)
        self.edge = (
            _f(p, "shore") * h
            + _f(p, "slope") * h * (xs / w - 0.5)
            + 0.012 * h * np.sin(xs / w * 5.0 + ph[0])
            + 0.006 * h * np.sin(xs / w * 13.0 + ph[1])
        ).astype(np.float32)
        self.lace = (0.5 + 0.5 * np.sin(xs / w * 61 + ph[2]) * np.sin(xs / w * 23)).astype(
            np.float32
        )

    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        p, w, h = self.p, self.w, self.h
        top = int(max(0, self.edge.min() - _f(p, "tide") * h * 2 - 4))
        s = np.linspace(0, 1, h - top, dtype=np.float32)[:, None]
        rgb = np.zeros((h, w, 3), np.float32)
        rgb[top:] = (d.rgb(p["sand"]) * (1 - s) + d.rgb(p["sand_near"]) * s)[:, None, :]
        grain = self.rng.random((h - top, w), dtype=np.float32)
        rgb[top:] *= (0.97 + 0.05 * grain)[..., None]
        return rgb, None

    def frame(self, t: float) -> Planes:
        p, w, h = self.p, self.w, self.h
        sand, _ = self.static()
        tide = _f(p, "tide") * h
        phase = (t / max(0.5, _f(p, "period"))) % 1.0
        reach = math.sin(phase * math.pi) ** 2  # wash up, slide back
        top = int(max(0, self.edge.min() - tide * 2 - 4))
        rows = np.arange(top, h, dtype=np.float32)[:, None]
        water_line = self.edge[None, :] - tide * (1.0 - 1.6 * reach)
        wet_line = self.edge[None, :] + tide * 0.9
        a = np.zeros((h, w), np.float32)
        a[top:] = np.clip(rows - water_line + 0.5, 0, 1)
        rgb = sand.copy()
        wet = np.clip((wet_line - rows) / max(1.0, tide), 0, 1) * a[top:]
        d.over(rgb[top:], np.broadcast_to(d.rgb(p["wet"]), rgb[top:].shape), wet * 0.55)
        foam_w = 3 + 0.008 * h * (0.5 + self.lace)
        foam = np.clip(1 - np.abs(rows - water_line - foam_w) / foam_w, 0, 1)
        foam *= 0.65 + 0.35 * reach
        d.over(rgb[top:], np.broadcast_to(d.rgb(p["foam"]), rgb[top:].shape), foam)
        a[top:] = np.maximum(a[top:], foam * 0.9)
        return rgb, a


# -- hills ----------------------------------------------------------------------------------
class Hills(_Scene):
    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        p, w, h = self.p, self.w, self.h
        n = max(1, int(p["layers"]))
        cv = d.canvas(w, h)
        horizon = _f(p, "horizon") * h
        xs = np.linspace(-4, w + 4, 200)
        tops: list[np.ndarray] = []
        for i in range(n):
            k = i / max(n - 1, 1)
            amp = _f(p, "height") * h * (1 - 0.35 * k)
            ph = self.rng.uniform(0, 6.28, 3)
            f1 = self.rng.uniform(1.2, 2.2)
            y = (
                horizon
                - amp * (0.55 + 0.25 * np.sin(xs / w * f1 * 3.14 + ph[0]))
                + amp * 0.18 * np.sin(xs / w * 7 + ph[1])
                + k * amp * 0.75
            )
            tops.append(y)
            c = d.mix(p["far"], p["near"], k)
            d.poly(
                cv, np.concatenate([np.stack([xs, y], 1), [[w + 4, h + 4], [-4, h + 4]]]), d.col(c)
            )
            # soft light on the crest
            d.line(cv, np.stack([xs, y + 2], 1), d.col(c, 1.08), max(1, h / 360))
            if i == n - 1 and _f(p, "tea") > 0:
                rows = int(10 + 20 * _f(p, "tea"))
                for j in range(1, rows):
                    off = (j / rows) ** 1.4 * (h - y.min())
                    d.line(cv, np.stack([xs, y + off], 1), d.col(c, 0.82), max(1, h / 300))
        houses = int(p["houses"])
        if houses and n >= 2:
            mid = tops[n // 2 if n > 2 else 0]
            for _ in range(houses):
                x = self.rng.uniform(0.08, 0.92) * w
                y = float(np.interp(x, xs, mid)) + h * 0.01
                _house(cv, x, y, h * self.rng.uniform(0.035, 0.05), p, self.rng, _f(p, "lit"))
        rgb, a = d.planes(cv)
        return rgb, a

    def frame(self, t: float) -> Planes:
        return self.static()


def _house(
    cv: np.ndarray,
    x: float,
    base: float,
    s: float,
    p: dict[str, Any],
    rng: np.random.Generator,
    lit: float,
    style: str = "cottage",
) -> None:
    wall, roof = d.rgb(p["wall"]), d.rgb(p["roof"])
    jitter = 0.9 + 0.2 * rng.random()
    wall = np.clip(wall * jitter, 0, 1)
    ww, wh = s * rng.uniform(1.1, 1.6), s
    win = d.col(d.mix("#3a4656", "#ffd36b", lit))
    if style == "hut":
        d.ellipse(cv, x, base - wh * 0.45, ww * 0.55, wh * 0.5, d.col(wall))
        d.poly(
            cv,
            [
                (x - ww * 0.75, base - wh * 0.75),
                (x, base - wh * 1.9),
                (x + ww * 0.75, base - wh * 0.75),
            ],
            d.col(roof),
        )
        d.poly(
            cv,
            [
                (x - ww * 0.12, base),
                (x - ww * 0.12, base - wh * 0.5),
                (x + ww * 0.12, base - wh * 0.5),
                (x + ww * 0.12, base),
            ],
            d.col("#3b2a1c"),
        )
        return
    d.poly(
        cv,
        [(x - ww / 2, base), (x - ww / 2, base - wh), (x + ww / 2, base - wh), (x + ww / 2, base)],
        d.col(wall),
    )
    d.poly(
        cv,
        [
            (x - ww * 0.62, base - wh * 0.95),
            (x - ww * 0.1, base - wh * 1.75),
            (x + ww * 0.1, base - wh * 1.75),
            (x + ww * 0.62, base - wh * 0.95),
        ],
        d.col(roof),
    )
    d.poly(
        cv,
        [
            (x - ww * 0.32, base - wh * 0.62),
            (x - ww * 0.32, base - wh * 0.35),
            (x - ww * 0.12, base - wh * 0.35),
            (x - ww * 0.12, base - wh * 0.62),
        ],
        win,
    )
    d.poly(
        cv,
        [
            (x + ww * 0.08, base),
            (x + ww * 0.08, base - wh * 0.55),
            (x + ww * 0.3, base - wh * 0.55),
            (x + ww * 0.3, base),
        ],
        d.col(roof, 0.6),
    )


class Houses(_Scene):
    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        p, w, h = self.p, self.w, self.h
        cv = d.canvas(w, h)
        n = int(p["count"])
        spread = _f(p, "spread")
        xs = sorted(w * (0.5 + spread * (self.rng.random(n) - 0.5)))
        lit = _f(p, "lit")
        for x in xs:
            s = _f(p, "size") * h * self.rng.uniform(0.85, 1.15)
            base = _f(p, "base") * h + self.rng.uniform(-0.01, 0.01) * h
            _house(cv, x, base, s, p, self.rng, lit, str(p["style"]))
        return d.planes(cv)

    def frame(self, t: float) -> Planes:
        return self.static()


# -- trees ----------------------------------------------------------------------------------
def _tree(
    cv: np.ndarray,
    style: str,
    x: float,
    base: float,
    size: float,
    leaf_c: np.ndarray,
    trunk_c: np.ndarray,
    rng: np.random.Generator,
) -> None:
    lc, dark, light = d.col(leaf_c), d.col(leaf_c, 0.78), d.col(leaf_c, 1.18)
    tc = d.col(trunk_c)
    if style == "pine":
        tw = size * 0.05
        d.poly(
            cv,
            [
                (x - tw, base),
                (x - tw, base - size * 0.3),
                (x + tw, base - size * 0.3),
                (x + tw, base),
            ],
            tc,
        )
        tiers = int(rng.integers(3, 5))
        for i in range(tiers):
            k = i / tiers
            yb = base - size * (0.18 + 0.72 * k)
            half = size * 0.3 * (1 - 0.6 * k)
            top = yb - size * 0.36
            d.poly(cv, [(x - half, yb), (x, top), (x + half, yb)], dark)
            d.poly(cv, [(x - half * 0.1, yb - size * 0.02), (x, top), (x + half, yb)], lc)
    elif style == "round":
        tw = size * 0.06
        d.poly(
            cv,
            [
                (x - tw, base),
                (x - tw * 0.7, base - size * 0.55),
                (x + tw * 0.7, base - size * 0.55),
                (x + tw, base),
            ],
            tc,
        )
        cy = base - size * 0.68
        for _ in range(7):
            d.circle(
                cv,
                x + rng.uniform(-0.22, 0.22) * size,
                cy + rng.uniform(-0.12, 0.15) * size,
                size * rng.uniform(0.17, 0.24),
                dark,
            )
        for _ in range(5):
            d.circle(
                cv,
                x + rng.uniform(-0.18, 0.1) * size,
                cy + rng.uniform(-0.2, 0.02) * size,
                size * rng.uniform(0.12, 0.18),
                lc,
            )
        d.circle(cv, x - size * 0.08, cy - size * 0.12, size * 0.09, light)
    elif style == "palm":
        lean = rng.uniform(-0.25, 0.25) * size
        crown = (x + lean, base - size * 0.85)
        pts = d.bezier((x, base), (x - lean * 0.2, base - size * 0.5), crown, 12)
        for i in range(len(pts) - 1):
            r = size * (0.024 - 0.008 * i / len(pts))
            d.limb(cv, pts[i], pts[i + 1], r, r * 0.95, tc if i % 2 else d.col(trunk_c, 0.85))
        for j in range(9):
            ang = -math.pi / 2 + (j - 4) / 4 * 1.45 + rng.uniform(-0.1, 0.1)
            ln = size * rng.uniform(0.45, 0.6)
            tip = (crown[0] + math.cos(ang) * ln, crown[1] + math.sin(ang) * ln * 0.55 + ln * 0.35)
            d.poly(
                cv,
                d.leaf(crown, tip, size * 0.1, 0.12 * (1 if math.cos(ang) > 0 else -1)),
                lc if j % 2 else dark,
            )
        d.circle(cv, crown[0], crown[1] + size * 0.03, size * 0.03, d.col("#6b4a2a"))
    elif style == "bamboo":
        for k in range(int(rng.integers(3, 6))):
            bx = x + rng.uniform(-0.12, 0.12) * size
            hgt = size * rng.uniform(0.8, 1.15)
            r = size * 0.018
            seg = hgt / 7
            for i in range(7):
                y0 = base - i * seg
                d.limb(
                    cv,
                    (bx, y0),
                    (bx + i * 0.4, y0 - seg * 0.92),
                    r,
                    r,
                    d.col(leaf_c, 1.1 - 0.05 * k),
                )
            for i in range(3, 8):
                y0 = base - i * seg
                side = 1 if (i + k) % 2 else -1
                d.poly(
                    cv,
                    d.leaf((bx, y0), (bx + side * size * 0.16, y0 + size * 0.04), size * 0.03, 0.1),
                    dark,
                )
    else:  # jungle: broad-leaf tree with big leaves
        tw = size * 0.07
        d.poly(
            cv,
            [
                (x - tw, base),
                (x - tw * 0.5, base - size * 0.6),
                (x + tw * 0.5, base - size * 0.6),
                (x + tw, base),
            ],
            tc,
        )
        cy = base - size * 0.72
        for _ in range(16):
            ang = rng.uniform(0, 6.28)
            ln = size * rng.uniform(0.16, 0.3)
            b0 = (x + rng.uniform(-0.1, 0.1) * size, cy + rng.uniform(-0.1, 0.1) * size)
            tip = (b0[0] + math.cos(ang) * ln, b0[1] + math.sin(ang) * ln * 0.7)
            d.poly(
                cv,
                d.leaf(b0, tip, ln * 0.45, rng.uniform(-0.1, 0.1)),
                dark if rng.random() < 0.5 else lc,
            )


class Trees(_Scene):
    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        p, w, h = self.p, self.w, self.h
        cv = d.canvas(w, h)
        style = str(p["style"])
        styles = ["pine", "round", "round", "pine"] if style == "mixed" else [style]
        rows = max(1, int(p["rows"]))
        count = int(p["count"])
        haze = d.rgb(p["haze"])
        side = str(p["side"])
        for r in range(rows):
            k = (r + 1) / rows  # 1 = nearest row
            size = _f(p, "size") * h * (0.45 + 0.55 * k)
            base = _f(p, "base") * h + (1 - _f(p, "base")) * h * 0.5 * (k - 1 / rows)
            leaf_c = d.mix(haze, d.rgb(p["color"]), 0.45 + 0.55 * k)
            trunk_c = d.mix(haze, d.rgb(p["trunk"]), 0.45 + 0.55 * k)
            n = max(1, int(count * (1.4 - 0.6 * k)))
            for _ in range(n):
                u = self.rng.random()
                if side == "left":
                    x = u * 0.35 * w
                elif side == "right":
                    x = w - u * 0.35 * w
                elif side == "both":
                    x = u * 0.32 * w if self.rng.random() < 0.5 else w - u * 0.32 * w
                else:
                    x = u * w
                st = styles[int(self.rng.integers(len(styles)))]
                _tree(
                    cv,
                    st,
                    x,
                    base + self.rng.uniform(-0.01, 0.02) * h,
                    size * self.rng.uniform(0.8, 1.2),
                    leaf_c,
                    trunk_c,
                    self.rng,
                )
        if style == "jungle":  # foreground frame: hanging vines and big leaves at the edges
            dark = d.rgb(p["color"]) * 0.6
            for _ in range(10):
                x = self.rng.uniform(0, w)
                pts = d.bezier(
                    (x, -5),
                    (x + self.rng.uniform(-60, 60), h * 0.25),
                    (x + self.rng.uniform(-30, 30), h * self.rng.uniform(0.25, 0.55)),
                )
                d.line(cv, pts, d.col(dark), max(2, h / 300))
                for q in pts[4::3]:
                    d.poly(
                        cv,
                        d.leaf(
                            q,
                            (q[0] + self.rng.uniform(-1, 1) * h * 0.03, q[1] + h * 0.025),
                            h * 0.012,
                        ),
                        d.col(dark, 1.3),
                    )
            for _ in range(26):
                left = self.rng.random() < 0.5
                b0 = (0.0 if left else float(w), self.rng.uniform(0, h * 0.95))
                ln = h * self.rng.uniform(0.12, 0.28)
                ang = self.rng.uniform(-0.6, 0.6) + (0 if left else math.pi)
                tip = (b0[0] + math.cos(ang) * ln, b0[1] + math.sin(ang) * ln)
                d.poly(
                    cv,
                    d.leaf(b0, tip, ln * 0.42, self.rng.uniform(-0.15, 0.15)),
                    d.col(dark, self.rng.uniform(0.8, 1.4)),
                )
            for _ in range(14):  # canopy from the top
                b0 = (self.rng.uniform(0, w), -10.0)
                ln = h * self.rng.uniform(0.1, 0.22)
                tip = (b0[0] + self.rng.uniform(-0.5, 0.5) * ln, ln)
                d.poly(cv, d.leaf(b0, tip, ln * 0.45), d.col(dark, self.rng.uniform(0.8, 1.3)))
        return d.planes(cv)

    def frame(self, t: float) -> Planes:
        return self.static()


# -- grass ----------------------------------------------------------------------------------
class Grass(_Scene):
    def __init__(self, p: dict[str, Any], w: int, h: int) -> None:
        super().__init__(p, w, h)
        n = int(p["density"])
        self.u = self.rng.random((n, 4))
        nf = int(p["flowers"])
        self.f = self.rng.random((nf, 3))
        self.fcols = [d.rgb(c.strip()) for c in str(p["flower_colors"]).split(",") if c.strip()]

    def frame(self, t: float) -> Planes:
        p, w, h = self.p, self.w, self.h
        gh = _f(p, "height") * h
        top = int(max(0, h - gh * 1.6))
        bh = h - top
        cv = d.canvas(w, bh)
        u = self.u
        base_x = u[:, 0] * w
        hgt = gh * (0.5 + 0.6 * u[:, 1])
        sway = _f(p, "sway") * gh * 0.18 * np.sin(t * 1.4 + base_x / w * 6 + u[:, 2] * 2)
        base_y = bh + 2 - u[:, 3] * gh * 0.15
        dark, light = d.rgb(p["color"]), d.rgb(p["tip"])
        for b in range(3):
            sel = np.minimum((u[:, 2] * 3).astype(int), 2) == b
            blades = [
                d.leaf((bx, by), (bx + sw, by - hh), max(2.0, h / 220), 0.08)
                for bx, by, hh, sw in zip(
                    base_x[sel], base_y[sel], hgt[sel], sway[sel], strict=True
                )
            ]
            d.polys(cv, blades, d.col(d.mix(dark, light, b / 2)))
        for fx, fh, fc in self.f:
            x0 = fx * w
            hh = gh * (0.7 + 0.5 * fh)
            sx = _f(p, "sway") * gh * 0.15 * math.sin(t * 1.4 + fx * 6)
            d.line(cv, [(x0, bh), (x0 + sx, bh - hh)], d.col(dark), max(1, h / 400))
            c = (
                self.fcols[int(fc * len(self.fcols)) % len(self.fcols)]
                if self.fcols
                else d.rgb("#ffffff")
            )
            r = max(2.0, h / 160)
            for k in range(5):
                a_ = k * 1.2566 + t * 0.2
                d.circle(
                    cv, x0 + sx + math.cos(a_) * r, bh - hh + math.sin(a_) * r, r * 0.7, d.col(c)
                )
            d.circle(cv, x0 + sx, bh - hh, r * 0.5, d.col("#f2c230"))
        srgb, sa = d.planes(cv)
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        rgb[top:], a[top:] = srgb, sa
        return rgb, a


# -- ice ------------------------------------------------------------------------------------
class Ice(_Scene):
    def __init__(self, p: dict[str, Any], w: int, h: int) -> None:
        super().__init__(p, w, h)
        self.g = self.rng.random((int(60 * _f(p, "glint")), 3))

    def build(self) -> tuple[np.ndarray, np.ndarray | None]:
        p, w, h = self.p, self.w, self.h
        r0 = _f(p, "horizon") * h
        cv = d.canvas(w, h)
        ice, shade = d.rgb(p["ice"]), d.rgb(p["shade"])
        for _ in range(int(p["bergs"])):  # icebergs on the horizon
            x = self.rng.uniform(0, w)
            bw = w * self.rng.uniform(0.06, 0.16)
            bh = bw * self.rng.uniform(0.35, 0.7)
            peak = x + self.rng.uniform(-0.3, 0.3) * bw
            pts = [
                (x - bw / 2, r0 + 2),
                (x - bw * 0.3, r0 - bh * 0.6),
                (peak, r0 - bh),
                (x + bw * 0.25, r0 - bh * 0.7),
                (x + bw / 2, r0 + 2),
            ]
            d.poly(cv, pts, d.col(ice, 1.02))
            d.poly(
                cv,
                [
                    (peak, r0 - bh),
                    (x + bw * 0.25, r0 - bh * 0.7),
                    (x + bw / 2, r0 + 2),
                    (peak + bw * 0.05, r0 + 2),
                ],
                d.col(shade),
            )
        d.poly(cv, [(0, r0), (w, r0), (w, h), (0, h)], d.col(shade, 1.05))
        rgb, a = d.planes(cv)
        rows = np.arange(h, dtype=np.float32)
        s = np.clip((rows - r0) / max(1.0, h - r0), 0, 1)[:, None, None]
        ground = d.mix(shade, ice, 0.0) * (1 - s) + ice * s
        below = (rows >= r0)[:, None]
        rgb = np.where(below[..., None], np.broadcast_to(ground, rgb.shape), rgb).astype(np.float32)
        cracks = np.zeros((h, w), np.float32)
        for _ in range(int(p["cracks"])):
            y = self.rng.uniform(r0 + 0.05 * h, h)
            x = self.rng.uniform(0, w)
            pts = [(x, y)]
            ln = (y - r0) / (h - r0) * w * 0.12 + 10
            for _ in range(6):
                x += self.rng.uniform(0.3, 1.0) * ln / 4 * (1 if self.rng.random() < 0.5 else -1)
                y += self.rng.uniform(-0.3, 0.3) * ln / 8
                pts.append((x, y))
            cv2.polylines(
                cracks,
                [np.round(np.array(pts) * 16).astype(np.int32)],
                False,
                0.6,
                max(1, int(h / 500)),
                cv2.LINE_AA,
                4,
            )
        d.over(rgb, np.broadcast_to(shade * 0.8, rgb.shape), cracks)
        return rgb, a

    def frame(self, t: float) -> Planes:
        rgb0, a = self.static()
        if not len(self.g):
            return rgb0, a
        p, w, h = self.p, self.w, self.h
        r0 = _f(p, "horizon") * h
        rgb = rgb0.copy()
        g = self.g
        tw = np.clip(np.sin(t * 2.5 * (0.5 + g[:, 2]) + g[:, 2] * 50), 0, 1) ** 6
        for (gx, gy, _), ti in zip(g, tw, strict=True):
            if ti < 0.05:
                continue
            x, y = int(gx * w), int(r0 + gy * (h - r0))
            r = int(2 + 6 * gy)
            for dx, dy in ((r, 0), (0, r)):
                seg = rgb[max(0, y - dy) : y + dy + 1, max(0, x - dx) : x + dx + 1]
                seg += (1.0 - seg) * float(ti) * 0.9
        return rgb, a


KINDS: dict[str, type[_Scene]] = {
    "sky": Sky,
    "sea": Sea,
    "beach": Beach,
    "hills": Hills,
    "trees": Trees,
    "grass": Grass,
    "ice": Ice,
    "houses": Houses,
}
