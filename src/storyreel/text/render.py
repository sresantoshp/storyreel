"""Text clip → RGBA frames: fill/gradient, outline, shadow, box, per-unit animation.

Layout (shaping + rasterising) happens once per clip; each frame only recombines cached
masks, so a 2-hour credits roll costs the same per frame as a 2-second title.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from storyreel.render.fxspec import TEXT_DEFAULTS
from storyreel.render.keyframes import ease
from storyreel.text.shaping import TextBlock, layout


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


@dataclass(frozen=True)
class UnitState:
    alpha: float
    dy: float
    scale: float


def unit_state(
    anim: str, t: float, order: int, stagger: float, dur: float, size: float
) -> UnitState:
    """Animation state of one unit at clip-local time t (seconds)."""
    if anim == "none":
        return UnitState(1.0, 0.0, 1.0)
    start = order * stagger
    if anim == "typewriter":
        return UnitState(1.0 if t >= start else 0.0, 0.0, 1.0)
    p = min(max((t - start) / max(dur, 1e-6), 0.0), 1.0)
    e = ease(p, 0.25, 0.1, 0.25, 1.0)
    if anim == "fade":
        return UnitState(e, 0.0, 1.0)
    if anim == "slide":
        return UnitState(e, (1.0 - e) * size * 0.6, 1.0)
    # pop: grow to 115 % then settle to 100 %
    sc = 0.3 + 0.85 * ease(min(p / 0.6, 1.0), 0.0, 0.0, 0.58, 1.0)
    if p > 0.6:
        sc = 1.15 - 0.15 * ease((p - 0.6) / 0.4, 0.42, 0.0, 0.58, 1.0)
    return UnitState(min(p * 3.0, 1.0), 0.0, sc)


class TextRenderer:
    def __init__(self, props: dict[str, Any]) -> None:
        self.p = {**TEXT_DEFAULTS, **props}
        p = self.p
        margin = p["outline"] + max(abs(p["shadow_dx"]), abs(p["shadow_dy"])) + 2 * p["shadow"]
        if p["box"]:
            margin = max(margin, p["box_padding"])
        self.block: TextBlock = layout(
            str(p["text"]),
            float(p["size"]),
            font=str(p["font"]),
            align=str(p["align"]),
            line_spacing=float(p["line_spacing"]),
            max_width=float(p["max_width"]),
            by_word=p["anim_unit"] == "word",
            margin=margin,
        )
        h, w = self.block.alpha.shape
        fill = np.empty((h, w, 3), np.float32)
        top = hex_rgb(p["color"])
        if p["gradient_to"]:
            ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
            fill[:] = top * (1 - ramp) + hex_rgb(p["gradient_to"]) * ramp
        else:
            fill[:] = top
        self.fill = fill
        n_units = max((u.order for u in self.block.units), default=-1) + 2
        self._lut = np.ones(n_units, np.float32)
        self.static = p["anim"] == "none"
        self._static_frame: tuple[np.ndarray, np.ndarray] | None = None
        self._memo: (
            tuple[tuple[tuple[float, float, float], ...], tuple[np.ndarray, np.ndarray]] | None
        ) = None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.block.alpha.shape
        return w, h

    def _coverage(self, t: float) -> np.ndarray:
        p, blk = self.p, self.block
        anim = str(p["anim"])
        states = {
            u.order: unit_state(
                anim,
                t,
                u.order,
                float(p["anim_stagger"]),
                float(p["anim_duration"]),
                float(p["size"]),
            )
            for u in blk.units
        }
        if anim in ("none", "typewriter", "fade"):
            lut = self._lut.copy()
            for order, st in states.items():
                lut[order + 1] = st.alpha
            return np.asarray(blk.alpha * lut[blk.unit_map + 1], np.float32)
        cov = np.zeros_like(blk.alpha)
        h, w = cov.shape
        for u in blk.units:
            st = states[u.order]
            if st.alpha <= 0.0:
                continue
            sl = (slice(u.y0, u.y1), slice(u.x0, u.x1))
            sprite = blk.alpha[sl] * (blk.unit_map[sl] == u.order) * st.alpha
            sh, sw = sprite.shape
            if st.scale != 1.0:
                nw, nh = max(1, round(sw * st.scale)), max(1, round(sh * st.scale))
                sprite = np.asarray(
                    cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_LINEAR), np.float32
                )
                sh, sw = nh, nw
            cx, cy = (u.x0 + u.x1) / 2, (u.y0 + u.y1) / 2 + st.dy
            x0, y0 = round(cx - sw / 2), round(cy - sh / 2)
            dx0, dy0 = max(x0, 0), max(y0, 0)
            dx1, dy1 = min(x0 + sw, w), min(y0 + sh, h)
            if dx0 < dx1 and dy0 < dy1:
                region = cov[dy0:dy1, dx0:dx1]
                np.maximum(region, sprite[dy0 - y0 : dy1 - y0, dx0 - x0 : dx1 - x0], out=region)
        return cov

    def _signature(self, t: float) -> tuple[tuple[float, float, float], ...]:
        p = self.p
        return tuple(
            (round(s.alpha, 3), round(s.dy, 1), round(s.scale, 3))
            for s in (unit_state(str(p["anim"]), t, u.order, float(p["anim_stagger"]),
                                 float(p["anim_duration"]), float(p["size"]))
                      for u in self.block.units)
        )  # fmt: skip

    def frame(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(rgb, alpha) float32 planes (straight alpha), block-sized, at clip-local time t (s).

        Frames are memoised on the per-unit animation state: a typewriter reveal only
        changes every few frames, and every animation ends static.
        """
        if self.static and self._static_frame is not None:
            return self._static_frame
        sig = self._signature(t)
        if self._memo is not None and self._memo[0] == sig:
            return self._memo[1]
        p = self.p
        cov = self._coverage(t)
        h, w = cov.shape
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)

        def over(src_rgb: np.ndarray, src_a: np.ndarray) -> None:
            nonlocal a
            inv = 1.0 - src_a
            rgb[:] = src_rgb * src_a[..., None] + rgb * inv[..., None]
            a = np.asarray(src_a + a * inv, np.float32)

        if p["box"] and cov.any():
            ys, xs = np.nonzero(cov > 0.01)
            pad = int(p["box_padding"])
            y0, y1 = max(int(np.min(ys)) - pad, 0), min(int(np.max(ys)) + pad, h - 1)
            x0, x1 = max(int(np.min(xs)) - pad, 0), min(int(np.max(xs)) + pad, w - 1)
            box = np.zeros((h, w), np.float32)
            box[y0 : y1 + 1, x0 : x1 + 1] = float(p["box_opacity"])
            over(np.broadcast_to(hex_rgb(p["box_color"]), (h, w, 3)), box)
        body = cov
        if p["outline"] > 0:
            k = int(p["outline"]) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            body = np.asarray(cv2.dilate(cov, kernel), np.float32)
        if p["shadow"] > 0:
            m = np.array([[1, 0, p["shadow_dx"]], [0, 1, p["shadow_dy"]]], np.float32)
            sh = np.asarray(cv2.warpAffine(body, m, (w, h)), np.float32)
            k = int(p["shadow"]) * 2 + 1
            sh = np.asarray(cv2.GaussianBlur(sh, (k, k), 0), np.float32) * 0.8
            over(np.broadcast_to(hex_rgb(p["shadow_color"]), (h, w, 3)), sh)
        if p["outline"] > 0:
            over(np.broadcast_to(hex_rgb(p["outline_color"]), (h, w, 3)), body)
        over(self.fill, cov)
        out = (
            np.asarray(rgb / np.maximum(a, 1e-6)[..., None], np.float32),
            np.asarray(a, np.float32),
        )
        if self.static:
            self._static_frame = out
        self._memo = (sig, out)
        return out
