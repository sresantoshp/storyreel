"""Procedural environments: gradient, starfield, rain, snow, fog, fire, particles,
and illustrated landscape pieces: mountains, river, birds.

Every generator is a pure function of (clip-local time, params, seed, frame size). No state
is carried between frames, so any chunk can render any frame independently and results are
bit-identical across chunk boundaries and re-renders. Particles get positions from a seeded
RNG once (per generator instance) and move along closed-form paths with wrap-around.

Output: (rgb float32 HxWx3, alpha HxW or None = opaque), full frame size.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from storyreel.render import actors, scenery
from storyreel.render.fxspec import GENERATORS
from storyreel.text.render import hex_rgb

Planes = tuple[np.ndarray, np.ndarray | None]


def _value_noise(rng: np.random.Generator, gw: int, gh: int) -> np.ndarray:
    return rng.random((gh, gw), dtype=np.float32)


class Generator:
    def __init__(self, gtype: str, params: dict[str, Any], width: int, height: int) -> None:
        self.type = gtype
        self.p = {**GENERATORS[gtype], **params}
        self.w, self.h = width, height
        self._solid_rgb: np.ndarray | None = None
        self._static: Planes | None = None
        self._river_geom: tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0))
        self._ext: scenery._Scene | actors.Actor | None = None
        if gtype in scenery.KINDS:
            self._ext = scenery.KINDS[gtype](self.p, width, height)
        elif gtype == "actor":
            self._ext = actors.Actor(self.p, width, height)
        seed = int(self.p.get("seed", 1))
        rng = np.random.default_rng(seed)
        n = int(self.p.get("count", self.p.get("ripples", 0)))  # particles per type
        self.u = rng.random((n, 4), dtype=np.float32)  # per-particle uniforms
        if gtype in ("fog", "fire"):
            # Tileable noise octaves; animated by scrolling, upscaled per frame.
            self.noise = [_value_noise(rng, 64, 64), _value_noise(rng, 128, 128)]

    def frame(self, t: float) -> Planes:
        if self._ext is not None:
            return self._ext.frame(t)
        out: Planes = getattr(self, f"_{self.type}")(t)
        return out

    # -- helpers -----------------------------------------------------------------------
    def _solid(self, color: str) -> np.ndarray:
        """Constant colour plane, built once per generator (read-only, shared per frame)."""
        if self._solid_rgb is None:
            rgb = np.empty((self.h, self.w, 3), np.float32)
            rgb[:] = hex_rgb(color)
            rgb.setflags(write=False)
            self._solid_rgb = rgb
        return self._solid_rgb

    def _blank(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((self.h, self.w, 3), np.float32), np.zeros((self.h, self.w), np.float32)

    def _scrolled(
        self, tile: np.ndarray, ox: float, oy: float, cell: float, w: int, h: int
    ) -> np.ndarray:
        """Upscale a tileable noise grid to (w, h), scrolled by (ox, oy) pixels."""
        th, tw = tile.shape
        gx = int(math.ceil(w / cell)) + 2
        gy = int(math.ceil(h / cell)) + 2
        sx, sy = int(math.floor(ox / cell)), int(math.floor(oy / cell))
        idx_y = (np.arange(gy) + sy) % th
        idx_x = (np.arange(gx) + sx) % tw
        grid = tile[np.ix_(idx_y, idx_x)]
        big = cv2.resize(grid, (int(gx * cell), int(gy * cell)), interpolation=cv2.INTER_CUBIC)
        fx, fy = int((ox / cell - sx) * cell), int((oy / cell - sy) * cell)
        return big[fy : fy + h, fx : fx + w]

    def _low(self) -> tuple[int, int, float]:
        """Working size for soft generators (fog, fire): quarter resolution, upscaled after."""
        k = 4.0
        return max(8, int(math.ceil(self.w / k))), max(8, int(math.ceil(self.h / k))), k

    def _up(self, x: np.ndarray) -> np.ndarray:
        """Bicubic upscale to frame size, clamped: bicubic overshoots outside [0, 1]."""
        out = np.asarray(cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_CUBIC), np.float32)
        np.clip(out, 0.0, 1.0, out=out)
        return out

    # -- generators --------------------------------------------------------------------
    def _gradient(self, t: float) -> Planes:
        """Linear gradient; computed at quarter resolution, cached when it does not drift."""
        p = self.p
        drift = float(p["drift"])
        if drift == 0.0 and self._static is not None:
            return self._static
        lw, lh, _ = self._low()
        a = math.radians(float(p["angle"]) + drift * t)
        yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        d = (xx / lw - 0.5) * math.cos(a) + (yy / lh - 0.5) * math.sin(a)
        k = np.clip(d + 0.5, 0, 1)[..., None]
        rgb = (hex_rgb(p["top"]) * (1 - k) + hex_rgb(p["bottom"]) * k).astype(np.float32)
        out: Planes = (self._up(rgb), None)
        if drift == 0.0:
            out[0].setflags(write=False)
            self._static = out
        return out

    def _starfield(self, t: float) -> Planes:
        a = np.zeros((self.h, self.w), np.float32)
        u, w, h = self.u, self.w, self.h
        z = (u[:, 2] - float(self.p["speed"]) * t) % 1.0 + 0.02  # depth, wraps
        x = (u[:, 0] - 0.5) / z * 0.5 * w + w / 2
        y = (u[:, 1] - 0.5) / z * 0.5 * h + h / 2
        b = (np.clip((1.0 - z) * 1.2, 0, 1) * (0.5 + 0.5 * u[:, 3])).astype(np.float32)
        r = np.clip((1.0 - z) * 3.0, 1, 3).astype(int)
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        small = inside & (r <= 1)
        # most stars are 1 px: write them directly; draw only the few near (big) ones
        np.maximum.at(a, (y[small].astype(int), x[small].astype(int)), b[small])
        for xi, yi, bi, ri in zip(x[inside & ~small], y[inside & ~small], b[inside & ~small],
                                  r[inside & ~small], strict=True):  # fmt: skip
            cv2.circle(a, (int(xi), int(yi)), int(ri), float(bi), -1, cv2.LINE_AA)
        c = hex_rgb(self.p["color"])
        # stars over black space: an opaque layer, usable as a background
        return np.asarray(cv2.merge([a * c[0], a * c[1], a * c[2]]), np.float32), None

    def _rain(self, t: float) -> Planes:
        a = np.zeros((self.h, self.w), np.float32)
        p, u, w, h = self.p, self.u, self.w, self.h
        ang = math.radians(float(p["angle"]))
        speed = float(p["speed"]) * (0.7 + 0.6 * u[:, 2])
        length = float(p["length"]) * (0.6 + 0.8 * u[:, 2])
        span_y = h + 2 * float(p["length"])
        y = (u[:, 1] * span_y + speed * t) % span_y - float(p["length"])
        drift = math.tan(ang)
        span_x = w + span_y * abs(drift)
        x = (u[:, 0] * span_x + drift * y) % span_x - span_y * max(drift, 0.0)
        dx, dy = math.sin(ang), math.cos(ang)
        seg = np.stack([x, y, x + dx * length, y + dy * length], -1).astype(np.int32)
        # one polylines call per (thickness, brightness bucket) instead of a call per drop
        bright = np.minimum((u[:, 3] * 4).astype(int), 3)
        for thick in (1, 2):
            for b in range(4):
                sel = (bright == b) & ((u[:, 3] > 0.8) == (thick == 2))
                if sel.any():
                    lines = list(seg[sel].reshape(-1, 2, 2))
                    cv2.polylines(a, lines, False, 0.5 + 0.125 * (b + 0.5), thick, cv2.LINE_AA)
        return self._solid(p["color"]), np.asarray(a * float(p["opacity"]), np.float32)

    def _snow(self, t: float) -> Planes:
        a = np.zeros((self.h, self.w), np.float32)
        p, u, w, h = self.p, self.u, self.w, self.h
        size = float(p["size"])
        speed = float(p["speed"]) * (0.5 + u[:, 2])
        y = (u[:, 1] * (h + 20) + speed * t) % (h + 20) - 10
        x = (u[:, 0] * w + float(p["sway"]) * np.sin(t * (0.5 + u[:, 3]) + u[:, 3] * 6.28)) % w
        r = np.maximum(1, size * (0.4 + u[:, 2]))
        for xi, yi, ri in zip(x, y, r, strict=True):
            cv2.circle(a, (int(xi), int(yi)), int(ri), 1.0, -1, cv2.LINE_AA)
        a = np.asarray(cv2.GaussianBlur(a, (3, 3), 0), np.float32)
        return self._solid("#ffffff"), np.asarray(a * float(p["opacity"]), np.float32)

    def _fog(self, t: float) -> Planes:
        p = self.p
        cell = float(p["scale"])
        sp = float(p["speed"])
        lw, lh, k = self._low()
        n1 = self._scrolled(self.noise[0], sp * t / k, 0.15 * sp * t / k, cell / k, lw, lh)
        n2 = self._scrolled(self.noise[1], -0.6 * sp * t / k, 0.0, cell / 2.5 / k, lw, lh)
        n = np.clip(0.65 * n1 + 0.35 * n2, 0, 1)
        # drifting patches of haze (smoothstep of the noise), thicker near the ground
        v = np.clip((n - 0.25) / 0.5, 0, 1)
        v = v * v * (3 - 2 * v)
        ground = np.linspace(0.5, 1.0, lh, dtype=np.float32)[:, None]
        a = float(p["density"]) * (0.15 + 0.85 * v * v) * ground
        return self._solid(p["color"]), self._up(np.clip(a, 0, 1))

    def _fire(self, t: float) -> Planes:
        p = self.p
        sp = float(p["speed"])
        lw, lh, k = self._low()
        # sampling window moves down the noise, so flame features rise
        n1 = self._scrolled(self.noise[0], 0.0, sp * t / k, 70.0 / k, lw, lh)
        n2 = self._scrolled(
            self.noise[1], 13.0 * math.sin(t) / k, 1.7 * sp * t / k, 28.0 / k, lw, lh
        )
        n = 0.6 * n1 + 0.4 * n2
        height = float(p["height"])
        yy = np.linspace(0.0, 1.0, lh, dtype=np.float32)[:, None]  # 0 at top, 1 at bottom
        width = float(p["width"])
        if width > 0:  # one campfire: a flame cone over the logs
            bottom = float(p["base"])
            xx = np.linspace(0.0, 1.0, lw, dtype=np.float32)[None, :]
            up = np.clip((bottom - yy) / max(height, 1e-3), 0, 1)  # 0 at the logs, 1 at tip
            half = width / 2 * (1 - up) ** 1.3 + 1e-3
            across = np.clip(1 - np.abs(xx - float(p["x"])) / half, 0, 1)
            base = np.where(yy <= bottom, across * (1 - up * 0.6), 0.0).astype(np.float32)
        else:
            base = np.clip((yy - (1 - height)) / max(height, 1e-3), 0, 1)
        heat = np.clip(base * (0.35 + 1.1 * n) - 0.15 * (1 - base), 0, 1)
        heat = heat * float(p["intensity"])
        # palette at low resolution: dark red -> orange -> yellow -> white; upscale once
        rgb = np.dstack([np.clip(heat * 2.2, 0, 1), np.clip(heat * 1.8 - 0.45, 0, 1),
                         np.clip(heat * 2.5 - 1.6, 0, 1)]).astype(np.float32)  # fmt: skip
        rgb_f, a_f = self._up(rgb), self._up(np.clip(heat * 1.6, 0, 1))
        if width > 0:  # logs under the flames (drawn over the fire's base)
            w_, h_ = self.w, self.h
            cx, by = float(p["x"]) * w_, float(p["base"]) * h_
            lg = width * w_ * 0.55
            for ang in (-14.0, 14.0):
                rad = math.radians(ang)
                dx, dy = math.cos(rad) * lg, math.sin(rad) * lg
                pts = np.array([[cx - dx, by - dy], [cx + dx, by + dy]], np.int32)
                cv2.line(
                    rgb_f,
                    tuple(pts[0]),
                    tuple(pts[1]),
                    (0.24, 0.13, 0.07),
                    max(3, int(h_ / 60)),
                    cv2.LINE_AA,
                )
                cv2.line(a_f, tuple(pts[0]), tuple(pts[1]), 1.0, max(3, int(h_ / 60)), cv2.LINE_AA)
        return rgb_f, a_f

    def _particles(self, t: float) -> Planes:
        a = np.zeros((self.h, self.w), np.float32)
        p, u, w, h = self.p, self.u, self.w, self.h
        size = float(p["size"])
        sp = float(p["speed"])
        x = (u[:, 0] * w + sp * t * (u[:, 2] - 0.5) * 2) % w
        y = (u[:, 1] * h - sp * t * (0.3 + u[:, 3])) % h
        twinkle = 0.5 + 0.5 * np.sin(t * (1 + 2 * u[:, 3]) + u[:, 2] * 6.28)
        for xi, yi, ri, bi in zip(x, y, size * (0.4 + u[:, 2]), twinkle, strict=True):
            cv2.circle(a, (int(xi), int(yi)), int(ri), float(bi), -1, cv2.LINE_AA)
        a = np.asarray(cv2.GaussianBlur(a, (0, 0), size * 0.35 + 0.5), np.float32)
        return self._solid(p["color"]), np.asarray(
            np.clip(a, 0, 1) * float(p["opacity"]), np.float32
        )

    # -- illustrated landscape ---------------------------------------------------------
    def _ridge(self, layer: int, x: np.ndarray, shift: float) -> np.ndarray:
        """Periodic 1D fractal ridge in [0, 1] (period 3 frame widths, so drift never seams)."""
        rng = np.random.default_rng(int(self.p.get("seed", 1)) * 97 + layer)
        period = 3.0 * self.w
        y = np.zeros_like(x, dtype=np.float64)
        amp, total = 1.0, 0.0
        for octave in range(5):
            k = int(rng.integers(2, 4)) * (2**octave)  # whole cycles per period: seamless
            y += amp * np.sin(2 * np.pi * k * (x + shift) / period + rng.uniform(0, 2 * np.pi))
            total += amp
            amp *= 0.5
        return np.asarray((y / total + 1.0) / 2.0)

    @staticmethod
    def _unpremul(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
        """rgb was composited over black; divide out coverage for straight alpha."""
        return np.asarray(rgb / np.maximum(a, 1e-6)[..., None], np.float32)

    def _mountains(self, t: float) -> Planes:
        """Layered ridges, far (pale, high) to near (dark, low), drifting with parallax."""
        p = self.p
        drift = float(p["drift"])
        if drift == 0.0 and self._static is not None:
            return self._static
        # Static mountains render once at full resolution. Drifting ones re-render every
        # frame, so they work at half resolution (measured: 400 ms/frame at 1080p otherwise).
        k_res = 1 if drift == 0.0 else 2
        w, h = self.w // k_res, self.h // k_res
        n = max(1, int(p["layers"]))
        horizon = float(p["horizon"]) * h
        top = horizon - float(p["height"]) * h
        far, near = hex_rgb(p["far"]), hex_rgb(p["near"])
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        xs = np.arange(0, w + 8, 8, dtype=np.float64)
        yy = np.arange(h, dtype=np.float32)[:, None]
        cols = np.arange(w, dtype=np.float64)
        white = np.array([0.96, 0.97, 1.0], np.float32)
        for i in range(n):
            k = i / max(n - 1, 1)  # 0 = farthest
            color = far * (1 - k) + near * k
            peak = top + k * 0.5 * (horizon - top)  # nearer ridges are lower
            ridge = self._ridge(i, xs * k_res, drift * t * (0.4 + k)) ** 1.6  # sharp peaks
            ys = peak + (1 - ridge) * (horizon - peak) * 0.92
            poly = np.concatenate([np.stack([xs, ys], 1), [[w + 8, h], [0, h]]]).astype(np.int32)
            layer = np.zeros((h, w), np.uint8)
            cv2.fillPoly(layer, [poly], 255, cv2.LINE_AA)
            m = layer.astype(np.float32) / 255.0
            cap = float(p["snowcap"]) * (1 - 0.8 * k)
            if cap > 0:  # a snow cap under every high peak, deeper on taller peaks
                yr = np.interp(cols, xs, ys).astype(np.float32)
                rr = np.interp(cols, xs, ridge).astype(np.float32)
                depth = cap * (horizon - peak) * np.clip((rr - 0.45) / 0.4, 0, 1)
                snow = np.clip((yr + depth - yy) / (6.0 / k_res), 0, 1) * (depth > 0.5) * m
                col = color + (white - color) * snow[..., None]
            else:
                col = np.broadcast_to(color, (h, w, 3))
            rgb = rgb * (1 - m[..., None]) + col * m[..., None]
            a = np.maximum(a, m)
        rgb = self._unpremul(rgb, a)
        if k_res != 1:
            rgb, a = self._up(rgb), self._up(a)
        out: Planes = (rgb, np.asarray(a, np.float32))
        if drift == 0.0:
            self._static = out
        return out

    def _river(self, t: float) -> Planes:
        """Meadow valley floor with a river winding toward the viewer; ripples flow downstream."""
        p, u = self.p, self.u
        w, h = self.w, self.h
        y0 = float(p["horizon"]) * h
        r0 = int(y0)
        span = h - y0
        if self._static is None:  # banks and water body never change: build once
            rgb = np.zeros((h, w, 3), np.float32)
            a = np.zeros((h, w), np.float32)
            ys = np.arange(r0, h, dtype=np.float32)
            s = np.clip((ys - y0) / span, 0.0, 1.0)  # 0 at the horizon, 1 at the bottom
            bank = (
                hex_rgb(p["bank_far"])[None, :] * (1 - s[:, None]) + hex_rgb(p["bank"]) * s[:, None]
            )
            rgb[r0:] = bank[:, None, :]
            a[r0:] = 1.0
            depth = np.sqrt(s)  # perspective: the near river widens quickly
            wf, wn = float(p["width_far"]) * w, float(p["width_near"]) * w
            half = (wf + (wn - wf) * depth**1.8) / 2  # stays narrow, then opens near the viewer
            # S-shaped course: one full swing between the horizon and the bottom edge
            cx = w * (0.5 + float(p["meander"]) * np.sin(2 * np.pi * depth + 0.4))
            water = (hex_rgb(p["water_far"])[None, :] * (1 - depth[:, None])
                     + hex_rgb(p["water"]) * depth[:, None])  # fmt: skip
            xx = np.arange(w, dtype=np.float32)[None, :]
            inside = np.clip(half[:, None] - np.abs(xx - cx[:, None]) + 0.5, 0, 1)[..., None]
            rgb[r0:] = rgb[r0:] * (1 - inside) + water[:, None, :] * inside
            self._static = (np.asarray(rgb, np.float32), np.asarray(a, np.float32))
            self._river_geom = (np.asarray(cx), np.asarray(half))
        base_rgb, base_a = self._static
        cx, half = self._river_geom
        n = len(u)
        # ripples move along the river over time: downstream = toward the viewer
        s = (u[:, 1] + float(p["speed"]) * t * (0.7 + 0.6 * u[:, 2])) % 1.0
        depth = s * s
        row = np.clip((y0 + depth * span).astype(int) - r0, 0, len(cx) - 1)
        x = cx[row] + (u[:, 0] - 0.5) * 1.6 * half[row]
        y = y0 + depth * span
        length = 4 + 40 * depth * (0.5 + u[:, 3])
        fade = np.clip(depth * 3, 0, 1) * (0.25 + 0.5 * u[:, 3])
        ripples = np.zeros((h, w), np.float32)
        segs = np.stack([x - length / 2, y, x + length / 2, y], 1).astype(np.int32).reshape(n, 2, 2)
        bucket = np.minimum((fade * 4).astype(int), 2)
        for b in range(3):  # three brightness buckets: three batched polylines calls
            sel = bucket == b
            if sel.any():
                cv2.polylines(ripples, list(segs[sel]), False, 0.35 + 0.25 * b, 2, cv2.LINE_AA)
        rgb = base_rgb.copy()
        kk = np.ascontiguousarray(ripples[r0:])
        white = np.ones_like(rgb[r0:])
        rgb[r0:] = cv2.blendLinear(white, np.ascontiguousarray(base_rgb[r0:]), kk, 1.0 - kk)
        return rgb, base_a

    def _birds(self, t: float) -> Planes:
        """A loose flock crossing the sky with flapping M-shaped wings."""
        p, u = self.p, self.u
        w, h = self.w, self.h
        size = float(p["size"])
        span = w + 4 * size
        x = (u[:, 0] * span + float(p["speed"]) * t * (0.8 + 0.4 * u[:, 2])) % span - 2 * size
        y = h * (0.1 + 0.25 * u[:, 1]) + 8 * np.sin(t * 0.8 + u[:, 3] * 6.28)
        s = size * (0.6 + 0.6 * u[:, 2])
        dy = s * 0.55 * np.sin(t * float(p["flap"]) * 2 * np.pi + u[:, 3] * 6.28)
        pts = np.stack([
            np.stack([x - s, y - dy], 1), np.stack([x - 0.35 * s, y - 0.2 * dy], 1),
            np.stack([x, y + 0.15 * s], 1), np.stack([x + 0.35 * s, y - 0.2 * dy], 1),
            np.stack([x + s, y - dy], 1),
        ], 1).astype(np.int32)  # fmt: skip
        a = np.zeros((h, w), np.float32)
        cv2.polylines(a, list(pts), False, 1.0, max(2, int(size / 7)), cv2.LINE_AA)
        return self._solid(p["color"]), a
