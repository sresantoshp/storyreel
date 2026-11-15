"""Procedural environments: gradient, starfield, rain, snow, fog, fire, particles.

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
        seed = int(self.p.get("seed", 1))
        rng = np.random.default_rng(seed)
        n = int(self.p.get("count", 0))
        self.u = rng.random((n, 4), dtype=np.float32)  # per-particle uniforms
        if gtype in ("fog", "fire"):
            # Tileable noise octaves; animated by scrolling, upscaled per frame.
            self.noise = [_value_noise(rng, 64, 64), _value_noise(rng, 128, 128)]

    def frame(self, t: float) -> Planes:
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
        base = np.clip((yy - (1 - height)) / max(height, 1e-3), 0, 1)
        heat = np.clip(base * (0.35 + 1.1 * n) - 0.15 * (1 - base), 0, 1)
        heat = heat * float(p["intensity"])
        # palette at low resolution: dark red -> orange -> yellow -> white; upscale once
        rgb = np.dstack([np.clip(heat * 2.2, 0, 1), np.clip(heat * 1.8 - 0.45, 0, 1),
                         np.clip(heat * 2.5 - 1.6, 0, 1)]).astype(np.float32)  # fmt: skip
        return self._up(rgb), self._up(np.clip(heat * 1.6, 0, 1))

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
