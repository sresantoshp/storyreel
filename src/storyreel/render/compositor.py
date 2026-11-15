"""Path B: numpy/OpenCV compositor for one chunk.

Per output frame, for each active layer (bottom to top):
  source frame → crop → effects → transition modifiers → affine warp (into the layer's
  bounding box only) → premultiplied "over" onto a float32 RGB canvas.
Adjustment layers apply their effects to the canvas itself.

Memory is O(layers), independent of chunk length: each source holds at most two decoded
frames (or one 12-frame reverse block), plus one float canvas (25 MB at 1080p).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from storyreel.render import effects as fx
from storyreel.render.fxspec import EFFECTS, TRANSFORM_DEFAULTS
from storyreel.render.keyframes import Key, Params, ease
from storyreel.render.layers import ColorSource, GeneratorSource, MediaSource, TextSource
from storyreel.timebase import FLICKS, Rate

Source = MediaSource | ColorSource | TextSource | GeneratorSource


class Layer:
    def __init__(self, spec: dict[str, Any], ffmpeg: Path, canvas: tuple[int, int], rate: Rate):
        self.spec = spec
        self.at, self.frames = int(spec["at"]), int(spec["frames"])
        self.offset = int(spec["clip_offset"])
        self.kind = str(spec["kind"])
        self.rate = rate
        props: dict[str, Any] = spec.get("props", {})
        statics = {
            k: float(v)
            for k, v in props.items()
            if k in TRANSFORM_DEFAULTS and isinstance(v, int | float)
        }
        tracks = {name: [Key(*k) for k in keys] for name, keys in spec.get("keyframes", {}).items()}
        self.params = Params(statics, tracks)
        self.effects = [e for e in spec.get("effects", []) if e["type"] != "lut"]
        self.trans: list[dict[str, Any]] = spec.get("transitions", [])
        fps = rate.fps
        if self.kind in ("video", "image"):
            max_scale = self.params.max("scale", 1.0) * max(
                self.params.max("scale_x", 1.0), self.params.max("scale_y", 1.0)
            )
            speed = spec["speed"][0] / spec["speed"][1]
            self.source: Source | None = MediaSource(
                ffmpeg, spec, canvas, fps, self._source_time(spec), max_scale, speed
            )
        elif self.kind == "color":
            self.source = ColorSource(str(props["color"]), canvas)
        elif self.kind == "text":
            self.source = TextSource(props, fps)
        elif self.kind == "generator":
            self.source = GeneratorSource(props, canvas, fps)
        else:  # adjust
            self.source = None

    def _source_time(self, spec: dict[str, Any]) -> Any:
        """Map clip-local frame → source seconds (constant speed, ramp, reverse, freeze)."""
        fps = self.rate.fps
        src_in = spec["src_in"] / FLICKS
        speed = spec["speed"][0] / spec["speed"][1]
        props = spec.get("props", {})
        total = int(spec["clip_frames"])
        if props.get("freeze"):
            return lambda c: src_in
        if self.params.animated("speed"):
            # integrate the speed curve once over the whole clip (+ transition margins)
            lo, hi = min(self.offset, 0) - 1, max(total, self.offset + self.frames) + 1
            sp = np.array(
                [
                    self.params.get("speed", self.rate.frame_to_flicks(c), 1.0)
                    for c in range(lo, hi)
                ],
                np.float64,
            )
            cum = np.concatenate([[0.0], np.cumsum(np.maximum(sp, 0.0) / fps)])
            zero = -lo  # index of c = 0 in cum
            return lambda c: src_in + float(cum[c - lo] - cum[zero])
        if props.get("reverse"):
            return lambda c: src_in + (total - 1 - c) * speed / fps
        return lambda c: src_in + c * speed / fps

    def close(self) -> None:
        if self.source is not None:
            self.source.close()


def _transition_mods(
    trans: list[dict[str, Any]], c: int, w: int
) -> tuple[float, float, float, float, float, tuple[str, float] | None]:
    """→ (opacity mult, dx, scale mult, brightness mult, blur px, wipe (dir, p) or None)."""
    op, dx, sc, br, bl = 1.0, 0.0, 1.0, 1.0, 0.0
    wipe: tuple[str, float] | None = None
    for t in trans:
        n = int(t["frames"])
        k = c - int(t["start"])
        if not 0 <= k < n:
            continue
        p = (k + 0.5) / n
        kind, role = t["kind"], t["role"]
        e = ease(p, 0.42, 0.0, 0.58, 1.0)
        if kind == "dissolve" and role == "in":
            op *= p
        elif kind == "fade_black":
            br *= max(0.0, 1 - 2 * p) if role == "out" else max(0.0, 2 * p - 1)
        elif kind.startswith("wipe_") and role == "in":
            wipe = (kind[5:], p)
        elif kind.startswith("slide_"):
            sign = -1.0 if kind == "slide_left" else 1.0
            dx += sign * w * (e if role == "out" else e - 1.0)
        elif kind == "zoom":
            if role == "in":
                sc *= 0.6 + 0.4 * e
                op *= e
            else:
                sc *= 1.0 + 0.3 * e
        elif kind == "whip":
            dx += (-w * e) if role == "out" else (w * (1.0 - e))
            bl = max(bl, 60.0 * math.sin(math.pi * p))
    return op, dx, sc, br, bl, wipe


def _over(
    canvas: np.ndarray, rgb: np.ndarray, a: np.ndarray | None, x0: int, y0: int, opacity: float
) -> None:
    """Straight-alpha "over" of planes at (x0, y0), clipped to the canvas, in place."""
    ch, cw = canvas.shape[:2]
    h, w = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, cw), min(y0 + h, ch)
    if cx0 >= cx1 or cy0 >= cy1 or opacity <= 0.0:
        return
    sy, sx = slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0)
    dst = canvas[cy0:cy1, cx0:cx1]
    if a is None and opacity >= 1.0:
        dst[:] = rgb[sy, sx]  # opaque fast path: plain copy
        return
    wgt = fx.ones(cy1 - cy0, cx1 - cx0) if a is None else np.ascontiguousarray(a[sy, sx])
    if opacity < 1.0:
        wgt = wgt * np.float32(opacity)
    # OpenCV's per-pixel weighted blend is multithreaded: 16 ms vs 34 ms numpy at 720p
    src = (
        rgb[sy, sx] if (sy, sx) == (slice(0, h), slice(0, w)) else np.ascontiguousarray(rgb[sy, sx])
    )
    full = dst.flags.c_contiguous
    out = cv2.blendLinear(src, dst if full else np.ascontiguousarray(dst), wgt,
                          np.float32(1.0) - wgt)  # fmt: skip
    dst[:] = out


class Compositor:
    def __init__(self, spec: dict[str, Any], ffmpeg: Path) -> None:
        s = spec["settings"]
        self.w, self.h = int(s["width"]), int(s["height"])
        self.rate = Rate(int(s["fps_num"]), int(s["fps_den"]))
        self.layers = [Layer(ls, ffmpeg, (self.w, self.h), self.rate) for ls in spec["layers"]]
        self.canvas = np.zeros((self.h, self.w, 3), np.float32)

    def close(self) -> None:
        for layer in self.layers:
            layer.close()

    def _effect_params(self, layer: Layer, eff: dict[str, Any], t: int) -> dict[str, Any]:
        params = {**EFFECTS[eff["type"]], **eff["params"]}
        for name, default in EFFECTS[eff["type"]].items():
            key = f"fx{eff['id']}.{name}"
            if layer.params.animated(key):
                params[name] = layer.params.get(key, t, float(default))
        return params

    def _apply_effects(
        self, layer: Layer, rgb: np.ndarray, a: np.ndarray | None, c: int, t: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        ctx = fx.Ctx(c, self.rate.fps)
        for eff in layer.effects:
            rgb, a = fx.REGISTRY[eff["type"]](rgb, a, self._effect_params(layer, eff, t), ctx)
        return rgb, a

    def render(self, f: int) -> np.ndarray:
        """Chunk-local frame f → uint8 RGB (H, W, 3)."""
        canvas = self.canvas
        canvas[:] = 0.0
        for layer in self.layers:
            if not layer.at <= f < layer.at + layer.frames:
                continue
            c = layer.offset + (f - layer.at)
            t = self.rate.frame_to_flicks(c) if c >= 0 else -self.rate.frame_to_flicks(-c)
            p = layer.params
            if layer.source is None:  # adjustment layer: effects on everything below
                out, _ = self._apply_effects(layer, canvas.copy(), None, c, t)
                op = p.get("opacity", t, 1.0)
                canvas[:] = cv2.addWeighted(canvas, 1.0 - op, out, op, 0.0)
                continue
            rgb, a = layer.source.frame(c)
            ih, iw = rgb.shape[:2]
            crop = [p.get(k, t, 0.0) for k in ("crop_l", "crop_t", "crop_r", "crop_b")]
            if any(crop):
                keep = np.zeros((ih, iw), np.float32)
                x0, y0 = int(crop[0] * iw), int(crop[1] * ih)
                keep[y0 : ih - int(crop[3] * ih), x0 : iw - int(crop[2] * iw)] = 1.0
                a = keep if a is None else a * keep
            if layer.effects:
                rgb, a = self._apply_effects(layer, rgb, a, c, t)
            op_m, dx, sc_m, br, blur_px, wipe = _transition_mods(layer.trans, c, self.w)
            if br != 1.0:
                rgb = fx.f32(rgb * np.float32(br))
            if wipe is not None:  # the incoming layer is revealed from one edge
                d, pr = wipe
                a = (fx.ones(ih, iw) if a is None else a).copy()
                if d == "left":
                    a[:, : int(iw * (1 - pr))] = 0.0
                elif d == "right":
                    a[:, int(iw * pr) :] = 0.0
                elif d == "up":
                    a[: int(ih * (1 - pr)), :] = 0.0
                else:
                    a[int(ih * pr) :, :] = 0.0
            if blur_px > 0.5:
                k = int(blur_px) * 2 + 1
                rgb = fx.f32(cv2.blur(rgb, (k, 1)))
            # transform: decoded size → fit size → user scale; rotate about centre
            base = layer.source.fit[0] / iw
            sx = base * p.get("scale", t, 1.0) * p.get("scale_x", t, 1.0) * sc_m
            sy = base * p.get("scale", t, 1.0) * p.get("scale_y", t, 1.0) * sc_m
            rot = math.radians(p.get("rotation", t, 0.0))
            cx = self.w / 2 + p.get("x", t, 0.0) + dx
            cy = self.h / 2 + p.get("y", t, 0.0)
            opacity = p.get("opacity", t, 1.0) * op_m
            cos, sin = math.cos(rot), math.sin(rot)
            m = np.array([[sx * cos, -sy * sin, 0.0], [sx * sin, sy * cos, 0.0]], np.float64)
            m[:, 2] = [cx, cy] - m[:, :2] @ [iw / 2, ih / 2]
            corners = m[:, :2] @ np.array([[0, iw, iw, 0], [0, 0, ih, ih]], np.float64) + m[:, 2:3]
            bx0 = max(int(math.floor(corners[0].min())), 0)
            by0 = max(int(math.floor(corners[1].min())), 0)
            bx1 = min(int(math.ceil(corners[0].max())), self.w)
            by1 = min(int(math.ceil(corners[1].max())), self.h)
            if bx0 >= bx1 or by0 >= by1:
                continue
            identity = abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6 and rot == 0.0
            if identity and float(m[0, 2]).is_integer() and float(m[1, 2]).is_integer():
                _over(canvas, rgb, a, int(m[0, 2]), int(m[1, 2]), opacity)
                continue
            m[:, 2] -= [bx0, by0]
            size = (bx1 - bx0, by1 - by0)
            interp = cv2.INTER_AREA if max(sx, sy) < 0.7 else cv2.INTER_LINEAR
            w_rgb = fx.f32(cv2.warpAffine(rgb, m, size, flags=interp,
                                          borderMode=cv2.BORDER_REPLICATE))  # fmt: skip
            # alpha carries the layer edge: transparent outside the warped source
            w_a = fx.f32(cv2.warpAffine(fx.ones(ih, iw) if a is None else a, m, size, flags=interp,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0))  # fmt: skip
            _over(canvas, w_rgb, w_a, bx0, by0, opacity)
        return np.asarray(cv2.convertScaleAbs(canvas, alpha=255.0), np.uint8)
