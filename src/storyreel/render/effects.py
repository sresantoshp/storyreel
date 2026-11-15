"""Per-layer effects on planar images: rgb (H, W, 3) float32 0..1 + alpha (H, W) or None.

`alpha is None` means fully opaque. Keeping colour and alpha as separate contiguous planes
avoids the channel-slicing copies that dominated the first profile (5.8 s of 14.8 s).

Each effect is `fn(rgb, alpha, params, ctx) -> (rgb, alpha)` and must not modify its inputs
(sources may hand out cached frames). Numeric params may be keyframed; the compositor
resolves them per frame. Randomised effects (grain, glitch) seed from the clip-local frame
index, so a frame renders identically whichever chunk computes it.

`lut` is not here: a static 3D LUT runs in the ffmpeg decoder (lut3d), see layers.MediaSource.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from storyreel.text.render import hex_rgb

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

Alpha = np.ndarray | None


@dataclass(frozen=True)
class Ctx:
    frame: int  # clip-local frame index
    fps: float


Effect = Callable[[np.ndarray, Alpha, dict[str, Any], Ctx], tuple[np.ndarray, Alpha]]


def f32(x: Any) -> np.ndarray:
    return np.asarray(x, np.float32)


def to_u8(rgb: np.ndarray) -> np.ndarray:
    """Float 0..1 → uint8 with rounding and saturation (OpenCV, multithreaded)."""
    return np.asarray(cv2.convertScaleAbs(rgb, alpha=255.0), np.uint8)


def u8_to_f(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.multiply(x, np.float32(1.0 / 255.0), dtype=np.float32), np.float32)


@functools.lru_cache(maxsize=4)
def ones(h: int, w: int) -> np.ndarray:
    o = np.ones((h, w), np.float32)
    o.setflags(write=False)
    return o


def _alpha(a: Alpha, h: int, w: int) -> np.ndarray:
    return ones(h, w) if a is None else a


@functools.lru_cache(maxsize=256)
def _color_lut(key: tuple[float, ...]) -> np.ndarray:
    """All per-channel colour operations folded into one 256-entry float LUT per channel."""
    exposure, bright, temp, tint, contrast, *rest = key
    lift, gain = np.array(rest[0:3]), np.array(rest[6:9])
    gamma = np.maximum(np.array(rest[3:6]), 0.05)
    x = np.linspace(0.0, 1.0, 256)[:, None]
    k = 2.0**exposure * bright
    x = x * np.array([k * (1 + 0.2 * temp), k * (1 - 0.2 * tint), k * (1 - 0.2 * temp)])
    x = gain * (x + lift * (1.0 - x))
    x = np.power(np.clip(x, 0.0, None), 1.0 / gamma)
    x = (x - 0.5) * contrast + 0.5
    return np.asarray(np.clip(x, 0.0, 1.0), np.float32).reshape(256, 1, 3)


def color(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    """Exposure/temperature/tint/lift/gamma/gain/contrast via one LUT; saturation via 3x3.

    The LUT indexes 8-bit-quantised input (sources and output are 8-bit). Measured at
    720p: 4 ms, against 125 ms for the same maths in float numpy.
    """
    names = ("exposure", "brightness", "temperature", "tint", "contrast", "lift_r", "lift_g",
             "lift_b", "gamma_r", "gamma_g", "gamma_b", "gain_r", "gain_g", "gain_b")  # fmt: skip
    key = tuple(round(float(p[n]), 5) for n in names)
    out = f32(cv2.LUT(to_u8(rgb), _color_lut(key)))
    s = float(p["saturation"])
    if s != 1.0:
        m = (s * np.eye(3) + (1.0 - s) * np.ones((3, 1)) * LUMA[None, :]).astype(np.float32)
        out = f32(cv2.transform(out, m))
        np.clip(out, 0.0, 1.0, out=out)
    return out, a


@functools.lru_cache(maxsize=64)
def _curves_lut(key: str) -> np.ndarray:
    pts = json.loads(key)

    def curve(points: list[list[float]]) -> np.ndarray:
        xs = np.array([q[0] for q in points], np.float64)
        ys = np.array([q[1] for q in points], np.float64)
        order = np.argsort(xs)
        return np.interp(np.linspace(0, 1, 256), xs[order], ys[order])

    master = curve(pts["master"])
    chans = [np.interp(master, np.linspace(0, 1, 256), curve(pts[c])) for c in "rgb"]
    return np.asarray(np.clip(np.stack(chans, -1), 0, 1), np.float32).reshape(256, 1, 3)


def curves(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    key = json.dumps({k: p[k] for k in ("master", "r", "g", "b")}, sort_keys=True)
    return f32(cv2.LUT(to_u8(rgb), _curves_lut(key))), a


def lab_stats(rgb: np.ndarray) -> tuple[list[float], list[float]]:
    """Mean/std per Lab channel; input RGB float 0..1. Used to set up colormatch."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2Lab)
    flat = lab.reshape(-1, 3)
    return flat.mean(0).tolist(), np.maximum(flat.std(0), 1e-3).tolist()


def colormatch(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    """Reinhard colour transfer in Lab: move src statistics onto dst statistics."""
    lab = f32(cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab))
    sm, ss = np.array(p["src_mean"], np.float32), np.array(p["src_std"], np.float32)
    dm, ds = np.array(p["dst_mean"], np.float32), np.array(p["dst_std"], np.float32)
    lab = (lab - sm) / ss * ds + dm
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    moved = np.clip(f32(cv2.cvtColor(f32(lab), cv2.COLOR_Lab2RGB)), 0, 1)
    m = float(p["mix"])
    return f32(cv2.addWeighted(rgb, 1 - m, moved, m, 0.0)), a


def blur(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    r = float(p["radius"])
    if r <= 0.05:
        return rgb, a
    sigma = r / 2
    if a is None:
        return f32(cv2.GaussianBlur(rgb, (0, 0), sigma)), None
    # premultiplied blur so transparent edges do not darken
    pm = f32(cv2.GaussianBlur(rgb * a[..., None], (0, 0), sigma))
    ab = f32(cv2.GaussianBlur(a, (0, 0), sigma))
    return f32(pm / np.maximum(ab, 1e-6)[..., None]), ab


def sharpen(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    soft = cv2.GaussianBlur(rgb, (0, 0), max(float(p["radius"]), 0.3))
    k = float(p["amount"])
    out = f32(cv2.addWeighted(rgb, 1 + k, soft, -k, 0.0))
    np.clip(out, 0, 1, out=out)
    return out, a


def glow(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    y = f32(cv2.transform(rgb, LUMA[None, :]))
    y = y.reshape(rgb.shape[:2])
    th = float(p["threshold"])
    bright = rgb * np.clip((y - th) / max(1 - th, 1e-3), 0, 1)[..., None]
    halo = f32(cv2.GaussianBlur(f32(bright), (0, 0), max(float(p["radius"]), 0.5)))
    halo *= np.float32(p["strength"])
    np.clip(halo, 0, 1, out=halo)
    return f32(1 - (1 - rgb) * (1 - halo)), a  # screen blend


@functools.lru_cache(maxsize=8)
def _radial(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    return f32(d)


@functools.lru_cache(maxsize=16)
def _vignette_map(w: int, h: int, amount: float, softness: float) -> np.ndarray:
    soft = max(softness, 0.05)
    k = np.clip((_radial(w, h) - (1 - soft)) / soft, 0, 1)
    m = (1 - amount * k * k).astype(np.float32)
    return f32(cv2.merge([m, m, m]))


def vignette(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    h, w = rgb.shape[:2]
    vmap = _vignette_map(w, h, round(float(p["amount"]), 4), round(float(p["softness"]), 4))
    return f32(cv2.multiply(rgb, vmap)), a


def grain(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    h, w = rgb.shape[:2]
    size = max(float(p["size"]), 1.0)
    rng = np.random.default_rng(ctx.frame * 7919 + 17)
    gh, gw = max(1, int(h / size)), max(1, int(w / size))
    noise = rng.standard_normal((gh, gw), dtype=np.float32) * np.float32(p["amount"])
    if (gh, gw) != (h, w):
        noise = f32(cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR))
    out = f32(cv2.add(rgb, cv2.merge([noise, noise, noise])))
    np.clip(out, 0, 1, out=out)
    return out, a


def _shift(ch: np.ndarray, dx: float) -> np.ndarray:
    m = np.array([[1, 0, dx], [0, 1, 0]], np.float32)
    h, w = ch.shape[:2]
    return f32(cv2.warpAffine(ch, m, (w, h), borderMode=cv2.BORDER_REPLICATE))


def chromatic(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    s = float(p["shift"])
    r, g, b = cv2.split(rgb)
    return f32(cv2.merge([_shift(r, -s), g, _shift(b, s)])), a


def glitch(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    amount = float(p["amount"])
    rng = np.random.default_rng(int(p["seed"]) * 100_003 + ctx.frame)
    if rng.random() > amount:
        return rgb, a
    h, w = rgb.shape[:2]
    out = rgb.copy()
    for _ in range(int(2 + 8 * amount)):
        y0 = int(rng.integers(0, h))
        bh = int(rng.integers(2, max(3, h // 12)))
        dx = int(rng.normal(0, w * 0.04 * amount))
        out[y0 : y0 + bh] = np.roll(out[y0 : y0 + bh], dx, axis=1)
    r, g, b = cv2.split(out)
    return f32(cv2.merge([_shift(r, float(rng.normal(0, 6 * amount))), g, b])), a


@functools.lru_cache(maxsize=8)
def _lens_maps(w: int, h: int, k1: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    f = 1 + k1 * (nx * nx + ny * ny)
    return f32(nx * f * (w / 2) + w / 2), f32(ny * f * (h / 2) + h / 2)


def lens(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    h, w = rgb.shape[:2]
    mx, my = _lens_maps(w, h, round(float(p["k1"]), 4))
    out = f32(cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
    alpha = f32(
        cv2.remap(_alpha(a, h, w), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    )
    return out, alpha


def mask(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    h, w = rgb.shape[:2]
    m = np.zeros((h, w), np.float32)
    pts = np.array([[x * w, y * h] for x, y in p["points"]], np.int32)
    cv2.fillPoly(m, [pts], 1.0, cv2.LINE_AA)
    f = float(p["feather"])
    if f > 0.5:
        m = f32(cv2.GaussianBlur(m, (0, 0), f / 2))
    if p["invert"]:
        m = 1 - m
    return rgb, f32(m if a is None else a * m)


def chromakey(rgb: np.ndarray, a: Alpha, p: dict[str, Any], ctx: Ctx) -> tuple[np.ndarray, Alpha]:
    """Distance in CbCr from the key colour → alpha; then green/blue spill suppression."""
    key = hex_rgb(p["key"])
    to_cbcr = np.array([[-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]], np.float32)
    cbcr = f32(cv2.transform(rgb, to_cbcr))
    d = np.linalg.norm(cbcr - key @ to_cbcr.T, axis=-1)
    sim, smooth = float(p["similarity"]), max(float(p["smoothness"]), 1e-3)
    k = f32(np.clip((d - sim) / smooth, 0, 1))
    ch = int(np.argmax(key))  # dominant key channel (1 = green, 2 = blue)
    others = [i for i in range(3) if i != ch]
    limit = np.maximum(rgb[..., others[0]], rgb[..., others[1]])
    out = rgb.copy()
    out[..., ch] = rgb[..., ch] - np.clip(rgb[..., ch] - limit, 0, None) * float(p["spill"])
    return out, f32(k if a is None else a * k)


REGISTRY: dict[str, Effect] = {
    "color": color, "curves": curves, "colormatch": colormatch, "blur": blur,
    "sharpen": sharpen, "glow": glow, "vignette": vignette, "grain": grain,
    "chromatic": chromatic, "glitch": glitch, "lens": lens, "mask": mask,
    "chromakey": chromakey,
}  # fmt: skip
