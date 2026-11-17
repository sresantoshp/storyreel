"""Small anti-aliased vector drawing helpers shared by scenery and actors.

Shapes are drawn with OpenCV's anti-aliased fill on a uint8 RGBA canvas that starts fully
transparent. With LINE_AA OpenCV blends every channel toward the shape colour by coverage, so
drawing opaque shapes onto (0,0,0,0) yields *premultiplied* RGBA; `planes()` converts that to
the engine's straight-alpha (rgb, alpha) float planes. Coordinates are float and use a 4-bit
sub-pixel shift, so slow movement does not jitter by whole pixels.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from storyreel.text.render import hex_rgb

Planes = tuple[np.ndarray, np.ndarray | None]
Color = tuple[int, int, int, int]
SHIFT = 4
_K = float(1 << SHIFT)


def rgb(color: str) -> np.ndarray:
    return np.asarray(hex_rgb(color), np.float32)


def col(c: np.ndarray | str, k: float = 1.0) -> Color:
    """Opaque canvas colour from '#rrggbb' or an rgb array, optionally scaled by k."""
    v = rgb(c) if isinstance(c, str) else np.asarray(c, np.float32)
    v = np.clip(v * k, 0, 1) * 255
    return (int(v[0]), int(v[1]), int(v[2]), 255)


def mix(a: np.ndarray | str, b: np.ndarray | str, k: float) -> np.ndarray:
    va = rgb(a) if isinstance(a, str) else a
    vb = rgb(b) if isinstance(b, str) else b
    return np.asarray(va * (1 - k) + vb * k, np.float32)


def canvas(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), np.uint8)


def _fx(pts: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    return np.round(np.asarray(pts, np.float64) * _K).astype(np.int32)


def poly(cv: np.ndarray, pts: np.ndarray | Sequence[Sequence[float]], c: Color) -> None:
    cv2.fillPoly(cv, [_fx(pts)], c, cv2.LINE_AA, SHIFT)


def polys(cv: np.ndarray, many: Sequence[np.ndarray], c: Color) -> None:
    if many:
        cv2.fillPoly(cv, [_fx(p) for p in many], c, cv2.LINE_AA, SHIFT)


def ellipse(
    cv: np.ndarray, cx: float, cy: float, rx: float, ry: float, c: Color, angle: float = 0.0
) -> None:
    cv2.ellipse(
        cv,
        (round(cx * _K), round(cy * _K)),
        (max(1, round(rx * _K)), max(1, round(ry * _K))),
        angle,
        0,
        360,
        c,
        -1,
        cv2.LINE_AA,
        SHIFT,
    )


def circle(cv: np.ndarray, cx: float, cy: float, r: float, c: Color) -> None:
    ellipse(cv, cx, cy, r, r, c)


def line(
    cv: np.ndarray, pts: np.ndarray | Sequence[Sequence[float]], c: Color, width: float
) -> None:
    cv2.polylines(cv, [_fx(pts)], False, c, max(1, round(width)), cv2.LINE_AA, SHIFT)


def limb(
    cv: np.ndarray, a: Sequence[float], b: Sequence[float], ra: float, rb: float, c: Color
) -> None:
    """Tapered capsule from point a (radius ra) to point b (radius rb)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    nx, ny = -dy / n, dx / n
    poly(
        cv,
        [
            (ax + nx * ra, ay + ny * ra),
            (bx + nx * rb, by + ny * rb),
            (bx - nx * rb, by - ny * rb),
            (ax - nx * ra, ay - ny * ra),
        ],
        c,
    )
    circle(cv, ax, ay, ra, c)
    circle(cv, bx, by, rb, c)


def bezier(
    p0: Sequence[float], p1: Sequence[float], p2: Sequence[float], n: int = 16
) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    a, b, c = (np.asarray(p, np.float64) for p in (p0, p1, p2))
    return np.asarray((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * c)


def leaf(
    base: Sequence[float], tip: Sequence[float], width: float, bend: float = 0.0, n: int = 10
) -> np.ndarray:
    """Pointed leaf / frond outline from base to tip, `bend` curves the midrib sideways."""
    b = np.asarray(base, np.float64)
    t_ = np.asarray(tip, np.float64)
    d = t_ - b
    length = max(1e-6, float(np.hypot(*d)))
    nrm = np.array([-d[1], d[0]]) / length
    s = np.linspace(0, 1, n)[:, None]
    mid = b + d * s + nrm * (bend * length * 4 * s * (1 - s))
    half = width / 2 * np.sin(np.pi * np.clip(s, 0, 1) ** 0.8)
    return np.concatenate([mid + nrm * half, (mid - nrm * half)[::-1]])


def planes(cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Premultiplied uint8 RGBA canvas -> straight-alpha float planes."""
    f = cv.astype(np.float32) * (1.0 / 255.0)
    a = np.ascontiguousarray(f[..., 3])
    out = f[..., :3] / np.maximum(a, 1e-6)[..., None]
    np.clip(out, 0.0, 1.0, out=out)
    return np.asarray(out, np.float32), a


def over(dst_rgb: np.ndarray, src_rgb: np.ndarray, src_a: np.ndarray) -> None:
    """In-place straight-alpha 'over' of src onto an opaque dst region (same shape)."""
    k = src_a[..., None]
    dst_rgb *= 1.0 - k
    dst_rgb += src_rgb * k


def paste(
    dst_rgb: np.ndarray,
    dst_a: np.ndarray | None,
    src_rgb: np.ndarray,
    src_a: np.ndarray,
    x: int,
    y: int,
) -> None:
    """Composite a sprite onto full-frame planes at (x, y), clipped to the frame."""
    h, w = dst_rgb.shape[:2]
    sh, sw = src_a.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + sw), min(h, y + sh)
    if x0 >= x1 or y0 >= y1:
        return
    s = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    d = (slice(y0, y1), slice(x0, x1))
    sa = src_a[s]
    if dst_a is None:  # opaque destination
        k = sa[..., None]
        dst_rgb[d] = dst_rgb[d] * (1 - k) + src_rgb[s] * k
        return
    da = dst_a[d]
    out_a = sa + da * (1 - sa)
    wd = (da * (1 - sa) / np.maximum(out_a, 1e-6))[..., None]
    dst_rgb[d] = dst_rgb[d] * wd + src_rgb[s] * (1 - wd)
    dst_a[d] = out_a
