"""Keyframe evaluation: hold, linear, and cubic-bezier easing (CSS cubic-bezier semantics).

A segment's interpolation is set by its *first* keyframe. Before the first key the value
holds at the first key; after the last key it holds at the last. A parameter with no keys
uses the clip's static value.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class Key:
    t: int  # flicks from clip start
    value: float
    interp: str = "linear"
    h1x: float = 0.42
    h1y: float = 0.0
    h2x: float = 0.58
    h2y: float = 1.0


def _bez(a: float, b: float, s: float) -> float:
    """1D cubic bezier with endpoints 0 and 1 and control values a, b."""
    u = 1.0 - s
    return 3 * u * u * s * a + 3 * u * s * s * b + s * s * s


def _bez_d(a: float, b: float, s: float) -> float:
    u = 1.0 - s
    return 3 * u * u * a + 6 * u * s * (b - a) + 3 * s * s * (1 - b)


def ease(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """y for progress x in [0, 1] on cubic-bezier(x1, y1, x2, y2)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    s = x
    for _ in range(8):  # Newton; converges in 3-5 steps for sane handles
        err = _bez(x1, x2, s) - x
        if abs(err) < 1e-7:
            return _bez(y1, y2, s)
        d = _bez_d(x1, x2, s)
        if abs(d) < 1e-6:
            break
        s -= err / d
    lo, hi = 0.0, 1.0  # bisection fallback for flat derivatives
    s = x
    for _ in range(40):
        if _bez(x1, x2, s) < x:
            lo = s
        else:
            hi = s
        s = (lo + hi) / 2
    return _bez(y1, y2, s)


class Track:
    """Sorted keys for one parameter."""

    def __init__(self, keys: list[Key]) -> None:
        self.keys = sorted(keys, key=lambda k: k.t)
        self._ts = [k.t for k in self.keys]

    def at(self, t: int) -> float:
        keys = self.keys
        if t <= keys[0].t:
            return keys[0].value
        if t >= keys[-1].t:
            return keys[-1].value
        i = bisect.bisect_right(self._ts, t) - 1
        k0, k1 = keys[i], keys[i + 1]
        if k0.interp == "hold":
            return k0.value
        x = (t - k0.t) / (k1.t - k0.t)
        if k0.interp == "bezier":
            x = ease(x, k0.h1x, k0.h1y, k0.h2x, k0.h2y)
        return k0.value + (k1.value - k0.value) * x

    def max_value(self) -> float:
        return max(k.value for k in self.keys)


class Params:
    """Static values + animated tracks for one clip. `get(name, t)` is the only API."""

    def __init__(self, static: dict[str, float], tracks: dict[str, list[Key]]) -> None:
        self.static = static
        self.tracks = {n: Track(ks) for n, ks in tracks.items() if ks}

    def get(self, name: str, t: int, default: float = 0.0) -> float:
        tr = self.tracks.get(name)
        if tr is not None:
            return tr.at(t)
        return float(self.static.get(name, default))

    def animated(self, name: str) -> bool:
        return name in self.tracks

    def max(self, name: str, default: float) -> float:
        tr = self.tracks.get(name)
        static = float(self.static.get(name, default))
        return tr.max_value() if tr is not None else static
