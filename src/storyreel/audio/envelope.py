"""Sample-exact gain automation, computed with numpy for one chunk window at a time.

Every envelope is a pure function of *timeline* (or clip-local) sample positions, never of
where a chunk starts, so a chunked render and a single-chunk render get bit-identical
gains. Arrays are at most one window long (chunk + pre-roll, ~15 s → ~3 MB float32).

Clip envelope  = clip fades (half-cosine) x volume keyframes (dB, any interpolation).
Duck envelope  = gain of a ducked bus: smoothstep ramp down `attack` s *before* each
                 dialogue clip (look-ahead: the timeline is known), hold, ramp up over
                 `release` s after it, depth `duck_db`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from storyreel.render.keyframes import Key, Track
from storyreel.timebase import FLICKS

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
KEY_GRID = 48  # volume keyframes are evaluated every 48 clip-local samples, then interpolated


def _ramp(x: F64) -> F64:
    """Half-cosine 0 → 1 for x in [0, 1]."""
    xc = np.clip(x, 0.0, 1.0)
    out: F64 = 0.5 - 0.5 * np.cos(np.pi * xc)
    return out


def clip_envelope(
    offset: int,
    n: int,
    clip_len: int,
    sample_rate: int,
    fade_in: int = 0,
    fade_out: int = 0,
    keys: Sequence[Sequence[float | str]] = (),
) -> F32:
    """Linear gain for clip-local samples [offset, offset + n).

    keys: [t_flicks, value_db, interp, h1x, h1y, h2x, h2y] as stored in the keyframe table.
    """
    c = np.arange(offset, offset + n, dtype=np.float64)
    g = np.ones(n, dtype=np.float64)
    if fade_in > 0:
        g *= _ramp(c / fade_in)
    if fade_out > 0:
        g *= _ramp((clip_len - c) / fade_out)
    if keys:
        track = Track(
            [
                Key(
                    int(k[0]),
                    float(k[1]),
                    str(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                    float(k[6]),
                )
                for k in keys
            ]
        )
        # grid points at clip-local multiples of KEY_GRID covering [offset, offset + n]
        g0 = (offset // KEY_GRID) * KEY_GRID
        g1 = -(-(offset + n) // KEY_GRID) * KEY_GRID
        grid = np.arange(g0, g1 + KEY_GRID, KEY_GRID, dtype=np.int64)
        db = np.array([track.at(int(s) * FLICKS // sample_rate) for s in grid], dtype=np.float64)
        g *= np.power(10.0, np.interp(c, grid.astype(np.float64), db) / 20.0)
    return g.astype(np.float32)


def duck_envelope(
    w0: int,
    n: int,
    speech: Sequence[tuple[int, int]],
    sample_rate: int,
    duck_db: float,
    attack_s: float,
    release_s: float,
) -> F32:
    """Linear gain for timeline samples [w0, w0 + n) given dialogue intervals (timeline
    samples, [a, b))."""
    t = np.arange(w0, w0 + n, dtype=np.float64)
    depth = np.zeros(n, dtype=np.float64)
    att = max(1.0, attack_s * sample_rate)
    rel = max(1.0, release_s * sample_rate)
    for a, b in speech:
        if b + rel < w0 or a - att > w0 + n:
            continue
        d = np.where(
            t < a,
            (t - (a - att)) / att,
            np.where(t < b, 1.0, 1.0 - (t - b) / rel),
        )
        d = np.clip(d, 0.0, 1.0)
        depth = np.maximum(depth, d * d * (3.0 - 2.0 * d))  # smoothstep
    gain: F64 = np.power(10.0, duck_db * depth / 20.0)
    return gain.astype(np.float32)


def write_f32(path: Path, data: F32) -> None:
    """Raw little-endian float32 (ffmpeg: -f f32le -ar <sr> -ch_layout <layout>)."""
    np.ascontiguousarray(data, dtype="<f4").tofile(path)
