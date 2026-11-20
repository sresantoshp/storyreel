"""Film-trailer sound design, synthesised in code (numpy only; CC0, no samples):

  whoosh   air rushing past: band of noise sweeping up then down, panned left → right.
           Its loudest point is at 65 % of the file, so place it to peak on a cut.
  impact   cinema "hit": a falling sub-bass thump, a short bright crack and a dark tail.
  riser    tension build: a noise band and a gliding tone climbing for the whole length,
           with a tremolo that speeds up. Ends abruptly: put an impact right after it.
  braam    big brass-like low chord (A1 + E2 + A2, detuned saw stacks) whose filter opens
           and closes, for the title card.

All are stereo, 48 kHz, peak-normalised to -1 dBFS; levels are set by the mix.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from storyreel.audio.sfx import env_ar, lowpass

F64 = npt.NDArray[np.float64]
KINDS = ("whoosh", "impact", "riser", "braam")
DEFAULT_SECONDS = {"whoosh": 1.4, "impact": 3.5, "riser": 4.0, "braam": 4.0}
WHOOSH_PEAK = 0.65  # fraction of the whoosh length where it is loudest


def _tv_filter(x: F64, sr: int, gain: Callable[[F64, float], F64], ctl: F64) -> F64:
    """Time-varying FFT filter: gain(freqs, ctl[at frame]) per 2048-sample frame,
    overlap-added with sqrt-Hann windows (50 % overlap sums to one)."""
    n = x.size
    frame, hop = 2048, 1024
    win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame))
    padded = np.concatenate([np.zeros(frame), x, np.zeros(frame)])
    out = np.zeros(n + 2 * frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    for s in range(0, n + frame, hop):
        c = float(ctl[min(max(s - frame // 2, 0), n - 1)])
        seg = np.fft.irfft(np.fft.rfft(padded[s : s + frame] * win) * gain(freqs, c), frame)
        out[s : s + frame] += seg * win
    res: F64 = out[frame : frame + n]
    return res


def _sweep_noise(rng: np.random.Generator, n: int, sr: int, fc: F64, q: float) -> F64:
    """White noise through a band-pass whose centre follows fc (per sample)."""

    def band(f: F64, c: float) -> F64:
        bw = max(c / q, 40.0)
        g: F64 = 1.0 / (1.0 + ((f - c) / (bw / 2)) ** 2)
        return g

    return _tv_filter(rng.standard_normal(n), sr, band, fc)


def _pan(x: F64, pan: F64 | float) -> F64:
    """Constant-power pan, -1 = left, +1 = right. Returns (2, n)."""
    p = (np.asarray(pan, np.float64) + 1.0) * math.pi / 4
    return np.stack([x * np.cos(p), x * np.sin(p)])


def _norm(x: F64, peak_db: float = -1.0) -> F64:
    m = float(np.max(np.abs(x))) or 1.0
    out: F64 = x * (10 ** (peak_db / 20) / m)
    return out


def whoosh(sr: int, seconds: float, rng: np.random.Generator) -> F64:
    n = int(seconds * sr)
    t = np.arange(n) / n
    pk = WHOOSH_PEAK
    bell = np.where(t < pk, (t / pk) ** 2.2, np.exp(-((t - pk) / (1 - pk)) * 4.0))
    fc = 250.0 + 3200.0 * bell
    air = _sweep_noise(rng, n, sr, fc, q=1.6)
    body = _sweep_noise(rng, n, sr, fc * 0.25 + 60, q=1.2)
    x = (air + 0.8 * body) * bell * env_ar(n, sr, 0.01, 0.05)
    return _norm(_pan(x, -0.75 + 1.5 * t))


def impact(sr: int, seconds: float, rng: np.random.Generator) -> F64:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f = 34.0 + 56.0 * np.exp(-t / 0.07)
    sub = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.9)
    crack = lowpass(rng.standard_normal(n), sr, 3500.0) * np.exp(-t / 0.025)
    chans = []
    for _ in range(2):  # decorrelated tails make it wide
        tail = lowpass(rng.standard_normal(n), sr, 350.0, order=1.5) * np.exp(-t / 1.1)
        chans.append(sub + 0.9 * crack + 0.6 * tail / (np.max(np.abs(tail)) or 1))
    x = np.stack(chans) * env_ar(n, sr, 0.002, 0.3)
    return _norm(np.tanh(1.6 * x))


def riser(sr: int, seconds: float, rng: np.random.Generator) -> F64:
    n = int(seconds * sr)
    t = np.arange(n) / n
    grow = t**2.2
    noise = _sweep_noise(rng, n, sr, 200.0 * (7000.0 / 200.0) ** t, q=2.0) * grow
    f = 110.0 * 4.0**t  # two octaves up
    ph = 2 * np.pi * np.cumsum(f) / sr
    tone = (np.sin(ph) + 0.5 * np.sin(2 * ph + 0.3) + 0.25 * np.sin(3 * ph)) * grow
    trem_rate = 2.0 + 14.0 * t**1.5
    trem = 0.65 + 0.35 * np.sin(2 * np.pi * np.cumsum(trem_rate) / sr)
    x = (0.8 * noise / (np.max(np.abs(noise)) or 1) + 0.5 * tone) * trem
    env = env_ar(n, sr, 0.3, 0.015)
    return _norm(np.stack([_pan(x * env, -0.4)[0], _pan(x * env, 0.4)[1]]))


def braam(sr: int, seconds: float, rng: np.random.Generator) -> F64:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rise, fall = 0.35, seconds * 0.45
    cutoff = 120.0 + 2300.0 * np.where(
        t < rise, (t / rise) ** 0.7, np.exp(-(t - rise) / max(fall, 0.1))
    )

    def lp4(f: F64, c: float) -> F64:
        g: F64 = 1.0 / np.sqrt(1.0 + (f / c) ** 4)
        return g

    out = np.zeros((2, n))
    for f0, gain in ((55.0, 1.0), (82.41, 0.6), (110.0, 0.5)):
        for det, pan in ((-7.0, -0.7), (0.0, 0.0), (7.0, 0.7)):
            f = f0 * 2 ** (det / 1200)
            ph = (f * t + rng.uniform()) % 1.0
            saw = 2.0 * ph - 1.0  # naive saw: aliasing is negligible at these low pitches
            out += gain * _pan(_tv_filter(saw, sr, lp4, cutoff), pan)
    out += 0.5 * np.sin(2 * np.pi * 27.5 * t)  # sub octave
    env = env_ar(n, sr, 0.08, seconds * 0.5)
    return _norm(np.tanh(0.9 * out / (np.max(np.abs(out)) or 1) * 1.8) * env)


_MAKERS = {"whoosh": whoosh, "impact": impact, "riser": riser, "braam": braam}


def write_wav(path: Path, x: F64, sr: int) -> Path:
    """(2, n) float → 24-bit stereo WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(x.T, -1.0, 1.0)
    ints = np.round(pcm * (2**23 - 1)).astype("<i4").reshape(-1)
    raw = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(raw)
    return path


def make_sfx(
    kind: str, seconds: float, out: Path, *, seed: int = 1, sample_rate: int = 48000
) -> Path:
    if kind not in _MAKERS:
        raise ValueError(f"unknown sound effect {kind!r}; use one of {KINDS}")
    if not 0.2 <= seconds <= 30:
        raise ValueError("sound effect length must be 0.2-30 s")
    rng = np.random.default_rng([seed, KINDS.index(kind)])
    return write_wav(out, _MAKERS[kind](sample_rate, seconds, rng), sample_rate)
