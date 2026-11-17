"""Procedural ambience and music, offline and deterministic (numpy only, no downloads).

Everything is generated in ~1 s stereo blocks and written straight to a 24-bit WAV, so
memory stays constant for any duration. The same (kind, seconds, seed, sample_rate)
always produces the same bytes.

Building blocks
  * Spectral noise: each 2048-sample frame gets Gaussian random spectrum shaped by a
    time-varying magnitude curve, inverse FFT, sqrt-Hann window, 50 % overlap-add. Gives
    rain hiss, wind with moving resonances, river babble, room tone.
  * Events: short synthesised sounds placed at random times (Poisson) into a carry buffer
    that spans block boundaries: rain drops, fire crackles, water bubbles (Minnaert
    model), bird calls.
  * Environments: long frame-rate noise events (_Swells: ocean waves, thunder rolls, cicada
    choruses), animals/objects at fixed places that call in bouts (_Singer, sounds from
    storyreel.audio.sfx), and reverb sends for distance (valley echo, forest).
  * Music: additive pads (detuned partials with a soft roll-off = a gentle low-pass),
    chord progressions per mood, bass, plucked arpeggios, an optional low drum, and a
    procedural hall reverb (storyreel.audio.ir) applied by FFT overlap-add convolution.

Looping: the stream is generated `xfade` seconds longer than asked; the extra tail is
equal-power crossfaded into the head, so the file's last sample flows into its first.
Level safety: a soft knee above -6 dBFS keeps every sample below -1 dBFS (no clipping).

These are synthetic sounds, judged here only by objective checks (duration, level,
spectrum, determinism), not by listening tests.
"""

from __future__ import annotations

import math
import os
import wave
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from storyreel.audio import ir, sfx
from storyreel.errors import StoryReelError

F64 = npt.NDArray[np.float64]
Gen = np.random.Generator

AMBIENCES = (
    "rain",
    "wind",
    "fire",
    "river",
    "birds",
    "night",
    "room",
    "ocean",
    "beach",
    "jungle",
    "forest",
    "thunderstorm",
    "ice",
    "waterfall",
    "hill_station",
    "village",
    "stream",
)
MOODS = ("calm", "happy", "sad", "tense", "mysterious", "epic")

MAX_SECONDS = 6 * 3600.0
_N = 2048  # spectral-noise frame
_HOP = _N // 2
_FRAMES_PER_BLOCK = 47  # block = 48128 samples (~1 s at 48 kHz)
_BLOCK = _HOP * _FRAMES_PER_BLOCK
_CEIL = 10 ** (-1.2 / 20)  # soft-clip ceiling, -1.2 dBFS
_KNEE = 0.5  # -6 dBFS: below this the signal is untouched
_XFADE_S = 2.0


class ProceduralError(StoryReelError):
    pass


# -- small helpers ---------------------------------------------------------------------
def _rngs(seed: int, name: str, n: int) -> list[Gen]:
    """Independent generators per layer, so each layer's draws never shift another's."""
    ss = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, zlib.crc32(name.encode())])
    return [np.random.default_rng(s) for s in ss.spawn(n)]


def _pan(p: float) -> tuple[float, float]:
    a = (p + 1.0) * math.pi / 4.0
    return math.cos(a), math.sin(a)


def _band(f: F64, lo: float, hi: float, order: float = 2.0) -> F64:
    """Smooth band-pass magnitude (Butterworth-like skirts) evaluated per frequency."""
    fs = np.maximum(f, 1.0)
    hp = 1.0 / np.sqrt(1.0 + (lo / fs) ** (2 * order))
    lp = 1.0 / np.sqrt(1.0 + (fs / hi) ** (2 * order))
    out: F64 = hp * lp
    return out


def _peak(f: F64, fc: F64 | float, q: float) -> F64:
    """Resonance shape, 1 at fc; fc may be a column (one centre per frame)."""
    bw = np.asarray(fc) / q
    out: F64 = 1.0 / (1.0 + ((f - np.asarray(fc)) / bw) ** 2)
    return out


def _soft_clip(x: F64) -> F64:
    """Identity below -6 dBFS; above, tanh-compress towards the -1.2 dBFS ceiling."""
    a = np.abs(x)
    over = a > _KNEE
    if np.any(over):
        span = _CEIL - _KNEE
        x = x.copy()
        x[over] = np.sign(x[over]) * (_KNEE + span * np.tanh((a[over] - _KNEE) / span))
    return x


def _midi_hz(m: float) -> float:
    return float(440.0 * 2.0 ** ((m - 69.0) / 12.0))


class _Smooth:
    """Smooth random control signal: random values every `period` s in [lo, hi],
    cosine-interpolated. Values are drawn in time order and old ones are dropped, so
    memory is constant however long the stream runs."""

    def __init__(self, rng: Gen, period: float, lo: float, hi: float) -> None:
        self.rng, self.period, self.lo, self.hi = rng, period, lo, hi
        self.k0 = 0
        self.vals: F64 = rng.random(2)

    def at(self, t: F64) -> F64:
        x = t / self.period
        k = np.floor(x).astype(np.int64)
        need = int(np.max(k)) + 2 - self.k0
        if need > self.vals.size:
            self.vals = np.concatenate([self.vals, self.rng.random(need - self.vals.size)])
        idx = k - self.k0
        w = 0.5 - 0.5 * np.cos(np.pi * (x - k))
        v = self.vals[idx] * (1.0 - w) + self.vals[idx + 1] * w
        drop = int(np.min(k)) - self.k0
        if drop > 0:
            self.vals = self.vals[drop:]
            self.k0 += drop
        out: F64 = self.lo + (self.hi - self.lo) * v
        return out


class _Events:
    """Carry buffer for sounds that may cross a block boundary."""

    def __init__(self, block: int, max_len: int) -> None:
        self.block, self.max_len = block, max_len
        self.buf = np.zeros((block + max_len, 2))

    def add(self, pos: int, mono: F64, pan: float) -> None:
        n = min(mono.size, self.max_len)
        seg = mono[:n]
        if mono.size > n:  # too long for the carry buffer: fade out instead of cutting
            m = min(n, 512)
            seg = seg.copy()
            seg[n - m :] *= np.linspace(1.0, 0.0, m)
        gl, gr = _pan(pan)
        self.buf[pos : pos + n, 0] += seg * gl
        self.buf[pos : pos + n, 1] += seg * gr

    def take(self) -> F64:
        out = self.buf[: self.block].copy()
        rest = self.buf[self.block :].copy()
        self.buf[:] = 0.0
        self.buf[: rest.shape[0]] = rest
        return out


class _SpectralNoise:
    """Stereo noise whose magnitude spectrum follows `shape(freqs, times) -> (frames, bins)`.

    Independent noise per channel; RMS of the output equals the RMS of the shape values
    (each frame's shape is used as-is, so shape carries the level)."""

    def __init__(self, rng: Gen, sr: int, shape: Callable[[F64, F64], F64]) -> None:
        self.rng, self.sr, self.shape = rng, sr, shape
        self.freqs: F64 = np.asarray(np.fft.rfftfreq(_N, 1.0 / sr), np.float64)
        self.win = np.sin(np.pi * (np.arange(_N) + 0.5) / _N)[None, :, None]
        self.carry = np.zeros((_HOP, 2))
        self.frame = 0
        # E|X|^2 = 2 mag^2 scale^2 per bin; summed over the mirrored spectrum and divided by
        # N^2 this gives var(x) = mean(mag^2), i.e. output RMS = RMS of the shape over bins
        self.scale = math.sqrt(_N / 2.0)

    def block(self) -> F64:
        k = _FRAMES_PER_BLOCK
        t = (self.frame + np.arange(k)) * _HOP / self.sr
        self.frame += k
        mag = self.shape(self.freqs, t)  # (k, bins) or (k, bins, 2) for per-channel shapes
        bins = self.freqs.size
        re = self.rng.standard_normal((k, bins, 2))
        im = self.rng.standard_normal((k, bins, 2))
        m3 = mag[:, :, None] if mag.ndim == 2 else mag
        spec = (re + 1j * im) * (m3 * self.scale)
        spec[:, 0, :] = 0.0
        frames = np.fft.irfft(spec, n=_N, axis=1) * self.win  # (k, N, 2)
        first, second = frames[:, :_HOP, :], frames[:, _HOP:, :]
        segs = first.copy()
        segs[0] += self.carry
        segs[1:] += second[:-1]
        self.carry = second[-1].copy()
        out: F64 = segs.reshape(k * _HOP, 2)
        return out


def _norm(shape: F64) -> F64:
    """Scale a (bins,) shape so that its RMS over bins is 1."""
    out: F64 = shape / max(1e-12, float(np.sqrt(np.mean(shape**2))))
    return out


# -- ambiences ---------------------------------------------------------------------------
Layer = Callable[[int], F64]  # block start sample -> (block, 2)


def _poisson_positions(rng: Gen, rate_per_s: float, sr: int) -> F64:
    n = rng.poisson(rate_per_s * _BLOCK / sr)
    out: F64 = np.sort(rng.integers(0, _BLOCK, size=n)).astype(np.float64)
    return out


def _decay_tone(sr: int, f0: float, tau: float, length: float, rise: float = 0.0) -> F64:
    n = max(2, int(length * sr))
    t = np.arange(n) / sr
    f = f0 * (1.0 + rise * t / max(length, 1e-6))
    phase = 2 * np.pi * np.cumsum(f) / sr
    env = np.exp(-t / tau) * np.minimum(1.0, t / 0.0005)
    out: F64 = np.sin(phase) * env
    return out


def _ambience_layers(kind: str, sr: int, seed: int) -> list[Layer]:
    r = _rngs(seed, "amb:" + kind, 48)  # spawn(n) children don't depend on n
    layers: list[Layer] = []

    def noise(rng: Gen, shape: Callable[[F64, F64], F64]) -> Layer:
        src = _SpectralNoise(rng, sr, shape)
        return lambda _s0: src.block()

    if kind == "rain":
        layers += _rain(r[0], r[1], r[2], r[3], sr)
    elif kind == "wind":
        gust = _Smooth(r[0], 2.2, 0.0, 1.0)
        slow = _Smooth(r[1], 9.0, 0.6, 1.0)

        def wind_shape(f: F64, t: F64) -> F64:
            g = gust.at(t)[:, None] ** 1.5
            amp = (0.3 + 0.7 * g) * slow.at(t)[:, None]
            fc = 250.0 + 900.0 * g
            body = _band(f[None, :], 70.0, 16000.0, 2.0) / np.maximum(f[None, :], 70.0) ** 0.3
            lp = 1.0 / np.sqrt(1.0 + (f[None, :] / fc) ** 4)
            whistle = 4.0 * g * _peak(f[None, :], 500.0 + 700.0 * g, 10.0)
            mag = body * lp * (1.0 + whistle)
            mag = mag / np.sqrt(np.mean(mag**2, axis=1, keepdims=True))
            out: F64 = 0.12 * amp * mag
            return out

        layers.append(noise(r[2], wind_shape))
    elif kind == "fire":
        flick = _Smooth(r[0], 0.25, 0.55, 1.2)
        base_rumble = _norm(_band(_freqs(sr), 30, 260, 2.0))
        base_hiss = _norm(_band(_freqs(sr), 2000, 10000, 2.0))

        def fire_shape(_f: F64, t: F64) -> F64:
            fl = flick.at(t)[:, None]
            out: F64 = 0.05 * fl * base_rumble[None, :] + 0.006 * base_hiss[None, :]
            return out

        layers.append(noise(r[1], fire_shape))
        burst = _Smooth(r[2], 0.7, 0.15, 2.2)
        ev = _Events(_BLOCK, int(0.03 * sr))
        crng = r[3]

        def crackle_layer(s0: int) -> F64:
            rate = 22.0 * float(burst.at(np.array([s0 / sr]))[0])
            for pos in _poisson_positions(crng, rate, sr):
                n = max(4, int(crng.uniform(0.0003, 0.003) * sr))
                x = crng.standard_normal(n + 1)
                click = np.diff(x) * np.exp(-np.arange(n) / (0.25 * n))
                click = np.convolve(click, (0.25, 0.5, 0.25))  # soften the very top end
                amp = min(0.3, 0.03 * (1.0 / max(crng.random(), 0.02)) ** 0.7)
                ev.add(int(pos), amp * click, crng.uniform(-0.6, 0.6))
            for pos in _poisson_positions(crng, 0.8, sr):  # bigger pops
                n = int(crng.uniform(0.005, 0.015) * sr)
                x = np.convolve(crng.standard_normal(n), np.ones(4) / 4, mode="same")
                pop = x * np.exp(-np.arange(n) / (0.2 * n))
                ev.add(int(pos), crng.uniform(0.15, 0.3) * pop, crng.uniform(-0.5, 0.5))
            return ev.take()

        layers.append(crackle_layer)
    elif kind == "river":
        flow = _Smooth(r[0], 4.0, 0.85, 1.1)
        brng = r[1]
        base = _norm(_band(_freqs(sr), 60, 2400, 1.5))

        def river_shape(f: F64, t: F64) -> F64:
            k = t.size
            # three moving resonances per frame = gurgle/babble
            fc = np.exp(brng.uniform(math.log(200), math.log(1500), size=(k, 3)))
            gain = brng.uniform(0.5, 3.0, size=(k, 3))
            res = np.zeros((k, f.size))
            for j in range(3):
                res += gain[:, j : j + 1] * _peak(f[None, :], fc[:, j : j + 1], 6.0)
            mag = base[None, :] * (1.0 + res)
            mag = mag / np.sqrt(np.mean(mag**2, axis=1, keepdims=True))
            out: F64 = 0.07 * flow.at(t)[:, None] * mag
            return out

        layers.append(noise(r[2], river_shape))
        ev = _Events(_BLOCK, int(0.08 * sr))
        bub = r[3]

        def bubble_layer(_s0: int) -> F64:
            for pos in _poisson_positions(bub, 55.0, sr):
                f0 = math.exp(bub.uniform(math.log(350), math.log(1700)))
                d = 0.043 * f0 + 0.0014 * f0**1.5  # Minnaert bubble damping (van den Doel)
                length = min(0.08, 5.0 / d)
                tone = _decay_tone(sr, f0, 1.0 / d, length, rise=0.1 * d * length)
                ev.add(int(pos), bub.uniform(0.008, 0.05) * tone, bub.uniform(-0.8, 0.8))
            return ev.take()

        layers.append(bubble_layer)
    elif kind == "birds":
        air = _Smooth(r[0], 5.0, 0.6, 1.0)
        base = _norm(_band(_freqs(sr), 250, 6000, 1.5))

        def leaves(_f: F64, t: F64) -> F64:
            out: F64 = 0.008 * air.at(t)[:, None] * base[None, :]
            return out

        layers.append(noise(r[1], leaves))
        # five resident songbirds, each its own species, singing bouts from its own spot
        styles = r[2].permutation(len(sfx.SONGBIRDS))[:5]
        birds = [
            _Singer(
                r[3 + i],
                sfx.songbird(r[3 + i], sfx.SONGBIRDS[int(st)]),
                sr,
                gain=float(r[3 + i].uniform(0.06, 0.2)),
                pan=float(r[3 + i].uniform(-0.9, 0.9)),
                cutoff=float(r[3 + i].choice([0.0, 0.0, 6000.0])),
                bout=(2, 6),
                gap=(0.6, 3.0),
                rest=(3.0, 15.0),
                first=(0.0, 4.0),
            )
            for i, st in enumerate(styles)
        ]
        layers.append(_wet(_chorus(birds, 2.0, sr), "hall", 0.15, sr))
    elif kind == "night":
        base = _norm(_band(_freqs(sr), 40, 600, 1.5))
        drift = _Smooth(r[0], 6.0, 0.6, 1.0)

        def night_air(_f: F64, t: F64) -> F64:
            out: F64 = 0.012 * drift.at(t)[:, None] * base[None, :]
            return out

        layers.append(noise(r[1], night_air))
        crickets = [_gated(_cricket(r[2 + i], sr, i), r[10 + i], sr) for i in range(4)]
        layers.append(lambda s0: np.sum([c(s0) for c in crickets], axis=0))
        owl = _Singer(
            r[6],
            sfx.hoot(r[6]),
            sr,
            gain=0.05,
            pan=float(r[6].uniform(-0.8, 0.8)),
            cutoff=1800.0,
            bout=(1, 3),
            gap=(3.0, 8.0),
            rest=(20.0, 60.0),
            first=(2.0, 25.0),
        )
        layers.append(_wet(_chorus([owl], 2.0, sr), "hall", 0.3, sr))
    elif kind == "room":
        hvac = _norm(_band(_freqs(sr), 40, 180, 2.0))
        tone = _norm(_band(_freqs(sr), 40, 3500, 1.0) / np.sqrt(np.maximum(_freqs(sr), 40.0)))

        def room_shape(_f: F64, t: F64) -> F64:
            row = 0.006 * hvac + 0.008 * tone
            out: F64 = np.broadcast_to(row, (t.size, row.size))
            return out

        layers.append(noise(r[1], room_shape))

        def hum(s0: int) -> F64:
            t = (s0 + np.arange(_BLOCK)) / sr
            x = 0.0010 * np.sin(2 * np.pi * 50 * t) + 0.0006 * np.sin(2 * np.pi * 100 * t)
            out: F64 = np.stack([x, x], axis=1)
            return out

        layers.append(hum)
    elif kind in _RECIPES:
        layers += _RECIPES[kind](r, sr)
    else:
        raise ProceduralError(
            f"Unknown ambience {kind!r}.", fix=f"Choose one of: {', '.join(AMBIENCES)}."
        )
    return layers


def _freqs(sr: int) -> F64:
    out: F64 = np.asarray(np.fft.rfftfreq(_N, 1.0 / sr), np.float64)
    return out


def _bird_species(rng: Gen) -> Callable[[Gen, int], F64]:
    """A call template with its own pitch range and rhythm; each call varies slightly."""
    style = int(rng.integers(0, 3))
    hi, lo = rng.uniform(4500, 7000), rng.uniform(2500, 4000)
    syl, gap = rng.uniform(0.04, 0.09), rng.uniform(0.04, 0.09)

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        if style == 0:  # descending "tweet" syllables
            for _ in range(int(r.integers(3, 7))):
                n = int(syl * sr * r.uniform(0.85, 1.15))
                f = np.linspace(hi, lo, n) * r.uniform(0.95, 1.05)
                parts += [_chirp(f, sr), np.zeros(int(gap * sr))]
        elif style == 1:  # trill: fast FM
            n = int(r.uniform(0.5, 1.0) * sr)
            t = np.arange(n) / sr
            f = (lo + hi) / 2 + (hi - lo) / 3 * np.sin(2 * np.pi * r.uniform(18, 30) * t)
            parts.append(_chirp(f, sr))
        else:  # two-note whistle with glide
            for a, b in ((lo, lo * 1.12), (lo * 1.3, lo * 1.25)):
                n = int(r.uniform(0.14, 0.24) * sr)
                parts += [_chirp(np.linspace(a, b, n), sr), np.zeros(int(0.05 * sr))]
        out: F64 = np.concatenate(parts)
        return out

    return call


def _chirp(f: F64, sr: int) -> F64:
    n = f.size
    phase = 2 * np.pi * np.cumsum(f) / sr
    env = np.sin(np.pi * np.arange(n) / max(n - 1, 1)) ** 2
    out: F64 = (np.sin(phase) + 0.15 * np.sin(2 * phase)) * env
    return out


def _cricket(rng: Gen, sr: int, i: int) -> Layer:
    """Pulse trains on a ~4-5 kHz carrier: chirps of 3-4 pulses, repeating."""
    fc = rng.uniform(4200, 5200)
    pulse_rate = rng.uniform(26, 34)
    n_pulses = int(rng.integers(3, 5))
    period = rng.uniform(0.55, 0.95)
    offset = rng.uniform(0, period)
    amp = rng.uniform(0.02, 0.05) * (0.6 if i == 3 else 1.0)
    gl, gr = _pan(rng.uniform(-0.9, 0.9))
    wobble = _Smooth(rng, 7.0, 0.97, 1.03)

    def layer(s0: int) -> F64:
        t = (s0 + np.arange(_BLOCK)) / sr
        per = period * float(wobble.at(np.array([s0 / sr]))[0])
        tau = np.mod(t + offset, per)
        on = tau < n_pulses / pulse_rate
        pulse = np.sin(np.pi * np.mod(tau * pulse_rate, 1.0)) ** 2
        x = amp * on * pulse * (np.sin(2 * np.pi * fc * t) + 0.2 * np.sin(4 * np.pi * fc * t))
        out: F64 = np.stack([x * gl, x * gr], axis=1)
        return out

    return layer


# -- environment building blocks -----------------------------------------------------------
def _rain(
    ri: Gen,
    rn: Gen,
    rd: Gen,
    rp: Gen,
    sr: int,
    level: float = 0.07,
    low: float = 0.35,
    drops: float = 45.0,
    plips: float = 4.0,
) -> list[Layer]:
    """Rain hiss (+ low roar) with individual drops and water "plips"."""
    inten = _Smooth(ri, 3.0, 0.75, 1.1)

    def rain_shape(f: F64, t: F64) -> F64:
        base = _norm(_band(f, 500, 9000, 2.0) + low * _band(f, 80, 450, 2.0))
        out: F64 = level * inten.at(t)[:, None] * base[None, :]
        return out

    src = _SpectralNoise(rn, sr, rain_shape)
    ev = _Events(_BLOCK, int(0.05 * sr))

    def drop_layer(_s0: int) -> F64:
        for pos in _poisson_positions(rd, drops, sr):
            amp = math.exp(rd.uniform(math.log(0.015), math.log(0.12)))
            tone = _decay_tone(sr, rd.uniform(2200, 7500), rd.uniform(0.0012, 0.004), 0.03)
            ev.add(int(pos), amp * tone, rd.uniform(-0.9, 0.9))
        for pos in _poisson_positions(rp, plips, sr):  # "plip": rising bubble on water
            amp = rp.uniform(0.04, 0.14)
            tone = _decay_tone(sr, rp.uniform(900, 2200), 0.007, 0.04, rise=0.6)
            ev.add(int(pos), amp * tone, rp.uniform(-0.8, 0.8))
        return ev.take()

    return [lambda _s0: src.block(), drop_layer]


class _Singer:
    """One animal (or bell, or cow) at a fixed place: a bout of calls with short gaps,
    then a longer rest. Every call is re-synthesised, so no two are identical; distance
    is a low-pass `cutoff` and a lower `gain`."""

    def __init__(
        self,
        rng: Gen,
        call: sfx.Call,
        sr: int,
        *,
        gain: float,
        pan: float,
        cutoff: float = 0.0,
        bout: tuple[int, int] = (1, 4),
        gap: tuple[float, float] = (1.0, 4.0),
        rest: tuple[float, float] = (5.0, 30.0),
        first: tuple[float, float] = (0.0, 10.0),
        busy: float = 1.0,
    ) -> None:
        self.rng, self.call, self.sr = rng, call, sr
        self.gain, self.pan, self.cutoff = gain, pan, cutoff
        self.bout, self.gap, self.rest, self.busy = bout, gap, rest, busy
        self.next = float(rng.uniform(*first))
        self.left = int(rng.integers(bout[0], bout[1] + 1))

    def fill(self, ev: _Events, s0: int) -> None:
        r, sr = self.rng, self.sr
        while int(self.next * sr) < s0 + _BLOCK:
            pos = max(0, int(self.next * sr) - s0)
            x = self.call(r, sr)
            if self.cutoff:
                x = sfx.lowpass(x, sr, self.cutoff)
            amp = self.gain * r.uniform(0.6, 1.0)
            ev.add(pos, amp * x, float(np.clip(self.pan + r.uniform(-0.08, 0.08), -1, 1)))
            self.left -= 1
            span = self.busy * x.size / sr
            if self.left > 0:
                self.next += span + r.uniform(*self.gap)
            else:
                self.left = int(r.integers(self.bout[0], self.bout[1] + 1))
                self.next += span + r.uniform(*self.rest)


def _chorus(singers: list[_Singer], max_s: float, sr: int) -> Layer:
    ev = _Events(_BLOCK, int(max_s * sr))

    def layer(s0: int) -> F64:
        for s in singers:
            s.fill(ev, s0)
        return ev.take()

    return layer


def _wet(layer: Layer, room: str, mix: float, sr: int) -> Layer:
    """Send one layer through the procedural reverb (distance, forest/valley echo)."""
    rv = _Reverb(room, sr, mix)
    return lambda s0: rv.process(layer(s0))


def _gated(layer: Layer, rng: Gen, sr: int) -> Layer:
    """An insect that stops and starts: a slow random gate (smooth, 0..1)."""
    g = _Smooth(rng, float(rng.uniform(7.0, 14.0)), -0.4, 1.8)

    def out(s0: int) -> F64:
        t = (s0 + np.arange(_BLOCK)) / sr
        v = np.clip(g.at(t), 0.0, 1.0)
        res: F64 = layer(s0) * (v * v * (3 - 2 * v))[:, None]
        return res

    return out


@dataclass
class _Swell:
    start: float
    dur: float
    gains: tuple[float, float]
    power: Callable[[F64], F64]  # local times (k,), clipped to [0, dur] -> (k, bins) power


class _Swells:
    """Long noise events (waves, thunder rolls, cicada choruses) evaluated per spectral
    frame. `spawn(t)` builds the event starting at t and returns the time to the next
    start, so events are made in time order and dropped once finished; their powers add
    (they are independent noises), giving a per-channel magnitude for _SpectralNoise."""

    def __init__(self, spawn: Callable[[float], tuple[_Swell, float]], first: float) -> None:
        self.spawn, self.next_t = spawn, first
        self.active: list[_Swell] = []

    def ensure(self, t_end: float) -> None:
        while self.next_t <= t_end:
            sw, gap = self.spawn(self.next_t)
            self.active.append(sw)
            self.next_t += max(gap, 0.05)

    def power(self, t: F64, bins: int) -> F64:
        self.ensure(float(t[-1]))
        acc = np.zeros((t.size, bins, 2))
        keep: list[_Swell] = []
        for sw in self.active:
            tl = t - sw.start
            on = (tl >= 0) & (tl <= sw.dur)
            if np.any(on):
                p = sw.power(np.clip(tl, 0.0, sw.dur)) * on[:, None]
                acc[:, :, 0] += p * sw.gains[0] ** 2
                acc[:, :, 1] += p * sw.gains[1] ** 2
            if sw.start + sw.dur >= t[-1]:
                keep.append(sw)
        self.active = keep
        return acc


def _rownorm(m: F64) -> F64:
    out: F64 = m / np.maximum(1e-12, np.sqrt(np.mean(m**2, axis=1, keepdims=True)))
    return out


def _breeze(
    rg: Gen,
    rs: Gen,
    rn: Gen,
    sr: int,
    level: float,
    fc: tuple[float, float] = (250.0, 1100.0),
    whistle: float = 2.0,
    period: float = 2.6,
) -> Layer:
    """Wind in gusts: brighter and louder as each gust rises (a lighter `wind`)."""
    gust = _Smooth(rg, period, 0.0, 1.0)
    slow = _Smooth(rs, 9.0, 0.6, 1.0)

    def shape(f: F64, t: F64) -> F64:
        g = gust.at(t)[:, None] ** 1.5
        amp = (0.35 + 0.65 * g) * slow.at(t)[:, None]
        body = _band(f[None, :], 60.0, 16000.0, 2.0) / np.maximum(f[None, :], 60.0) ** 0.3
        lp = 1.0 / np.sqrt(1.0 + (f[None, :] / (fc[0] + (fc[1] - fc[0]) * g)) ** 4)
        mag = body * lp * (1.0 + whistle * g * _peak(f[None, :], 450.0 + 800.0 * g, 10.0))
        out: F64 = level * amp * _rownorm(mag)
        return out

    src = _SpectralNoise(rn, sr, shape)
    return lambda _s0: src.block()


def _bed(rng: Gen, sr: int, level: float, lo: float, hi: float, period: float = 6.0) -> Layer:
    """Steady air/surf bed with a slow drift."""
    drift = _Smooth(rng, period, 0.65, 1.0)
    base = _norm(_band(_freqs(sr), lo, hi, 1.5))

    def shape(_f: F64, t: F64) -> F64:
        out: F64 = level * drift.at(t)[:, None] * base[None, :]
        return out

    src = _SpectralNoise(rng, sr, shape)
    return lambda _s0: src.block()


def _waves(
    rw: Gen,
    rn: Gen,
    sr: int,
    *,
    level: float,
    gap: tuple[float, float],
    dur: tuple[float, float],
    crash: float,
    foam: float,
    spread: float = 0.6,
) -> Layer:
    """Ocean swells: each wave builds (a dark rumble that brightens as it steepens), crashes
    (broadband burst) and washes up as fizzing foam that slides back; waves overlap and
    differ in size, length, timing and position."""
    f = _freqs(sr)
    body = _band(f, 35, 20000, 2.0)[None, :]
    crash_s = (_norm(_band(f, 80, 12000, 2.0) / np.maximum(f, 80.0) ** 0.25) ** 2)[None, :]
    foam_b = _band(f, 900, 16000, 2.0)[None, :]

    def spawn(t0: float) -> tuple[_Swell, float]:
        d = rw.uniform(*dur)
        tc = d * rw.uniform(0.35, 0.55)
        big = level * math.exp(rw.normal(0.0, 0.3))
        bright = rw.uniform(0.7, 1.3)
        tw = (d - tc) / rw.uniform(2.0, 3.0)
        tcr = rw.uniform(0.4, 0.9)
        fade = max(0.5, min(1.5, d - tc))

        def power(tl: F64) -> F64:
            xb = np.clip(tl / tc, 0.0, 1.0)
            after = np.maximum(tl - tc, 0.0)
            a_b = xb**2 * np.exp(-after / 0.6)
            fc_b = 200.0 + 1200.0 * bright * xb[:, None]
            sb = _rownorm(body / np.sqrt(1.0 + (f[None, :] / fc_b) ** 4))
            rise = np.clip((tl - (tc - 0.35)) / 0.35, 0.0, 1.0) ** 2
            a_c = crash * rise * np.exp(-after / tcr)
            a_f = foam * (1.0 - np.exp(-after / 0.3)) * np.exp(-after / tw)
            fc_f = 2500.0 + 9000.0 * bright * np.exp(-after / (2.0 * tw))
            sf = _rownorm(foam_b / np.sqrt(1.0 + (f[None, :] / fc_f[:, None]) ** 4))
            grain = 0.5 + rw.random((tl.size, 1))  # fizz: frame-to-frame flicker
            end = np.clip((d - tl) / fade, 0.0, 1.0) ** 2
            p = (
                (a_b**2)[:, None] * 0.5 * sb**2
                + (a_c**2)[:, None] * crash_s
                + (a_f**2)[:, None] * grain * sf**2
            )
            out: F64 = ((big * end) ** 2)[:, None] * p
            return out

        return _Swell(t0, d, _pan(rw.uniform(-spread, spread)), power), rw.uniform(*gap)

    sw = _Swells(spawn, rw.uniform(0.0, 2.0))
    src = _SpectralNoise(rn, sr, lambda fr, t: np.sqrt(sw.power(t, fr.size)))
    return lambda _s0: src.block()


class _Thunder:
    """Lightning strikes at random: distant ones are slow, low rolling rumbles; close ones
    start with a tearing crack (sample-accurate) and a heavier, brighter rumble."""

    def __init__(
        self, rng: Gen, rn: Gen, sr: int, gap: tuple[float, float], close_p: float
    ) -> None:
        self.rng, self.sr, self.gap, self.close_p = rng, sr, gap, close_p
        self.f = _freqs(sr)
        self.body = _band(self.f, 18.0, 20000.0, 2.0)[None, :] / np.maximum(self.f, 30.0) ** 0.5
        self.swells = _Swells(self._spawn, rng.uniform(1.0, 6.0))
        self.cracks: list[tuple[float, float, float]] = []
        self.ev = _Events(_BLOCK, int(1.0 * sr))
        self.src = _SpectralNoise(rn, sr, lambda fr, t: np.sqrt(self.swells.power(t, fr.size)))

    def _spawn(self, t0: float) -> tuple[_Swell, float]:
        r = self.rng
        close = bool(r.random() < self.close_p)
        d = r.uniform(9.0, 16.0) if close else r.uniform(6.0, 13.0)
        lvl = (0.4 if close else 0.18) * math.exp(r.normal(0.0, 0.3))
        pan = float(r.uniform(-0.7, 0.7))
        nb = int(r.integers(2, 7))
        bt = np.sort(r.uniform(0.0, 0.55 * d, nb))
        bt[0] = 0.0
        ba = r.uniform(0.3, 1.0, nb)
        ba[0] = 1.0
        bdec = r.uniform(0.8, 2.5, nb)
        batt = 0.04 if close else float(r.uniform(0.3, 1.2))
        fc0 = r.uniform(900.0, 1600.0) if close else r.uniform(150.0, 380.0)
        f, body = self.f, self.body

        def power(tl: F64) -> F64:
            e = np.zeros(tl.size)
            for i in range(nb):
                x = tl - bt[i]
                rise = np.clip(x / batt, 0.0, 1.0) ** 2
                e += ba[i] * rise * np.exp(-np.maximum(x, 0.0) / bdec[i])
            e *= np.clip((d - tl) / 1.5, 0.0, 1.0)
            fc = 60.0 + fc0 * np.exp(-tl / (d / 3.0))
            s = _rownorm(body / np.sqrt(1.0 + (f[None, :] / fc[:, None]) ** 4))
            out: F64 = ((lvl * e) ** 2)[:, None] * s**2
            return out

        if close:
            self.cracks.append((t0, lvl, pan))
        return _Swell(t0, d, _pan(pan), power), float(r.uniform(*self.gap))

    def rumble(self, _s0: int) -> F64:
        return self.src.block()

    def crack(self, s0: int) -> F64:
        sr = self.sr
        self.swells.ensure((s0 + _BLOCK) / sr)
        keep = []
        for t0, lvl, pan in self.cracks:
            if int(t0 * sr) < s0 + _BLOCK:
                x = sfx.thunder_crack(self.rng, sr)
                self.ev.add(max(0, int(t0 * sr) - s0), min(1.0, 1.5 * lvl) * x, pan)
            else:
                keep.append((t0, lvl, pan))
        self.cracks = keep
        return self.ev.take()


def _cicada(rw: Gen, rn: Gen, sr: int, level: float) -> Layer:
    """Cicada chorus: a narrow buzzing band, pulsed at 90-220 Hz, that swells up over
    several seconds, holds, and dies away, then rests."""
    fc = rw.uniform(3800.0, 7000.0)
    rate = rw.uniform(90.0, 220.0)
    syl = rw.uniform(6.0, 12.0) if rw.random() < 0.5 else 0.0
    f = _freqs(sr)
    spec = (_norm(_peak(f, fc, 3.0) * _band(f, 1500, 16000, 2.0)) ** 2)[None, :]

    def spawn(t0: float) -> tuple[_Swell, float]:
        d = rw.uniform(6.0, 18.0)
        att = rw.uniform(0.25, 0.5) * d
        rel = rw.uniform(1.0, 2.5)
        lvl = level * rw.uniform(0.5, 1.0)

        def power(tl: F64) -> F64:
            e = lvl * np.minimum(1.0, tl / att) ** 2 * np.clip((d - tl) / rel, 0.0, 1.0)
            out: F64 = (e**2)[:, None] * spec
            return out

        return _Swell(t0, d, _pan(rw.uniform(-0.8, 0.8)), power), d + rw.uniform(2.0, 18.0)

    sw = _Swells(spawn, rw.uniform(0.0, 4.0))
    src = _SpectralNoise(rn, sr, lambda fr, t: np.sqrt(sw.power(t, fr.size)))

    def layer(s0: int) -> F64:
        t = (s0 + np.arange(_BLOCK)) / sr
        am = (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)) ** 2
        if syl:
            am = am * (0.5 + 0.5 * np.sin(2 * np.pi * syl * t)) ** 0.5
        out: F64 = src.block() * am[:, None]
        return out

    return layer


def _katydid(rng: Gen, sr: int, amp: float) -> Layer:
    """Fast insect pulse train (higher and quicker than the night crickets)."""
    fc = rng.uniform(5500, 9000)
    rate = rng.uniform(40, 70)
    n_p = int(rng.integers(2, 8))
    per = rng.uniform(0.25, 0.6)
    off = rng.uniform(0, per)
    gl, gr = _pan(rng.uniform(-0.9, 0.9))

    def layer(s0: int) -> F64:
        t = (s0 + np.arange(_BLOCK)) / sr
        tau = np.mod(t + off, per)
        on = tau < n_p / rate
        pulse = np.sin(np.pi * np.mod(tau * rate, 1.0)) ** 2
        x = amp * on * pulse * np.sin(2 * np.pi * fc * t)
        out: F64 = np.stack([x * gl, x * gr], axis=1)
        return out

    return layer


def _drips(rng: Gen, sr: int, spots: int, amp: float) -> Layer:
    """Leaves dripping into puddles: a few spots, each with its own pitch and slowly
    varying rhythm, that start and stop, plus stray drops."""
    ev = _Events(_BLOCK, int(0.1 * sr))
    # per spot: pitch, period, pan, next time, on (1) / off (0)
    st = [
        [
            rng.uniform(700, 2200),
            rng.uniform(0.8, 3.5),
            rng.uniform(-0.8, 0.8),
            rng.uniform(0, 3),
            1.0,
        ]
        for _ in range(spots)
    ]

    def layer(s0: int) -> F64:
        end = s0 + _BLOCK
        for s in st:
            while int(s[3] * sr) < end:
                if s[4] > 0:
                    a = amp * rng.uniform(0.5, 1.0)
                    ev.add(max(0, int(s[3] * sr) - s0), a * sfx.drip(rng, sr, s[0]), s[2])
                s[3] += s[1] * rng.uniform(0.8, 1.25)
                if rng.random() < 0.06:  # this spot dries up, or starts again
                    s[4] = 1.0 - s[4]
                    s[1] = rng.uniform(0.8, 3.5)
        for pos in _poisson_positions(rng, 0.4, sr):
            x = sfx.drip(rng, sr, rng.uniform(600, 2600))
            ev.add(int(pos), amp * rng.uniform(0.3, 0.8) * x, rng.uniform(-0.9, 0.9))
        return ev.take()

    return layer


def _bubbles(
    rng: Gen,
    sr: int,
    rate: Callable[[int], float],
    f_lo: float,
    f_hi: float,
    amp: tuple[float, float],
) -> Layer:
    """Minnaert-model water bubbles (as in `river`), at a time-varying rate."""
    ev = _Events(_BLOCK, int(0.08 * sr))

    def layer(s0: int) -> F64:
        for pos in _poisson_positions(rng, rate(s0), sr):
            f0 = math.exp(rng.uniform(math.log(f_lo), math.log(f_hi)))
            d = 0.043 * f0 + 0.0014 * f0**1.5
            length = min(0.08, 5.0 / d)
            tone = _decay_tone(sr, f0, 1.0 / d, length, rise=0.1 * d * length)
            ev.add(int(pos), rng.uniform(*amp) * tone, rng.uniform(-0.85, 0.85))
        return ev.take()

    return layer


def _rustle(rg: Gen, rn: Gen, sr: int, level: float) -> Layer:
    """Leaves rustling as gusts pass: bright noise, flickering frame to frame."""
    gust = _Smooth(rg, 1.8, 0.0, 1.0)
    base = _norm(_band(_freqs(sr), 1200, 9000, 1.5))

    def shape(_f: F64, t: F64) -> F64:
        g = np.clip(gust.at(t) - 0.25, 0.0, 1.0)[:, None] ** 1.5
        flick = 0.4 + 1.2 * rg.random((t.size, 1))
        out: F64 = level * g * flick * base[None, :]
        return out

    src = _SpectralNoise(rn, sr, shape)
    return lambda _s0: src.block()


def _songbirds(
    r: list[Gen],
    base: int,
    n: int,
    sr: int,
    gain: tuple[float, float],
    cutoff: list[float],
    rest: tuple[float, float],
) -> list[_Singer]:
    """`n` resident songbirds of different species, uses generators r[base .. base+n]."""
    order = r[base].permutation(len(sfx.SONGBIRDS))
    out = []
    for i in range(n):
        g = r[base + 1 + i]
        style = sfx.SONGBIRDS[int(order[i % len(order)])]
        out.append(
            _Singer(
                g,
                sfx.songbird(g, style),
                sr,
                gain=float(g.uniform(*gain)),
                pan=float(g.uniform(-0.9, 0.9)),
                cutoff=float(g.choice(cutoff)),
                bout=(2, 6),
                gap=(0.8, 4.0),
                rest=rest,
                first=(0.0, 6.0),
            )
        )
    return out


def _caller(
    rng: Gen,
    call: sfx.Call,
    sr: int,
    gain: float,
    cutoff: float,
    rest: tuple[float, float],
    gap: tuple[float, float] = (3.0, 8.0),
    first: tuple[float, float] = (2.0, 20.0),
) -> _Singer:
    """An occasional caller (owl, cuckoo, rooster...) somewhere around the listener."""
    pan = float(rng.uniform(-0.9, 0.9))
    return _Singer(
        rng,
        call,
        sr,
        gain=gain,
        pan=pan,
        cutoff=cutoff,
        bout=(1, 3),
        gap=gap,
        rest=rest,
        first=first,
    )


# -- environment recipes -------------------------------------------------------------------
def _amb_ocean(r: list[Gen], sr: int) -> list[Layer]:
    return [
        _bed(r[0], sr, 0.015, 30.0, 500.0, 8.0),  # far surf
        _waves(r[1], r[2], sr, level=0.06, gap=(4.5, 9.0), dur=(6.0, 12.0), crash=1.0, foam=1.0),
        _breeze(r[3], r[4], r[5], sr, 0.012),
    ]


def _amb_beach(r: list[Gen], sr: int) -> list[Layer]:
    gulls = [
        _Singer(
            r[10 + i],
            sfx.gull(r[10 + i]),
            sr,
            gain=float(r[10 + i].uniform(0.05, 0.12)),
            pan=float(r[10 + i].uniform(-0.9, 0.9)),
            cutoff=float(r[10 + i].choice([0.0, 4500.0])),
            bout=(1, 4),
            gap=(0.6, 3.0),
            rest=(8.0, 35.0),
            first=(0.5, 12.0),
        )
        for i in range(3)
    ]
    return [
        _bed(r[0], sr, 0.008, 30.0, 400.0, 8.0),
        _waves(r[1], r[2], sr, level=0.045, gap=(5.0, 10.0), dur=(5.0, 9.0), crash=0.45, foam=1.3),
        _breeze(r[3], r[4], r[5], sr, 0.018, (400.0, 1500.0), 1.0),
        _wet(_chorus(gulls, 3.0, sr), "hall", 0.12, sr),
    ]


def _amb_jungle(r: list[Gen], sr: int) -> list[Layer]:
    birds = _songbirds(r, 20, 5, sr, (0.04, 0.13), [0.0, 6000.0, 4000.0], (3.0, 14.0))
    birds += [
        _caller(r[26], sfx.parrot(r[26]), sr, 0.07, 6000.0, (6.0, 25.0), (0.5, 2.0), (1.0, 10.0)),
        _caller(r[27], sfx.koel(r[27]), sr, 0.06, 3500.0, (15.0, 45.0)),
    ]
    far = [
        _caller(r[28], sfx.whoop(r[28]), sr, 0.07, 1800.0, (15.0, 45.0)),
        _caller(r[29], sfx.hoot(r[29]), sr, 0.06, 1500.0, (20.0, 60.0)),
    ]
    insects = [_gated(_katydid(r[30 + i], sr, 0.012), r[34 + i], sr) for i in range(3)]
    return [
        _bed(r[0], sr, 0.02, 60.0, 2500.0, 7.0),  # warm humid air
        _cicada(r[2], r[3], sr, 0.05),
        _cicada(r[4], r[5], sr, 0.035),
        lambda s0: np.sum([c(s0) for c in insects], axis=0),
        _wet(_chorus(birds, 7.0, sr), "hall", 0.28, sr),
        _wet(_chorus(far, 7.0, sr), "cathedral", 0.5, sr),
        _drips(r[6], sr, 3, 0.05),
    ]


def _amb_forest(r: list[Gen], sr: int) -> list[Layer]:
    birds = _songbirds(r, 20, 4, sr, (0.05, 0.14), [0.0, 5500.0], (6.0, 25.0))
    other = sfx.cuckoo(r[26]) if r[26].random() < 0.5 else sfx.dove(r[26])
    birds.append(_caller(r[26], other, sr, 0.06, 2500.0, (15.0, 45.0), (1.0, 3.0)))
    pecker = sfx.woodpecker(r[27])
    birds.append(_caller(r[27], pecker, sr, 0.1, 4500.0, (20.0, 70.0), (2.0, 6.0), (4.0, 30.0)))
    return [
        _breeze(r[0], r[1], r[2], sr, 0.02, (300.0, 1200.0), 1.0, 3.0),
        _rustle(r[3], r[4], sr, 0.02),
        _wet(_chorus(birds, 4.0, sr), "hall", 0.22, sr),
    ]


def _amb_thunderstorm(r: list[Gen], sr: int) -> list[Layer]:
    th = _Thunder(r[6], r[7], sr, (7.0, 22.0), 0.35)
    return [
        *_rain(r[0], r[1], r[2], r[3], sr, level=0.11, low=0.7, drops=90.0, plips=8.0),
        _breeze(r[4], r[5], r[8], sr, 0.05, (200.0, 900.0), 2.0, 3.5),
        th.rumble,
        th.crack,
    ]


def _amb_ice(r: list[Gen], sr: int) -> list[Layer]:
    gust = _Smooth(r[0], 3.0, 0.0, 1.0)
    slow = _Smooth(r[1], 11.0, 0.5, 1.0)
    howl = [
        _Smooth(r[2], 4.0, 280.0, 520.0),
        _Smooth(r[3], 5.5, 520.0, 950.0),
        _Smooth(r[4], 3.2, 1400.0, 2600.0),
    ]
    wq = ((22.0, 30.0), (12.0, 45.0), (5.0, 70.0))  # (weight, Q) of each howling resonance
    snow = _norm(_band(_freqs(sr), 3000, 12000, 2.0))[None, :]

    def ice_shape(f: F64, t: F64) -> F64:
        g = gust.at(t)[:, None] ** 1.5
        body = _band(f[None, :], 60.0, 14000.0, 2.0) / np.maximum(f[None, :], 60.0) ** 0.4
        body = body / np.sqrt(1.0 + (f[None, :] / (600.0 + 2500.0 * g)) ** 4)
        res = np.ones((t.size, f.size))
        for h, (w, q) in zip(howl, wq, strict=True):
            res += w * (0.2 + g) * _peak(f[None, :], h.at(t)[:, None], q)
        wind = (0.25 + 0.75 * g) * slow.at(t)[:, None] * _rownorm(body * res)
        out: F64 = 0.05 * wind + 0.008 * g * snow
        return out

    src = _SpectralNoise(r[5], sr, ice_shape)
    ev = _Events(_BLOCK, int(1.3 * sr))
    er = r[6]

    def ice_events(_s0: int) -> F64:
        for pos in _poisson_positions(er, 0.12, sr):
            x = sfx.lowpass(sfx.ice_creak(er, sr), sr, er.uniform(2000, 6000))
            ev.add(int(pos), er.uniform(0.04, 0.12) * x, er.uniform(-0.8, 0.8))
        for pos in _poisson_positions(er, 0.07, sr):
            x = sfx.lowpass(sfx.ice_crack(er, sr), sr, er.uniform(2500, 9000))
            ev.add(int(pos), er.uniform(0.06, 0.2) * x, er.uniform(-0.8, 0.8))
        return ev.take()

    return [lambda _s0: src.block(), _wet(ice_events, "hall", 0.35, sr)]


def _amb_waterfall(r: list[Gen], sr: int) -> list[Layer]:
    f = _freqs(sr)
    edges = ((30, 200), (200, 900), (900, 3500), (3500, 14000))
    bands = [_norm(_band(f, lo, hi, 2.0)) ** 2 for lo, hi in edges]
    weights = (1.0, 0.9, 0.6, 0.35)
    churn = [_Smooth(r[i], p, 0.55, 1.45) for i, p in enumerate((0.35, 0.6, 1.1, 1.7))]
    surge = _Smooth(r[4], 7.0, 0.8, 1.0)
    jr = r[5]
    mist = _norm(_band(f, 5000, 16000, 2.0))[None, :]

    def shape(_f: F64, t: F64) -> F64:
        p = np.zeros((t.size, f.size))
        for b, w, c in zip(bands, weights, churn, strict=True):
            p += ((w * c.at(t)) ** 2)[:, None] * b[None, :]
        jitter = 0.8 + 0.4 * jr.random((t.size, 1))  # turbulence
        roar = surge.at(t)[:, None] * jitter * np.sqrt(p / len(bands))
        out: F64 = 0.06 * roar + 0.012 * mist
        return out

    src = _SpectralNoise(r[6], sr, shape)
    return [
        lambda _s0: src.block(),
        _bubbles(r[7], sr, lambda _s0: 120.0, 250.0, 1500.0, (0.005, 0.03)),
    ]


def _amb_hill_station(r: list[Gen], sr: int) -> list[Layer]:
    birds = _songbirds(r, 20, 4, sr, (0.03, 0.07), [4500.0, 3500.0], (6.0, 25.0))
    birds.append(_caller(r[26], sfx.cuckoo(r[26]), sr, 0.05, 2500.0, (12.0, 40.0), (1.0, 2.5)))
    temple = _Singer(
        r[27],
        sfx.bell(r[27]),
        sr,
        gain=0.06,
        pan=float(r[27].uniform(-0.7, 0.7)),
        cutoff=3000.0,
        bout=(1, 5),
        gap=(1.5, 3.5),
        rest=(25.0, 70.0),
        first=(2.0, 15.0),
        busy=0.0,  # strikes overlap the previous bell's ring
    )
    return [
        _breeze(r[0], r[1], r[2], sr, 0.03, (400.0, 1800.0), 2.0, 3.0),
        _wet(_chorus(birds, 3.0, sr), "hall", 0.35, sr),
        _wet(_chorus([temple], 6.0, sr), "cathedral", 0.6, sr),
    ]


def _amb_village(r: list[Gen], sr: int) -> list[Layer]:
    near = [
        _caller(r[10], sfx.rooster(r[10]), sr, 0.16, 0.0, (25.0, 80.0), (3.0, 8.0), (1.0, 10.0)),
        _caller(r[11], sfx.rooster(r[11]), sr, 0.05, 2000.0, (30.0, 90.0), first=(8.0, 40.0)),
        _caller(r[12], sfx.koel(r[12]), sr, 0.06, 4000.0, (15.0, 50.0)),
        _caller(r[13], sfx.crow(r[13]), sr, 0.06, 3500.0, (10.0, 40.0), (1.0, 4.0)),
        _caller(r[14], sfx.dove(r[14]), sr, 0.05, 0.0, (10.0, 35.0), (1.0, 3.0)),
    ]
    near += _songbirds(r, 20, 2, sr, (0.04, 0.08), [0.0, 5000.0], (5.0, 20.0))
    cows = [
        _Singer(
            r[30 + i],
            sfx.cowbell(r[30 + i]),
            sr,
            gain=float(r[30 + i].uniform(0.03, 0.06)),
            pan=float(r[30 + i].uniform(-0.8, 0.8)),
            cutoff=5000.0,
            bout=(1, 3),
            gap=(1.0, 4.0),
            rest=(3.0, 15.0),
            first=(0.0, 6.0),
        )
        for i in range(2)
    ]
    return [
        _breeze(r[0], r[1], r[2], sr, 0.02, (300.0, 1300.0), 1.0, 3.0),
        _wet(_chorus(near + cows, 7.0, sr), "hall", 0.15, sr),
    ]


def _amb_stream(r: list[Gen], sr: int) -> list[Layer]:
    flow = _Smooth(r[0], 3.0, 0.8, 1.15)
    br = r[1]
    base = _norm(_band(_freqs(sr), 250, 6000, 1.5))[None, :]

    def shape(f: F64, t: F64) -> F64:
        k = t.size
        fc = np.exp(br.uniform(math.log(450), math.log(3000), size=(k, 4)))
        gain = br.uniform(0.5, 3.0, size=(k, 4))
        res = np.zeros((k, f.size))
        for j in range(4):
            res += gain[:, j : j + 1] * _peak(f[None, :], fc[:, j : j + 1], 8.0)
        out: F64 = 0.03 * flow.at(t)[:, None] * _rownorm(base * (1.0 + res))
        return out

    src = _SpectralNoise(r[2], sr, shape)
    burst = _Smooth(r[3], 0.5, 0.3, 1.8)

    def rate(s0: int) -> float:
        return 75.0 * float(burst.at(np.array([s0 / sr]))[0])

    return [
        lambda _s0: src.block(),
        _bubbles(r[4], sr, rate, 600.0, 3200.0, (0.006, 0.035)),
        _bubbles(r[5], sr, lambda _s0: 1.5, 250.0, 600.0, (0.03, 0.08)),  # bigger "gloops"
    ]


_RECIPES: dict[str, Callable[[list[Gen], int], list[Layer]]] = {
    "ocean": _amb_ocean,
    "beach": _amb_beach,
    "jungle": _amb_jungle,
    "forest": _amb_forest,
    "thunderstorm": _amb_thunderstorm,
    "ice": _amb_ice,
    "waterfall": _amb_waterfall,
    "hill_station": _amb_hill_station,
    "village": _amb_village,
    "stream": _amb_stream,
}


# -- music -------------------------------------------------------------------------------
_MOOD = {
    # bpm, beats per chord, chords (MIDI notes), brightness, reverb room, mix, arp step
    # (beats; 0 = none), bass style, drums, tremolo Hz, output gain (-> about -18 LUFS)
    "calm": (
        66,
        4,
        [[60, 64, 67, 71], [57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 64]],
        0.5,
        "hall",
        0.35,
        0.5,
        "held",
        False,
        0.0,
        0.624,
    ),
    "happy": (
        100,
        4,
        [[55, 59, 62, 69], [50, 57, 62, 66], [52, 59, 64, 67], [48, 55, 60, 64]],
        0.7,
        "hall",
        0.25,
        0.5,
        "pulse",
        False,
        0.0,
        0.608,
    ),
    "sad": (
        60,
        4,
        [[57, 60, 64, 71], [53, 57, 60, 64], [48, 55, 60, 64], [55, 59, 62, 67]],
        0.35,
        "hall",
        0.45,
        1.0,
        "held",
        False,
        0.0,
        0.640,
    ),
    "tense": (
        92,
        4,
        [[50, 53, 57, 62], [51, 55, 58, 62], [50, 53, 57, 60], [49, 52, 56, 61]],
        0.45,
        "hall",
        0.3,
        0.0,
        "ostinato",
        False,
        6.0,
        0.812,
    ),
    "mysterious": (
        54,
        4,
        [[52, 59, 66, 67], [48, 55, 59, 66], [45, 52, 59, 60], [47, 52, 54, 57]],
        0.4,
        "cathedral",
        0.5,
        1.0,
        "held",
        False,
        0.0,
        0.537,
    ),
    "epic": (
        80,
        4,
        [[50, 57, 62, 65], [46, 53, 58, 62], [53, 57, 60, 65], [48, 55, 60, 64]],
        0.75,
        "hall",
        0.35,
        0.5,
        "octave",
        True,
        0.0,
        0.553,
    ),
}


class _Reverb:
    """Stereo convolution with a procedural IR, uniform FFT overlap-add per block."""

    def __init__(self, room: str, sr: int, mix: float) -> None:
        h = ir.generate(room, sr, ("FL", "FR")).astype(np.float64)
        self.mix = mix
        self.nfft = 1 << int(math.ceil(math.log2(_BLOCK + h.shape[0] - 1)))
        self.h = np.fft.rfft(h, n=self.nfft, axis=0)
        self.tail = np.zeros((self.nfft - _BLOCK, 2))

    def process(self, x: F64) -> F64:
        if not x.any():  # silent input (e.g. a resting caller): only the tail rings on
            y = np.zeros((self.nfft, 2))
        else:
            y = np.fft.irfft(np.fft.rfft(x, n=self.nfft, axis=0) * self.h, n=self.nfft, axis=0)
        y[: self.tail.shape[0]] += self.tail
        out = y[:_BLOCK]
        self.tail = y[_BLOCK:].copy()
        result: F64 = (1.0 - self.mix) * x + self.mix * out
        return result


def _music_layers(mood: str, sr: int, seed: int) -> tuple[list[Layer], _Reverb]:
    if mood not in _MOOD:
        raise ProceduralError(f"Unknown mood {mood!r}.", fix=f"Choose one of: {', '.join(MOODS)}.")
    bpm, beats, chords, bright, room, mix, arp_step, bass, drums, trem, _g = _MOOD[mood]
    r = _rngs(seed, "music:" + mood, 6)
    beat = 60.0 / bpm
    seg = beats * beat
    release = min(2.0, seg * 0.6)
    attack = min(1.6, seg * 0.4)
    rolloff = 2.2 - 1.6 * bright
    n_harm = 8
    detune = [r[0].uniform(-7, 7, size=2) for _ in range(64)]  # cents per voice (L, R)
    phases = r[0].uniform(0, 2 * np.pi, size=(64, n_harm, 2))
    order = np.arange(len(chords))
    # vary the progression order a little by seed: rotate the start chord
    start = int(r[1].integers(0, len(chords)))
    order = np.roll(order, -start)
    breathe = _Smooth(r[2], 6.0, 0.85, 1.0)

    def chord_at(k: int) -> list[int]:
        return list(chords[int(order[k % len(order)])])

    def env_for(k: int, t: F64) -> F64:
        local = t - k * seg
        a = np.clip(local / attack, 0.0, 1.0)
        a = a * a * (3 - 2 * a)
        rel = np.clip((local - seg) / release, 0.0, 1.0)
        out: F64 = a * (1.0 - rel * rel * (3 - 2 * rel)) * (local >= 0)
        return out

    def pad(s0: int) -> F64:
        t = (s0 + np.arange(_BLOCK)) / sr
        out = np.zeros((_BLOCK, 2))
        k_lo = max(0, int(math.floor((t[0] - seg - release) / seg)))
        k_hi = int(math.floor(t[-1] / seg))
        for k in range(k_lo, k_hi + 1):
            env = env_for(k, t)
            if not np.any(env):
                continue
            notes = chord_at(k)
            for ni, m in enumerate(notes):
                f0 = _midi_hz(m)
                v = (k * 7 + ni) % 64
                for ch in range(2):
                    f = f0 * 2 ** (detune[v][ch] / 1200.0)
                    # partials by the Chebyshev recurrence sin((h+1)p) = 2cos(p)sin(hp) -
                    # sin((h-1)p): two transcendental calls per voice instead of one per partial
                    ph = 2 * np.pi * f * t + phases[v, 0, ch]
                    two_c = 2.0 * np.cos(ph)
                    s_prev, s_cur = np.zeros(_BLOCK), np.sin(ph)
                    wave_ = s_cur.copy()
                    for h in range(2, n_harm + 1):
                        if f * h > min(7000.0, sr * 0.45):
                            break
                        s_prev, s_cur = s_cur, two_c * s_cur - s_prev
                        wave_ += h**-rolloff * s_cur
                    out[:, ch] += wave_ * env / math.sqrt(len(notes))
            # bass
            root = min(notes) - 12
            fb = _midi_hz(root)
            if bass == "held":
                b = 0.3 * (np.sin(2 * np.pi * fb * t) + 0.25 * np.sin(4 * np.pi * fb * t)) * env
                out += b[:, None]
            elif bass == "octave":
                b = 0.3 * (np.sin(2 * np.pi * fb / 2 * t) + 0.5 * np.sin(2 * np.pi * fb * t))
                out += (b * env)[:, None]
            else:
                step = beat if bass == "pulse" else beat / 2
                local = np.mod(t - k * seg, step)
                pl = np.exp(-local / (0.18 if bass == "pulse" else 0.09)) * np.minimum(
                    1.0, local / 0.004
                )
                inside = (t >= k * seg) & (t < (k + 1) * seg)
                b = 0.35 * np.sin(2 * np.pi * fb * t) * pl * inside
                out += b[:, None]
        if trem:
            out *= (0.8 + 0.2 * np.sin(2 * np.pi * trem * t))[:, None]
        out *= (0.25 * breathe.at(t))[:, None]
        result: F64 = out
        return result

    layers: list[Layer] = [pad]
    if arp_step:
        ev = _Events(_BLOCK, int(2.0 * sr))
        arng = r[3]
        step_s = arp_step * beat
        pattern = [0, 1, 2, 3, 2, 1] if mood == "happy" else [0, 1, 2, 3]

        def arp(s0: int) -> F64:
            t0, t1 = s0 / sr, (s0 + _BLOCK) / sr
            i0 = int(math.ceil(t0 / step_s - 1e-9))
            i = i0
            while i * step_s < t1:
                ts = i * step_s
                k = int(ts // seg)
                notes = sorted(chord_at(k))
                if mood == "mysterious" and arng.random() < 0.45:
                    i += 1
                    continue
                m = notes[pattern[i % len(pattern)] % len(notes)] + 12
                n = int(1.6 * sr)
                tt = np.arange(n) / sr
                f = _midi_hz(m)
                x = (
                    (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt))
                    * np.exp(-tt / (0.45 if bright < 0.5 else 0.3))
                    * np.minimum(1.0, tt / 0.005)
                )
                pos = int(round(ts * sr)) - s0
                if 0 <= pos < _BLOCK:
                    ev.add(pos, 0.11 * arng.uniform(0.8, 1.0) * x, 0.35 * (1 if i % 2 else -1))
                i += 1
            return ev.take()

        layers.append(arp)
    if drums:
        dev = _Events(_BLOCK, int(1.2 * sr))
        drng = r[4]

        def drum(s0: int) -> F64:
            t0, t1 = s0 / sr, (s0 + _BLOCK) / sr
            i = int(math.ceil(t0 / beat - 1e-9))
            while i * beat < t1:
                hit = i % 4 in (0, 2) or (i % 8 == 7 and drng.random() < 0.6)
                if hit:
                    n = int(1.0 * sr)
                    tt = np.arange(n) / sr
                    f = 45.0 + 70.0 * np.exp(-tt / 0.04)
                    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt / 0.32)
                    pos = int(round(i * beat * sr)) - s0
                    if 0 <= pos < _BLOCK:
                        dev.add(pos, 0.5 * x, 0.0)
                i += 1
            return dev.take()

        layers.append(drum)
    return layers, _Reverb(room, sr, mix)


# -- rendering ---------------------------------------------------------------------------
class _Stream:
    """Pull any number of samples from block layers (+ optional reverb)."""

    def __init__(self, layers: list[Layer], gain: float, reverb: _Reverb | None) -> None:
        self.layers, self.gain, self.reverb = layers, gain, reverb
        self.s0 = 0
        self.left = np.zeros((0, 2))

    def _block(self) -> F64:
        x = np.zeros((_BLOCK, 2))
        for layer in self.layers:
            x += layer(self.s0)
        if self.reverb is not None:
            x = self.reverb.process(x)
        self.s0 += _BLOCK
        out: F64 = x * self.gain
        return out

    def pull(self, n: int) -> F64:
        parts = [self.left]
        have = self.left.shape[0]
        while have < n:
            b = self._block()
            parts.append(b)
            have += b.shape[0]
        allx = np.concatenate(parts)
        self.left = allx[n:]
        out: F64 = allx[:n]
        return out


def _to_pcm24(x: F64) -> bytes:
    q = np.round(np.clip(x, -1.0, 1.0) * 8388607.0).astype("<i4")
    return q.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def _check(seconds: float, out: Path, sample_rate: int) -> int:
    if not (0 < seconds <= MAX_SECONDS) or not math.isfinite(seconds):
        raise ProceduralError(
            f"Duration {seconds:g} s is out of range.",
            fix=f"Use a duration above 0 and up to {MAX_SECONDS / 3600:g} hours.",
        )
    if not 8000 <= sample_rate <= 192_000:
        raise ProceduralError(
            f"Sample rate {sample_rate} Hz is not supported.", fix="Use 44100 or 48000."
        )
    if out.suffix.lower() != ".wav":
        raise ProceduralError(f"Cannot write {out.name}: only .wav is supported.", fix="Use .wav.")
    total = int(round(seconds * sample_rate))
    if total < 1:
        raise ProceduralError("Duration is shorter than one sample.")
    return total


def _render(stream: _Stream, total: int, out: Path, sample_rate: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + f".{os.getpid()}.part")
    xf = min(int(_XFADE_S * sample_rate), total // 4)
    try:
        with wave.open(str(part), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(3)
            w.setframerate(sample_rate)
            # output y[t] = s[t + xf] for t < total - xf, then s's extra tail crossfaded
            # into its head s[0:xf]: the last sample of y leads straight into y[0] = s[xf]
            head = stream.pull(xf)
            body = total - xf
            while body > 0:
                n = min(body, _BLOCK)
                w.writeframes(_to_pcm24(_soft_clip(stream.pull(n))))
                body -= n
            if xf:
                tail = stream.pull(xf)
                ph = (np.arange(xf) + 0.5) / xf * (np.pi / 2)
                mixed = tail * np.cos(ph)[:, None] + head * np.sin(ph)[:, None]
                w.writeframes(_to_pcm24(_soft_clip(mixed)))
        part.replace(out)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise ProceduralError(
            f"Could not write {out}: {exc.strerror or exc}.",
            fix="Check free disk space and folder permissions.",
        ) from exc
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return out


# Per-kind output gain, calibrated so the files land at typical bed levels (measured with
# ebur128, see docs/PROGRESS.md); the project's loudness pass sets the final level anyway.
# Targets (LUFS): rain -24, wind/fire -26, river -24, birds -28, night -32, room -42 (room
# tone is meant to be barely audible); ocean/thunderstorm/waterfall -24, beach -25,
# jungle/ice/stream -26, forest/village -27, hill_station -28 (sparse kinds measured as the
# mean of two seeds x 180 s, since a short stretch may hold no rooster or thunder at all).
_AMB_GAIN_DB = {
    "rain": -5.8,
    "wind": -3.9,
    "fire": -0.8,
    "river": -4.1,
    "birds": -4.2,
    "night": 2.4,
    "room": -6.4,
    "ocean": 3.4,
    "beach": 4.0,
    "jungle": 0.3,
    "forest": 1.8,
    "thunderstorm": -11.0,
    "ice": 4.4,
    "waterfall": -1.7,
    "hill_station": 4.8,
    "village": 6.7,
    "stream": -0.5,
}


def make_ambience(
    kind: str, seconds: float, out: Path, *, seed: int = 1, sample_rate: int = 48000
) -> Path:
    """Write a loopable stereo 24-bit WAV of ambience `kind` (see AMBIENCES)."""
    if kind not in AMBIENCES:
        raise ProceduralError(
            f"Unknown ambience {kind!r}.", fix=f"Choose one of: {', '.join(AMBIENCES)}."
        )
    total = _check(seconds, out, sample_rate)
    gain = 10 ** (_AMB_GAIN_DB[kind] / 20)
    stream = _Stream(_ambience_layers(kind, sample_rate, seed), gain, None)
    return _render(stream, total, out, sample_rate)


def make_music(
    mood: str, seconds: float, out: Path, *, seed: int = 1, sample_rate: int = 48000
) -> Path:
    """Write a loopable stereo 24-bit WAV of ambient/pad music for `mood` (see MOODS)."""
    if mood not in MOODS:
        raise ProceduralError(f"Unknown mood {mood!r}.", fix=f"Choose one of: {', '.join(MOODS)}.")
    total = _check(seconds, out, sample_rate)
    layers, reverb = _music_layers(mood, sample_rate, seed)
    stream = _Stream(layers, float(_MOOD[mood][-1]), reverb)
    return _render(stream, total, out, sample_rate)
