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
from pathlib import Path

import numpy as np
import numpy.typing as npt

from storyreel.audio import ir
from storyreel.errors import StoryReelError

F64 = npt.NDArray[np.float64]
Gen = np.random.Generator

AMBIENCES = ("rain", "wind", "fire", "river", "birds", "night", "room")
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
        gl, gr = _pan(pan)
        self.buf[pos : pos + n, 0] += mono[:n] * gl
        self.buf[pos : pos + n, 1] += mono[:n] * gr

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
        self.freqs: F64 = np.fft.rfftfreq(_N, 1.0 / sr)
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
        mag = self.shape(self.freqs, t)  # (k, bins)
        bins = self.freqs.size
        re = self.rng.standard_normal((k, bins, 2))
        im = self.rng.standard_normal((k, bins, 2))
        spec = (re + 1j * im) * (mag[:, :, None] * self.scale)
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
    r = _rngs(seed, "amb:" + kind, 8)
    layers: list[Layer] = []

    def noise(rng: Gen, shape: Callable[[F64, F64], F64]) -> Layer:
        src = _SpectralNoise(rng, sr, shape)
        return lambda _s0: src.block()

    if kind == "rain":
        inten = _Smooth(r[0], 3.0, 0.75, 1.1)

        def rain_shape(f: F64, t: F64) -> F64:
            base = _norm(_band(f, 500, 9000, 2.0) + 0.35 * _band(f, 80, 450, 2.0))
            return 0.07 * inten.at(t)[:, None] * base[None, :]

        layers.append(noise(r[1], rain_shape))
        drops = _Events(_BLOCK, int(0.05 * sr))
        drng, prng = r[2], r[3]

        def drop_layer(s0: int) -> F64:
            for pos in _poisson_positions(drng, 45.0, sr):
                amp = math.exp(drng.uniform(math.log(0.015), math.log(0.12)))
                tone = _decay_tone(sr, drng.uniform(2200, 7500), drng.uniform(0.0012, 0.004), 0.03)
                drops.add(int(pos), amp * tone, drng.uniform(-0.9, 0.9))
            for pos in _poisson_positions(prng, 4.0, sr):  # "plip": rising bubble on water
                amp = prng.uniform(0.04, 0.14)
                tone = _decay_tone(sr, prng.uniform(900, 2200), 0.007, 0.04, rise=0.6)
                drops.add(int(pos), amp * tone, prng.uniform(-0.8, 0.8))
            return drops.take()

        layers.append(drop_layer)
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
        ev = _Events(_BLOCK, int(1.6 * sr))
        cr = r[2]
        species = [_bird_species(r[3 + i]) for i in range(3)]

        def call_layer(_s0: int) -> F64:
            for pos in _poisson_positions(cr, 0.7, sr):
                sp = species[int(cr.integers(0, len(species)))]
                call = sp(cr, sr)
                ev.add(int(pos), cr.uniform(0.04, 0.2) * call, cr.uniform(-0.9, 0.9))
            return ev.take()

        layers.append(call_layer)
    elif kind == "night":
        base = _norm(_band(_freqs(sr), 40, 600, 1.5))
        drift = _Smooth(r[0], 6.0, 0.6, 1.0)

        def night_air(_f: F64, t: F64) -> F64:
            out: F64 = 0.012 * drift.at(t)[:, None] * base[None, :]
            return out

        layers.append(noise(r[1], night_air))
        crickets = [_cricket(r[2 + i], sr, i) for i in range(4)]
        layers.append(lambda s0: np.sum([c(s0) for c in crickets], axis=0))
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
    else:
        raise ProceduralError(
            f"Unknown ambience {kind!r}.", fix=f"Choose one of: {', '.join(AMBIENCES)}."
        )
    return layers


def _freqs(sr: int) -> F64:
    out: F64 = np.fft.rfftfreq(_N, 1.0 / sr)
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
_AMB_GAIN_DB = {  # measured -> target LUFS: rain -24, wind/fire -26, river -24, birds -28,
    "rain": -5.8,  # night -32, room -42 (room tone is meant to be barely audible)
    "wind": -3.9,
    "fire": -0.8,
    "river": -4.1,
    "birds": -7.6,
    "night": 1.8,
    "room": -6.4,
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
