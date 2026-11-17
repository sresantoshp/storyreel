"""Short synthesised sound events for the procedural ambiences (numpy only, no samples).

Every function returns a mono float64 signal that starts and ends at zero, so it can be
dropped anywhere into a stream without a click; most are normalised to a peak of 1 and
the caller sets the level. The `make_*` factories draw an animal's or object's constants
once (pitch, rhythm, timbre), so one "bird" or "bell" stays recognisable while each call
still varies a little. These are stylised imitations, not recordings.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
Gen = np.random.Generator
Call = Callable[[Gen, int], F64]  # (rng, sample rate) -> mono signal

SONGBIRDS = ("tweet", "trill", "whistle", "warble", "chip", "sweep", "chatter")


# -- helpers ---------------------------------------------------------------------------
def env_ar(n: int, sr: int, attack: float, release: float) -> F64:
    """Raised-cosine fade-in/out envelope; first and last samples are exactly 0."""
    e = np.ones(n)
    if n < 4:
        return np.zeros(n)
    a = min(n // 2, max(1, int(attack * sr)))
    r = min(n - a, max(1, int(release * sr)))
    e[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    e[n - r :] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(r) + 1) / r)
    return e


def tone(f: F64, sr: int, harm: Sequence[float], phase0: float = 0.0) -> F64:
    """Harmonic oscillator following the per-sample frequency `f` (Hz); partials that
    would pass 0.45 * sr are left out (no aliasing)."""
    ph = 2.0 * np.pi * np.cumsum(f) / sr + phase0
    fmax = float(np.max(f)) if f.size else 0.0
    out = np.zeros(f.size)
    for h, a in enumerate(harm, start=1):
        if fmax * h >= 0.45 * sr:
            break
        if a:
            out += a * np.sin(h * ph)
    return out


def _filt(x: F64, sr: int, gain: Callable[[F64], F64]) -> F64:
    """Zero-phase FFT filter (padded, so no wrap-around), then a 2 ms edge fade."""
    if x.size < 8:
        return np.zeros(x.size)
    n = 1 << int(math.ceil(math.log2(x.size + 4096)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    y = np.fft.irfft(np.fft.rfft(x, n) * gain(f), n)[: x.size]
    out: F64 = y * env_ar(x.size, sr, 0.002, 0.002)
    return out


def lowpass(x: F64, sr: int, fc: float, order: float = 2.0) -> F64:
    return _filt(x, sr, lambda f: 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order)))


def bandpass(x: F64, sr: int, lo: float, hi: float, order: float = 2.0) -> F64:
    def g(f: F64) -> F64:
        fs = np.maximum(f, 1.0)
        out: F64 = 1.0 / np.sqrt(
            (1.0 + (lo / fs) ** (2 * order)) * (1.0 + (fs / hi) ** (2 * order))
        )
        return out

    return _filt(x, sr, g)


def formants(x: F64, sr: int, peaks: Sequence[tuple[float, float]], floor: float) -> F64:
    """Shape a buzzy source with resonances (centre Hz, bandwidth Hz) over a floor."""

    def g(f: F64) -> F64:
        out = np.full(f.size, floor)
        for fc, bw in peaks:
            out += 1.0 / (1.0 + ((f - fc) / bw) ** 2)
        return out

    return _filt(x, sr, g)


def peak_norm(x: F64) -> F64:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    out: F64 = x / m if m > 0 else x
    return out


def smooth_noise(rng: Gen, n: int, sr: int, rate: float) -> F64:
    """Slow random wobble (about `rate` Hz), unit RMS: for pitch jitter / roughness."""
    k = max(2, int(n / sr * rate) + 2)
    pts = rng.standard_normal(k)
    x = np.linspace(0.0, k - 1.0, n)
    i = np.minimum(x.astype(np.int64), k - 2)
    w = 0.5 - 0.5 * np.cos(np.pi * (x - i))
    out: F64 = pts[i] * (1 - w) + pts[i + 1] * w
    return out


def _cat(parts: list[F64]) -> F64:
    out: F64 = np.concatenate(parts) if parts else np.zeros(1)
    return out


def _sil(sec: float, sr: int) -> F64:
    return np.zeros(max(0, int(sec * sr)))


def _glide(a: float, b: float, n: int, curve: float = 1.0) -> F64:
    x = np.linspace(0.0, 1.0, n) ** curve
    out: F64 = a + (b - a) * x
    return out


# -- songbirds -------------------------------------------------------------------------
def songbird(rng: Gen, style: str | None = None, low: float = 2200.0) -> Call:
    """A small songbird: one of SONGBIRDS with its own pitch range and rhythm."""
    st = style or SONGBIRDS[int(rng.integers(0, len(SONGBIRDS)))]
    lo = rng.uniform(max(low, 2400.0), 4000.0)
    hi = lo * rng.uniform(1.3, 1.9)
    syl, gap = rng.uniform(0.04, 0.09), rng.uniform(0.03, 0.08)
    harm = (1.0, rng.uniform(0.08, 0.3), rng.uniform(0.0, 0.08))
    scale = lo * 2.0 ** (np.sort(rng.choice(np.arange(13), size=6, replace=False)) / 12.0)
    motif = rng.integers(0, 6, size=int(rng.integers(4, 8)))  # warbler's own phrase
    chip_iv = rng.uniform(0.07, 0.16)

    def note(f: F64, sr: int, att: float = 0.006, rel: float = 0.012) -> F64:
        return tone(f, sr, harm) * env_ar(f.size, sr, att, rel)

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        k = r.uniform(0.96, 1.04)
        if st == "tweet":  # descending syllables, the last often shorter
            for _ in range(int(r.integers(3, 8))):
                n = int(syl * sr * r.uniform(0.85, 1.15))
                parts += [note(_glide(hi * k, lo * k, n, 0.7), sr), _sil(gap, sr)]
        elif st == "trill":  # fast FM, rising or falling overall
            n = int(r.uniform(0.4, 1.1) * sr)
            t = np.arange(n) / sr
            drift = _glide(1.0, r.uniform(0.85, 1.15), n)
            f = ((lo + hi) / 2 + (hi - lo) / 3 * np.sin(2 * np.pi * r.uniform(18, 32) * t)) * drift
            parts.append(note(f * k, sr, 0.03, 0.05))
        elif st == "whistle":  # two or three clear glides
            for a, b in ((lo, lo * 1.12), (lo * 1.3, lo * 1.22), (lo * 1.1, lo * 0.95))[
                : int(r.integers(2, 4))
            ]:
                n = int(r.uniform(0.14, 0.26) * sr)
                parts += [note(_glide(a * k, b * k, n), sr, 0.02, 0.04), _sil(0.06, sr)]
        elif st == "warble":  # a remembered melodic phrase, slightly varied each time
            idx = motif.copy()
            if r.random() < 0.5:
                idx[int(r.integers(0, idx.size))] = int(r.integers(0, 6))
            for j in idx:
                f0 = float(scale[int(j)]) * k
                n = int(r.uniform(0.06, 0.15) * sr)
                bend = 2.0 ** (r.uniform(-1.0, 1.0) / 12.0)
                parts += [note(_glide(f0, f0 * bend, n), sr), _sil(r.uniform(0.01, 0.04), sr)]
        elif st == "chip":  # dry repeated chips, sparrow-like
            for i in range(int(r.integers(2, 9))):
                n = int(0.022 * sr)
                parts += [
                    note(_glide(hi * 1.2 * k, lo * k, n, 0.5), sr, 0.002, 0.008),
                    _sil(chip_iv * (1.0 - 0.03 * i) * r.uniform(0.85, 1.15), sr),
                ]
        elif st == "sweep":  # rising "wheet"
            for _ in range(int(r.integers(1, 4))):
                n = int(r.uniform(0.18, 0.3) * sr)
                parts += [note(_glide(lo * k, hi * k, n, 1.6), sr, 0.02, 0.03), _sil(0.15, sr)]
        else:  # chatter: quick buzzy notes of random pitch
            buzz = (1.0, 0.5, 0.3, 0.2)
            for _ in range(int(r.integers(6, 18))):
                n = int(r.uniform(0.025, 0.05) * sr)
                f = np.full(n, r.uniform(lo, hi) * k)
                t = np.arange(n) / sr
                x = tone(f, sr, buzz) * (0.6 + 0.4 * np.sin(2 * np.pi * 60 * t))
                parts += [x * env_ar(n, sr, 0.003, 0.01), _sil(r.uniform(0.01, 0.03), sr)]
        return peak_norm(_cat(parts))

    return call


# -- larger birds and animals ------------------------------------------------------------
def gull(rng: Gen) -> Call:
    """Seagull: one to three nasal "kee-ow" notes, sometimes a "ha-ha-ha" laugh."""
    base = rng.uniform(800.0, 1200.0)
    harm = (1.0, 0.9, 0.75, 0.55, 0.4, 0.28, 0.18, 0.12, 0.08)

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        for _ in range(int(r.integers(1, 4))):
            d = r.uniform(0.3, 0.55)
            n = int(d * sr)
            t = np.arange(n) / sr
            x = t / d
            top = r.uniform(1.4, 1.65)
            c = np.where(
                x < 0.18,
                1.0 + (top - 1.0) * np.sin(np.pi / 2 * x / 0.18),
                top - (top - 0.72) * np.clip((x - 0.18) / 0.82, 0, 1) ** 1.3,
            )
            vib = 1.0 + 0.018 * np.sin(2 * np.pi * r.uniform(18, 28) * t)
            f = base * r.uniform(0.94, 1.06) * c * vib
            y = tone(f, sr, harm) * env_ar(n, sr, 0.02, 0.12)
            parts += [y, _sil(r.uniform(0.12, 0.35), sr)]
        if r.random() < 0.35:  # laugh
            for i in range(int(r.integers(3, 7))):
                n = int(r.uniform(0.08, 0.12) * sr)
                f0 = base * 1.25 * (1.0 - 0.02 * i)
                f = _glide(f0 * 1.1, f0 * 0.9, n)
                parts += [0.8 * tone(f, sr, harm) * env_ar(n, sr, 0.01, 0.04), _sil(0.07, sr)]
        return peak_norm(_cat(parts))

    return call


def rooster(rng: Gen) -> Call:
    """Cock-a-doodle-doo: four buzzy syllables, the last one long and falling."""
    base = rng.uniform(430.0, 600.0)
    long_s = rng.uniform(0.8, 1.3)
    syl = [(0.11, 0.92, 1.0), (0.09, 1.05, 1.08), (0.15, 1.18, 1.25), (long_s, 1.3, 0.95)]
    harm = tuple(h**-0.75 for h in range(1, 26))
    f1, f2 = rng.uniform(1100, 1500), rng.uniform(2400, 3100)

    def call(r: Gen, sr: int) -> F64:
        k = base * r.uniform(0.96, 1.04)
        f_parts, e_parts = [], []
        for i, (d, a, b) in enumerate(syl):
            n = int(d * r.uniform(0.9, 1.1) * sr)
            f_parts.append(_glide(a * k, b * k, n, 0.6 if i == 3 else 1.0))
            # syllables run together: the envelope dips between them instead of stopping
            e = 0.3 + 0.7 * env_ar(n, sr, 0.025, 0.03)
            if i == 3:
                e *= np.linspace(1.0, 0.75, n)
            e_parts.append(e)
        f = np.concatenate(f_parts)
        n = f.size
        f = f * (1.0 + 0.02 * smooth_noise(r, n, sr, 25.0))
        t = np.arange(n) / sr
        y = tone(f, sr, harm) * (1.0 + 0.3 * np.sin(2 * np.pi * r.uniform(35, 55) * t))
        y = formants(y, sr, ((f1, 250.0), (f2, 400.0)), 0.12)
        y *= np.concatenate(e_parts) * env_ar(n, sr, 0.02, 0.15)
        return peak_norm(y)

    return call


def crow(rng: Gen) -> Call:
    """Harsh "caw", one to four times."""
    base = rng.uniform(450.0, 650.0)
    harm = tuple(h**-0.4 for h in range(1, 15))
    form = rng.uniform(1200, 1800)

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        for _ in range(int(r.integers(1, 5))):
            n = int(r.uniform(0.22, 0.4) * sr)
            f = _glide(base * 1.05, base * 0.88, n) * (1 + 0.03 * smooth_noise(r, n, sr, 40))
            y = tone(f, sr, harm) + 0.35 * r.standard_normal(n)
            y = formants(y, sr, ((form, 500.0),), 0.1) * env_ar(n, sr, 0.03, 0.08)
            parts += [y, _sil(r.uniform(0.25, 0.45), sr)]
        return peak_norm(_cat(parts))

    return call


def dove(rng: Gen) -> Call:
    """Soft low cooing phrase, e.g. coo-COO-coo-coo."""
    base = rng.uniform(430.0, 620.0)
    pattern = [(0.3, 1.0, 0.7), (0.55, 1.06, 1.0), (0.35, 0.96, 0.7), (0.35, 0.94, 0.6)]
    pattern = pattern[: int(rng.integers(3, 5))]

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        for d, p, a in pattern:
            n = int(d * r.uniform(0.9, 1.1) * sr)
            t = np.arange(n) / sr
            f = _glide(base * p * 1.02, base * p * 0.96, n) * (1 + 0.01 * np.sin(2 * np.pi * 6 * t))
            parts += [a * tone(f, sr, (1.0, 0.2, 0.05)) * env_ar(n, sr, 0.08, 0.15), _sil(0.12, sr)]
        return peak_norm(_cat(parts))

    return call


def cuckoo(rng: Gen) -> Call:
    """The two-note "cu-ckoo", a falling minor third."""
    f1 = rng.uniform(640.0, 780.0)

    def call(r: Gen, sr: int) -> F64:
        k = r.uniform(0.98, 1.02)
        a = int(0.17 * sr)
        b = int(0.3 * sr)
        x1 = tone(_glide(f1 * k, f1 * k * 0.99, a), sr, (1.0, 0.08)) * env_ar(a, sr, 0.03, 0.06)
        f2 = f1 * k * 2 ** (-3.5 / 12)
        x2 = tone(_glide(f2, f2 * 0.98, b), sr, (1.0, 0.08)) * env_ar(b, sr, 0.03, 0.12)
        return peak_norm(_cat([x1, _sil(0.12, sr), 0.85 * x2]))

    return call


def koel(rng: Gen) -> Call:
    """Asian koel: "ku-OO" repeated, climbing in pitch and loudness."""
    base = rng.uniform(720.0, 900.0)

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        reps = int(r.integers(3, 7))
        for i in range(reps):
            b = base * (1.0 + r.uniform(0.03, 0.06)) ** i
            a1, a2 = int(0.11 * sr), int(r.uniform(0.3, 0.4) * sr)
            x1 = tone(_glide(b, b * 1.05, a1), sr, (1.0, 0.2, 0.06)) * env_ar(a1, sr, 0.02, 0.03)
            x2 = tone(_glide(b * 1.12, b * 1.42, a2, 0.7), sr, (1.0, 0.2, 0.06))
            x2 *= env_ar(a2, sr, 0.03, 0.08)
            lvl = 0.55 + 0.45 * i / max(1, reps - 1)
            parts += [lvl * x1, _sil(0.03, sr), lvl * x2, _sil(r.uniform(0.35, 0.55), sr)]
        return peak_norm(_cat(parts))

    return call


def hoot(rng: Gen) -> Call:
    """Owl: soft low hoots in a fixed rhythm (hoo ... hoo-hoo)."""
    base = rng.uniform(300.0, 430.0)
    rhythm = [(0.4, 0.55), (0.18, 0.12), (0.45, 0.0)]
    if rng.random() < 0.5:
        rhythm = [(0.5, 0.0)]

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        for d, g in rhythm:
            n = int(d * r.uniform(0.9, 1.1) * sr)
            f = _glide(base * 1.02, base * 0.95, n)
            parts += [tone(f, sr, (1.0, 0.12, 0.03)) * env_ar(n, sr, 0.07, 0.15), _sil(g, sr)]
        return peak_norm(_cat(parts))

    return call


def whoop(rng: Gen) -> Call:
    """Distant tropical whooping (gibbon/hornbill-like): rising glides that speed up."""
    base = rng.uniform(350.0, 500.0)

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        reps = int(r.integers(4, 10))
        for i in range(reps):
            b = base * 1.04**i
            n = int(max(0.18, 0.5 - 0.035 * i) * sr)
            f = _glide(b, b * r.uniform(1.8, 2.2), n, 1.5)
            x = tone(f, sr, (1.0, 0.35, 0.12, 0.05)) * env_ar(n, sr, 0.04, 0.07)
            parts += [(0.6 + 0.4 * i / reps) * x, _sil(max(0.06, 0.25 - 0.02 * i), sr)]
        return peak_norm(_cat(parts))

    return call


def parrot(rng: Gen) -> Call:
    """Raucous squawks."""
    base = rng.uniform(1200.0, 2000.0)
    harm = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2)

    def call(r: Gen, sr: int) -> F64:
        parts: list[F64] = []
        for _ in range(int(r.integers(1, 4))):
            n = int(r.uniform(0.12, 0.3) * sr)
            t = np.arange(n) / sr
            f = base * r.uniform(0.9, 1.1) * (1 + 0.08 * np.sin(2 * np.pi * 70 * t))
            f = f * (1 + 0.05 * smooth_noise(r, n, sr, 80))
            y = tone(f, sr, harm) + 0.5 * bandpass(r.standard_normal(n), sr, 1000, 5000)
            parts += [y * env_ar(n, sr, 0.01, 0.05), _sil(r.uniform(0.1, 0.25), sr)]
        return peak_norm(_cat(parts))

    return call


# -- percussive / objects --------------------------------------------------------------
def woodpecker(rng: Gen) -> Call:
    """A drum roll of 12-28 knocks (or, sometimes, a few slow feeding taps)."""
    rate = rng.uniform(15.0, 22.0)
    fr, fl = rng.uniform(900, 1500), rng.uniform(170, 260)

    def knock(sr: int, r: Gen) -> F64:
        n = int(0.03 * sr)
        t = np.arange(n) / sr
        k = r.uniform(0.95, 1.05)
        x = np.sin(2 * np.pi * fr * k * t) * np.exp(-t / 0.005)
        x += 0.7 * np.sin(2 * np.pi * fl * t) * np.exp(-t / 0.011)
        x[: int(0.001 * sr)] += 0.4 * r.standard_normal(int(0.001 * sr))
        out: F64 = x * env_ar(n, sr, 0.0004, 0.004)
        return out

    def call(r: Gen, sr: int) -> F64:
        if r.random() < 0.3:
            times = np.cumsum(r.uniform(0.25, 0.5, size=int(r.integers(2, 6))))
            amps = r.uniform(0.5, 1.0, size=times.size)
        else:
            m = int(r.integers(12, 29))
            iv = 1.0 / rate * np.linspace(1.0, r.uniform(0.85, 1.15), m)
            times = np.cumsum(iv)
            amps = np.exp(-np.arange(m) / m * 1.2)
        y = np.zeros(int((times[-1] + 0.05) * sr))
        for t0, a in zip(times, amps, strict=True):
            kn = knock(sr, r)
            p = int(t0 * sr)
            y[p : p + kn.size] += a * kn[: y.size - p]
        return peak_norm(y)

    return call


_CHURCH = (0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 3.011, 4.166, 5.433)
_CHURCH_A = (0.35, 1.0, 0.55, 0.35, 0.6, 0.3, 0.22, 0.15, 0.08)


def bell(rng: Gen, f0: float | None = None, decay: float | None = None) -> Call:
    """A struck bell with inharmonic (church/temple-bell) partials and slow beating."""
    fb = f0 if f0 is not None else rng.uniform(320.0, 700.0)
    tau = decay if decay is not None else rng.uniform(1.4, 2.4)
    ratios = np.array(_CHURCH) * (1 + rng.uniform(-0.01, 0.01, size=len(_CHURCH)))

    def call(r: Gen, sr: int) -> F64:
        n = int(min(6.0, tau * 3.0) * sr)
        t = np.arange(n) / sr
        y = np.zeros(n)
        for rt, a in zip(ratios, _CHURCH_A, strict=True):
            f = fb * float(rt)
            if f > 0.45 * sr:
                break
            d = tau * (1.0 / float(rt)) ** 0.7
            beat = 1.0 + 0.25 * np.sin(2 * np.pi * r.uniform(0.3, 1.2) * t + r.uniform(0, 6.3))
            y += a * np.sin(2 * np.pi * f * t + r.uniform(0, 6.3)) * np.exp(-t / d) * beat
        m = int(0.003 * sr)
        y[:m] += 0.3 * r.standard_normal(m) * np.linspace(1, 0, m)
        return peak_norm(y * env_ar(n, sr, 0.001, 0.3))

    return call


def cowbell(rng: Gen) -> Call:
    """A grazing cow's bell: a few irregular hollow clinks, sometimes double-struck."""
    f0 = rng.uniform(420.0, 800.0)
    ratios = np.array([1.0, 1.47, 2.09, 2.56, 3.53]) * (1 + rng.uniform(-0.03, 0.03, size=5))
    amps = np.array([1.0, 0.7, 0.5, 0.3, 0.2])
    taus = np.array([0.35, 0.25, 0.18, 0.12, 0.08]) * rng.uniform(0.7, 1.3)

    def clink(r: Gen, sr: int) -> F64:
        n = int(1.2 * sr)
        t = np.arange(n) / sr
        y = np.zeros(n)
        for rt, a, d in zip(ratios, amps, taus, strict=True):
            y += a * np.sin(2 * np.pi * f0 * float(rt) * t + r.uniform(0, 6.3)) * np.exp(-t / d)
        out: F64 = y * env_ar(n, sr, 0.0008, 0.2)
        return out

    def call(r: Gen, sr: int) -> F64:
        hits = int(r.integers(1, 6))
        times = np.concatenate([[0.0], np.cumsum(r.uniform(0.15, 0.6, size=hits - 1))])
        y = np.zeros(int((times[-1] + 1.3) * sr))
        for t0 in times:
            c = clink(r, sr)
            p = int(t0 * sr)
            y[p : p + c.size] += r.uniform(0.3, 1.0) * c
            if r.random() < 0.35:  # clapper bounce
                q = p + int(r.uniform(0.025, 0.05) * sr)
                y[q : q + c.size] += 0.35 * c[: y.size - q]
        return peak_norm(y)

    return call


def drip(r: Gen, sr: int, f0: float) -> F64:
    """A water drop falling into a pool: tiny click then a rising "plip"."""
    n = int(0.08 * sr)
    t = np.arange(n) / sr
    f = f0 * (1.0 + 0.7 * np.minimum(1.0, t / r.uniform(0.02, 0.04)))
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / r.uniform(0.008, 0.02))
    m = int(0.0008 * sr)
    y[:m] += 0.3 * r.standard_normal(m)
    out: F64 = y * env_ar(n, sr, 0.0005, 0.01)
    return out


def thunder_crack(r: Gen, sr: int) -> F64:
    """A close lightning strike's tearing crack (the rolling rumble is made separately)."""
    n = int(r.uniform(0.6, 0.9) * sr)
    y = np.zeros(n)
    t = 0.0
    a = 1.0
    for _ in range(int(r.integers(5, 13))):
        m = int(r.uniform(0.003, 0.02) * sr)
        p = int(t * sr)
        if p + m >= n:
            break
        burst = r.standard_normal(m) * np.exp(-np.arange(m) / (0.3 * m))
        y[p : p + m] += a * burst
        t += r.uniform(0.01, 0.06)
        a *= r.uniform(0.6, 0.9)
    tt = np.arange(n) / sr
    y = lowpass(y, sr, r.uniform(4000, 7000)) * 2.0
    y += (
        0.8
        * np.sin(2 * np.pi * r.uniform(40, 65) * tt)
        * np.exp(-tt / 0.25)
        * (1 - np.exp(-tt / 0.01))
    )
    return peak_norm(y * env_ar(n, sr, 0.0015, 0.2))


def ice_crack(r: Gen, sr: int) -> F64:
    """Ice sheet crack: sharp click then a dispersive falling "pew", sometimes a boom."""
    d = r.uniform(0.25, 0.6)
    n = int(d * sr)
    t = np.arange(n) / sr
    f = 150.0 + r.uniform(2500, 5000) * np.exp(-t / r.uniform(0.04, 0.1))
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / (0.35 * d))
    m = int(0.002 * sr)
    y[:m] += 1.5 * r.standard_normal(m)
    if r.random() < 0.4:
        y += (
            0.9
            * np.sin(2 * np.pi * r.uniform(40, 70) * t)
            * np.exp(-t / 0.15)
            * (1 - np.exp(-t / 0.01))
        )
    return peak_norm(y * env_ar(n, sr, 0.0005, 0.05))


def ice_creak(r: Gen, sr: int) -> F64:
    """Stick-slip creak: a ringing pulse train whose rate glides (like a stressed board)."""
    d = r.uniform(0.3, 1.1)
    rate0, rate1 = r.uniform(25, 90), r.uniform(25, 90)
    fr = r.uniform(400, 1300)
    n = int(d * sr)
    y = np.zeros(n + int(0.03 * sr))
    kt = np.arange(int(0.02 * sr)) / sr
    t = 0.0
    while t < d:
        x = t / d
        rate = rate0 + (rate1 - rate0) * x
        k = np.sin(2 * np.pi * fr * r.uniform(0.97, 1.03) * kt) * np.exp(-kt / 0.003)
        p = int(t * sr)
        y[p : p + kt.size] += math.sin(math.pi * x) * r.uniform(0.5, 1.0) * k
        t += 1.0 / rate * r.uniform(0.85, 1.15)
    return peak_norm(y * env_ar(y.size, sr, 0.005, 0.02))
