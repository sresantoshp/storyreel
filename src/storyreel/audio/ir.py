"""Procedurally generated impulse responses for the convolution reverb (ffmpeg `afir`).

These are not recordings of real rooms. Each IR is synthesised deterministically:
  * pre-delay, then a few discrete early reflections (room-specific delays/gains),
  * a late tail of decorrelated noise per channel, split into a low and a high band with
    a one-pole filter; each band decays exponentially to -60 dB at its own RT60
    (high band decays faster = air and wall absorption, the "darkness" of big rooms),
  * "cave" uses a sparse tail (most noise samples zeroed) so it sounds echoey/rocky.
The IR is scaled to unit energy, so a reverb at mix=1 has roughly the input's loudness.
LFE gets no reverb. Files are cached per (room, sample rate, layout, version).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from storyreel.audio import spec
from storyreel.audio.envelope import F32, F64, write_f32

IR_VERSION = 1


def _one_pole_lowpass(x: F64, cutoff_hz: float, sr: int) -> F64:
    """One-pole low-pass y[n] = y[n-1] + a (x[n] - y[n-1]), without scipy."""
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sr)
    # Closed form of the recursion: y = a * sum_k (1-a)^k x[n-k]; computed as an FIR
    # truncated where (1-a)^k < 1e-7 (a few hundred taps at 48 kHz and 2 kHz cutoff).
    taps = int(np.ceil(np.log(1e-7) / np.log(1.0 - a))) + 1
    h = a * (1.0 - a) ** np.arange(taps)
    y: F64 = np.convolve(x, h)[: x.size]
    return y


def generate(room_name: str, sample_rate: int, channels: tuple[str, ...]) -> F32:
    """Interleaved float32 IR, shape (samples, channels)."""
    room = spec.ROOMS[room_name]
    n = int(np.ceil(spec.reverb_tail_s(room_name) * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    out = np.zeros((n, len(channels)), dtype=np.float64)
    pre = int(room.predelay * sample_rate)
    seed_base = sum(ord(ch) for ch in room_name) * 1000
    for ci, name in enumerate(channels):
        if name == "LFE":
            continue
        rng = np.random.default_rng(seed_base + ci)
        noise = rng.standard_normal(n)
        if room.density < 1.0:
            noise *= rng.random(n) < room.density
            noise /= np.sqrt(room.density)
        low = _one_pole_lowpass(noise, 2000.0, sample_rate)
        high = noise - low
        tt = np.maximum(t - room.predelay, 0.0)
        k = np.log(1000.0)  # -60 dB
        tail = low * np.exp(-k * tt / room.rt60) + high * np.exp(
            -k * tt / (room.rt60 * room.hf_ratio)
        )
        tail[:pre] = 0.0
        # fade the tail in over 10 ms after the pre-delay (no hard edge)
        fade = int(0.01 * sample_rate)
        m = max(0, min(fade, n - pre))
        tail[pre : pre + m] *= np.linspace(0.0, 1.0, fade)[:m]
        tail *= 0.5
        # early reflections: alternate sides slightly so the stereo image widens
        for j, (delay, gain) in enumerate(room.early):
            jitter = 0.0013 * ((ci + j) % 3)
            pos = int((delay + jitter) * sample_rate)
            if pos < n:
                tail[pos] += gain * (1.0 if (ci + j) % 2 == 0 else -1.0)
        out[:, ci] = tail
    energy = float(np.sqrt(np.sum(out**2) / max(1, sum(1 for c in channels if c != "LFE"))))
    if energy > 0:
        out /= energy
    result: F32 = out.astype(np.float32)
    return result


def ensure_ir(cache_dir: Path, room_name: str, sample_rate: int, channels: int) -> Path:
    """Path of the cached raw f32le IR for this room and layout; written once, atomically."""
    lay = spec.layout(channels)
    path = cache_dir / f"{room_name}_{sample_rate}_{lay.name}_v{IR_VERSION}.f32"
    if path.is_file():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = generate(room_name, sample_rate, lay.channels)
    part = path.with_name(path.name + f".{os.getpid()}.part")
    write_f32(part, data)
    with part.open("rb+") as fh:
        os.fsync(fh.fileno())
    part.replace(path)
    return path
