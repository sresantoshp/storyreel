"""Loudness: measure with ffmpeg `ebur128` (streaming, O(1) RAM) and normalise.

Two passes over the concatenated master PCM:
  1. measure integrated loudness (I), loudness range (LRA) and true peak (TP);
  2. final encode applies `gain = target - I`, then a true-peak limiter: the signal is
     upsampled 4x (192 kHz at 48 kHz), limited with `alimiter` (lookahead, latency
     compensated, auto-level off) at the ceiling, and resampled back. Peaks between
     samples show up as samples at 4x, so this approximates a true-peak limiter.
Lossy encoders overshoot (AAC by up to ~1-2 dB on dense material), so the encoded file
is measured again; if TP is over the ceiling or I is off by more than the tolerance, the
encode is repeated with a corrected gain/ceiling (at most twice).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from storyreel.errors import StoryReelError
from storyreel.jobs.runner import run_ffmpeg
from storyreel.watchdog import Watchdog

_NUM = r"(-?(?:\d+(?:\.\d+)?|inf))"
_RE = {
    "i": re.compile(r"I:\s+" + _NUM + r" LUFS"),
    "lra": re.compile(r"LRA:\s+" + _NUM + r" LU\b"),
    "tp": re.compile(r"Peak:\s+" + _NUM + r" dBFS"),
}
SILENCE_LUFS = -70.0  # ebur128 reports -70 for digital silence (absolute gate)


class LoudnessError(StoryReelError):
    pass


@dataclass(frozen=True)
class Loudness:
    integrated: float  # LUFS
    lra: float  # LU
    true_peak: float  # dBTP (ebur128 prints it as dBFS)

    @property
    def silent(self) -> bool:
        return self.integrated <= SILENCE_LUFS + 0.05


def parse_summary(stderr: str) -> Loudness:
    """Parse the `Summary:` block that ebur128 prints when it finishes."""
    tail = stderr[stderr.rfind("Summary:") :] if "Summary:" in stderr else ""
    found = {k: rx.search(tail) for k, rx in _RE.items()}
    missing = [k for k, m in found.items() if m is None]
    if not tail or missing:
        raise LoudnessError(
            "Could not read the loudness measurement from ffmpeg "
            f"(missing {', '.join(missing) or 'summary'})."
        )
    vals = {k: float(m.group(1)) for k, m in found.items() if m is not None}
    return Loudness(vals["i"], vals["lra"], vals["tp"])


def measure_argv(
    ffmpeg: Path,
    inputs: list[str],
    *,
    stream: str = "0:a:0",
    pre_filter: str = "",
) -> list[str]:
    af = (pre_filter + "," if pre_filter else "") + "ebur128=peak=true:framelog=quiet"
    return [
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-progress",
        "pipe:1",
        *inputs,
        "-map",
        stream,
        "-af",
        af,
        "-f",
        "null",
        "-",
    ]


def measure(
    ffmpeg: Path,
    inputs: list[str],
    *,
    stream: str = "0:a:0",
    pre_filter: str = "",
    limit_bytes: int | None = None,
    watchdog: Watchdog | None = None,
) -> Loudness:
    """Measure one audio stream. `inputs` are ffmpeg input args, e.g. ["-i", "x.mp4"]."""
    stats = run_ffmpeg(
        measure_argv(ffmpeg, inputs, stream=stream, pre_filter=pre_filter),
        limit_bytes=limit_bytes,
        watchdog=watchdog,
        keep_stderr=True,
    )
    return parse_summary(stats.stderr_tail)


def measure_file(ffmpeg: Path, path: Path) -> Loudness:
    if not path.is_file():
        raise LoudnessError(f"{path} does not exist.")
    return measure(ffmpeg, ["-i", str(path)])


def normalize_filter(
    gain_db: float, ceiling_db: float, sample_rate: int, *, limit: bool = True
) -> str:
    """Gain, then 4x-oversampled lookahead limiting at `ceiling_db`, back to `sample_rate`.

    limit=False: gain only (used when the measured true peak + gain is clearly below the
    ceiling; saves the resampling and keeps the mix bit-identical to the stems' sum)."""
    if not limit:
        return f"volume={gain_db:.3f}dB"
    lim = max(0.0625, 10.0 ** (ceiling_db / 20.0))
    return (
        f"volume={gain_db:.3f}dB,aresample={sample_rate * 4},"
        f"alimiter=limit={lim:.6f}:attack=5:release=50:level=false:latency=true,"
        f"aresample={sample_rate}"
    )
