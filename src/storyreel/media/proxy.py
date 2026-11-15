"""Proxies for editing: low-res video, audio waveform peaks, thumbnail sprites.

All three stream through ffmpeg; Python never holds more than one read block (peaks) in RAM.
Outputs are written to *.part and renamed when complete.
"""

from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from storyreel.errors import StoryReelError, ToolFailedError
from storyreel.proc import run, tail

PEAKS_MAGIC = b"SRPK"
PEAKS_VERSION = 1
PEAKS_PER_SECOND = 100
_PEAK_SR = 8000  # decode audio at 8 kHz mono for peaks: plenty for a waveform display
_SAMPLES_PER_PEAK = _PEAK_SR // PEAKS_PER_SECOND
THUMB_W, THUMB_H, THUMBS_PER_SPRITE = 160, 90, 10


@dataclass(frozen=True)
class Peaks:
    rate: int
    mins: bytes  # int8 per peak
    maxs: bytes


def _finish(part: Path, final: Path) -> None:
    with part.open("rb+") as fh:
        os.fsync(fh.fileno())
    part.replace(final)


def make_video_proxy(
    ffmpeg: Path, src: Path, out: Path, *, height: int, timeout_s: float = 6 * 3600
) -> None:
    """H.264 proxy, short GOP (15) for fast scrubbing, AAC 96k. Even-width scaling."""
    part = out.with_name(out.name + ".part")
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            src,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"scale=-2:{height}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-g",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-f",
            "mp4",
            "-movflags",
            "+faststart",
            part,
        ],
        timeout_s=timeout_s,
    )
    _finish(part, out)


def make_thumbs(ffmpeg: Path, src: Path, out_dir: Path, *, timeout_s: float = 3600) -> int:
    """One JPEG sprite per 10 s of video: 10 tiles of 160x90, one frame per second."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps=1,scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=decrease,"
        f"pad={THUMB_W}:{THUMB_H}:(ow-iw)/2:(oh-ih)/2,tile={THUMBS_PER_SPRITE}x1"
    )
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            src,
            "-map",
            "0:v:0",
            "-vf",
            vf,
            "-q:v",
            "5",
            out_dir / "sprite_%05d.jpg",
        ],
        timeout_s=timeout_s,
    )
    return len(list(out_dir.glob("sprite_*.jpg")))


def make_still_thumb(ffmpeg: Path, src: Path, out: Path, *, timeout_s: float = 60) -> None:
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            src,
            "-vf",
            f"scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=decrease",
            "-frames:v",
            "1",
            "-q:v",
            "5",
            out,
        ],
        timeout_s=timeout_s,
    )


def make_peaks(ffmpeg: Path, src: Path, out: Path, *, timeout_s: float = 3600) -> int:
    """Stream mono 8 kHz s16 PCM from ffmpeg; write min/max int8 pairs at 100 per second.

    Memory: one 64 KB read block. File size: 200 bytes per second of audio (720 KB/hour).
    """
    import numpy as np

    argv = [
        str(a)
        for a in (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            src,
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(_PEAK_SR),
            "-f",
            "s16le",
            "pipe:1",
        )
    ]
    part = out.with_name(out.name + ".part")
    block = _SAMPLES_PER_PEAK * 2 * 400  # bytes per read: 400 peaks = 4 s of audio
    count = 0
    carry = b""
    with (
        subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p,
        part.open("wb") as fh,
    ):
        assert p.stdout is not None and p.stderr is not None
        fh.write(PEAKS_MAGIC + struct.pack("<BHI", PEAKS_VERSION, PEAKS_PER_SECOND, 0))
        while True:
            data = p.stdout.read(block)
            if not data:
                break
            data = carry + data
            usable = len(data) - len(data) % (_SAMPLES_PER_PEAK * 2)
            carry = data[usable:]
            if usable:
                s = np.frombuffer(data[:usable], dtype="<i2").reshape(-1, _SAMPLES_PER_PEAK)
                pairs = np.empty((s.shape[0], 2), dtype=np.int8)
                pairs[:, 0] = s.min(axis=1) >> 8
                pairs[:, 1] = s.max(axis=1) >> 8
                fh.write(pairs.tobytes())
                count += s.shape[0]
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            p.kill()
            raise StoryReelError(f"Waveform extraction for {src.name} timed out.") from exc
        err = p.stderr.read().decode("utf-8", "replace")
        if p.returncode != 0:
            raise ToolFailedError(
                f"Waveform extraction failed for {src.name}",
                returncode=p.returncode,
                stderr_tail=tail(err),
            )
        fh.seek(len(PEAKS_MAGIC) + 3)
        fh.write(struct.pack("<I", count))
    _finish(part, out)
    return count


def read_peaks(path: Path) -> Peaks:
    raw = path.read_bytes()
    if raw[:4] != PEAKS_MAGIC:
        raise StoryReelError(f"{path.name} is not a peaks file.")
    _ver, rate, count = struct.unpack("<BHI", raw[4:11])
    body = raw[11 : 11 + 2 * count]
    return Peaks(rate, body[0::2], body[1::2])
