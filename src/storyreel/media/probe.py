"""ffprobe wrapper: file → MediaInfo, or a typed CorruptMediaError with a plain message."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from storyreel.errors import StoryReelError
from storyreel.proc import run
from storyreel.timebase import seconds_to_flicks

IMAGE_CODECS = frozenset({"png", "mjpeg", "webp", "bmp", "tiff", "gif", "jpegls"})
IMAGE_FORMATS = ("image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe")
MAX_DURATION_S = 48 * 3600
MAX_DIM = 16384
FINGERPRINT_BLOCK = 4 * 2**20


class CorruptMediaError(StoryReelError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    kind: str  # video | image | audio
    duration: int | None  # flicks; None for stills
    width: int | None
    height: int | None
    fps_num: int | None
    fps_den: int | None
    vcodec: str | None
    acodec: str | None
    has_video: bool
    has_audio: bool
    sample_rate: int | None
    channels: int | None
    color_space: str | None = None
    pix_fmt: str | None = None


def fingerprint(path: Path) -> str:
    """size + mtime + BLAKE2b of first/middle/last 4 MB. ~5 ms for any file size."""
    st = path.stat()
    h = hashlib.blake2b(digest_size=20)
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with path.open("rb") as fh:
        for offset in sorted(
            {
                0,
                max(0, st.st_size // 2 - FINGERPRINT_BLOCK // 2),
                max(0, st.st_size - FINGERPRINT_BLOCK),
            }
        ):
            fh.seek(offset)
            h.update(fh.read(FINGERPRINT_BLOCK))
    return h.hexdigest()


def _rate(s: str | None) -> Fraction | None:
    if not s or s in ("0/0", "0/1"):
        return None
    try:
        r = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return r if 0 < r <= 1000 else None


def probe(ffprobe: Path, path: Path, *, timeout_s: float = 30.0) -> MediaInfo:
    if not path.is_file():
        raise CorruptMediaError(f"{path.name}: file not found.")
    if path.stat().st_size == 0:
        raise CorruptMediaError(f"{path.name}: file is empty (0 bytes).")
    res = run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        timeout_s=timeout_s,
        check=False,
    )
    if res.returncode != 0:
        detail = res.stderr.strip().splitlines()[-1:] or ["unknown error"]
        raise CorruptMediaError(f"{path.name}: not a readable media file ({detail[0][:200]}).")
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise CorruptMediaError(f"{path.name}: ffprobe returned garbage.") from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})
    v = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None and a is None:
        raise CorruptMediaError(f"{path.name}: no video or audio streams.")

    fmt_name = str(fmt.get("format_name", ""))
    is_image = (
        v is not None
        and a is None
        and (
            fmt_name.startswith(IMAGE_FORMATS)
            or (v.get("codec_name") in IMAGE_CODECS and int(v.get("nb_frames", "1") or 1) <= 1)
        )
    )
    duration: int | None = None
    if not is_image:
        try:
            dur_s = float(fmt.get("duration") or (v or a or {}).get("duration") or 0)
        except ValueError:
            dur_s = 0.0
        if not 0 < dur_s <= MAX_DURATION_S:
            raise CorruptMediaError(f"{path.name}: unreadable or absurd duration ({dur_s:.3f} s).")
        duration = seconds_to_flicks(dur_s)

    width = height = None
    fr: Fraction | None = None
    if v is not None:
        width, height = int(v.get("width") or 0), int(v.get("height") or 0)
        if not (0 < width <= MAX_DIM and 0 < height <= MAX_DIM):
            raise CorruptMediaError(f"{path.name}: invalid picture size {width}x{height}.")
        fr = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
    return MediaInfo(
        kind="image" if is_image else ("video" if v is not None else "audio"),
        duration=duration,
        width=width,
        height=height,
        fps_num=fr.numerator if fr and not is_image else None,
        fps_den=fr.denominator if fr and not is_image else None,
        vcodec=v.get("codec_name") if v else None,
        acodec=a.get("codec_name") if a else None,
        has_video=v is not None,
        has_audio=a is not None,
        sample_rate=int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        channels=int(a["channels"]) if a and a.get("channels") else None,
        color_space=(v.get("color_space") or None) if v else None,
        pix_fmt=(v.get("pix_fmt") or None) if v else None,
    )
