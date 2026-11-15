from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@dataclass(frozen=True)
class Media:
    clip: Path  # 12 s, 640x360, 25 fps, 440 Hz sine
    image: Path  # 320x240 png
    tone: Path  # 20 s, 220 Hz wav
    ffmpeg: Path
    ffprobe: Path


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Media]:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    d = tmp_path_factory.mktemp("media")
    clip, image, tone = d / "clip.mp4", d / "image.png", d / "tone.wav"
    _ff(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=640x360:r=25:d=12",
        "-f",
        "lavfi",
        "-i",
        "sine=f=440:d=12",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-shortest",
        str(clip),
    )
    _ff("-f", "lavfi", "-i", "mandelbrot=s=320x240", "-frames:v", "1", str(image))
    _ff("-f", "lavfi", "-i", "sine=f=220:d=20", str(tone))
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ff and fp
    yield Media(clip, image, tone, Path(ff), Path(fp))
