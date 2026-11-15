from __future__ import annotations

from pathlib import Path

import pytest

from conftest import needs_ffmpeg
from storyreel.errors import StoryReelError
from storyreel.media import ffbin

# Copied from real `ffmpeg 9.0.2 -encoders` / `-filters` output, legend lines included.
ENCODERS = """Encoders:
 V..... = Video
 A..... = Audio
 .....D = Supports direct rendering method 1
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D libvpx-vp9           libvpx VP9 (codec vp9)
 A....D aac                  AAC (Advanced Audio Coding)
"""
FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  | = Source or sink filter
  ------
 TS aap               AA->A      Apply Affine Projection algorithm to first audio stream.
 .. amix              N->A       Audio mixing.
 .S xfade             VV->V      Cross fade one video with another video.
 .. color             |->V       Provide an uniformly colored input.
"""


def test_parse_encoders_ignores_legend() -> None:
    assert ffbin.parse_encoders(ENCODERS) == {"libx264", "libvpx-vp9", "aac"}


def test_parse_filters_ignores_legend() -> None:
    assert ffbin.parse_filters(FILTERS) == {"aap", "amix", "xfade", "color"}


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026", (9, 0)),
        ("ffmpeg version n7.1 Copyright (c) 2000-2024", (7, 1)),
        ("ffmpeg version N-117000-g1234abcd-20241001 Copyright", None),
        ("", None),
    ],
)
def test_parse_version(line: str, expected: tuple[int, int] | None) -> None:
    assert ffbin.parse_version(line) == expected


def test_parse_hwaccels() -> None:
    assert ffbin.parse_hwaccels("Hardware acceleration methods:\ncuda\nqsv\n\n") == ("cuda", "qsv")


def test_locate_prefers_env_var(tmp_path: Path) -> None:
    fake = tmp_path / "my-ffmpeg"
    fake.write_bytes(b"")
    ff, fp = ffbin.locate(env={"STORYREEL_FFMPEG": str(fake), "PATH": ""}, bin_dir=tmp_path / "x")
    assert ff == fake
    assert fp is None


def test_locate_env_var_to_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(StoryReelError, match="STORYREEL_FFMPEG"):
        ffbin.locate(env={"STORYREEL_FFMPEG": str(tmp_path / "nope")}, bin_dir=tmp_path)


def test_locate_nothing_found(tmp_path: Path) -> None:
    assert ffbin.locate(env={"PATH": str(tmp_path)}, bin_dir=tmp_path) == (None, None)


@needs_ffmpeg
def test_probe_real_ffmpeg() -> None:
    ff, fp = ffbin.locate()
    assert ff is not None
    info = ffbin.probe(ff, fp)
    assert "libx264" in info.encoders or "aac" in info.encoders
    assert "scale" in info.filters
    assert info.version_line.startswith("ffmpeg version")
