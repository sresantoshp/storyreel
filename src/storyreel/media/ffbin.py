"""Find ffmpeg/ffprobe and discover what the build can actually do.

Lookup order: $STORYREEL_FFMPEG / $STORYREEL_FFPROBE (file paths), the StoryReel bin dir,
then PATH. Capability lists come from the binary itself (`-encoders`, `-filters`,
`-buildconf`); hardware encoders are verified with a real 5-frame encode, because a
listed encoder (e.g. h264_nvenc) is only usable if the matching hardware and driver exist.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from storyreel import paths
from storyreel.errors import StoryReelError
from storyreel.proc import run

_EXE = ".exe" if sys.platform == "win32" else ""

# Hardware encoders worth probing, in preference order per family.
HW_ENCODERS: tuple[str, ...] = (
    "h264_nvenc",
    "hevc_nvenc",
    "av1_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "av1_qsv",
    "h264_amf",
    "hevc_amf",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_vaapi",
    "hevc_vaapi",
)

# Names start with a letter/digit; this rejects legend lines such as " V..... = Video".
_ENCODER_RE = re.compile(r"^\s([VAS][F.][S.][X.][B.][D.])\s+([A-Za-z0-9][\w.-]*)\s")
_FILTER_RE = re.compile(r"^\s([T.][S.][C.]?)\s+([A-Za-z0-9][\w.-]*)\s+(\S+->\S+)\s")
_VERSION_RE = re.compile(r"version\s+n?(\d+)\.(\d+)")


@dataclass(frozen=True)
class FFmpegInfo:
    ffmpeg: Path
    ffprobe: Path | None
    version_line: str
    version: tuple[int, int] | None  # None for git snapshot builds without a release number
    buildconf: frozenset[str]
    encoders: frozenset[str]
    filters: frozenset[str]
    hwaccels: tuple[str, ...]
    hw_encoders_working: tuple[str, ...] = field(default=())
    hw_encoder_errors: Mapping[str, str] = field(default_factory=dict)

    def has_lib(self, name: str) -> bool:
        return f"--enable-{name}" in self.buildconf


def _from_env(var: str, env: Mapping[str, str]) -> Path | None:
    value = env.get(var)
    if not value:
        return None
    p = Path(value)
    if not p.is_file():
        raise StoryReelError(
            f"{var} points to {value}, which is not a file.",
            fix=f"Fix or unset the {var} environment variable.",
        )
    return p


def locate(
    env: Mapping[str, str] | None = None, bin_dir: Path | None = None
) -> tuple[Path | None, Path | None]:
    """Return (ffmpeg, ffprobe); either may be None if not found."""
    env = os.environ if env is None else env
    bin_dir = paths.bin_dir() if bin_dir is None else bin_dir
    search_path = env.get("PATH", "")

    found: list[Path | None] = []
    for name, var in (("ffmpeg", "STORYREEL_FFMPEG"), ("ffprobe", "STORYREEL_FFPROBE")):
        p = _from_env(var, env)
        if p is None:
            candidate = bin_dir / f"{name}{_EXE}"
            if candidate.is_file():
                p = candidate
        if p is None:
            which = shutil.which(name, path=search_path)
            p = Path(which) if which else None
        found.append(p)
    return found[0], found[1]


def parse_encoders(text: str) -> frozenset[str]:
    return frozenset(m.group(2) for line in text.splitlines() if (m := _ENCODER_RE.match(line)))


def parse_filters(text: str) -> frozenset[str]:
    return frozenset(m.group(2) for line in text.splitlines() if (m := _FILTER_RE.match(line)))


def parse_version(first_line: str) -> tuple[int, int] | None:
    m = _VERSION_RE.search(first_line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_hwaccels(text: str) -> tuple[str, ...]:
    lines = [ln.strip() for ln in text.splitlines()]
    if "Hardware acceleration methods:" in lines:
        lines = lines[lines.index("Hardware acceleration methods:") + 1 :]
    return tuple(ln for ln in lines if ln)


def probe(ffmpeg: Path, ffprobe: Path | None, *, timeout_s: float = 20.0) -> FFmpegInfo:
    """Query an ffmpeg binary for version, build flags, encoders, filters, hwaccels."""
    ver = run([ffmpeg, "-hide_banner", "-version"], timeout_s=timeout_s).stdout
    first = ver.splitlines()[0] if ver else ""
    conf = run([ffmpeg, "-hide_banner", "-buildconf"], timeout_s=timeout_s).stdout
    enc = run([ffmpeg, "-hide_banner", "-encoders"], timeout_s=timeout_s).stdout
    flt = run([ffmpeg, "-hide_banner", "-filters"], timeout_s=timeout_s).stdout
    hwa = run([ffmpeg, "-hide_banner", "-hwaccels"], timeout_s=timeout_s).stdout
    return FFmpegInfo(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        version_line=first,
        version=parse_version(first),
        buildconf=frozenset(tok.strip() for tok in conf.split() if tok.startswith("--")),
        encoders=parse_encoders(enc),
        filters=parse_filters(flt),
        hwaccels=parse_hwaccels(hwa),
    )


def _hw_test_argv(ffmpeg: Path, encoder: str) -> list[str]:
    argv = [str(ffmpeg), "-hide_banner", "-loglevel", "error"]
    if encoder.endswith("_vaapi"):
        argv += ["-vaapi_device", "/dev/dri/renderD128"]
    argv += ["-f", "lavfi", "-i", "color=c=gray:s=320x240:r=30:d=0.2", "-frames:v", "5"]
    if encoder.endswith("_vaapi"):
        argv += ["-vf", "format=nv12,hwupload"]
    elif encoder.endswith("_qsv"):
        argv += ["-pix_fmt", "nv12"]
    else:
        argv += ["-pix_fmt", "yuv420p"]
    argv += ["-c:v", encoder, "-f", "null", "-"]
    return argv


def verify_hw_encoders(info: FFmpegInfo, *, timeout_s: float = 20.0) -> FFmpegInfo:
    """Return a copy of `info` with hw encoders that really encode 5 frames on this machine."""
    working: list[str] = []
    errors: dict[str, str] = {}
    for enc in HW_ENCODERS:
        if enc not in info.encoders:
            continue
        if enc.endswith("_vaapi") and not Path("/dev/dri/renderD128").exists():
            errors[enc] = "no /dev/dri/renderD128"
            continue
        try:
            res = run(_hw_test_argv(info.ffmpeg, enc), timeout_s=timeout_s, check=False)
        except StoryReelError as exc:
            errors[enc] = exc.message
            continue
        if res.returncode == 0:
            working.append(enc)
        else:
            last = res.stderr.strip().splitlines()
            errors[enc] = last[-1] if last else f"exit code {res.returncode}"
    return FFmpegInfo(
        ffmpeg=info.ffmpeg,
        ffprobe=info.ffprobe,
        version_line=info.version_line,
        version=info.version,
        buildconf=info.buildconf,
        encoders=info.encoders,
        filters=info.filters,
        hwaccels=info.hwaccels,
        hw_encoders_working=tuple(working),
        hw_encoder_errors=errors,
    )


def install_hint() -> str:
    if sys.platform == "win32":
        return "winget install --id Gyan.FFmpeg -e   (or set STORYREEL_FFMPEG / STORYREEL_FFPROBE)"
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    return "sudo apt install ffmpeg   (or dnf/pacman; or set STORYREEL_FFMPEG)"
