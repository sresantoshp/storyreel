"""`storyreel doctor`: check the environment and print a fix for every problem.

Status levels: OK, INFO (optional feature absent), WARN (works, degraded), FAIL (StoryReel
cannot do its core job). Exit code is ENV_CHECK_FAILED if any check FAILs.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import psutil

from storyreel import paths
from storyreel.errors import StoryReelError
from storyreel.hw.profile import CLASS_4GB, CLASS_8GB
from storyreel.hw.sysinfo import GB
from storyreel.media import ffbin


class Status(StrEnum):
    OK = "OK"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    fix: str | None = None


MIN_FFMPEG = (6, 0)
API_KEY_VARS = ("ANTHROPIC_API_KEY", "PEXELS_API_KEY", "PIXABAY_API_KEY")

# (what, how to detect, level if missing, why it matters)
_FF_FEATURES: tuple[tuple[str, str, Status, str], ...] = (
    ("libx264", "enc", Status.FAIL, "default H.264 encoder"),
    ("aac", "enc", Status.FAIL, "default audio encoder"),
    ("loudnorm", "flt", Status.FAIL, "EBU R128 loudness normalisation"),
    ("ebur128", "flt", Status.FAIL, "loudness measurement"),
    ("xfade", "flt", Status.WARN, "fast-path transitions"),
    ("sidechaincompress", "flt", Status.WARN, "ducking music under speech"),
    ("lut3d", "flt", Status.WARN, ".cube LUT colour grading"),
    ("subtitles", "flt", Status.WARN, "burned-in subtitles (needs libass)"),
    ("drawtext", "flt", Status.INFO, "ffmpeg text (StoryReel renders text itself)"),
    ("libx265", "enc", Status.INFO, "H.265 export"),
    ("libsvtav1", "enc", Status.INFO, "AV1 export (fast encoder)"),
    ("libvpx-vp9", "enc", Status.INFO, "VP9 export"),
    ("libopus", "enc", Status.INFO, "Opus audio export"),
    ("ac3", "enc", Status.INFO, "AC-3 5.1 export"),
    ("eac3", "enc", Status.INFO, "E-AC-3 7.1 export"),
    ("prores_ks", "enc", Status.INFO, "ProRes archive export"),
    ("ffv1", "enc", Status.INFO, "FFV1 lossless archive"),
    ("flac", "enc", Status.INFO, "FLAC audio / stems"),
)


def _check_python() -> Check:
    v = sys.version_info
    if v >= (3, 11):
        return Check("Python", Status.OK, f"{v.major}.{v.minor}.{v.micro}")
    return Check("Python", Status.FAIL, f"{v.major}.{v.minor}", "Install Python 3.11 or newer.")


def _check_ram() -> Check:
    total = psutil.virtual_memory().total
    avail = psutil.virtual_memory().available
    detail = f"{total / GB:.1f} GB total, {avail / GB:.1f} GB available"
    if total < CLASS_4GB:
        return Check("RAM", Status.FAIL, detail, "StoryReel needs a 4 GB-class machine or better.")
    if avail < 2 * GB:
        return Check(
            "RAM",
            Status.WARN,
            detail,
            "Under 2 GB free right now: renders will pause under memory pressure. "
            "Close browsers/IDEs before long renders.",
        )
    if total < CLASS_8GB:
        return Check("RAM", Status.WARN, detail, "Works; 1 render job at a time, 360p proxies.")
    return Check("RAM", Status.OK, detail)


def _check_disk(directory: Path) -> Check:
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = psutil.disk_usage(str(probe)).free
    detail = f"{free / GB:.0f} GB free at {probe}"
    if free < 5 * GB:
        return Check("Disk space", Status.FAIL, detail, "Free disk space: renders need >= 5 GB.")
    if free < 30 * GB:
        return Check(
            "Disk space",
            Status.WARN,
            detail,
            "Long projects need ~10-30 GB for proxies and render chunks.",
        )
    return Check("Disk space", Status.OK, detail)


def _check_writable(directory: Path) -> Check:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, delete=True) as fh:
            fh.write(b"ok")
    except OSError as exc:
        return Check(
            "Data directory",
            Status.FAIL,
            f"{directory}: {exc.strerror}",
            "Check permissions on the folder, or set a different data dir.",
        )
    return Check("Data directory", Status.OK, str(directory))


def _check_sqlite(directory: Path) -> list[Check]:
    out = [Check("SQLite", Status.OK, sqlite3.sqlite_version)]
    mem = sqlite3.connect(":memory:")
    try:
        mem.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        out.append(Check("SQLite FTS5", Status.OK, "asset search available"))
    except sqlite3.OperationalError:
        out.append(
            Check(
                "SQLite FTS5",
                Status.WARN,
                "missing: asset search falls back to LIKE",
                "Use a Python build whose sqlite3 has FTS5 (python.org builds do).",
            )
        )
    finally:
        mem.close()
    db_path = directory / ".doctor_wal_check.db"
    try:
        con = sqlite3.connect(db_path)
        try:
            mode = con.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        finally:
            con.close()
        status = Status.OK if mode == "wal" else Status.FAIL
        out.append(
            Check(
                "SQLite WAL",
                status,
                f"journal_mode={mode} on {directory}",
                None
                if status is Status.OK
                else "Data dir is on a filesystem without WAL support (network share?).",
            )
        )
    except sqlite3.Error as exc:
        out.append(Check("SQLite WAL", Status.FAIL, str(exc), "Use a local disk for projects."))
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    return out


def _check_ffmpeg(test_hw: bool) -> list[Check]:
    hint = ffbin.install_hint()
    try:
        ff, fp = ffbin.locate()
    except StoryReelError as exc:
        return [Check("ffmpeg", Status.FAIL, exc.message, exc.fix)]
    out: list[Check] = []
    if fp is None:
        out.append(Check("ffprobe", Status.FAIL, "not found", hint))
    else:
        out.append(Check("ffprobe", Status.OK, str(fp)))
    if ff is None:
        return [Check("ffmpeg", Status.FAIL, "not found", hint), *out]
    try:
        info = ffbin.probe(ff, fp)
    except StoryReelError as exc:
        return [Check("ffmpeg", Status.FAIL, f"{ff}: {exc.message}", "Reinstall ffmpeg."), *out]

    if info.version is None:
        out.insert(
            0,
            Check(
                "ffmpeg",
                Status.WARN,
                f"{info.version_line} (git build, unversioned)",
                "Fine if recent; release builds are easier to support.",
            ),
        )
    elif info.version < MIN_FFMPEG:
        out.insert(
            0,
            Check(
                "ffmpeg",
                Status.FAIL,
                info.version_line,
                f"Need ffmpeg >= {MIN_FFMPEG[0]}.{MIN_FFMPEG[1]}. {hint}",
            ),
        )
    else:
        out.insert(0, Check("ffmpeg", Status.OK, f"{info.version_line}  [{ff}]"))

    for name, kind, level, why in _FF_FEATURES:
        present = name in (info.encoders if kind == "enc" else info.filters)
        label = f"ffmpeg {'encoder' if kind == 'enc' else 'filter'} {name}"
        if present:
            out.append(Check(label, Status.OK, why))
        else:
            out.append(
                Check(
                    label,
                    level,
                    f"missing: {why}",
                    "Install a 'full' ffmpeg build (e.g. gyan.dev full, BtbN GPL).",
                )
            )
    shaping = info.has_lib("libharfbuzz")
    out.append(
        Check(
            "ffmpeg libharfbuzz",
            Status.OK if shaping else Status.INFO,
            "complex-script shaping in drawtext"
            if shaping
            else "absent; not needed, StoryReel shapes Indic/RTL text itself (Phase 2)",
        )
    )

    if test_hw:
        info = ffbin.verify_hw_encoders(info)
        if info.hw_encoders_working:
            out.append(Check("Hardware encoders", Status.OK, ", ".join(info.hw_encoders_working)))
        else:
            listed = sorted(e for e in ffbin.HW_ENCODERS if e in info.encoders)
            out.append(
                Check(
                    "Hardware encoders",
                    Status.INFO,
                    f"none working (listed in build: {', '.join(listed) or 'none'}); "
                    "CPU libx264 will be used",
                    "Update GPU drivers if you expected one to work." if listed else None,
                )
            )
    return out


def _check_memory_limits() -> Check:
    name = "Worker memory hard limit"
    if sys.platform == "win32":
        return Check(name, Status.OK, "Windows Job Objects (job-wide commit limit)")
    if sys.platform.startswith("linux"):
        cg2 = Path("/sys/fs/cgroup/cgroup.controllers").exists()
        from shutil import which

        if cg2 and which("systemd-run"):
            return Check(name, Status.OK, "cgroup v2 via systemd-run --user --scope")
        return Check(
            name,
            Status.WARN,
            "RLIMIT_AS fallback only (address-space, imprecise)",
            "Use a distro with cgroup v2 + systemd for strict limits.",
        )
    return Check(name, Status.WARN, "macOS: no reliable per-process RAM limit; watchdog only")


def _check_keys(env: Mapping[str, str]) -> list[Check]:
    return [
        Check(
            f"Key {var}",
            Status.OK if env.get(var) else Status.INFO,
            "set" if env.get(var) else "not set (cloud features using it are disabled)",
        )
        for var in API_KEY_VARS
    ]


def run_checks(
    *, test_hw: bool = True, data_dir: Path | None = None, env: Mapping[str, str] | None = None
) -> list[Check]:
    data_dir = paths.data_dir() if data_dir is None else data_dir
    env = os.environ if env is None else env
    steps: list[Callable[[], Check | list[Check]]] = [
        _check_python,
        _check_ram,
        lambda: _check_disk(data_dir),
        lambda: _check_writable(data_dir),
        lambda: _check_sqlite(data_dir),
        lambda: _check_ffmpeg(test_hw),
        _check_memory_limits,
        lambda: _check_keys(env),
    ]
    checks: list[Check] = []
    for step in steps:
        res = step()
        checks.extend(res if isinstance(res, list) else [res])
    return checks


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        lines.append(f"[{c.status.value:<4}] {c.name:<{width}}  {c.detail}")
        if c.fix and c.status in (Status.WARN, Status.FAIL):
            lines.append(f"{'':>7} {'':<{width}}  fix: {c.fix}")
    counts = {s: sum(1 for c in checks if c.status is s) for s in Status}
    lines.append("")
    lines.append(" ".join(f"{s.value}={n}" for s, n in counts.items()))
    return "\n".join(lines)
