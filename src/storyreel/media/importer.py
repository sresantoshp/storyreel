"""Import media into a project: probe, fingerprint, register, build proxies.

Bad files never stop an import: each file gets its own result, and corrupt ones are reported
with a plain-English reason. Originals are referenced in place and never modified.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from storyreel.errors import StoryReelError
from storyreel.media import proxy
from storyreel.media.probe import CorruptMediaError, MediaInfo, fingerprint, probe
from storyreel.project.store import Project

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportResult:
    path: Path
    asset_id: int | None
    status: str  # imported | duplicate | skipped
    message: str
    proxy_errors: tuple[str, ...] = ()


def _register(project: Project, path: Path, info: MediaInfo, fp: str) -> int:
    cur = project.con.execute(
        """INSERT INTO asset(kind, path, name, fingerprint, size_bytes, duration, width, height,
               fps_num, fps_den, vcodec, acodec, has_video, has_audio, sample_rate, channels,
               imported_at, color_space, pix_fmt)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            info.kind,
            str(path),
            path.name,
            fp,
            path.stat().st_size,
            info.duration,
            info.width,
            info.height,
            info.fps_num,
            info.fps_den,
            info.vcodec,
            info.acodec,
            int(info.has_video),
            int(info.has_audio),
            info.sample_rate,
            info.channels,
            time.time(),
            info.color_space,
            info.pix_fmt,
        ),
    )
    return int(cur.lastrowid or 0)


def _set_proxy(project: Project, asset_id: int, kind: str, path: Path, err: str | None) -> None:
    project.con.execute(
        "INSERT OR REPLACE INTO proxy(asset_id, kind, path, params_hash, status, error) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (asset_id, kind, str(path), "v1", "failed" if err else "ok", err),
    )


def build_proxies(
    project: Project, ffmpeg: Path, asset_id: int, src: Path, info: MediaInfo, proxy_height: int
) -> list[str]:
    """Generate proxies; return error messages (empty = all good).

    A decode failure in the full-file video proxy is the strongest corruption signal we
    have (ffprobe only reads headers), so it marks the asset corrupt.
    """
    out = project.proxies_dir / str(asset_id)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    jobs: list[tuple[str, Path]] = []
    if info.kind == "video":
        jobs += [
            (f"video{proxy_height}", out / f"proxy_{proxy_height}.mp4"),
            ("thumbs", out / "thumbs"),
        ]
    if info.kind == "image":
        jobs.append(("thumbs", out / "thumb.jpg"))
    if info.has_audio:
        jobs.append(("peaks", out / "audio.peaks"))
    for kind, path in jobs:
        try:
            if kind.startswith("video"):
                proxy.make_video_proxy(ffmpeg, src, path, height=proxy_height)
            elif kind == "thumbs" and info.kind == "image":
                proxy.make_still_thumb(ffmpeg, src, path)
            elif kind == "thumbs":
                proxy.make_thumbs(ffmpeg, src, path)
            else:
                proxy.make_peaks(ffmpeg, src, path)
            _set_proxy(project, asset_id, kind, path, None)
        except StoryReelError as exc:
            msg = f"{kind}: {exc.message}"
            errors.append(msg)
            _set_proxy(project, asset_id, kind, path, msg)
            log.warning("proxy failed", extra={"asset_id": asset_id, "kind": kind, "error": msg})
    if errors and any(e.startswith("video") for e in errors):
        project.con.execute(
            "UPDATE asset SET status = 'corrupt', error = ? WHERE id = ?",
            ("could not decode the whole file (damaged or truncated)", asset_id),
        )
    return errors


def import_files(
    project: Project,
    files: list[Path],
    *,
    ffmpeg: Path,
    ffprobe: Path,
    proxy_height: int = 540,
    make_proxies: bool = True,
) -> list[ImportResult]:
    results: list[ImportResult] = []
    for raw in files:
        path = raw.resolve()
        try:
            info = probe(ffprobe, path)
            fp = fingerprint(path)
        except (CorruptMediaError, OSError) as exc:
            msg = exc.message if isinstance(exc, StoryReelError) else f"{path.name}: {exc}"
            results.append(ImportResult(path, None, "skipped", msg))
            log.warning("import skipped", extra={"path": str(path), "reason": msg})
            continue
        dup = project.con.execute(
            "SELECT id FROM asset WHERE fingerprint = ? AND path = ?", (fp, str(path))
        ).fetchone()
        if dup is not None:
            results.append(ImportResult(path, int(dup["id"]), "duplicate", "already imported"))
            continue
        asset_id = _register(project, path, info, fp)
        perr: list[str] = []
        if make_proxies:
            perr = build_proxies(project, ffmpeg, asset_id, path, info, proxy_height)
        status = project.con.execute(
            "SELECT status FROM asset WHERE id = ?", (asset_id,)
        ).fetchone()
        if status["status"] == "corrupt":
            results.append(
                ImportResult(
                    path,
                    asset_id,
                    "skipped",
                    f"{path.name}: file is damaged and cannot be fully decoded.",
                    tuple(perr),
                )
            )
            continue
        dur = f"{info.duration / 705_600_000:.2f} s" if info.duration else "still"
        size = f"{info.width}x{info.height}" if info.width else "audio"
        results.append(
            ImportResult(path, asset_id, "imported", f"{info.kind}, {size}, {dur}", tuple(perr))
        )
    return results
