"""End-to-end: import → timeline → chunked render → verify frames, cache, resume."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import psutil

from conftest import Media
from storyreel.media.importer import import_files
from storyreel.media.proxy import read_peaks
from storyreel.project import store
from storyreel.project.edits import Editor
from storyreel.render.plan import RenderSettings
from storyreel.render.renderer import Renderer
from storyreel.timebase import FLICKS, Rate
from storyreel.watchdog import Sample, Watchdog

S = RenderSettings(320, 180, Rate(30), chunk_seconds=2.0, preset="ultrafast")


def _project(tmp: Path, media: Media, proxies: bool = False) -> store.Project:
    p = store.create(tmp / "p.storyreel", store.ProjectSettings("t", 320, 180, Rate(30)))
    res = import_files(
        p,
        [media.clip, media.image, media.tone],
        ffmpeg=media.ffmpeg,
        ffprobe=media.ffprobe,
        make_proxies=proxies,
        proxy_height=180,
    )
    assert [r.status for r in res] == ["imported"] * 3
    ed = Editor(p.con)
    q = "SELECT id FROM track WHERE kind = ?"
    v1, a1 = (p.con.execute(q, (k,)).fetchone()[0] for k in ("video", "audio"))
    a2 = ed.add_track("audio")
    ed.add_clip(v1, 0, 4 * FLICKS, asset_id=1)
    ed.add_clip(a1, 0, 4 * FLICKS, asset_id=1)
    ed.add_clip(v1, 4 * FLICKS, 3 * FLICKS, asset_id=2)
    ed.add_clip(v1, 7 * FLICKS, 1 * FLICKS, color="#ff0000")
    ed.add_clip(v1, 8 * FLICKS, 3 * FLICKS, asset_id=1, src_in=6 * FLICKS)
    ed.add_clip(a1, 8 * FLICKS, 3 * FLICKS, asset_id=1, src_in=6 * FLICKS)
    ed.add_clip(a2, 0, 11 * FLICKS, asset_id=3, gain_db=-12.0)
    return p


def _render(p: store.Project, media: Media, out: Path, wd: Watchdog | None = None) -> object:
    wd = wd or Watchdog(2 * 2**30).start()
    try:
        return Renderer(
            p, S, ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, watchdog=wd, resume_wait_s=0.2
        ).render(out)
    finally:
        wd.stop()


def test_import_builds_proxies_and_skips_corrupt(tmp_path: Path, media: Media) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\xff" * 5000)
    truncated = tmp_path / "trunc.mp4"
    data = media.clip.read_bytes()
    truncated.write_bytes(data[: len(data) // 3])
    with store.create(tmp_path / "p", store.ProjectSettings("t", 320, 180, Rate(30))) as p:
        res = import_files(
            p,
            [media.clip, bad, truncated, tmp_path / "missing.mp4"],
            ffmpeg=media.ffmpeg,
            ffprobe=media.ffprobe,
            proxy_height=180,
        )
        assert [r.status for r in res] == ["imported", "skipped", "skipped", "skipped"]
        assert all(r.message for r in res)
        proxies = {r["kind"]: r for r in p.con.execute("SELECT * FROM proxy WHERE asset_id = 1")}
        assert {k: v["status"] for k, v in proxies.items()} == {
            "video180": "ok",
            "thumbs": "ok",
            "peaks": "ok",
        }
        peaks = read_peaks(Path(proxies["peaks"]["path"]))
        assert peaks.rate == 100 and 1150 <= len(peaks.maxs) <= 1210  # ~12 s
        # ffmpeg sine source amplitude is 1/8 → int8 peak ~ 0.125 * 127 = 16
        assert 14 <= max(peaks.maxs) <= 30 and min(peaks.mins) >= 256 - 30
        assert p.con.execute("SELECT name FROM asset_fts WHERE asset_fts MATCH 'clip'").fetchone()


def test_render_exact_frames_and_cache_invalidation(tmp_path: Path, media: Media) -> None:
    with _project(tmp_path, media) as p:
        r1 = _render(p, media, tmp_path / "a.mp4")
        assert r1.frames_written == r1.frames_expected == 330  # type: ignore[attr-defined]
        assert r1.hit_rate == 0  # type: ignore[attr-defined]
        r2 = _render(p, media, tmp_path / "b.mp4")
        assert r2.hit_rate == 1.0  # type: ignore[attr-defined]
        # Edit one shot's audio gain: only the chunk(s) under that clip re-render, audio only.
        ed = Editor(p.con)
        ed.update_clip(6, "gain", gain_db=-6.0)  # A1 clip at 8-11 s
        r3 = _render(p, media, tmp_path / "c.mp4")
        assert r3.video_rendered == 0  # type: ignore[attr-defined]
        assert 1 <= r3.audio_rendered <= 2  # type: ignore[attr-defined]
        assert r3.frames_written == 330  # type: ignore[attr-defined]


def test_render_pauses_on_critical_and_resumes(tmp_path: Path, media: Media) -> None:
    """Watchdog says CRITICAL for the first ticks: render must wait, then finish."""
    ticks = {"n": 0}

    def sampler() -> Sample:
        ticks["n"] += 1
        own = 0.99 if ticks["n"] <= 6 else 0.1
        return Sample(int(own * 2**30), 2**30, 4 * 2**30, 8 * 2**30, 0.1)

    with _project(tmp_path, media) as p:
        wd = Watchdog(2**30, sampler=sampler, interval_s=0.1).start()
        r = _render(p, media, tmp_path / "out.mp4", wd)
        assert r.frames_written == 330  # type: ignore[attr-defined]


def test_kill9_mid_render_then_resume(tmp_path: Path, media: Media) -> None:
    with _project(tmp_path, media) as p:
        root = p.root
    out = tmp_path / "k.mp4"
    cmd = [
        sys.executable,
        "-m",
        "storyreel",
        "render",
        str(root),
        "-o",
        str(out),
        "--chunk-seconds",
        "1",
        "--preset",
        "ultrafast",
        "--json",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    done = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        ev = json.loads(line)
        if ev["event"] == "chunk_done":
            done += 1
            if done == 4:
                for c in psutil.Process(proc.pid).children(recursive=True):
                    c.kill()
                proc.kill()  # TerminateProcess: no cleanup handlers run
                break
    proc.wait()
    assert (root / "project.lock").exists()  # the crash left its lock behind
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    summary = json.loads(res.stdout.strip().splitlines()[-1])
    assert summary["video_hits"] >= 4
    assert summary["frames_written"] == summary["frames_expected"] == 330
    assert "recovered" in res.stderr
