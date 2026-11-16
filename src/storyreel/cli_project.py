"""Project commands: new, import, append, info, render."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil

from storyreel.audio import spec as aspec
from storyreel.errors import ExitCode, StoryReelError
from storyreel.media import ffbin
from storyreel.media.importer import import_files
from storyreel.project import store
from storyreel.project.edits import Editor
from storyreel.render.plan import RenderSettings
from storyreel.render.renderer import Renderer
from storyreel.timebase import FLICKS, Rate, seconds_to_flicks
from storyreel.watchdog import Watchdog

MB = 2**20
GB = 2**30


def _tools() -> tuple[Path, Path]:
    ff, fp = ffbin.locate()
    if ff is None or fp is None:
        err = StoryReelError(
            f"{'ffmpeg' if ff is None else 'ffprobe'} was not found.", fix=ffbin.install_hint()
        )
        err.exit_code = ExitCode.ENV_CHECK_FAILED
        raise err
    return ff, fp


def default_cap() -> int:
    return min(3 * GB, psutil.virtual_memory().total // 2)


def _parse_rate(text: str) -> Rate:
    table = {"23.976": Rate(24000, 1001), "29.97": Rate(30000, 1001), "59.94": Rate(60000, 1001)}
    if text in table:
        return table[text]
    num, _, den = text.partition("/")
    return Rate(int(num), int(den or 1))


def _note_recovery(p: store.Project) -> None:
    r = p.recovery
    if r.recovered:
        print(
            f"note: recovered from an unclean shutdown (stale lock PID {r.stale_lock_pid}, "
            f"{r.jobs_requeued} interrupted job(s), "
            f"{r.part_files_removed} partial file(s) removed). "
            "No committed edits were lost.",
            file=sys.stderr,
        )


def cmd_new(a: argparse.Namespace) -> int:
    w, _, h = a.size.partition("x")
    root = Path(a.path)
    settings = store.ProjectSettings(a.name or root.stem, int(w), int(h), _parse_rate(a.fps))
    with store.create(root, settings):
        pass
    print(f"Created {root} ({w}x{h} @ {a.fps} fps)")
    return int(ExitCode.OK)


def cmd_import(a: argparse.Namespace) -> int:
    ff, fp = _tools()
    with store.open_project(Path(a.project)) as p:
        _note_recovery(p)
        results = import_files(
            p,
            [Path(f) for f in a.files],
            ffmpeg=ff,
            ffprobe=fp,
            proxy_height=a.proxy_height,
            make_proxies=not a.no_proxy,
        )
    ok = 0
    for r in results:
        tag = {"imported": "OK  ", "duplicate": "DUP ", "skipped": "SKIP"}[r.status]
        ident = f"#{r.asset_id}" if r.asset_id else "  "
        print(f"[{tag}] {ident:>4} {r.path.name}: {r.message}")
        for e in r.proxy_errors:
            print(f"            proxy warning: {e}")
        ok += r.status in ("imported", "duplicate")
    print(f"{ok} of {len(results)} file(s) usable.")
    return int(ExitCode.OK if ok or not results else ExitCode.INPUT_INVALID)


def cmd_append(a: argparse.Namespace) -> int:
    with store.open_project(Path(a.project)) as p:
        _note_recovery(p)
        ed = Editor(p.con)
        tracks = {(r["kind"], r["idx"]): r["id"] for r in p.con.execute("SELECT * FROM track")}
        vt = tracks.get(("video", a.video_track))
        at = tracks.get(("audio", a.audio_track))
        while vt is None or at is None:
            ed.add_track("video" if vt is None else "audio")
            tracks = {(r["kind"], r["idx"]): r["id"] for r in p.con.execute("SELECT * FROM track")}
            vt = tracks.get(("video", a.video_track))
            at = tracks.get(("audio", a.audio_track))
        audio_only = False
        if not a.color:
            row = p.con.execute("SELECT has_video FROM asset WHERE id = ?", (a.asset,)).fetchone()
            audio_only = row is not None and not row["has_video"]
        end = int(
            p.con.execute(
                "SELECT coalesce(max(start + duration), 0) FROM clip WHERE track_id = ?",
                (at if audio_only else vt,),
            ).fetchone()[0]
        )
        start = seconds_to_flicks(a.at) if a.at is not None else end
        src_in = seconds_to_flicks(a.src_in)
        if a.color:
            if a.dur is None:
                raise StoryReelError("A color clip needs --dur.")
            cid = ed.add_clip(vt, start, seconds_to_flicks(a.dur), color=a.color)
            print(f"Added color clip #{cid} at {start / FLICKS:.3f} s")
            return int(ExitCode.OK)
        asset = p.con.execute("SELECT * FROM asset WHERE id = ?", (a.asset,)).fetchone()
        if asset is None:
            raise StoryReelError(f"Asset {a.asset} not found.", fix="See `storyreel info`.")
        if a.dur is not None:
            dur = seconds_to_flicks(a.dur)
        elif asset["duration"] is not None:
            dur = asset["duration"] - src_in
        else:
            dur = 5 * FLICKS  # still image default
        # Snap to the frame grid so chunk boundaries are exact.
        rate = p.settings().rate
        start = rate.frame_to_flicks(rate.flicks_to_frame(start))
        dur = rate.frame_to_flicks(max(1, rate.flicks_to_frame(dur)))
        ids = []
        if asset["has_video"]:
            ids.append(ed.add_clip(vt, start, dur, asset_id=a.asset, src_in=src_in))
        if asset["has_audio"] and at is not None:
            ids.append(
                ed.add_clip(at, start, dur, asset_id=a.asset, src_in=src_in, gain_db=a.gain_db)
            )
        print(f"Added clip(s) {ids} at {start / FLICKS:.3f} s, {dur / FLICKS:.3f} s long")
    return int(ExitCode.OK)


def cmd_info(a: argparse.Namespace) -> int:
    with store.open_project(Path(a.project)) as p:
        _note_recovery(p)
        s = p.settings()
        lay = aspec.layout(s.channels).name
        loud = f"{s.target_lufs:g} LUFS / {s.true_peak:g} dBTP" if s.normalize else "no loudness"
        print(f"{s.name}: {s.width}x{s.height} @ {s.rate} fps, {s.sample_rate} Hz {lay}, {loud}")
        for r in p.con.execute("SELECT * FROM asset ORDER BY id"):
            dur = f"{r['duration'] / FLICKS:9.2f} s" if r["duration"] else "    still  "
            print(f"  asset #{r['id']:<4} {r['kind']:<5} {dur}  {r['status']:<7} {r['name']}")
        for t in p.con.execute("SELECT * FROM track ORDER BY kind DESC, idx"):
            row = p.con.execute(
                "SELECT count(*) n, coalesce(max(start + duration), 0) e "
                "FROM clip WHERE track_id = ?",
                (t["id"],),
            ).fetchone()
            bus = f"  -> {t['bus']}" if t["kind"] == "audio" else ""
            mute = "  MUTED" if t["muted"] else ""
            print(
                f"  track {t['name']:<4} {row['n']:>5} clips, ends at {row['e'] / FLICKS:.2f} s"
                f"{bus}{mute}"
            )
            if a.clips:
                for c in p.con.execute(
                    "SELECT c.*, a.name AS aname FROM clip c "
                    "LEFT JOIN asset a ON a.id = c.asset_id "
                    "WHERE track_id = ? ORDER BY start",
                    (t["id"],),
                ):
                    props = json.loads(c["props"])
                    what = (
                        c["aname"]
                        or props.get("type")
                        or str(props.get("text", ""))[:30]
                        or props.get("color", c["kind"])
                    )
                    speed = c["speed_num"] / c["speed_den"]
                    extra = f" {speed:g}x" if speed != 1 else ""
                    if t["kind"] == "audio":
                        extra += _audio_desc(p, c["id"], c["gain_db"], props)
                    print(
                        f"      clip #{c['id']:<5} {c['start'] / FLICKS:8.2f} s +"
                        f"{c['duration'] / FLICKS:7.2f} s  {c['kind']:<9} {what}{extra}"
                    )
        if a.clips:
            from storyreel.cli_audio import _print_buses

            _print_buses(p)
    return int(ExitCode.OK)


def _audio_desc(p: store.Project, clip_id: int, gain_db: float, props: dict[str, Any]) -> str:
    parts = [f"gain {gain_db:+.1f} dB"]
    if props.get("pan"):
        parts.append(f"pan {props['pan']:+.2f}")
    if props.get("fade_in") or props.get("fade_out"):
        parts.append(f"fades {props.get('fade_in', 0):g}/{props.get('fade_out', 0):g} s")
    keys = p.con.execute(
        "SELECT t, value FROM keyframe WHERE clip_id = ? AND param = 'volume' ORDER BY t",
        (clip_id,),
    ).fetchall()
    if keys:
        parts.append("volume " + " ".join(f"{k['t'] / FLICKS:g}s:{k['value']:+g}dB" for k in keys))
    fx = [
        f"{e['type']}#{e['id']}"
        for e in p.con.execute(
            "SELECT id, type FROM effect WHERE clip_id = ? ORDER BY idx", (clip_id,)
        )
    ]
    if fx:
        parts.append("fx " + ",".join(fx))
    return "  [" + "; ".join(parts) + "]"


def cmd_render(a: argparse.Namespace) -> int:
    ff, fp = _tools()
    root = Path(a.project)
    cap = a.cap_mb * MB if a.cap_mb else default_cap()

    def emit(ev: dict[str, Any]) -> None:
        if a.json:
            print(json.dumps(ev), flush=True)
        elif ev["event"] == "chunk_done":
            print(f"\r  chunk {ev['idx'] + 1}/{ev['of']}", end="", flush=True)
        elif ev["event"] in ("paused_memory", "memory_retry"):
            print(f"\n  {ev['event']}: {ev}", flush=True)

    with store.open_project(root) as p:
        _note_recovery(p)
        ps = p.settings()
        settings = RenderSettings(
            ps.width,
            ps.height,
            ps.rate,
            ps.sample_rate,
            ps.channels,
            preset=a.preset,
            crf=a.crf,
            chunk_seconds=a.chunk_seconds,
            acodec=a.acodec,
            abitrate=a.abitrate or "",
            target_lufs=(
                None
                if a.no_normalize or not (ps.normalize or a.target_lufs is not None)
                else (a.target_lufs if a.target_lufs is not None else ps.target_lufs)
            ),
            true_peak=ps.true_peak,
            out_channels=aspec.LAYOUT_NAMES[a.downmix] if a.downmix else None,
            stems=a.stems,
        )
        out = Path(a.output) if a.output else p.renders_dir / f"{ps.name}.mp4"
        wd = Watchdog(cap).start()
        try:
            r = Renderer(p, settings, ffmpeg=ff, ffprobe=fp, watchdog=wd, on_event=emit).render(out)
        finally:
            wd.stop()
    if not a.json:
        print()
        d = asdict(r)
        print(f"Output     {r.output}  ({r.output_bytes / MB:.1f} MB, {r.duration_s:.2f} s)")
        print(
            f"Chunks     {r.chunks}  video hits {r.video_hits}, rendered {r.video_rendered}; "
            f"audio hits {r.audio_hits}, rendered {r.audio_rendered}; hit rate {r.hit_rate:.1%}"
        )
        print(
            f"Frames     expected {r.frames_expected}, written {r.frames_written}, "
            f"dropped {r.frames_expected - r.frames_written}"
        )
        print(f"Wall       {r.wall_s:.1f} s ({r.duration_s / r.wall_s:.2f}x real time)")
        if r.loudness_i is not None:
            print(
                f"Loudness   {r.loudness_i:.1f} LUFS, LRA {r.loudness_lra:.1f} LU, true peak "
                f"{r.true_peak:.1f} dBTP (gain {r.loudness_gain_db:+.1f} dB, "
                f"{r.audio_passes} encode pass(es))"
            )
        for stem in r.stems:
            print(f"Stem       {stem}")
        print(
            f"Memory     peak tree RSS {r.peak_tree_rss / MB:.0f} MB, peak job commit "
            f"{d['peak_job_commit'] / MB:.0f} MB, cap {cap / MB:.0f} MB, "
            f"memory retries {r.memory_retries}"
        )
    if r.frames_written != r.frames_expected:
        print("error: output frame count does not match the timeline", file=sys.stderr)
        return int(ExitCode.RENDER_FAILED)
    return int(ExitCode.OK)


def register(sub: Any) -> None:
    n = sub.add_parser("new", help="create a project folder")
    n.add_argument("path")
    n.add_argument("--name")
    n.add_argument("--size", default="1920x1080")
    n.add_argument("--fps", default="30", help="30, 25, 24, 29.97, 23.976, 60 or N/D")
    n.set_defaults(func=cmd_new)

    i = sub.add_parser("import", help="import media files (bad files are skipped)")
    i.add_argument("project")
    i.add_argument("files", nargs="+")
    i.add_argument("--no-proxy", action="store_true")
    i.add_argument("--proxy-height", type=int, default=540)
    i.set_defaults(func=cmd_import)

    ap = sub.add_parser("append", help="add a clip (default: at the end of track V1/A1)")
    ap.add_argument("project")
    ap.add_argument("asset", type=int, nargs="?", default=0)
    ap.add_argument("--color", help="add a solid color clip instead, e.g. #203040")
    ap.add_argument("--at", type=float, help="timeline position in seconds")
    ap.add_argument("--in", dest="src_in", type=float, default=0.0, help="source in-point, s")
    ap.add_argument("--dur", type=float, help="duration in seconds")
    ap.add_argument("--gain-db", type=float, default=0.0)
    ap.add_argument("--video-track", type=int, default=0)
    ap.add_argument("--audio-track", type=int, default=0)
    ap.set_defaults(func=cmd_append)

    f = sub.add_parser("info", help="show assets and tracks")
    f.add_argument("project")
    f.add_argument("--clips", action="store_true", help="list every clip with its id")
    f.set_defaults(func=cmd_info)

    r = sub.add_parser("render", help="render the timeline (resumes from cached chunks)")
    r.add_argument("project")
    r.add_argument("-o", "--output")
    r.add_argument("--chunk-seconds", type=float, default=10.0)
    r.add_argument(
        "--cap-mb", type=int, help="memory cap for the render (default min(3 GB, RAM/2))"
    )
    r.add_argument("--crf", type=int, default=20)
    r.add_argument("--preset", default="veryfast")
    r.add_argument("--json", action="store_true", help="JSON-lines progress events on stdout")
    r.add_argument("--acodec", default="aac", choices=sorted(aspec.CODECS))
    r.add_argument("--abitrate", help="e.g. 256k (default depends on codec and channels)")
    r.add_argument("--downmix", choices=["stereo", "5.1"], help="downmix the final audio")
    r.add_argument("--stems", choices=["flac", "wav"], help="also write per-bus stems")
    r.add_argument("--target-lufs", type=float, help="override the project loudness target")
    r.add_argument("--no-normalize", action="store_true", help="skip loudness normalisation")
    r.set_defaults(func=cmd_render)
