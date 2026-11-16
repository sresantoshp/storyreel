"""Chunked renderer with a content-addressed chunk cache.

For each chunk: video and audio are separate jobs with separate keys (a gain change
re-renders only audio). A chunk counts as cached only if its cache_entry row exists and
the file has the recorded size; rows are written after fsync + atomic rename, so a crash
at any moment leaves either a complete, recorded chunk or nothing.

Resume after a crash = run the same render again: finished chunks are cache hits.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from storyreel.audio import graph as agraph
from storyreel.audio import loudness
from storyreel.audio import spec as aspec
from storyreel.audio.envelope import write_f32
from storyreel.audio.ir import ensure_ir
from storyreel.errors import ExitCode, StoryReelError, ToolFailedError
from storyreel.jobs.runner import JobStats, MemoryPressureError, run_ffmpeg
from storyreel.proc import run
from storyreel.project.store import Project
from storyreel.render import ffgraph
from storyreel.render.plan import (
    ChunkPlan,
    RenderSettings,
    load_mix,
    load_timeline,
    plan_chunks,
)
from storyreel.watchdog import MemState, Watchdog

log = logging.getLogger(__name__)
Event = dict[str, Any]


class RenderError(StoryReelError):
    exit_code = ExitCode.RENDER_FAILED


@dataclass
class RenderSummary:
    output: str
    chunks: int
    video_hits: int
    audio_hits: int
    video_rendered: int
    audio_rendered: int
    frames_expected: int
    frames_written: int
    duration_s: float
    wall_s: float
    peak_tree_rss: int
    peak_job_commit: int
    output_bytes: int
    memory_retries: int
    loudness_i: float | None = None  # measured on the final output
    loudness_lra: float | None = None
    true_peak: float | None = None
    loudness_gain_db: float = 0.0  # gain applied by normalisation
    audio_passes: int = 0  # final encodes needed to meet loudness / true-peak targets
    stems: tuple[str, ...] = ()

    @property
    def hit_rate(self) -> float:
        total = 2 * self.chunks
        return (self.video_hits + self.audio_hits) / total if total else 0.0


def _fsync_replace(part: Path, final: Path) -> None:
    with part.open("rb+") as fh:
        os.fsync(fh.fileno())
    part.replace(final)


class Renderer:
    def __init__(
        self,
        project: Project,
        settings: RenderSettings,
        *,
        ffmpeg: Path,
        ffprobe: Path,
        watchdog: Watchdog,
        on_event: Callable[[Event], None] | None = None,
        max_memory_retries: int = 3,
        resume_wait_s: float = 5.0,
    ) -> None:
        self.p = project
        self.s = settings
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.wd = watchdog
        self.emit = on_event or (lambda e: None)
        self.max_memory_retries = max_memory_retries
        self.resume_wait_s = resume_wait_s
        self.peak_commit = 0
        self.memory_retries = 0

    # -- cache -------------------------------------------------------------------------
    def _cached(self, key: str) -> Path | None:
        row = self.p.con.execute(
            "SELECT path, bytes FROM cache_entry WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        path = Path(row["path"])
        try:
            ok = path.stat().st_size == row["bytes"]
        except OSError:
            ok = False
        if not ok:
            self.p.con.execute("DELETE FROM cache_entry WHERE key = ?", (key,))
            return None
        self.p.con.execute(
            "UPDATE cache_entry SET last_used_at = ? WHERE key = ?", (time.time(), key)
        )
        return path

    def _store(self, key: str, kind: str, path: Path) -> None:
        now = time.time()
        self.p.con.execute(
            "INSERT OR REPLACE INTO cache_entry(key, kind, path, bytes, created_at, last_used_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, kind, str(path), path.stat().st_size, now, now),
        )

    # -- one job with memory policy ----------------------------------------------------
    def _wait_until_not_critical(self) -> None:
        stable_since: float | None = None
        last_note = 0.0
        while True:
            now = time.monotonic()
            if self.wd.state is not MemState.CRITICAL:
                stable_since = stable_since or now
                if now - stable_since >= self.resume_wait_s:
                    return
            else:
                stable_since = None
                if now - last_note > 10:
                    last = self.wd.last
                    self.emit(
                        {
                            "event": "paused_memory",
                            "sys_available_mb": (last.sys_available >> 20) if last else None,
                        }
                    )
                    last_note = now
            time.sleep(0.25)

    def _run_job(self, build: Callable[[Path, bool], list[str]], final: Path) -> JobStats:
        part = final.with_name(final.name + ".part")
        low_mem = False
        for attempt in range(self.max_memory_retries + 1):
            if self.wd.state is MemState.CRITICAL:
                self._wait_until_not_critical()
            try:
                stats = run_ffmpeg(build(part, low_mem), limit_bytes=self.wd.cap, watchdog=self.wd)
            except MemoryPressureError:
                part.unlink(missing_ok=True)
                self.memory_retries += 1
                if attempt == self.max_memory_retries:
                    raise
                self.emit({"event": "memory_retry", "attempt": attempt + 1, "low_mem": True})
                low_mem = True
                continue
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            finally:
                part.with_name(part.name + ".spec.json").unlink(missing_ok=True)
                for side in part.parent.glob(part.name + ".env*.f32"):
                    side.unlink(missing_ok=True)
            _fsync_replace(part, final)
            self.peak_commit = max(self.peak_commit, stats.peak_commit or 0)
            return stats
        raise AssertionError("unreachable")

    def _video_argv(self, plan: ChunkPlan, part: Path, low_mem: bool) -> list[str]:
        if not plan.path_b:
            return ffgraph.video_argv(self.ffmpeg, plan, self.s, part, low_mem=low_mem)
        spec_path = part.with_name(part.name + ".spec.json")
        spec_path.write_text(
            json.dumps(worker_spec(plan, self.s, self.ffmpeg, part, low_mem)), encoding="utf-8"
        )
        return [sys.executable, "-m", "storyreel.render.worker", str(spec_path)]

    def _audio_argv(self, plan: ChunkPlan, part: Path, low_mem: bool) -> list[str]:
        sr, ch = self.s.sample_rate, self.s.channels
        ir_dir = self.p.root / "cache" / "ir"
        ctx = agraph.AudioContext(
            sr, aspec.layout(ch), lambda room: ensure_ir(ir_dir, room, sr, ch), part
        )
        argv, side = agraph.build(self.ffmpeg, plan, part, ctx)
        for path, data in side:
            write_f32(path, data)
        return argv

    def _check_audio_settings(self) -> int:
        s = self.s
        if s.acodec not in aspec.CODECS:
            raise RenderError(
                f"Unknown audio codec {s.acodec!r}.", fix=f"Use one of {sorted(aspec.CODECS)}."
            )
        try:
            aspec.layout(s.channels)
        except ValueError as exc:
            raise RenderError(f"Project audio layout: {exc}.") from exc
        out_ch = s.out_channels or s.channels
        try:
            agraph.downmix_filter(s.channels, out_ch)
        except ValueError as exc:
            raise RenderError(f"{exc}.", fix="Downmix to stereo (2) instead.") from exc
        if out_ch > aspec.CODECS[s.acodec].max_channels:
            raise RenderError(
                f"{s.acodec} cannot carry {aspec.layout(out_ch).name} audio.",
                fix="Use aac, opus or flac, or downmix to 5.1/stereo.",
            )
        if s.stems not in (None, "flac", "wav"):
            raise RenderError("Stems are written as flac or wav.")
        return out_ch

    def _measure(self, inputs: list[str], pre_filter: str = "") -> loudness.Loudness:
        return loudness.measure(
            self.ffmpeg,
            inputs,
            pre_filter=pre_filter,
            limit_bytes=self.wd.cap,
            watchdog=self.wd,
        )

    def _finish_audio(
        self, vlist: Path, alist: Path, out: Path, out_ch: int
    ) -> tuple[loudness.Loudness, float, int]:
        """Measure → encode → verify; re-encode with corrections if a target was missed.

        Returns (loudness of the output, gain applied, number of encodes).
        """
        s = self.s
        downmix = agraph.downmix_filter(s.channels, out_ch)
        concat = ["-f", "concat", "-safe", "0", "-i", str(alist)]
        gain, ceiling, limit = 0.0, s.true_peak, False
        target = s.target_lufs
        if target is not None:
            t0 = time.monotonic()
            src = self._measure(concat, downmix)
            self.emit(
                {
                    "event": "loudness_measure",
                    "integrated": src.integrated,
                    "true_peak": src.true_peak,
                    "wall_s": round(time.monotonic() - t0, 3),
                }
            )
            if src.silent:
                self.emit({"event": "warning", "message": "The mix is silent; no loudness gain."})
                target = None
            else:
                gain = target - src.integrated
                # lossy encoders overshoot the limiter; start with a margin
                ceiling = s.true_peak - (0.5 if aspec.CODECS[s.acodec].lossy else 0.1)
                # the limiter (and its 4x resampling) only runs when peaks could reach it
                limit = src.true_peak + gain > ceiling - 0.5
        part = out.with_name(out.name + ".part")
        result: loudness.Loudness | None = None
        passes = 0
        for passes in range(1, 4):
            t_pass = time.monotonic()
            af = ",".join(
                f
                for f in (
                    downmix,
                    loudness.normalize_filter(gain, ceiling, s.sample_rate, limit=limit)
                    if target is not None
                    else "",
                )
                if f
            )
            try:
                run_ffmpeg(
                    ffgraph.mux_argv(
                        self.ffmpeg, vlist, alist, s, part, audio_filter=af, channels=out_ch
                    ),
                    limit_bytes=self.wd.cap,
                    watchdog=self.wd,
                )
                result = self._measure(["-i", str(part)])
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            if target is None:
                break
            tp_over = result.true_peak - s.true_peak
            i_off = target - result.integrated
            self.emit(
                {
                    "event": "loudness_pass",
                    "pass": passes,
                    "integrated": result.integrated,
                    "true_peak": result.true_peak,
                    "gain_db": round(gain, 3),
                    "ceiling_db": round(ceiling, 3),
                    "limiter": limit,
                    "wall_s": round(time.monotonic() - t_pass, 3),
                }
            )
            if tp_over <= 0.0 and abs(i_off) <= 0.3:
                break
            if passes == 3:
                self.emit(
                    {
                        "event": "warning",
                        "message": (
                            f"Loudness {result.integrated:.1f} LUFS / true peak "
                            f"{result.true_peak:.1f} dBTP after 3 encodes "
                            f"(target {target:.1f} LUFS, {s.true_peak:.1f} dBTP)."
                        ),
                    }
                )
                break
            if tp_over > 0.0:
                if limit:
                    ceiling -= tp_over + 0.2
                limit = True
            gain += i_off
        _fsync_replace(part, out)
        assert result is not None
        return result, gain, passes

    def _write_stems(self, alist: Path, out: Path, gain: float) -> tuple[str, ...]:
        fmt = self.s.stems or "flac"
        folder = out.with_name(out.stem + "_stems")
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for k, bus in enumerate(aspec.BUSES, start=1):
            final = folder / f"{bus}.{fmt}"
            part = final.with_name(final.name + ".part")
            try:
                run_ffmpeg(
                    ffgraph.stem_argv(self.ffmpeg, alist, k, gain, fmt, part),
                    limit_bytes=self.wd.cap,
                    watchdog=self.wd,
                )
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            _fsync_replace(part, final)
            paths.append(str(final))
        return tuple(paths)

    # -- main --------------------------------------------------------------------------
    def plan(self) -> list[ChunkPlan]:
        warnings: list[str] = []
        plans = plan_chunks(load_timeline(self.p), self.s, warnings, load_mix(self.p))
        for w in warnings:
            self.emit({"event": "warning", "message": w})
        return plans

    def render(self, out: Path) -> RenderSummary:
        t0 = time.monotonic()
        out_ch = self._check_audio_settings()
        plans = self.plan()
        if not plans:
            raise RenderError(
                "The timeline is empty; nothing to render.",
                fix="Add clips with `storyreel append`.",
            )
        self.p.chunks_dir.mkdir(parents=True, exist_ok=True)
        job_id = self._job_start(out)
        rate = self.s.rate
        vhits = ahits = vdone = adone = 0
        ventries: list[tuple[Path, float]] = []
        aentries: list[tuple[Path, float]] = []
        self.emit({"event": "start", "chunks": len(plans), "frames": plans[-1].f1})
        try:
            for plan in plans:
                dur = plan.frames * rate.den / rate.num
                vkey, akey = plan.video_key(self.s), plan.audio_key(self.s)
                vpath = self._cached(vkey)
                if vpath is None:
                    vpath = self.p.chunks_dir / f"{vkey}.mkv"
                    st = self._run_job(partial(self._video_argv, plan), vpath)
                    self._store(vkey, "vchunk", vpath)
                    vdone += 1
                    self.emit(
                        {
                            "event": "chunk",
                            "kind": "video",
                            "idx": plan.idx,
                            "cached": False,
                            "wall_s": round(st.wall_s, 3),
                            "peak_rss_mb": st.peak_rss >> 20,
                            "fps": round(plan.frames / st.wall_s, 1) if st.wall_s else None,
                            "path": "B" if plan.path_b else "A",
                        }
                    )
                else:
                    vhits += 1
                    self.emit({"event": "chunk", "kind": "video", "idx": plan.idx, "cached": True})
                apath = self._cached(akey)
                if apath is None:
                    apath = self.p.chunks_dir / f"{akey}.mka"
                    st = self._run_job(partial(self._audio_argv, plan), apath)
                    self._store(akey, "achunk", apath)
                    adone += 1
                    self.emit(
                        {
                            "event": "audio_chunk",
                            "idx": plan.idx,
                            "cached": False,
                            "wall_s": round(st.wall_s, 3),
                            "peak_rss_mb": st.peak_rss >> 20,
                            "preroll_s": round(plan.preroll / self.s.sample_rate, 3),
                        }
                    )
                else:
                    ahits += 1
                self.emit({"event": "chunk_done", "idx": plan.idx, "of": len(plans)})
                ventries.append((vpath, dur))
                aentries.append((apath, plan.samples / self.s.sample_rate))
                self.p.con.execute(
                    "UPDATE job SET progress = ? WHERE id = ?",
                    ((plan.idx + 1) / len(plans), job_id),
                )

            self.emit({"event": "mux"})
            out.parent.mkdir(parents=True, exist_ok=True)
            vlist = self.p.renders_dir / "concat_video.ffconcat"
            alist = self.p.renders_dir / "concat_audio.ffconcat"
            self.p.renders_dir.mkdir(parents=True, exist_ok=True)
            vlist.write_text(ffgraph.concat_list(ventries), encoding="utf-8")
            alist.write_text(ffgraph.concat_list(aentries), encoding="utf-8")
            loud, gain, passes = self._finish_audio(vlist, alist, out, out_ch)
            stems: tuple[str, ...] = ()
            if self.s.stems:
                self.emit({"event": "stems"})
                stems = self._write_stems(alist, out, gain)
        except StoryReelError as exc:
            self._job_end(job_id, "failed", exc.message)
            if isinstance(exc, MemoryPressureError | RenderError):
                raise
            detail = ""
            if isinstance(exc, ToolFailedError) and exc.stderr_tail.strip():
                detail = f" ({exc.stderr_tail.strip().splitlines()[-1][:300]})"
            raise RenderError(f"Render failed: {exc.message}{detail}", fix=exc.fix) from exc
        except BaseException as exc:
            self._job_end(job_id, "failed", repr(exc))
            raise

        expected = plans[-1].f1
        written = self.count_frames(out)
        summary = RenderSummary(
            output=str(out),
            chunks=len(plans),
            video_hits=vhits,
            audio_hits=ahits,
            video_rendered=vdone,
            audio_rendered=adone,
            frames_expected=expected,
            frames_written=written,
            duration_s=expected * rate.den / rate.num,
            wall_s=time.monotonic() - t0,
            peak_tree_rss=self.wd.peak_tree_rss,
            peak_job_commit=self.peak_commit,
            output_bytes=out.stat().st_size,
            memory_retries=self.memory_retries,
            loudness_i=loud.integrated,
            loudness_lra=loud.lra,
            true_peak=loud.true_peak,
            loudness_gain_db=round(gain, 3),
            audio_passes=passes,
            stems=stems,
        )
        self._job_end(job_id, "done", None, summary.peak_tree_rss)
        self.emit({"event": "summary", **asdict(summary), "hit_rate": round(summary.hit_rate, 4)})
        return summary

    def count_frames(self, path: Path) -> int:
        res = run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                path,
            ],
            timeout_s=600,
        )
        try:
            return int(res.stdout.strip().splitlines()[0])
        except (ValueError, IndexError) as exc:
            raise RenderError(f"Could not verify the output: {res.stdout[:200]!r}") from exc

    # -- job bookkeeping ---------------------------------------------------------------
    def _job_start(self, out: Path) -> int:
        spec = json.dumps(
            {"output": str(out), "settings": {**asdict(self.s), "rate": str(self.s.rate)}}
        )
        cur = self.p.con.execute(
            "INSERT INTO job(kind, state, spec, attempts, created_at, started_at) "
            "VALUES ('render', 'running', ?, 1, ?, ?)",
            (spec, time.time(), time.time()),
        )
        return int(cur.lastrowid or 0)

    def _job_end(self, job_id: int, state: str, err: str | None, peak: int | None = None) -> None:
        self.p.con.execute(
            "UPDATE job SET state = ?, last_error = ?, finished_at = ?, peak_rss = ? WHERE id = ?",
            (state, err, time.time(), peak, job_id),
        )


def worker_spec(
    plan: ChunkPlan, s: RenderSettings, ffmpeg: Path, out: Path, low_mem: bool = False
) -> dict[str, Any]:
    return {
        "settings": {
            "width": s.width,
            "height": s.height,
            "fps_num": s.rate.num,
            "fps_den": s.rate.den,
        },
        "frames": plan.frames,
        "layers": plan.worker_layers(),
        "ffmpeg": str(ffmpeg),
        "encoder_args": ffgraph.rgb_encoder_args(s, low_mem=low_mem),
        "out": str(out),
    }


def render_frame(project: Project, s: RenderSettings, ffmpeg: Path, frame: int) -> Any:
    """One timeline frame as uint8 RGB via the compositor (preview / golden tests)."""
    from dataclasses import replace

    from storyreel.render.compositor import Compositor

    sb = replace(s, force_path_b=True)
    for plan in plan_chunks(load_timeline(project), sb):
        if plan.f0 <= frame < plan.f1:
            comp = Compositor(worker_spec(plan, sb, ffmpeg, Path("unused")), ffmpeg)
            try:
                return comp.render(frame - plan.f0)
            finally:
                comp.close()
    raise RenderError(f"Frame {frame} is past the end of the timeline.")
