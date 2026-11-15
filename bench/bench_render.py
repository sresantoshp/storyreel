"""Render benchmark: synthetic 1080p project of any length, peak RSS, wall time, frames.

    python bench/bench_render.py --length 10s
    python bench/bench_render.py --length 10min --kill-after 20     # kill -9 + resume test
    python bench/bench_render.py --length 2h --budget-mb 3072

Peak RSS = sum of working sets of the `storyreel render` process and all its children
(ffmpeg), sampled every 100 ms from outside. Exit 1 if it exceeds the budget, if frames are
dropped, or if a resumed render did not reuse the chunks finished before the kill.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from storyreel.media.importer import import_files
from storyreel.project import store
from storyreel.project.edits import Editor
from storyreel.timebase import FLICKS, Rate

HERE = Path(__file__).resolve().parent
MEDIA = HERE / ".media"
WORK = HERE / ".work"
RESULTS = HERE / "results"
MB = 2**20
SOURCES = [("testsrc2", 440), ("smptehdbars", 330), ("rgbtestsrc", 550)]


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_media() -> list[Path]:
    """3 x 60 s 1080p30 H.264 clips with tones, 2 stills, 1 music bed. Cached on disk."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for name, hz in SOURCES:
        p = MEDIA / f"{name}.mp4"
        if not p.exists():
            ff(
                "-f",
                "lavfi",
                "-i",
                f"{name}=s=1920x1080:r=30:d=60",
                "-f",
                "lavfi",
                "-i",
                f"sine=f={hz}:d=60",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-g",
                "60",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(p),
            )
        out.append(p)
    for i, src in enumerate(("gradients=s=1920x1080:seed=3", "cellauto=s=1920x1080")):
        p = MEDIA / f"still{i}.png"
        if not p.exists():
            ff("-f", "lavfi", "-i", src, "-frames:v", "1", str(p))
        out.append(p)
    music = MEDIA / "music.wav"
    if not music.exists():
        ff(
            "-f",
            "lavfi",
            "-i",
            "sine=f=196:d=60",
            "-f",
            "lavfi",
            "-i",
            "sine=f=247:d=60",
            "-filter_complex",
            "amix=inputs=2",
            "-ac",
            "2",
            str(music),
        )
    out.append(music)
    return out


def parse_length(text: str) -> float:
    for suffix, mult in (("min", 60), ("h", 3600), ("s", 1)):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * mult
    return float(text)


def build_project(root: Path, seconds: float, media: list[Path]) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    rng = random.Random(7)  # noqa: S311  # reproducible test data, not security
    clips = 0
    t_import = time.perf_counter()
    with store.create(root, store.ProjectSettings("bench", 1920, 1080, Rate(30))) as p:
        res = import_files(p, media, ffmpeg=Path(ffmpeg), ffprobe=Path(ffprobe), make_proxies=False)
        ids = [r.asset_id for r in res]
        videos, stills, music = ids[:3], ids[3:5], ids[5]
        t_import = time.perf_counter() - t_import
        ed = Editor(p.con)
        q = "SELECT id FROM track WHERE kind = ?"
        v1, a1 = (p.con.execute(q, (k,)).fetchone()[0] for k in ("video", "audio"))
        v2, a2 = ed.add_track("video"), ed.add_track("audio")
        rate = Rate(30)
        end = rate.frame_to_flicks(rate.flicks_to_frame(round(seconds * FLICKS)))
        t = 0
        n = 0
        while t < end:
            n += 1
            if n % 5 == 0:
                dur = min(4 * FLICKS, end - t)
                ed.add_clip(v1, t, dur, asset_id=rng.choice(stills))
                clips += 1
            else:
                dur = min(rate.frame_to_flicks(rng.randint(90, 360)), end - t)
                src_in = rate.frame_to_flicks(rng.randint(0, 1800 - 361))
                aid = rng.choice(videos)
                ed.add_clip(v1, t, dur, asset_id=aid, src_in=src_in)
                ed.add_clip(a1, t, dur, asset_id=aid, src_in=src_in, gain_db=-3.0)
                clips += 2
            t += dur
        for o in range(int(30 * FLICKS), end - 3 * FLICKS, int(45 * FLICKS)):
            ed.add_clip(v2, o, 3 * FLICKS, color="#204060")
            clips += 1
        for m in range(0, end, 60 * FLICKS):
            ed.add_clip(a2, m, min(60 * FLICKS, end - m), asset_id=music, gain_db=-14.0)
            clips += 1
    return {"clips": clips, "import_s": round(t_import, 2)}


class RssSampler(threading.Thread):
    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid, self.peak, self.samples, self._stop_evt = pid, 0, 0, threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                root = psutil.Process(self.pid)
                procs = [root, *root.children(recursive=True)]
                total = 0
                for pr in procs:
                    try:
                        total += pr.memory_info().rss
                    except psutil.Error:
                        continue
                self.peak, self.samples = max(self.peak, total), self.samples + 1
            except psutil.Error:
                pass
            time.sleep(0.1)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=2)


def run_render(root: Path, out: Path, kill_after: int | None) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "storyreel", "render", str(root), "-o", str(out), "--json"]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = RssSampler(proc.pid)
    sampler.start()
    events: list[dict[str, Any]] = []
    done = 0
    killed = False
    assert proc.stdout is not None
    for line in proc.stdout:
        ev = json.loads(line)
        events.append(ev)
        if ev["event"] == "chunk_done":
            done += 1
            if done % 10 == 0:
                print(
                    f"  {done}/{ev['of']} chunks, peak tree RSS so far {sampler.peak / MB:.0f} MB",
                    flush=True,
                )
            if kill_after is not None and done >= kill_after:
                for c in psutil.Process(proc.pid).children(recursive=True):
                    c.kill()
                proc.kill()
                killed = True
                break
    proc.wait()
    sampler.stop()
    stderr = proc.stderr.read() if proc.stderr else ""
    summary = next((e for e in reversed(events) if e["event"] == "summary"), None)
    vchunks = [
        e for e in events if e["event"] == "chunk" and e["kind"] == "video" and not e["cached"]
    ]
    fps = [e["fps"] for e in vchunks if e.get("fps")]
    return {
        "returncode": proc.returncode,
        "killed": killed,
        "chunks_done": done,
        "wall_s": round(time.perf_counter() - t0, 2),
        "peak_tree_rss_mb": round(sampler.peak / MB),
        "rss_samples": sampler.samples,
        "summary": summary,
        "chunk_fps_min": min(fps, default=None),
        "chunk_fps_median": sorted(fps)[len(fps) // 2] if fps else None,
        "stderr_tail": stderr[-500:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", default="10s")
    ap.add_argument("--budget-mb", type=int, default=2560)
    ap.add_argument("--kill-after", type=int, help="kill -9 after N chunks, then resume")
    ap.add_argument("--keep", action="store_true", help="keep the project and output")
    a = ap.parse_args()
    seconds = parse_length(a.length)
    label = a.length + ("-kill" if a.kill_after else "")
    print(f"== bench {label}: building sources and project")
    media = make_media()
    root = WORK / f"bench-{a.length}.storyreel"
    built = build_project(root, seconds, media)
    out = WORK / f"bench-{a.length}.mp4"
    print(f"   {built['clips']} clips; rendering {seconds:.0f} s of 1080p30")
    runs = [run_render(root, out, a.kill_after)]
    if a.kill_after:
        print(f"   killed after {runs[0]['chunks_done']} chunks; resuming")
        runs.append(run_render(root, out, None))
    final = runs[-1]
    s = final["summary"] or {}
    peak = max(r["peak_tree_rss_mb"] for r in runs)
    result = {
        "label": label,
        "seconds": seconds,
        "budget_mb": a.budget_mb,
        "built": built,
        "peak_tree_rss_mb": peak,
        "runs": runs,
        "machine": {
            "ram_total_mb": psutil.virtual_memory().total // MB,
            "cpu_logical": psutil.cpu_count(),
        },
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    failures = []
    if final["returncode"] != 0 or not s:
        failures.append(f"render failed: {final['stderr_tail']}")
    else:
        dropped = s["frames_expected"] - s["frames_written"]
        print(f"   output      {s['output_bytes'] / MB:.1f} MB, {s['duration_s']:.1f} s video")
        print(
            f"   frames      expected {s['frames_expected']}, written {s['frames_written']}, "
            f"dropped {dropped}"
        )
        print(
            f"   wall        {sum(r['wall_s'] for r in runs):.1f} s total "
            f"({s['duration_s'] / sum(r['wall_s'] for r in runs):.2f}x real time); "
            f"chunk fps min {final['chunk_fps_min']} median {final['chunk_fps_median']}"
        )
        print(
            f"   peak RSS    {peak} MB (budget {a.budget_mb} MB), "
            f"job commit peak {s['peak_job_commit'] / MB:.0f} MB"
        )
        print(f"   cache       hits v{s['video_hits']}/a{s['audio_hits']} of {s['chunks']} chunks")
        if dropped:
            failures.append(f"{dropped} dropped frames")
        if a.kill_after and s["video_hits"] < runs[0]["chunks_done"]:
            failures.append(
                f"resume reused {s['video_hits']} chunks, expected >= {runs[0]['chunks_done']}"
            )
    if peak > a.budget_mb:
        failures.append(f"peak RSS {peak} MB > budget {a.budget_mb} MB")
    print(f"   result file {path}")
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)
        out.unlink(missing_ok=True)
    for f in failures:
        print(f"FAIL: {f}")
    print("PASS" if not failures else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
