"""Phase 3 demo: a 4-track mix (dialogue, music, ambience, fx) built only with `storyreel`
CLI commands, fully offline, from procedural test audio (ffmpeg lavfi).

    python examples/phase3_audio_demo.py              # 60 s, stereo, AAC, -14 LUFS
    python examples/phase3_audio_demo.py --dur 30 --layout 5.1 --acodec eac3

Output: examples/out/audio.storyreel (project), examples/out/audio.mp4 and
examples/out/audio_stems/{dialogue,music,fx,ambience}.flac.
The video track is a small solid colour so the run measures the audio engine.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
PROJECT = OUT / "audio.storyreel"


def sr(*args: str) -> str:
    """Run one storyreel CLI command; stop on any failure."""
    cmd = [sys.executable, "-m", "storyreel", *args]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if res.returncode != 0:
        sys.exit(f"FAILED ({res.returncode}): {' '.join(args)}\n{res.stdout}\n{res.stderr}")
    lines = res.stdout.strip().splitlines()
    print(f"  $ storyreel {' '.join(args)}\n    {lines[-1] if lines else ''}")
    return res.stdout


def ids(output: str) -> list[int]:
    m = re.search(r"\[([\d, ]+)\]", output)
    if m is None:
        sys.exit(f"no clip ids in: {output}")
    return [int(x) for x in m.group(1).split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--layout", default="stereo", choices=["stereo", "5.1", "7.1"])
    ap.add_argument("--acodec", default="aac")
    a = ap.parse_args()
    dur = a.dur

    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    src = {}
    for kind in ("speech", "music", "ambience", "fx"):
        src[kind] = OUT / f"{kind}.wav"
        sr("gen-audio", kind, str(src[kind]), "--dur", f"{dur:g}")
    sr("new", str(PROJECT), "--size", "320x180", "--fps", "25", "--name", "audio")
    sr("audio", str(PROJECT), f"layout={a.layout}", "target_lufs=youtube", "true_peak=-1")
    sr("import", str(PROJECT), *(str(p) for p in src.values()), "--no-proxy")
    sr("append", str(PROJECT), "--color", "#203040", "--dur", f"{dur:g}")
    # dialogue: three lines with gaps, so ducking has something to do
    lines = [
        (2.0, 0.0, dur * 0.25),
        (dur * 0.4, dur * 0.3, dur * 0.2),
        (dur * 0.7, dur * 0.5, dur * 0.25),
    ]
    dialogue = []
    for at, src_in, length in lines:
        out = sr("append", str(PROJECT), "1", "--at", f"{at:g}", "--in", f"{src_in:g}",
                 "--dur", f"{min(length, dur - at):g}", "--audio-track", "0")  # fmt: skip
        dialogue += ids(out)
    music = ids(sr("append", str(PROJECT), "2", "--at", "0", "--audio-track", "1"))[0]
    amb = ids(sr("append", str(PROJECT), "3", "--at", "0", "--audio-track", "2"))[0]
    fx = ids(sr("append", str(PROJECT), "4", "--at", "1", "--dur", f"{dur - 2:g}",
                "--audio-track", "3"))[0]  # fmt: skip
    sr("track", str(PROJECT), "A2", "bus=music")
    sr("track", str(PROJECT), "A3", "bus=ambience")
    sr("track", str(PROJECT), "A4", "bus=fx")
    # per-clip: dialogue chain, music fades, ambience swell, fx panned left
    for cid in dialogue:
        sr("afx", str(PROJECT), str(cid), "highpass", "freq=90")
        sr("afx", str(PROJECT), str(cid), "denoise", "amount=10")
        sr("afx", str(PROJECT), str(cid), "deesser", "intensity=0.4")
        sr("afx", str(PROJECT), str(cid), "compressor", "threshold=-24", "ratio=3")
    sr("set", str(PROJECT), str(music), "fade_in=2", "fade_out=4")
    sr("keyframe", str(PROJECT), str(amb), "volume", "0", "-12")
    sr("keyframe", str(PROJECT), str(amb), "volume", f"{dur / 2:g}", "-3", "--ease", "ease_in_out")
    sr("set", str(PROJECT), str(fx), "pan=-0.6")
    # buses
    sr("afx", str(PROJECT), "music", "eq", "freq=2500", "gain=-3", "q=0.8")
    sr("afx", str(PROJECT), "fx", "reverb", "room=hall", "mix=0.35")
    sr("afx", str(PROJECT), "dialogue", "reverb", "room=small_room", "mix=0.12")
    sr("bus", str(PROJECT), "music", "duck_db=-12", "duck_attack=0.3", "duck_release=0.8")
    sr("bus", str(PROJECT), "ambience", "gain_db=-4")
    sr("afx", str(PROJECT), "master", "compressor", "threshold=-16", "ratio=2", "attack=20")
    print(sr("info", str(PROJECT), "--clips"))

    out = OUT / "audio.mp4"
    t1 = time.perf_counter()
    res = sr("render", str(PROJECT), "-o", str(out), "--json", "--stems", "flac",
             "--acodec", a.acodec)  # fmt: skip
    events = [json.loads(x) for x in res.splitlines() if x.startswith("{")]
    summary = [e for e in events if e["event"] == "summary"][-1]
    achunks = [e for e in events if e["event"] == "audio_chunk"]
    vchunks = [e for e in events if e["event"] == "chunk" and e.get("kind") == "video"]
    measure = [e for e in events if e["event"] == "loudness_measure"]
    passes = [e for e in events if e["event"] == "loudness_pass"]
    loud = json.loads(sr("loudness", str(out), "--json"))
    print(
        json.dumps(
            {
                "timeline_s": dur,
                "render_wall_s": round(time.perf_counter() - t1, 2),
                "total_wall_s": round(time.perf_counter() - t0, 2),
                "peak_tree_rss_mb": summary["peak_tree_rss"] >> 20,
                "audio_chunks": len(achunks),
                "audio_chunks_wall_s": round(sum(e["wall_s"] for e in achunks), 2),
                "audio_chunk_max_rss_mb": max((e["peak_rss_mb"] for e in achunks), default=0),
                "audio_preroll_s": max((e["preroll_s"] for e in achunks), default=0),
                "video_chunks_wall_s": round(sum(e.get("wall_s") or 0 for e in vchunks), 2),
                "loudness_measure_wall_s": measure[0]["wall_s"] if measure else None,
                "encode_passes": [(e["wall_s"], e["limiter"]) for e in passes],
                "audio_passes": summary["audio_passes"],
                "loudness_gain_db": summary["loudness_gain_db"],
                **loud,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
