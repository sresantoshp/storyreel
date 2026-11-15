"""Phase 2 demo: an 18 s animated story opening, built only with `storyreel` CLI commands,
fully offline, from procedural sources plus one still image.

    python examples/phase2_demo.py            # 1920x1080
    python examples/phase2_demo.py --size 1280x720

Output: examples/out/demo.storyreel (project) and examples/out/demo.mp4.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
PROJECT = OUT / "demo.storyreel"


def sr(*args: str) -> str:
    """Run one storyreel CLI command; stop on any failure."""
    cmd = [sys.executable, "-m", "storyreel", *args]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if res.returncode != 0:
        sys.exit(f"FAILED ({res.returncode}): {' '.join(args)}\n{res.stdout}\n{res.stderr}")
    print(f"  $ storyreel {' '.join(args)}\n    {res.stdout.strip().splitlines()[-1]}")
    return res.stdout


def clip_id(output: str) -> int:
    m = re.search(r"[#\[](\d+)", output)  # "clip #4" or "clip(s) [4]"
    if m is None:
        sys.exit(f"no clip id in: {output}")
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1920x1080")
    a = ap.parse_args()
    k = int(a.size.split("x")[1]) / 1080  # scale text and offsets with the frame height

    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    photo = OUT / "valley.png"
    if not photo.exists():  # a synthetic "landscape" still, made offline
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "gradients=s=1920x1080:c0=0x1b3a5c:c1=0x5c7a3b:c2=0xd9a066:"
                        "x0=0:y0=0:x1=1920:y1=1080:nb_colors=3:seed=7",
                        "-frames:v", "1", str(photo)], check=True)  # fmt: skip

    t0 = time.perf_counter()
    sr("new", str(PROJECT), "--size", a.size, "--fps", "30", "--name", "demo")
    sr("import", str(PROJECT), str(photo), "--no-proxy")
    p = str(PROJECT)

    # Scene 1 (0-6 s): starfield + titles
    s1 = clip_id(sr("env", p, "starfield", "--at", "0", "--dur", "6", "--track", "0",
                    "count=1400", "speed=0.08"))  # fmt: skip
    sr("text", p, "The Night Sky", "--at", "0.4", "--dur", "5.2", "--size", f"{110 * k:.0f}",
       "--anim", "pop", "--y", f"{-40 * k:.0f}", "outline=3", "shadow=6",
       "gradient_to=#ffd27a")  # fmt: skip
    sr("text", p, "रात का आसमान", "--at", "1.6", "--dur", "4.0", "--size", f"{64 * k:.0f}",
       "--anim", "fade", "--unit", "word", "--y", f"{70 * k:.0f}", "--track", "3",
       "color=#cfe3ff", "anim_stagger=0.35")  # fmt: skip

    # Scene 2 (6-12 s): night valley photo, Ken Burns, rain, fog, lower third
    s2 = clip_id(sr("append", p, "1", "--at", "6", "--dur", "6"))
    sr("kenburns", p, str(s2), "--zoom-from", "1.05", "--zoom-to", "1.25", "--pan-x",
       f"{-60 * k:.0f}")  # fmt: skip
    sr("fx", p, str(s2), "--preset", "night")
    sr("env", p, "fog", "--at", "6", "--dur", "6", "--track", "1", "density=0.35")
    sr("env", p, "rain", "--at", "6", "--dur", "6", "--track", "2", "count=1100")
    sr("text", p, "Chapter 1 - The Storm", "--at", "7", "--dur", "4.5", "--size",
       f"{46 * k:.0f}", "--anim", "slide", "--unit", "word", "--y", f"{380 * k:.0f}",
       "--track", "3", "box=true", "box_opacity=0.55", "align=left")  # fmt: skip

    # Scene 3 (12-18 s): campfire: gradient sky, fire, embers, typewriter line
    s3 = clip_id(sr("env", p, "gradient", "--at", "12", "--dur", "6", "--track", "0",
                    "top=#05060f", "bottom=#3a1a08"))  # fmt: skip
    sr("env", p, "fire", "--at", "12", "--dur", "6", "--track", "1", "height=0.42")
    sr("env", p, "particles", "--at", "12", "--dur", "6", "--track", "2", "color=#ffb347",
       "count=90", "speed=60")  # fmt: skip
    sr("text", p, "Every story starts with a spark...", "--at", "12.8", "--dur", "5",
       "--size", f"{58 * k:.0f}", "--anim", "typewriter", "--y", f"{-260 * k:.0f}",
       "--track", "3", "anim_stagger=0.05", "shadow=5")  # fmt: skip

    # Transitions on the base track
    sr("transition", p, str(s1), str(s2), "fade_black", "--dur", "1.0")
    sr("transition", p, str(s2), str(s3), "dissolve", "--dur", "1.2")
    build_s = time.perf_counter() - t0

    sr("info", p, "--clips")
    sr("frame", p, "--at", "9.0", "-o", str(OUT / "demo_frame_9s.png"))
    t1 = time.perf_counter()
    out = sr("render", p, "-o", str(OUT / "demo.mp4"))
    print(out)
    print(f"build {build_s:.1f} s (CLI process start-ups included), "
          f"render {time.perf_counter() - t1:.1f} s")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
