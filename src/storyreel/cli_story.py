"""`storyreel make`: idea or script → finished video, headless and offline.

    storyreel make --idea "A fox finds a frozen river" --length 60s --out fox.mp4
    storyreel make --script-file my_hindi_story.txt --aspect 9:16 --out short.mp4
    storyreel make --idea "..." --no-render --project-dir work/fox.storyreel

Exit codes: 0 done, 2 usage, 3 missing tools/models, 4 render failed, 5 out of resources.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from storyreel.errors import ExitCode, StoryReelError
from storyreel.story import assemble as asm
from storyreel.story.planner import PlanRequest, plan_story

MB = 2**20


def parse_length(text: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(s|sec|m|min|h)?\s*", text)
    if m is None:
        raise StoryReelError(f"Cannot read length {text!r}.", fix="Use e.g. 45s, 2min or 1h.")
    value, unit = float(m.group(1)), (m.group(2) or "s")
    return value * {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600}[unit]


def _engines(voice: str | None) -> tuple[asm.SpeakFn, asm.SoundFn, asm.SoundFn]:
    from storyreel.audio import procedural
    from storyreel.voice import tts

    def speak(text: str, lang: str, out: Path) -> asm.Speech:
        tts_lang = lang if tts.available(lang) else "en"
        if not tts.available(tts_lang):
            raise StoryReelError(
                f"No voice model for {lang!r}.", fix=f"storyreel models download --lang {lang}"
            )
        return tts.synthesize(text, out, lang=tts_lang, voice=voice)

    def ambience(kind: str, seconds: float, out: Path) -> Path:
        return procedural.make_ambience(kind, seconds, out)

    def music(mood: str, seconds: float, out: Path) -> Path:
        return procedural.make_music(mood, seconds, out)

    return speak, ambience, music


def _voice_licence(lang: str, voice: str | None) -> str:
    from storyreel.voice import models, tts

    tts_lang = lang if tts.available(lang) else "en"
    return tts.license_text(tts_lang, voice or models.DEFAULT_VOICE[tts_lang])


def cmd_make(a: argparse.Namespace) -> int:
    from storyreel.cli_project import _tools, default_cap
    from storyreel.project import store
    from storyreel.render.plan import RenderSettings
    from storyreel.render.renderer import Renderer
    from storyreel.watchdog import Watchdog

    t_start = time.perf_counter()
    if bool(a.idea) == bool(a.script_file):
        raise StoryReelError("Give exactly one of --idea or --script-file.")
    text = a.idea or Path(a.script_file).read_text(encoding="utf-8")
    mode = "script" if a.script_file and a.mode == "auto" else a.mode
    out = Path(a.out)
    root = Path(a.project_dir) if a.project_dir else out.with_suffix(".storyreel")
    if root.exists():
        raise StoryReelError(
            f"{root} already exists.", fix="Pick another --project-dir or delete it."
        )
    ff, fp = _tools()

    def say(msg: str) -> None:
        if a.json:
            print(json.dumps({"event": "progress", "message": msg}, ensure_ascii=False), flush=True)
        else:
            print(f"[{time.perf_counter() - t_start:6.1f}s] {msg}", flush=True)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    plan = plan_story(PlanRequest(text, parse_length(a.length), a.lang, mode, a.writer), say)
    timings["plan_s"] = time.perf_counter() - t0
    say(f"plan: '{plan.title}' — {len(plan.scenes)} scenes, {plan.language}, via {plan.source}")
    for w in plan.warnings:
        say(f"WARNING: {w}")

    t0 = time.perf_counter()
    speak, ambience, music = _engines(a.voice)
    voice_licence = _voice_licence(plan.language, a.voice)
    res = asm.assemble(
        plan,
        root,
        speak=speak,
        ambience=ambience,
        music=music,
        ffmpeg=ff,
        ffprobe=fp,
        aspect=a.aspect,
        height=a.height,
        voice_licence=voice_licence,
        progress=say,
        captions=a.captions,
    )
    timings["assemble_s"] = time.perf_counter() - t0
    target = parse_length(a.length)
    if mode != "script" and not 0.7 * target <= res.duration_s <= 1.4 * target:
        say(
            f"WARNING: the narration runs {res.duration_s:.0f} s, target was {target:.0f} s "
            "(timing follows the spoken words; edit plan.json or use --script-file)"
        )
    summary: dict[str, Any] = {
        "project": str(res.project),
        "srt": str(res.srt),
        "duration_s": round(res.duration_s, 2),
        "scenes": res.scenes,
        "source": plan.source,
        "language": plan.language,
    }
    if not a.no_render:
        t0 = time.perf_counter()
        with store.open_project(res.project) as p:
            ps = p.settings()
            settings = RenderSettings(
                ps.width, ps.height, ps.rate, ps.sample_rate, ps.channels, preset=a.preset
            )
            wd = Watchdog(default_cap()).start()
            try:
                r = Renderer(
                    p,
                    settings,
                    ffmpeg=ff,
                    ffprobe=fp,
                    watchdog=wd,
                    on_event=lambda e: (
                        say(f"render chunk {e['idx'] + 1}/{e['of']}")
                        if e["event"] == "chunk_done"
                        else None
                    ),
                ).render(out)
            finally:
                wd.stop()
        timings["render_s"] = time.perf_counter() - t0
        summary |= {
            "output": str(out),
            "output_mb": round(r.output_bytes / MB, 1),
            "frames": r.frames_written,
            "peak_rss_mb": r.peak_tree_rss // MB,
        }
        if r.frames_written != r.frames_expected:
            raise StoryReelError("The rendered video is missing frames.")
    timings["total_s"] = time.perf_counter() - t_start
    summary["timings"] = {k: round(v, 1) for k, v in timings.items()}
    if a.json:
        print(json.dumps({"event": "summary", **summary}, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    return int(ExitCode.OK)


def register(sub: Any) -> None:
    m = sub.add_parser("make", help="idea or script -> finished video (offline)")
    m.add_argument("--idea", help="one-line idea or short story (any language)")
    m.add_argument("--script-file", help="your own narration text file (UTF-8)")
    m.add_argument("--length", default="60s", help="target length for --idea, e.g. 45s, 3min")
    m.add_argument("--lang", default="auto", help="auto (from the text), en, hi, ta, te, kn")
    m.add_argument("--mode", default="auto", choices=["auto", "llm", "script", "rules"])
    m.add_argument("--writer", default="auto", help="auto, qwen2.5-1.5b, phi-3.5-mini")
    m.add_argument("--voice", help="voice id (see `storyreel voices`)")
    m.add_argument("--aspect", default="16:9", choices=sorted(asm.SIZES))
    m.add_argument("--height", type=int, default=1080, help="1080 (default), 720 or 480 for drafts")
    m.add_argument("--preset", default="veryfast", help="x264 preset")
    m.add_argument("--out", required=True)
    m.add_argument("--project-dir")
    m.add_argument("--no-render", action="store_true", help="build the editable project only")
    m.add_argument("--captions", action="store_true", help="also show short on-screen captions")
    m.add_argument("--json", action="store_true", help="JSON-lines progress")
    m.set_defaults(func=cmd_make)


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use: storyreel make ...")
