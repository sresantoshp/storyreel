"""Voice and generated-sound commands: voices, models, say, ambience, music.

  storyreel models download --lang hi        one-time model download (SHA-256 verified)
  storyreel voices --lang hi
  storyreel say film.storyreel "नमस्ते दुनिया।" --lang hi --at 2.5
  storyreel ambience film.storyreel rain --at 0 --dur 30
  storyreel music film.storyreel calm --at 0 --dur 60

Generated files go to <project>/assets/generated/ and are imported as assets with
is_generated = 1 and their licence. Speech goes on a dialogue-bus track, ambience on an
ambience-bus track, music on a music-bus track (which ducks under dialogue).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from storyreel.audio import procedural
from storyreel.errors import ExitCode
from storyreel.media import generated
from storyreel.project import store
from storyreel.timebase import FLICKS, seconds_to_flicks
from storyreel.voice import models, tts


def cmd_voices(a: argparse.Namespace) -> int:
    ok = tts.available(a.lang)
    default = models.DEFAULT_VOICE[a.lang]
    for v in tts.voices(a.lang):
        print(f"{v}{'  (default)' if v == default else ''}")
    if not ok:
        print(
            f"note: the {a.lang} speech model is not installed; run "
            f"`{models.download_command(a.lang)}`.",
            file=sys.stderr,
        )
    return int(ExitCode.OK)


def cmd_models(a: argparse.Namespace) -> int:
    if a.action == "download":
        tts.ensure_models(lang=a.lang, progress=lambda m: print(m, flush=True))
        spec = models.spec_for(a.lang)
        print(f"Model {spec.id} ready in {models.model_dir(spec)}")
        print(f"Licence: {spec.license}  ({spec.license_source})")
        return int(ExitCode.OK)
    for lang in models.LANGS:
        spec = models.spec_for(lang)
        state = "installed" if tts.available(lang) else "not installed"
        print(f"{lang}: {spec.id} {state}  [{models.model_dir(spec)}]")
        print(f"    licence: {spec.license}")
    return int(ExitCode.OK)


def _tools() -> tuple[Path, Path]:
    from storyreel.cli_project import _tools as tools

    return tools()


def _placed(kind: str, aid: int, cid: int, start: int, dur: int, extra: str = "") -> None:
    print(
        f"Added {kind} clip #{cid} (asset #{aid}) at {start / FLICKS:.3f} s, "
        f"{dur / FLICKS:.3f} s long{extra}"
    )


def cmd_say(a: argparse.Namespace) -> int:
    ff, fp = _tools()
    voice = a.voice or models.DEFAULT_VOICE[a.lang]
    with store.open_project(Path(a.project)) as p:
        spec = models.spec_for(a.lang)
        params = {"text": a.text, "lang": a.lang, "voice": voice, "speed": a.speed, "m": spec.id}
        out = generated.generated_dir(p) / generated.generated_name("tts", params)
        if out.exists():
            print(f"Reusing {out.name} (same text, voice and speed)")
        else:
            t0 = time.monotonic()
            res, st = tts.synthesize_with_stats(
                a.text,
                out,
                lang=a.lang,
                voice=voice,
                speed=a.speed,
                on_progress=lambda i, n: print(f"\r  sentence {i}/{n}", end="", flush=True),
            )
            wall = time.monotonic() - t0
            print(
                f"\rSpoke {res.duration_s:.2f} s of audio in {wall:.1f} s "
                f"({res.duration_s / wall:.2f}x real time; model load {st.load_s:.1f} s), "
                f"peak worker memory {st.peak_rss / 2**20:.0f} MB, "
                f"word times from model for {st.timed_sentences}/{st.sentences} sentence(s)"
            )
        aid = generated.import_generated(
            p,
            out,
            ffmpeg=ff,
            ffprobe=fp,
            license=tts.license_text(a.lang, voice),
            source=f"storyreel:tts?model={spec.id}&voice={voice}&lang={a.lang}&speed={a.speed:g}",
            tags=f"tts speech {a.lang} {voice}",
        )
        cid, start, dur = generated.place_audio(p, aid, bus="dialogue", at=a.at)
    _placed("speech", aid, cid, start, dur, " on the dialogue bus")
    return int(ExitCode.OK)


def _procedural(a: argparse.Namespace, what: str) -> int:
    ff, fp = _tools()
    with store.open_project(Path(a.project)) as p:
        s = p.settings()
        rate = s.rate
        frames = max(1, rate.flicks_to_frame(seconds_to_flicks(a.dur)))
        seconds = rate.frame_to_flicks(frames) / FLICKS
        params = {"kind": a.kind, "seconds": round(seconds, 6), "seed": a.seed, "sr": s.sample_rate}
        out = generated.generated_dir(p) / generated.generated_name(what, params)
        if not out.exists():
            t0 = time.monotonic()
            if what == "music":
                procedural.make_music(a.kind, seconds, out, seed=a.seed, sample_rate=s.sample_rate)
            else:
                procedural.make_ambience(
                    a.kind, seconds, out, seed=a.seed, sample_rate=s.sample_rate
                )
            wall = time.monotonic() - t0
            print(f"Generated {seconds:.2f} s of {a.kind} {what} in {wall:.1f} s")
        aid = generated.import_generated(
            p,
            out,
            ffmpeg=ff,
            ffprobe=fp,
            license=generated.PROCEDURAL_LICENSE,
            source=f"storyreel:{what}?kind={a.kind}&seed={a.seed}&seconds={seconds:g}",
            tags=f"procedural {what} {a.kind}",
        )
        bus = "music" if what == "music" else "ambience"
        cid, start, dur = generated.place_audio(p, aid, bus=bus, at=a.at, gain_db=a.gain_db)
    _placed(f"{a.kind} {what}", aid, cid, start, dur, f" on the {bus} bus")
    return int(ExitCode.OK)


def cmd_ambience(a: argparse.Namespace) -> int:
    return _procedural(a, "ambience")


def cmd_music(a: argparse.Namespace) -> int:
    return _procedural(a, "music")


def register(sub: Any) -> None:
    v = sub.add_parser("voices", help="list text-to-speech voices for a language")
    v.add_argument("--lang", default="en", choices=list(models.LANGS))
    v.set_defaults(func=cmd_voices)

    m = sub.add_parser("models", help="download / list offline speech models")
    m.add_argument("action", choices=["download", "list"])
    m.add_argument("--lang", default="en", choices=list(models.LANGS))
    m.set_defaults(func=cmd_models)

    s = sub.add_parser("say", help="speak text (offline TTS) onto a dialogue track")
    s.add_argument("project")
    s.add_argument("text")
    s.add_argument("--lang", default="en", choices=list(models.LANGS))
    s.add_argument("--voice", help="voice id (see `storyreel voices`)")
    s.add_argument("--speed", type=float, default=1.0, help="0.5-2.0")
    s.add_argument("--at", type=float, help="timeline position, s (default: after last line)")
    s.set_defaults(func=cmd_say)

    for name, choices, helptext, func in (
        (
            "ambience",
            procedural.AMBIENCES,
            "add procedural ambience on an ambience track",
            cmd_ambience,
        ),
        (
            "music",
            procedural.MOODS,
            "add procedural music on a music track (ducks under speech)",
            cmd_music,
        ),
    ):
        g = sub.add_parser(name, help=helptext)
        g.add_argument("project")
        g.add_argument("kind", choices=list(choices))
        g.add_argument("--at", type=float, help="timeline position, s (default: after last)")
        g.add_argument("--dur", type=float, required=True, help="length in seconds")
        g.add_argument("--seed", type=int, default=1)
        g.add_argument("--gain-db", type=float, default=0.0)
        g.set_defaults(func=func)
