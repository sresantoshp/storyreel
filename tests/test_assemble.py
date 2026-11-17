"""Story plan → project → rendered video, with tone stand-ins for voice and music.

The real voice/music engines live in storyreel.voice / audio.procedural; here they are
replaced by ffmpeg-generated tones so the test is fast, offline and model-free.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from conftest import Media, needs_ffmpeg
from storyreel.project import store
from storyreel.render.plan import RenderSettings
from storyreel.render.renderer import Renderer
from storyreel.story import assemble, subtitles
from storyreel.story.planner import PlanRequest, plan_story
from storyreel.timebase import FLICKS
from storyreel.watchdog import Watchdog

pytestmark = needs_ffmpeg

SCRIPT = (
    "Snow fell over the quiet valley. The river froze under the snow. "
    "That night a storm came with heavy rain and thunder. "
    "In the morning the sun rose over the village and birds sang."
)


@dataclass
class ToneSpeech:
    path: Path
    duration_s: float
    words: list[tuple[str, float, float]] = field(default_factory=list)


def _tone(ffmpeg: Path, src: str, seconds: float, out: Path) -> Path:
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            src,
            "-t",
            f"{seconds:.3f}",
            "-ac",
            "2",
            "-ar",
            "48000",
            str(out),
        ],
        check=True,
    )
    return out


def stand_ins(ffmpeg: Path) -> tuple[assemble.SpeakFn, assemble.SoundFn, assemble.SoundFn]:
    def speak(text: str, lang: str, out: Path) -> ToneSpeech:
        dur = max(1.0, 0.055 * len(text))
        _tone(ffmpeg, "sine=f=300", dur, out)
        return ToneSpeech(out, dur, subtitles.estimate_words(text, 0.0, dur))

    def amb(kind: str, seconds: float, out: Path) -> Path:
        return _tone(ffmpeg, "anoisesrc=c=pink:a=0.2", seconds, out)

    def mus(mood: str, seconds: float, out: Path) -> Path:
        return _tone(ffmpeg, "sine=f=220", seconds, out)

    return speak, amb, mus


def test_assemble_and_render(tmp_path: Path, media: Media) -> None:
    plan = plan_story(PlanRequest(SCRIPT, mode="script"))
    kits_used = [s.kit for s in plan.scenes]
    speak, amb, mus = stand_ins(media.ffmpeg)
    res = assemble.assemble(
        plan,
        tmp_path / "story.storyreel",
        speak=speak,
        ambience=amb,
        music=mus,
        ffmpeg=media.ffmpeg,
        ffprobe=media.ffprobe,
        aspect="16:9",
        height=180,
    )
    assert res.scenes == len(plan.scenes)
    # same-kit neighbours merge into one segment
    expected_runs = sum(1 for i, k in enumerate(kits_used) if i == 0 or kits_used[i - 1] != k)
    assert res.runs == expected_runs
    assert res.srt.read_text(encoding="utf-8").startswith("1\n00:00:00")
    assert json.loads((res.project / "plan.json").read_text(encoding="utf-8"))["title"]

    with store.open_project(res.project) as p:
        con = p.con
        assert con.execute("SELECT count(*) FROM asset WHERE is_generated = 1").fetchone()[0] >= 3
        licences = {r[0] for r in con.execute("SELECT DISTINCT license FROM asset")}
        assert "procedural (CC0)" in licences
        ducked = {
            r["name"]
            for r in con.execute("SELECT * FROM bus")
            if json.loads(r["params"]).get("duck")
        }
        assert ducked == {"music", "ambience"}
        timeline_end = con.execute("SELECT max(start + duration) FROM clip").fetchone()[0]
        assert abs(timeline_end / FLICKS - res.duration_s) < 0.1
        s = p.settings()
        wd = Watchdog(2**31).start()
        try:
            r = Renderer(
                p,
                RenderSettings(s.width, s.height, s.rate, preset="ultrafast"),
                ffmpeg=media.ffmpeg,
                ffprobe=media.ffprobe,
                watchdog=wd,
            ).render(tmp_path / "story.mp4")
        finally:
            wd.stop()
    assert r.frames_written == r.frames_expected
    assert abs(r.duration_s - res.duration_s) < 0.1


@pytest.mark.parametrize(
    ("aspect", "size"), [("16:9", (1920, 1080)), ("9:16", (1080, 1920)), ("1:1", (1080, 1080))]
)
def test_frame_sizes(aspect: str, size: tuple[int, int]) -> None:
    assert assemble.frame_size(aspect, 1080) == (size if aspect != "9:16" else (608, 1080))
