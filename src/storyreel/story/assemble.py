"""StoryPlan → editable StoryReel project (then render with the normal renderer).

Layout (bottom → top):
  V1..V5  kit layers (base gradient on V1, overlays above)      dialogue bus  A1 narration
  V6      title / on-screen text                                 ambience bus  A2 kit ambience
  V7      subtitles (also written as .srt)                        music bus     A3 music bed

Scenes are timed by their spoken narration (speech duration + lead/tail). Consecutive scenes
with the same kit become ONE clip per layer, so rain or snow never restarts mid-story; kit
changes dissolve on the base layer and fade the overlays. Music and ambience duck under the
narration (Phase 3 ducking). Everything stays an ordinary, editable project.

Voice, ambience and music come in through callables, so this module does not depend on a
particular TTS engine (storyreel.voice provides the default implementation).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from storyreel.errors import StoryReelError
from storyreel.media.importer import import_files
from storyreel.project import store
from storyreel.project.edits import Editor
from storyreel.render import actors
from storyreel.story import cast as castmod
from storyreel.story import subtitles
from storyreel.story.cast import MAX_CAST
from storyreel.story.kits import KITS
from storyreel.story.schema import StoryPlan
from storyreel.timebase import FLICKS, Rate

log = logging.getLogger(__name__)
Progress = Callable[[str], None]

LEAD_S = 0.4  # silence before narration starts in a scene
TAIL_S = 0.7  # breathing room after it ends
MIN_SCENE_S = 3.0
XFADE_S = 0.8  # dissolve between kits
OVERLAY_FADE_S = 0.6
SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}


class Speech(Protocol):
    @property
    def path(self) -> Path: ...
    @property
    def duration_s(self) -> float: ...
    @property
    def words(self) -> list[tuple[str, float, float]]: ...


SpeakFn = Callable[[str, str, Path], Speech]  # (text, lang, out path) -> speech
SoundFn = Callable[[str, float, Path], Path]  # (kind or mood, seconds, out path) -> file


@dataclass(frozen=True)
class Assembled:
    project: Path
    srt: Path
    duration_s: float
    scenes: int
    runs: int  # visual segments after merging same-kit scenes


def frame_size(aspect: str, height: int) -> tuple[int, int]:
    if aspect not in SIZES:
        raise StoryReelError(f"Unknown aspect {aspect!r}.", fix=f"Use one of {', '.join(SIZES)}.")
    w, h = SIZES[aspect]
    k = height / max(w, h) if aspect in ("9:16", "4:5") else height / h
    return max(2, round(w * k / 2) * 2), max(2, round(h * k / 2) * 2)


def _snap(rate: Rate, seconds: float) -> int:
    return rate.frame_to_flicks(rate.flicks_to_frame(round(seconds * FLICKS)))


def _snap_down(rate: Rate, flicks: int) -> int:
    """Largest whole-frame length not longer than `flicks` (media clips must fit the file)."""
    return rate.frame_to_flicks(flicks * rate.num // (rate.den * FLICKS))


def _asset_len(p: store.Project, asset_id: int) -> int:
    return int(p.con.execute("SELECT duration FROM asset WHERE id = ?", (asset_id,)).fetchone()[0])


def _import_generated(
    p: store.Project, ffmpeg: Path, ffprobe: Path, path: Path, licence: str
) -> int:
    res = import_files(p, [path], ffmpeg=ffmpeg, ffprobe=ffprobe, make_proxies=False)[0]
    if res.asset_id is None or res.status == "skipped":
        raise StoryReelError(f"Generated file could not be imported: {res.message}")
    p.con.execute(
        "UPDATE asset SET is_generated = 1, license = ? WHERE id = ?", (licence, res.asset_id)
    )
    return int(res.asset_id)


_RUN_WORDS = (
    "run",
    "ran",
    "runs",
    "running",
    "raced",
    "races",
    "hurried",
    "hurries",
    "rushed",
    "dashed",
    "chased",
    "fled",
    "sprinted",
    "bolted",
    "भाग",
    "दौड़",
)
_IDLE_WORDS = ("sat", "sits", "sitting", "slept", "sleeps", "sleeping", "rested", "resting",
               "watched", "watches", "waited", "waits", "stood", "standing", "listened",
               "lay", "gazed", "ate", "eating", "बैठ", "सो ", "देख", "रुक")  # fmt: skip


def _action(narration: str) -> str:
    low = f" {narration.lower()} "
    words = set(re.findall(r"[a-z]+", low))

    def hit(vocab: tuple[str, ...]) -> bool:
        return any((v in words) if v.isascii() else (v in low) for v in vocab)

    if hit(_RUN_WORDS):
        return "run"
    if hit(_IDLE_WORDS):
        return "idle"
    return "walk"


def _place_cast(
    ed: Editor,
    plan: StoryPlan,
    starts: list[float],
    durs: list[float],
    runs: list[tuple[str, float, float]],
    rate: Rate,
    tracks: list[int],
    w: int,
    h: int,
) -> int:
    """Add `actor` generator clips for every scene's cast. A species that stays on screen
    through consecutive scenes of the same kit is ONE clip pair (walk in, then idle), so it
    does not restart at every scene cut."""
    portrait = h > w
    base = 0.2 if portrait else 0.3
    n = 0
    scene_run = [next(r for r, (_, a, b) in enumerate(runs) if a - 1e-6 <= s0 < b) for s0 in starts]
    stretches: list[tuple[str, int, int, int]] = []  # (species, first, last, slot)
    open_: dict[str, tuple[int, int]] = {}
    for i, scene in enumerate(plan.scenes):
        for sp in list(open_):
            first, slot = open_[sp]
            if sp not in scene.cast or scene_run[i] != scene_run[first]:
                stretches.append((sp, first, i - 1, slot))
                del open_[sp]
        for slot, sp in enumerate(scene.cast[: len(tracks)]):
            if sp not in open_:
                open_[sp] = (i, slot)
    for sp, (first, slot) in open_.items():
        stretches.append((sp, first, len(plan.scenes) - 1, slot))

    def clip(track: int, t0: float, t1: float, props: dict[str, Any], fade_in: bool) -> None:
        nonlocal n
        c0, c1 = _snap(rate, t0), _snap(rate, t1)
        if c1 - c0 < _snap(rate, 0.2):
            return
        cid = ed.add_clip(track, c0, c1 - c0, kind="generator", props={"type": "actor", **props})
        fade = min(_snap(rate, 0.4), (c1 - c0) // 3)
        if fade_in:
            ed.add_keyframe(cid, "opacity", 0, 0.0)
            ed.add_keyframe(cid, "opacity", fade, 1.0)
        ed.add_keyframe(cid, "opacity", c1 - c0 - fade, 1.0)
        ed.add_keyframe(cid, "opacity", c1 - c0, 0.0)
        n += 1

    for sp, first, last, slot in stretches:
        kit = KITS[plan.scenes[first].kit]
        t0, t1 = starts[first], starts[last] + durs[last]
        dur = t1 - t0
        track = tracks[len(tracks) - 1 - slot]  # slot 0 (main character) drawn in front
        hab = castmod.habitat(sp)
        seed = 1 + sorted(actors.SPECIES).index(sp)  # same look in every scene
        common = {"species": sp, "seed": seed}
        if hab == "air":
            size = 0.35 * base / 0.3 * actors.SCALE[sp]
            speed = min(0.2, max(0.06, 1.25 / dur))
            end = min(t1, t0 + 1.3 / speed)
            props = {**common, "action": "fly", "x": -0.1, "y": 0.16 + 0.07 * slot,
                     "size": size, "speed": speed}  # fmt: skip
            clip(track, t0, end, props, False)
            continue
        if hab == "water":
            size = base * actors.SCALE[sp] * (0.9 if sp != "whale" else 0.6)
            speed = min(0.12, max(0.04, 1.2 / dur))
            y = min(0.95, kit.water + (0.2 if kit.ground == 0 else 0.07) + 0.06 * slot)
            props = {**common, "action": "swim", "x": -0.15, "size": size, "speed": speed,
                     "y": y, "surface": kit.water}  # fmt: skip
            clip(track, t0, min(t1, t0 + 1.4 / speed), props, False)
            continue
        ground_bird = sp in actors.BIRDS
        size = max(0.06, base * actors.SCALE[sp] * (2.2 if ground_bird else 1.0))
        y = min(kit.ground, 0.83) - 0.025 * slot
        action = _action(plan.scenes[first].narration)
        height_px = size * h
        if ground_bird:
            per_s = 0.5
        elif action == "run":
            per_s = 2.6 if sp in actors.QUADS else 1.8
        else:
            per_s = 1.1 if sp in actors.QUADS else 0.8
        speed = per_s * height_px / w  # frame widths per second
        x_stop = 0.36 - 0.13 * slot  # left of centre: clear of rivers and the campfire
        props = {**common, "y": y, "size": size}
        if action == "idle":
            clip(track, t0, t1, {**props, "action": "idle", "x": x_stop, "speed": 0.0}, True)
            continue
        x0 = -0.12 - 0.12 * slot
        if action == "run":
            end = min(t1, t0 + (1.25 - x0) / speed)
            clip(track, t0, end, {**props, "action": "run", "x": x0, "speed": speed}, False)
            continue
        t_walk = (x_stop - x0) / speed
        if t_walk >= dur - 0.5:
            clip(track, t0, t1, {**props, "action": "walk", "x": x0, "speed": speed}, False)
            continue
        walk_props = {**props, "action": "walk", "x": x0, "speed": speed}
        c0, c1 = _snap(rate, t0), _snap(rate, t0 + t_walk)
        ed.add_clip(track, c0, c1 - c0, kind="generator", props={"type": "actor", **walk_props})
        n += 1
        idle = {**props, "action": "idle", "x": x_stop, "speed": 0.0, "facing": "right"}
        cid = ed.add_clip(
            track, c1, _snap(rate, t1) - c1, kind="generator", props={"type": "actor", **idle}
        )
        fade = _snap(rate, 0.4)
        ed.add_keyframe(cid, "opacity", _snap(rate, t1) - c1 - fade, 1.0)
        ed.add_keyframe(cid, "opacity", _snap(rate, t1) - c1, 0.0)
        n += 1
    return n


def assemble(
    plan: StoryPlan,
    root: Path,
    *,
    speak: SpeakFn,
    ambience: SoundFn,
    music: SoundFn,
    ffmpeg: Path,
    ffprobe: Path,
    aspect: str = "16:9",
    height: int = 1080,
    voice_licence: str = "TTS",
    progress: Progress | None = None,
    captions: bool = False,
) -> Assembled:
    say = progress or (lambda m: None)
    w, h = frame_size(aspect, height)
    k = h / 1080 if w >= h else w / 1080  # text scale relative to a 1080-wide short side
    rate = Rate(30)
    p = store.create(root, store.ProjectSettings(plan.title[:60] or "story", w, h, rate))
    media = root / "generated"
    media.mkdir(exist_ok=True)
    (root / "plan.json").write_text(
        json.dumps(plan.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    try:
        ed = Editor(p.con)
        n_layers = max(len(KITS[s.kit].layers) for s in plan.scenes)
        vtracks = [p.con.execute("SELECT id FROM track WHERE kind='video'").fetchone()[0]]
        while len(vtracks) < n_layers + MAX_CAST + 2:
            vtracks.append(ed.add_track("video"))
        title_t, sub_t = vtracks[n_layers + MAX_CAST], vtracks[n_layers + MAX_CAST + 1]
        dia_t = p.con.execute("SELECT id FROM track WHERE kind='audio'").fetchone()[0]
        amb_t, mus_t = ed.add_track("audio", "Ambience"), ed.add_track("audio", "Music")
        ed.set_track(amb_t, bus="ambience")
        ed.set_track(mus_t, bus="music")
        for bus in ("ambience", "music"):
            ed.set_bus(bus, duck=True)

        # 1. narration, which sets every scene's timing
        starts: list[float] = []
        durs: list[float] = []
        scene_words: list[list[tuple[str, float, float]]] = []
        t = 0.0
        for i, scene in enumerate(plan.scenes, start=1):
            say(f"voice {i}/{len(plan.scenes)}")
            sp = speak(scene.narration, plan.language, media / f"voice_{i:03d}.wav")
            aid = _import_generated(p, ffmpeg, ffprobe, Path(sp.path), voice_licence)
            dur = max(MIN_SCENE_S, LEAD_S + sp.duration_s + TAIL_S)
            clip_len = _snap_down(rate, _asset_len(p, aid))
            ed.add_clip(dia_t, _snap(rate, t + LEAD_S), clip_len, asset_id=aid)
            timed = [(wd, t + LEAD_S + a, t + LEAD_S + b) for wd, a, b in sp.words]
            scene_words.append(
                subtitles.align_words(
                    scene.narration, timed, t + LEAD_S, t + LEAD_S + sp.duration_s
                )
            )
            starts.append(t)
            durs.append(dur)
            t += dur
        total = t

        # 2. visuals: merge consecutive scenes with the same kit into runs
        runs: list[tuple[str, float, float]] = []  # (kit, start, end)
        for scene, s0, d in zip(plan.scenes, starts, durs, strict=True):
            if runs and runs[-1][0] == scene.kit:
                runs[-1] = (scene.kit, runs[-1][1], s0 + d)
            else:
                runs.append((scene.kit, s0, s0 + d))
        say(f"visuals: {len(plan.scenes)} scenes → {len(runs)} continuous segment(s)")
        prev_base: int | None = None
        for r, (kit_name, r0, r1) in enumerate(runs):
            kit = KITS[kit_name]
            c0, c1 = _snap(rate, r0), _snap(rate, r1)
            for j, (gtype, params) in enumerate(kit.layers):
                cid = ed.add_clip(
                    vtracks[j], c0, c1 - c0, kind="generator", props={"type": gtype, **params}
                )
                if j == 0:
                    if prev_base is not None:
                        ed.add_transition(prev_base, cid, "dissolve", _snap(rate, XFADE_S))
                    prev_base = cid
                    continue
                fade = _snap(rate, OVERLAY_FADE_S)
                if r > 0:
                    ed.add_keyframe(cid, "opacity", 0, 0.0)
                    ed.add_keyframe(cid, "opacity", fade, 1.0)
                if r < len(runs) - 1:
                    ed.add_keyframe(cid, "opacity", c1 - c0 - fade, 1.0)
                    ed.add_keyframe(cid, "opacity", c1 - c0, 0.0)
            # ambience for this segment
            amb = ambience(kit.ambience, r1 - r0 + 1.0, media / f"amb_{r:03d}_{kit.ambience}.wav")
            aid = _import_generated(p, ffmpeg, ffprobe, amb, "procedural (CC0)")
            alen = min(c1 - c0, _snap_down(rate, _asset_len(p, aid)))
            cid = ed.add_clip(amb_t, c0, alen, asset_id=aid, gain_db=-14.0)
            ed.set_props(cid, fade_in=min(1.0, (r1 - r0) / 3), fade_out=min(1.0, (r1 - r0) / 3))

        # 2b. characters: one clip per species per stretch of consecutive scenes it is in
        n_actor_clips = _place_cast(
            ed, plan, starts, durs, runs, rate, vtracks[n_layers : n_layers + MAX_CAST], w, h
        )
        if n_actor_clips:
            say(f"characters: {n_actor_clips} animated clip(s)")

        # 3. music: one bed in the story's dominant mood
        mood = Counter(s.mood for s in plan.scenes).most_common(1)[0][0]
        say(f"music: {mood}, {total:.0f} s")
        mfile = music(mood, total + 1.0, media / f"music_{mood}.wav")
        aid = _import_generated(p, ffmpeg, ffprobe, mfile, "procedural (CC0)")
        mlen = min(_snap(rate, total), _snap_down(rate, _asset_len(p, aid)))
        mid = ed.add_clip(mus_t, 0, mlen, asset_id=aid, gain_db=-8.0)
        ed.set_props(mid, fade_in=1.5, fade_out=2.5)

        # 4. title and on-screen captions
        ed.add_clip(
            title_t,
            _snap(rate, 0.3),
            _snap(rate, min(4.5, durs[0] - 0.3)),
            kind="text",
            props={
                "text": plan.title,
                "size": round(92 * k),
                "anim": "pop",
                "outline": 3.0,
                "shadow": 6.0,
                "y": round(-0.3 * h),
                "max_width": round(0.85 * w),
            },
        )
        for scene, s0, d in list(zip(plan.scenes, starts, durs, strict=True))[1:]:
            if captions and scene.onscreen and d > 2.5:
                ed.add_clip(
                    title_t,
                    _snap(rate, s0 + 0.2),
                    _snap(rate, min(3.0, d - 0.4)),
                    kind="text",
                    props={
                        "text": scene.onscreen,
                        "size": round(56 * k),
                        "anim": "fade",
                        "anim_unit": "word",
                        "outline": 2.0,
                        "y": round(-0.32 * h),
                    },
                )

        # 5. subtitles: burned-in text clips + .srt sidecar
        # cues are built per scene and never outlive their scene
        cues: list[subtitles.Cue] = []
        for sw, s0, d in zip(scene_words, starts, durs, strict=True):
            scene_cues = subtitles.build_cues(sw, max_chars=42 if w >= h else 26)
            for i, c in enumerate(scene_cues):
                # linger 0.25 s for reading, but never into the next cue or the next scene
                limit = scene_cues[i + 1].start if i + 1 < len(scene_cues) else s0 + d - 0.05
                end = min(c.end + 0.25, limit)
                if end > c.start:
                    cues.append(subtitles.Cue(c.start, end, c.text))
        prev_end = 0
        for c in cues:
            c0, c1 = max(_snap(rate, c.start), prev_end), _snap(rate, c.end)
            prev_end = max(prev_end, c1)
            if c1 > c0:
                ed.add_clip(
                    sub_t,
                    c0,
                    c1 - c0,
                    kind="text",
                    props={
                        "text": c.text,
                        "size": round(44 * k),
                        "box": True,
                        "box_opacity": 0.55,
                        "box_padding": round(14 * k),
                        "y": round(0.41 * h),
                        "line_spacing": 1.15,
                    },
                )
        srt = root / "subtitles.srt"
        subtitles.write_srt(cues, srt)
        say(f"assembled: {total:.1f} s, {len(cues)} subtitle cues")
        log.info("assembled", extra={"scenes": len(plan.scenes), "runs": len(runs), "s": total})
    finally:
        p.close()
    return Assembled(root, srt, total, len(plan.scenes), len(runs))
