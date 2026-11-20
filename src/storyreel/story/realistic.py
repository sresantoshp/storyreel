"""The realistic style: real footage cut like a film, a film grade and trailer sound.

Separate from the animation style (cartoon kits + puppets): a realistic video never falls
back to drawn scenery. If no footage is found for a scene, making stops with a fix.

Picture (bottom → top): footage shots · grade (adjustment layer) · letterbox bars ·
fade-from/to-black. Shots last ~3.5-6 s; each has a slow push-in (and a small drift) so
even static stock clips feel like a moving camera. Scene cuts are hard cuts, as in films.

Sound: a broadcast voice chain on the dialogue bus (rumble cut, de-mud, presence, air,
compression, de-esser); whooshes peaking on every scene cut; a braam under the title;
a riser into the final scene and an impact on its first frame; hall reverb on the
effects bus; glue compression on the master (the renderer then sets -14 LUFS).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from storyreel.errors import StoryReelError
from storyreel.media import stock
from storyreel.story import cast as castmod
from storyreel.story.schema import Scene

# Kit → what a camera would see there (best first). Kits stay the writer's vocabulary.
KIT_QUERIES: dict[str, tuple[str, ...]] = {
    "night_sky": ("night sky stars", "moon night", "milky way"),
    "storm": ("thunderstorm lightning", "storm clouds rain", "heavy rain"),
    "snow_valley": ("snowy mountain valley", "snowfall forest", "snow landscape"),
    "sunny_valley": ("green valley sunny", "meadow sunlight", "countryside landscape"),
    "campfire": ("campfire night", "bonfire flames", "fire burning"),
    "misty_forest": ("misty forest", "fog trees", "foggy woods"),
    "forest": ("forest sunlight", "forest trees", "woodland"),
    "jungle": ("tropical jungle", "rainforest", "jungle trees"),
    "beach": ("tropical beach", "beach waves sand", "sea shore"),
    "ocean": ("ocean waves", "underwater ocean", "sea aerial"),
    "hill_station": ("tea plantation hills", "mountain village clouds", "green hills mist"),
    "ice_land": ("arctic ice", "glacier", "iceberg"),
    "village": ("indian village", "village life", "rural village"),
    "desert": ("desert dunes", "sand dunes sunset", "desert landscape"),
}
KIT_PLACE = {
    "night_sky": "night sky", "storm": "rain", "snow_valley": "snow", "sunny_valley": "meadow",
    "campfire": "campfire", "misty_forest": "forest", "forest": "forest", "jungle": "jungle",
    "beach": "beach", "ocean": "ocean", "hill_station": "mountains", "ice_land": "snow",
    "village": "village", "desert": "desert",
}  # fmt: skip
# Things worth filming that the narration may name (English). First match wins.
VISUAL_WORDS = (
    "waterfall", "river", "lake", "mountain", "volcano", "sunrise", "sunset", "moon",
    "stars", "lightning", "rain", "snow", "fog", "flowers", "garden", "field", "farm",
    "temple", "market", "city", "street", "road", "train", "boat", "ship", "bridge",
    "castle", "cave", "island", "waves", "fire", "candle", "lantern", "house", "school",
    "children", "family", "crowd", "kitchen", "tea", "bird", "birds", "butterfly",
)  # fmt: skip
_WORD = re.compile(r"[a-z]+")
PEOPLE = frozenset({"man", "woman", "boy", "girl", "grandpa", "grandma"})
SHOT_MIN_S, SHOT_MAX_S = 3.5, 6.0
PUSH_IN = 1.07  # zoom over one shot
LETTERBOX = 2.39  # cinema scope


def scene_queries(scene: Scene) -> list[str]:
    """Searches for one scene, most specific first; the kit's generic shots last."""
    kit = scene.kit if scene.kit in KIT_QUERIES else "forest"
    place = KIT_PLACE[kit]
    cast = scene.cast or castmod.detect(scene.narration)
    words = set(_WORD.findall(scene.narration.lower()))
    thing = next((w for w in VISUAL_WORDS if w in words), None)
    animals = [c for c in cast if c not in PEOPLE]
    people = [c for c in cast if c in PEOPLE]
    kit_q = list(KIT_QUERIES[kit])
    # order: the animal, the named thing, the scene's main shot, then people (a stock
    # "girl at night" can be anything; "campfire night" is what the scene is about)
    out: list[str] = []
    if animals:
        out += [f"{animals[0]} {place}", animals[0]]
    if thing:
        out.append(f"{thing} {place}" if thing not in place else thing)
    out.append(kit_q[0])
    if people:
        out.append(f"{people[0]} {place}")
    out += kit_q[1:]
    return list(dict.fromkeys(out))


def shot_lengths(seconds: float) -> list[float]:
    """Split a scene into film-length shots (equal, each 3.5-6 s when possible)."""
    n = max(1, math.ceil(seconds / SHOT_MAX_S))
    if n > 1 and seconds / n < SHOT_MIN_S:
        n -= 1
    return [seconds / n] * n


def letterbox_bar(w: int, h: int) -> int:
    """Height of each black bar for a 2.39:1 picture; 0 for portrait/square frames."""
    if w <= h:
        return 0
    return max(0, round((h - w / LETTERBOX) / 2))


@dataclass(frozen=True)
class Shot:
    scene: int
    start_s: float
    length_s: float
    clip: stock.Clip


FindFn = Callable[[list[str], str, float, int, set[str]], list[stock.Clip]]
# (queries, narration to match, min seconds, count, keys already used) -> clips


def plan_shots(
    scenes: list[Scene], starts: list[float], durs: list[float], find: FindFn
) -> list[Shot]:
    """Footage for every shot of every scene, never the same clip twice in one story
    (unless a scene has nothing else). Raises if a scene has no footage at all."""
    used: set[str] = set()
    shots: list[Shot] = []
    for i, (scene, s0, d) in enumerate(zip(scenes, starts, durs, strict=True)):
        lengths = shot_lengths(d)
        need = max(lengths) + 1.0  # skip the first second of each clip (often a shaky start)
        describe = scene.narration if scene.narration.isascii() else ""  # CLIP: English
        clips = find(scene_queries(scene), describe, need, len(lengths), used)
        if not clips:  # reuse rather than fail: still real footage
            clips = find(scene_queries(scene), describe, need, len(lengths), set())
        if not clips:
            raise StoryReelError(
                f"No footage found for scene {i + 1} ({scene.kit}: {scene.narration[:50]!r}).",
                fix="Set a free PEXELS_API_KEY or PIXABAY_API_KEY, or pass --footage-dir with "
                "your own clips (named by what they show, e.g. forest_fox.mp4). "
                "The realistic style does not fall back to animation.",
            )
        t = s0
        for j, ln in enumerate(lengths):
            c = clips[j % len(clips)]
            used.add(c.key)
            shots.append(Shot(i, t, ln, c))
            t += ln
    return shots


def credits(shots: list[Shot]) -> str:
    """Footage credits for the video description (not required, but appreciated)."""
    seen: dict[str, stock.Clip] = {}
    for s in shots:
        seen.setdefault(s.clip.key, s.clip)
    lines = ["Footage:"]
    for c in seen.values():
        who = f" by {c.author}" if c.author else ""
        where = f" — {c.page_url}" if c.page_url.startswith("http") else ""
        lines.append(f"- {c.key.split(':')[0].title()}{who}{where} ({c.licence})")
    return "\n".join(lines) + "\n"


def credits_path(root: Path) -> Path:
    return root / "footage_credits.txt"


# -- grade and sound settings (applied by story.assemble) ------------------------------
def film_grade(k: float) -> list[tuple[str, dict[str, object]]]:
    """Filmic look: soft contrast with lifted blacks, warm highlights/cool shadows,
    a little halation, vignette and fine grain. k = frame short side / 1080."""
    return [
        ("color", {"contrast": 1.08, "saturation": 0.9, "temperature": 0.03, "lift_b": 0.025,
                   "lift_g": 0.008, "gain_r": 1.04, "gain_b": 0.95}),
        ("curves", {"master": [[0, 0.03], [0.25, 0.21], [0.75, 0.81], [1, 0.97]]}),
        ("glow", {"radius": round(16 * k, 1), "strength": 0.22, "threshold": 0.78}),
        ("vignette", {"amount": 0.3, "softness": 0.6}),
        ("grain", {"amount": 0.03, "size": max(1.0, round(1.5 * k, 1))}),
    ]  # fmt: skip


VOICE_CHAIN: list[tuple[str, dict[str, float]]] = [
    ("highpass", {"freq": 85.0}),
    ("eq", {"freq": 250.0, "gain": -2.5, "q": 1.0}),  # mud
    ("eq", {"freq": 3200.0, "gain": 2.5, "q": 1.0}),  # presence
    ("eq", {"freq": 11000.0, "gain": 2.0, "q": 0.7}),  # air
    ("compressor", {"threshold": -22.0, "ratio": 3.0, "attack": 6.0, "release": 120.0,
                    "makeup": 3.0}),
    ("deesser", {"intensity": 0.4}),
]  # fmt: skip
FX_CHAIN: list[tuple[str, dict[str, object]]] = [("reverb", {"room": "hall", "mix": 0.2})]
MASTER_CHAIN: list[tuple[str, dict[str, float]]] = [
    ("compressor", {"threshold": -16.0, "ratio": 2.0, "attack": 25.0, "release": 250.0,
                    "makeup": 1.0}),
]  # fmt: skip


def shot_match(
    stats: list[tuple[list[float], list[float]]],
) -> list[dict[str, list[float] | float]]:
    """Colour-match parameters per shot (Lab mean/std from a frame of each): every shot
    moves part of the way towards the story's average look, so cuts between clips from
    different cameras stop jumping, while a campfire stays warmer than the sea."""
    if not stats:
        return []
    ref_m = np.median(np.array([m for m, _ in stats]), axis=0)
    ref_s = np.median(np.array([s for _, s in stats]), axis=0)
    pull_m = np.array([0.3, 0.5, 0.5])  # L (exposure) less than a/b (tint)
    out: list[dict[str, list[float] | float]] = []
    for m, sd in stats:
        mm, ss = np.array(m), np.array(sd)
        out.append({
            "src_mean": [round(float(x), 4) for x in mm],
            "src_std": [round(float(x), 4) for x in ss],
            "dst_mean": [round(float(x), 4) for x in mm + pull_m * (ref_m - mm)],
            "dst_std": [round(float(x), 4) for x in ss + 0.5 * (ref_s - ss)],
            "mix": 1.0,
        })  # fmt: skip
    return out


SFX_GAIN_DB = {"whoosh": -13.0, "impact": -7.0, "riser": -12.0, "braam": -9.0}
