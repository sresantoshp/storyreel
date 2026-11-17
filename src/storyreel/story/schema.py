"""Story plan data model + the JSON schema the local writer is forced to follow.

The writer's output is constrained by llama.cpp's JSON-schema grammar (exact scene count,
allowed kits and moods, bounded lengths), then validated again here: the grammar guarantees
shape, this module guarantees meaning (non-empty narration, known kit, sane lengths).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from storyreel.errors import StoryReelError

MOODS: tuple[str, ...] = ("calm", "happy", "sad", "tense", "mysterious", "epic")
# Narration speed used to size scenes before speech exists (measured TTS will replace it).
WORDS_PER_SECOND = {"en": 2.5, "hi": 2.2}
SCENE_SECONDS = 10.0  # target narration per scene
MAX_SCENES_PER_CALL = 12  # longer stories are written section by section


class PlanError(StoryReelError):
    pass


@dataclass
class Scene:
    narration: str
    kit: str
    mood: str
    onscreen: str = ""  # optional short headline shown on screen


@dataclass
class StoryPlan:
    title: str
    language: str
    scenes: list[Scene] = field(default_factory=list)
    source: str = "rules"  # llm:<model> | rules | script
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> StoryPlan:
        return StoryPlan(
            title=str(d["title"]),
            language=str(d["language"]),
            scenes=[Scene(**s) for s in d["scenes"]],
            source=str(d.get("source", "rules")),
            warnings=list(d.get("warnings", [])),
        )


# Measured: the local 1.5B writer averages ~15 words (~6.5 s spoken) per scene even when
# asked for ~25, so scenes are counted at 7 s each to land near the requested length.
EFFECTIVE_SCENE_SECONDS = 7.0


def scene_count(length_s: float) -> int:
    return max(1, min(500, math.ceil(length_s / EFFECTIVE_SCENE_SECONDS)))


def narration_chars(lang: str) -> int:
    """Upper bound on characters of narration per scene (≈ SCENE_SECONDS of speech)."""
    words = WORDS_PER_SECOND.get(lang, 2.3) * SCENE_SECONDS
    return int(words * (7 if lang == "en" else 8)) + 40


def story_schema(n_scenes: int, kits: list[str], lang: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 2, "maxLength": 70},
            "scenes": {
                "type": "array",
                "minItems": n_scenes,
                "maxItems": n_scenes,
                "items": {
                    "type": "object",
                    "properties": {
                        "narration": {
                            "type": "string",
                            "minLength": 10,
                            # generous: a grammar limit cuts mid-word; validate() trims
                            "maxLength": 2 * narration_chars(lang),
                        },
                        "kit": {"enum": kits},
                        "mood": {"enum": list(MOODS)},
                        "onscreen": {"type": "string", "maxLength": 40},
                    },
                    "required": ["narration", "kit", "mood", "onscreen"],
                },
            },
        },
        "required": ["title", "scenes"],
    }


def outline_schema(n_sections: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 2, "maxLength": 70},
            "sections": {
                "type": "array",
                "minItems": n_sections,
                "maxItems": n_sections,
                "items": {"type": "string", "minLength": 10, "maxLength": 300},
            },
        },
        "required": ["title", "sections"],
    }


def trim_to_sentence(text: str, limit: int) -> str:
    """Text that hit the length limit may end mid-word: cut back to the last full sentence."""
    if len(text) < limit - 2 or text.endswith((".", "!", "?", "।", "॥")):
        return text
    ends = [text.rfind(p) for p in (". ", "! ", "? ", "। ", "॥ ", ".", "।")]
    cut = max(ends)
    return text[: cut + 1].strip() if cut > len(text) // 3 else text


def validate(raw: dict[str, Any], *, lang: str, kits: list[str], source: str) -> StoryPlan:
    try:
        title = str(raw["title"]).strip()
        scenes_raw = list(raw["scenes"])
    except (KeyError, TypeError) as exc:
        raise PlanError(f"Story output is missing {exc}.") from exc
    scenes: list[Scene] = []
    for i, s in enumerate(scenes_raw, start=1):
        narration = trim_to_sentence(str(s.get("narration", "")).strip(), 2 * narration_chars(lang))
        if len(narration) < 3:
            raise PlanError(f"Scene {i} has no narration.")
        kit = str(s.get("kit", ""))
        if kit not in kits:
            raise PlanError(f"Scene {i} uses unknown visual kit {kit!r}.")
        mood = str(s.get("mood", "calm"))
        if mood not in MOODS:
            mood = "calm"
        scenes.append(Scene(narration, kit, mood, str(s.get("onscreen", "")).strip()[:40]))
    if not scenes:
        raise PlanError("The story has no scenes.")
    return StoryPlan(title or scenes[0].narration[:40], lang, scenes, source)
