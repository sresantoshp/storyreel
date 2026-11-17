"""Visual kits: named, offline scene recipes built from the procedural generators.

A kit is what a scene looks and sounds like when no photo/footage is supplied: generator
layers bottom-to-top, an ambience loop, a default mood. The writer picks a kit per scene
(constrained to these names); `refine_kits` then corrects lazy picks with keywords from the
narration itself (English + Hindi), because small local models tend to repeat one kit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from storyreel.story.schema import MOODS, Scene


@dataclass(frozen=True)
class Kit:
    name: str
    description: str  # shown to the writer
    layers: tuple[tuple[str, dict[str, Any]], ...]  # (generator type, params) bottom → top
    ambience: str  # procedural ambience kind
    mood: str
    keywords: tuple[str, ...] = field(default=())


KITS: dict[str, Kit] = {
    k.name: k
    for k in (
        Kit(
            "night_sky",
            "a calm night sky full of stars",
            (
                ("gradient", {"top": "#05060f", "bottom": "#1b2440"}),
                ("starfield", {"count": 1200, "speed": 0.05}),
            ),
            "night",
            "calm",
            (
                "night",
                "star",
                "stars",
                "moon",
                "dark",
                "dream",
                "sky",
                "sleep",
                "रात",
                "तारे",
                "तारा",
                "चाँद",
                "चांद",
                "आसमान",
                "सपना",
                "नींद",
            ),
        ),
        Kit(
            "storm",
            "a dark rainy storm with fog",
            (
                ("gradient", {"top": "#232a35", "bottom": "#4a5566"}),
                ("fog", {"density": 0.35, "color": "#9aa3ae"}),
                ("rain", {"count": 1100}),
            ),
            "rain",
            "tense",
            (
                "rain",
                "storm",
                "thunder",
                "cloud",
                "clouds",
                "wet",
                "lightning",
                "flood",
                "बारिश",
                "तूफ़ान",
                "तूफान",
                "बादल",
                "बिजली",
                "बाढ़",
            ),
        ),
        Kit(
            "snow_valley",
            "snowy mountains, a river valley and falling snow",
            (
                ("gradient", {"top": "#2f6fd0", "bottom": "#bfe3f7"}),
                ("mountains", {}),
                ("river", {}),
                ("birds", {"count": 6}),
                ("snow", {"count": 450, "opacity": 0.85}),
            ),
            "wind",
            "calm",
            (
                "snow",
                "winter",
                "cold",
                "ice",
                "frozen",
                "freezing",
                "बर्फ",
                "बर्फ़",
                "सर्दी",
                "ठंड",
                "जाड़ा",
            ),
        ),
        Kit(
            "sunny_valley",
            "a sunny green valley with mountains, a river and birds",
            (
                ("gradient", {"top": "#2f80ed", "bottom": "#cde9fb"}),
                ("mountains", {"snowcap": 0.1, "near": "#4a7a3c"}),
                ("river", {}),
                ("birds", {"count": 9}),
            ),
            "birds",
            "happy",
            (
                "valley",
                "river",
                "morning",
                "sun",
                "sunny",
                "village",
                "field",
                "bird",
                "birds",
                "mountain",
                "mountains",
                "journey",
                "walk",
                "spring",
                "farm",
                "घाटी",
                "नदी",
                "सुबह",
                "सूरज",
                "गाँव",
                "गांव",
                "खेत",
                "पक्षी",
                "चिड़िया",
                "पहाड़",
                "पहाड़ों",
                "यात्रा",
            ),
        ),
        Kit(
            "campfire",
            "a warm campfire glowing in the dark with rising embers",
            (
                ("gradient", {"top": "#05060f", "bottom": "#3a1a08"}),
                ("fire", {"height": 0.42}),
                ("particles", {"color": "#ffb347", "count": 90, "speed": 60.0}),
            ),
            "fire",
            "calm",
            (
                "fire",
                "campfire",
                "warm",
                "flame",
                "flames",
                "camp",
                "spark",
                "embers",
                "hearth",
                "आग",
                "अलाव",
                "गर्म",
                "लौ",
                "चिंगारी",
            ),
        ),
        Kit(
            "misty_forest",
            "misty green hills with drifting fog, mysterious",
            (
                ("gradient", {"top": "#1d2b24", "bottom": "#5d7a63"}),
                ("mountains", {"far": "#6f8c78", "near": "#20352a", "snowcap": 0.0}),
                ("fog", {"density": 0.5, "color": "#c9d6cc"}),
                ("particles", {"color": "#e8f5c8", "count": 40, "size": 6.0, "opacity": 0.35}),
            ),
            "wind",
            "mysterious",
            (
                "forest",
                "tree",
                "trees",
                "mist",
                "misty",
                "fog",
                "mystery",
                "jungle",
                "secret",
                "strange",
                "जंगल",
                "पेड़",
                "धुंध",
                "कोहरा",
                "रहस्य",
                "अजीब",
            ),
        ),
    )
}

_MOOD_WORDS: dict[str, tuple[str, ...]] = {
    "happy": (
        "happy",
        "joy",
        "smile",
        "laugh",
        "friend",
        "friends",
        "celebrate",
        "fun",
        "खुश",
        "खुशी",
        "मुस्कान",
        "हँसी",
        "दोस्त",
        "मज़ा",
    ),
    "sad": (
        "sad",
        "alone",
        "lonely",
        "cry",
        "tears",
        "lost",
        "miss",
        "goodbye",
        "उदास",
        "अकेला",
        "अकेली",
        "रोना",
        "आँसू",
        "खो",
    ),
    "tense": (
        "danger",
        "fear",
        "afraid",
        "run",
        "chase",
        "storm",
        "dark",
        "scream",
        "डर",
        "खतरा",
        "भागो",
        "भाग",
        "तूफ़ान",
    ),
    "mysterious": (
        "secret",
        "strange",
        "mystery",
        "hidden",
        "unknown",
        "whisper",
        "रहस्य",
        "अजीब",
        "छिपा",
        "अनजान",
    ),
    "epic": (
        "hero",
        "battle",
        "victory",
        "brave",
        "legend",
        "kingdom",
        "war",
        "वीर",
        "जीत",
        "बहादुर",
        "युद्ध",
        "राजा",
    ),
}

_WORD = re.compile(r"[\wऀ-෿]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def _hits(tokens: set[str], text: str, words: tuple[str, ...]) -> int:
    # Latin words match whole tokens; Indic words match as substrings because Hindi
    # postpositions/inflections attach to the stem (e.g. पहाड़ों contains पहाड़).
    return sum(1 for w in words if (w in tokens) if w.isascii()) + sum(
        1 for w in words if not w.isascii() and w in text
    )


def kit_scores(text: str) -> dict[str, int]:
    tokens = _tokens(text)
    return {name: _hits(tokens, text, kit.keywords) for name, kit in KITS.items()}


def guess_kit(text: str, fallback: str = "sunny_valley") -> str:
    scores = kit_scores(text)
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else fallback


def guess_mood(text: str, fallback: str = "calm") -> str:
    tokens = _tokens(text)
    scores = {m: _hits(tokens, text, words) for m, words in _MOOD_WORDS.items()}
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else fallback


def refine_kits(scenes: list[Scene]) -> int:
    """Correct kit picks the narration contradicts. Returns how many scenes changed.

    A scene keeps the writer's kit when its own words support it (or nothing better is
    supported); otherwise it moves to the kit its narration names most strongly.
    """
    changed = 0
    for s in scenes:
        scores = kit_scores(s.narration)
        best, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score > 0 and scores.get(s.kit, 0) < best_score:
            s.kit = best
            changed += 1
        if s.mood not in MOODS:
            s.mood = KITS[s.kit].mood
    return changed


def describe_for_writer() -> str:
    return "\n".join(f"- {k.name}: {k.description}" for k in KITS.values())
