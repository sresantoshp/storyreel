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
    ground: float = 0.84  # where walking characters stand (fraction of height); 0 = no land
    water: float = 0.0  # sea surface line for swimmers; 0 = no water


def _w(text: str) -> tuple[str, ...]:
    return tuple(text.split())


SKY: dict[str, Any] = {"clouds": 4}
KITS: dict[str, Kit] = {
    k.name: k
    for k in (
        Kit(
            "night_sky",
            "a calm night under the moon and stars, quiet hills",
            (
                ("sky", {"top": "#04050d", "bottom": "#1b2440", "body": "moon",
                         "body_color": "#f4f1e4", "glow": 0.35, "clouds": 0, "stars": 900,
                         "body_y": 0.16, "body_size": 0.05}),
                ("hills", {"far": "#1c2638", "near": "#0c121c", "layers": 2, "height": 0.16,
                           "horizon": 0.8, "houses": 3, "lit": 1.0, "wall": "#3a3f4c",
                           "roof": "#272a33"}),
                ("particles", {"color": "#d8ff9a", "count": 25, "size": 5.0, "opacity": 0.5}),
            ),
            "night", "calm",
            _w("night star stars moon dark dream sleep midnight owl firefly fireflies "
               "रात तारे तारा चाँद चांद आसमान सपना नींद उल्लू"),
            ground=0.84,
        ),
        Kit(
            "storm",
            "a dark thunderstorm with heavy rain over dark hills and trees",
            (
                ("gradient", {"top": "#1d232c", "bottom": "#4a5566"}),
                ("hills", {"far": "#3a4452", "near": "#1f262e", "layers": 2, "height": 0.2,
                           "horizon": 0.8}),
                ("trees", {"style": "pine", "color": "#1c2a24", "haze": "#4a5566",
                           "count": 7, "base": 0.74, "size": 0.32}),
                ("fog", {"density": 0.35, "color": "#9aa3ae"}),
                ("rain", {"count": 1100}),
            ),
            "thunderstorm", "tense",
            _w("rain storm thunder cloud clouds wet lightning flood monsoon umbrella "
               "बारिश तूफ़ान तूफान बादल बिजली बाढ़ मानसून"),
            ground=0.84,
        ),
        Kit(
            "snow_valley",
            "snowy mountains, a river valley, pine trees and falling snow",
            (
                ("sky", {"top": "#2f6fd0", "bottom": "#bfe3f7", "clouds": 3}),
                ("mountains", {}),
                ("river", {"meander": -0.22, "width_near": 0.38}),
                ("trees", {"style": "pine", "color": "#2a5a46", "haze": "#cfe3f0",
                           "count": 5, "base": 0.66, "size": 0.24}),
                ("birds", {"count": 5}),
                ("snow", {"count": 450, "opacity": 0.85}),
            ),
            "wind", "calm",
            _w("snow snowy winter cold freezing snowflakes बर्फ बर्फ़ सर्दी ठंड जाड़ा"),
            ground=0.84,
        ),
        Kit(
            "sunny_valley",
            "a sunny green valley with mountains, a river, trees and birds",
            (
                ("sky", SKY),
                ("mountains", {"snowcap": 0.1, "near": "#4a7a3c"}),
                ("river", {"meander": -0.22, "width_near": 0.38}),
                ("trees", {"style": "round", "count": 5, "base": 0.68, "size": 0.26}),
                ("birds", {"count": 8}),
                ("grass", {"flowers": 14}),
            ),
            "birds", "happy",
            _w("valley river morning sun sunny meadow bird birds mountain mountains journey "
               "spring घाटी नदी सुबह सूरज पक्षी चिड़िया पहाड़ पहाड़ों यात्रा"),
            ground=0.84,
        ),
        Kit(
            "campfire",
            "a warm campfire glowing at night among dark trees, rising embers",
            (
                ("gradient", {"top": "#05060f", "bottom": "#3a1a08"}),
                ("trees", {"style": "pine", "color": "#0c0e0d", "haze": "#20120a",
                           "count": 6, "base": 0.7, "size": 0.4}),
                ("hills", {"far": "#1a120c", "near": "#120c08", "layers": 2,
                           "height": 0.06, "horizon": 0.84}),
                ("fire", {"height": 0.3, "width": 0.16, "x": 0.66, "base": 0.88}),
                ("particles", {"color": "#ffb347", "count": 60, "speed": 60.0,
                               "size": 5.0}),
            ),
            "fire", "calm",
            _w("fire campfire warm flame flames camp spark embers hearth cozy "
               "आग अलाव गर्म लौ चिंगारी"),
            ground=0.86,
        ),
        Kit(
            "misty_forest",
            "a misty, mysterious pine forest with drifting fog",
            (
                ("gradient", {"top": "#1d2b24", "bottom": "#5d7a63"}),
                ("mountains", {"far": "#6f8c78", "near": "#20352a", "snowcap": 0.0}),
                ("trees", {"style": "pine", "color": "#1d3327", "haze": "#6f8c78",
                           "rows": 3, "count": 9, "side": "spread", "base": 0.62,
                           "size": 0.34}),
                ("fog", {"density": 0.5, "color": "#c9d6cc"}),
                ("particles", {"color": "#e8f5c8", "count": 40, "size": 6.0, "opacity": 0.35}),
            ),
            "wind", "mysterious",
            _w("mist misty fog foggy mystery secret strange haunted shadow shadows "
               "धुंध कोहरा रहस्य अजीब"),
            ground=0.86,
        ),
        Kit(
            "forest",
            "a bright green forest with tall trees, sunlight and flowers",
            (
                ("sky", {"clouds": 3, "body_x": 0.2}),
                ("hills", {"far": "#9cc59a", "near": "#4f8a45", "horizon": 0.74,
                           "height": 0.18, "layers": 2}),
                ("trees", {"style": "mixed", "rows": 3, "count": 9, "side": "spread",
                           "base": 0.66, "size": 0.38, "color": "#2f6b3a"}),
                ("grass", {"flowers": 18}),
            ),
            "forest", "happy",
            _w("forest woods wood tree trees deer squirrel leaves path oak pine "
               "वन पेड़ पेड़ों हिरण पत्ते"),
            ground=0.86,
        ),
        Kit(
            "jungle",
            "a dense tropical jungle with big leaves, vines and wild animals",
            (
                ("gradient", {"top": "#3f7a52", "bottom": "#9fc98e"}),
                ("hills", {"far": "#5f9a62", "near": "#2c5f30", "horizon": 0.8,
                           "height": 0.18, "layers": 2}),
                ("trees", {"style": "jungle", "rows": 2, "count": 8, "side": "spread",
                           "base": 0.78, "size": 0.55, "color": "#2b6531",
                           "haze": "#9cc59a"}),
                ("grass", {"height": 0.09, "color": "#3c7a2e", "flowers": 8,
                           "flower_colors": "#ff5a36,#ffb020,#ff3d7f"}),
                ("particles", {"color": "#f2ffb0", "count": 30, "size": 5.0, "opacity": 0.35}),
            ),
            "jungle", "mysterious",
            _w("jungle rainforest tropical tiger tigers monkey monkeys parrot parrots "
               "elephant elephants vines wild जंगल बाघ बंदर तोता हाथी"),
            ground=0.88,
        ),
        Kit(
            "beach",
            "a sunny beach with sand, gentle waves and palm trees",
            (
                ("sky", SKY),
                ("sea", {"horizon": 0.55}),
                ("beach", {"shore": 0.7}),
                ("trees", {"style": "palm", "count": 2, "rows": 1, "side": "right",
                           "base": 0.98, "size": 0.62}),
                ("birds", {"count": 4, "color": "#5b6670"}),
            ),
            "beach", "happy",
            _w("beach sand sandy shore coast seaside shell shells crab holiday "
               "तट किनारा किनारे रेत"),
            ground=0.86, water=0.55,
        ),
        Kit(
            "ocean",
            "the open blue sea with waves, under a wide sky",
            (
                ("sky", {"clouds": 3}),
                ("sea", {"horizon": 0.45, "waves": 150}),
                ("birds", {"count": 3, "color": "#5b6670"}),
            ),
            "ocean", "calm",
            _w("ocean sea seas waves boat ship sail sailor island dolphin dolphins whale "
               "whales fish underwater समुद्र सागर लहरें नाव जहाज़ मछली"),
            ground=0.0, water=0.45,
        ),
        Kit(
            "hill_station",
            "a cool hill station: green hills with cottages, tea gardens, pines and peaks",
            (
                ("sky", {"clouds": 6}),
                ("mountains", {"far": "#b8c9dc", "near": "#6d8fa8", "snowcap": 0.3,
                               "horizon": 0.58, "height": 0.3}),
                ("hills", {"houses": 7, "tea": 0.6, "horizon": 0.74}),
                ("trees", {"style": "pine", "count": 6, "base": 0.78, "size": 0.32}),
                ("fog", {"density": 0.15, "color": "#eef3f6"}),
            ),
            "hill_station", "calm",
            _w("hill hills station tea cottage cottages shimla manali darjeeling ooty munnar "
               "mussoorie toy train hillside slopes पहाड़ी चाय"),
            ground=0.86,
        ),
        Kit(
            "ice_land",
            "a frozen land of ice and icebergs under a pale sky",
            (
                ("sky", {"top": "#8fb8d8", "bottom": "#e8f2f8", "clouds": 2,
                         "body_color": "#fffbe8", "glow": 0.3}),
                ("ice", {}),
                ("snow", {"count": 220, "opacity": 0.8}),
            ),
            "ice", "calm",
            _w("ice icy glacier glaciers arctic antarctic polar iceberg icebergs frozen "
               "penguin penguins igloo बर्फीला हिमनद"),
            ground=0.84,
        ),
        Kit(
            "village",
            "a peaceful village with cottages, fields, trees and flowers",
            (
                ("sky", SKY),
                ("hills", {"layers": 2, "far": "#b9d7a0", "near": "#7fb055",
                           "height": 0.15, "horizon": 0.7}),
                ("houses", {"count": 5, "base": 0.72}),
                ("trees", {"style": "round", "count": 5, "base": 0.74, "size": 0.3}),
                ("grass", {"flowers": 20}),
            ),
            "village", "happy",
            _w("village farm farmer field fields cow cows goat goats hen hens rooster "
               "house home hut barn harvest गाँव गांव खेत किसान गाय बकरी मुर्गी घर"),
            ground=0.86,
        ),
        Kit(
            "desert",
            "a hot desert of golden sand dunes under a blazing sun",
            (
                ("sky", {"top": "#3f8fe0", "bottom": "#f6e2b0", "clouds": 1,
                         "body_size": 0.08, "glow": 0.7}),
                ("hills", {"far": "#ecd09a", "near": "#d4a25c", "layers": 3,
                           "height": 0.16, "horizon": 0.76}),
            ),
            "wind", "epic",
            _w("desert dune dunes sand camel camels oasis thirst hot "
               "रेगिस्तान मरुस्थल ऊँट"),
            ground=0.86,
        ),
    )
}  # fmt: skip

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
