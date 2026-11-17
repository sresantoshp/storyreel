"""Who appears on screen: characters found in the story's own words.

The writer models are small and unreliable at structured "character lists", so the cast is
read from the narration (English and Hindi words) by code:

* the story's main characters (named in the idea, or mentioned in several scenes) appear in
  every scene, so the fox of "a fox explores..." is never missing;
* any other animal or person a scene mentions joins that scene;
* sea creatures need water: a scene that mentions a dolphin but uses a dry kit moves to the
  ocean (or the beach when it talks about the shore); land animals are left out of open sea.

Only characters the renderer can draw (render.actors.SPECIES) are ever chosen.
"""

from __future__ import annotations

import re
from collections import Counter

from storyreel.render.actors import BIRDS, SPECIES, SWIMMERS
from storyreel.story.kits import KITS
from storyreel.story.schema import Scene

MAX_CAST = 4  # 3 from the scene itself + 1 for an idea character the writer left out

_EN: dict[str, tuple[str, ...]] = {
    "fox": ("fox", "foxes", "vixen"),
    "dog": ("dog", "dogs", "puppy", "puppies", "pup"),
    "wolf": ("wolf", "wolves"),
    "cat": ("cat", "cats", "kitten", "kittens", "kitty"),
    "deer": ("deer", "fawn", "stag", "doe"),
    "horse": ("horse", "horses", "pony", "ponies", "foal", "stallion", "mare"),
    "cow": ("cow", "cows", "calf", "cattle", "bull", "ox", "oxen", "buffalo"),
    "goat": ("goat", "goats", "kid goat"),
    "sheep": ("sheep", "lamb", "lambs", "ram"),
    "tiger": ("tiger", "tigers", "tigress"),
    "lion": ("lion", "lions", "lioness"),
    "elephant": ("elephant", "elephants"),
    "bear": ("bear", "bears"),
    "rabbit": ("rabbit", "rabbits", "bunny", "bunnies", "hare", "hares"),
    "monkey": ("monkey", "monkeys", "ape", "apes", "chimp"),
    "man": ("man", "men", "father", "dad", "farmer", "king", "fisherman", "hunter",
            "traveler", "traveller", "uncle", "brother", "prince", "soldier", "teacher"),
    "woman": ("woman", "women", "mother", "mom", "mum", "queen", "lady", "aunt", "sister",
              "princess", "wife"),
    "boy": ("boy", "boys", "son", "child", "kid", "kids", "children", "lad"),
    "girl": ("girl", "girls", "daughter"),
    "grandpa": ("grandpa", "grandfather", "grandad", "old man", "sage", "monk", "hermit"),
    "grandma": ("grandma", "grandmother", "granny", "old woman"),
    "eagle": ("eagle", "eagles", "hawk", "hawks", "falcon", "vulture", "kite bird"),
    "parrot": ("parrot", "parrots", "parakeet"),
    "crow": ("crow", "crows", "raven", "ravens"),
    "sparrow": ("sparrow", "sparrows", "robin", "songbird"),
    "seagull": ("seagull", "seagulls", "gull", "gulls"),
    "owl": ("owl", "owls"),
    "peacock": ("peacock", "peacocks", "peahen"),
    "hen": ("hen", "hens", "chicken", "chickens", "rooster", "chick", "chicks"),
    "duck": ("duck", "ducks", "duckling", "ducklings", "goose", "swan"),
    "fish": ("fish", "fishes", "goldfish"),
    "dolphin": ("dolphin", "dolphins"),
    "whale": ("whale", "whales"),
    "turtle": ("turtle", "turtles", "tortoise"),
}  # fmt: skip

# Hindi stems match as substrings (postpositions attach to the word). Longer stems are
# checked first and consumed, so भेड़िया (wolf) does not also count as भेड़ (sheep).
_HI: dict[str, tuple[str, ...]] = {
    "fox": ("लोमड़ी", "लोमड़"),
    "dog": ("कुत्त", "पिल्ल"),
    "wolf": ("भेड़िय", "भेड़िए"),
    "cat": ("बिल्ली", "बिल्लि"),
    "deer": ("हिरण", "हिरन"),
    "horse": ("घोड़",),
    "cow": ("गाय", "बछड़", "बैल"),
    "goat": ("बकरी", "बकरा", "बकरि"),
    "sheep": ("भेड़",),
    "tiger": ("बाघ",),
    "lion": ("शेर",),
    "elephant": ("हाथी", "हाथि"),
    "bear": ("भालू",),
    "rabbit": ("खरगोश",),
    "monkey": ("बंदर", "बन्दर"),
    "man": ("आदमी", "किसान", "पिता", "राजा"),
    "woman": ("औरत", "महिला", "माँ", "रानी"),
    "boy": ("लड़का", "लड़के", "बच्च", "बेटा"),
    "girl": ("लड़की", "लड़कि", "बेटी"),
    "grandpa": ("दादा", "नाना", "बूढ़ा"),
    "grandma": ("दादी", "नानी", "बूढ़ी"),
    "eagle": ("चील", "बाज़", "गरुड़"),
    "parrot": ("तोता", "तोते"),
    "crow": ("कौआ", "कौए", "कौवा"),
    "sparrow": ("गौरैया", "चिड़िया"),
    "owl": ("उल्लू",),
    "peacock": ("मोर",),
    "hen": ("मुर्गी", "मुर्गा", "मुर्गे"),
    "duck": ("बत्तख", "हंस"),
    "fish": ("मछली", "मछलि"),
    "dolphin": ("डॉल्फ़िन", "डॉल्फिन"),
    "whale": ("व्हेल",),
    "turtle": ("कछुआ", "कछुए"),
}

assert set(_EN) == set(SPECIES) and set(_HI) <= set(SPECIES)

_HI_ORDER = sorted(((stem, sp) for sp, stems in _HI.items() for stem in stems),
                   key=lambda x: -len(x[0]))  # fmt: skip
_EN_LOOKUP = {w: sp for sp, words in _EN.items() for w in words}


def detect(text: str) -> list[str]:
    """Species mentioned in `text`, in order of first mention."""
    found: list[tuple[int, str]] = []
    low = text.lower()
    for m in re.finditer(r"[a-z]+", low):  # single words
        sp = _EN_LOOKUP.get(m.group(0))
        if sp:
            found.append((m.start(), sp))
    for phrase, sp in _EN_LOOKUP.items():  # two-word names ("old man")
        if " " in phrase:
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
                found = [(p, s) for p, s in found if not (m.start() <= p < m.end())]
                found.append((m.start(), sp))
    rest = text
    for stem, sp in _HI_ORDER:
        while (k := rest.find(stem)) >= 0:
            found.append((k, sp))
            rest = rest[:k] + " " * len(stem) + rest[k + len(stem) :]
    out: list[str] = []
    for _, sp in sorted(found):
        if sp not in out:
            out.append(sp)
    return out


def habitat(species: str) -> str:
    if species in SWIMMERS:
        return "water"
    if species in BIRDS and species not in ("hen", "peacock", "duck"):
        return "air"
    return "land"


def fits(species: str, kit: str) -> bool:
    k = KITS[kit]
    hab = habitat(species)
    if hab == "water":
        return k.water > 0
    if hab == "land":
        return k.ground > 0
    return True


def assign(scenes: list[Scene], idea: str = "") -> list[str]:
    """Fill every scene's `cast`; returns the main characters. Moves scenes whose sea
    creatures need water onto a water kit."""
    per_scene = [detect(s.narration) for s in scenes]
    counts: Counter[str] = Counter(sp for names in per_scene for sp in names)
    mains = detect(idea)[:2]
    if not mains:
        mains = [sp for sp, n in counts.most_common(2) if n >= max(2, len(scenes) // 3)]
    for scene, named in zip(scenes, per_scene, strict=True):
        if any(habitat(sp) == "water" for sp in named) and KITS[scene.kit].water == 0:
            words = scene.narration.lower()
            scene.kit = "beach" if any(w in words for w in ("shore", "beach", "sand")) else "ocean"
        cast: list[str] = []
        for sp in [*named, *mains]:
            if sp not in cast and fits(sp, scene.kit) and len(cast) < MAX_CAST - 1:
                cast.append(sp)
        scene.cast = cast
    # Characters the idea asked for but the writer never wrote in: show them in the scenes
    # whose environment is their home (a tiger in the jungle scenes), if there is room.
    for sp in detect(idea)[2:]:
        if any(sp in s.cast for s in scenes):
            continue
        homes = [s for s in scenes if set(_EN[sp]) & set(KITS[s.kit].keywords)]
        for scene in homes or [s for s in scenes if fits(sp, s.kit)][:1]:
            if len(scene.cast) < MAX_CAST and fits(sp, scene.kit):
                scene.cast.append(sp)
    return mains
