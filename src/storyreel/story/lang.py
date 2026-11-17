"""Language detection from script, and the writer model chosen for each language.

Detection is by Unicode script share, not a statistical model: a story typed in Devanagari is
Hindi for our purposes. Mixed text (Hinglish in Latin letters) counts as English, which is
what the writers handle best.
"""

from __future__ import annotations

import unicodedata

from storyreel.text.shaping import script_of

# script -> language code used across StoryReel (writer choice, voice, fonts)
SCRIPT_LANG: dict[str, str] = {
    "Latn": "en",
    "Deva": "hi",
    "Taml": "ta",
    "Telu": "te",
    "Knda": "kn",
    "Beng": "bn",
    "Gujr": "gu",
    "Arab": "ar",
}

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ar": "Arabic",
}


def detect_language(text: str) -> str:
    """Language whose script covers most letters; 'en' when there are no letters."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha() and not unicodedata.category(ch).startswith("M"):
            continue
        sc = script_of(ch)
        lang = SCRIPT_LANG.get(sc or "")
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "en"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def resolve_language(text: str, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in LANG_NAMES:
            from storyreel.errors import StoryReelError

            raise StoryReelError(
                f"Unknown language {requested!r}.", fix=f"Use one of: auto, {', '.join(LANG_NAMES)}"
            )
        return requested
    return detect_language(text)
