"""Text handling for speech: sentence splitting and word timing estimates.

Pure functions, no model needed, so they are unit-tested without downloads.
"""

from __future__ import annotations

import re

# Sentence ends: . ! ? and the Devanagari danda / double danda, followed by space or end.
_SENT_END = re.compile("([.!?\u0964\u0965][\"'\u201d\u2019)\\]]*)\\s+")
_CLAUSE = re.compile("(?<=[,;:\u2014])\\s+")
_WORD = re.compile(r"[^\s]+")
# A token with no letter/digit in any script (e.g. "-", "...", "।") is not a spoken word.
_SPOKEN = re.compile(r"\w", re.UNICODE)

MAX_PIECE_CHARS = 300
_STRIP = ".,!?;:\"'()[]{}\u0964\u0965\u201c\u201d\u2018\u2019"


def _split_long(sentence: str, limit: int) -> list[str]:
    """Split an over-long sentence at clause marks, then at word boundaries."""
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    for clause in _CLAUSE.split(sentence):
        if pieces and len(pieces[-1]) + 1 + len(clause) <= limit:
            pieces[-1] = f"{pieces[-1]} {clause}"
            continue
        while len(clause) > limit:
            cut = clause.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(clause[:cut].strip())
            clause = clause[cut:].strip()
        if clause:
            pieces.append(clause)
    return pieces


def split_sentences(text: str, limit: int = MAX_PIECE_CHARS) -> list[str]:
    """Sentences (and paragraph lines), each at most `limit` characters.

    The model is run once per piece, so memory per step stays small however long the
    text is, and a failure points at one sentence.
    """
    out: list[str] = []
    for para in text.splitlines():
        para = " ".join(para.split())
        if not para:
            continue
        parts = _SENT_END.split(para)  # [text, end mark(s), text, end mark(s), ..., text]
        for k in range(0, len(parts), 2):
            sent = (parts[k] + (parts[k + 1] if k + 1 < len(parts) else "")).strip()
            if sent:
                out.extend(_split_long(sent, limit))
    return [s for s in out if _SPOKEN.search(s)]


def spoken_words(text: str) -> list[str]:
    """Whitespace-separated tokens that contain a letter or digit, punctuation stripped."""
    words = []
    for tok in _WORD.findall(text):
        if _SPOKEN.search(tok):
            words.append(tok.strip(_STRIP))
    return [w for w in words if w]


def estimate_word_times(
    words: list[str], start_s: float, end_s: float
) -> list[tuple[str, float, float]]:
    """Spread words over [start_s, end_s] in proportion to their length (+1 for the gap).

    Used when the engine's timings cannot be matched to the words of the text.
    """
    if not words:
        return []
    weights = [len(w) + 1 for w in words]
    total = float(sum(weights))
    span = max(0.0, end_s - start_s)
    out: list[tuple[str, float, float]] = []
    t = start_s
    for w, wt in zip(words, weights, strict=True):
        d = span * wt / total
        out.append((w, round(t, 4), round(t + d * len(w) / wt, 4)))
        t += d
    return out
