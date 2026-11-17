"""Subtitle cues from narration + speech timing, and .srt output.

Timing comes from word timestamps when the voice engine provides them, otherwise words are
spread over the spoken duration in proportion to their length (stated in the cue source).
Lines are broken at word boundaries (never inside an Indic cluster) to fit the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 42  # per line; two lines per cue
MIN_CUE_S = 1.0


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str  # may contain one newline


def estimate_words(text: str, start: float, end: float) -> list[tuple[str, float, float]]:
    words = text.split()
    if not words:
        return []
    weights = [len(w) + 1 for w in words]
    total = sum(weights)
    out, t = [], start
    for w, k in zip(words, weights, strict=True):
        d = (end - start) * k / total
        out.append((w, t, t + d))
        t += d
    return out


def align_words(
    narration: str, timed: list[tuple[str, float, float]], start: float, end: float
) -> list[tuple[str, float, float]]:
    """The narration's own words (punctuation kept) on the voice engine's timings.

    TTS word timestamps drop punctuation, which subtitles need for sentence breaks. When the
    counts match, each narration word takes its spoken word's time; otherwise (numbers or
    abbreviations expanded by the voice) the words are spread over the spoken span.
    """
    words = narration.split()
    if timed and len(timed) == len(words):
        return [(w, a, b) for w, (_, a, b) in zip(words, timed, strict=True)]
    if timed:
        start, end = timed[0][1], timed[-1][2]
    return estimate_words(narration, start, end)


def _two_lines(words: list[str], max_chars: int) -> str:
    text = " ".join(words)
    if len(text) <= max_chars or len(words) < 2:
        return text
    best, acc = 1, 0
    for k, w in enumerate(words[:-1]):  # break nearest the middle
        acc += len(w) + 1
        if acc <= len(text) / 2 + 4:
            best = k + 1
    return " ".join(words[:best]) + "\n" + " ".join(words[best:])


def _sentences(words: list[tuple[str, float, float]]) -> list[list[tuple[str, float, float]]]:
    out: list[list[tuple[str, float, float]]] = [[]]
    for w in words:
        out[-1].append(w)
        if w[0].endswith((".", "!", "?", "।", "॥")):
            out.append([])
    return [s for s in out if s]


def build_cues(words: list[tuple[str, float, float]], max_chars: int = MAX_CHARS) -> list[Cue]:
    """One cue per sentence when it fits two lines; longer sentences are split into the
    fewest *balanced* parts (no one-word orphans); short neighbours may share a cue."""
    groups: list[list[tuple[str, float, float]]] = []
    cap = 2 * max_chars
    for sent in _sentences(words):
        length = len(" ".join(w for w, _, _ in sent))
        parts = max(1, -(-length // cap))
        target = length / parts
        cur: list[tuple[str, float, float]] = []
        for w in sent:
            if (
                cur
                and len(groups) < 10**9
                and len(" ".join(x for x, _, _ in cur)) >= target
                and len(" ".join(x for x, _, _ in [*cur, w])) > target
            ):
                groups.append(cur)
                cur = []
            cur.append(w)
        if cur:
            groups.append(cur)
    merged: list[list[tuple[str, float, float]]] = []
    for g in groups:  # let a short sentence share the previous cue if both fit
        if merged and len(" ".join(w for w, _, _ in [*merged[-1], *g])) <= max_chars:
            merged[-1] = [*merged[-1], *g]
        else:
            merged.append(g)
    cues = [
        Cue(
            g[0][1], max(g[-1][2], g[0][1] + MIN_CUE_S), _two_lines([w for w, _, _ in g], max_chars)
        )
        for g in merged
    ]
    for k in range(len(cues) - 1):  # never overlap the next cue
        if cues[k].end > cues[k + 1].start:
            cues[k] = Cue(cues[k].start, cues[k + 1].start, cues[k].text)
    return cues


def _ts(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[Cue], path: Path) -> None:
    lines = []
    for i, c in enumerate(cues, start=1):
        lines += [str(i), f"{_ts(c.start)} --> {_ts(c.end)}", c.text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
