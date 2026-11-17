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


def build_cues(words: list[tuple[str, float, float]], max_chars: int = MAX_CHARS) -> list[Cue]:
    """Greedy: fill up to two lines of max_chars, cut at sentence ends when possible."""
    cues: list[Cue] = []
    line: list[tuple[str, float, float]] = []

    def flush() -> None:
        if not line:
            return
        text = " ".join(w for w, _, _ in line)
        if len(text) > max_chars:  # break into two balanced lines
            best, acc = 0, 0
            for i, (w, _, _) in enumerate(line[:-1]):
                acc += len(w) + 1
                if acc <= len(text) / 2 + 4:
                    best = i + 1
            best = max(best, 1)
            text = (
                " ".join(w for w, _, _ in line[:best])
                + "\n"
                + " ".join(w for w, _, _ in line[best:])
            )
        start, end = line[0][1], line[-1][2]
        cues.append(Cue(start, max(end, start + MIN_CUE_S), text))
        line.clear()

    for word in words:
        if line and len(" ".join(w for w, _, _ in line)) + 1 + len(word[0]) > 2 * max_chars:
            flush()
        line.append(word)
        if (
            word[0].endswith((".", "!", "?", "।", "॥"))
            and len(" ".join(w for w, _, _ in line)) > max_chars // 2
        ):
            flush()
    flush()
    for i in range(len(cues) - 1):  # never overlap the next cue
        if cues[i].end > cues[i + 1].start:
            cues[i] = Cue(cues[i].start, cues[i + 1].start, cues[i].text)
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
