"""TTS child process: `python -m storyreel.voice.worker`, one job spec as JSON on stdin.

Runs under a Job Object memory cap (started by `storyreel.voice.tts`). Loads Kokoro once,
synthesises one sentence at a time and appends 16-bit PCM to a WAV file, so memory does
not grow with text length. Reports JSON lines on stdout:
  {"event": "loaded", "load_s": ...}
  {"event": "sentence", "i": 0, "audio_s": ..., "words": [[word, start, end], ...],
   "timed": true}
  {"event": "done", "samples": ..., "sample_rate": 24000}
  {"event": "error", "kind": "memory" | "failed", "message": "..."}
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from storyreel.voice.text import estimate_word_times, spoken_words

# Phoneme symbols that are punctuation in Kokoro's vocabulary (not spoken sounds).
_PUNCT = set(';:,.!?\u00a1\u00bf\u2014\u2026"\u00ab\u00bb\u201c\u201d()')


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def phoneme_words(timings: list[Any]) -> list[tuple[float, float]]:
    """(start, end) of each spoken phoneme group between spaces."""
    groups: list[tuple[float, float]] = []
    start: float | None = None
    end = 0.0
    for tm in timings:
        ph = str(tm.phoneme)
        if ph == " ":
            if start is not None:
                groups.append((start, end))
            start = None
            continue
        if ph in _PUNCT:
            continue
        if start is None:
            start = float(tm.start)
        end = float(tm.end)
    if start is not None:
        groups.append((start, end))
    return groups


def align_words(
    sentence: str, timings: list[Any], audio_s: float
) -> tuple[list[tuple[str, float, float]], bool]:
    """Word times for one sentence. Exact when the model's phoneme words match the text's
    words one to one; otherwise estimated over the voiced span. Returns (words, timed)."""
    words = spoken_words(sentence)
    groups = phoneme_words(timings)
    if groups and len(groups) == len(words):
        return [
            (w, round(s, 4), round(e, 4)) for w, (s, e) in zip(words, groups, strict=True)
        ], True
    lo = groups[0][0] if groups else 0.0
    hi = groups[-1][1] if groups else audio_s
    return estimate_word_times(words, lo, hi), False


def speech_span(pcm: np.ndarray, sr: int, floor_db: float = 35.0) -> tuple[float, float] | None:
    """(onset, offset) in seconds: first and last 10 ms frame within `floor_db` of the
    loudest frame. None for silence."""
    hop = max(1, sr // 100)
    n = pcm.size // hop
    if n == 0:
        return None
    x = pcm[: n * hop].astype(np.float64).reshape(n, hop)
    db = 10 * np.log10(np.mean(x * x, axis=1) + 1e-12)
    on = np.flatnonzero(db > db.max() - floor_db)
    if on.size == 0 or db.max() < -100:
        return None
    return float(on[0] * hop / sr), float((on[-1] + 1) * hop / sr)


def anchor_to_audio(
    words: list[tuple[str, float, float]], pcm: np.ndarray, sr: int
) -> list[tuple[str, float, float]]:
    """Map the sentence's word times linearly so the first word starts at the measured
    speech onset and the last ends at the measured offset.

    Measured on this model: its duration output drifts up to ~0.1-0.25 s against the trimmed
    audio (late starts, ends that run into the trailing silence); the order and relative
    lengths of words are kept, only the span is corrected.
    """
    span = speech_span(pcm, sr)
    if not words or span is None:
        return words
    s0, e1 = words[0][1], words[-1][2]
    on, off = span
    if e1 <= s0 or off <= on:
        return words
    k = (off - on) / (e1 - s0)
    return [(w, round(on + (s - s0) * k, 4), round(on + (e - s0) * k, 4)) for w, s, e in words]


def run(spec: dict[str, Any]) -> int:
    import onnxruntime as rt
    from kokoro_onnx import Kokoro

    t0 = time.monotonic()
    so = rt.SessionOptions()
    so.enable_cpu_mem_arena = False  # measured: -100 MB peak, same speed
    sess = rt.InferenceSession(spec["model"], sess_options=so, providers=["CPUExecutionProvider"])
    kok = Kokoro.from_session(sess, spec["voices"])
    _emit({"event": "loaded", "load_s": round(time.monotonic() - t0, 3)})

    sr = 24_000
    pause = np.zeros(int(float(spec["pause_s"]) * sr), dtype=np.int16)
    offset = 0
    out = Path(spec["out"])
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        sentences: list[str] = spec["sentences"]
        for i, sentence in enumerate(sentences):
            audio, rate, timings = kok.create_timed(
                sentence, spec["voice"], float(spec["speed"]), spec["espeak_lang"]
            )
            if rate != sr:
                raise RuntimeError(f"model returned {rate} Hz audio, expected {sr} Hz")
            pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
            w.writeframes(pcm.tobytes())
            words, timed = align_words(sentence, list(timings), pcm.size / sr)
            words = anchor_to_audio(words, pcm, sr)
            base = offset / sr
            _emit(
                {
                    "event": "sentence",
                    "i": i,
                    "of": len(sentences),
                    "audio_s": round(pcm.size / sr, 4),
                    "timed": timed,
                    "words": [[wd, round(s + base, 4), round(e + base, 4)] for wd, s, e in words],
                }
            )
            offset += pcm.size
            if i + 1 < len(sentences):
                w.writeframes(pause.tobytes())
                offset += pause.size
    _emit({"event": "done", "samples": offset, "sample_rate": sr})
    return 0


def main() -> int:
    try:
        spec = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        return run(spec)
    except MemoryError:
        _emit({"event": "error", "kind": "memory", "message": "out of memory"})
        return 3
    except Exception as exc:  # noqa: BLE001 (report every failure to the parent as JSON)
        _emit({"event": "error", "kind": "failed", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
