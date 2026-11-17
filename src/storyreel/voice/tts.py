"""Offline text-to-speech (Kokoro-82M via kokoro-onnx), English and Hindi.

The model never runs in the caller's process: `synthesize` starts
`python -m storyreel.voice.worker` under a Windows Job Object memory cap (`jobs.limits`),
with a heartbeat and an overall timeout. A crash, hang or out-of-memory in the model only
ends that child; the caller gets a typed error with a plain-English message.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import psutil

from storyreel.errors import ExitCode
from storyreel.jobs.limits import spawn
from storyreel.voice import models
from storyreel.voice.models import ModelMissingError, VoiceError
from storyreel.voice.text import split_sentences

MB = 2**20
# Measured peak working set of the worker: ~0.5 GB (fp32 model, arena off). The cap leaves
# room for long sentences; if it is hit the child is stopped by the OS, never the caller.
DEFAULT_LIMIT_BYTES = 1536 * MB
HEARTBEAT_S = 240.0  # one sentence (<= 300 chars) takes < 60 s at RTF ~1 on the dev laptop
SENTENCE_PAUSE_S = 0.35

# Kokoro v1.0 voices (hexgrad/Kokoro-82M VOICES.md); used when the voices file is absent.
KNOWN_VOICES = {
    "en": [
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    ],
    "hi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
}


class SynthesisError(VoiceError):
    exit_code = ExitCode.RENDER_FAILED


class SynthesisTimeoutError(VoiceError):
    exit_code = ExitCode.RENDER_FAILED


class SynthesisMemoryError(VoiceError):
    exit_code = ExitCode.RESOURCE_EXHAUSTED


@dataclass(frozen=True)
class SpeechResult:
    """A synthesised speech file.

    `words` are (word, start_s, end_s) from the start of the file. They come from the
    model's per-phoneme durations when its phoneme words line up one-to-one with the
    words of the sentence; otherwise (numbers, abbreviations, mixed scripts) that
    sentence's words are *estimated*: spread over its voiced span in proportion to their
    length. Either way each sentence's word times are then mapped linearly onto the
    speech onset/offset measured in its audio (the model's durations drift by ~0.1-0.25 s),
    so expect word edges to be within roughly a syllable, not sample-exact. Sentence
    boundaries are exact.
    """

    path: Path  # WAV/FLAC file
    duration_s: float
    sample_rate: int
    words: list[tuple[str, float, float]]


@dataclass(frozen=True)
class SynthesisStats:
    wall_s: float
    load_s: float
    peak_rss: int  # sampled working set of the worker process
    peak_commit: int | None  # Job Object peak committed memory (Windows)
    sentences: int
    timed_sentences: int  # sentences whose word times came from the model


def _paths(spec: models.ModelSpec) -> tuple[Path, Path]:
    d = models.model_dir(spec)
    return d / spec.files[0].name, d / spec.files[1].name


def _file_voices(voices_path: Path) -> list[str]:
    """Voice ids in the voices .npz without loading the arrays (reads the zip directory)."""
    with zipfile.ZipFile(voices_path) as z:
        return sorted(n.removesuffix(".npy") for n in z.namelist() if n.endswith(".npy"))


def available(lang: str) -> bool:
    """True when the model files are present (size + verified marker), the voices file
    opens and has voices for `lang`, and the runtime packages are installed.

    The ONNX model itself is not loaded here (that costs ~0.5 GB); it is loaded only in
    the memory-capped worker.
    """
    spec = models.spec_for(lang)
    if not models.files_present(spec):
        return False
    if find_spec("kokoro_onnx") is None or find_spec("onnxruntime") is None:
        return False
    try:
        return any(v.startswith(models.VOICE_PREFIXES[lang]) for v in _file_voices(_paths(spec)[1]))
    except (OSError, zipfile.BadZipFile):
        return False


def voices(lang: str) -> list[str]:
    """Voice ids for "en" / "hi" (from the voices file when downloaded, else the known list)."""
    spec = models.spec_for(lang)
    vpath = _paths(spec)[1]
    if models.files_present(spec):
        try:
            return [v for v in _file_voices(vpath) if v.startswith(models.VOICE_PREFIXES[lang])]
        except (OSError, zipfile.BadZipFile):
            pass
    return list(KNOWN_VOICES[lang])


def ensure_models(*, lang: str, progress: Callable[[str], None] | None = None) -> None:
    """Download the model for `lang` once and verify its SHA-256. Raises ModelDownloadError
    (with a fix) when offline or when a file fails verification."""
    models.ensure(models.spec_for(lang), progress)


def license_text(lang: str, voice: str) -> str:
    spec = models.spec_for(lang)
    return f"TTS {spec.id}, voice {voice}: {spec.license}"


def _resolve_voice(lang: str, voice: str | None) -> str:
    v = voice or models.DEFAULT_VOICE[lang]
    have = voices(lang)
    if v not in have:
        raise VoiceError(
            f"Voice {v!r} is not a {lang} voice.",
            fix=f"Pick one of: {', '.join(have)} (see `storyreel voices --lang {lang}`).",
        )
    return v


def _to_flac(wav: Path, out: Path) -> None:
    from storyreel.media import ffbin
    from storyreel.proc import run

    ff, _ = ffbin.locate()
    if ff is None:
        raise SynthesisError(
            "FLAC output needs ffmpeg, which was not found.", fix=ffbin.install_hint()
        )
    run(
        [
            ff,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            wav,
            "-c:a",
            "flac",
            out,
        ],
        timeout_s=600,
    )


def synthesize_with_stats(
    text: str,
    out: Path,
    *,
    lang: str,
    voice: str | None = None,
    speed: float = 1.0,
    limit_bytes: int = DEFAULT_LIMIT_BYTES,
    timeout_s: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[SpeechResult, SynthesisStats]:
    """`synthesize`, plus measurements of the worker (wall time, peak memory)."""
    models.check_lang(lang)
    if not 0.5 <= speed <= 2.0:
        raise VoiceError(f"Speed {speed:g} is out of range.", fix="Use a speed from 0.5 to 2.0.")
    sentences = split_sentences(text)
    if not sentences:
        raise VoiceError("There is no text to speak.", fix="Give a sentence with words in it.")
    spec = models.spec_for(lang)
    if not available(lang):
        raise ModelMissingError(
            f"The speech model for {lang!r} is not installed.",
            fix=f"Run `{models.download_command(lang)}` once (needs internet, ~354 MB).",
        )
    v = _resolve_voice(lang, voice)
    suffix = out.suffix.lower()
    if suffix not in (".wav", ".flac"):
        raise VoiceError(
            f"Cannot write speech as {suffix or 'no extension'}.", fix="Use .wav or .flac."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=out.stem + ".", suffix=".part.wav", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    model_path, voices_path = _paths(spec)
    job = {
        "model": str(model_path),
        "voices": str(voices_path),
        "out": str(tmp),
        "sentences": sentences,
        "voice": v,
        "speed": speed,
        "espeak_lang": models.ESPEAK_LANG[lang],
        "pause_s": SENTENCE_PAUSE_S,
    }
    timeout = timeout_s if timeout_s is not None else 120.0 + 0.5 * len(text)
    try:
        events, stats = _run_worker(job, limit_bytes, timeout, len(sentences), on_progress)
        done = events[-1]
        sr = int(done["sample_rate"])
        words = [
            (str(w), float(s), float(e))
            for ev in events
            if ev["event"] == "sentence"
            for w, s, e in ev["words"]
        ]
        if suffix == ".flac":
            _to_flac(tmp, out)
        else:
            tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    result = SpeechResult(out, int(done["samples"]) / sr, sr, words)
    return result, stats


def synthesize(
    text: str, out: Path, *, lang: str, voice: str | None = None, speed: float = 1.0
) -> SpeechResult:
    """Speak `text` into `out` (.wav, 24 kHz 16-bit mono; or .flac).

    Long text is split into sentences (and over-long sentences at clauses); the worker
    synthesises one at a time and appends to the file, so memory does not grow with
    length. Inference runs in a memory-capped child process with a timeout.
    Raises ModelMissingError, VoiceError (bad input), SynthesisError,
    SynthesisTimeoutError, SynthesisMemoryError.
    """
    return synthesize_with_stats(text, out, lang=lang, voice=voice, speed=speed)[0]


def _tree_rss(ps: psutil.Process) -> int:
    """Working set of the worker and its children (a venv's python.exe is a launcher that
    starts the real interpreter as a child process on Windows)."""
    total = 0
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        for proc in [ps, *ps.children(recursive=True)]:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                total += proc.memory_info().rss
    return total


def _run_worker(
    job: dict[str, Any],
    limit_bytes: int,
    timeout_s: float,
    n_sentences: int,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[dict[str, Any]], SynthesisStats]:
    t0 = time.monotonic()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p, limiter = spawn(
        [sys.executable, "-m", "storyreel.voice.worker"],
        limit_bytes=limit_bytes,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert p.stdin is not None and p.stdout is not None and p.stderr is not None
    events: list[dict[str, Any]] = []
    err_tail: collections.deque[str] = collections.deque(maxlen=40)
    last_beat = [time.monotonic()]

    def read_events() -> None:
        assert p.stdout is not None
        for raw in p.stdout:
            last_beat[0] = time.monotonic()
            try:
                ev = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                err_tail.append(raw.decode("utf-8", "replace").rstrip())
                continue
            if isinstance(ev, dict):
                events.append(ev)
                if ev.get("event") == "sentence" and on_progress:
                    on_progress(int(ev["i"]) + 1, n_sentences)

    def drain_stderr() -> None:
        assert p.stderr is not None
        for raw in p.stderr:
            err_tail.append(raw.decode("utf-8", "replace").rstrip())

    readers = [threading.Thread(target=read_events, daemon=True)]
    readers.append(threading.Thread(target=drain_stderr, daemon=True))
    for t in readers:
        t.start()
    try:
        p.stdin.write(json.dumps(job, ensure_ascii=False).encode("utf-8"))
        p.stdin.close()
    except OSError:
        pass  # child died at once; reported below from its exit code
    peak_rss = 0
    reason: str | None = None
    ps = psutil.Process(p.pid)
    try:
        while True:
            try:
                p.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
            peak_rss = max(peak_rss, _tree_rss(ps))
            now = time.monotonic()
            if now - last_beat[0] > HEARTBEAT_S:
                reason = "hung"
            elif now - t0 > timeout_s:
                reason = "timeout"
            if reason:
                p.kill()
                p.wait()
                break
    finally:
        for t in readers:
            t.join(timeout=5)
        peak_commit = limiter.peak_commit_bytes()
        limiter.close()
    wall = time.monotonic() - t0
    if reason == "hung":
        raise SynthesisTimeoutError(
            f"The speech engine stopped responding for {HEARTBEAT_S:.0f} s and was stopped.",
            fix="Try again; if it repeats, split the text into shorter sentences.",
        )
    if reason == "timeout":
        raise SynthesisTimeoutError(
            f"Speech synthesis took longer than its {timeout_s:.0f} s limit and was stopped.",
            fix="Speak the text in smaller parts, or close other programs to free the CPU.",
        )
    err = next((e for e in events if e.get("event") == "error"), None)
    hit_cap = peak_commit is not None and peak_commit >= 0.97 * limit_bytes
    if (err is not None and err.get("kind") == "memory") or (p.returncode != 0 and hit_cap):
        raise SynthesisMemoryError(
            f"The speech engine needed more than its {limit_bytes / MB:.0f} MB memory cap "
            "and was stopped.",
            fix="Close other programs, or split very long sentences.",
        )
    if err is not None or p.returncode != 0 or not events or events[-1].get("event") != "done":
        detail = str(err["message"]) if err else "\n".join(list(err_tail)[-5:])
        raise SynthesisError(
            f"The speech engine failed (exit code {p.returncode}): {detail or 'no output'}",
            fix="Run `storyreel doctor`; re-download the model with "
            f"`{models.download_command('en')}` if it reports a problem.",
        )
    load = next((float(e["load_s"]) for e in events if e.get("event") == "loaded"), 0.0)
    timed = sum(1 for e in events if e.get("event") == "sentence" and e.get("timed"))
    stats = SynthesisStats(wall, load, peak_rss, peak_commit, n_sentences, timed)
    return events, stats
