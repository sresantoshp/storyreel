"""Text-to-speech: text handling, model download/verification, typed errors (always run),
and real synthesis in English and Hindi (skipped when the model is not downloaded)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import Media
from storyreel import cli, doctor
from storyreel.voice import models, tts
from storyreel.voice.text import estimate_word_times, split_sentences, spoken_words
from storyreel.voice.worker import align_words

HI = "एक समय की बात है।"
# "ek samay ki baat hai." (once upon a time) in Devanagari, written as escapes on purpose
HI2 = "वह बहुत मेहनती था।"

have_en = tts.available("en")
have_hi = tts.available("hi")
needs_en = pytest.mark.skipif(not have_en, reason="Kokoro model not downloaded")
needs_hi = pytest.mark.skipif(not have_hi, reason="Kokoro model not downloaded")


# -- text ---------------------------------------------------------------------------------
def test_split_sentences_english_and_hindi() -> None:
    assert split_sentences('He said "Go!" Then he left.\n\nThe end') == [
        'He said "Go!"',
        "Then he left.",
        "The end",
    ]
    assert split_sentences(f"{HI} {HI2}") == [HI, HI2]
    assert split_sentences(" ... \n - ") == []


def test_long_sentence_is_split_under_limit() -> None:
    text = ", ".join(["the quick brown fox jumps over the lazy dog"] * 30) + "."
    parts = split_sentences(text, limit=120)
    assert all(len(p) <= 120 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_spoken_words_and_estimates() -> None:
    assert spoken_words('"Hello," she said - twice!') == ["Hello", "she", "said", "twice"]
    assert spoken_words(HI) == HI.rstrip("।").split()
    est = estimate_word_times(["a", "bbbb", "cc"], 1.0, 2.0)
    assert [w for w, _, _ in est] == ["a", "bbbb", "cc"]
    starts = [s for _, s, _ in est]
    assert starts == sorted(starts) and est[0][1] == 1.0 and est[-1][2] <= 2.0
    assert all(e > s for _, s, e in est)


def _tm(ph: str, s: float, e: float) -> SimpleNamespace:
    return SimpleNamespace(phoneme=ph, start=s, end=e)


def test_align_words_uses_model_timings_when_counts_match() -> None:
    timings = [_tm("h", 0.1, 0.2), _tm("i", 0.2, 0.3), _tm(" ", 0.3, 0.35)]
    timings += [_tm("j", 0.35, 0.5), _tm("o", 0.5, 0.7), _tm(".", 0.7, 0.8)]
    words, timed = align_words("Hi Joe.", timings, 1.0)
    assert timed and words == [("Hi", 0.1, 0.3), ("Joe", 0.35, 0.7)]
    words, timed = align_words("Hi Joe 42.", timings, 1.0)  # 3 words vs 2 groups
    assert not timed and [w for w, _, _ in words] == ["Hi", "Joe", "42"]
    assert words[0][1] == 0.1 and words[-1][2] <= 0.7


# -- models -------------------------------------------------------------------------------
@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "models"
    monkeypatch.setenv("STORYREEL_MODELS_DIR", str(d))
    return d


def _fake_spec(src: Path, sha: str, size: int) -> models.ModelSpec:
    return models.ModelSpec(
        id="fake",
        files=(models.ModelFile("w.bin", src.as_uri(), sha, size),),
        license="test",
        license_source="test",
        langs=("en",),
        sample_rate=24000,
    )


def test_download_verifies_sha256(tmp_path: Path, models_dir: Path) -> None:
    src = tmp_path / "w.bin"
    src.write_bytes(b"weights" * 1000)
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    bad = _fake_spec(src, "0" * 64, src.stat().st_size)
    with pytest.raises(models.ModelDownloadError) as e:
        models.ensure(bad)
    assert "verification" in e.value.message and e.value.fix
    assert not list((models_dir / "fake").glob("*.bin*"))  # nothing half-written kept
    spec = _fake_spec(src, good, src.stat().st_size)
    msgs: list[str] = []
    d = models.ensure(spec, progress=msgs.append)
    assert (d / "w.bin").read_bytes() == src.read_bytes()
    assert models.files_present(spec) and any("SHA-256 OK" in m for m in msgs)
    info = json.loads((d / "verified.json").read_text(encoding="utf-8"))
    assert info["sources"]["w.bin"]["sha256"] == good and info["license"] == "test"
    (d / "w.bin").write_bytes(b"tampered" * 1000)  # size/mtime marker no longer matches
    assert not models.files_present(spec)


def test_offline_download_is_typed_error(models_dir: Path) -> None:
    spec = models.ModelSpec(
        "offline",
        (models.ModelFile("x.onnx", "http://127.0.0.1:9/x.onnx", "0" * 64, 10),),
        "test",
        "test",
        ("en",),
        24000,
    )
    with pytest.raises(models.ModelDownloadError) as e:
        models.ensure(spec)
    assert "internet" in e.value.message and "x.onnx" in (e.value.fix or "")


def test_missing_model_is_reported_with_fix(models_dir: Path, tmp_path: Path) -> None:
    assert not tts.available("hi")
    assert "hf_alpha" in tts.voices("hi") and "af_heart" in tts.voices("en")
    with pytest.raises(models.ModelMissingError) as e:
        tts.synthesize("Hello there.", tmp_path / "x.wav", lang="en")
    assert "storyreel models download --lang en" in (e.value.fix or "")
    with pytest.raises(models.VoiceError):
        tts.synthesize("Hello.", tmp_path / "x.wav", lang="fr")
    with pytest.raises(models.VoiceError):
        tts.synthesize("   ", tmp_path / "x.wav", lang="en")
    checks = {c.name: c for c in doctor.run_checks(test_hw=False, env={})}
    c = checks["TTS model (hi)"]
    assert c.status is doctor.Status.INFO and "models download --lang hi" in c.detail


# -- real synthesis (needs the downloaded model) -------------------------------------------
def _pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768.0
    return x, sr


def _check_speech(r: tts.SpeechResult, words: list[str]) -> None:
    x, sr = _pcm(r.path)
    assert sr == r.sample_rate == 24000
    assert abs(x.size / sr - r.duration_s) < 1e-6
    assert r.duration_s > 0.8
    assert 20 * np.log10(np.sqrt(np.mean(x**2))) > -35  # not silent
    assert [w for w, _, _ in r.words] == words
    t = [s for _, s, _ in r.words]
    assert t == sorted(t) and r.words[-1][2] <= r.duration_s + 1e-3
    # word times line up with the audio: the signal is louder inside words than outside
    inside = np.zeros(x.size, dtype=bool)
    for _, s, e in r.words:
        inside[int(s * sr) : int(e * sr)] = True
    if (~inside).sum() > sr // 10:
        assert np.mean(x[inside] ** 2) > 2 * np.mean(x[~inside] ** 2)


@needs_en
def test_english_speech(tmp_path: Path) -> None:
    text = "The lamp went dark. Nobody saw it."
    r, st = tts.synthesize_with_stats(text, tmp_path / "en.wav", lang="en")
    _check_speech(r, spoken_words(text))
    assert st.timed_sentences == 2 and st.wall_s > st.load_s > 0
    assert st.peak_commit is None or st.peak_commit > 50 * 2**20


@needs_hi
def test_hindi_speech(tmp_path: Path) -> None:
    r = tts.synthesize(f"{HI} {HI2}", tmp_path / "hi.wav", lang="hi", voice="hm_omega")
    _check_speech(r, spoken_words(f"{HI} {HI2}"))


@needs_en
def test_memory_cap_gives_typed_error(tmp_path: Path) -> None:
    with pytest.raises(tts.SynthesisMemoryError) as e:
        tts.synthesize_with_stats(
            "Hello there.", tmp_path / "x.wav", lang="en", limit_bytes=150 * 2**20
        )
    assert e.value.message and not (tmp_path / "x.wav").exists()
    assert not list(tmp_path.glob("*.part*"))


@needs_en
def test_bad_voice_is_rejected_before_starting(tmp_path: Path) -> None:
    with pytest.raises(models.VoiceError) as e:
        tts.synthesize("Hi.", tmp_path / "x.wav", lang="hi", voice="af_heart")
    assert "hf_alpha" in (e.value.fix or "")


# -- CLI ----------------------------------------------------------------------------------
def test_cli_generated_sound_on_buses(
    tmp_path: Path, media: Media, capsys: pytest.CaptureFixture[str]
) -> None:
    proj = tmp_path / "v.storyreel"
    run = cli.main
    assert run(["new", str(proj), "--size", "64x36", "--fps", "10"]) == 0
    assert run(["append", str(proj), "--color", "#101010", "--dur", "6"]) == 0
    assert run(["music", str(proj), "calm", "--at", "0", "--dur", "6"]) == 0
    assert run(["ambience", str(proj), "rain", "--at", "0", "--dur", "6", "--gain-db", "-6"]) == 0
    assert run(["ambience", str(proj), "wind", "--at", "1", "--dur", "2"]) == 0  # 2nd track
    if have_en:
        assert run(["say", str(proj), "The lamp went dark.", "--at", "1.5"]) == 0
    capsys.readouterr()
    assert run(["voices", "--lang", "hi"]) == 0
    assert "hf_alpha  (default)" in capsys.readouterr().out
    con = sqlite3.connect(proj / "project.db")
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT a.is_generated, a.license, a.source_url, t.bus, t.name FROM clip c "
        "JOIN asset a ON a.id = c.asset_id JOIN track t ON t.id = c.track_id ORDER BY c.id"
    ).fetchall()
    con.close()
    assert all(r["is_generated"] == 1 for r in rows)
    assert [r["bus"] for r in rows[:3]] == ["music", "ambience", "ambience"]
    assert rows[1]["name"] != rows[2]["name"]  # overlapping ambience went to a new track
    assert "CC0" in rows[0]["license"] and "kind=calm" in rows[0]["source_url"]
    if have_en:
        assert rows[3]["bus"] == "dialogue" and "Apache-2.0" in rows[3]["license"]
    out = tmp_path / "v.mp4"
    assert run(["render", str(proj), "-o", str(out)]) == 0
    capsys.readouterr()
    assert run(["loudness", str(out), "--json"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert abs(m["integrated_lufs"] + 14.0) <= 0.5 and m["true_peak_dbtp"] <= -1.0
