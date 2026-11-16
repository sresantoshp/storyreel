"""Phase 3 end-to-end audio: loudness targets, true peak, ducking depth, chunk-boundary
continuity, stems vs master, 5.1 / downmix, CLI. Media is procedural (ffmpeg lavfi) and
short; the video track is a 64x36 colour clip."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from conftest import Media
from storyreel import cli
from storyreel.audio import loudness, testsignals
from storyreel.media.importer import import_files
from storyreel.project import store
from storyreel.project.edits import Editor
from storyreel.render import plan as plan_mod
from storyreel.render.plan import RenderSettings
from storyreel.render.renderer import Renderer, RenderError, RenderSummary
from storyreel.timebase import FLICKS, Rate
from storyreel.watchdog import Watchdog

SR = 48_000
S = RenderSettings(64, 36, Rate(10), chunk_seconds=2.0, preset="ultrafast")
F = FLICKS


@dataclass(frozen=True)
class AudioMedia:
    speech: Path  # 12 s procedural speech
    music: Path  # 12 s procedural music bed
    tone: Path  # 12 s steady 330 Hz sine, stereo
    ambience: Path  # 12 s pink noise


@pytest.fixture(scope="module")
def amedia(media: Media, tmp_path_factory: pytest.TempPathFactory) -> Iterator[AudioMedia]:
    d = tmp_path_factory.mktemp("amedia")
    ff = media.ffmpeg
    tone = d / "tone.wav"
    subprocess.run(
        [str(ff), "-v", "error", "-y", "-f", "lavfi", "-i",
         "sine=f=330:d=12:r=48000,volume=0.5,aformat=channel_layouts=stereo", str(tone)],
        check=True,
    )  # fmt: skip
    yield AudioMedia(
        testsignals.generate(ff, "speech", d / "speech.wav", 12),
        testsignals.generate(ff, "music", d / "music.wav", 12),
        tone,
        testsignals.generate(ff, "ambience", d / "amb.wav", 12),
    )


def _project(tmp: Path, media: Media, am: AudioMedia, channels: int = 2) -> store.Project:
    ps = store.ProjectSettings("t", 64, 36, Rate(10), channels=channels)
    p = store.create(tmp / "p.storyreel", ps)
    files = [am.speech, am.music, am.tone, am.ambience]
    res = import_files(p, files, ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, make_proxies=False)
    assert [r.status for r in res] == ["imported"] * 4
    v1 = p.con.execute("SELECT id FROM track WHERE kind = 'video'").fetchone()[0]
    Editor(p.con).add_clip(v1, 0, 10 * F, color="#203040")
    return p


def _tracks(p: store.Project, *buses: str) -> list[int]:
    ed = Editor(p.con)
    a1 = p.con.execute("SELECT id FROM track WHERE kind = 'audio'").fetchone()[0]
    ids = [a1] + [ed.add_track("audio") for _ in buses[1:]]
    for tid, bus in zip(ids, buses, strict=True):
        if bus != "dialogue":
            ed.set_track(tid, bus=bus)
    return ids


def _render(p: store.Project, media: Media, out: Path, s: RenderSettings = S) -> RenderSummary:
    wd = Watchdog(2 * 2**30).start()
    try:
        return Renderer(p, s, ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, watchdog=wd).render(out)
    finally:
        wd.stop()


def _decode(media: Media, args: list[str], channels: int = 2) -> np.ndarray:
    raw = subprocess.run(
        [str(media.ffmpeg), "-v", "error", *args, "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels)


def _chunk_pcm(media: Media, p: store.Project, stream: int) -> np.ndarray:
    alist = p.renders_dir / "concat_audio.ffconcat"
    return _decode(media, ["-f", "concat", "-safe", "0", "-i", str(alist), "-map", f"0:a:{stream}"])


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


@pytest.mark.parametrize(
    ("target", "acodec", "suffix"), [(-14.0, "aac", ".mp4"), (-23.0, "opus", ".mkv")]
)
def test_loudness_target_and_true_peak(
    tmp_path: Path, media: Media, amedia: AudioMedia, target: float, acodec: str, suffix: str
) -> None:
    with _project(tmp_path, media, amedia) as p:
        dia, mus = _tracks(p, "dialogue", "music")
        ed = Editor(p.con)
        ed.add_clip(dia, 1 * F, 8 * F, asset_id=1, gain_db=6.0)  # hot: forces the limiter
        ed.add_clip(mus, 0, 10 * F, asset_id=2)
        out = tmp_path / f"out{suffix}"
        r = _render(p, media, out, replace(S, target_lufs=target, acodec=acodec))
    m = loudness.measure_file(media.ffmpeg, out)  # independent ebur128 pass on the output
    assert abs(m.integrated - target) <= 0.5, m
    assert m.true_peak <= -1.0, m
    assert r.loudness_i == m.integrated and r.frames_written == 100


def test_ducking_depth_follows_dialogue(tmp_path: Path, media: Media, amedia: AudioMedia) -> None:
    with _project(tmp_path, media, amedia) as p:
        dia, mus = _tracks(p, "dialogue", "music")
        ed = Editor(p.con)
        ed.set_bus("music", duck=True, duck_db=-12.0, duck_attack=0.5, duck_release=1.0)
        ed.add_clip(dia, 4 * F, 3 * F, asset_id=1)
        ed.add_clip(mus, 0, 10 * F, asset_id=3)  # steady tone, so RMS ratio = duck depth
        _render(p, media, tmp_path / "o.mp4", replace(S, target_lufs=None))
        music = _chunk_pcm(media, p, 2)
    free = _db(music[1 * SR : 3 * SR])
    ducked = _db(music[int(4.2 * SR) : int(6.8 * SR)])
    assert ducked - free == pytest.approx(-12.0, abs=0.1)
    assert _db(music[int(8.5 * SR) : int(9.5 * SR)]) == pytest.approx(free, abs=0.05)
    ramp = _db(music[int(3.7 * SR) : int(3.8 * SR)])
    assert free - 12 < ramp < free  # look-ahead: ramps down before the line starts


def test_chunked_render_matches_single_chunk(
    tmp_path: Path, media: Media, amedia: AudioMedia, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stateful effects (compressor, denoise, EQ, reverb) and automation across 1 s chunk
    boundaries must match a one-chunk render of the same timeline."""
    with _project(tmp_path, media, amedia) as p:
        dia, mus, amb = _tracks(p, "dialogue", "music", "ambience")
        ed = Editor(p.con)
        line = ed.add_clip(dia, int(0.5 * F), 9 * F, asset_id=1)
        ed.add_effect(line, "compressor", threshold=-30.0, ratio=6.0, release=400.0)
        ed.add_effect(line, "deesser", intensity=0.6)
        bed = ed.add_clip(mus, 0, 10 * F, asset_id=2)
        ed.add_effect(bed, "eq", freq=200.0, gain=4.0)
        ed.set_props(bed, fade_in=1.5, fade_out=2.5)
        noise = ed.add_clip(amb, 0, 10 * F, asset_id=4)
        ed.add_effect(noise, "denoise", amount=20.0)
        ed.add_keyframe(noise, "volume", 0, -20.0)
        ed.add_keyframe(noise, "volume", 8 * F, 0.0, interp="bezier")
        ed.add_bus_effect("dialogue", "reverb", room="small_room", mix=0.3)
        ed.add_bus_effect("master", "limiter", ceiling=-3.0)
        base = replace(S, target_lufs=None)
        _render(p, media, tmp_path / "a.mp4", replace(base, chunk_seconds=1.0))
        chunked = [_chunk_pcm(media, p, k) for k in range(5)]
        _render(p, media, tmp_path / "b.mp4", replace(base, chunk_seconds=30.0))
        single = [_chunk_pcm(media, p, k) for k in range(5)]
        # negative control: the same 1 s chunks without pre-roll really do break
        monkeypatch.setattr(plan_mod, "preroll_seconds", lambda *a: 0.0)
        _render(p, media, tmp_path / "c.mp4", replace(base, chunk_seconds=1.0))
        cold = _chunk_pcm(media, p, 0)
    assert float(np.max(np.abs(cold - single[0]))) > 0.01
    for k, (x, y) in enumerate(zip(chunked, single, strict=True)):
        assert x.shape == y.shape == (10 * SR, 2), k
        err = float(np.max(np.abs(x - y)))
        peak = float(np.max(np.abs(y)))
        if k in (0, 4):
            # master and ambience contain afftdn, whose adaptive noise estimate never fully
            # forgets its start: measured 1.4e-3 / 2.7e-3 of peak (about -50 dB)
            assert err <= 1e-2 * peak, (k, err, peak)
            assert np.corrcoef(x.ravel(), y.ravel())[0, 1] > 0.9999, k
        else:
            # compressor, de-esser, EQ, reverb, fades, keyframes: exact to float rounding
            assert err <= 1e-6, (k, err, peak)


def test_stems_sum_to_master(tmp_path: Path, media: Media, amedia: AudioMedia) -> None:
    with _project(tmp_path, media, amedia) as p:
        dia, mus, amb = _tracks(p, "dialogue", "music", "ambience")
        ed = Editor(p.con)
        ed.add_clip(dia, 2 * F, 5 * F, asset_id=1)
        ed.add_clip(mus, 0, 10 * F, asset_id=2)
        ed.add_clip(amb, 0, 10 * F, asset_id=4)
        ed.add_bus_effect("music", "reverb", room="hall", mix=0.3)
        out = tmp_path / "o.mkv"
        r = _render(p, media, out, replace(S, target_lufs=-23.0, acodec="flac", stems="wav"))
    assert [Path(x).name for x in r.stems] == [
        "dialogue.wav",
        "music.wav",
        "fx.wav",
        "ambience.wav",
    ]
    master = _decode(media, ["-i", str(out), "-map", "0:a:0"])
    stems = [_decode(media, ["-i", x]) for x in r.stems]
    assert all(st.shape == master.shape == (10 * SR, 2) for st in stems)
    total = np.sum(stems, axis=0)
    residual = _db(master - total) - _db(master)
    assert residual < -60.0, residual  # 24-bit FLAC master vs float stems
    assert not stems[2].any()  # empty fx bus → silent stem


def test_51_layout_downmix_and_codec_limits(
    tmp_path: Path, media: Media, amedia: AudioMedia
) -> None:
    with _project(tmp_path, media, amedia, channels=6) as p:
        dia, mus = _tracks(p, "dialogue", "music")
        ed = Editor(p.con)
        ed.add_clip(dia, 1 * F, 6 * F, asset_id=1)  # mono speech → centre channel
        ed.add_clip(mus, 0, 10 * F, asset_id=2)
        out = tmp_path / "o51.mp4"
        s51 = replace(S, channels=6)
        _render(p, media, out, s51)
        pcm = _decode(media, ["-i", str(out)], channels=6)
        assert pcm.shape[1] == 6
        out2 = tmp_path / "o20.mp4"
        r = _render(p, media, out2, replace(s51, out_channels=2, acodec="ac3"))
        assert _decode(media, ["-i", str(out2)], channels=2).shape[0] > 9 * SR
        assert abs(r.loudness_i + 14.0) <= 0.5  # type: ignore[operator]
        chunks = _chunk_pcm_layout(media, p)
    centre, left = chunks[:, 2], chunks[:, 0]
    speech_only = slice(int(1.5 * SR), int(6.5 * SR))
    assert _db(centre[speech_only]) > -40 and np.max(np.abs(chunks[:, 3])) == 0.0  # LFE silent
    assert _db(left[speech_only]) > -40  # music in FL/FR


def _chunk_pcm_layout(media: Media, p: store.Project) -> np.ndarray:
    alist = p.renders_dir / "concat_audio.ffconcat"
    return _decode(media, ["-f", "concat", "-safe", "0", "-i", str(alist), "-map", "0:a:0"], 6)


def test_codec_that_cannot_carry_the_layout_fails_early(
    tmp_path: Path, media: Media, amedia: AudioMedia
) -> None:
    with _project(tmp_path, media, amedia, channels=8) as p:
        with pytest.raises(RenderError, match=r"ac3 cannot carry 7.1"):
            _render(p, media, tmp_path / "x.mp4", replace(S, channels=8, acodec="ac3"))
        assert not list(p.chunks_dir.glob("*"))  # failed before rendering anything


def test_cli_audio_commands(
    tmp_path: Path, media: Media, amedia: AudioMedia, capsys: pytest.CaptureFixture[str]
) -> None:
    proj = tmp_path / "c.storyreel"
    run = cli.main
    assert run(["new", str(proj), "--size", "64x36", "--fps", "10"]) == 0
    assert run(["import", str(proj), str(amedia.speech), str(amedia.music), "--no-proxy"]) == 0
    assert run(["append", str(proj), "--color", "#101010", "--dur", "4"]) == 0
    assert run(["append", str(proj), "1", "--at", "0.5", "--dur", "3"]) == 0
    assert run(["append", str(proj), "2", "--at", "0", "--dur", "4", "--audio-track", "1"]) == 0
    assert run(["track", str(proj), "A2", "bus=music"]) == 0
    assert run(["bus", str(proj), "music", "gain_db=-2", "duck_db=-15"]) == 0
    assert run(["afx", str(proj), "2", "compressor", "threshold=-20"]) == 0
    assert run(["afx", str(proj), "fx", "reverb", "room=cave"]) == 0
    assert (
        run(["afx", str(proj), "music", "reverb", "room=nowhere"]) == 1
    )  # EditError, plain message
    assert run(["set", str(proj), "3", "pan=0.4", "fade_in=0.5"]) == 0
    assert run(["keyframe", str(proj), "3", "volume", "1.0", "-9"]) == 0
    assert run(["audio", str(proj), "layout=5.1", "target_lufs=broadcast"]) == 0
    capsys.readouterr()
    assert run(["info", str(proj), "--clips"]) == 0
    info = capsys.readouterr().out
    assert "5.1, -23 LUFS / -1 dBTP" in info
    assert "-> music" in info and "pan +0.40" in info and "volume 1s:-9dB" in info
    assert "fx compressor#1" in info and "duck -15 dB" in info and "reverb" in info
    assert run(["audio", str(proj), "layout=stereo"]) == 0
    out = tmp_path / "c.mp4"
    assert run(["render", str(proj), "-o", str(out), "--stems", "flac"]) == 0
    capsys.readouterr()
    assert run(["loudness", str(out), "--json"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert abs(m["integrated_lufs"] + 23.0) <= 0.5 and m["true_peak_dbtp"] <= -1.0
    assert (tmp_path / "c_stems" / "music.flac").is_file()
