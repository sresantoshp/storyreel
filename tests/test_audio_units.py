"""Phase 3 unit tests: envelopes, IRs, filter strings, loudness parsing, planning, editing.
No ffmpeg needed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from storyreel.audio import envelope, graph, ir, loudness
from storyreel.audio import spec as aspec
from storyreel.project import store
from storyreel.project.edits import EditError, Editor
from storyreel.render.plan import RenderSettings, TimelineClip, plan_chunks, preroll_seconds
from storyreel.timebase import FLICKS, Rate, seconds_to_flicks

SR = 48_000


# -- envelopes -----------------------------------------------------------------------------
def test_clip_envelope_is_independent_of_chunking() -> None:
    keys = [[0, -12.0, "bezier", 0.42, 0.0, 0.58, 1.0], [3 * FLICKS, 0.0, "linear", 0, 0, 1, 1]]
    full = envelope.clip_envelope(0, 4 * SR, 4 * SR, SR, SR // 2, SR, keys)
    parts = [
        envelope.clip_envelope(o, n, 4 * SR, SR, SR // 2, SR, keys)
        for o, n in ((0, 12_345), (12_345, 70_001), (82_346, 4 * SR - 82_346))
    ]
    assert np.array_equal(full, np.concatenate(parts))


def test_clip_envelope_fades_and_volume_keys() -> None:
    env = envelope.clip_envelope(0, 4 * SR, 4 * SR, SR, SR, SR)
    assert env[0] == 0.0 and env[SR // 2] == pytest.approx(0.5, abs=1e-3)
    assert env[2 * SR] == 1.0 and env[-1] == pytest.approx(0.0, abs=1e-4)
    keys = [[0, -6.0, "linear", 0, 0, 1, 1], [2 * FLICKS, 0.0, "linear", 0, 0, 1, 1]]
    env = envelope.clip_envelope(0, 4 * SR, 4 * SR, SR, 0, 0, keys)
    assert 20 * np.log10(env[0]) == pytest.approx(-6.0, abs=1e-4)
    assert 20 * np.log10(env[SR]) == pytest.approx(-3.0, abs=1e-3)
    assert env[3 * SR] == pytest.approx(1.0)


def test_duck_envelope_depth_ramps_and_window_invariance() -> None:
    speech = [(2 * SR, 4 * SR)]
    env = envelope.duck_envelope(0, 6 * SR, speech, SR, -12.0, 0.5, 1.0)
    db = 20 * np.log10(env)
    assert db[3 * SR] == pytest.approx(-12.0) and db[2 * SR] == pytest.approx(-12.0)
    assert db[SR] == pytest.approx(0.0) and db[int(5.5 * SR)] == pytest.approx(0.0)
    assert -12.0 < db[int(1.75 * SR)] < 0.0  # look-ahead ramp before the line starts
    assert -12.0 < db[int(4.5 * SR)] < 0.0  # release after it ends
    # same samples when rendered as a window that starts elsewhere
    w0 = int(1.3 * SR)
    win = envelope.duck_envelope(0, 3 * SR, [(a - w0, b - w0) for a, b in speech], SR, -12, 0.5, 1)
    assert np.array_equal(win, env[w0 : w0 + 3 * SR])


# -- impulse responses ----------------------------------------------------------------------
def test_ir_is_deterministic_unit_energy_and_skips_lfe() -> None:
    lay = aspec.layout(6)
    a = ir.generate("small_room", SR, lay.channels)
    b = ir.generate("small_room", SR, lay.channels)
    assert np.array_equal(a, b)
    assert a.shape == (int(np.ceil(aspec.reverb_tail_s("small_room") * SR)), 6)
    assert not a[:, lay.channels.index("LFE")].any()
    energy = float(np.sum(a.astype(np.float64) ** 2)) / 5  # per non-LFE channel
    assert energy == pytest.approx(1.0, rel=1e-3)
    left, right = a[:, 0].astype(np.float64), a[:, 1].astype(np.float64)
    assert abs(np.corrcoef(left, right)[0, 1]) < 0.5  # decorrelated → wide


def test_ir_cache_written_once(tmp_path: Path) -> None:
    p1 = ir.ensure_ir(tmp_path, "cave", SR, 2)
    mtime = p1.stat().st_mtime_ns
    assert ir.ensure_ir(tmp_path, "cave", SR, 2) == p1 and p1.stat().st_mtime_ns == mtime
    assert p1.stat().st_size == 4 * 2 * int(np.ceil(aspec.reverb_tail_s("cave") * SR))


# -- filter strings -----------------------------------------------------------------------
def test_pan_and_downmix_filters() -> None:
    assert graph.pan_filter(aspec.layout(2), -0.5) == "pan=stereo|c0=1*c0|c1=0.5*c1"
    assert graph.pan_filter(aspec.layout(6), 1.0) == (
        "pan=5.1|c0=0*c0|c1=1*c1|c2=1*c2|c3=1*c3|c4=0*c4|c5=1*c5"
    )
    assert graph.downmix_filter(6, 2) == (
        "pan=stereo|FL=FL+0.7071*FC+0.7071*BL|FR=FR+0.7071*FC+0.7071*BR"
    )
    assert "SL" in graph.downmix_filter(8, 2) and graph.downmix_filter(2, 2) == ""
    with pytest.raises(ValueError):
        graph.downmix_filter(2, 6)


def test_effect_filters_convert_units() -> None:
    fx = graph.effect_list([{"type": "compressor", "params": {"threshold": -20.0}}])[0]
    s = graph.simple_fx(fx["type"], fx["params"])
    assert s.startswith("acompressor=threshold=0.1:") and "makeup=1:" in s
    lim = graph.simple_fx("limiter", aspec.EFFECTS["limiter"])
    assert "level=false" in lim and "latency=true" in lim and "limit=0.891251" in lim
    assert graph.simple_fx("denoise", aspec.EFFECTS["denoise"]) == "afftdn=nr=12:nf=-50"
    with pytest.raises(ValueError):
        graph.simple_fx("reverb", aspec.EFFECTS["reverb"])  # needs an IR input


# -- loudness parsing ---------------------------------------------------------------------
SUMMARY = """[Parsed_ebur128_0 @ 0000] Summary:

  Integrated loudness:
    I:         -14.2 LUFS
    Threshold: -24.6 LUFS

  Loudness range:
    LRA:         5.6 LU
    Threshold: -34.4 LUFS
    LRA low:   -17.8 LUFS
    LRA high:  -12.2 LUFS

  True peak:
    Peak:       -1.5 dBFS
[out#0/null @ 0000] video:0KiB audio:1875KiB
"""


def test_parse_ebur128_summary() -> None:
    m = loudness.parse_summary("noise\n" + SUMMARY)
    assert (m.integrated, m.lra, m.true_peak) == (-14.2, 5.6, -1.5)
    silent = loudness.parse_summary(SUMMARY.replace("-14.2", "-70.0").replace("-1.5", "-inf"))
    assert silent.silent and silent.true_peak == float("-inf")
    with pytest.raises(loudness.LoudnessError):
        loudness.parse_summary("ffmpeg crashed")
    assert loudness.normalize_filter(3.0, -1.5, SR, limit=False) == "volume=3.000dB"
    assert "aresample=192000,alimiter=limit=0.841395" in loudness.normalize_filter(3, -1.5, SR)


# -- planning -----------------------------------------------------------------------------
def _aclip(cid: int, start: float, dur: float, bus: str = "dialogue", **kw: object) -> TimelineClip:
    base = TimelineClip(
        cid, "audio", 0, "media", seconds_to_flicks(start), seconds_to_flicks(dur), 0, 0.0,
        None, "audio", f"/m/{cid}.wav", f"fp{cid}", bus=bus,
    )  # fmt: skip
    return TimelineClip(**{**base.__dict__, **kw})


def _mix(**bus_changes: dict[str, object]) -> dict[str, object]:
    mix = graph.default_mix()
    mix["buses"]["music"]["duck"] = True
    for name, ch in bus_changes.items():
        mix["buses"][name] |= ch
    return {"buses": mix["buses"]}


def test_preroll_only_when_effects_are_used() -> None:
    plain = [_aclip(1, 0, 20)]
    assert preroll_seconds(plain, _mix()) == 0.0
    comp = [_aclip(1, 0, 20, effects=[{"type": "compressor", "params": {}}])]
    assert preroll_seconds(comp, _mix()) == aspec.WARMUP_S
    rev = {"fx": [{"type": "reverb", "params": {"room": "cathedral", "mix": 0.3}}]}
    assert preroll_seconds(plain, _mix(fx=rev)) == pytest.approx(
        aspec.WARMUP_S + aspec.reverb_tail_s("cathedral")
    )
    s = RenderSettings(64, 36, Rate(10), chunk_seconds=2.0)
    plans = plan_chunks(comp, s, mix=_mix())
    assert plans[0].preroll == 0 and plans[1].preroll == 2 * SR
    assert plans[1].alayers[0].at == 0 and plans[1].alayers[0].samples == 4 * SR


def test_mix_carries_dialogue_intervals_and_keys_follow_bus_changes() -> None:
    s = RenderSettings(64, 36, Rate(10), chunk_seconds=4.0)
    clips = [_aclip(1, 0, 12, bus="music"), _aclip(2, 5, 2), _aclip(3, 10, 1)]
    plans = plan_chunks(clips, s, mix=_mix())
    speech = [json.loads(p.mix)["speech"] for p in plans]
    assert [(p.s0, p.s1) for p in plans][1:3] == [(4 * SR, 7 * SR), (7 * SR, 11 * SR)]
    assert speech[1] == [[1 * SR, 3 * SR]]  # window-relative
    assert speech[2] == [[-2 * SR, 0], [3 * SR, 4 * SR]]  # the first line's release counts
    louder = plan_chunks(clips, s, mix=_mix(music={"gain_db": -3.0}))
    assert all(a.audio_key(s) != b.audio_key(s) for a, b in zip(plans, louder, strict=True))
    assert all(a.video_key(s) == b.video_key(s) for a, b in zip(plans, louder, strict=True))
    muted = plan_chunks(clips, s, mix=_mix(dialogue={"muted": True}))
    assert all(not any(la.bus == "dialogue" for la in p.alayers) for p in muted)
    assert all(json.loads(p.mix)["speech"] == [] for p in muted)


# -- editing ------------------------------------------------------------------------------
@pytest.fixture
def proj(tmp_path: Path) -> store.Project:
    p = store.create(tmp_path / "p.storyreel", store.ProjectSettings("t", 64, 36, Rate(10)))
    p.con.execute(
        "INSERT INTO asset(kind, path, name, fingerprint, size_bytes, duration, has_video, "
        "has_audio, imported_at) VALUES ('audio', 'x.wav', 'x.wav', 'fp', 1, ?, 0, 1, 0)",
        (60 * FLICKS,),
    )
    return p


def test_audio_props_keyframes_and_effects_are_validated(proj: store.Project) -> None:
    ed = Editor(proj.con)
    a1 = proj.con.execute("SELECT id FROM track WHERE kind = 'audio'").fetchone()[0]
    v1 = proj.con.execute("SELECT id FROM track WHERE kind = 'video'").fetchone()[0]
    cid = ed.add_clip(a1, 0, 10 * FLICKS, asset_id=1)
    ed.set_props(cid, pan=-0.25, fade_in=1.0, fade_out=2.0)
    with pytest.raises(EditError, match="pan must be between"):
        ed.set_props(cid, pan=1.5)
    with pytest.raises(EditError, match="longer than the clip"):
        ed.set_props(cid, fade_in=6.0, fade_out=6.0)
    ed.add_keyframe(cid, "volume", 0, -6.0)
    with pytest.raises(EditError, match="volume must be between"):
        ed.add_keyframe(cid, "volume", FLICKS, 40.0)
    with pytest.raises(EditError, match="Audio clips animate"):
        ed.add_keyframe(cid, "scale", 0, 2.0)
    vid = ed.add_clip(v1, 0, 10 * FLICKS, color="#000000")
    with pytest.raises(EditError, match="cannot be animated"):
        ed.add_keyframe(vid, "volume", 0, -6.0)
    with pytest.raises(EditError, match="audio effect"):
        ed.add_effect(vid, "compressor")
    with pytest.raises(EditError, match="video effect"):
        ed.add_effect(cid, "blur")
    with pytest.raises(EditError, match="room must be one of"):
        ed.add_effect(cid, "reverb", room="bathroom")
    with pytest.raises(EditError, match="ratio must be between"):
        ed.add_effect(cid, "compressor", ratio=50.0)
    assert ed.add_effect(cid, "reverb", room="cave", mix=0.5) > 0


def test_bus_track_and_project_audio_edits_are_undoable(proj: store.Project) -> None:
    ed = Editor(proj.con)
    a1 = proj.con.execute("SELECT id FROM track WHERE kind = 'audio'").fetchone()[0]
    ed.set_track(a1, bus="music")
    ed.set_bus("music", gain_db=-3.0, duck_db=-18.0)
    eid = ed.add_bus_effect("master", "limiter", ceiling=-2.0)
    ed.set_audio(layout="5.1", target_lufs="broadcast")
    s = proj.settings()
    assert (s.channels, s.target_lufs) == (6, -23.0)
    with pytest.raises(EditError, match="duck under dialogue"):
        ed.set_bus("dialogue", duck=True)
    with pytest.raises(EditError, match="master is the sum"):
        ed.set_track(a1, bus="master")
    for _ in range(4):
        assert ed.undo() is not None
    s = proj.settings()
    assert (s.channels, s.target_lufs) == (2, -14.0)
    assert proj.con.execute("SELECT count(*) FROM bus_effect").fetchone()[0] == 0
    music = proj.con.execute("SELECT * FROM bus WHERE name = 'music'").fetchone()
    assert music["gain_db"] == 0.0 and json.loads(music["params"]) == {"duck": True}
    assert ed.redo() == "Track A1"
    assert proj.con.execute("SELECT bus FROM track WHERE id = ?", (a1,)).fetchone()[0] == "music"
    assert eid > 0


def test_phase2_project_migrates_to_audio_schema(tmp_path: Path) -> None:
    from storyreel.project.schema import MIGRATIONS
    from storyreel.project.store import _split_sql, connect

    root = tmp_path / "old.storyreel"
    root.mkdir()
    con = connect(root / "project.db")
    for version, script in enumerate(MIGRATIONS[:2], start=1):
        for stmt in _split_sql(script):
            con.execute(stmt)
        con.execute(f"PRAGMA user_version={version}")
    con.execute("INSERT INTO project VALUES (1, 'old', 64, 36, 10, 1, 48000, 2, 0)")
    con.execute("INSERT INTO track(kind, idx, name) VALUES ('audio', 0, 'A1')")
    con.close()
    with store.open_project(root) as p:
        s = p.settings()
        assert (s.target_lufs, s.true_peak, s.normalize) == (-14.0, -1.0, True)
        assert p.con.execute("SELECT bus FROM track").fetchone()[0] == "dialogue"
        names = [r[0] for r in p.con.execute("SELECT name FROM bus ORDER BY id")]
        assert names == list(aspec.ALL_BUSES)
