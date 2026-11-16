from __future__ import annotations

from itertools import pairwise

import pytest

from storyreel.render.plan import RenderSettings, TimelineClip, chunk_bounds, plan_chunks
from storyreel.timebase import FLICKS, Rate, seconds_to_flicks


@pytest.mark.parametrize(
    "rate", [Rate(24000, 1001), Rate(24), Rate(25), Rate(30000, 1001), Rate(30), Rate(60)]
)
def test_frames_roundtrip_exactly(rate: Rate) -> None:
    for f in (0, 1, 2, 1001, 86_400 * 60):
        assert rate.flicks_to_frame(rate.frame_to_flicks(f)) == f


def test_common_frame_and_sample_durations_are_whole_flicks() -> None:
    for num, den in ((24000, 1001), (30000, 1001), (24, 1), (25, 1), (30, 1), (60, 1)):
        assert (FLICKS * den) % num == 0
    for sr in (44_100, 48_000, 96_000):
        assert FLICKS % sr == 0


def test_samples_tile_exactly_at_29_97() -> None:
    r = Rate(30000, 1001)
    # 1001 frames of 29.97 = 1601.6 samples/frame; chunk sums must equal the direct total.
    bounds = [0, 300, 601, 1001]
    total = sum(
        r.frame_to_sample(b, 48_000) - r.frame_to_sample(a, 48_000) for a, b in pairwise(bounds)
    )
    # 1001 frames = 33.4 s = 1,603,201.6 samples; floor at every boundary still tiles exactly
    assert total == r.frame_to_sample(1001, 48_000) == 1_603_201


def test_chunk_bounds_prefers_edit_points() -> None:
    assert chunk_bounds([0, 210, 300, 360, 555], 555, 300) == [(0, 300), (300, 555)]
    assert chunk_bounds([0, 750], 750, 300) == [(0, 300), (300, 600), (600, 750)]
    assert chunk_bounds([0, 100, 150, 160], 160, 300) == [(0, 160)]


def _clip(cid: int, start_s: float, dur_s: float, fp: str, track: str = "video") -> TimelineClip:
    return TimelineClip(
        cid,
        track,
        0,
        "media",
        seconds_to_flicks(start_s),
        seconds_to_flicks(dur_s),
        0,
        0.0,
        None,
        "video" if track == "video" else "audio",
        f"/m/{fp}",
        fp,
    )


def test_shift_after_insert_keeps_later_chunk_keys() -> None:
    """Inserting a 2 s clip at the start and pushing everything back must not invalidate
    the chunks after the first edit point (cache keys use chunk-relative times)."""
    s = RenderSettings(640, 360, Rate(30))
    base = [_clip(i, 4.0 * i, 4.0, f"fp{i}") for i in range(30)]  # 120 s of 4 s shots
    shifted = [_clip(99, 0, 2.0, "new")] + [
        _clip(c.clip_id, c.start / FLICKS + 2.0, 4.0, str(c.fingerprint)) for c in base
    ]
    k1 = {p.video_key(s) for p in plan_chunks(base, s)}
    plans2 = plan_chunks(shifted, s)
    k2 = [p.video_key(s) for p in plans2]
    reused = sum(k in k1 for k in k2)
    assert reused >= len(k2) - 2, (reused, len(k2))


def test_audio_and_video_keys_are_independent() -> None:
    s = RenderSettings(640, 360, Rate(30))
    a = [_clip(1, 0, 10, "v"), _clip(2, 0, 10, "v", track="audio")]
    b = [a[0], TimelineClip(**{**a[1].__dict__, "gain_db": -6.0})]
    pa, pb = plan_chunks(a, s)[0], plan_chunks(b, s)[0]
    assert pa.video_key(s) == pb.video_key(s)
    assert pa.audio_key(s) != pb.audio_key(s)


def test_key_ignores_file_path_but_not_content() -> None:
    s = RenderSettings(640, 360, Rate(30))
    a = [_clip(1, 0, 5, "same")]
    moved = [TimelineClip(**{**a[0].__dict__, "path": "/elsewhere/x.mp4"})]
    edited = [TimelineClip(**{**a[0].__dict__, "fingerprint": "changed"})]
    k = plan_chunks(a, s)[0].video_key(s)
    assert plan_chunks(moved, s)[0].video_key(s) == k
    assert plan_chunks(edited, s)[0].video_key(s) != k


def test_engine_fingerprint_is_stable_and_in_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from storyreel.render import plan as plan_mod

    s = RenderSettings(640, 360, Rate(30))
    clips = [_clip(1, 0, 5, "x")]
    k1 = plan_chunks(clips, s)[0].video_key(s)
    assert plan_chunks(clips, s)[0].video_key(s) == k1  # deterministic
    monkeypatch.setattr(plan_mod, "engine_fingerprint", lambda: "changed-code")
    assert plan_chunks(clips, s)[0].video_key(s) != k1  # code change -> new key
