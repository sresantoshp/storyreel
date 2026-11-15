"""Phase 2 end-to-end: exact-pixel checks on composited frames, path A/B equivalence,
speed/reverse source mapping, transitions, worker renders, and golden frames (SSIM)."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import Media, needs_ffmpeg
from storyreel.media.importer import import_files
from storyreel.project import store
from storyreel.project.edits import EditError, Editor
from storyreel.render.plan import RenderSettings, load_timeline, plan_chunks
from storyreel.render.renderer import Renderer, render_frame
from storyreel.timebase import FLICKS, Rate
from storyreel.watchdog import Watchdog

W, H = 320, 180
S = RenderSettings(W, H, Rate(30), chunk_seconds=2.0, preset="ultrafast")
GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("STORYREEL_UPDATE_GOLDEN") == "1"
RED, BLUE = (255, 0, 0), (0, 0, 255)

pytestmark = needs_ffmpeg


@pytest.fixture
def proj(tmp_path: Path, media: Media) -> Iterator[tuple[store.Project, Editor, list[int]]]:
    p = store.create(tmp_path / "p.storyreel", store.ProjectSettings("t", W, H, Rate(30)))
    import_files(
        p, [media.clip, media.image], ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, make_proxies=False
    )
    ed = Editor(p.con)
    v1 = p.con.execute("SELECT id FROM track WHERE kind = 'video'").fetchone()[0]
    tracks = [v1, ed.add_track("video"), ed.add_track("video")]
    yield p, ed, tracks
    p.close()


def frame(p: store.Project, media: Media, t: float) -> np.ndarray:
    return np.asarray(render_frame(p, S, media.ffmpeg, round(t * 30)))


def close(px: np.ndarray, rgb: tuple[int, int, int], tol: int = 3) -> bool:
    return bool(np.all(np.abs(px.astype(int) - np.array(rgb)) <= tol))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over luma, 11x11 Gaussian window (Wang et al. 2004 constants)."""
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(m.mean())


def source_frame(media: Media, t: float) -> np.ndarray:
    """Decode the test clip at source time t, fitted to W x H, as RGB."""
    out = subprocess.run(
        [
            str(media.ffmpeg),
            "-v",
            "error",
            "-ss",
            f"{t:.4f}",
            "-i",
            str(media.clip),
            "-frames:v",
            "1",
            "-vf",
            f"scale={W}:{H}:in_color_matrix=bt601",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W, 3)


# -- transforms and blending ---------------------------------------------------------------
def test_scale_position_opacity(
    proj: tuple[store.Project, Editor, list[int]], media: Media
) -> None:
    p, ed, (v1, v2, _) = proj
    ed.add_clip(v1, 0, 2 * FLICKS, color="#0000ff")
    red = ed.add_clip(v2, 0, 2 * FLICKS, color="#ff0000")
    ed.set_props(red, scale=0.5, x=40.0)
    f = frame(p, media, 0.5)
    assert close(f[H // 2, W // 2 + 40], RED)  # moved right by 40 px
    assert close(f[H // 2, 10], BLUE)  # outside the scaled layer
    assert close(f[H // 2, W // 2 + 40 + W // 4 + 4], BLUE)  # just past its right edge
    ed.set_props(red, scale=1.0, x=0.0, opacity=0.5)
    f = frame(p, media, 0.5)
    assert close(f[H // 2, W // 2], (128, 0, 128), tol=2)  # 50 % red over blue


def test_rotation_direction(proj: tuple[store.Project, Editor, list[int]], media: Media) -> None:
    p, ed, (v1, v2, _) = proj
    ed.add_clip(v1, 0, 2 * FLICKS, color="#000000")
    bar = ed.add_clip(v2, 0, 2 * FLICKS, color="#ffffff")
    ed.set_props(bar, scale_x=0.5, scale_y=0.1, rotation=90.0)  # wide bar → tall bar
    f = frame(p, media, 0.1)
    assert close(f[H // 2 - 30, W // 2], (255, 255, 255))  # now extends vertically
    assert close(f[H // 2, W // 2 + 60], (0, 0, 0))  # and no longer horizontally


def test_keyframed_opacity(proj: tuple[store.Project, Editor, list[int]], media: Media) -> None:
    p, ed, (v1, v2, _) = proj
    ed.add_clip(v1, 0, 2 * FLICKS, color="#000000")
    w = ed.add_clip(v2, 0, 2 * FLICKS, color="#ffffff")
    ed.add_keyframe(w, "opacity", 0, 0.0)
    ed.add_keyframe(w, "opacity", 2 * FLICKS, 1.0, interp="linear")
    ed.add_keyframe(w, "opacity", 0, 0.0, interp="linear")  # same t: updates, not duplicates
    assert p.con.execute("SELECT count(*) FROM keyframe").fetchone()[0] == 2
    v = frame(p, media, 1.0)[H // 2, W // 2, 0]
    assert abs(int(v) - 128) <= 3  # halfway


# -- transitions ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kind", "check"),
    [
        ("dissolve", lambda f: abs(int(f[90, 160, 0]) - int(f[90, 160, 2])) < 20),
        ("fade_black", lambda f: int(f[90, 160].max()) < 20),
        ("wipe_right", lambda f: close(f[90, 40], BLUE) and close(f[90, 280], RED)),
        ("slide_left", lambda f: close(f[90, 40], RED) and close(f[90, 280], BLUE)),
    ],
)
def test_transition_midpoints(
    proj: tuple[store.Project, Editor, list[int]], media: Media, kind: str, check: object
) -> None:
    p, ed, (v1, _, _) = proj
    a = ed.add_clip(v1, 0, 2 * FLICKS, color="#ff0000")
    b = ed.add_clip(v1, 2 * FLICKS, 2 * FLICKS, color="#0000ff")
    ed.add_transition(a, b, kind, FLICKS)
    assert close(frame(p, media, 1.0), RED) and close(frame(p, media, 3.0), BLUE)
    mid = frame(p, media, 2.0)
    assert check(mid), (kind, mid[90, 40], mid[90, 160], mid[90, 280])  # type: ignore[operator]


def test_transition_validation(proj: tuple[store.Project, Editor, list[int]]) -> None:
    _, ed, (v1, v2, _) = proj
    a = ed.add_clip(v1, 0, 2 * FLICKS, color="#ff0000")
    b = ed.add_clip(v1, 3 * FLICKS, 2 * FLICKS, color="#0000ff")
    c = ed.add_clip(v2, 2 * FLICKS, FLICKS, color="#00ff00")
    with pytest.raises(EditError, match="start exactly"):
        ed.add_transition(a, b, "dissolve", FLICKS)
    with pytest.raises(EditError, match="same track"):
        ed.add_transition(a, c, "dissolve", FLICKS)
    with pytest.raises(EditError, match="must be one of"):
        ed.add_transition(a, b, "explode", FLICKS)


# -- media: path A/B equivalence, speed, reverse -------------------------------------------
def test_path_a_and_b_match(
    proj: tuple[store.Project, Editor, list[int]], media: Media, tmp_path: Path
) -> None:
    p, ed, (v1, _, _) = proj
    ed.add_clip(v1, 0, 2 * FLICKS, asset_id=1, src_in=3 * FLICKS)
    outs = []
    for force in (False, True):
        s = RenderSettings(
            W, H, Rate(30), chunk_seconds=2.0, preset="ultrafast", crf=10, force_path_b=force
        )
        plans = plan_chunks(load_timeline(p), s)
        assert plans[0].path_b is force
        wd = Watchdog(2**31).start()
        try:
            out = tmp_path / f"o{int(force)}.mp4"
            Renderer(p, s, ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, watchdog=wd).render(out)
        finally:
            wd.stop()
        argv = [
            str(media.ffmpeg),
            "-v",
            "error",
            "-i",
            str(out),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        raw = subprocess.run(argv, capture_output=True, check=True)
        outs.append(np.frombuffer(raw.stdout, np.uint8).reshape(-1, H, W, 3))
    assert outs[0].shape == outs[1].shape == (60, H, W, 3)
    worst = min(psnr(a, b) for a, b in zip(outs[0], outs[1], strict=True))
    print(f"path A vs B worst-frame PSNR {worst:.1f} dB")
    # Path B round-trips chroma (4:2:0 -> RGB -> 4:2:0) and path A converts the matrix with
    # ffmpeg's colorspace filter; measured residual ~37 dB worst frame (mean offset 1-2
    # levels), not visible at a chunk boundary. Frame timing must match exactly.
    assert worst > 35.0


def test_speed_and_reverse_map_source_time(
    proj: tuple[store.Project, Editor, list[int]], media: Media
) -> None:
    p, ed, (v1, _, _) = proj
    c = ed.add_clip(v1, 0, 4 * FLICKS, asset_id=1, src_in=FLICKS)
    ed.set_speed(c, 2.0)  # 4 s of source in 2 s of timeline
    assert p.con.execute("SELECT duration FROM clip WHERE id = ?", (c,)).fetchone()[0] == 2 * FLICKS
    got = frame(p, media, 1.0)  # timeline 1 s → source 1 + 1 * 2 = 3 s
    assert psnr(got, source_frame(media, 3.0)) > 30
    assert psnr(got, source_frame(media, 2.0)) < psnr(got, source_frame(media, 3.0))
    ed.set_props(c, reverse=True)
    got = frame(p, media, 0.0)  # reversed: first timeline frame shows the source end (~5 s)
    assert psnr(got, source_frame(media, 4.92)) > 30


def test_speed_limits(proj: tuple[store.Project, Editor, list[int]]) -> None:
    _, ed, (v1, _, _) = proj
    c = ed.add_clip(v1, 0, FLICKS, asset_id=1)
    with pytest.raises(EditError, match=r"between 0\.1x and 16x"):
        ed.set_speed(c, 20)
    with pytest.raises(EditError, match="needs"):
        ed.set_speed(c, 15, keep_duration=True)  # 15 s of source from a 12 s clip


# -- full renders through the worker -------------------------------------------------------
def test_mixed_path_render_exact_frames(
    proj: tuple[store.Project, Editor, list[int]], media: Media, tmp_path: Path
) -> None:
    p, ed, (v1, v2, v3) = proj
    ed.add_clip(v1, 0, 6 * FLICKS, asset_id=1)  # plain video: path A where alone
    ed.add_clip(v2, 3 * FLICKS, 2 * FLICKS, kind="generator", props={"type": "snow"})
    t = ed.add_clip(
        v3, 2 * FLICKS, 3 * FLICKS, kind="text", props={"text": "Hello", "size": 40, "anim": "fade"}
    )
    ed.add_effect(t, "glow")
    wd = Watchdog(2**31).start()
    events: list[dict[str, object]] = []
    try:
        r = Renderer(
            p, S, ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, watchdog=wd, on_event=events.append
        ).render(tmp_path / "m.mp4")
    finally:
        wd.stop()
    paths = {e["path"] for e in events if e.get("event") == "chunk" and not e.get("cached")}
    assert paths == {"A", "B"}
    assert r.frames_written == r.frames_expected == 180


def test_undo_ken_burns_is_one_step(proj: tuple[store.Project, Editor, list[int]]) -> None:
    p, ed, (v1, _, _) = proj
    c = ed.add_clip(v1, 0, 3 * FLICKS, asset_id=2)
    ed.ken_burns(c, zoom_to=1.4, pan_x=30)
    assert p.con.execute("SELECT count(*) FROM keyframe").fetchone()[0] == 6
    assert ed.undo() == "Ken Burns"
    assert p.con.execute("SELECT count(*) FROM keyframe").fetchone()[0] == 0
    ed.redo()
    ed.add_effect(c, "vignette")
    ed.delete_clip(c)  # cascades
    assert p.con.execute("SELECT count(*) FROM keyframe").fetchone()[0] == 0
    assert p.con.execute("SELECT count(*) FROM effect").fetchone()[0] == 0
    ed.undo()
    assert p.con.execute("SELECT count(*) FROM keyframe").fetchone()[0] == 6
    assert p.con.execute("SELECT count(*) FROM effect").fetchone()[0] == 1


def test_validation_messages(proj: tuple[store.Project, Editor, list[int]]) -> None:
    _, ed, (v1, _, _) = proj
    c = ed.add_clip(v1, 0, FLICKS, color="#101010")
    with pytest.raises(EditError, match="cannot be animated"):
        ed.add_keyframe(c, "volume_of_doom", 0, 1.0)
    with pytest.raises(EditError, match="no parameter"):
        ed.add_effect(c, "blur", strength=3.0)
    with pytest.raises(EditError, match="color like"):
        ed.add_effect(c, "chromakey", key="green")
    eid = ed.add_effect(c, "blur", radius=2.0)
    ed.add_keyframe(c, f"fx{eid}.radius", 0, 0.0)  # effect params are animatable
    with pytest.raises(EditError, match="needs some text"):
        ed.add_clip(v1, 2 * FLICKS, FLICKS, kind="text", props={"text": " "})


# -- golden frames ---------------------------------------------------------------------------
GOLDEN_TIMES = (0.5, 1.6, 2.9)


def test_golden_scene(proj: tuple[store.Project, Editor, list[int]], media: Media) -> None:
    """Regression guard for the whole compositor. Regenerate on purpose with
    STORYREEL_UPDATE_GOLDEN=1 after an intended visual change, and review the PNGs."""
    p, ed, (v1, v2, v3) = proj
    bg = ed.add_clip(v1, 0, 3 * FLICKS, asset_id=2)
    ed.ken_burns(bg, zoom_to=1.3, pan_x=-20)
    ed.apply_color_preset(bg, "teal-orange")
    ed.add_clip(v2, 0, 3 * FLICKS, kind="generator", props={"type": "rain", "count": 200})
    ed.add_clip(
        v3,
        0,
        3 * FLICKS,
        kind="text",
        props={
            "text": "Golden 123",
            "size": 28,
            "outline": 2,
            "anim": "pop",
            "font": "arial.ttf" if shutil.which("cmd") else "",
        },
    )
    GOLDEN.mkdir(exist_ok=True)
    for t in GOLDEN_TIMES:
        got = frame(p, media, t)
        path = GOLDEN / f"scene_{t}.png"
        if UPDATE or not path.exists():
            if not UPDATE:
                pytest.skip(f"golden {path.name} missing; run with STORYREEL_UPDATE_GOLDEN=1")
            cv2.imwrite(str(path), got[..., ::-1])
            continue
        ref = cv2.imread(str(path))[..., ::-1]
        score = ssim(got, ref)
        assert score >= 0.98, f"{path.name}: SSIM {score:.4f}"
