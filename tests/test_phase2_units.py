"""Phase 2 unit tests: keyframes, effects, generators, text shaping. No ffmpeg needed."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from storyreel.render import effects as fx
from storyreel.render.generators import Generator
from storyreel.render.keyframes import Key, Params, Track, ease
from storyreel.text import shaping
from storyreel.timebase import FLICKS

CTX = fx.Ctx(frame=5, fps=30.0)


# -- keyframes ---------------------------------------------------------------------------
def test_linear_hold_and_clamping() -> None:
    tr = Track([Key(0, 0.0), Key(10, 100.0, "hold"), Key(20, 50.0)])
    assert tr.at(-5) == 0.0 and tr.at(5) == 50.0
    assert tr.at(15) == 100.0  # hold until the next key
    assert tr.at(20) == 50.0 and tr.at(99) == 50.0


def test_bezier_ease_shape() -> None:
    assert ease(0.0, 0.42, 0, 0.58, 1) == 0.0 and ease(1.0, 0.42, 0, 0.58, 1) == 1.0
    assert ease(0.5, 0.42, 0, 0.58, 1) == pytest.approx(0.5, abs=1e-4)  # symmetric
    assert ease(0.25, 0.42, 0, 0.58, 1) < 0.25  # slow start
    xs = [ease(i / 100, 0.42, 0, 0.58, 1) for i in range(101)]
    assert all(b >= a for a, b in pairwise(xs))  # monotone
    assert ease(0.3, 0, 0, 1, 1) == pytest.approx(0.3, abs=1e-4)  # linear handles


def test_params_static_vs_animated() -> None:
    p = Params({"scale": 2.0}, {"x": [Key(0, 0.0), Key(FLICKS, 100.0)]})
    assert p.get("scale", 0) == 2.0 and not p.animated("scale")
    assert p.get("x", FLICKS // 2) == pytest.approx(50.0)
    assert p.get("missing", 0, 7.0) == 7.0
    assert p.max("x", 0.0) == 100.0


# -- effects -----------------------------------------------------------------------------
def gray(v: float, h: int = 8, w: int = 8) -> np.ndarray:
    return np.full((h, w, 3), v, np.float32)


def params(kind: str, **kw: object) -> dict[str, object]:
    return {**fx_defaults(kind), **kw}


def fx_defaults(kind: str) -> dict[str, object]:
    from storyreel.render.fxspec import EFFECTS

    return dict(EFFECTS[kind])


def test_color_identity_and_exposure() -> None:
    img = gray(0.25)
    same, _ = fx.color(img, None, params("color"), CTX)
    assert np.allclose(same, img, atol=1 / 255)
    up, _ = fx.color(img, None, params("color", exposure=1.0), CTX)  # +1 stop = x2
    assert np.allclose(up, 0.5, atol=2 / 255)


def test_saturation_zero_is_grayscale() -> None:
    img = np.zeros((4, 4, 3), np.float32)
    img[..., 0] = 1.0  # pure red
    out, _ = fx.color(img, None, params("color", saturation=0.0), CTX)
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert np.allclose(out[..., 0], 0.2126, atol=2 / 255)  # Rec.709 luma of red


def test_curves_identity_and_inversion() -> None:
    img = gray(0.3)
    out, _ = fx.curves(img, None, params("curves"), CTX)
    assert np.allclose(out, 0.3, atol=1 / 255)
    inv, _ = fx.curves(img, None, params("curves", master=[[0, 1], [1, 0]]), CTX)
    assert np.allclose(inv, 0.7, atol=1 / 255)


def test_mask_and_chromakey_alpha() -> None:
    img = gray(0.5, 100, 100)
    _, a = fx.mask(img, None, params("mask", feather=0.0), CTX)
    assert a is not None and a[50, 50] == 1.0 and a[5, 5] == 0.0
    green = np.zeros((10, 10, 3), np.float32)
    green[..., 1] = 1.0
    green[:, 5:] = (1.0, 0.0, 0.0)
    _, k = fx.chromakey(green, None, params("chromakey"), CTX)
    assert k is not None and k[0, 0] == 0.0 and k[0, 9] == 1.0


def test_effects_never_modify_inputs() -> None:
    from storyreel.render.fxspec import EFFECTS

    img = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
    alpha = np.ones((32, 32), np.float32) * 0.5
    for name, fn in fx.REGISTRY.items():
        rgb_before, a_before = img.copy(), alpha.copy()
        p = dict(EFFECTS[name])
        if name == "glitch":
            p["amount"] = 1.0
        rgb, _ = fn(img, alpha, p, CTX)
        assert np.array_equal(img, rgb_before) and np.array_equal(alpha, a_before), name
        assert rgb.shape == img.shape and rgb.dtype == np.float32, name


def test_grain_is_deterministic_per_frame() -> None:
    img = gray(0.5, 16, 16)
    a1, _ = fx.grain(img, None, params("grain"), fx.Ctx(7, 30.0))
    a2, _ = fx.grain(img, None, params("grain"), fx.Ctx(7, 30.0))
    b, _ = fx.grain(img, None, params("grain"), fx.Ctx(8, 30.0))
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)


# -- generators --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gtype",
    [
        "gradient",
        "starfield",
        "rain",
        "snow",
        "fog",
        "fire",
        "particles",
        "mountains",
        "river",
        "birds",
    ],
)
def test_generators_deterministic_and_well_formed(gtype: str) -> None:
    g1, g2 = Generator(gtype, {}, 160, 90), Generator(gtype, {}, 160, 90)
    for t in (0.0, 1.5, 7.25):
        r1, a1 = g1.frame(t)
        r2, a2 = g2.frame(t)
        assert r1.shape == (90, 160, 3) and r1.dtype == np.float32
        assert np.array_equal(r1, r2)
        assert (a1 is None and a2 is None) or (a1 is not None and np.array_equal(a1, a2))
        if a1 is not None:
            assert float(a1.min()) >= 0.0 and float(a1.max()) <= 1.0
    r_a, a_a = g1.frame(0.0)
    r_b, a_b = g1.frame(1.0)
    moved = not np.array_equal(r_a, r_b) or (a_a is not None and not np.array_equal(a_a, a_b))
    assert moved or gtype in ("gradient", "mountains")  # static by default (cached)


# -- text shaping --------------------------------------------------------------------------
needs_indic_font = pytest.mark.skipif(
    not any(n in shaping._font_index() for n in ("nirmala.ttc", "notosansdevanagari-regular.ttf")),
    reason="no Devanagari font installed",
)


@needs_indic_font
def test_hindi_is_shaped_not_mapped_per_codepoint() -> None:
    line = shaping.shape_line("हिन्दी", 64)
    # 6 code points shape to fewer glyphs: the conjunct न्द is either one stacked ligature
    # (Nirmala UI: 4 glyphs) or half-न + द (Noto: 5 glyphs); both are correct Hindi.
    assert len(line.glyphs) < 6
    assert line.glyphs[0].cluster == line.glyphs[1].cluster == 0  # ि sits before ह in cluster 0


@needs_indic_font
@pytest.mark.parametrize("text", ["क्षत्रिय", "தமிழ்", "తెలుగు", "ಕನ್ನಡ"])
def test_indic_scripts_produce_fewer_glyphs_than_codepoints(text: str) -> None:
    line = shaping.shape_line(text, 48)
    assert 0 < len(line.glyphs) < len(text)
    block = shaping.layout(text, 48)
    assert block.alpha.max() > 0.9  # something was actually drawn


def test_itemize_splits_scripts_and_keeps_marks() -> None:
    runs = shaping.itemize("Hi नमस्ते!")
    assert [r.script for r in runs] == ["Latn", "Deva"]
    assert runs[1].text == "नमस्ते!"


def test_letter_units_follow_clusters() -> None:
    block = shaping.layout("ab c", 40)
    assert len(block.units) == 3  # spaces are not units
    words = shaping.layout("ab c", 40, by_word=True)
    assert len(words.units) == 2


def test_drifting_mountains_move_and_static_ones_are_cached() -> None:
    still = Generator("mountains", {}, 160, 90)
    assert still.frame(0.0) is still.frame(5.0)  # same cached planes object
    moving = Generator("mountains", {"drift": 40.0}, 160, 90)
    assert not np.array_equal(moving.frame(0.0)[0], moving.frame(2.0)[0])


def test_bundled_fonts_win_over_system_fonts() -> None:
    for script, name in (
        ("Latn", "NotoSans-Regular.ttf"),
        ("Deva", "NotoSansDevanagari-Regular.ttf"),
        ("Taml", "NotoSansTamil-Regular.ttf"),
        ("Telu", "NotoSansTelugu-Regular.ttf"),
        ("Knda", "NotoSansKannada-Regular.ttf"),
        ("Arab", "NotoSansArabic-Regular.ttf"),
    ):
        path, _ = shaping.find_font(script)
        assert path.parent == shaping.BUNDLED_FONTS and path.name == name, (script, path)
