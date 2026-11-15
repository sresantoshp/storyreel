"""Text layout: script itemisation, font fallback, HarfBuzz shaping, FreeType rasterising.

Why not Pillow: Pillow's complex-script layout needs libraqm + fribidi DLLs that are not
reliably present on Windows. uharfbuzz and freetype-py ship wheels with everything inside.

Output is a `TextBlock`: an alpha mask of the whole block plus one bounding box and alpha
sprite per animation unit (grapheme cluster or word). Clusters come from HarfBuzz, so an
Indic conjunct like क्ष or a Tamil ligature is always one unit and never split.

Limit (stated honestly): mixed-direction text inside one line (Arabic + English) is laid
out per run in logical order without the Unicode bidi algorithm. Single-direction lines,
including pure RTL, are correct.
"""

from __future__ import annotations

import functools
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb

from storyreel.errors import StoryReelError

# Script ranges for itemisation; anything else (spaces, digits, punctuation) joins the
# current run. Order matters only for readability.
_SCRIPTS: list[tuple[str, int, int]] = [
    ("Arab", 0x0600, 0x06FF),
    ("Hebr", 0x0590, 0x05FF),
    ("Deva", 0x0900, 0x097F),
    ("Beng", 0x0980, 0x09FF),
    ("Guru", 0x0A00, 0x0A7F),
    ("Gujr", 0x0A80, 0x0AFF),
    ("Orya", 0x0B00, 0x0B7F),
    ("Taml", 0x0B80, 0x0BFF),
    ("Telu", 0x0C00, 0x0C7F),
    ("Knda", 0x0C80, 0x0CFF),
    ("Mlym", 0x0D00, 0x0D7F),
    ("Sinh", 0x0D80, 0x0DFF),
    ("Thai", 0x0E00, 0x0E7F),
    ("Hani", 0x4E00, 0x9FFF),
    ("Hira", 0x3040, 0x30FF),
    ("Hang", 0xAC00, 0xD7AF),
    ("Cyrl", 0x0400, 0x04FF),
    ("Grek", 0x0370, 0x03FF),
]
_INDIC = {"Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym", "Sinh"}

# Font candidates per script: (file name, ttc index). First one found wins.
_CANDIDATES: dict[str, list[tuple[str, int]]] = {
    "Latn": [
        ("segoeui.ttf", 0),
        ("arial.ttf", 0),
        ("DejaVuSans.ttf", 0),
        ("NotoSans-Regular.ttf", 0),
        ("Helvetica.ttc", 0),
    ],
    "Indic": [
        ("Nirmala.ttc", 0),
        ("NotoSansDevanagari-Regular.ttf", 0),
        ("Mangal.ttf", 0),
        ("Kohinoor.ttc", 0),
    ],
    "Taml": [("Nirmala.ttc", 0), ("NotoSansTamil-Regular.ttf", 0), ("Latha.ttf", 0)],
    "Telu": [("Nirmala.ttc", 0), ("NotoSansTelugu-Regular.ttf", 0), ("Gautami.ttf", 0)],
    "Knda": [("Nirmala.ttc", 0), ("NotoSansKannada-Regular.ttf", 0), ("Tunga.ttf", 0)],
    "Arab": [("arial.ttf", 0), ("segoeui.ttf", 0), ("NotoSansArabic-Regular.ttf", 0)],
    "Hebr": [("arial.ttf", 0), ("segoeui.ttf", 0), ("NotoSansHebrew-Regular.ttf", 0)],
    "Hani": [("msyh.ttc", 0), ("NotoSansCJK-Regular.ttc", 0)],
    "Hira": [("YuGothR.ttc", 0), ("msgothic.ttc", 0), ("NotoSansCJK-Regular.ttc", 0)],
    "Hang": [("malgun.ttf", 0), ("NotoSansCJK-Regular.ttc", 0)],
}


class FontNotFoundError(StoryReelError):
    pass


def _font_dirs() -> list[Path]:
    if sys.platform == "win32":
        import os

        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
        return [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", local]
    if sys.platform == "darwin":
        return [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
    return [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]


@functools.lru_cache(maxsize=1)
def _font_index() -> dict[str, Path]:
    """lower-case file name -> path, across system font dirs (searched recursively once)."""
    out: dict[str, Path] = {}
    for d in _font_dirs():
        if d.is_dir():
            for p in d.rglob("*"):
                if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                    out.setdefault(p.name.lower(), p)
    return out


def script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, lo, hi in _SCRIPTS:
        if lo <= cp <= hi:
            return name
    if ch.isalpha():
        return "Latn"
    return None


def find_font(script: str, override: str = "") -> tuple[Path, int]:
    if override:
        p = Path(override)
        if p.is_file():
            return p, 0
        hit = _font_index().get(override.lower())
        if hit:
            return hit, 0
        raise FontNotFoundError(f"Font {override!r} not found.", fix="Give a font file path.")
    key = "Indic" if script in _INDIC and script not in _CANDIDATES else script
    for name, idx in _CANDIDATES.get(key, []) + _CANDIDATES["Latn"]:
        hit = _font_index().get(name.lower())
        if hit:
            return hit, idx
    raise FontNotFoundError(
        f"No font found for script {script}.",
        fix="Install Noto Sans for that script (fonts.google.com/noto), or set a font path.",
    )


@dataclass(frozen=True)
class Run:
    text: str
    start: int  # char offset in the line
    script: str


def itemize(line: str) -> list[Run]:
    runs: list[Run] = []
    cur, cur_start, cur_script = "", 0, None
    for i, ch in enumerate(line):
        sc = script_of(ch)
        # combining marks and neutral characters stay with the current run
        if sc is None or unicodedata.category(ch).startswith("M") or sc == cur_script:
            cur += ch
            continue
        if cur_script is None:
            cur_script = sc
            cur += ch
            continue
        runs.append(Run(cur, cur_start, cur_script))
        cur, cur_start, cur_script = ch, i, sc
    if cur:
        runs.append(Run(cur, cur_start, cur_script or "Latn"))
    return runs


class _Font:
    def __init__(self, path: Path, index: int) -> None:
        self.hb_face = hb.Face(hb.Blob.from_file_path(str(path)), index)
        self.hb_font = hb.Font(self.hb_face)
        self.ft = freetype.Face(str(path), index)
        self.upem = self.hb_face.upem


@functools.lru_cache(maxsize=16)
def _load(path: str, index: int) -> _Font:
    return _Font(Path(path), index)


@dataclass(frozen=True)
class Glyph:
    font: tuple[str, int]
    gid: int
    x: float  # pen position, pixels
    y: float
    cluster: int  # char offset in the line


@dataclass(frozen=True)
class ShapedLine:
    glyphs: list[Glyph]
    width: float
    text: str


def shape_line(line: str, size: float, font_override: str = "") -> ShapedLine:
    glyphs: list[Glyph] = []
    pen = 0.0
    for run in itemize(line):
        path, idx = find_font(run.script, font_override)
        f = _load(str(path), idx)
        buf = hb.Buffer()
        buf.add_str(run.text)
        buf.guess_segment_properties()
        hb.shape(f.hb_font, buf, {"kern": True, "liga": True})
        k = size / f.upem
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
            glyphs.append(
                Glyph(
                    (str(path), idx),
                    info.codepoint,
                    pen + pos.x_offset * k,
                    -pos.y_offset * k,
                    run.start + info.cluster,
                )
            )
            pen += pos.x_advance * k
    return ShapedLine(glyphs, pen, line)


@dataclass(frozen=True)
class Unit:
    """One animation unit (letter cluster or word) with its pixel box in the block."""

    x0: int
    y0: int
    x1: int
    y1: int
    order: int  # reveal order


@dataclass
class TextBlock:
    alpha: np.ndarray  # float32 HxW coverage 0..1, whole block (fill only)
    units: list[Unit]
    unit_map: np.ndarray  # int32 HxW: unit index per pixel, -1 = none
    line_boxes: list[tuple[int, int, int, int]]


def _unit_bounds(line: ShapedLine, by_word: bool) -> list[int]:
    """Glyph index -> unit id within the line (clusters or words)."""
    ids: list[int] = []
    unit = -1
    prev_cluster: int | None = None
    prev_space = True
    for g in line.glyphs:
        ch = line.text[g.cluster] if g.cluster < len(line.text) else " "
        is_space = ch.isspace()
        if by_word:
            if not is_space and prev_space:
                unit += 1
            prev_space = is_space
        elif g.cluster != prev_cluster:
            unit += 1
        prev_cluster = g.cluster
        ids.append(-1 if is_space else max(unit, 0))
    return ids


def layout(
    text: str,
    size: float,
    *,
    font: str = "",
    align: str = "center",
    line_spacing: float = 1.2,
    max_width: float = 0.0,
    by_word: bool = False,
    margin: float = 0.0,
) -> TextBlock:
    """Shape, wrap and rasterise text into a coverage mask with per-unit boxes."""
    lines: list[ShapedLine] = []
    for para in text.split("\n"):
        if max_width <= 0:
            lines.append(shape_line(para, size, font))
            continue
        words, cur = para.split(" "), ""
        for w in words:
            trial = f"{cur} {w}" if cur else w
            if cur and shape_line(trial, size, font).width > max_width:
                lines.append(shape_line(cur, size, font))
                cur = w
            else:
                cur = trial
        lines.append(shape_line(cur, size, font))

    pad = int(size * 0.5 + margin)
    line_h = size * line_spacing
    width = int(max((ln.width for ln in lines), default=1) + 2 * pad) + 1
    height = int(line_h * len(lines) + 2 * pad) + 1
    alpha = np.zeros((height, width), np.float32)
    unit_map = np.full((height, width), -1, np.int32)
    units: list[Unit] = []
    line_boxes: list[tuple[int, int, int, int]] = []
    order = 0
    for li, ln in enumerate(lines):
        ox: float
        if align == "left":
            ox = pad
        elif align == "right":
            ox = width - pad - ln.width
        else:
            ox = (width - ln.width) / 2
        baseline = pad + li * line_h + size * 0.95
        uid = _unit_bounds(ln, by_word)
        boxes: dict[int, list[int]] = {}
        for g, u in zip(ln.glyphs, uid, strict=True):
            f = _load(*g.font)
            f.ft.set_char_size(int(size * 64))
            f.ft.load_glyph(g.gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
            bm = f.ft.glyph.bitmap
            if bm.width == 0 or bm.rows == 0:
                continue
            arr = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, : bm.width]
            gx = int(round(ox + g.x + f.ft.glyph.bitmap_left))
            gy = int(round(baseline + g.y - f.ft.glyph.bitmap_top))
            x0, y0 = max(gx, 0), max(gy, 0)
            x1, y1 = min(gx + bm.width, width), min(gy + bm.rows, height)
            if x0 >= x1 or y0 >= y1:
                continue
            src = arr[y0 - gy : y1 - gy, x0 - gx : x1 - gx].astype(np.float32) / 255.0
            region = alpha[y0:y1, x0:x1]
            np.maximum(region, src, out=region)
            if u >= 0:
                umap = unit_map[y0:y1, x0:x1]
                umap[src > 0.0] = order + u
                b = boxes.setdefault(u, [x0, y0, x1, y1])
                b[0], b[1] = min(b[0], x0), min(b[1], y0)
                b[2], b[3] = max(b[2], x1), max(b[3], y1)
        for u in sorted(boxes):
            b = boxes[u]
            units.append(Unit(b[0], b[1], b[2], b[3], order + u))
        order += (max(uid) + 1) if uid else 0
        lx0 = int(ox)
        line_boxes.append(
            (lx0, int(pad + li * line_h), int(lx0 + ln.width), int(pad + (li + 1) * line_h))
        )
    return TextBlock(alpha, units, unit_map, line_boxes)
