"""Animated characters: animals, people, birds and sea creatures as cut-out puppets.

Every character is a small 2D rig drawn from vector shapes each frame (side view, facing
right; mirrored when moving left). Legs use 2-bone IK on a foot path whose stride is tied to
the walking speed, so feet stay planted instead of sliding. Everything is a pure function of
(time, params), like the other generators: deterministic, chunk-safe, no assets to download.

The `actor` generator params: species, action (walk|run|idle|fly|swim|hop), x (start centre,
fraction of width), y (ground/flight line, fraction of height), size (character height,
fraction of frame height), speed (frame widths per second, sign = direction), surface
(optional water line: parts below it are seen "through the water"), seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from storyreel.render import draw as d
from storyreel.render.draw import Planes
from storyreel.render.fxspec import ACTOR_DEFAULTS  # noqa: F401  (re-export)

P = tuple[float, float]


# -- species tables ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Quad:
    leg: float  # leg length (fraction of character height)
    body: float  # torso length
    body_h: float
    neck: float
    neck_ang: float  # degrees above horizontal
    head: float
    head_h: float
    snout: float  # 0 flat face .. 1 long muzzle
    ears: str  # point | round | flop | long | elephant | none
    ear: float
    tail: str  # bush | thin | tuft | short | puff | hair | curl
    tail_len: float
    color: str
    belly: str
    dark: str
    accent: str = ""
    pattern: str = ""  # stripes | spots | patches | wool
    extras: tuple[str, ...] = ()  # antlers horns tusks trunk mane beard
    leg_w: float = 0.05
    gait: str = "walk"  # walk | hop
    socks: bool = False  # lower legs in the dark colour (fox, horse)


@dataclass(frozen=True)
class Human:
    height: float = 1.0
    head: float = 0.068
    skin: tuple[str, ...] = ("#f1c9a5", "#d9a57b", "#b9805a", "#8d5a3b")
    hair: tuple[str, ...] = ("#2b1d14", "#4a3020", "#121212", "#6b4a2a")
    top: tuple[str, ...] = ("#3d6fb6", "#c0392b", "#2e8b57", "#e0a030", "#7b4fa0")
    bottom: tuple[str, ...] = ("#2f3640", "#44546a", "#5b4636")
    style: str = "pants"  # pants | skirt | long
    long_hair: bool = False
    stoop: float = 0.0
    stick: bool = False


@dataclass(frozen=True)
class Bird:
    body: str
    belly: str
    wing: str
    beak: str
    head: str = ""
    size: float = 1.0
    tail: float = 0.3
    flap: float = 3.0  # flaps per second
    glide: bool = False
    round: bool = False  # owl
    crest: str = ""
    train: bool = False  # peacock
    legs: str = "#d9a441"


@dataclass(frozen=True)
class Swimmer:
    kind: str  # fish | dolphin | turtle | whale
    color: str
    belly: str
    fin: str
    stripes: tuple[str, ...] = field(default=())


QUADS: dict[str, Quad] = {
    "fox": Quad(0.36, 0.78, 0.28, 0.2, 48, 0.3, 0.2, 0.75, "point", 0.15, "bush", 0.62,
                "#d8662a", "#f6eee4", "#3a2a22", "#ffffff", socks=True),
    "dog": Quad(0.4, 0.74, 0.3, 0.2, 52, 0.28, 0.21, 0.55, "flop", 0.13, "thin", 0.35,
                "#a8743f", "#ecd7b8", "#5a3b20"),
    "wolf": Quad(0.42, 0.82, 0.32, 0.2, 40, 0.3, 0.21, 0.75, "point", 0.13, "bush", 0.55,
                 "#8a8d93", "#dadcdf", "#3b3d42"),
    "cat": Quad(0.3, 0.62, 0.25, 0.16, 55, 0.22, 0.2, 0.25, "point", 0.11, "thin", 0.6,
                "#e2a35a", "#f8e6cf", "#a0662c", pattern="stripes"),
    "deer": Quad(0.55, 0.7, 0.28, 0.32, 65, 0.25, 0.15, 0.55, "long", 0.12, "short", 0.1,
                 "#b07a45", "#f1e2c8", "#5a3a20", pattern="spots", extras=("antlers",)),
    "horse": Quad(0.52, 0.86, 0.34, 0.38, 55, 0.34, 0.17, 0.7, "point", 0.08, "hair", 0.5,
                  "#7a4b2a", "#8d5b36", "#2b1d14", extras=("mane",), leg_w=0.055, socks=True),
    "cow": Quad(0.42, 0.95, 0.42, 0.14, 20, 0.3, 0.2, 0.45, "round", 0.09, "tuft", 0.5,
                "#f4f1ea", "#f4f1ea", "#3b3330", pattern="patches", extras=("horns",),
                leg_w=0.07),
    "goat": Quad(0.42, 0.62, 0.28, 0.2, 55, 0.24, 0.15, 0.5, "flop", 0.08, "short", 0.12,
                 "#ebe4d6", "#f6f2ea", "#6b5b4b", extras=("horns", "beard")),
    "sheep": Quad(0.3, 0.7, 0.42, 0.1, 35, 0.22, 0.17, 0.4, "flop", 0.08, "short", 0.1,
                  "#f2efe8", "#f2efe8", "#3a3533", pattern="wool", socks=True),
    "tiger": Quad(0.4, 1.0, 0.34, 0.15, 28, 0.3, 0.25, 0.35, "round", 0.08, "thin", 0.75,
                  "#e8832a", "#fbf1e3", "#2a1a10", pattern="stripes", leg_w=0.065),
    "lion": Quad(0.42, 0.98, 0.34, 0.15, 30, 0.3, 0.25, 0.35, "round", 0.07, "tuft", 0.7,
                 "#d6a35a", "#f3dfb8", "#7a4a1e", extras=("mane",), leg_w=0.065),
    "elephant": Quad(0.5, 1.0, 0.56, 0.04, 10, 0.36, 0.4, 0.0, "elephant", 0.3, "thin",
                     0.3, "#8f9399", "#a2a6ab", "#6e7278", extras=("trunk", "tusks"),
                     leg_w=0.13),
    "bear": Quad(0.32, 0.9, 0.5, 0.12, 25, 0.28, 0.24, 0.45, "round", 0.08, "short",
                 0.06, "#5b3b25", "#6e4a30", "#3a2516", leg_w=0.09),
    "rabbit": Quad(0.2, 0.45, 0.32, 0.06, 60, 0.22, 0.2, 0.25, "long", 0.32, "puff",
                   0.1, "#cdb8a0", "#f4ece2", "#8d7660", gait="hop", leg_w=0.05),
    "monkey": Quad(0.32, 0.5, 0.26, 0.08, 40, 0.24, 0.24, 0.12, "round", 0.07, "curl",
                   0.65, "#8a6440", "#e8c9a0", "#5a3f24", leg_w=0.045),
}  # fmt: skip

HUMANS: dict[str, Human] = {
    "man": Human(),
    "woman": Human(long_hair=True, style="skirt",
                   top=("#c2185b", "#e67e22", "#16a085", "#8e44ad", "#d35400")),
    "boy": Human(height=0.72, head=0.085),
    "girl": Human(height=0.7, head=0.085, long_hair=True, style="skirt",
                  top=("#e84393", "#f1c40f", "#00b894", "#6c5ce7")),
    "grandpa": Human(hair=("#d9d9d9", "#bfbfbf"), stoop=0.12, stick=True),
    "grandma": Human(hair=("#d9d9d9", "#c8c8c8"), style="long", long_hair=True, stoop=0.08,
                     top=("#8e2a5a", "#2a6f8e", "#b5651d")),
}  # fmt: skip

BIRDS: dict[str, Bird] = {
    "eagle": Bird("#5b3a22", "#6e4a2e", "#4a2e1a", "#f2c12e", head="#f5f2ea", size=1.3,
                  flap=1.6, glide=True),
    "parrot": Bird("#2fa84f", "#7bd389", "#1f7a3a", "#e8433a", head="#e8433a", flap=5.0),
    "crow": Bird("#1d1d22", "#2a2a30", "#141418", "#2a2a2a", flap=3.5),
    "sparrow": Bird("#8a6240", "#e9dcc8", "#6b4a2e", "#3a3028", size=0.7, flap=7.0),
    "seagull": Bird("#f5f7f8", "#ffffff", "#9aa5ad", "#f2b134", size=1.1, flap=2.2,
                    glide=True),
    "owl": Bird("#8b6a48", "#d9c3a0", "#6e5236", "#3a3028", round=True, flap=2.5),
    "peacock": Bird("#1f5fbf", "#1f5fbf", "#2b6b5a", "#cfc6a8", crest="#1f5fbf", size=1.2,
                    train=True, flap=2.5),
    "hen": Bird("#b5652c", "#d48a4a", "#8a4a20", "#f2c12e", crest="#e0302a", size=0.9,
                flap=6.0),
    "duck": Bird("#f2f2ee", "#ffffff", "#d9d9d2", "#f29a1e", size=0.9, flap=5.0),
}  # fmt: skip

SWIMMERS: dict[str, Swimmer] = {
    "fish": Swimmer("fish", "#f08a24", "#ffd8a8", "#ffffff", ("#ffffff",)),
    "dolphin": Swimmer("dolphin", "#6f8fa6", "#dfe8ee", "#5a7a91"),
    "whale": Swimmer("whale", "#3d5470", "#c9d3dc", "#34495f"),
    "turtle": Swimmer("turtle", "#5f8f4a", "#c9b27a", "#4b7238"),
}

SPECIES: dict[str, str] = {
    **dict.fromkeys(QUADS, "quad"),
    **dict.fromkeys(HUMANS, "human"),
    **dict.fromkeys(BIRDS, "bird"),
    **dict.fromkeys(SWIMMERS, "swim"),
}

# relative real-world height, so a fox stands lower than a horse in the same scene
SCALE: dict[str, float] = {
    "fox": 0.45, "dog": 0.5, "wolf": 0.6, "cat": 0.32, "deer": 0.85, "horse": 1.0,
    "cow": 0.85, "goat": 0.6, "sheep": 0.55, "tiger": 0.75, "lion": 0.8, "elephant": 1.5,
    "bear": 0.8, "rabbit": 0.25, "monkey": 0.45,
    "man": 1.0, "woman": 0.95, "boy": 0.72, "girl": 0.7, "grandpa": 0.95, "grandma": 0.9,
    "eagle": 0.5, "parrot": 0.3, "crow": 0.3, "sparrow": 0.2, "seagull": 0.4, "owl": 0.35,
    "peacock": 0.6, "hen": 0.3, "duck": 0.3,
    "fish": 0.25, "dolphin": 0.7, "whale": 1.6, "turtle": 0.4,
}  # fmt: skip


# -- rig helpers ------------------------------------------------------------------------------
def _ik(root: P, foot: P, a: float, b: float, bend: float) -> P:
    dx, dy = foot[0] - root[0], foot[1] - root[1]
    dist = max(1e-6, math.hypot(dx, dy))
    dist_c = min(max(dist, abs(a - b) + 1e-3), a + b - 1e-3)
    cos_a = (a * a + dist_c * dist_c - b * b) / (2 * a * dist_c)
    alpha = math.acos(max(-1.0, min(1.0, cos_a)))
    base = math.atan2(dy, dx)
    ang = base + bend * alpha
    return root[0] + a * math.cos(ang), root[1] + a * math.sin(ang)


def _foot(phase: float, stride: float, lift: float, duty: float = 0.6) -> P:
    """Foot offset (dx forward, dy up-negative) over one gait cycle; planted while c<duty."""
    c = phase % 1.0
    if c < duty:
        return stride / 2 - stride * c / duty, 0.0
    s = (c - duty) / (1 - duty)
    return -stride / 2 + stride * s, -lift * math.sin(math.pi * s)


@dataclass
class _Ctx:
    cv: np.ndarray
    s: float  # character height in px
    gx: float  # body centre x on canvas
    gy: float  # ground line y on canvas
    t: float
    phase: float  # gait cycle position
    moving: bool
    running: bool
    rng: np.random.Generator


# -- quadrupeds -------------------------------------------------------------------------------
def _draw_quad(q: Quad, c: _Ctx) -> None:
    s, cv = c.s, c.cv
    leg = q.leg * s
    bob = 0.0
    if q.gait == "hop" and c.moving:
        hop = max(0.0, math.sin(2 * math.pi * c.phase))
        bob = -hop * leg * 0.9
    elif c.moving:
        bob = -abs(math.sin(2 * math.pi * c.phase)) * s * (0.03 if c.running else 0.012)
    else:
        bob = math.sin(c.t * 1.6) * s * 0.004  # breathing
    pitch = math.sin(2 * math.pi * c.phase) * (0.06 if c.running else 0.0)
    cx, gy = c.gx, c.gy
    top = gy - leg + bob
    body_c, belly_c, dark_c = d.rgb(q.color), d.rgb(q.belly), d.rgb(q.dark)
    far_k = 0.78
    half = q.body * s / 2
    shoulder = (cx + half * 0.72, top - pitch * half)
    hip = (cx - half * 0.72, top + pitch * half)
    stride = leg * (1.5 if c.running else 0.85)
    lift = leg * (0.35 if c.running else 0.2)
    offs = (
        {"nh": 0.0, "fh": 0.12, "nf": 0.5, "ff": 0.62}
        if c.running
        else {"nh": 0.0, "nf": 0.25, "fh": 0.5, "ff": 0.75}
    )
    lw = q.leg_w * s

    def leg_at(root: P, key: str, front: bool, k: float) -> None:
        if q.gait == "hop" and c.moving:
            ph = c.phase
            tuck = max(0.0, math.sin(2 * math.pi * ph))
            dx = (0.25 if front else -0.1) * leg * tuck
            foot = (root[0] + dx, root[1] + leg - tuck * leg * 0.35 - bob)
        elif c.moving:
            fx, fy = _foot(c.phase + offs[key], stride, lift)
            foot = (root[0] + fx, gy + fy)
        else:
            foot = (root[0] + (0.04 if front else -0.04) * leg, gy)
        a = b = leg * 0.52
        knee = _ik(root, foot, a, b, -1.0 if front else 1.0)
        col = d.col(body_c, k)
        d.limb(cv, root, knee, lw, lw * 0.75, col)
        d.limb(cv, knee, foot, lw * 0.75, lw * 0.55, d.col(dark_c if q.socks else body_c, k))
        d.ellipse(
            cv, foot[0] + lw * 0.3, foot[1] - lw * 0.25, lw * 0.75, lw * 0.4, d.col(dark_c, k * 0.9)
        )

    # tail (behind the body)
    tail_base = (hip[0] - half * 0.2, top - q.body_h * s * 0.25)
    wag = math.sin(c.t * (5.0 if c.moving else 2.2)) * 0.18
    tl = q.tail_len * s
    if q.tail in ("bush", "hair"):
        droop = 0.55 if q.tail == "hair" else 0.25
        ang = math.pi + droop + wag
        tip = (
            tail_base[0] + math.cos(ang) * tl,
            tail_base[1] + math.sin(ang) * tl * -0.2 + tl * droop * 0.7,
        )
        width = tl * (0.34 if q.tail == "bush" else 0.22)
        d.poly(
            cv, d.leaf(tail_base, tip, width, 0.12), d.col(body_c if q.tail == "bush" else dark_c)
        )
        if q.accent:
            k0 = 0.72
            mid = (
                tail_base[0] + (tip[0] - tail_base[0]) * k0,
                tail_base[1] + (tip[1] - tail_base[1]) * k0,
            )
            d.poly(cv, d.leaf(mid, tip, width * 0.55, 0.05), d.col(q.accent))
    elif q.tail in ("thin", "tuft", "curl"):
        up = 0.9 if q.tail == "thin" and q.tail_len > 0.5 else -0.6
        pts = d.bezier(
            tail_base,
            (tail_base[0] - tl * 0.5, tail_base[1] + tl * 0.3 * (1 if up < 0 else -0.2)),
            (tail_base[0] - tl * 0.8, tail_base[1] - up * tl * 0.4 + wag * tl),
            10,
        )
        if q.tail == "curl":
            pts = np.concatenate(
                [
                    pts,
                    d.bezier(
                        pts[-1],
                        (pts[-1][0] - tl * 0.15, pts[-1][1] - tl * 0.3),
                        (pts[-1][0] + tl * 0.05, pts[-1][1] - tl * 0.3),
                        6,
                    ),
                ]
            )
        d.line(cv, pts, d.col(body_c), max(2, lw * 0.4))
        if q.tail == "tuft":
            d.poly(cv, d.leaf(pts[-2], (pts[-1][0] - lw, pts[-1][1] + lw), lw * 0.8), d.col(dark_c))
    elif q.tail in ("short", "puff"):
        d.circle(
            cv,
            tail_base[0] - lw * 0.3,
            tail_base[1],
            lw * (1.0 if q.tail == "puff" else 0.6),
            d.col("#ffffff" if q.tail == "puff" else q.color),
        )

    # far legs, body, near legs
    leg_at((hip[0] + lw * 0.4, hip[1]), "fh", False, far_k)
    leg_at((shoulder[0] - lw * 0.4, shoulder[1]), "ff", True, far_k)
    body_cy = top - q.body_h * s * 0.32
    bh = q.body_h * s / 2
    d.ellipse(cv, cx, body_cy, half, bh, d.col(body_c), math.degrees(pitch) * -0.6)
    d.ellipse(cv, cx + half * 0.05, body_cy + bh * 0.45, half * 0.78, bh * 0.5, d.col(belly_c))
    d.ellipse(cv, cx, body_cy - bh * 0.15, half * 0.95, bh * 0.75, d.col(body_c))
    d.ellipse(cv, hip[0] + half * 0.1, body_cy, half * 0.42, bh * 0.95, d.col(body_c))  # haunch
    d.ellipse(cv, shoulder[0] - half * 0.1, body_cy, half * 0.38, bh * 0.92, d.col(body_c))
    if q.pattern == "stripes":
        for i in range(7):
            x = cx - half * 0.8 + i * half * 0.26
            d.poly(
                cv,
                d.leaf(
                    (x, body_cy - bh * 0.95), (x - half * 0.05, body_cy + bh * 0.4), lw * 0.45, 0.05
                ),
                d.col(dark_c),
            )
    elif q.pattern == "spots":
        for _ in range(9):
            d.circle(
                cv,
                cx + c.rng.uniform(-0.6, 0.5) * half,
                body_cy + c.rng.uniform(-0.6, 0.1) * bh,
                lw * 0.22,
                d.col("#f6efe2"),
            )
    elif q.pattern == "patches":
        for _ in range(4):
            d.ellipse(
                cv,
                cx + c.rng.uniform(-0.6, 0.6) * half,
                body_cy + c.rng.uniform(-0.5, 0.3) * bh,
                half * c.rng.uniform(0.15, 0.3),
                bh * c.rng.uniform(0.25, 0.45),
                d.col("#2b2523"),
                c.rng.uniform(0, 180),
            )
    elif q.pattern == "wool":
        for i in range(14):
            a_ = i / 14 * 2 * math.pi
            d.circle(
                cv,
                cx + math.cos(a_) * half * 0.88,
                body_cy + math.sin(a_) * bh * 0.85,
                bh * 0.32,
                d.col(body_c, 0.95 if i % 2 else 1.0),
            )
    leg_at((hip[0], hip[1]), "nh", False, 1.0)
    leg_at((shoulder[0], shoulder[1]), "nf", True, 1.0)

    # neck and head
    na = math.radians(q.neck_ang)
    if c.moving:
        na += math.sin(2 * math.pi * c.phase * 2) * 0.04
    elif q.extras and "trunk" not in q.extras:
        na += -0.25 * max(0.0, math.sin(c.t * 0.35)) ** 3  # occasionally lowers to graze
    neck0 = (cx + half * 0.78, body_cy - bh * 0.35)
    nl = max(q.neck * s, s * 0.03)
    neck1 = (neck0[0] + math.cos(na) * nl, neck0[1] - math.sin(na) * nl)
    hl, hh = q.head * s, q.head_h * s
    d.limb(cv, neck0, neck1, bh * 0.6, hh * 0.45, d.col(body_c))
    if "mane" in q.extras:
        mane_c = d.col("#8a5524" if q.color == "#d6a35a" else q.dark)
        if q.color == "#d6a35a":  # lion: big round mane
            d.circle(cv, neck1[0] - hl * 0.05, neck1[1] + hh * 0.1, hh * 1.05, mane_c)
        else:
            d.limb(
                cv,
                (neck0[0] - bh * 0.1, neck0[1] - bh * 0.45),
                (neck1[0] - hl * 0.15, neck1[1] - hh * 0.45),
                lw * 0.6,
                lw * 0.5,
                mane_c,
            )
    hx, hy = neck1[0] + hl * 0.12, neck1[1]
    # ears behind the head
    if q.ears == "elephant":
        d.ellipse(
            cv,
            hx - hl * 0.15,
            hy + hh * 0.05,
            hl * 0.42,
            hh * 0.62,
            d.col(body_c, 0.9),
            -10 + 6 * math.sin(c.t * 1.5),
        )
    d.ellipse(cv, hx, hy, hl * 0.5, hh * 0.5, d.col(body_c))
    if q.snout > 0:
        ml = hl * (0.25 + 0.55 * q.snout)
        tip = (hx + hl * 0.35 + ml * 0.6, hy + hh * 0.18)
        d.poly(
            cv,
            [
                (hx + hl * 0.05, hy - hh * 0.38),
                tip,
                (tip[0] - ml * 0.2, tip[1] + hh * 0.22),
                (hx, hy + hh * 0.45),
            ],
            d.col(body_c),
        )
        d.poly(
            cv,
            [
                (hx + hl * 0.1, hy + hh * 0.1),
                tip,
                (tip[0] - ml * 0.2, tip[1] + hh * 0.22),
                (hx, hy + hh * 0.45),
            ],
            d.col(belly_c),
        )
        d.circle(cv, tip[0] - lw * 0.1, tip[1] - lw * 0.05, max(2.0, hh * 0.1), d.col("#1a1412"))
    else:
        d.ellipse(cv, hx + hl * 0.32, hy + hh * 0.12, hl * 0.22, hh * 0.28, d.col(belly_c))
    eye = (hx + hl * 0.12, hy - hh * 0.12)
    blink = 0.15 if (c.t % 3.7) < 0.12 else 1.0
    d.ellipse(
        cv, eye[0], eye[1], max(1.6, hh * 0.08), max(1.0, hh * 0.08 * blink), d.col("#141010")
    )
    if blink > 0.5:
        d.circle(
            cv, eye[0] + hh * 0.025, eye[1] - hh * 0.03, max(0.8, hh * 0.025), d.col("#ffffff")
        )
    ear_c = d.col(body_c, 0.92)
    er = q.ear * s
    ex, ey = hx - hl * 0.15, hy - hh * 0.38
    if q.ears == "point":
        d.poly(
            cv,
            [
                (ex - er * 0.35, ey + er * 0.2),
                (ex - er * 0.05, ey - er),
                (ex + er * 0.35, ey + er * 0.15),
            ],
            ear_c,
        )
        d.poly(
            cv,
            [
                (ex - er * 0.15, ey + er * 0.05),
                (ex - er * 0.05, ey - er * 0.65),
                (ex + er * 0.15, ey + er * 0.05),
            ],
            d.col(dark_c),
        )
    elif q.ears == "round":
        d.circle(cv, ex, ey, er * 0.5, ear_c)
    elif q.ears == "flop":
        d.poly(
            cv,
            d.leaf((ex, ey + er * 0.2), (ex - er * 0.35, ey + er * 1.2), er * 0.55, 0.1),
            d.col(dark_c),
        )
    elif q.ears == "long":
        d.poly(cv, d.leaf((ex, ey + er * 0.1), (ex - er * 0.35, ey - er), er * 0.32, -0.08), ear_c)
        d.poly(
            cv,
            d.leaf(
                (ex + er * 0.12, ey + er * 0.1), (ex + er * 0.05, ey - er * 0.95), er * 0.3, 0.05
            ),
            ear_c,
        )
    elif q.ears == "elephant":
        d.ellipse(
            cv,
            hx - hl * 0.12,
            hy + hh * 0.05,
            hl * 0.3,
            hh * 0.48,
            d.col(body_c, 0.86),
            -10 + 6 * math.sin(c.t * 1.5),
        )
    if "trunk" in q.extras:
        sw = math.sin(c.t * 1.3) * hl * 0.15
        pts = d.bezier(
            (hx + hl * 0.4, hy + hh * 0.05),
            (hx + hl * 0.8, hy + hh * 0.6),
            (hx + hl * 0.55 + sw, hy + hh * 1.45),
            14,
        )
        for i in range(len(pts) - 1):
            r = hh * (0.17 - 0.1 * i / len(pts))
            d.limb(cv, pts[i], pts[i + 1], r, r * 0.95, d.col(body_c))
    if "tusks" in q.extras:
        d.poly(
            cv,
            d.leaf(
                (hx + hl * 0.35, hy + hh * 0.3), (hx + hl * 0.75, hy + hh * 0.55), hh * 0.08, -0.15
            ),
            d.col("#f4efe2"),
        )
    if "antlers" in q.extras:
        ac = d.col("#8a6a48")
        base = (ex + er * 0.2, ey - er * 0.2)
        for side in (-1, 1):
            tip = (base[0] + side * hl * 0.25, base[1] - hh * 1.6)
            d.line(
                cv,
                d.bezier(base, (base[0] - hl * 0.3, base[1] - hh), tip, 10),
                ac,
                max(2, hh * 0.08),
            )
            for k in (0.4, 0.7):
                p0 = (base[0] + (tip[0] - base[0]) * k - hl * 0.1, base[1] + (tip[1] - base[1]) * k)
                d.line(cv, [p0, (p0[0] + hl * 0.2, p0[1] - hh * 0.35)], ac, max(2, hh * 0.07))
    if "horns" in q.extras:
        hc = d.col("#d9cdb4")
        d.poly(
            cv,
            d.leaf(
                (ex + er * 0.3, ey + er * 0.1), (ex - hl * 0.25, ey - hh * 0.45), hh * 0.12, -0.2
            ),
            hc,
        )
    if "beard" in q.extras:
        d.poly(
            cv,
            d.leaf((hx + hl * 0.3, hy + hh * 0.45), (hx + hl * 0.22, hy + hh * 0.95), hh * 0.18),
            d.col("#d8d0c0"),
        )


# -- people -----------------------------------------------------------------------------------
def _draw_human(hu: Human, c: _Ctx) -> None:
    s, cv = c.s * hu.height, c.cv
    rng = c.rng
    skin_rgb = d.rgb(hu.skin[int(rng.integers(len(hu.skin)))])
    skin = d.col(skin_rgb)
    hair = d.col(hu.hair[int(rng.integers(len(hu.hair)))])
    top_c = d.rgb(hu.top[int(rng.integers(len(hu.top)))])
    bot_c = d.rgb(hu.bottom[int(rng.integers(len(hu.bottom)))])
    shoe = d.col("#2a2420")
    gy, cx = c.gy, c.gx
    leg = 0.47 * s
    bob = (
        -abs(math.sin(2 * math.pi * c.phase)) * s * (0.025 if c.running else 0.012)
        if c.moving
        else math.sin(c.t * 1.5) * s * 0.003
    )
    hip = (cx, gy - leg + bob)
    lean = hu.stoop + (0.12 if c.running else 0.03 if c.moving else 0.0)
    torso_len = 0.3 * s
    neck = (hip[0] + math.sin(lean) * torso_len, hip[1] - math.cos(lean) * torso_len)
    stride = leg * (1.2 if c.running else 0.7)
    lift = leg * (0.3 if c.running else 0.15)
    lw = s * 0.045

    def leg_at(off: float, k: float) -> None:
        if c.moving:
            fx, fy = _foot(c.phase + off, stride, lift, 0.4 if c.running else 0.6)
        else:
            fx, fy = (0.05 if off else -0.03) * leg, 0.0
        foot = (hip[0] + fx, gy + fy)
        knee = _ik(hip, foot, leg * 0.5, leg * 0.5, -1.0)
        color = d.col(bot_c, k)
        d.limb(cv, hip, knee, lw, lw * 0.85, color)
        d.limb(
            cv,
            knee,
            foot,
            lw * 0.85,
            lw * 0.7,
            color if hu.style != "skirt" else d.col(skin_rgb, k),
        )
        d.ellipse(cv, foot[0] + lw * 0.5, foot[1] - lw * 0.35, lw * 1.05, lw * 0.5, shoe)

    def arm_at(off: float, k: float) -> None:
        swing = (
            (0.75 if c.running else 0.4) * math.sin(2 * math.pi * (c.phase + off))
            if c.moving
            else 0.05 * math.sin(c.t)
        )
        sh = (neck[0] - lw * 0.2, neck[1] + s * 0.03)
        al = 0.17 * s
        elbow = (sh[0] + math.sin(swing) * al, sh[1] + math.cos(swing) * al)
        fore = swing + (0.9 if c.running else 0.25)
        hand = (elbow[0] + math.sin(fore) * al, elbow[1] + math.cos(fore) * al)
        d.limb(cv, sh, elbow, lw * 0.9, lw * 0.75, d.col(top_c, k))
        d.limb(cv, elbow, hand, lw * 0.7, lw * 0.6, d.col(skin_rgb, k))
        if hu.stick and off == 0.0:
            d.line(cv, [hand, (hand[0] + s * 0.05, gy)], d.col("#6b4a2a"), max(2, lw * 0.5))

    leg_at(0.5, 0.8)
    arm_at(0.0, 0.8)
    if hu.long_hair:  # hair behind the shoulders
        d.ellipse(cv, neck[0] - s * 0.03, neck[1] - s * 0.02, s * 0.05, s * 0.08, hair)
    tw = s * 0.075
    d.poly(
        cv,
        [
            (hip[0] - tw, hip[1] + s * 0.02),
            (neck[0] - tw * 1.05, neck[1] + s * 0.02),
            (neck[0] + tw * 0.9, neck[1] + s * 0.02),
            (hip[0] + tw, hip[1] + s * 0.02),
        ],
        d.col(top_c),
    )
    if hu.style in ("skirt", "long"):
        hem = hip[1] + (0.22 if hu.style == "skirt" else 0.44) * s
        flare = s * (0.12 if hu.style == "skirt" else 0.1)
        sway = math.sin(2 * math.pi * c.phase) * s * 0.02 if c.moving else 0.0
        d.poly(
            cv,
            [
                (hip[0] - tw, hip[1] - s * 0.02),
                (hip[0] - flare + sway, hem),
                (hip[0] + flare + sway, hem),
                (hip[0] + tw, hip[1] - s * 0.02),
            ],
            d.col(bot_c if hu.style == "skirt" else top_c, 0.9),
        )
    leg_at(0.0, 1.0)
    hr = hu.head * s / hu.height
    head = (neck[0] + s * 0.015, neck[1] - hr * 1.15)
    d.limb(cv, neck, (head[0], head[1] + hr * 0.6), lw * 0.6, lw * 0.6, skin)
    d.circle(cv, head[0], head[1], hr, skin)
    d.poly(
        cv,
        [
            (head[0] + hr * 0.85, head[1] - hr * 0.1),
            (head[0] + hr * 1.18, head[1] + hr * 0.25),
            (head[0] + hr * 0.85, head[1] + hr * 0.35),
        ],
        skin,
    )
    d.ellipse(cv, head[0] - hr * 0.15, head[1] - hr * 0.35, hr * 1.02, hr * 0.7, hair)
    d.ellipse(cv, head[0] - hr * 0.6, head[1] - hr * 0.05, hr * 0.45, hr * 0.6, hair)
    blink = 0.2 if (c.t % 4.1) < 0.12 else 1.0
    d.ellipse(
        cv,
        head[0] + hr * 0.5,
        head[1] - hr * 0.05,
        max(1.2, hr * 0.1),
        max(1.0, hr * 0.12 * blink),
        d.col("#1a1412"),
    )
    d.line(
        cv,
        [(head[0] + hr * 0.55, head[1] + hr * 0.45), (head[0] + hr * 0.78, head[1] + hr * 0.42)],
        d.col("#8a4a3a"),
        max(1, hr * 0.07),
    )
    arm_at(0.5, 1.0)


# -- birds ------------------------------------------------------------------------------------
def _draw_bird(b: Bird, c: _Ctx, flying: bool) -> None:
    s, cv = c.s * 0.5 * b.size, c.cv
    cx = c.gx
    cy = c.gy - (0.0 if flying else s * 0.55)
    body_c, belly_c, wing_c = d.rgb(b.body), d.rgb(b.belly), d.rgb(b.wing)
    if flying:
        flap = math.sin(2 * math.pi * b.flap * c.t)
        if b.glide:  # long glides with a few flaps
            flap *= 0.25 + 0.75 * max(0.0, math.sin(c.t * 0.7)) ** 2
            flap = flap * 0.9 + 0.15
    else:
        flap = -1.0  # folded
    # tail
    tail_len = s * b.tail * (1.0 if not b.train else 3.2)
    if b.train and not flying:
        fan = 0.4 + 0.1 * math.sin(c.t)
        for i in range(9):
            a_ = math.pi + (i - 4) / 4 * fan
            tip = (
                cx - s * 0.3 + math.cos(a_) * tail_len,
                cy + math.sin(a_) * tail_len * 0.4 + s * 0.15,
            )
            d.poly(cv, d.leaf((cx - s * 0.3, cy), tip, s * 0.18), d.col("#2b6b5a"))
            d.circle(cv, tip[0] + s * 0.12, tip[1], s * 0.07, d.col("#e0b030"))
            d.circle(cv, tip[0] + s * 0.12, tip[1], s * 0.04, d.col("#1f3fa0"))
    else:
        d.poly(
            cv,
            [
                (cx - s * 0.3, cy - s * 0.04),
                (cx - s * 0.3 - tail_len, cy - s * 0.1),
                (cx - s * 0.3 - tail_len, cy + s * 0.08),
                (cx - s * 0.3, cy + s * 0.06),
            ],
            d.col(wing_c),
        )
    shoulder = (cx + s * 0.02, cy - s * 0.06)
    span = s * (1.1 if b.glide else 0.8)

    def wing(k: float, phase_shift: float) -> None:
        f = flap if flying else -1.0
        tip = (shoulder[0] - span * 0.45, shoulder[1] - span * 0.75 * (f + phase_shift))
        if not flying:
            tip = (shoulder[0] - s * 0.42, shoulder[1] + s * 0.06)
        d.poly(
            cv,
            d.leaf(shoulder, tip, s * (0.32 if flying else 0.22), 0.12 if flying else 0.05),
            d.col(wing_c, k),
        )

    if flying:
        wing(0.75, 0.15)
    if not flying:  # legs
        step = math.sin(2 * math.pi * c.phase) if c.moving else 0.0
        for k, sg in ((0.8, -1), (1.0, 1)):
            foot = (cx + sg * step * s * 0.15, c.gy)
            d.line(cv, [(cx, cy + s * 0.1), foot], d.col(b.legs, k), max(2, s * 0.03))
    if b.round:
        d.ellipse(cv, cx, cy - s * 0.05, s * 0.26, s * 0.32, d.col(body_c))
        d.ellipse(cv, cx + s * 0.06, cy + s * 0.04, s * 0.17, s * 0.22, d.col(belly_c))
        hc = (cx + s * 0.06, cy - s * 0.38)
        d.circle(cv, hc[0], hc[1], s * 0.2, d.col(body_c))
        for dx in (-0.08, 0.1):
            d.circle(cv, hc[0] + s * dx, hc[1], s * 0.07, d.col("#f2e6c8"))
            d.circle(cv, hc[0] + s * dx, hc[1], s * 0.035, d.col("#141010"))
        d.poly(
            cv,
            [
                (hc[0] + s * 0.02, hc[1] + s * 0.03),
                (hc[0] + s * 0.06, hc[1] + s * 0.1),
                (hc[0] + s * 0.09, hc[1] + s * 0.03),
            ],
            d.col(b.beak),
        )
    else:
        d.ellipse(cv, cx, cy, s * 0.33, s * 0.15, d.col(body_c), -8 if flying else -20)
        d.ellipse(
            cv,
            cx + s * 0.05,
            cy + s * 0.05,
            s * 0.24,
            s * 0.09,
            d.col(belly_c),
            -8 if flying else -20,
        )
        hx, hy = cx + s * 0.32, cy - s * (0.08 if flying else 0.2)
        if not flying:
            d.limb(
                cv,
                (cx + s * 0.18, cy - s * 0.02),
                (hx, hy),
                s * 0.1,
                s * 0.08,
                d.col(b.head or b.body),
            )
        d.circle(cv, hx, hy, s * 0.12, d.col(b.head or b.body))
        d.poly(
            cv,
            [
                (hx + s * 0.08, hy - s * 0.04),
                (hx + s * 0.28, hy + s * 0.02),
                (hx + s * 0.08, hy + s * 0.06),
            ],
            d.col(b.beak),
        )
        d.circle(cv, hx + s * 0.04, hy - s * 0.03, max(1.5, s * 0.025), d.col("#141010"))
        if b.crest:
            d.poly(
                cv,
                d.leaf((hx - s * 0.02, hy - s * 0.1), (hx - s * 0.06, hy - s * 0.28), s * 0.08),
                d.col(b.crest),
            )
    wing(1.0, 0.0)


# -- swimmers ---------------------------------------------------------------------------------
def _draw_swimmer(sw: Swimmer, c: _Ctx) -> None:
    s, cv = c.s, c.cv
    cx, cy = c.gx, c.gy - s * 0.3
    beat = math.sin(c.t * (6.0 if sw.kind == "fish" else 2.5))
    body_c, belly_c, fin_c = d.col(sw.color), d.col(sw.belly), d.col(sw.fin)
    if sw.kind == "turtle":
        for k, ph in ((0.8, 0.5), (1.0, 0.0)):
            a_ = math.sin(c.t * 2 + ph * 6.28) * 0.5
            for fx in (0.25, -0.25):
                tip = (
                    cx + s * fx + math.cos(a_ + (0.6 if fx > 0 else 2.5)) * s * 0.3,
                    cy + math.sin(a_ + 0.6) * s * 0.15 + s * 0.08,
                )
                d.poly(cv, d.leaf((cx + s * fx, cy + s * 0.05), tip, s * 0.12), d.col(sw.fin, k))
        d.circle(cv, cx + s * 0.48, cy, s * 0.1, d.col(sw.fin))
        d.circle(cv, cx + s * 0.52, cy - s * 0.02, max(1.5, s * 0.02), d.col("#141010"))
        d.ellipse(cv, cx, cy, s * 0.4, s * 0.2, body_c)
        for i in range(-2, 3):
            d.ellipse(cv, cx + i * s * 0.14, cy - s * 0.04, s * 0.06, s * 0.05, d.col(sw.belly))
        return
    length = s * (1.6 if sw.kind == "whale" else 1.0)
    tail = (cx - length * 0.55, cy + beat * s * 0.05)
    if sw.kind == "fish":
        d.poly(
            cv,
            [
                tail,
                (tail[0] - s * 0.22, tail[1] - s * 0.2 + beat * s * 0.05),
                (tail[0] - s * 0.22, tail[1] + s * 0.2 + beat * s * 0.05),
            ],
            fin_c,
        )
        d.poly(cv, d.leaf((cx - length * 0.55, cy), (cx + length * 0.5, cy), s * 0.42, 0.0), body_c)
        for i, st in enumerate(sw.stripes * 2):
            x = cx - length * 0.1 + i * length * 0.25
            d.poly(cv, d.leaf((x, cy - s * 0.2), (x, cy + s * 0.2), s * 0.07), d.col(st))
        d.circle(cv, cx + length * 0.3, cy - s * 0.04, max(2.0, s * 0.05), d.col("#ffffff"))
        d.circle(cv, cx + length * 0.31, cy - s * 0.04, max(1.2, s * 0.025), d.col("#141010"))
        return
    # dolphin / whale: tapered body, flukes, dorsal fin
    fl = s * (0.3 if sw.kind == "dolphin" else 0.45)
    d.poly(
        cv,
        d.leaf(tail, (tail[0] - fl * 0.6, tail[1] - fl * 0.7 - beat * fl * 0.3), fl * 0.35),
        fin_c,
    )
    d.poly(
        cv,
        d.leaf(tail, (tail[0] - fl * 0.6, tail[1] + fl * 0.5 - beat * fl * 0.3), fl * 0.3),
        fin_c,
    )
    th = s * (0.32 if sw.kind == "dolphin" else 0.55)
    d.poly(cv, d.leaf(tail, (cx + length * 0.55, cy + th * 0.05), th, -0.06), body_c)
    d.poly(
        cv,
        d.leaf(
            (cx - length * 0.2, cy + th * 0.15), (cx + length * 0.45, cy + th * 0.2), th * 0.45, 0.0
        ),
        belly_c,
    )
    if sw.kind == "dolphin":
        d.poly(
            cv,
            [
                (cx - length * 0.05, cy - th * 0.4),
                (cx - length * 0.18, cy - th * 0.95),
                (cx + length * 0.08, cy - th * 0.42),
            ],
            body_c,
        )
        d.poly(
            cv,
            d.leaf(
                (cx + length * 0.52, cy + th * 0.02),
                (cx + length * 0.68, cy + th * 0.12),
                th * 0.18,
            ),
            body_c,
        )
    d.circle(cv, cx + length * 0.38, cy - th * 0.05, max(2.0, th * 0.06), d.col("#141010"))
    d.poly(
        cv,
        d.leaf((cx + length * 0.15, cy + th * 0.2), (cx - length * 0.02, cy + th * 0.6), th * 0.22),
        fin_c,
    )


# -- the generator ----------------------------------------------------------------------------
class Actor:
    def __init__(self, p: dict[str, Any], w: int, h: int) -> None:
        species = str(p["species"])
        if species not in SPECIES:
            raise ValueError(
                f"Unknown species {species!r}; choose one of {', '.join(sorted(SPECIES))}."
            )
        self.p, self.w, self.h = p, w, h
        self.kind = SPECIES[species]
        self.species = species
        self.seed = int(p["seed"])
        self.size = max(4.0, float(p["size"]) * h)

    def _phase(self, t: float, speed_px: float) -> float:
        s = self.size
        if self.kind == "quad":
            q = QUADS[self.species]
            if q.gait == "hop":
                return t * max(1.2, abs(speed_px) / (q.leg * s * 3.0))
            run = self.p["action"] == "run"
            stride = q.leg * s * (1.5 if run else 0.85)
            duty = 0.6
        elif self.kind == "human":
            run = self.p["action"] == "run"
            stride = 0.47 * s * HUMANS[self.species].height * (1.2 if run else 0.7)
            duty = 0.4 if run else 0.6
        else:
            return t * 2.0
        # a planted foot moves `stride` backwards (relative to the body) during the stance part
        return t * abs(speed_px) * duty / max(1e-6, stride)

    def frame(self, t: float) -> Planes:
        p, w, h = self.p, self.w, self.h
        action = str(p["action"])
        speed_px = float(p["speed"]) * w if action != "idle" else 0.0
        moving = abs(speed_px) > 1e-6
        facing = str(p["facing"])
        left = facing == "left" if facing != "auto" else speed_px < 0
        x = float(p["x"]) * w + speed_px * t
        y = float(p["y"]) * h
        s = self.size
        cw, ch = int(s * 2.8) + 8, int(s * 1.5) + 8
        flying = self.kind == "bird" and action == "fly"
        swim = self.kind == "swim"
        if flying:
            y += math.sin(t * 1.1 + self.seed) * s * 0.08
        if swim and self.species == "dolphin" and moving:
            arc = math.sin(t * 1.4 + self.seed)
            y -= max(0.0, arc) * s * 0.9  # leaps
        gx, gy = cw / 2, ch - s * 0.12
        cv = d.canvas(cw, ch)
        if not flying and not swim:  # soft contact shadow
            d.ellipse(
                cv, gx, gy, s * (0.45 if self.kind != "human" else 0.22), s * 0.04, (0, 0, 0, 255)
            )
            shadow = cv[..., 3].astype(np.float32) / 255.0 * 0.28
            cv[:] = 0
        else:
            shadow = None
        ctx = _Ctx(cv, s, gx, gy, t, self._phase(t, speed_px), moving, action == "run",
                   np.random.default_rng(self.seed))  # fmt: skip
        if self.kind == "quad":
            _draw_quad(QUADS[self.species], ctx)
        elif self.kind == "human":
            _draw_human(HUMANS[self.species], ctx)
        elif self.kind == "bird":
            _draw_bird(BIRDS[self.species], ctx, flying)
        else:
            _draw_swimmer(SWIMMERS[self.species], ctx)
        if swim and self.species == "dolphin" and moving:
            ang = math.degrees(math.atan(math.cos(t * 1.4 + self.seed) * 0.9)) * 0.6
            m = cv2.getRotationMatrix2D((gx, gy - s * 0.3), ang, 1.0)
            cv = cv2.warpAffine(cv, m, (cw, ch), flags=cv2.INTER_LINEAR)
        srgb, sa = d.planes(cv)
        if shadow is not None:
            na = sa + shadow * (1 - sa)
            srgb = srgb * (sa / np.maximum(na, 1e-6))[..., None]
            sa = na
        if left:
            srgb, sa = srgb[:, ::-1], sa[:, ::-1]
        ox, oy = int(round(x - gx)), int(round(y - gy))
        surface = float(p["surface"])
        if surface > 0:  # below the water line: seen faintly through the water
            rows = np.arange(ch, dtype=np.float32) + oy
            under = np.clip((rows - surface * h) / max(2.0, s * 0.05), 0, 1)[:, None]
            sa = sa * (1 - 0.7 * under)
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        d.paste(
            rgb, a, np.ascontiguousarray(srgb), np.ascontiguousarray(sa, dtype=np.float32), ox, oy
        )
        return rgb, a
