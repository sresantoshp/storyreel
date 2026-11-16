"""Timeline → chunk plans. Pure data; no ffmpeg, no I/O except reading the project DB.

Chunking: walk the timeline in frames; each chunk ends at the farthest edit point within
`chunk_seconds`, or at the limit if there is none. Because chunks start at edit points
whenever possible, an edit that only shifts later clips reproduces the same chunks (and
the same cache keys) after the first edit point that follows it.

Two render paths per chunk:
  A (ffmpeg filtergraph): every layer is a plain video/image/color clip at 1x speed with
    no transform, keyframes, effects or transitions. Fastest.
  B (numpy compositor in a worker process): anything else. Layer details travel in
    `VLayer.extra` (canonical JSON), which is also part of the cache key.

Cache keys use times relative to the chunk / clip start and asset fingerprints (not paths).
Every frame is a pure function of its inputs, so transitions may straddle chunk boundaries.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from storyreel.audio import spec as aspec
from storyreel.errors import StoryReelError
from storyreel.project.store import Project
from storyreel.timebase import FLICKS, Rate

ENGINE_VERSION = "3"  # manual bump for changes outside the fingerprinted code (e.g. ffmpeg)
AUDIO_VERSION = "1"  # Phase 3 audio graph (buses, automation, effects, pre-roll)


@functools.lru_cache(maxsize=1)
def engine_fingerprint() -> str:
    """Hash of every source file that decides what a frame or a sample looks like.

    Part of every cache key, so editing a generator, effect, text, audio or compositor
    module invalidates old chunks automatically. A cosmetic edit costs a re-render; a
    forgotten manual version bump would have served stale output, which is worse.
    """
    pkg = Path(__file__).resolve().parent.parent  # src/storyreel
    h = hashlib.blake2b(ENGINE_VERSION.encode(), digest_size=12)
    for sub in ("render", "text", "audio"):
        for f in sorted((pkg / sub).glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class RenderSettings:
    width: int
    height: int
    rate: Rate
    sample_rate: int = 48_000
    channels: int = 2
    vcodec: str = "libx264"
    preset: str = "veryfast"
    crf: int = 20
    pix_fmt: str = "yuv420p"
    acodec: str = "aac"  # aac | ac3 | eac3 | opus | flac (audio.spec.CODECS)
    abitrate: str = ""  # "" = codec default for the channel count
    chunk_seconds: float = 10.0
    force_path_b: bool = False  # tests: render simple chunks through the compositor too
    target_lufs: float | None = -14.0  # None = no loudness normalisation
    true_peak: float = -1.0  # dBTP ceiling of the final encode
    out_channels: int | None = None  # downmix the final encode (e.g. 5.1 project -> 2)
    stems: str | None = None  # "flac" | "wav": per-bus stems next to the output

    @property
    def chunk_frames(self) -> int:
        return max(1, int(self.chunk_seconds * self.rate.num / self.rate.den))

    def video_key(self) -> dict[str, Any]:
        return {
            "w": self.width,
            "h": self.height,
            "r": str(self.rate),
            "c": self.vcodec,
            "p": self.preset,
            "crf": self.crf,
            "pf": self.pix_fmt,
        }

    def audio_key(self) -> dict[str, Any]:
        return {"sr": self.sample_rate, "ch": self.channels}


@dataclass(frozen=True)
class TimelineClip:
    clip_id: int
    track_kind: str  # video | audio
    track_idx: int
    kind: str  # media | color | text | generator | adjust
    start: int  # flicks
    duration: int
    src_in: int
    gain_db: float
    color: str | None
    asset_kind: str | None  # video | image | audio
    path: str | None
    fingerprint: str | None
    speed: tuple[int, int] = (1, 1)
    props: dict[str, Any] = field(default_factory=dict)
    keyframes: dict[str, list[list[float]]] = field(default_factory=dict)
    effects: list[dict[str, Any]] = field(default_factory=list)
    asset: dict[str, Any] = field(default_factory=dict)
    trans_in: dict[str, Any] | None = None  # this clip is B of a transition
    trans_out: dict[str, Any] | None = None  # this clip is A of a transition
    bus: str = "dialogue"  # audio track bus

    @property
    def simple(self) -> bool:
        extra_props = {k for k in self.props if k != "color"}
        return (
            self.kind in ("media", "color")
            and not extra_props
            and not self.keyframes
            and not self.effects
            and self.speed == (1, 1)
            and self.trans_in is None
            and self.trans_out is None
        )


@dataclass(frozen=True)
class VLayer:
    kind: str  # video | image | color | text | generator | adjust
    at: int  # first frame within the chunk
    frames: int
    src_in: int  # path A: flicks at the layer's first frame; path B: clip src_in
    color: str | None
    fingerprint: str | None
    extra: str = ""  # path B: canonical JSON (clip offset, props, keyframes, effects, ...)
    matrix: str = ""  # path A video: YUV matrix of the source (bt709 / bt601 / bt2020)
    path: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ALayer:
    at: int  # first sample within the chunk
    samples: int
    src_in: int  # flicks: source position where this layer's audio starts (pre-reverse)
    gain_db: float
    fingerprint: str
    path: str = field(default="", compare=False)
    speed: tuple[int, int] = (1, 1)
    reverse: bool = False
    bus: str = "dialogue"
    pan: float = 0.0
    env: str = ""  # canonical JSON gain automation (clip-relative), "" = none
    effects: str = ""  # canonical JSON effect list, "" = none
    src_ch: int = 0  # source channel count (0 = unknown): mono → FC, stereo → FL/FR in 5.1+


@dataclass(frozen=True)
class ChunkPlan:
    idx: int
    f0: int
    f1: int
    s0: int
    s1: int
    vlayers: tuple[VLayer, ...]  # bottom to top
    alayers: tuple[ALayer, ...]  # `at` is relative to the audio window start (s0 - preroll)
    preroll: int = 0  # audio samples rendered before s0 and trimmed (warm filter state)
    mix: str = ""  # canonical JSON: bus config + dialogue intervals (window-relative)

    @property
    def frames(self) -> int:
        return self.f1 - self.f0

    @property
    def samples(self) -> int:
        return self.s1 - self.s0

    @property
    def path_b(self) -> bool:
        return any(layer.extra for layer in self.vlayers)

    def _key(
        self,
        kind: str,
        settings: dict[str, Any],
        layers: tuple[Any, ...],
        n: int,
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "engine": engine_fingerprint(),
            "kind": kind,
            "settings": settings,
            "n": n,
            "layers": [{k: v for k, v in asdict(layer).items() if k != "path"} for layer in layers],
        }
        if extra:
            payload["extra"] = extra
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=20).hexdigest()

    def video_key(self, s: RenderSettings) -> str:
        return self._key("v", s.video_key(), self.vlayers, self.frames)

    def audio_key(self, s: RenderSettings) -> str:
        extra = {"audio": AUDIO_VERSION, "preroll": self.preroll, "mix": self.mix}
        return self._key("a", s.audio_key(), self.alayers, self.samples, extra)

    def worker_layers(self) -> list[dict[str, Any]]:
        """Layer specs for the path-B worker (paths included)."""
        out = []
        for layer in self.vlayers:
            if not layer.extra:
                raise StoryReelError("internal: simple layer in a path-B chunk")
            spec: dict[str, Any] = json.loads(layer.extra)
            spec |= {
                "kind": layer.kind,
                "at": layer.at,
                "frames": layer.frames,
                "src_in": layer.src_in,
                "path": layer.path,
            }
            out.append(spec)
        return out


def color_matrix(height: int | None, tag: str | None) -> str:
    """YUV matrix for a source: its tag if known, else BT.709 for >= 720p, BT.601 below."""
    tags = {
        "bt709": "bt709",
        "smpte170m": "bt601",
        "bt470bg": "bt601",
        "bt2020nc": "bt2020",
        "bt2020c": "bt2020",
    }
    if tag in tags:
        return tags[tag]
    return "bt709" if (height or 1080) >= 720 else "bt601"


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def load_timeline(project: Project) -> list[TimelineClip]:
    con = project.con
    kfs: dict[int, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for k in con.execute("SELECT * FROM keyframe ORDER BY clip_id, param, t"):
        kfs[k["clip_id"]][k["param"]].append(
            [k["t"], k["value"], k["interp"], k["h1x"], k["h1y"], k["h2x"], k["h2y"]]
        )
    fxs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in con.execute("SELECT * FROM effect WHERE enabled = 1 ORDER BY clip_id, idx"):
        fxs[e["clip_id"]].append(
            {"id": e["id"], "type": e["type"], "params": json.loads(e["params"])}
        )
    t_in: dict[int, dict[str, Any]] = {}
    t_out: dict[int, dict[str, Any]] = {}
    for t in con.execute("SELECT * FROM transition"):
        row = {"kind": t["kind"], "duration": t["duration"]}
        t_out[t["clip_a"]] = row
        t_in[t["clip_b"]] = row

    rows = con.execute(
        """SELECT c.id, t.kind AS tkind, t.idx, t.bus, c.kind, c.start, c.duration, c.src_in,
                  c.speed_num, c.speed_den, c.gain_db, c.props,
                  a.kind AS akind, a.path, a.fingerprint, a.status, a.name, a.width, a.height,
                  a.fps_num, a.fps_den, a.duration AS adur, a.color_space, a.pix_fmt,
                  a.channels AS achannels
           FROM clip c JOIN track t ON t.id = c.track_id
           LEFT JOIN asset a ON a.id = c.asset_id
           WHERE t.muted = 0
           ORDER BY t.kind, t.idx, c.start"""
    ).fetchall()
    out: list[TimelineClip] = []
    for r in rows:
        if r["kind"] == "media" and r["status"] != "ok":
            raise StoryReelError(
                f"Clip {r['id']} uses {r['name']}, which is {r['status']}.",
                fix="Relink or remove the clip.",
            )
        props = json.loads(r["props"])
        color = None
        if r["kind"] == "color":
            m = _HEX.match(str(props.get("color", "")))
            if m is None:
                raise StoryReelError(f"Clip {r['id']} has an invalid color.")
            color = m.group(1).lower()
        effects = fxs.get(r["id"], [])
        if any(e["type"] == "lut" for e in effects) and r["akind"] not in ("video", "image"):
            raise StoryReelError(
                f"Clip {r['id']}: LUTs apply to video and image clips only.",
                fix="Put the LUT on the media clip, not on text/adjustment.",
            )
        asset = (
            {
                "kind": r["akind"],
                "width": r["width"],
                "height": r["height"],
                "fps_num": r["fps_num"],
                "fps_den": r["fps_den"],
                "duration": r["adur"],
                "color_space": r["color_space"],
                "pix_fmt": r["pix_fmt"],
                "channels": r["achannels"],
            }
            if r["akind"]
            else {}
        )
        out.append(
            TimelineClip(
                r["id"],
                r["tkind"],
                r["idx"],
                r["kind"],
                r["start"],
                r["duration"],
                r["src_in"],
                r["gain_db"],
                color,
                r["akind"],
                r["path"],
                r["fingerprint"],
                (r["speed_num"], r["speed_den"]),
                props,
                {k: v for k, v in kfs.get(r["id"], {}).items()},
                effects,
                asset,
                t_in.get(r["id"]),
                t_out.get(r["id"]),
                r["bus"],
            )
        )
    return out


def load_mix(project: Project) -> dict[str, Any]:
    """Bus settings and bus effect chains: {"buses": {name: {...}}}."""
    con = project.con
    buses: dict[str, dict[str, Any]] = {}
    for r in con.execute("SELECT * FROM bus"):
        params = {**aspec.BUS_PARAMS, **json.loads(r["params"])}
        fx = [
            {"type": e["type"], "params": {**aspec.EFFECTS[e["type"]], **json.loads(e["params"])}}
            for e in con.execute(
                "SELECT * FROM bus_effect WHERE bus_id = ? AND enabled = 1 ORDER BY idx",
                (r["id"],),
            )
        ]
        buses[r["name"]] = {
            "gain_db": float(r["gain_db"]),
            "muted": bool(r["muted"]),
            "fx": fx,
            **params,
        }
    return {"buses": buses}


def _sample_of(t: int, sr: int) -> int:
    q, r = divmod(t * sr, FLICKS)
    return q + (1 if 2 * r >= FLICKS else 0)


def chunk_bounds(edit_frames: list[int], total: int, max_frames: int) -> list[tuple[int, int]]:
    points = sorted({p for p in edit_frames if 0 < p < total} | {total})
    out: list[tuple[int, int]] = []
    s = 0
    while s < total:
        limit = s + max_frames
        i = bisect.bisect_right(points, limit) - 1
        e = points[i] if i >= 0 and points[i] > s else min(limit, total)
        out.append((s, e))
        s = e
    return out


@dataclass(frozen=True)
class _VSpan:
    lo: int  # visible frames [lo, hi) including transition extensions
    hi: int
    a: int  # clip start frame (clip-local 0)
    clip_frames: int
    transitions: tuple[dict[str, Any], ...]
    clip: TimelineClip


def _vspan(c: TimelineClip, rate: Rate) -> _VSpan:
    a = rate.flicks_to_frame(c.start)
    b = rate.flicks_to_frame(c.start + c.duration)
    lo, hi = a, b
    trans: list[dict[str, Any]] = []
    if c.trans_out is not None:
        d = max(1, rate.flicks_to_frame(c.trans_out["duration"]))
        hi = b + (d - d // 2)
        trans.append(
            {"kind": c.trans_out["kind"], "role": "out", "start": b - d // 2 - a, "frames": d}
        )
    if c.trans_in is not None:
        d = max(1, rate.flicks_to_frame(c.trans_in["duration"]))
        lo = a - d // 2
        trans.append({"kind": c.trans_in["kind"], "role": "in", "start": -(d // 2), "frames": d})
    return _VSpan(max(lo, 0), hi, a, b - a, tuple(trans), c)


def _vlayer(sp: _VSpan, o0: int, o1: int, f0: int, rate: Rate, force_b: bool) -> VLayer:
    c = sp.clip
    kind = c.kind if c.kind != "media" else str(c.asset_kind)
    if c.simple and not force_b:
        src = c.src_in + rate.frame_to_flicks(o0) - rate.frame_to_flicks(sp.a)
        matrix = (
            color_matrix(c.asset.get("height"), c.asset.get("color_space"))
            if kind == "video"
            else ""
        )
        return VLayer(
            kind,
            o0 - f0,
            o1 - o0,
            src if kind == "video" else 0,
            c.color,
            c.fingerprint,
            "",
            matrix,
            c.path,
        )
    extra = {
        "clip_offset": o0 - sp.a,
        "clip_frames": sp.clip_frames,
        "speed": list(c.speed),
        "props": c.props,
        "keyframes": c.keyframes,
        "effects": c.effects,
        "transitions": list(sp.transitions),
        "asset": c.asset,
    }
    if c.kind == "color":
        extra["props"] = {**c.props, "color": f"#{c.color}"}
    return VLayer(
        kind, o0 - f0, o1 - o0, c.src_in, c.color, c.fingerprint, _canon(extra), "", c.path
    )


def _atempo(speed: float) -> str:
    parts = []
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    while speed > 100.0:
        parts.append("atempo=100")
        speed /= 100.0
    if abs(speed - 1.0) > 1e-9:
        parts.append(f"atempo={speed:.6f}")
    return ",".join(parts)


def _audio_env(c: TimelineClip, a: int, b: int, o0: int, sr: int) -> str:
    fi = round(float(c.props.get("fade_in", 0.0)) * sr)
    fo = round(float(c.props.get("fade_out", 0.0)) * sr)
    keys = c.keyframes.get("volume", [])
    if not (fi or fo or keys):
        return ""
    return _canon({"off": o0 - a, "len": b - a, "fi": fi, "fo": fo, "kf": keys})


def _audio_effects(c: TimelineClip) -> str:
    fx = [
        {"type": e["type"], "params": {**aspec.EFFECTS[e["type"]], **e["params"]}}
        for e in c.effects
        if e["type"] in aspec.EFFECTS
    ]
    return _canon(fx) if fx else ""


def preroll_seconds(aclips: list[TimelineClip], mix: dict[str, Any] | None) -> float:
    """0 when no audio effect is used anywhere; else warm-up + the longest reverb tail."""
    fx: list[dict[str, Any]] = [e for c in aclips for e in c.effects if e["type"] in aspec.EFFECTS]
    if mix:
        fx += [e for cfg in mix["buses"].values() for e in cfg["fx"]]
    if not fx:
        return 0.0
    tails = [aspec.reverb_tail_s(str(e["params"].get("room", "hall"))) for e in fx
             if e["type"] == "reverb"]  # fmt: skip
    return aspec.WARMUP_S + max(tails, default=0.0)


def plan_chunks(
    clips: list[TimelineClip],
    s: RenderSettings,
    warnings: list[str] | None = None,
    mix: dict[str, Any] | None = None,
) -> list[ChunkPlan]:
    """`mix` = load_mix(project); None = all buses at unity, no effects, no ducking."""
    rate, sr = s.rate, s.sample_rate
    vclips = [c for c in clips if c.track_kind == "video"]
    aclips = [c for c in clips if c.track_kind == "audio"]
    vspans = [_vspan(c, rate) for c in vclips]
    vspans.sort(key=lambda v: (v.clip.track_idx, v.a))
    edits = [f for v in vspans for f in (v.lo, v.hi)]
    edits += [rate.flicks_to_frame(t) for c in aclips for t in (c.start, c.start + c.duration)]
    total = max(edits, default=0)
    if total == 0:
        return []

    aspans: list[tuple[int, int, TimelineClip]] = []
    for c in aclips:
        if c.props.get("freeze") or "speed" in c.keyframes:
            if warnings is not None:
                why = "frozen" if c.props.get("freeze") else "has a speed ramp"
                warnings.append(
                    f"Audio clip {c.clip_id} {why}; its sound is muted "
                    "(ramped audio arrives with the Phase 3 audio engine)."
                )
            continue
        aspans.append((_sample_of(c.start, sr), _sample_of(c.start + c.duration, sr), c))
    buses = (mix or {}).get("buses", {})
    muted = {name for name, cfg in buses.items() if cfg.get("muted")}
    aspans = [sp for sp in aspans if sp[2].bus not in muted]
    pre = round(preroll_seconds([sp[2] for sp in aspans], mix) * sr)
    speech = sorted((a, b) for a, b, c in aspans if c.bus == "dialogue")
    ducking = [cfg for name, cfg in buses.items() if cfg.get("duck") and name != "dialogue"]
    margin_before = round(max((float(c["duck_attack"]) for c in ducking), default=0.0) * sr)
    margin_after = round(max((float(c["duck_release"]) for c in ducking), default=0.0) * sr)

    plans: list[ChunkPlan] = []
    for idx, (f0, f1) in enumerate(chunk_bounds(edits, total, s.chunk_frames)):
        vl: list[VLayer] = []
        for sp in vspans:
            o0, o1 = max(sp.lo, f0), min(sp.hi, f1)
            if o0 < o1:
                vl.append(_vlayer(sp, o0, o1, f0, rate, s.force_path_b))
        if any(layer.extra for layer in vl) and not all(layer.extra for layer in vl):
            # mixed chunk: express simple layers for the compositor too
            vl = [
                _vlayer(sp, max(sp.lo, f0), min(sp.hi, f1), f0, rate, True)
                for sp in vspans
                if max(sp.lo, f0) < min(sp.hi, f1)
            ]
        s0, s1 = rate.frame_to_sample(f0, sr), rate.frame_to_sample(f1, sr)
        w0 = max(0, s0 - pre)
        al: list[ALayer] = []
        for a, b, c in aspans:
            o0, o1 = max(a, w0), min(b, s1)
            if o0 >= o1 or c.fingerprint is None or c.path is None:
                continue
            speed = Fraction(*c.speed)
            if c.props.get("reverse"):
                src = c.src_in + int((b - o1) * speed * FLICKS / sr)
            else:
                src = c.src_in + int((o0 - a) * speed * FLICKS / sr)
            al.append(
                ALayer(
                    o0 - w0,
                    o1 - o0,
                    src,
                    c.gain_db,
                    c.fingerprint,
                    c.path,
                    c.speed,
                    bool(c.props.get("reverse")),
                    c.bus,
                    float(c.props.get("pan", 0.0)),
                    _audio_env(c, a, b, o0, sr),
                    _audio_effects(c),
                    int(c.asset.get("channels") or 0),
                )
            )
        mix_json = ""
        if mix is not None:
            near = [
                [a - w0, b - w0]
                for a, b in speech
                if b + margin_after > w0 and a - margin_before < s1
            ]
            mix_json = _canon({"buses": buses, "speech": near if ducking else []})
        plans.append(ChunkPlan(idx, f0, f1, s0, s1, tuple(vl), tuple(al), s0 - w0, mix_json))
    return plans


def atempo_chain(speed: tuple[int, int]) -> str:
    return _atempo(speed[0] / speed[1])
