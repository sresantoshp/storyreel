"""Timeline → chunk plans. Pure data; no ffmpeg, no I/O except reading the project DB.

Chunking: walk the timeline in frames; each chunk ends at the farthest edit point within
`chunk_seconds`, or at the limit if there is none. Because chunks start at edit points
whenever possible, an edit that only shifts later clips reproduces the same chunks (and
the same cache keys) after the first edit point that follows it.

Cache keys use times relative to the chunk start and asset fingerprints (not paths).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from storyreel.errors import StoryReelError
from storyreel.project.store import Project
from storyreel.timebase import FLICKS, Rate

ENGINE_VERSION = "1"  # bump when render output changes for the same inputs
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
    acodec: str = "aac"
    abitrate: str = "192k"
    chunk_seconds: float = 10.0

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
    kind: str  # media | color
    start: int  # flicks
    duration: int
    src_in: int
    gain_db: float
    color: str | None
    asset_kind: str | None  # video | image | audio
    path: str | None
    fingerprint: str | None


@dataclass(frozen=True)
class VLayer:
    kind: str  # video | image | color
    at: int  # first frame within the chunk
    frames: int
    src_in: int  # flicks into the source at the layer's first frame (video only)
    color: str | None
    fingerprint: str | None
    path: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ALayer:
    at: int  # first sample within the chunk
    samples: int
    src_in: int  # flicks
    gain_db: float
    fingerprint: str
    path: str = field(default="", compare=False)


@dataclass(frozen=True)
class ChunkPlan:
    idx: int
    f0: int
    f1: int
    s0: int
    s1: int
    vlayers: tuple[VLayer, ...]  # bottom to top
    alayers: tuple[ALayer, ...]

    @property
    def frames(self) -> int:
        return self.f1 - self.f0

    @property
    def samples(self) -> int:
        return self.s1 - self.s0

    def _key(self, kind: str, settings: dict[str, Any], layers: tuple[Any, ...], n: int) -> str:
        payload = {
            "engine": ENGINE_VERSION,
            "kind": kind,
            "settings": settings,
            "n": n,
            "layers": [{k: v for k, v in asdict(layer).items() if k != "path"} for layer in layers],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=20).hexdigest()

    def video_key(self, s: RenderSettings) -> str:
        return self._key("v", s.video_key(), self.vlayers, self.frames)

    def audio_key(self, s: RenderSettings) -> str:
        return self._key("a", s.audio_key(), self.alayers, self.samples)


def load_timeline(project: Project) -> list[TimelineClip]:
    rows = project.con.execute(
        """SELECT c.id, t.kind AS tkind, t.idx, c.kind, c.start, c.duration, c.src_in,
                  c.speed_num, c.speed_den, c.gain_db, c.props,
                  a.kind AS akind, a.path, a.fingerprint, a.status, a.name
           FROM clip c JOIN track t ON t.id = c.track_id
           LEFT JOIN asset a ON a.id = c.asset_id
           WHERE t.muted = 0
           ORDER BY t.kind, t.idx, c.start"""
    ).fetchall()
    out: list[TimelineClip] = []
    for r in rows:
        if (r["speed_num"], r["speed_den"]) != (1, 1):
            raise StoryReelError(
                f"Clip {r['id']} has a speed change; speed ramps arrive with the Phase 2 "
                "compositor and the Phase 1 renderer refuses rather than rendering it wrong."
            )
        if r["kind"] == "media" and r["status"] != "ok":
            raise StoryReelError(
                f"Clip {r['id']} uses {r['name']}, which is {r['status']}.",
                fix="Relink or remove the clip.",
            )
        color = None
        if r["kind"] == "color":
            m = _HEX.match(str(json.loads(r["props"]).get("color", "")))
            if m is None:
                raise StoryReelError(f"Clip {r['id']} has an invalid color.")
            color = m.group(1).lower()
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
            )
        )
    return out


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


def plan_chunks(clips: list[TimelineClip], s: RenderSettings) -> list[ChunkPlan]:
    rate, sr = s.rate, s.sample_rate
    vclips = [c for c in clips if c.track_kind == "video"]
    aclips = [c for c in clips if c.track_kind == "audio"]
    vspan = [
        (rate.flicks_to_frame(c.start), rate.flicks_to_frame(c.start + c.duration), c)
        for c in vclips
    ]
    edits = [f for a, b, _ in vspan for f in (a, b)]
    edits += [rate.flicks_to_frame(t) for c in aclips for t in (c.start, c.start + c.duration)]
    total = max(edits, default=0)
    if total == 0:
        return []
    aspan = [(_sample_of(c.start, sr), _sample_of(c.start + c.duration, sr), c) for c in aclips]
    vspan.sort(key=lambda x: (x[2].track_idx, x[0]))

    plans: list[ChunkPlan] = []
    for idx, (f0, f1) in enumerate(chunk_bounds(edits, total, s.chunk_frames)):
        vl: list[VLayer] = []
        for a, b, c in vspan:
            o0, o1 = max(a, f0), min(b, f1)
            if o0 >= o1:
                continue
            src = c.src_in + rate.frame_to_flicks(o0) - rate.frame_to_flicks(a)
            kind = "color" if c.kind == "color" else str(c.asset_kind)
            vl.append(
                VLayer(
                    kind,
                    o0 - f0,
                    o1 - o0,
                    src if kind == "video" else 0,
                    c.color,
                    c.fingerprint,
                    c.path,
                )
            )
        s0, s1 = rate.frame_to_sample(f0, sr), rate.frame_to_sample(f1, sr)
        al: list[ALayer] = []
        for a, b, c in aspan:
            o0, o1 = max(a, s0), min(b, s1)
            if o0 >= o1 or c.fingerprint is None or c.path is None:
                continue
            src = c.src_in + (o0 - a) * FLICKS // sr
            al.append(ALayer(o0 - s0, o1 - o0, src, c.gain_db, c.fingerprint, c.path))
        plans.append(ChunkPlan(idx, f0, f1, s0, s1, tuple(vl), tuple(al)))
    return plans
