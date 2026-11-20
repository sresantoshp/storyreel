"""Pick the footage that best shows a scene, with CLIP (storyreel.vision.clip).

For every candidate a preview picture is scored against the scene's narration and the
search words that found it:

    score = 0.6 * sim(preview, narration) + 0.4 * sim(preview, query)
            + (0.015 + 0.03 * share of query words in the clip's tags) * query priority

Candidates below MIN_SIM on both are rejected (a "dolphin" search that returns an
aquarium crowd), near-duplicates of an already chosen clip are skipped, and a scene's
shots are ordered wide → close, the way films introduce a place and then show detail.
Previews are cached next to the footage; local clips use a frame from their middle.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from storyreel.media.stock import Candidate, Clip, user_agent
from storyreel.vision.clip import F32, read_image

log = logging.getLogger(__name__)
RGB = npt.NDArray[np.uint8]
Grab = Callable[[Path], RGB | None]  # local video -> a frame from its middle

MIN_SIM = 0.215  # measured: matching previews score 0.26-0.33, unrelated 0.15-0.22
DUPLICATE = 0.93  # cosine between previews above this = the same shot
W_NARRATION, W_QUERY, W_ORDER = 0.6, 0.4, 0.015
# The library's own tags confirming the words of the search that found a clip; earlier
# (more specific) searches weigh more. CLIP alone ranks "waves" next to "dolphins in the
# waves"; the tags know which clip actually has dolphins (measured on Pixabay).
W_TAGS = 0.03
_WORDS = re.compile(r"[a-z]+")
# Tags that break a realistic film: never picked (drawn or rendered) ...
NOT_REAL = frozenset(
    {
        "animation",
        "animated",
        "cartoon",
        "illustration",
        "cgi",
        "3d",
        "render",
        "rendering",
        "anime",
        "drawing",
        "vector",
        "motion graphics",
    }
)
# ... or picked only when nothing better exists (animals in captivity, screens, text)
CAPTIVE = frozenset({"aquarium", "zoo", "exhibit", "captivity", "cage", "enclosure"})
W_CAPTIVE = 0.04
WIDE = "a wide establishing shot of a landscape"
CLOSE = "a close-up photo of a face or an animal"


class Embedder(Protocol):
    """What the ranker needs from CLIP (vision.clip.Clip implements it)."""

    def texts(self, texts: list[str]) -> F32: ...

    def images(self, rgbs: list[RGB]) -> F32: ...


@dataclass
class Scored:
    cand: Candidate
    score: float
    emb: F32


def _tags(c: Candidate) -> set[str]:
    """The clip's tags as single lower-case words plus whole phrases."""
    phrases = {t.strip().lower() for t in c.tags.split(",") if t.strip()}
    return phrases | {w for ph in phrases for w in _WORDS.findall(ph)}


def _tag_match(query: str, c: Candidate) -> float:
    """Share of the query's words found in the clip's tags (or file name), plural-blind."""
    words = {w.rstrip("s") for w in _WORDS.findall(query.lower()) if len(w) > 2}
    if not words:
        return 0.0
    have = {w.rstrip("s") for w in _WORDS.findall((c.tags or c.key).lower())}
    return len(words & have) / len(words)


class Ranker:
    def __init__(self, clip: Embedder, cache_dir: Path, grab: Grab, *, offline: bool = False):
        self.clip = clip
        self.dir = cache_dir / "thumbs"
        self.grab = grab
        self.offline = offline
        self._emb: dict[str, F32] = {}
        self._wide: F32 | None = None

    # -- previews ------------------------------------------------------------------
    def _thumb_path(self, c: Candidate) -> Path:
        h = hashlib.sha1(c.key.encode()).hexdigest()[:20]  # noqa: S324 (file name only)
        return self.dir / f"{h}.jpg"

    def preview(self, c: Candidate) -> RGB | None:
        p = self._thumb_path(c)
        if p.is_file():
            return read_image(p)
        if c.key.startswith("folder:"):
            img = self.grab(Path(c.url))
            if img is not None:
                import cv2

                p.parent.mkdir(parents=True, exist_ok=True)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if ok:
                    p.write_bytes(buf.tobytes())
            return img
        if not c.thumb.startswith("https://") or self.offline:
            return None
        try:
            req = urllib.request.Request(c.thumb, headers={"User-Agent": user_agent(c.thumb)})  # noqa: S310
            with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310 (https checked)
                data = r.read(5 * 2**20)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("preview %s: %s", c.key, exc)
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return read_image(p)

    def _embed(self, cands: list[Candidate]) -> list[F32 | None]:
        todo = [c for c in cands if c.key not in self._emb]
        imgs, keys = [], []
        for c in todo:
            img = self.preview(c)
            if img is not None:
                imgs.append(img)
                keys.append(c.key)
        for i in range(0, len(imgs), 16):
            for k, e in zip(keys[i : i + 16], self.clip.images(imgs[i : i + 16]), strict=True):
                self._emb[k] = e
        return [self._emb.get(c.key) for c in cands]

    # -- ranking ---------------------------------------------------------------------
    def rank(self, describe: str, pool: list[tuple[int, str, Candidate]]) -> list[Scored]:
        """pool = (query index, query, candidate). Best first, weak and duplicate removed."""
        if not pool:
            return []
        queries = sorted({q for _, q, _ in pool})
        qemb = dict(zip(queries, self.clip.texts([f"a photo of {q}" for q in queries]),
                        strict=True))  # fmt: skip
        demb = self.clip.texts([describe])[0] if describe.strip() else None
        n_q = max(i for i, _, _ in pool) + 1
        scored: list[Scored] = []
        pool = [(qi, q, c) for qi, q, c in pool if not (_tags(c) & NOT_REAL)]
        if not pool:
            return []
        for (qi, q, c), e in zip(pool, self._embed([c for _, _, c in pool]), strict=True):
            if e is None:
                continue
            sq = float(e @ qemb[q])
            sd = float(e @ demb) if demb is not None else sq
            if max(sq, sd) < MIN_SIM:
                continue
            bonus = (W_ORDER + W_TAGS * _tag_match(q, c)) * (n_q - qi) / n_q
            if _tags(c) & CAPTIVE:
                bonus -= W_CAPTIVE
            scored.append(Scored(c, W_NARRATION * sd + W_QUERY * sq + bonus, e))
        scored.sort(key=lambda s: -s.score)
        out: list[Scored] = []
        for s in scored:
            if all(float(s.emb @ o.emb) < DUPLICATE for o in out):
                out.append(s)
        return out

    def wide_first(self, clips: list[Clip], scored: dict[str, Scored]) -> list[Clip]:
        """Order a scene's shots from the widest to the closest."""
        if len(clips) < 2:
            return clips
        if self._wide is None:
            t = self.clip.texts([WIDE, CLOSE])
            self._wide = (t[0] - t[1]).astype(np.float32)
        wide = self._wide

        def width(c: Clip) -> float:
            s = scored.get(c.key)
            return float(s.emb @ wide) if s is not None else 0.0

        return sorted(clips, key=lambda c: -width(c))
