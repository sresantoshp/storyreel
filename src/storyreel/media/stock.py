"""Real video footage for the realistic style: free stock libraries + your own clip folder.

Providers (all free; licences allow commercial use and YouTube monetisation):
  folder   your own clips (any .mp4/.mov/.mkv/.webm); matched by words in file/folder names.
  pexels   Pexels Videos API, key in PEXELS_API_KEY (free account). Pexels License: free
           for commercial use, no attribution required (credit is appreciated).
  pixabay  Pixabay Videos API, key in PIXABAY_API_KEY (free account). Pixabay Content
           License: free for commercial use, no attribution required.
  commons  Wikimedia Commons, no key, OPT-IN ONLY (--footage commons). Only public-domain,
           CC0 and CC BY files (credit in footage_credits.txt); share-alike is skipped.
           Wikimedia requires a contact in the User-Agent: set STORYREEL_CONTACT (your
           email or site); without it downloads are refused (HTTP 429). Results are NOT
           curated (war, rituals, medical footage appear), so check every shot.

Network use is optional: every search result and every downloaded file is cached under
<data dir>/stock, so a story made once re-makes fully offline (`offline=True` never
touches the network and uses only the cache). Only the search words leave the machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from storyreel.errors import StoryReelError
from storyreel.paths import data_dir

if TYPE_CHECKING:
    from storyreel.media.rank import Ranker

log = logging.getLogger(__name__)

USER_AGENT = "StoryReel/0.1 (offline video maker)"
MAX_DOWNLOAD_MB = 150
# Bump when Candidate gains fields: older cached searches are then fetched again online.
SEARCH_CACHE_VERSION = "v2"  # v2: preview thumbnails + library tags
RANKED_QUERIES = 4  # searches pooled per scene when ranking with CLIP
PER_QUERY = 12  # candidates per search that get a preview scored
COMMONS_MAX_S = 180.0
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v")
PEXELS_LICENCE = "Pexels License (free commercial use, no attribution required)"
PIXABAY_LICENCE = "Pixabay Content License (free commercial use, no attribution required)"


class StockError(StoryReelError):
    pass


@dataclass(frozen=True)
class Candidate:
    """One search hit, before download."""

    key: str  # provider:id, unique per source video
    width: int
    height: int
    duration_s: float
    url: str  # file to download (or local path for the folder provider)
    page_url: str
    author: str
    licence: str
    thumb: str = ""  # preview picture URL (for CLIP ranking)
    tags: str = ""  # the library's own keywords, comma separated


@dataclass(frozen=True)
class Clip:
    """A footage file on disk, ready to import."""

    key: str
    path: Path
    width: int
    height: int
    duration_s: float
    page_url: str
    author: str
    licence: str
    tags: str = ""  # library keywords (or the file name), used to pick matching sound


class Provider(Protocol):
    name: str

    def search(self, query: str, orientation: str, short_side: int) -> list[Candidate]: ...


# -- helpers ---------------------------------------------------------------------------
def _short(w: int, h: int) -> int:
    return min(w, h)


def _orientation_ok(w: int, h: int, orientation: str) -> bool:
    if orientation == "portrait":
        return h > w
    if orientation == "landscape":
        return w > h
    return True


def user_agent(url: str) -> str:
    """Wikimedia's policy needs a contact in the User-Agent; only sent to Wikimedia."""
    contact = os.environ.get("STORYREEL_CONTACT", "").strip()
    host = urllib.parse.urlparse(url).hostname or ""
    if contact and (host == "wikimedia.org" or host.endswith(".wikimedia.org")):
        return f"{USER_AGENT} ({contact})"
    return USER_AGENT


def _get_json(url: str, headers: dict[str, str], timeout: float = 30.0) -> Any:
    if not url.startswith("https://"):
        raise StockError(f"Refusing a non-https stock URL: {url[:40]}")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent(url), **headers})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https only)
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise StockError(
                f"The stock library refused the API key (HTTP {exc.code}).",
                fix="Check PEXELS_API_KEY / PIXABAY_API_KEY (free keys from pexels.com/api "
                "and pixabay.com/api/docs).",
            ) from exc
        if exc.code == 429:
            raise StockError(
                "The stock library rate limit was reached (HTTP 429).",
                fix="Wait an hour, or re-run: everything already downloaded is cached.",
            ) from exc
        raise StockError(f"Stock search failed: HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise StockError(
            f"Cannot reach the stock library: {exc}",
            fix="Check the internet connection, or use --offline to use only cached footage.",
        ) from exc


def _pick_file(files: Iterable[tuple[int, int, str]], short_side: int) -> tuple[int, int, str]:
    """Smallest file whose short side covers the frame; else the largest available."""
    files = [f for f in files if f[2]]
    if not files:
        raise ValueError("no files")
    enough = [f for f in files if _short(f[0], f[1]) >= short_side]
    if enough:
        return min(enough, key=lambda f: f[0] * f[1])
    return max(files, key=lambda f: f[0] * f[1])


# -- providers -------------------------------------------------------------------------
class Pexels:
    name = "pexels"

    def __init__(self, key: str) -> None:
        self.key = key

    def search(self, query: str, orientation: str, short_side: int) -> list[Candidate]:
        params: dict[str, str | int] = {"query": query, "per_page": 30, "size": "medium"}
        if orientation in ("landscape", "portrait", "square"):
            params["orientation"] = orientation
        q = urllib.parse.urlencode(params)
        data = _get_json(f"https://api.pexels.com/videos/search?{q}", {"Authorization": self.key})
        out = []
        for v in data.get("videos", []):
            files = [
                (int(f.get("width") or 0), int(f.get("height") or 0), str(f.get("link") or ""))
                for f in v.get("video_files", [])
                if f.get("file_type") == "video/mp4" and f.get("width") and f.get("height")
            ]
            try:
                w, h, url = _pick_file(files, short_side)
            except ValueError:
                continue
            if not _orientation_ok(w, h, orientation):
                continue
            user = v.get("user") or {}
            out.append(
                Candidate(
                    f"pexels:{v['id']}",
                    w,
                    h,
                    float(v.get("duration") or 0),
                    url,
                    str(v.get("url") or ""),
                    str(user.get("name") or ""),
                    PEXELS_LICENCE,
                    thumb=str(v.get("image") or ""),
                )
            )
        return out


class Pixabay:
    name = "pixabay"

    def __init__(self, key: str) -> None:
        self.key = key

    def search(self, query: str, orientation: str, short_side: int) -> list[Candidate]:
        q = urllib.parse.urlencode(
            # film = live action only: Pixabay also hosts animation and AI-made clips
            {
                "key": self.key,
                "q": query[:100],
                "per_page": 30,
                "safesearch": "true",
                "video_type": "film",
            }
        )
        data = _get_json(f"https://pixabay.com/api/videos/?{q}", {})
        out = []
        for v in data.get("hits", []):
            if v.get("isAiGenerated") or v.get("isLowQuality"):
                continue
            files = [
                (int(f.get("width") or 0), int(f.get("height") or 0), str(f.get("url") or ""))
                for f in (v.get("videos") or {}).values()
                if f.get("width") and f.get("height")
            ]
            try:
                w, h, url = _pick_file(files, short_side)
            except ValueError:
                continue
            if not _orientation_ok(w, h, orientation):
                continue
            out.append(
                Candidate(
                    f"pixabay:{v['id']}",
                    w,
                    h,
                    float(v.get("duration") or 0),
                    url,
                    str(v.get("pageURL") or ""),
                    str(v.get("user") or ""),
                    PIXABAY_LICENCE,
                    thumb=_pixabay_thumb(v),
                    tags=str(v.get("tags") or ""),
                )
            )
        return out


def _pixabay_thumb(v: dict[str, Any]) -> str:
    vids = v.get("videos") or {}
    for size in ("small", "medium", "tiny", "large"):
        t = str((vids.get(size) or {}).get("thumbnail") or "")
        if t:
            return t
    return ""


_TAG = re.compile(r"<[^>]+>")
# Commercial use allowed; CC BY needs credit (written to footage_credits.txt).
# Share-alike (CC BY-SA) is skipped: it would put licence terms on the whole video.
_COMMONS_OK = re.compile(r"^(cc0|public domain|pd\b|pd-|cc by \d)", re.IGNORECASE)


class Commons:
    """Wikimedia Commons videos. No key needed; only public-domain, CC0 and CC BY files."""

    name = "commons"

    def search(self, query: str, orientation: str, short_side: int) -> list[Candidate]:
        q = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:video {query}",
                "gsrnamespace": 6,
                "gsrlimit": 40,
                "prop": "videoinfo",
                "viprop": "url|size|extmetadata|derivatives",
            }
        )
        data = _get_json(f"https://commons.wikimedia.org/w/api.php?{q}", {})
        pages = sorted((data.get("query") or {}).get("pages", {}).values(),
                       key=lambda p: p.get("index", 0))  # fmt: skip
        out = []
        for page in pages:
            info = (page.get("videoinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            lic = str((meta.get("LicenseShortName") or {}).get("value") or "")
            if not _COMMONS_OK.match(lic.strip()):
                continue
            if float(info.get("duration") or 0) > COMMONS_MAX_S:  # films/lectures, not b-roll
                continue
            files = [
                (int(d.get("width") or 0), int(d.get("height") or 0), str(d.get("src") or ""))
                for d in info.get("derivatives", [])
                if "webm" in str(d.get("type", "")) and d.get("width") and d.get("height")
            ]
            try:
                w, h, url = _pick_file(files, short_side)
            except ValueError:
                continue
            if not _orientation_ok(w, h, orientation) or _short(w, h) < short_side * 2 // 3:
                continue
            artist = _TAG.sub("", str((meta.get("Artist") or {}).get("value") or "")).strip()
            credit = "no attribution required" if not lic.lower().startswith("cc by") else (
                "credit the author")  # fmt: skip
            out.append(
                Candidate(
                    f"commons:{page.get('pageid')}",
                    w,
                    h,
                    float(info.get("duration") or 0),
                    url,
                    str(info.get("descriptionurl") or ""),
                    artist[:80],
                    f"{lic} via Wikimedia Commons ({credit})",
                )
            )
        return out


_WORD = re.compile(r"[a-z]+")
Prober = Callable[[Path], tuple[int, int, float]]  # path -> (width, height, seconds)


class Folder:
    """Your own clips. A file matches a query when its name (or its folder's name) shares
    words with it: forest/fox_running.mp4 matches "fox forest"."""

    name = "folder"

    def __init__(self, root: Path, prober: Prober) -> None:
        if not root.is_dir():
            raise StockError(f"Footage folder {root} does not exist.")
        self.root = root
        self.prober = prober
        self._files = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
        self._info: dict[Path, tuple[int, int, float] | None] = {}

    def _words(self, p: Path) -> set[str]:
        rel = p.relative_to(self.root).with_suffix("")
        return set(_WORD.findall(str(rel).lower()))

    def _probe(self, p: Path) -> tuple[int, int, float] | None:
        if p not in self._info:
            try:
                self._info[p] = self.prober(p)
            except StoryReelError as exc:
                log.warning("skipping footage %s: %s", p, exc)
                self._info[p] = None
        return self._info[p]

    def search(self, query: str, orientation: str, short_side: int) -> list[Candidate]:
        want = set(_WORD.findall(query.lower()))
        scored = []
        for p in self._files:
            score = len(want & self._words(p))
            if score:
                scored.append((-score, str(p), p))
        out = []
        for _, _, p in sorted(scored):
            info = self._probe(p)
            if info is None or not _orientation_ok(info[0], info[1], orientation):
                continue
            out.append(
                Candidate(
                    f"folder:{p.relative_to(self.root).as_posix()}",
                    info[0],
                    info[1],
                    info[2],
                    str(p),
                    p.as_uri(),
                    "",
                    "your own footage",
                )
            )
        return out


# -- library: providers + cache --------------------------------------------------------
def default_cache() -> Path:
    return data_dir() / "stock"


def providers_from_env(names: str = "auto") -> list[Provider]:
    """'auto' = Pexels/Pixabay when their free key is set (Commons only when named).
    Or a comma list of names: pexels, pixabay, commons."""
    keys = {"pexels": os.environ.get("PEXELS_API_KEY", ""),
            "pixabay": os.environ.get("PIXABAY_API_KEY", "")}  # fmt: skip
    # Commons is never automatic: its results are uncurated (see the module docstring)
    wanted = list(keys) if names == "auto" else [n.strip() for n in names.split(",")]
    out: list[Provider] = []
    for n in filter(None, wanted):
        if n == "commons":
            if not os.environ.get("STORYREEL_CONTACT"):
                raise StockError(
                    "Wikimedia Commons requires contact details in every request.",
                    fix="setx STORYREEL_CONTACT you@example.com (then a new terminal); "
                    "check every Commons shot: its results are not curated.",
                )
            out.append(Commons())
            continue
        if n not in keys:
            raise StockError(
                f"Unknown footage library {n!r}.", fix="Use pexels, pixabay or commons."
            )
        if keys[n]:
            out.append(Pexels(keys[n]) if n == "pexels" else Pixabay(keys[n]))
        elif names != "auto":
            raise StockError(
                f"{n} needs a free API key in {n.upper()}_API_KEY.",
                fix=f"Get one at {'pexels.com/api' if n == 'pexels' else 'pixabay.com/api/docs'}"
                f", then: setx {n.upper()}_API_KEY <key>  (open a new terminal afterwards)",
            )
    return out


def _read_hits(f: Path) -> list[Candidate] | None:
    if not f.is_file():
        return None
    try:
        return [Candidate(**c) for c in json.loads(f.read_text(encoding="utf-8"))]
    except (ValueError, TypeError):
        return None


class Library:
    def __init__(
        self,
        providers: list[Provider],
        cache_dir: Path | None = None,
        *,
        offline: bool = False,
        progress: Callable[[str], None] | None = None,
        ranker: Ranker | None = None,
    ) -> None:
        self.providers = providers
        self.cache = cache_dir or default_cache()
        self.offline = offline
        self.say = progress or (lambda m: None)
        self.ranker = ranker

    # search results are cached forever: offline re-makes need them
    def _search(self, prov: Provider, query: str, orientation: str, short: int) -> list[Candidate]:
        if prov.name == "folder":
            return prov.search(query, orientation, short)
        key = f"{query}|{orientation}|{short}"
        f = self._search_file(prov.name, f"{SEARCH_CACHE_VERSION}|{key}")
        cached = _read_hits(f)
        if cached is not None:
            return cached
        if self.offline:  # an older StoryReel's results (no previews/tags) beat nothing
            return _read_hits(self._search_file(prov.name, key)) or []
        res = prov.search(query, orientation, short)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps([asdict(c) for c in res]), encoding="utf-8")
        return res

    def _search_file(self, provider: str, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()[:16]  # noqa: S324 (file name only)
        return self.cache / "search" / provider / f"{h}.json"

    def _file_for(self, c: Candidate) -> Path:
        prov, ident = c.key.split(":", 1)
        if prov == "folder":
            return Path(c.url)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", ident)
        ext = Path(urllib.parse.urlparse(c.url).path).suffix.lower()
        ext = ext if ext in VIDEO_EXTS else ".mp4"
        return self.cache / "files" / f"{prov}_{safe}_{_short(c.width, c.height)}{ext}"

    def _download(self, c: Candidate, dest: Path) -> bool:
        if dest.is_file() and dest.stat().st_size > 0:
            return True
        if self.offline or not c.url.startswith("https://"):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        req = urllib.request.Request(c.url, headers={"User-Agent": user_agent(c.url)})  # noqa: S310
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as out:  # noqa: S310
                size = int(r.headers.get("Content-Length") or 0)
                if size > MAX_DOWNLOAD_MB * 2**20:
                    return False
                got = 0
                while chunk := r.read(1 << 20):
                    got += len(chunk)
                    if got > MAX_DOWNLOAD_MB * 2**20:
                        return False
                    out.write(chunk)
            part.replace(dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("download failed %s: %s", c.key, exc)
            return False
        finally:
            part.unlink(missing_ok=True)
        self.say(
            f"downloaded {c.key} ({dest.stat().st_size / 2**20:.1f} MB, "
            f"{time.perf_counter() - t0:.0f} s)"
        )
        dest.with_suffix(".json").write_text(json.dumps(asdict(c)), encoding="utf-8")
        return True

    def _hits(self, q: str, orientation: str, short: int, failures: list[StockError]
              ) -> list[Candidate]:  # fmt: skip
        out: list[Candidate] = []
        for prov in self.providers:
            try:
                out += self._search(prov, q, orientation, short)
            except StockError as exc:
                failures.append(exc)
                log.warning("%s: %s", prov.name, exc)
        return out

    def _clip(self, c: Candidate) -> Clip | None:
        dest = self._file_for(c)
        if not self._download(c, dest):
            return None
        tags = c.tags or c.key.split(":", 1)[1].replace("/", " ").replace("_", " ")
        return Clip(c.key, dest, c.width, c.height, c.duration_s, c.page_url, c.author,
                    c.licence, tags)  # fmt: skip

    def find(
        self,
        queries: list[str],
        *,
        seconds: float,
        count: int,
        orientation: str,
        short_side: int,
        exclude: set[str],
        describe: str = "",
    ) -> list[Clip]:
        """Up to `count` different clips at least `seconds` long. Clips in `exclude`
        (already used in this story) are skipped.

        With a ranker, candidates from the first RANKED_QUERIES searches are scored with
        CLIP against `describe` (the narration) and their query; otherwise the first
        hits of the best query win."""
        failures: list[StockError] = []
        out: list[Clip] = []
        seen = set(exclude)
        if self.ranker is None:
            for q in queries:
                for c in self._hits(q, orientation, short_side, failures):
                    if c.key in seen or 0 < c.duration_s < seconds:
                        continue
                    clip = self._clip(c)
                    if clip is None:
                        continue
                    seen.add(c.key)
                    out.append(clip)
                    if len(out) >= count:
                        return out
        else:
            pool: list[tuple[int, str, Candidate]] = []
            for qi, q in enumerate(queries[:RANKED_QUERIES]):
                for c in self._hits(q, orientation, short_side, failures)[:PER_QUERY]:
                    if c.key in seen or 0 < c.duration_s < seconds:
                        continue
                    seen.add(c.key)
                    pool.append((qi, q, c))
            ranked = self.ranker.rank(describe, pool)
            for s in ranked:
                clip = self._clip(s.cand)
                if clip is not None:
                    out.append(clip)
                if len(out) >= count:
                    break
            out = self.ranker.wide_first(out, {s.cand.key: s for s in ranked})
        if not out and failures and len(failures) >= len(self.providers):
            raise failures[0]
        return out
