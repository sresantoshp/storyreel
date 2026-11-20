"""Realistic style: footage search/cache, shot planning, trailer sound, assembly + render.

No network: the stock libraries are replaced by a fake provider or a local folder of
ffmpeg-generated clips.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from conftest import Media, needs_ffmpeg
from storyreel.audio import cinematic
from storyreel.errors import StoryReelError
from storyreel.media import stock
from storyreel.project import store
from storyreel.render.plan import RenderSettings
from storyreel.render.renderer import Renderer
from storyreel.story import assemble
from storyreel.story import realistic as real
from storyreel.story.schema import Scene, StoryPlan
from storyreel.watchdog import Watchdog
from test_assemble import stand_ins


# -- queries and shots -----------------------------------------------------------------
def test_scene_queries_specific_first() -> None:
    s = Scene("The fox ran to the waterfall.", "forest", "happy", cast=["fox"])
    q = real.scene_queries(s)
    assert q[:4] == ["fox forest", "fox", "waterfall forest", "forest sunlight"]
    camp = Scene("The girl sat by the fire.", "campfire", "calm", cast=["girl", "tiger"])
    assert real.scene_queries(camp)[:4] == [
        "tiger campfire",
        "tiger",
        "fire",
        "campfire night",
    ]
    assert real.scene_queries(camp).index("girl campfire") == 4
    assert q[-1] == real.KIT_QUERIES["forest"][-1]
    assert len(q) == len(set(q))


def test_every_kit_has_queries() -> None:
    from storyreel.story.kits import KITS

    assert set(KITS) == set(real.KIT_QUERIES) == set(real.KIT_PLACE)


@pytest.mark.parametrize("seconds", [2.0, 3.6, 6.0, 7.0, 11.9, 13.0, 30.0])
def test_shot_lengths(seconds: float) -> None:
    lengths = real.shot_lengths(seconds)
    assert sum(lengths) == pytest.approx(seconds)
    assert all(ln <= real.SHOT_MAX_S + 1e-9 for ln in lengths)
    if seconds >= 2 * real.SHOT_MIN_S:
        assert all(ln >= real.SHOT_MIN_S for ln in lengths)


def test_letterbox() -> None:
    assert real.letterbox_bar(1920, 1080) == 138
    assert real.letterbox_bar(1080, 1920) == 0


def _clip(key: str, path: Path = Path("x.mp4")) -> stock.Clip:
    return stock.Clip(key, path, 1920, 1080, 20.0, f"https://p/{key}", "Ann", "L")


def test_plan_shots_no_repeats_and_no_footage_fails() -> None:
    scenes = [Scene("a", "forest", "calm"), Scene("b", "beach", "calm")]
    calls: list[set[str]] = []
    pool = [_clip(f"pexels:{i}") for i in range(10)]

    def find(q: list[str], d: str, secs: float, n: int, used: set[str]) -> list[stock.Clip]:
        calls.append(set(used))
        assert d in ("a", "b")
        return [c for c in pool if c.key not in used][:n]

    shots = real.plan_shots(scenes, [0.0, 12.0], [12.0, 5.0], find)
    assert [s.scene for s in shots] == [0, 0, 1]
    assert len({s.clip.key for s in shots}) == 3
    assert calls[1] == {"pexels:0", "pexels:1"}
    assert "Ann" in real.credits(shots)

    with pytest.raises(StoryReelError, match="No footage found for scene 1") as ei:
        real.plan_shots(scenes, [0.0, 12.0], [12.0, 5.0], lambda *a: [])
    assert "does not fall back to animation" in str(ei.value.fix)


# -- stock library ---------------------------------------------------------------------
def test_pick_file_prefers_smallest_covering() -> None:
    files = [(3840, 2160, "4k"), (1920, 1080, "hd"), (1280, 720, "sd"), (640, 360, "")]
    assert stock._pick_file(files, 1080)[2] == "hd"
    assert stock._pick_file(files, 720)[2] == "sd"
    assert stock._pick_file(files, 4000)[2] == "4k"


def test_pexels_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "videos": [
            {
                "id": 7,
                "duration": 15,
                "url": "https://www.pexels.com/video/7/",
                "user": {"name": "Ann"},
                "video_files": [
                    {"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "https://a"},
                    {"file_type": "video/mp4", "width": 3840, "height": 2160, "link": "https://b"},
                ],
            },
            {"id": 8, "duration": 9, "video_files": []},
        ]
    }
    seen: dict[str, Any] = {}

    def fake(url: str, headers: dict[str, str], timeout: float = 30) -> Any:
        seen.update(url=url, headers=headers)
        return data

    monkeypatch.setattr(stock, "_get_json", fake)
    hits = stock.Pexels("KEY").search("fox forest", "landscape", 1080)
    assert [(h.key, h.url, h.author) for h in hits] == [("pexels:7", "https://a", "Ann")]
    assert seen["headers"] == {"Authorization": "KEY"}
    assert "query=fox+forest" in seen["url"]


def test_commons_licences(monkeypatch: pytest.MonkeyPatch) -> None:
    def page(i: int, lic: str) -> dict[str, Any]:
        return {
            "pageid": i,
            "index": i,
            "videoinfo": [
                {
                    "duration": 30.0,
                    "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{i}",
                    "extmetadata": {
                        "LicenseShortName": {"value": lic},
                        "Artist": {"value": "<a href='x'>Bo</a>"},
                    },
                    "derivatives": [
                        {"type": 'video/webm; codecs="vp9"', "width": 1920, "height": 1080,
                         "src": f"https://upload.wikimedia.org/{i}.1080p.vp9.webm"},
                        {"type": "video/quicktime", "width": 640, "height": 360, "src": "q"},
                    ],
                }
            ],
        }  # fmt: skip

    lics = ["CC BY 3.0", "CC BY-SA 4.0", "Public domain", "CC0", "GFDL"]
    data = {"query": {"pages": {str(i): page(i, lic) for i, lic in enumerate(lics)}}}
    monkeypatch.setattr(stock, "_get_json", lambda url, headers, timeout=30: data)
    hits = stock.Commons().search("forest", "landscape", 1080)
    assert [h.key for h in hits] == ["commons:0", "commons:2", "commons:3"]
    assert hits[0].author == "Bo" and "credit the author" in hits[0].licence
    assert "no attribution required" in hits[1].licence
    assert hits[0].url.endswith(".webm")


class FakeProvider:
    name = "pexels"

    def __init__(self, hits: list[stock.Candidate]) -> None:
        self.hits = hits
        self.calls = 0

    def search(self, query: str, orientation: str, short_side: int) -> list[stock.Candidate]:
        self.calls += 1
        return self.hits


def test_library_cache_and_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        stock.Candidate(f"pexels:{i}", 1920, 1080, d, f"https://f/{i}", "", "", "L")
        for i, d in enumerate([3.0, 20.0, 20.0])
    ]
    prov = FakeProvider(hits)
    downloads: list[str] = []

    def fake_download(self: stock.Library, c: stock.Candidate, dest: Path) -> bool:
        if not dest.is_file() and self.offline:
            return False
        downloads.append(c.key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(stock.Library, "_download", fake_download)
    lib = stock.Library([prov], tmp_path)
    kw: dict[str, Any] = dict(seconds=5.0, count=1, orientation="landscape", short_side=1080)
    got = lib.find(["forest"], exclude=set(), **kw)
    assert [c.key for c in got] == ["pexels:1"]  # the 3 s clip is too short
    got = lib.find(["forest"], exclude={"pexels:1"}, **kw)
    assert [c.key for c in got] == ["pexels:2"]
    assert prov.calls == 1  # second search came from the cache

    off = stock.Library([FakeProvider([])], tmp_path, offline=True)
    assert [c.key for c in off.find(["forest"], exclude=set(), **kw)] == ["pexels:1"]
    assert off.find(["never searched"], exclude=set(), **kw) == []


def test_providers_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setenv("PIXABAY_API_KEY", "k")
    monkeypatch.delenv("STORYREEL_CONTACT", raising=False)
    assert [p.name for p in stock.providers_from_env()] == ["pixabay"]  # commons is opt-in
    with pytest.raises(StoryReelError, match="contact"):
        stock.providers_from_env("commons")
    monkeypatch.setenv("STORYREEL_CONTACT", "me@example.com")
    assert [p.name for p in stock.providers_from_env("commons")] == ["commons"]
    assert "me@example.com" in stock.user_agent("https://upload.wikimedia.org/x.webm")
    assert "me@example.com" not in stock.user_agent("https://api.pexels.com/videos")
    with pytest.raises(StoryReelError, match="PEXELS_API_KEY"):
        stock.providers_from_env("pexels")


def test_folder_provider(tmp_path: Path) -> None:
    (tmp_path / "forest").mkdir()
    for name in ("forest/fox_running.mp4", "beach_waves.mov", "notes.txt", "fox_tall.mp4"):
        (tmp_path / name).write_bytes(b"x")
    dims = {"fox_tall.mp4": (1080, 1920, 9.0)}
    folder = stock.Folder(tmp_path, lambda p: dims.get(p.name, (1920, 1080, 12.0)))
    hits = folder.search("fox forest", "landscape", 1080)
    assert [h.key for h in hits] == ["folder:forest/fox_running.mp4"]  # tall clip filtered
    assert [h.key for h in folder.search("fox", "portrait", 1080)] == ["folder:fox_tall.mp4"]
    assert folder.search("desert", "landscape", 1080) == []


# -- trailer sound ---------------------------------------------------------------------
@pytest.mark.parametrize("kind", cinematic.KINDS)
def test_sfx(tmp_path: Path, kind: str) -> None:
    import wave

    out = cinematic.make_sfx(kind, 1.5, tmp_path / f"{kind}.wav")
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 3, 48000)
        raw = w.readframes(w.getnframes())
    b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
    x = b[:, 0].astype(np.int32) | b[:, 1].astype(np.int32) << 8 | b[:, 2].astype(np.int32) << 16
    x = np.where(x >= 2**23, x - 2**24, x) / 2**23
    assert x.size == 2 * 72000
    assert 0.85 < np.max(np.abs(x)) < 0.9  # -1 dBFS
    assert np.max(np.abs(x[:20])) < 0.05  # no click at the start
    with pytest.raises(ValueError):
        cinematic.make_sfx("boing", 1.0, tmp_path / "x.wav")


# -- end to end ------------------------------------------------------------------------
def _footage(ffmpeg: Path, out: Path, src: str) -> Path:
    subprocess.run(
        [str(ffmpeg), "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}=s=480x270:r=25", "-t", "9",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    return out


@needs_ffmpeg
def test_realistic_assemble_and_render(tmp_path: Path, media: Media) -> None:
    lib = tmp_path / "footage"
    lib.mkdir()
    for i, name in enumerate(["snow_valley", "river_snow", "storm_rain", "village_sun", "sky"]):
        _footage(media.ffmpeg, lib / f"{name}.mp4", "testsrc2" if i % 2 else "mandelbrot")
    folder = stock.Folder(lib, lambda p: (480, 270, 9.0))
    library = stock.Library([folder], tmp_path / "cache")

    def find(q: list[str], d: str, secs: float, n: int, used: set[str]) -> list[stock.Clip]:
        # every scene finds something: fall back to any clip, like a broad stock search
        return library.find(
            [*q, "sky"], seconds=secs, count=n, orientation="landscape", short_side=180,
            exclude=used,
        )  # fmt: skip

    plan = StoryPlan(
        "The Long Winter",
        "en",
        [
            Scene("Snow fell over the quiet valley.", "snow_valley", "calm"),
            Scene("The river froze under the snow.", "snow_valley", "sad"),
            Scene("That night a storm came with heavy rain.", "storm", "tense"),
            Scene("In the morning the sun rose over the village.", "village", "happy"),
        ],
    )
    speak, amb, mus = stand_ins(media.ffmpeg)
    res = assemble.assemble(
        plan, tmp_path / "real.storyreel", speak=speak, ambience=amb, music=mus,
        ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, aspect="16:9", height=180,
        style="realistic", find_footage=find, sfx=cinematic.make_sfx,
    )  # fmt: skip
    credits = (res.project / "footage_credits.txt").read_text(encoding="utf-8")
    assert "your own footage" in credits
    with store.open_project(res.project) as p:
        con = p.con
        kinds = [r[0] for r in con.execute("SELECT kind FROM clip")]
        assert "generator" not in kinds  # never falls back to animation
        assert kinds.count("adjust") == 1
        effects = {r[0] for r in con.execute("SELECT type FROM effect")}
        assert {"color", "curves", "glow", "vignette", "grain"} <= effects
        fx: dict[str, list[str]] = {}
        for name, etype in con.execute(
            "SELECT b.name, e.type FROM bus_effect e JOIN bus b ON b.id = e.bus_id ORDER BY e.idx"
        ):
            fx.setdefault(name, []).append(etype)
        assert fx["dialogue"][0] == "highpass" and "deesser" in fx["dialogue"]
        assert fx["fx"] == ["reverb"] and fx["master"] == ["compressor"]
        sfx_names = {Path(r[0]).name.split("_")[1] for r in con.execute(
            "SELECT path FROM asset WHERE name LIKE 'sfx_%'")}  # fmt: skip
        assert {"braam", "whoosh", "riser", "impact"} <= sfx_names
        # sound bed: ambience + music on alternating tracks, crossfaded, music keyframed
        names = {r[0] for r in con.execute("SELECT name FROM track WHERE kind='audio'")}
        assert {"Ambience", "Ambience B", "Music", "Music B"} <= names
        amb = [Path(r[0]).name for r in con.execute(
            "SELECT path FROM asset WHERE name LIKE 'amb_%' ORDER BY id")]  # fmt: skip
        # the snow scene's sound comes from its footage (ice/stream), not the kit's wind
        assert amb[0].split("_")[2] in ("ice.wav", "stream.wav")
        vol = con.execute("SELECT count(*) FROM keyframe WHERE param = 'volume'").fetchone()
        assert vol[0] >= 3
        s = p.settings()
        wd = Watchdog(2**31).start()
        try:
            r = Renderer(
                p, RenderSettings(s.width, s.height, s.rate, preset="ultrafast"),
                ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, watchdog=wd,
            ).render(tmp_path / "real.mp4")  # fmt: skip
        finally:
            wd.stop()
    assert r.frames_written == r.frames_expected
    assert abs(r.duration_s - res.duration_s) < 0.1
    # the first frame is black (fade from black) and the letterbox bars stay black
    frames = subprocess.run(
        [str(media.ffmpeg), "-v", "error", "-i", str(tmp_path / "real.mp4"), "-vf",
         "select=eq(n\\,0)+eq(n\\,90)", "-fps_mode", "passthrough", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    img = np.frombuffer(frames, np.uint8).reshape(2, s.height, s.width, 3)
    assert img[0].mean() < 12
    bar = real.letterbox_bar(s.width, s.height)
    mid = img[1]
    assert mid[: bar - 2].mean() < 16 and mid[s.height - bar + 2 :].mean() < 16
    assert mid[bar + 4 : s.height - bar - 4].mean() > 30


# -- CLIP ranking (fake embedder: one axis per concept) ----------------------------------
CONCEPTS = ["dolphin", "aquarium", "ocean", "landscape", "close"]


def _vec(*names: str) -> np.ndarray:
    v = np.zeros(len(CONCEPTS), np.float32)
    for n in names:
        v[CONCEPTS.index(n)] = 1.0
    out: np.ndarray = v / np.linalg.norm(v)
    return out


class FakeClip:
    """Images are tagged by their first pixel's red value; texts by the words in them."""

    looks: ClassVar[dict[int, tuple[str, ...]]] = {
        1: ("dolphin", "ocean", "close"),
        2: ("aquarium",),
        3: ("ocean", "landscape"),
        4: ("dolphin", "ocean", "landscape"),
    }

    def texts(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            names = [c for c in CONCEPTS if c in t] or ["landscape"]
            if "wide" in t:
                names = ["landscape"]
            rows.append(_vec(*names) * 0.3)  # real CLIP similarities are ~0.3
        return np.array(rows, np.float32)

    def images(self, rgbs: list[np.ndarray]) -> np.ndarray:
        return np.array([_vec(*self.looks[int(x[0, 0, 0])]) for x in rgbs], np.float32)


def _cand(i: int) -> stock.Candidate:
    return stock.Candidate(f"pixabay:{i}", 1280, 720, 20.0, f"https://v/{i}", "", "", "L",
                           thumb=f"https://t/{i}.jpg")  # fmt: skip


def test_ranker_prefers_narration_match_and_orders_wide_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storyreel.media import rank

    def fake_preview(self: rank.Ranker, c: stock.Candidate) -> np.ndarray:
        img = np.zeros((4, 4, 3), np.uint8)
        img[0, 0, 0] = int(c.key.split(":")[1])
        return img

    monkeypatch.setattr(rank.Ranker, "preview", fake_preview)
    r = rank.Ranker(FakeClip(), tmp_path, lambda p: None)
    pool = [(0, "dolphin ocean", _cand(i)) for i in (2, 3, 1, 4)]
    ranked = r.rank("Dolphins dance in the ocean", pool)
    keys = [s.cand.key for s in ranked]
    assert keys[:2] in (["pixabay:1", "pixabay:4"], ["pixabay:4", "pixabay:1"])
    assert "pixabay:2" not in keys  # the aquarium crowd is rejected
    clips = [_clip(k) for k in keys[:2]]
    ordered = r.wide_first(clips, {s.cand.key: s for s in ranked})
    assert [c.key for c in ordered] == ["pixabay:4", "pixabay:1"]  # wide, then close


def test_library_uses_ranker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from storyreel.media import rank

    class Fixed:
        def __init__(self) -> None:
            self.pool: list[tuple[int, str, stock.Candidate]] = []

        def rank(self, describe: str, pool: list[Any]) -> list[rank.Scored]:
            self.pool = pool
            return [rank.Scored(c, 1.0, np.zeros(2, np.float32)) for _, _, c in reversed(pool)]

        def wide_first(self, clips: list[stock.Clip], scored: Any) -> list[stock.Clip]:
            return clips

    monkeypatch.setattr(stock.Library, "_download", lambda self, c, dest: True)
    fixed = Fixed()
    lib = stock.Library([FakeProvider([_cand(i) for i in range(3)])], tmp_path,
                        ranker=fixed)  # type: ignore[arg-type]  # fmt: skip
    got = lib.find(["a", "b"], seconds=5, count=2, orientation="landscape", short_side=720,
                   exclude={"pixabay:0"}, describe="x")  # fmt: skip
    assert [c.key for c in got] == ["pixabay:2", "pixabay:1"]
    assert [c.key for _, _, c in fixed.pool] == ["pixabay:1", "pixabay:2"]  # no duplicates


def test_pixabay_skips_ai_and_low_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    def hit(i: int, **flags: bool) -> dict[str, Any]:
        return {"id": i, "duration": 10, "pageURL": "", "user": "u", "tags": "sea, waves",
                "videos": {"medium": {"width": 1280, "height": 720, "url": "https://m",
                                      "thumbnail": f"https://t/{i}.jpg"}}, **flags}  # fmt: skip

    data = {"hits": [hit(1), hit(2, isAiGenerated=True), hit(3, isLowQuality=True)]}
    monkeypatch.setattr(stock, "_get_json", lambda url, headers, timeout=30: data)
    hits = stock.Pixabay("k").search("sea", "landscape", 720)
    assert [(h.key, h.thumb, h.tags) for h in hits] == [
        ("pixabay:1", "https://t/1.jpg", "sea, waves")]  # fmt: skip


def test_shot_match_pulls_towards_story_average() -> None:
    warm = ([60.0, 20.0, 40.0], [20.0, 10.0, 10.0])
    cool = ([50.0, -5.0, -20.0], [10.0, 5.0, 8.0])
    mid = ([55.0, 5.0, 5.0], [15.0, 6.0, 9.0])
    p = real.shot_match([warm, cool, mid])
    ref = np.array(mid[0])
    for (m, _), q in zip([warm, cool, mid], p, strict=True):
        moved = np.abs(np.array(q["dst_mean"]) - ref)
        assert np.all(moved <= np.abs(np.array(m) - ref) + 1e-6)  # never further away
        assert q["src_mean"] == m
    assert p[0]["dst_mean"][1] > p[1]["dst_mean"][1]  # the warm shot stays warmer


@needs_ffmpeg
def test_grab_frame(tmp_path: Path, media: Media) -> None:
    from storyreel.vision.frames import grab

    clip = _footage(media.ffmpeg, tmp_path / "c.mp4", "testsrc2")
    img = grab(media.ffmpeg, clip, at_s=2.0, width=160)
    assert img is not None and img.shape == (90, 160, 3) and img.dtype == np.uint8
    assert grab(media.ffmpeg, clip, width=64) is not None
    assert grab(media.ffmpeg, tmp_path / "missing.mp4") is None


def test_tag_match() -> None:
    from storyreel.media.rank import _tag_match

    c = stock.Candidate("pixabay:1", 1, 1, 1.0, "", "", "", "L", tags="ocean, dolphins, sea")
    assert _tag_match("dolphin ocean", c) == 1.0
    assert _tag_match("tiger jungle", c) == 0.0
    f = stock.Candidate("folder:jungle/tiger_walk.mp4", 1, 1, 1.0, "", "", "", "L")
    assert _tag_match("tiger jungle", f) == 1.0


def test_search_cache_versioned(tmp_path: Path) -> None:
    old = [stock.Candidate("pixabay:1", 1920, 1080, 20.0, "https://f/1", "", "", "L")]
    lib = stock.Library([FakeProvider([])], tmp_path, offline=True)
    legacy = lib._search_file("pexels", "sea|landscape|720")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps([dataclasses.asdict(c) for c in old]), encoding="utf-8")
    assert [c.key for c in lib._search(FakeProvider([]), "sea", "landscape", 720)] == [
        "pixabay:1"]  # offline: the old cache still works  # fmt: skip
    fresh = FakeProvider([_cand(7)])
    online = stock.Library([fresh], tmp_path)
    assert [c.key for c in online._search(fresh, "sea", "landscape", 720)] == ["pixabay:7"]
    assert fresh.calls == 1  # online: re-searched, so previews and tags are filled in


def test_ranker_drops_drawn_and_demotes_captive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storyreel.media import rank

    def fake_preview(self: rank.Ranker, c: stock.Candidate) -> np.ndarray:
        img = np.zeros((4, 4, 3), np.uint8)
        img[0, 0, 0] = 1  # every preview looks like a dolphin close-up
        return img

    monkeypatch.setattr(rank.Ranker, "preview", fake_preview)
    r = rank.Ranker(FakeClip(), tmp_path, lambda p: None)

    def tagged(i: int, tags: str) -> stock.Candidate:
        return dataclasses.replace(_cand(i), tags=tags)

    pool = [(0, "dolphin ocean", tagged(1, "dolphin, aquarium, zoo")),
            (0, "dolphin ocean", tagged(2, "dolphin, 3d, animation")),
            (0, "dolphin ocean", tagged(3, "dolphin, ocean, wild"))]  # fmt: skip
    keys = [s.cand.key for s in r.rank("Dolphins in the ocean", pool)]
    assert "pixabay:2" not in keys  # drawn/rendered: never
    assert keys[0] == "pixabay:3"  # wild beats the aquarium (which is a duplicate here)


# -- step 2: sound that follows the picture and the story ------------------------------
def test_scene_ambience_from_tags() -> None:
    beach = "beach, sea, sand, waves, summer"
    assert real.scene_ambience("forest", [beach]) == "beach"  # beach beats its "sea" tag
    assert real.scene_ambience("forest", ["ocean, waves", "dolphins, sea"]) == "ocean"
    assert real.scene_ambience("forest", ["fire, night", "campfire"]) == "fire"
    assert real.scene_ambience("jungle", ["monkey, playing", "beach"]) == "jungle"  # tie:
    assert real.scene_ambience("jungle", ["beach", "monkey, playing"]) == "beach"  # 1st
    assert real.scene_ambience("village", ["person, smiling"]) == "village"  # no clue
    assert real.scene_ambience("village", []) == "village"


def test_segments_merge_short_moods() -> None:
    starts, durs = [0.0, 5.0, 10.0, 13.0, 30.0], [5.0, 5.0, 3.0, 17.0, 6.0]
    keys = ["calm", "calm", "tense", "epic", "calm"]
    assert real.segments(keys, starts, durs) == [
        ("calm", 0.0, 10.0), ("tense", 10.0, 13.0), ("epic", 13.0, 30.0),
        ("calm", 30.0, 36.0)]  # fmt: skip
    merged = real.segments(keys, starts, durs, min_s=8.0)
    assert all(b - a >= 8.0 for _, a, b in merged)
    assert merged[0][1] == 0.0 and merged[-1][2] == 36.0
    assert [k for k, _, _ in merged] == ["calm", "epic"]
    assert real.segments(["sad"], [0.0], [4.0], min_s=8.0) == [("sad", 0.0, 4.0)]


def test_music_arc_builds_to_climax() -> None:
    curve = [real.music_arc_db(t, 100.0) for t in range(0, 101, 5)]
    assert curve[0] == pytest.approx(-4.0)
    assert max(curve) == pytest.approx(0.0, abs=1e-9)
    peak = curve.index(max(curve))
    assert 80 <= peak * 5 <= 90
    rising = curve[: peak + 1]
    assert all(b >= a for a, b in itertools.pairwise(rising))
    assert -1.01 < curve[-1] < 0
