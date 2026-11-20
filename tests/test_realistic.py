"""Realistic style: footage search/cache, shot planning, trailer sound, assembly + render.

No network: the stock libraries are replaced by a fake provider or a local folder of
ffmpeg-generated clips.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

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

    def find(q: list[str], secs: float, n: int, used: set[str]) -> list[stock.Clip]:
        calls.append(set(used))
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

    def find(q: list[str], secs: float, n: int, used: set[str]) -> list[stock.Clip]:
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
