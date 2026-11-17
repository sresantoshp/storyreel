"""Illustrated environments, animated characters, and how stories cast them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import Media
from storyreel.project import store
from storyreel.render import actors, scenery
from storyreel.render.fxspec import GENERATORS
from storyreel.render.generators import Generator
from storyreel.story import assemble, cast
from storyreel.story.kits import KITS
from storyreel.story.planner import PlanRequest, plan_story
from storyreel.story.schema import Scene
from test_assemble import stand_ins

W, H = 320, 180


def _check(planes: tuple[np.ndarray, np.ndarray | None]) -> None:
    rgb, a = planes
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.float32
    assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1.0001
    if a is not None:
        assert a.shape == (H, W) and a.min() >= 0 and a.max() <= 1.0001


@pytest.mark.parametrize("kind", sorted(scenery.KINDS))
def test_scenery_renders_and_is_deterministic(kind: str) -> None:
    g1, g2 = Generator(kind, {}, W, H), Generator(kind, {}, W, H)
    for t in (0.0, 1.7):
        f1, f2 = g1.frame(t), g2.frame(t)
        _check(f1)
        assert np.array_equal(f1[0], f2[0])


@pytest.mark.parametrize("species", sorted(actors.SPECIES))
def test_every_species_draws_something(species: str) -> None:
    kind = actors.SPECIES[species]
    action = {"bird": "fly", "swim": "swim"}.get(kind, "walk")
    g = Generator(
        "actor", {"species": species, "action": action, "x": 0.5, "y": 0.7, "size": 0.5}, W, H
    )
    rgb, a = g.frame(0.4)
    _check((rgb, a))
    assert a is not None and (a > 0.5).sum() > 40, "character is (nearly) invisible"


def test_actor_moves_and_faces_its_direction() -> None:
    def centre(speed: float, t: float) -> float:
        p = {"species": "fox", "x": 0.5, "y": 0.8, "size": 0.4, "speed": speed}
        _, a = Generator("actor", p, W, H).frame(t)
        assert a is not None
        return float((a.sum(0) * np.arange(W)).sum() / a.sum())

    assert centre(0.1, 2.0) - centre(0.1, 0.0) == pytest.approx(0.1 * W * 2.0, abs=4)
    assert centre(-0.1, 1.0) < centre(-0.1, 0.0)
    # legs move while walking, nothing but breathing while idle
    p = {"species": "man", "x": 0.5, "y": 0.9, "size": 0.6, "speed": 0.0001}
    walk = Generator("actor", {**p, "speed": 0.05}, W, H)
    idle = Generator("actor", {**p, "action": "idle"}, W, H)
    shift = lambda g, t: g.frame(t)[1]  # noqa: E731
    assert np.abs(shift(walk, 0.0) - np.roll(shift(walk, 0.3), 0)).sum() > 50
    assert (
        np.abs(shift(idle, 0.0) - shift(idle, 0.3)).sum()
        < np.abs(shift(walk, 0.0) - shift(walk, 0.3)).sum()
    )


def test_unknown_species_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown species"):
        Generator("actor", {"species": "dragon"}, W, H)


def test_kits_use_registered_generators_and_known_params() -> None:
    for kit in KITS.values():
        for gtype, params in kit.layers:
            assert gtype in GENERATORS
            assert set(params) <= set(GENERATORS[gtype]), (kit.name, gtype)


# -- casting -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("A curious fox met an old man and his dog.", ["fox", "grandpa", "dog"]),
        ("Two wolves watched the sheep.", ["wolf", "sheep"]),
        ("एक भेड़िया और लोमड़ी जंगल में थे, दादी ने देखा।", ["wolf", "fox", "grandma"]),
        ("Dolphins jumped near the boat.", ["dolphin"]),
        ("The mountains were quiet.", []),
    ],
)
def test_detect(text: str, found: list[str]) -> None:
    assert cast.detect(text) == found


def test_main_character_appears_in_every_scene_and_sea_creatures_get_water() -> None:
    scenes = [
        Scene("Snow fell on the valley.", "snow_valley", "calm"),
        Scene("A dolphin leapt from the waves.", "sunny_valley", "happy"),
        Scene("Night came and the stars shone.", "night_sky", "calm"),
    ]
    mains = cast.assign(scenes, "A fox goes on a journey")
    assert mains == ["fox"]
    assert scenes[0].cast == ["fox"] and scenes[2].cast == ["fox"]
    assert scenes[1].kit == "ocean" and scenes[1].cast == ["dolphin"]  # no fox at sea


def test_action_words() -> None:
    assert assemble._action("The fox hurried home.") == "run"
    assert assemble._action("She sat by the fire.") == "idle"
    assert assemble._action("They walked on.") == "walk"


def test_story_places_characters(tmp_path: Path, media: Media) -> None:
    text = (
        "A little fox walked across the sunny valley. The fox met a friendly dog. "
        "Later the fox sat by the warm campfire."
    )
    plan = plan_story(PlanRequest(text, mode="script"))
    plan.scenes = [Scene(s, k, "calm") for s, k in (
        ("A little fox walked across the sunny valley.", "sunny_valley"),
        ("The fox met a friendly dog.", "sunny_valley"),
        ("Later the fox sat by the warm campfire.", "campfire"),
    )]  # fmt: skip
    cast.assign(plan.scenes, "A little fox's day")  # the idea names only the fox
    speak, amb, mus = stand_ins(media.ffmpeg)
    res = assemble.assemble(
        plan, tmp_path / "p.storyreel", speak=speak, ambience=amb, music=mus,
        ffmpeg=media.ffmpeg, ffprobe=media.ffprobe, height=180,
    )  # fmt: skip
    with store.open_project(res.project) as p:
        rows = p.con.execute(
            "SELECT track_id, start, duration, props FROM clip WHERE kind = 'generator'"
        ).fetchall()
    acts = [(r[0], r[1], r[2], json.loads(r[3])) for r in rows if '"actor"' in r[3]]
    fox = sorted((a for a in acts if a[3]["species"] == "fox"), key=lambda a: a[1])
    # valley stretch (2 scenes, one segment): walk in then stand; campfire: sitting (idle)
    assert [a[3]["action"] for a in fox] == ["walk", "idle", "idle"]
    assert fox[0][1] + fox[0][2] == fox[1][1]  # idle starts where the walk ends
    dog = [a for a in acts if a[3]["species"] == "dog"]
    assert 1 <= len(dog) <= 2 and min(d[1] for d in dog) > fox[0][1]  # joins in scene 2
    assert {a[3]["seed"] for a in fox} == {fox[0][3]["seed"]}  # same look throughout
