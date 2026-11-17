"""Phase 4 story layer: language pick, kits, planning modes, schema, subtitles, writer."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from storyreel.story import kits, llm, planner, schema, subtitles
from storyreel.story.lang import detect_language, resolve_language
from storyreel.story.schema import PlanError, Scene

HINDI = "पहाड़ों में रहने वाली एक छोटी लड़की और उसकी बकरी की कहानी। बर्फ़ गिर रही थी। रात को तारे चमके।"


# -- language ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("A fox in the snow.", "en"),
        (HINDI, "hi"),
        ("தமிழ் கதை", "ta"),
        ("తెలుగు కథ", "te"),
        ("ಕನ್ನಡ ಕಥೆ", "kn"),
        ("123 !!", "en"),
        ("Hello दोस्त, कैसे हो आप?", "hi"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text) == lang


def test_resolve_language_override_and_validation() -> None:
    assert resolve_language(HINDI, "en") == "en"
    with pytest.raises(Exception, match="Unknown language"):
        resolve_language("x", "klingon")


def test_auto_writer_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "installed", lambda key: True)
    assert llm.pick_model("en") == "qwen2.5-1.5b"
    assert llm.pick_model("hi") == "phi-3.5-mini"
    assert llm.pick_model("hi", "qwen2.5-1.5b") == "qwen2.5-1.5b"
    monkeypatch.setattr(llm, "installed", lambda key: key == "qwen2.5-1.5b")
    assert llm.pick_model("hi") == "qwen2.5-1.5b"  # falls back to what is installed
    monkeypatch.setattr(llm, "installed", lambda key: False)
    with pytest.raises(llm.LLMUnavailableError):
        llm.pick_model("en")


def test_licences_are_commercial_ok() -> None:
    assert {m.licence for m in llm.MODELS.values()} <= {"Apache-2.0", "MIT"}
    assert not any("3b" in k and "qwen" in k for k in llm.MODELS)  # qwen-research licence


def test_prompt_templates() -> None:
    p, stops = llm.build_prompt("chatml", "S", "U")
    assert p.endswith("<|im_start|>assistant\n") and stops == ["<|im_end|>"]
    p, stops = llm.build_prompt("phi3", "S", "U")
    assert p.startswith("<|system|>\nS<|end|>") and stops == ["<|end|>"]


# -- kits ----------------------------------------------------------------------------------
def test_kits_reference_real_generators() -> None:
    from storyreel.render.fxspec import GENERATORS

    for k in kits.KITS.values():
        for gtype, params in k.layers:
            assert gtype in GENERATORS, (k.name, gtype)
            assert set(params) <= set(GENERATORS[gtype]), (k.name, gtype, params)
        assert k.mood in schema.MOODS


@pytest.mark.parametrize(
    ("text", "kit"),
    [
        ("Rain and thunder all night.", "storm"),
        ("Snow covered the frozen pass.", "snow_valley"),
        ("They sat by the warm campfire.", "campfire"),
        ("बारिश और बादल।", "storm"),
        ("पहाड़ों और नदी के पास एक गाँव।", "sunny_valley"),
        ("रात में तारे चमक रहे थे।", "night_sky"),
    ],
)
def test_guess_kit(text: str, kit: str) -> None:
    assert kits.guess_kit(text) == kit


def test_refine_kits_fixes_contradicting_picks_only() -> None:
    scenes = [
        Scene("Heavy rain and thunder.", "night_sky", "calm"),
        Scene("A quiet moment.", "campfire", "calm"),
    ]
    assert kits.refine_kits(scenes) == 1
    assert scenes[0].kit == "storm" and scenes[1].kit == "campfire"


def test_guess_mood() -> None:
    assert kits.guess_mood("She was alone and cried.") == "sad"
    assert kits.guess_mood("वह बहुत खुश थी और दोस्त के साथ हँसी।") == "happy"
    assert kits.guess_mood("A table.") == "calm"


# -- planning modes ------------------------------------------------------------------------
def test_script_mode_splits_and_keeps_words() -> None:
    text = " ".join(f"Sentence number {i} about the snowy river." for i in range(20))
    plan = planner.plan_story(planner.PlanRequest(text, mode="script"))
    assert plan.source == "script" and len(plan.scenes) > 1
    assert " ".join(s.narration for s in plan.scenes) == text  # nothing lost or invented
    limit = schema.narration_chars("en")
    assert all(len(s.narration) <= limit or "." not in s.narration[:-1] for s in plan.scenes)


def test_script_mode_hindi() -> None:
    plan = planner.plan_story(planner.PlanRequest(HINDI, mode="script"))
    assert plan.language == "hi"
    assert {s.kit for s in plan.scenes} <= set(kits.KITS)
    assert planner.split_sentences(HINDI)[1] == "बर्फ़ गिर रही थी।"


def test_auto_without_writer_falls_back_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "installed", lambda key: False)
    plan = planner.plan_story(planner.PlanRequest("A fox walks in the snow.", mode="auto"))
    assert plan.source == "rules" and plan.warnings
    with pytest.raises(llm.LLMUnavailableError):
        planner.plan_story(planner.PlanRequest("A fox.", mode="llm"))


def test_empty_text_is_a_clear_error() -> None:
    with pytest.raises(PlanError, match="empty"):
        planner.plan_story(planner.PlanRequest("   ", mode="script"))


# -- schema --------------------------------------------------------------------------------
def test_schema_fixes_scene_count_and_kits() -> None:
    sch = schema.story_schema(4, list(kits.KITS), "en")
    scenes = sch["properties"]["scenes"]
    assert scenes["minItems"] == scenes["maxItems"] == 4
    assert scenes["items"]["properties"]["kit"]["enum"] == list(kits.KITS)


def test_validate_rejects_and_repairs() -> None:
    names = list(kits.KITS)
    good = {
        "title": "T",
        "scenes": [{"narration": "A fox runs.", "kit": "storm", "mood": "weird", "onscreen": ""}],
    }
    plan = schema.validate(good, lang="en", kits=names, source="x")
    assert plan.scenes[0].mood == "calm"  # unknown mood repaired
    with pytest.raises(PlanError, match="unknown visual kit"):
        schema.validate(
            {"title": "T", "scenes": [{"narration": "Hello there", "kit": "moon"}]},
            lang="en",
            kits=names,
            source="x",
        )
    with pytest.raises(PlanError, match="no narration"):
        schema.validate(
            {"title": "T", "scenes": [{"narration": "", "kit": "storm"}]},
            lang="en",
            kits=names,
            source="x",
        )


def test_trim_to_sentence() -> None:
    cut = "First sentence. Second one ends here. Third is cut mid-wo"
    assert schema.trim_to_sentence(cut, len(cut) + 1) == "First sentence. Second one ends here."
    assert schema.trim_to_sentence("Short.", 100) == "Short."
    hi = "पहला वाक्य। दूसरा वाक्य। तीसरा अधू"
    assert schema.trim_to_sentence(hi, len(hi)) == "पहला वाक्य। दूसरा वाक्य।"


def test_scene_count() -> None:
    assert schema.scene_count(15) == 2 and schema.scene_count(60) == 6
    assert schema.scene_count(3600) == 360


def test_plan_json_roundtrip() -> None:
    plan = planner.plan_story(planner.PlanRequest(HINDI, mode="script"))
    again = schema.StoryPlan.from_json(plan.to_json())
    assert again == plan


# -- subtitles -----------------------------------------------------------------------------
def test_cues_cover_words_in_order_without_overlap() -> None:
    words = subtitles.estimate_words(
        "The fox ran across the frozen river. It stopped to look at the stars above "
        "the silent mountains and smiled.",
        0.0,
        8.0,
    )
    cues = subtitles.build_cues(words)
    assert len(cues) >= 2
    assert all(c.start < c.end for c in cues)
    assert all(a.end <= b.start + 1e-9 for a, b in pairwise(cues))
    assert all(len(line) <= subtitles.MAX_CHARS + 12 for c in cues for line in c.text.split("\n"))
    joined = " ".join(c.text.replace("\n", " ") for c in cues)
    assert joined.split() == [w for w, _, _ in words]


def test_srt_format(tmp_path: Path) -> None:
    cues = subtitles.build_cues(subtitles.estimate_words("नमस्ते दोस्तों। आज की कहानी।", 61.5, 64.0))
    out = tmp_path / "s.srt"
    subtitles.write_srt(cues, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("1\n00:01:01,500 --> ")
    assert "नमस्ते" in text


# -- real writer (only where the model is installed; never in CI) --------------------------
@pytest.mark.skipif(not llm.installed("qwen2.5-1.5b"), reason="writer model not installed")
def test_real_writer_english_story() -> None:
    plan = planner.plan_story(
        planner.PlanRequest(
            "A kite gets stuck in a tree and two friends rescue it.", length_s=30, mode="llm"
        )
    )
    assert plan.source == "llm:qwen2.5-1.5b"
    assert len(plan.scenes) == schema.scene_count(30)
    assert all(s.kit in kits.KITS and s.narration for s in plan.scenes)
