"""Text → StoryPlan.

Modes
  llm     the local writer invents the story from an idea (auto language pick of model)
  script  the user's own text IS the narration; we only split it into scenes and choose
          visuals/moods (instant, best quality for a hand-written Hindi script)
  rules   no model available: the idea text is used as narration, split like a script
  auto    llm if a writer model is installed, else rules (with a warning)

Long stories (more scenes than one call can write well) are planned as an outline first,
then each section is expanded in its own call, so memory and prompt size stay constant.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from storyreel.errors import StoryReelError
from storyreel.story import kits as kitmod
from storyreel.story import llm
from storyreel.story.lang import LANG_NAMES, resolve_language
from storyreel.story.schema import (
    MAX_SCENES_PER_CALL,
    PlanError,
    Scene,
    StoryPlan,
    narration_chars,
    outline_schema,
    scene_count,
    story_schema,
    validate,
)

log = logging.getLogger(__name__)
Progress = Callable[[str], None]

_SENT = re.compile(r"(?<=[.!?।॥])\s+|\n+")


@dataclass(frozen=True)
class PlanRequest:
    text: str
    length_s: float = 60.0
    lang: str = "auto"
    mode: str = "auto"  # auto | llm | script | rules
    writer: str = "auto"  # auto | qwen2.5-1.5b | phi-3.5-mini


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s and s.strip()]


def script_to_plan(text: str, lang: str, *, source: str, title: str = "") -> StoryPlan:
    """Group sentences into scenes of about one scene's worth of narration each."""
    sentences = split_sentences(text)
    if not sentences:
        raise PlanError("The text is empty.", fix="Write at least one sentence.")
    limit = narration_chars(lang)
    scenes_text: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > limit:
            scenes_text.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    scenes_text.append(cur)
    scenes = []
    prev_kit = "sunny_valley"
    for t in scenes_text:
        kit = kitmod.guess_kit(t, fallback=prev_kit)
        scenes.append(Scene(t, kit, kitmod.guess_mood(t, fallback=kitmod.KITS[kit].mood)))
        prev_kit = kit
    first = sentences[0]
    if not title:
        title = first if len(first) <= 60 else first[:57].rsplit(" ", 1)[0] + "..."
    return StoryPlan(title, lang, scenes, source)


def _system_prompt(lang: str) -> str:
    name = LANG_NAMES.get(lang, "English")
    return (
        "You are a scriptwriter for short narrated story videos on YouTube. "
        f"Write ALL narration, the title and on-screen text in {name}"
        + (" using Devanagari script" if lang == "hi" else "")
        + ". Each scene's narration is one to three vivid spoken sentences that continue the "
        "story; scenes must connect into one story with a beginning, middle and end. "
        "For each scene choose the visual kit that best fits it from this list:\n"
        + kitmod.describe_for_writer()
        + "\nChoose a mood per scene. onscreen is an optional short caption (or empty)."
    )


def _tokens_for(n_scenes: int, lang: str) -> int:
    # Devanagari costs ~3x the tokens of English per character in these tokenizers.
    per_scene = narration_chars(lang) // (2 if lang == "en" else 1) + 60
    return min(8000, 120 + n_scenes * per_scene)


def _llm_plan(req: PlanRequest, lang: str, progress: Progress) -> StoryPlan:
    key = llm.pick_model(lang, req.writer)
    info = llm.MODELS[key]
    n = scene_count(req.length_s)
    kits = list(kitmod.KITS)
    warnings: list[str] = []
    if lang not in info.langs:
        warnings.append(
            f"{key} writes {LANG_NAMES.get(lang, lang)} poorly; install phi-3.5-mini "
            "for better results."
        )
    progress(f"writer {key} ({info.licence}) loading...")
    ctx = 8192 if lang != "en" or n > 6 else 4096
    with llm.LocalWriter(key, ctx=ctx) as w:
        progress(f"writer ready in {w.load_s:.1f} s; writing {n} scenes in {LANG_NAMES[lang]}...")
        system = _system_prompt(lang)
        scenes: list[Scene] = []
        title = ""
        if n <= MAX_SCENES_PER_CALL:
            res = _chat_retry(
                w,
                system,
                f"Story idea: {req.text}\nWrite exactly {n} scenes.",
                story_schema(n, kits, lang),
                _tokens_for(n, lang),
                progress,
            )
            plan = validate(res.data, lang=lang, kits=kits, source=f"llm:{key}")
            title, scenes = plan.title, plan.scenes
        else:
            sections = max(2, -(-n // MAX_SCENES_PER_CALL))
            progress(f"long story: outline of {sections} sections first")
            out = _chat_retry(
                w,
                system,
                f"Story idea: {req.text}\nOutline the whole story in "
                f"exactly {sections} sections (one paragraph each).",
                outline_schema(sections),
                300 * sections,
                progress,
            )
            title = str(out.data["title"])
            per = [n // sections + (1 if i < n % sections else 0) for i in range(sections)]
            for i, (summary, k) in enumerate(zip(out.data["sections"], per, strict=True), 1):
                progress(f"section {i}/{sections}: {k} scenes")
                prompt = (
                    f"Story: {title}\nWhole outline: {' | '.join(out.data['sections'])}\n"
                    f"Now write section {i}: {summary}\nWrite exactly {k} scenes."
                )
                res = _chat_retry(
                    w, system, prompt, story_schema(k, kits, lang), _tokens_for(k, lang), progress
                )
                scenes += validate(res.data, lang=lang, kits=kits, source="").scenes
        log.info("writer done", extra={"model": key, "peak_rss": w.peak_rss, "load_s": w.load_s})
    changed = kitmod.refine_kits(scenes)
    if changed:
        progress(f"visuals: corrected {changed} scene kit(s) from the narration's own words")
    if lang != "en" and any(_mostly_latin(s.narration) for s in scenes):
        warnings.append(f"Some narration came back in English instead of {LANG_NAMES[lang]}.")
    return StoryPlan(title, lang, scenes, f"llm:{key}", warnings)


def _mostly_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.6


def _chat_retry(
    w: llm.LocalWriter,
    system: str,
    user: str,
    schema: dict[str, object],
    max_tokens: int,
    progress: Progress,
    attempts: int = 2,
) -> llm.ChatResult:
    last: StoryReelError | None = None
    for attempt in range(1, attempts + 1):
        try:
            res = w.chat(system, user, schema=schema, max_tokens=max_tokens, seed=7 + attempt)
            progress(f"  {res.tokens} tokens in {res.wall_s:.0f} s ({res.tok_per_s:.1f} tok/s)")
            return res
        except llm.LLMError as exc:
            last = exc
            progress(f"  attempt {attempt} failed: {exc.message}")
            max_tokens = int(max_tokens * 1.5)
    assert last is not None
    raise last


def plan_story(req: PlanRequest, progress: Progress | None = None) -> StoryPlan:
    say = progress or (lambda m: None)
    lang = resolve_language(req.text, req.lang)
    mode = req.mode
    if mode == "script":
        return script_to_plan(req.text, lang, source="script")
    if mode == "rules":
        return script_to_plan(req.text, lang, source="rules")
    if mode not in ("auto", "llm"):
        raise PlanError(f"Unknown mode {mode!r}.", fix="Use auto, llm, script or rules.")
    try:
        return _llm_plan(req, lang, say)
    except llm.LLMUnavailableError as exc:
        if mode == "llm":
            raise
        say(f"no local writer ({exc.message}); using the rule-based planner")
        plan = script_to_plan(req.text, lang, source="rules")
        plan.warnings.append("No AI writer installed: your text was used as the narration.")
        return plan
    except llm.LLMError as exc:
        if mode == "llm":
            raise
        say(f"writer failed ({exc.message}); using the rule-based planner")
        plan = script_to_plan(req.text, lang, source="rules")
        plan.warnings.append(
            f"The AI writer failed ({exc.message}); your text was used as the narration."
        )
        return plan
