# Progress

## 2026-11-20 — Realistic step 2: sound that follows the picture and the story

### Works
- **Ambience from the footage:** each scene's nature sound comes from the tags of the clips
  on screen (waves → ocean, campfire → fire, beach → surf, monkey → jungle …), most votes
  wins and the establishing shot breaks ties; the kit's ambience only when no tag says
  anything. Consecutive same-sound scenes are one clip; changes crossfade 1.5 s on two
  alternating tracks.
- **Music by mood:** stretches of scenes with one mood (at least 8 s; shorter ones join
  the longer neighbour) each get their own music, crossfaded 2.5 s on alternating tracks.
- **Music arc:** volume keyframes start the music 4 dB down under the opening narration
  and build to full at 85 % of the film (the climax), then ease 1 dB.

### Measured (journey, 720p)
| | value |
|---|---|
| sound bed | ambience village → jungle → forest → jungle → beach → ocean → beach → fire; music mysterious → happy → calm |
| loudness | -14.0 LUFS overall; 0-5 s -10.4 (title braam), 10-35 s -15.1..-15.6, 40-45 s -12.9 (climax) |
| render | 1459 s (machine busy; colour match adds per-frame cost) |
| Tests | 362 passed |

## 2026-11-20 — Realistic step 1: footage that shows what the story says

### Works
- **CLIP matching (vision/clip.py, media/rank.py):** OpenAI CLIP ViT-B/32 (MIT; Xenova
  int8 ONNX, pinned revision, SHA-256 verified, 146 MB, used offline). Each candidate's
  preview is scored against the narration (0.6) and its search words (0.4), plus a bonus
  when the library's own tags confirm the words of a specific search. Weak matches,
  near-duplicates, drawn/CGI-tagged clips are dropped; zoo/aquarium clips are demoted.
  Shots within a scene go wide → close. `--no-match` turns it off.
- **Pixabay:** AI-generated and low-quality clips are skipped (API flags); previews and
  tags kept. Search cache is versioned (v2) so old entries without them are re-fetched.
- **Shot colour matching:** every shot moves part of the way towards the story's median
  Lab look (colormatch effect), so cuts between cameras stop jumping.
- **Vertical videos** can use landscape footage (centre crop; files tall enough).
- **ffmpeg hang guard:** ffmpeg >= 7 prints progress on a timer even when stuck, so the
  heartbeat now needs moving output; a hung step is killed after 60 s and run once more.

### Measured
| | value |
|---|---|
| CLIP load / 9 frames / 1 text | ~1.5 s / 1.25 s / ~15 ms |
| journey (9 scenes) footage step, previews uncached | ~140 s of 278 s make-before-render |
| journey render 720p with colour match | 530 s resumed (3 of 6 chunks cached) |
| loudness | -14.2 LUFS, -1.3 dBTP |
| Tests | 357 full suite + new ranking/cache/hang tests |

### Bugs found (fixed)
- Render froze for an hour: stuck ffmpeg kept printing progress (see hang guard).
- Cached Pixabay searches had no tags, so the dolphin scene lost to plain waves.

### Honest limits
- CLIP reads English only (Hindi stories rank by search words alone).
- It does not know the story's setting: a "forest" after a jungle can still be any forest.

## 2026-11-19 — Realistic style (`storyreel make --style realistic`)

A second, separate video type next to the animation style: real footage cut like a film,
a film grade and trailer sound. The two styles never mix: a realistic video never falls back
to drawn scenery or puppets; a scene without footage stops with a fix.

### Works
- **Footage (media/stock.py):** your own clip folder (`--footage-dir`, matched by file
  names), Pixabay (`PIXABAY_API_KEY`, live action only via `video_type=film`), Pexels
  (`PEXELS_API_KEY`; new keys paused by Pexels as of 2026-11-19). Searches and downloads
  are cached in the data dir; `--offline` re-makes from the cache. Licence, author and page
  of every clip go into the project and `footage_credits.txt`.
- **Wikimedia Commons:** opt-in only (`--footage commons` + `STORYREEL_CONTACT`). Without
  a contact User-Agent downloads get HTTP 429, and results are uncurated (a "village" search
  returned war and ritual footage), so it is never automatic.
- **Shots (story/realistic.py):** searches per scene ranked animal → named thing → the
  kit's main shot → people → other kit shots; scenes split into 3.5-6 s shots, each with a
  slow push-in and drift, scaled to cover the frame; no clip repeats in a story.
- **Look:** one adjustment layer (soft S-curve with lifted blacks, warm highlights / cool
  shadows, halation glow, vignette, grain), 2.39:1 letterbox bars, subtitles inside the
  bottom bar, letter-by-letter title fade, fade from/to black.
- **Sound (audio/cinematic.py, numpy, CC0):** braam under the title, whoosh peaking on
  every scene cut, riser into the last scene + impact on its first frame, hall reverb on
  the fx bus; dialogue chain (high-pass, de-mud, presence, air, 3:1 compression,
  de-esser); master glue compressor; -14 LUFS master as before.

### Measured (i3-1315U, ~0.5-1 GB RAM free)
| | value |
|---|---|
| journey story, 9 scenes, 720p, Pixabay | 47.3 s video, 9 shots from 9 clips |
| ... voice + footage + assembly | 139-245 s (footage downloads 1-4 s per clip, 2-15 MB) |
| ... render | 692-767 s (≈1.9 fps) |
| ... peak RSS | 818 MB |
| ... loudness | -14.0 LUFS, -1.5 dBFS peak |
| SFX synthesis | whoosh 0.2 s, riser 0.7 s, braam 1.2 s, impact 0.3-1.7 s |
| Tests | 349 + new tests/test_realistic.py (22) |

### Bugs found (fixed)
- Pixabay returned an anime illustration for "girl night": live-action filter added and
  people searches moved after the scene's main shot.
- Braam additive synthesis took 11 s: replaced by filtered detuned saws (1.2 s).

### Honest limits
- Stock footage cannot keep the same characters across shots; the story's people do not
  appear. Best for nature/documentary-style narration.
- Weak matches still happen (the dolphin scene showed plain aerial water).
- Rendering ≈2 fps at 720p on this laptop.

## 2026-11-17 — Phase 4: story → video (`storyreel make`), characters, environments

One command turns an idea (or your own script, or a saved plan) into a finished video,
fully offline: `storyreel make --idea "..." --length 60s --out story.mp4`.

### Works
- **Writer:** local llama.cpp server (Qwen2.5-1.5B default, Phi-3.5-mini via `--writer`),
  JSON-schema-constrained; outline-then-sections for long stories; rules fallback.
- **Other languages:** written in English, then translated by IndicTrans2-200M (MIT,
  pinned revision): Hindi, Tamil, Telugu, Kannada, Bengali, Gujarati.
- **Characters (new):** 34 animated cut-out puppets drawn in code (render/actors.py):
  - 15 animals: fox, dog, wolf, cat, deer, horse, cow, goat, sheep, tiger, lion, elephant,
    bear, rabbit, monkey.
  - 6 people: man, woman, boy, girl, grandpa, grandma.
  - 9 birds: eagle, parrot, crow, sparrow, seagull, owl, peacock, hen, duck.
  - 4 sea creatures: fish, dolphin, whale, turtle.
  - Movement: legs use IK with planted feet (stride tied to speed); hop, flight with
    glides, dolphin leaps, idle breathing and blinking, contact shadows.
- **Casting (story/cast.py):** characters come from the story's own words (English + Hindi
  stems). The idea's main characters appear in every scene; characters named in the idea
  but left out by the writer appear in their home scenes (a tiger in the jungle); a scene
  with a sea creature moves to water. On screen they walk in and stop, run ("hurried"),
  or rest ("sat", "watched"); one clip per stretch of same-kit scenes.
- **Environments (new):** 14 kits: night_sky, storm, snow_valley, sunny_valley, campfire,
  misty_forest, forest, jungle, beach, ocean, hill_station, ice_land, village, desert.
  - New layers: sky (sun/moon/stars/drifting clouds), animated sea and beach surf, hills
    (tea rows, houses, lit windows), trees (pine/round/palm/jungle/bamboo), swaying grass
    with flowers, ice with icebergs, houses/huts.
  - The fire layer has a campfire mode with logs.
- **Sound:** a Kokoro voice, a procedural ambience per kit (see "Phase 4 sound" above),
  a mood music bed, and ducking.
  - Mix: ambience −10 dB, music −11 dB.
  - Final master: −14 LUFS / −1 dBTP.
- **Subtitles:** per-scene cues, burned in and written as an .srt.
- **`--plan-file`:** re-make a written story (plan.json) without running the writer again.

### Measured (i3-1315U, 0.6-1.3 GB RAM free, user apps open)
| | value |
|---|---|
| Translation, 3 sentences | hi 6.4 s, ta 6.5 s, te 5.0 s; 1.0 GB RSS (first load ~10 s) |
| Showcase "journey" (9 scenes, 6 kits, 26 character clips, 720p) | 47.3 s video |
| ... voice + sound + assembly | 296 s |
| ... render | 900 s (≈1.6 fps; machine paging) |
| ... peak RSS | 505 MB |
| ... loudness | −14.1 LUFS, −1.4 dBFS peak |
| Fox story (Phi writer, 720p) | 44.5 s for a 45 s target, 373 s total, 458 MB |
| Scenery cost per frame, 960×540 | 45-100 ms (static parts drawn once) |
| Tests | 330+ (new: tests/test_world.py) |

### Bugs found this phase (fixed)
- IndicTrans2's remote code crashes on transformers 4.57 cache objects: KV cache disabled.
- Windows memory cap race: a child that exits instantly cannot be capped
  (ERROR_ACCESS_DENIED). Children now start suspended, join the job, then resume.
- Colour-list parameters (flower_colors) were rejected by the editor.
- Characters stood on the river and in a full-width wall of fire: river curve flipped,
  campfire mode added, stop positions moved left of centre.

### Honest limits
- Flat cut-out cartoon style, not realistic footage.
- Characters walk in again at each kit change; no talking mouths.
- One generic look per species; people get random clothes per seed.
- Rendering is slow on this laptop (≈1.5-4 fps at 720p with characters).
- The writer sometimes drops characters; the cast fallback places only those named in the
  idea.
- Nobody has listened to the voice or ambiences yet: every judgement so far is from
  measurements.
- Weakest sounds (from measurements): rooster, crow, parrot; the icy wind may sound
  synthetic.

## 2026-11-17 — Phase 4 sound: environments

Ten new procedural ambiences, all made in code (numpy; no samples, no downloads), through
the same `make_ambience` / `storyreel ambience` as before: `ocean`, `beach`, `jungle`,
`forest`, `thunderstorm`, `ice`, `waterfall`, `hill_station`, `village`, `stream`.

### How they are built
- **Long noise events** (`_Swells`): ocean waves (each 5-12 s: a dark swell that brightens,
  a crash, then fizzing foam that slides back), thunder rolls (several decaying rumbles;
  distant = slow and low, close = sharp crack then a bright heavy roll) and cicada choruses
  that swell up and die away. Size, length, timing and left/right position are random for
  every event, and events overlap.
- **Animals and objects** (new `audio/sfx.py`): seven songbird types (tweet, trill, whistle,
  warble with its own remembered phrase, chip, sweep, chatter), seagull "kee-ow" + laugh,
  rooster, crow, dove, cuckoo, Asian koel, owl, tropical whoops, parrot, woodpecker drum,
  temple bell (inharmonic partials, slow beating), cow bell clinks, water drips, thunder
  crack, ice crack ("pew" + boom) and ice creak. Each animal sits at a fixed place, sings
  bouts then rests, and every call is re-synthesised with small changes. Distant ones are
  low-passed and sent through the procedural reverb (valley echo for bells and far calls).
- `birds` now uses five resident songbirds of different species with a light forest echo;
  `night` gained an occasional distant owl and crickets that stop and start.

### Measured (seed 1; loudness = ffmpeg ebur128)
| kind | LUFS (20 s file) | true peak | speed (x realtime, 20 s) |
|---|---|---|---|
| ocean | -23.5 | -4.1 dBTP | 14x |
| beach | -23.1 | -8.8 | 9x |
| jungle | -26.4 | -15.7 | 4-10x |
| forest | -25.3 | -14.7 | 9x |
| thunderstorm | -23.7 | -7.8 | 12x |
| ice | -27.7 | -14.0 | 28x |
| waterfall | -24.2 | -12.9 | 16x |
| hill_station | -27.8 | -13.8 | 6x |
| village | -28.4 | -15.0 | 9x |
| stream | -26.2 | -15.1 | 22x |

Gains were calibrated on two seeds x 180 s (sparse sounds such as roosters and thunder need
a long stretch) to targets between -24 and -28 LUFS. Short files vary by a few dB because a
5-20 s stretch may or may not contain a crash, a thunder roll or a rooster.
Memory stays flat: a 10-minute jungle (the heaviest) peaked at 86 MB working set, the same
as a 20 s one (84 MB); 10-minute thunderstorm 49 MB. Speed varies with machine load: the
10-minute jungle ran at 5.4x realtime, thunderstorm at 15x.

Listening files (20 s each): `examples/out/ambience/<kind>.wav` (not in git).

### Tests
273 pass. New: every environment renders cleanly (exact length, no NaN, peak < -1 dBFS,
loudness inside its window, true peak <= -1 dBTP), no click at block boundaries or at the
loop seam, deterministic per seed, a 30 s ocean does not repeat itself, 120 s thunderstorm
memory equals 8 s memory, and every synthesised call starts and ends at silence.

### Honest limits
These are stylised synthetic sounds checked by numbers, not by listening here. Weakest
likely: the rooster and crow (buzzy synthesis, recognisable but not natural), the parrot,
and the ice wind's howl (resonant tones can sound "synth"). Strongest likely: stream,
waterfall, rain/thunderstorm and ocean, which are shaped noise.
Story kits still use the old kinds (e.g. `storm` -> rain); switching them to the new ones
is a one-line change per kit.

## 2026-11-17 — Phase 4a: voice and procedural sound

Design: ARCHITECTURE.md §3b.

### Works
- **Offline TTS** (`storyreel.voice`): Kokoro-82M through `kokoro-onnx`, English (28 voices)
  and Hindi (`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`). API: `available`, `voices`,
  `ensure_models`, `synthesize` -> `SpeechResult(path, duration_s, sample_rate, words)`.
  Inference runs only in a child process under a Job Object cap (1.5 GB), with heartbeat and
  timeout; text is spoken sentence by sentence and streamed into the WAV. Typed errors:
  `ModelMissingError` (fix = download command), `ModelDownloadError` (offline / bad hash),
  `SynthesisMemoryError`, `SynthesisTimeoutError`, `SynthesisError`, `VoiceError` (bad input).
- **Hindi works**: espeak-ng turns Devanagari into Hindi IPA (e.g. "एक समय की बात है" ->
  `ˈeːk sˈʌmˌɛj ki bˈaːt hɛː`) and the Hindi voices speak it; word timings matched the
  text one-to-one on every Hindi test sentence. Intelligibility was not judged by ear
  (no listening possible here); objective checks only: non-silent, energy inside word spans.
- **Word timestamps** from the model's phoneme durations, matched to words, then anchored
  to the measured speech onset/offset of each sentence (raw durations drift 0.1-0.25 s).
  Sentences where the words do not match one-to-one fall back to a length-proportional
  estimate. 40/40 sentences of the 131 s test narration had model timings.
- **Procedural sound** (`audio/procedural.py`): ambiences rain, wind, fire, river, birds,
  night, room; music moods calm, happy, sad, tense, mysterious, epic (chord progression,
  pads, bass, arpeggio, drum for epic, procedural reverb). Deterministic per seed, 24-bit
  stereo WAV, any length at constant memory, loop-crossfaded head/tail, peak < -1 dBFS.
- **Integration**: `media/generated.py` writes into `<project>/assets/generated/`, imports
  with `is_generated=1`, licence and a generator `source_url`, and places the clip on a free
  track of the right bus (speech -> dialogue, music -> music, ambience -> ambience). Music
  and ambience duck under speech through the Phase 3 buses.
- **CLI**: `voices --lang en|hi`, `models download|list --lang`, `say PROJECT "text" --lang
  hi [--voice] [--speed] [--at]`, `ambience PROJECT rain --at --dur [--seed] [--gain-db]`,
  `music PROJECT calm --at --dur ...`. `doctor` reports `TTS model (en|hi)` as OK, or INFO
  with the download command.
- 180 tests (123 + 57): procedural length/clipping/seam/spectrum/loudness/determinism/memory,
  sentence splitting (incl. danda), word alignment, SHA-256 verification + offline error with
  a fake model, missing-model errors, doctor, real English + Hindi synthesis, memory-cap error,
  CLI end to end with a rendered mix at -14 LUFS. TTS tests skip cleanly without the model
  (checked with an empty model dir: 8 pass, 4 skip). ruff, ruff format, mypy --strict clean.

### Measured (same laptop; another session's LLM + IDE running: 22-55 % CPU busy, 0.7-1.0 GB free)

| Test | Result |
|---|---|
| TTS English, 5 sentences, 16.0 s audio | 24.0 s wall incl. 2.3 s model load = **0.67x real time** (0.74x without load) |
| TTS Hindi, 4 sentences, 16.1 s audio | 23.6 s wall incl. 2.5 s load = **0.68x real time** (0.76x without load) |
| TTS English, 40 sentences, 130.8 s audio | 170.9 s wall = **0.77x real time** |
| TTS worker memory (all three runs) | **peak working set 0.47-0.49 GB**, Job Object commit peak 0.68-0.70 GB; does not grow with length |
| TTS worst seen under heavy load | Hindi 3.4 s in 21.1 s (0.16x) during a busy period; speed depends on free CPU |
| Kokoro variants, same sentence, warm | fp32 ~1.0x / fp16 ~0.85x / int8 0.15-0.22x real time; RSS ~0.48 / 0.59 / 0.40 GB (fp32 with arena off) |
| Memory cap 150 MB | `SynthesisMemoryError`, caller unaffected, no partial file left |
| 60 s ambience, each kind | **2.0-5.2 s** (12-30x real time); night (4 cricket layers) is slowest |
| 60 s music, each mood | **15-24 s** typical (2.5-4x real time); one run at 195 s while the machine was saturated |
| 10 min rain vs 30 s rain | peak working set **45 MB both**; 16.3 s for 10 min |
| 10 min calm music vs 30 s | peak working set **61 MB both**; 103 s for 10 min |
| Levels (60 s, ebur128) | rain -23.6, wind -26.8, fire -25.9, river -23.8, birds -28.7, night -32.1, room -41.9 LUFS; music -17.6..-18.3 LUFS; true peak <= -5.8 dBTP everywhere |
| Spectra (60 s) | rain 60 % of energy above 4 kHz; wind 89 % below 1 kHz; birds 82 % in 4-8 kHz; night 52 % in 4-8 kHz (crickets) |

### Downloaded files (kept in `%LOCALAPPDATA%\StoryReel\models\kokoro-v1.0\`)

| File | Size | SHA-256 | Verified against | Licence |
|---|---|---|---|---|
| `kokoro-v1.0.onnx` | 325,505,369 B | `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a` | GitHub release asset digest (thewh1teagle/kokoro-onnx `model-files-v1.1`): match | Kokoro-82M weights Apache-2.0 (HF `hexgrad/Kokoro-82M` cardData.license); ONNX export repo MIT |
| `voices-v1.0.bin` | 28,214,398 B | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` | same release, digest match | Apache-2.0 (voices are part of Kokoro-82M) |

Evaluated and deleted (not used): `kokoro-v1.0.int8.onnx` (114,119,327 B, `ae315a79…ee9c`,
digest match; 5x slower) and `kokoro-v1.0.fp16.onnx` (163,527,961 B, `f3a290d3…ac96`, digest
match; slower than fp32, same RAM).

New Python dependencies (one: `kokoro-onnx`, needed to run Kokoro; the rest are its
dependencies): kokoro-onnx 0.6.1 (MIT), onnxruntime 1.32.0 (MIT), phonemizer 3.6.0
(**GPL-3.0**), espeakng-loader 0.2.4 (MIT, bundles **espeak-ng, GPL-3.0**), attrs (MIT),
dlinfo (MIT), joblib (BSD-3), cloudpickle (BSD-3), flatbuffers (Apache-2.0), protobuf
(BSD-3). All free and open source. The GPL parts cover redistributing StoryReel's binaries
(source must be offered); they place no restriction on the audio produced, so monetised
videos are fine.

### Honest limits
- **TTS is slower than real time** on this laptop (0.7-0.8x when the CPU is shared, ~1x
  when idle): a 60 s narration takes ~1.3 min, an 8 min one ~10 min. Model load adds 2-5 s
  per `synthesize` call (one worker per call); batching many lines into one call is cheaper.
- Hindi quality is Kokoro's (its Hindi voices are rated lower than its English ones in
  VOICES.md) and espeak-ng's Hindi phonemiser: no schwa-deletion tuning, English words in
  Latin script inside Hindi text are read by espeak's rules. Not judged by a listener.
- Word times are approximate (syllable-level), not forced alignment.
- Procedural music is simple ambient pad music (4-chord loops per mood, one seeded variation:
  the start chord); it will sound repetitive over minutes. The loop crossfade can land in the
  middle of a chord. No listening test was possible; checks are objective only.
- Ambiences are synthetic approximations; birds/crickets are stylised.
- `available()` checks file size + mtime against the verified marker, not a full re-hash.
- Measured on Windows only, with CPU shared with another session; numbers vary run to run.

### Next
Story planner (other session) calls `voice.synthesize` and `audio.procedural`; captions from
`SpeechResult.words`; optional Piper voices if more Indic languages are needed.

## 2026-11-16 — Phase 3: audio engine

Design and the parts that differ from the first draft: ARCHITECTURE.md §3a.

### Works
- **Buses**: every audio track routes to dialogue / music / fx / ambience; master = their
  sum. Per-bus gain, mute, effect chain; master gain + effect chain. Schema migration 3
  (`track.bus`, `bus`, `bus_effect`, `project.audio`); old projects migrate in place.
- **Per clip**: gain, balance pan, fade in/out (half-cosine), `volume` keyframes in dB
  (hold / linear / bezier, same keyframe table and easings as video), audio effect chain.
- **Effects** (ffmpeg filters, all offline): parametric EQ band (stackable), high/low-pass,
  compressor, limiter, de-esser, noise reduction (`afftdn`), reverb = `afir` convolution
  with procedurally generated impulse responses (small_room, hall, cathedral, cave; synthetic,
  not measured rooms).
- **Ducking**: music / ambience (and fx if enabled) duck under dialogue clips by a
  deterministic, look-ahead gain envelope (default −12 dB, 0.25 s attack before the line,
  0.6 s release), computed by numpy per chunk window, applied with `amultiply`.
- **Chunk safety**: chunks with effects render from a pre-roll (2 s + longest reverb tail)
  and trim it; all automation is a pure function of timeline position.
- **Loudness**: two-pass. `ebur128` measures the concatenated master PCM; the final encode
  applies the gain and (only if needed) a 4x-oversampled lookahead limiter; the output is
  measured again and re-encoded with corrections if it misses (≤ 3 encodes). Target −14
  LUFS (YouTube), `broadcast` −23, any value −36..−5; true peak ≤ −1 dBTP by default.
- **Layouts / codecs**: stereo, 5.1, 7.1 projects; mono → FC and stereo → FL/FR (no fake
  upmix); downmix to stereo (or 7.1 → 5.1) in the final encode; AAC, AC-3, E-AC-3, Opus,
  FLAC; impossible combinations (e.g. AC-3 7.1) fail before any chunk renders.
- **Stems**: per-bus FLAC (24-bit) or float WAV, same gain as the master, sample-aligned.
- **Test audio**: `storyreel gen-audio speech|music|ambience|fx` (ffmpeg lavfi only).
- **CLI**: `loudness <file> [--json]`, `track <p> A2 bus=music`, `bus <p> music gain_db=-3
  duck_db=-14`, `afx <p> <clip|bus> <type> k=v` / `--remove N`, `audio <p> layout=5.1
  target_lufs=broadcast`, `set <p> <clip> pan=.. fade_in=.. fade_out=..`, `keyframe <p>
  <clip> volume <t> <dB>`; `render --acodec --abitrate --downmix --stems --target-lufs
  --no-normalize`; `info --clips` shows bus routing, clip gain/pan/fades/volume keys/effects,
  bus settings and chains. `examples/phase3_audio_demo.py` builds a 4-track mix with CLI
  calls only. Every edit is one undoable command.
- 115 tests (94 before + 21): loudness ±0.5 LU and TP ≤ −1 dBTP checked by an independent
  `ebur128` pass (AAC −14, Opus −23), ducking depth, chunked-vs-single continuity with a
  negative control, stems sum vs master, 5.1 channel routing + downmix to AC-3 stereo,
  codec/layout refusal, CLI end to end, envelope chunk-invariance, IR, migration, undo.
  ruff, ruff format, mypy --strict clean.

### Measured (same laptop, another render running concurrently, ~0.5-1 GB RAM free)

| Test | Result |
|---|---|
| 60 s, 4 audio tracks (3 dialogue lines with HP+denoise+de-esser+compressor, music with fades + bus EQ + ducking, ambience with volume keyframes, fx panned + hall reverb, dialogue bus small-room reverb, master compressor), stereo AAC, 320x180 video | **render 9.2 s wall** (CLI incl. start-up): 8 audio chunks 3.4 s (each 10 s chunk + 4.0 s pre-roll), video 0.55 s, loudness pass 1 0.23 s, encode with limiter 1.2 s, then verify + 4 FLAC stems |
| same, memory | **peak tree RSS 106 MB**; largest audio-chunk ffmpeg 64 MB |
| same, output | **−14.1 LUFS, true peak −1.5 dBTP**, LRA 6.7 LU, gain +8.3 dB, 1 encode |
| re-render unchanged | 100 % cache hits, 1.5 s |
| 30 s 5.1 E-AC-3 | −14.1 LUFS, −1.5 dBTP, 1 encode; peak tree RSS 132 MB |
| 15 s 7.1: Opus / FLAC / AAC / AAC downmixed to stereo | −14.1 LUFS each; TP −1.4 / −1.1 / −1.5 / −1.5 dBTP; 1 / 1 / 2 / 1 encodes |
| worst case seen: 7.1 with 4 reverbs (cathedral, cave, hall, small room) | 6.4 s pre-roll; 218 MB per audio-chunk ffmpeg (8-channel 4.4 s IRs in `afir`) |
| chunk continuity, 1 s chunks vs one chunk (compressor, de-esser, EQ, reverb, fades, keyframes) | max error ≤ 3e-8 (float rounding) |
| same, streams through `afftdn` | max error 2.2e-4 = −57 dB (master) / −51 dB (ambience) of peak, correlation > 0.9999 |
| same chunks **without** pre-roll (negative control) | max error 0.02-0.04 (−8 to −12 dB of peak): audible clicks |
| ducking, steady tone under one dialogue clip | −12.0 dB ± 0.1 (test tolerance), ramp starts before the line |
| stems summed vs 24-bit FLAC master (no limiter engaged) | residual < −60 dB of master |

### Honest limits
- `afftdn` is not chunk-exact (its adaptive noise estimate depends on where it started);
  the boundary difference is ~−50 dB of peak, measured above. Everything else matches to float rounding.
- Ducking follows dialogue *clip spans*, not detected speech: pauses inside a long dialogue
  clip do not release the duck. Split the clip, or use a shallower `duck_db`.
- Only `volume` is keyframable on audio. Pan and audio-effect parameters are static.
  Audio of frozen clips and speed *ramps* is still muted (constant speed and reverse work).
- The true-peak limiter is `alimiter` on a 4x-oversampled signal (a close approximation,
  not an ITU-certified true-peak limiter). Lossy encoders overshoot it (AAC: +2.3 dB on a
  heavily limited test signal; up to 0.4 dB on the demo mixes); the measure-and-re-encode
  loop is what guarantees TP ≤ −1 dBTP. Each extra encode re-muxes the video by stream
  copy (cheap, but a full pass over the file).
- Reverb impulse responses are synthetic: they model RT60, pre-delay and early reflections
  of a room of that size; they are not recordings of real spaces and have not been judged
  in listening tests. Long reverbs make every chunk render up to ~6.5 s of pre-roll.
- AC-3 / E-AC-3 stop at 5.1 (encoder limit); ffmpeg labels E-AC-3 5.1 as `5.1(side)`.
- Stems are post bus-fx/ducking, pre master-fx: with master effects or when the limiter
  engages they do not sum exactly to the master. FLAC stems are 24-bit integer and would
  clip above 0 dBFS after a large gain; WAV stems are float.
- Any bus change re-renders every audio chunk (bus settings are in every audio key);
  audio chunks cost ~0.4 s per 10 s.
- No level meters yet (UI phase); `storyreel loudness` reports integrated loudness, LRA and
  true peak only (no momentary/short-term maxima).
- Measured on Windows only. Benchmarks over 60 s (10 min / 2 h) not re-run this phase
  (RAM was shared with another render); audio memory per chunk does not grow with length.

### Next: Phase 4 (story-to-video pipeline)
Offline story engine, TTS (Piper/Kokoro) on the dialogue bus, auto-assembly, captions.

## 2026-11-15 — Phase 2: compositor, effects, text, environments

Owner requirements recorded this phase: (1) the output is for a YouTube channel (Shorts +
long-form), (2) **free, open source, offline only — no paid services** (ARCHITECTURE.md §0a).

### Works
- **Compositor (path B)** in a separate worker process under the Job Object cap; planar
  `(rgb, alpha|None)` frames; affine warp into the layer's bounding box only; OpenCV blending.
  Simple chunks still use the ffmpeg-only path A; chunks of both kinds concat with `-c copy`.
- **Keyframes** on any transform or numeric effect parameter: hold / linear / cubic-bezier
  (CSS semantics) with named easings. Ken Burns helper = one undo step.
- **Transforms**: position, scale, scale_x/y, rotation, opacity, crop; decode at up to 2x
  detail when a clip zooms in, so Ken Burns stays sharp.
- **Text**: HarfBuzz shaping + FreeType rasterising (uharfbuzz, freetype-py); script
  itemisation with per-script font fallback; Hindi, Tamil, Telugu, Kannada, Arabic verified
  visually; fill/gradient, outline, shadow, box (lower thirds); animations typewriter / fade /
  slide / pop per letter-cluster or word (clusters from HarfBuzz, so conjuncts never split).
- **Effects**: color (exposure, temperature, tint, lift/gamma/gain, contrast, saturation),
  curves, 3D LUT (.cube via ffmpeg lut3d, with mix), color match (Reinhard/Lab), blur,
  sharpen, glow, vignette, grain, chromatic aberration, glitch, lens distortion, polygon
  mask with feather, chroma key with spill suppression; presets teal-orange, noir,
  documentary, night; adjustment clips.
- **Transitions**: dissolve, fade_black, wipe ×4, slide ×2, zoom, whip; centred on the cut,
  source handles used, frozen first/last frame when there are none.
- **Speed**: constant 0.1x-16x (audio via atempo), speed ramps via keyframes (frame blend
  optional), reverse (video in 12-frame blocks; audio per chunk), freeze.
- **Environments** (pure functions of time + seed): gradient, starfield, rain, snow, fog,
  fire, particles.
- **CLI**: text, env, adjust, fx, kenburns, keyframe, transition, set, speed, frame, undo,
  redo; `info --clips`. `examples/phase2_demo.py` builds an 18 s scene with CLI calls only.
- 94 tests (exact-pixel transform/blend/transition checks, path A/B equivalence, speed and
  reverse source mapping, Indic shaping, golden frames by SSIM); ruff + mypy --strict clean.

### Measured (same laptop; 0.5-1.1 GB RAM free during runs)

| Test | Result |
|---|---|
| Demo, 18 s 1080p, 3 scenes, up to 7 layers | 540/540 frames, **peak RSS 833 MB**, job commit 1110 MB, 175 s wall (3.1 fps) |
| Compositor fps at 1080p, per demo scene | starfield+2 titles 8.2; photo+KB+grade+fog+rain+lower-third+dissolve 3.8; fire+embers+typewriter 5.1 |
| Same 4-layer scene at 720p / 1080p | 9.5 / 4.9 fps (was 4.5 fps at 720p before optimisation) |
| Colour effect at 720p | 4 ms (LUT) vs 125 ms (float numpy) |
| Generators at 1080p | rain 7, fog 9, snow 11, particles 16, fire 19, starfield(1400) 19 ms/frame |
| Text layout / styled frame | 26 ms once / memoised per animation state |
| Path A vs path B, same plain clip | frame timing identical; worst-frame PSNR 36.9 dB (chroma round trip) |
| Path A 10 s 1080p bench (regression) | 300/300 frames, 634 MB peak, PASS |

### Bugs found by tests/visual review this phase (all fixed)
Decoder `-map` vs filtergraph conflict; fog nearly invisible (6 % mean alpha); fire drawn at
the top of the frame and blocky; bicubic upscale overshoot (alpha −0.02/1.01); path A not
converting the BT.601 matrix (swscale skips yuv→yuv matrix changes); path B picking a
different source frame than ffmpeg's fps filter at 25→30 fps.

### Honest limits
- **Render speed for effect-heavy 1080p is 3-5 fps on this laptop.** A 60 s story ≈ 4-6 min;
  an 8-min fully-composited video ≈ 30-45 min at 1080p (≈ half at 720p). Simple footage
  edits stay on path A at 35-90 fps. Next levers: parallel chunk workers when RAM is free,
  OpenCL (cv2.UMat on the Intel iGPU), 8-bit blending.
- Mixed-direction (bidi) text within one line is not reordered; single-direction lines are.
- Text currently uses OS fonts (Nirmala UI on Windows). Bundling Noto (OFL) is planned so
  projects render identically on any machine.
- Not done (need models; Phase 4+): depth-map 2.5D parallax, AI background removal,
  stabilisation, optical-flow slow motion (frame blend instead).

### Next: Phase 3 (audio engine)
Buses, ducking, EQ/compressor/limiter/reverb, loudness targeting (−14 LUFS YouTube),
5.1/7.1, stems, meters; verify with ffmpeg ebur128 to ±0.5 LU.

## 2026-11-15 — Phase 1: engine core (no UI)

### Works
- **Project store**: folder + SQLite (WAL, `synchronous=FULL`), schema migrations, lock file
  with stale-lock takeover, crash recovery (deletes `*.part`, re-queues interrupted jobs).
- **Edits + undo/redo**: row-level command deltas, persisted across restarts, 200 kept.
  Overlap and source-range validation.
- **Import**: ffprobe + fast fingerprint, FTS5 search index, corrupt/empty/truncated/missing
  files skipped with plain messages. Proxies: 540p H.264 (GOP 15), waveform peaks
  (100/s, streamed via numpy, 200 B per second of audio), thumbnail sprites.
- **Timebase**: int64 flicks; exact frame/sample tiling incl. 29.97 and 23.976.
- **Renderer** (path A, ffmpeg filtergraph): chunks ≤ 10 s cut at edit points; video and
  audio as separate cached jobs; content-addressed keys with chunk-relative times; concat
  with exact durations; single AAC encode at the end; output frame count verified.
- **Memory safety**: Windows Job Object hard cap per ffmpeg job (Linux `RLIMIT_AS` path
  written, not tested on Linux yet); watchdog thread with hysteresis; CRITICAL → kill chunk,
  wait, retry with low-memory settings; exit code 5 after 3 failures.
- **Runner**: heartbeat via `-progress` (hung ffmpeg killed after 60 s silence), stderr
  drained into a 50-line buffer, typed errors.
- **CLI**: `new`, `import`, `append`, `info`, `render [--json]`. Exit codes: 0 ok,
  3 missing tools, 4 render failed, 5 resources exhausted, 6 every input unusable.
- `bench/bench_render.py --length 10s|10min|2h [--kill-after N]`.
- 56 tests; ruff, ruff format, mypy --strict clean.

### Measured (i3-1315U, 7.6 GB RAM with 0.3-0.9 GB free, Windows 11)

| Test | Result | Target |
|---|---|---|
| 10 min 1080p30 render, 189 clips, 2 video + 2 audio tracks | **peak tree RSS 724 MB**, Job Object commit peak 835 MB | < 2.5 GB |
| same, frames | 18,000 expected / 18,000 written, 0 dropped | 0 dropped |
| same, wall time incl. kill + resume | 231.9 s = 2.59x real time; chunk fps min 60, median 88 | — |
| `kill -9` after 20 of 78 chunks, then re-run | resumed; 22 video / 20 audio chunks reused; PASS | resumes |
| 10 s 1080p render | 593 MB peak, 300/300 frames, 1.0x real time (startup-dominated) | < 2.5 GB |
| Re-render unchanged project | 100 % cache hits, 0.5 s for 18.5 s video | — |
| Edit one clip's gain | only 1-2 audio chunks re-rendered, 0 video | only that chunk |
| 50 consecutive undos | worst 7.6 ms, mean 2.1 ms | < 50 ms each |
| Import 60 s 1080p clip with proxies | 10.0 s (6x real time); proxy 4.9 MB, peaks 12 KB, thumbs 128 KB | — |
| Output correctness | PSNR 44-46 dB vs source at the right times, 22 dB at a wrong frame; color clip exact; audio placement exact | — |

### Known gaps / honest notes
- Synthetic test patterns compress far better than real footage (10 min = 155 MB). Real
  camera footage will produce larger files and somewhat slower encodes; to be measured with
  real media in Phase 6.
- The 2-hour benchmark has not been run yet (Phase 6 per plan; ETA from these numbers:
  ~45-60 min wall on this laptop).
- Linux/macOS memory-limit paths are written but untested (no machine here).
- Speed changes, transforms, text, transitions: the planner refuses clips with speed ≠ 1
  instead of rendering them wrong. Those arrive in Phase 2.
- Watchdog samples at 2 Hz in-process; the bench samples at 10 Hz from outside. A spike
  shorter than 100 ms could be missed by both; the Job Object cap still bounds it.

### Next: Phase 2 (compositor & effects, the start of "animation")
numpy compositor (path B) in worker processes, transforms with keyframes + bezier,
Ken Burns/parallax, text engine with Indic shaping (Hindi/Tamil/Kannada/Telugu),
color tools + LUTs, transitions, masks, procedural environments (rain, snow, fog, fire,
particles), golden-frame SSIM tests.

## 2026-11-15 — Step 0: architecture, skeleton, doctor, hardware profiler

### Works
- `docs/ARCHITECTURE.md`: process model, SQLite schema, render graph, cache keys, watchdog
  state machine, failure-mode table. **Waiting for review before Phase 1.**
- `storyreel doctor [--json] [--no-hw-test]`: 32 checks (Python, RAM, disk, data dir,
  SQLite/FTS5/WAL, ffmpeg + 18 codec/filter features, harfbuzz, real hardware-encoder
  test encodes, memory-limit mechanism, API-key presence). Exit 0 = no FAIL, 3 = FAIL.
- `storyreel profile [--json] [--quick] [--no-bench]`: CPU/RAM/GPU/disk facts, disk write
  benchmark (fsync), libx264 and hardware H.264 encode benchmarks with ffmpeg peak RSS,
  safe defaults, honest can/cannot list. Saved atomically to the user data dir.
- JSON-lines logging (rotating 5 MB × 5) with stack traces in the log only.
- 27 tests pass; `ruff check`, `ruff format --check`, `mypy --strict` clean.

### Measured on the dev machine
i3-1315U (6 cores / 8 threads), 7.6 GB RAM, Intel UHD iGPU, Windows 11, Python 3.13.5,
ffmpeg 9.0.2 (gyan.dev full build).

| What | Number |
|---|---|
| Edge `--app` window, 2000-rect canvas | 18 processes, 1023 MB working set, 525 MB private |
| PySide6 window, same drawing | 1 process, 56 MB working set, 28 MB private |
| Bare Python 3.13 + psutil | 17 MB RSS |
| RAM available during work (IDE + browser open) | 0.3-0.9 GB of 7.6 GB |
| Disk sequential write with fsync, 256 MB / 64 MB | 248 MB/s / 180 MB/s |
| libx264 veryfast crf20, 1080p30, 120 frames | 101 fps and 79 fps on two runs; peak RSS 430 MB |
| h264_qsv 1080p30, 120 frames | 73.4 fps; peak RSS 233 MB |
| Hardware encoders that really encode | h264_qsv, hevc_qsv |
| `storyreel doctor` wall time (with hw test encodes) | 3.6 s |
| `storyreel profile` wall time | 4.7 s |

### Bugs found and fixed in this step
- Windows cp1252 console crashed on `≈` in profile output → stdout/stderr now use
  `errors="replace"`, output kept ASCII.
- Encoder parser accepted the legend line ` V..... = Video` as an encoder named `=`.
- "8 GB" laptops report ~7.6 GB, so `>= 8 GB` thresholds said "no" to 4K and local LLM →
  thresholds use an 8 GB-class constant (7 GB).
- Profile recommended 2 heavy workers with 0.9 GB free → default is now 1 (spec), with a
  separate ceiling the run-time scheduler may scale to.
- QSV was recommended without a speed test → now benchmarked; chosen only if ≥ libx264.

### Known gaps / honest notes
- Encode benchmark is 4 s of video: run-to-run noise of ~25 % on a busy machine.
  Fine for defaults, not for ETAs; Phase 1 will refine ETAs from real chunk timings.
- winget installed ffmpeg but could not create the `ffmpeg` alias symlinks (Windows
  Developer Mode off). It did add the bin dir to the user PATH, so **new** terminals find it.
- Local AI video/image generation is not possible on this machine (no CUDA GPU).

### Next
Phase 1 after architecture review: project DB, asset import + proxies/peaks/thumbs,
timeline model, chunked renderer, checkpoint/resume, watchdog, Job Objects, `storyreel render`,
10-min synthetic benchmark with `kill -9` + resume.
