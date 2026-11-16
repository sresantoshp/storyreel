# Progress

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
