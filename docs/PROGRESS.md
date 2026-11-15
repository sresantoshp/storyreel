# Progress

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
