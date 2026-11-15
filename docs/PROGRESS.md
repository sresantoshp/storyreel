# Progress

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
