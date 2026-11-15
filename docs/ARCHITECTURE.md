# StoryReel Architecture

Status: draft for review (2026-11-15). Nothing below Phase 1 is implemented yet.
Numbers marked **measured** were taken on the dev machine (Windows 11, Intel UHD iGPU,
Python 3.13). Numbers marked **estimate** must be replaced by measurements in the phase
that builds the component. If a measurement contradicts this document, the document changes.

---

## 0. Decisions in one screen

| Decision | Choice | Why (numbers) |
|---|---|---|
| UI toolkit | **PySide6 (Qt Widgets), not a web UI** | Measured idle + 2000-rect canvas redraw at 60 Hz: PySide6 = **56 MB working set / 28 MB private, 1 process**. Edge `--app` window, same drawing in `<canvas>` = **18 processes, 1023 MB working set / 525 MB private**. A web UI alone blows the 400 MB idle budget. Spec said "default (a) unless (b) is lighter"; (b) is ~19x lighter. |
| Engine / UI split | Separate processes. UI talks to engine over HTTP + WebSocket on 127.0.0.1 (FastAPI) | UI crash never kills a render; render OOM never kills the UI; CLI, UI and tests drive the same API. Cost: FastAPI + uvicorn ~40-60 MB (**estimate**, measure Phase 1). |
| Time unit | int64 **flicks** (1/705,600,000 s) | Every common frame rate (23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120) and sample rate (44.1k, 48k, 96k) is an exact integer number of flicks. No float drift over 2 h. int64 range = 415 years. |
| Video path | ffmpeg subprocess pipes (rawvideo), never whole-clip objects | Decoded 1080p RGB24 frame = 6.2 MB. 1 hour at 30 fps = 671 GB raw. Nothing else is possible. |
| Render unit | **Chunk** = ≤ 10 s of timeline, cut at edit points | 2 h = ~720 chunks. Crash loses ≤ 1 chunk. Per-chunk ffmpeg start-up ~0.1-0.3 s (**estimate**) = 1-3 % overhead at 10 s. |
| Audio path | Render audio per chunk to PCM, concat sample-exact, encode once at the end | AAC chunks concatenated with `-c copy` click/gap at every boundary (encoder priming ~1024 samples). PCM concat is exact. 2 h 48 kHz stereo float = 2.76 GB on disk, 0 in RAM. |
| Project store | SQLite, WAL, `synchronous=FULL`, one writer (engine process) | Each edit is its own durable transaction (fsync ~1-10 ms on SSD). Edit loss on power cut ≈ 0 s, beats the 5 s target. |
| Hashing | BLAKE2b (stdlib) | No extra dependency, ~1 GB/s/core. |
| Packaging | `uv`, Python ≥ 3.11 (dev box runs 3.13) | |
| Hard memory limit | Windows Job Object (job-wide commit limit), Linux cgroup v2 via `systemd-run --user --scope` if available else `RLIMIT_AS`, macOS watchdog only | See section 5; `RLIMIT_AS` and macOS limits are weak, said honestly below. |

---

## 1. Process model

```
 ┌────────────────────────────────────┐
 │ UI process  (PySide6, ~60-150 MB)  │  never blocks: all I/O via QNetworkAccessManager /
 │  home, wizard, timeline, inspector │  QWebSocket on the Qt event loop. Preview frames
 │  preview widget, job panel         │  arrive as JPEG bytes, decoded at display size.
 └──────────────┬─────────────────────┘
                │ HTTP + WebSocket, 127.0.0.1:<random port>
                │ Bearer token from a 0600 file in the user data dir; Host header checked
                ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Engine process  `storyreel serve`  (asyncio, target < 150 MB)            │
 │                                                                          │
 │  FastAPI routes ─► ProjectService ─► SQLite project.db  (ONLY writer)    │
 │                     undo/redo, autosave, versions                        │
 │  Scheduler: priority queue, admission control, retries, resume           │
 │  Watchdog thread: 1 Hz, sums RSS of the whole process tree               │
 │  Preview service: scrub requests coalesced (latest wins), LRU 256 MB     │
 └──────┬──────────────────────────────────────┬────────────────────────────┘
        │ spawn `python -m storyreel.worker`    │ stdin: one JSON job spec
        │ inside a Job Object / cgroup          │ stdout: JSON-lines progress/heartbeat
        ▼                                       ▼
 ┌─────────────────────────────┐   ┌──────────────────────────────────────┐
 │ Render worker (1 chunk/job) │   │ Other workers: proxy, waveform,      │
 │  ffmpeg decode ×L ──pipe──┐ │   │ thumbnails, TTS (Piper), ASR         │
 │  numpy compositor ◄───────┘ │   │ (faster-whisper), cloud gen clients  │
 │  ffmpeg encode ◄──pipe──    │   └──────────────────────────────────────┘
 └──────────┬──────────────────┘
            ▼ files only (atomic rename)
   <project>/cache/chunks/<key>.mkv   <project>/proxies/…   <project>/assets/…
```

Rules:
- **Workers never touch `project.db`.** They read a job spec, write files, report on stdout.
  The scheduler records results. This removes SQLITE_BUSY from the design entirely.
- Workers are fresh processes (Windows has no `fork`; spawn + `import numpy` ≈ 0.3 s
  **estimate**). One worker runs one job, then exits, which returns all memory to the OS.
  Fragmentation leaks cannot accumulate over an 8-hour soak.
- `storyreel render project/` and `storyreel make …` run the same scheduler **in-process**
  (no HTTP). Same code path as the UI, so the CLI is the test harness.
- Heartbeat: workers print a progress line at least every 2 s. No line for 30 s = hung;
  the scheduler kills the job tree and retries.
- Children die with their parent: Windows `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; Linux
  `PR_SET_PDEATHSIG` + process group kill; macOS process group kill on scheduler exit.

Default concurrency: **1 heavy job** (render, generation, ASR, TTS). Light jobs (thumbnails,
waveform peaks) run 1 extra. Max heavy workers = `min(physical_cores // 2, floor((available_RAM - 1.5 GB) / per_job_budget))`, never above the user setting.

---

## 2. Project data model

Project folder (portable, zippable):

```
MyFilm.storyreel/
  project.db        SQLite, WAL
  project.lock      PID + hostname + start time; stale if PID dead
  assets/           imported originals are referenced by path (never moved, never deleted);
                    generated assets (AI, TTS, stock downloads) are COPIED here: they cost money
  proxies/          540p/360p H.264 all-intra-ish (GOP 15) proxies, waveform peaks, thumb sprites
  cache/chunks/     rendered chunk intermediates, content-addressed
  cache/preview/    preview segments
  renders/          final outputs
  versions/         snapshots: `VACUUM INTO versions/<ts>.db` (MBs, assets not copied)
  logs/             JSON-lines
```

Schema (all times int64 flicks; JSON columns are small and validated by pydantic):

```
project(id, name, fps_num, fps_den, width, height, sample_rate, channel_layout,
        created_at, schema_version, settings_json)
asset(id, kind[video|image|audio|font|lut], path, is_generated, fingerprint,
      duration, width, height, fps_num, fps_den, codec, sample_rate, channels,
      license, source_url, attribution, tags, status[ok|missing|corrupt], error)
asset_fts       FTS5 over (name, tags, license, source_url, description)
proxy(asset_id, kind[video540|video360|peaks|thumbs], path, params_hash, status)
sequence(id, name, is_main)                   nested sequences = compound clips
track(id, sequence_id, kind[video|audio|text|subtitle|adjust], idx, name,
      muted, locked, solo, bus[dialogue|music|fx|ambience|master])
clip(id, track_id, kind[media|text|shape|color|sequence|adjust|generator],
     asset_id NULL, ref_sequence_id NULL, start, duration, src_in,
     speed_num, speed_den, reverse, props_json)
effect(id, clip_id, plugin_id, plugin_version, idx, enabled, params_json)
keyframe(id, owner_kind[clip|effect], owner_id, param, t, value_json,
         interp[hold|linear|bezier], bez_out_x, bez_out_y, bez_in_x, bez_in_y)
transition(id, track_id, clip_a, clip_b, kind, duration, params_json)
marker(id, sequence_id, t, duration, label, color, kind[marker|chapter])
scene(id, idx, title, summary, location_id, bible_json)       story layer
shot(id, scene_id, idx, spec_json, status[fresh|stale|generating|failed], clip_id NULL)
bible_entry(id, kind[character|location|style], name, spec_json, ref_asset_id)
command(id, seq, label, forward_json, inverse_json, created_at)  undo log, ring of 50+
job(id, kind, priority, state, spec_json, attempts, last_error, progress,
    created_at, started_at, finished_at, peak_rss)
cache_entry(key, kind, path, bytes, created_at, last_used_at, pinned)
spend(id, provider, op, est_cost, actual_cost, currency, at)    budget cap enforcement
```

Undo/redo: every edit is a `Command` with `forward` and `inverse` patches (row-level
inserts/updates/deletes). Undo = apply `inverse` in one transaction. Target **< 5 ms**
per step (spec: < 50 ms). Command log is persisted, so undo survives a restart.

Durability: `PRAGMA journal_mode=WAL; synchronous=FULL; foreign_keys=ON`.
Drags commit on mouse-up and every 1 s during long drags. Spec asks for "autosave every
5 s"; per-transaction commit is strictly better, so the 5 s autosave becomes a no-op
safety net for UI-side unsent state.

---

## 3. Render graph: how one frame is computed

```
timeline (sequence)                 per chunk                          output
──────────────────────   split at   ────────────────────────────────   ──────────
tracks/clips/effects/  ─► cuts and ─► ChunkPlan (pure data, hashable) ─► chunk.mkv  ┐
keyframes/transitions     ≤ 10 s       ├─ path A: ffmpeg filtergraph               ├► concat
                                       │   (all ops expressible in ffmpeg)         │  demuxer
                                       └─ path B: numpy compositor                 │  -c copy
                                           (text, masks, custom effects, plugins)  ┘  + audio mux
```

**Chunking.** Boundaries are placed at edit points first, then every ≤ 10 s inside long
clips. A transition never straddles a boundary (boundary moves to the transition's edge).

**Path A (fast).** If every layer in a chunk maps to ffmpeg filters (scale, crop, overlay,
fade, xfade, eq, lut3d, zoompan, setpts, …), the chunk is one ffmpeg process with one
filtergraph. No frames cross into Python. Fastest and lowest memory.

**Path B (general).** For each active layer, one ffmpeg decoder outputs `rawvideo rgba` at
the *target* resolution (ffmpeg scales, not numpy) into a pipe. The compositor pulls one
frame per layer, evaluates keyframes at time t, applies effects, blends bottom-to-top into a
float32 accumulator, converts to uint8, writes to the encoder's stdin.

Memory per chunk is O(layers), independent of duration:

```
1080p RGBA uint8 frame          8.3 MB
float32 accumulator            33.2 MB  (1 per compositor)
ring buffer per decoder        N = 3 frames -> 24.9 MB/layer   (N drops to 2 under pressure)
pipe buffers                   ~1 MB/pipe (OS), not counted in RSS
x264 veryfast 1080p encoder    430 MB   (MEASURED, ffmpeg process peak RSS incl. testsrc2 source)
h264_qsv 1080p encoder         233 MB   (MEASURED, Intel UHD iGPU)
ffmpeg decoder (H.264 1080p)   ~40-80 MB each (estimate; measure Phase 1)
```
Worked example, 6 active layers, libx264: 6 × (24.9 + 60) + 33 + 430 + Python/numpy ~80
≈ **1.05 GB**. A 2-hour film renders with the same peak as a 10-second one because
chunks are sequential and each worker exits after its chunk. Budget 2.5 GB leaves room for
~20 simultaneous 1080p layers (the profiler computes the exact number per machine); the planner refuses to exceed the budget and instead
pre-renders nested sequences to their own chunk files (spill to disk, then composite).

**Encoding for lossless concat.** All chunks of one render use identical encoder settings,
pixel format, time base, and start with an IDR frame; intermediates are MKV. The concat
demuxer then joins them with `-c copy`. Risk: some hardware encoders emit different
SPS/PPS between sessions; Phase 1 tests this per encoder and falls back to MPEG-TS
intermediates, then to a final re-encode, and logs which path was used.

Rate control: CRF/CQP per chunk gives uniform quality. Target-bitrate and two-pass modes run
per chunk too, so bitrate varies a little at boundaries. Trade-off accepted; users who need
a strict file size get a "final single-pass re-encode" option (slower, still streaming).

**Audio.** Per chunk: ffmpeg mixes bus graphs (`amix`, `volume` envelopes, `sidechaincompress`,
EQ, reverb) into 32-bit float PCM. Chunks are concatenated as PCM (sample-exact), loudness is
measured once over the full file (`ebur128`/`loudnorm` pass 1, streaming, O(1) RAM), then a
final encode applies the gain and true-peak limit. Dolby Atmos: not possible without a Dolby
license; an `Exporter` plugin hook receives the stems and bed/object metadata for a licensed
encoder later. We ship stereo, 5.1, 7.1 (AAC, AC-3, E-AC-3, Opus, FLAC, PCM).

**Preview.** Same graph at proxy resolution reading proxy files. Scrub = single-frame
request; the engine keeps only the newest pending request per client (older ones dropped),
returns a JPEG (~30 KB at 540p). Playback = the engine renders 2-second preview segments
ahead of the playhead with the cheapest encoder (`-preset ultrafast`), the Qt widget plays
them. Quality ladder Full → Half → Quarter → Proxy, auto-downgrade under pressure or when
the segment renderer falls behind real time.

---

## 4. Cache keys and invalidation

Key = `blake2b(canonical_json(inputs), digest_size=20)` where inputs are:

| Cached thing | Key inputs |
|---|---|
| Asset fingerprint | file size, mtime_ns, BLAKE2b of first 4 MB + middle 4 MB + last 4 MB. Full-file hash computed in background for dedup. |
| Proxy / peaks / thumbs | asset fingerprint, proxy kind, params, engine version |
| Render chunk | ChunkPlan with times **relative to chunk start**, every referenced asset fingerprint, effect plugin id + version, render settings (codec, crf, res, fps, pix_fmt), engine version |
| Audio chunk | same, for the audio graph |
| Generated asset (AI image/video/TTS/music) | provider, model, prompt (normalized), seed, references (fingerprints), params |
| Preview frame | ChunkPlan slice at t, quality level |

Invalidation rules:
1. Nothing is ever "invalidated" in place. An edit changes inputs → new key → miss → render.
   Old entries age out by LRU.
2. Relative times mean a ripple edit that only *shifts* later clips does not change their
   keys, so inserting a shot at minute 1 of a 2-hour film re-renders only the chunks that
   actually changed content. Chunk boundaries are anchored to edit points for the same reason.
3. Engine or plugin version bump changes keys: you get correct output after an upgrade.
4. Before trusting a hit: file exists, size matches `cache_entry.bytes`, `ffprobe` duration
   matches within 1 frame (~50 ms per chunk, **estimate**). Fail → delete, re-render.
5. Generated assets are **pinned** (they cost money) and live in `assets/`, not `cache/`.
   Chunk/preview/proxy caches are evictable; default disk cap = min(20 GB, 25 % of free space);
   RAM preview cache 256 MB (128 MB on machines with < 6 GB RAM).
6. Writes are atomic: `<key>.part` → fsync → `os.replace` → insert `cache_entry` row.
   A crash leaves at worst an orphan `.part` file, removed on startup.

---

## 5. Memory watchdog

Measured quantity: **sum of RSS (working set) over the engine process and all descendants**
(workers + ffmpeg), plus system available memory. 1 Hz thread in the engine, plus the
same check inside each worker every 1 s for its own tree.

```
                ≥ 80 % cap  or  sys avail < 10 %
     ┌────────┐ ─────────────────────────────────► ┌──────────┐
     │ NORMAL │                                    │ PRESSURE │
     └────────┘ ◄───────────────────────────────── └──────────┘
          ▲      < 70 % cap and sys avail > 15 %    │      ▲
          │                                         │      │ < 85 % cap and sys avail > 8 %
          │        ≥ 95 % cap  or  sys avail < 5 %  ▼      │
          │                                       ┌──────────┐
          └──── (via PRESSURE, after 10 s stable) │ CRITICAL │
                                                  └──────────┘
```

Default cap = **min(3 GB, 50 % of physical RAM)** (8 GB machine → 3 GB; 4 GB machine → 2 GB).

The cap is a ceiling, not a promise. The dev machine (7.6 GB) measured **0.3-0.9 GB available**
with an IDE and browser open, so in practice the "sys avail" triggers fire long before the cap.
That is the normal case on the target hardware, and the reason the scheduler sizes concurrency
from *available* RAM at run time, never from the profile.

| State | Actions |
|---|---|
| NORMAL | Admit jobs. Full preview quality allowed. |
| PRESSURE | Drop RAM caches to 25 %, `gc.collect()`, preview quality down one step, decoder ring buffers 3 → 2, scheduler admits **no new** jobs. |
| CRITICAL | Send `stop` to running heavy workers: worker finishes the current frame, deletes the partial chunk, exits 0 with `stopped_for_memory`. Job → `PAUSED_MEMORY`. Not exited in 5 s → kill tree. Resume when NORMAL for 10 s. Preview → Proxy. |

Hysteresis gaps (80/70, 95/85) stop flapping. Losing a chunk costs ≤ 10 s of output.

Hard limits, the backstop if the watchdog loses a race:
- **Windows**: each worker in a Job Object with `JOB_OBJECT_LIMIT_JOB_MEMORY` = cap. Limits
  *commit* (private bytes), which is ≥ working set for numpy, so it is a real ceiling. The OS
  kills only the job; the engine sees the exit code and retries with `ring=2, threads=2`.
- **Linux**: cgroup v2 `memory.max` through `systemd-run --user --scope` when available
  (real RSS limit). Fallback `RLIMIT_AS` = 3 × cap, because address-space limits count
  reserved-but-unused memory (OpenBLAS, glibc arenas) and a tight `RLIMIT_AS` causes
  false OOMs. Weak, but better than nothing.
- **macOS**: `RLIMIT_AS` is not enforced reliably. Watchdog only. Said plainly in `doctor`.
- The engine process itself has no hard limit (it cannot kill itself cleanly); its own
  caches are byte-capped and it never decodes frames.

---

## 6. Failure modes

| Failure | Detection | Recovery | User message |
|---|---|---|---|
| Worker OOM / killed | exit code ≠ 0 with Job Object / cgroup OOM signal, or watchdog CRITICAL | delete partial chunk; retry up to 3× with ring=2, threads=2, then 540p intermediate; then fail job | "Ran low on memory while rendering scene 12. Retrying with lower memory settings." Fix: close other apps / lower preview quality |
| ffmpeg crash / non-zero exit | exit code, stderr tail (last 50 lines kept, full to log) | retry once; if same error twice, classify (decode error → mark asset corrupt; encoder error → fall back to libx264) | "Couldn't decode `clip.mov` at 00:03:12. The file looks damaged." Fix: re-encode with one click (ffmpeg remux / transcode to proxy-style intermediate) |
| ffmpeg / worker hang | no heartbeat 30 s, or no output-file growth 60 s | kill tree, retry | "A render step stopped responding and was restarted." |
| API timeout / 5xx / 429 | per-call timeout (connect 10 s, read 120 s for generation), status code | exponential backoff 1, 2, 4, 8, 16 s + jitter, max 5; 429 honours `Retry-After`; then next backend in priority list (A → B → C) | "Image service is slow; using stock footage for shot 4. You can regenerate later." |
| Budget cap reached | `spend` sum + estimate > cap **before** the call | refuse the call, fall back to backend C | "This would cost $1.80 and your monthly cap leaves $0.40." |
| Disk full | free space checked before each job (need ≥ 2 × estimated output + 1 GB); `ENOSPC` on write | pause queue, keep `.part` files deleted, offer cache cleanup | "Disk almost full (1.2 GB left). Free 14 GB of render cache?" |
| Corrupt / odd media | ffprobe fails, 0 streams, absurd duration, decode error in proxy job | mark asset `corrupt`, skip, continue import | "Skipped `bad.mp4`: not a readable video. 41 other files imported." |
| Power loss / kill -9 | on start: `project.lock` with dead PID; jobs in RUNNING state | SQLite WAL recovers committed edits; `.part` files deleted; RUNNING jobs → QUEUED; render resumes from the first chunk with no valid cache entry | "StoryReel closed unexpectedly. Your project was restored (no edits lost). Resume render? 412/720 chunks done." |
| UI crash | engine sees WebSocket drop | engine keeps rendering; UI restart reconnects and re-reads state | — |
| Engine crash | UI loses socket | UI shows "Reconnecting", restarts engine; engine performs power-loss recovery | "The engine restarted. Nothing was lost." |
| Missing asset (moved file) | stat fails on open / render | mark `missing`, render a slate frame for preview, block final render | "3 files are missing. Locate folder?" (relink by fingerprint) |

All user messages come from a typed error → message table; stack traces go only to
`logs/*.jsonl`.

---

## 7. Story-to-video pipeline (Phase 4 outline)

```
text ─► StoryEngine (LLM provider, strict pydantic schema, repair + retry ≤ 3)
     ─► Outline ─► Scenes ─► Shots (duration, camera, visual, narration, text, sfx, music, mood, transition)
     ─► Bible (characters, locations, style) ─► per-shot prompt builder
     ─► Estimator (cost $, time s per backend) ─► user confirms if cost > 0
     ─► Jobs: visuals (A cloud → B local → C stock/procedural), TTS (cloud → Piper),
              music/SFX/ambience (cloud → library → procedural)
     ─► Auto-assembly: narration word timestamps drive shot cuts and subtitles
     ─► Timeline in project.db; every shot linked back to its spec (stale tracking)
```

Honest limits on the target laptop (8 GB, 4 cores, no discrete GPU):
- Local AI video (Wan, LTX, Hunyuan, CogVideoX): **not possible**. Needs 8-24 GB VRAM.
- Local image diffusion on CPU: possible but slow (**estimate** 20-90 s per 512 px image for
  1-4 step turbo models). Off by default; profiler decides.
- Offline story writing: no LLM offline by default. Offline mode uses a deterministic
  splitter (sentences → shots, keyword → stock/procedural visuals). Lower quality, always
  works. A local small LLM (llama.cpp, ~3B Q4, ~2.5 GB RAM) is an optional backend B.
- Piper TTS on CPU: faster than real time on 4 cores (**estimate**, measure Phase 4).
- Acceptance test "60 s video from text in < 10 min via backend C + Piper": feasible;
  stock search needs internet unless the user's local media library is used.

---

## 8. Security and safety

- API bound to 127.0.0.1 only, random port, bearer token in a user-only file, `Host` header
  check (blocks DNS-rebinding from web pages).
- API keys: OS keyring when available, else a user-only file; never logged; `doctor` prints
  presence only.
- AI disclosure metadata on by default; C2PA via `c2pa-python` if it works on the target
  platforms (self-signed certificates show as "untrusted issuer" in verifiers; real trust
  needs a CA-issued signing cert — the user's decision).
- Real-person likeness: generation prompts naming a real person are blocked unless the
  character sheet has `consent=true` and a consent note.

---

## 9. Open questions for you

1. UI: PySide6 per the measurement above. OK, or do you want the web UI despite ~500 MB?
2. Chunk length default 10 s: shorter = less loss on crash, more overhead. OK?
3. Default memory cap min(3 GB, 50 % RAM): on a 4 GB machine that is 2 GB, below the
   2.5 GB render target. Lower-RAM machines get fewer layers / 720p intermediates.
4. Intermediates: x264 CRF 16 `veryfast` for chunks (near-transparent, ~2× final size on
   disk) vs. the final codec directly. I chose the final codec directly when concat-copy
   works, so no generational loss and no second encode.
