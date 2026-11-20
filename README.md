# StoryReel

Text-to-video studio for weak laptops (4-8 GB RAM, no discrete GPU). Everything streams
from disk through ffmpeg; renders are chunked, cached and resumable.

Status: Phase 3 (audio engine: buses, ducking, effects, loudness, 5.1/7.1, stems) done. Free, open source, offline. See `docs/ARCHITECTURE.md` and `docs/PROGRESS.md`.

```
python -m pip install uv
python -m uv sync
.venv/Scripts/storyreel doctor     # Windows; .venv/bin/storyreel on Linux/macOS
.venv/Scripts/storyreel profile

.venv/Scripts/storyreel new myfilm.storyreel --size 1920x1080 --fps 30
.venv/Scripts/storyreel import myfilm.storyreel clip1.mp4 photo.jpg music.wav
.venv/Scripts/storyreel append myfilm.storyreel 1 --in 2 --dur 6     # asset #1, 6 s from 0:02
.venv/Scripts/storyreel append myfilm.storyreel --color "#203040" --dur 2
.venv/Scripts/storyreel render myfilm.storyreel -o film.mp4         # re-run to resume

# animation, text, environments (see `storyreel <cmd> -h`)
.venv/Scripts/storyreel env myfilm.storyreel rain --at 0 --dur 6 count=1200
.venv/Scripts/storyreel text myfilm.storyreel "एक रात की कहानी" --anim typewriter --size 80 outline=3
.venv/Scripts/storyreel kenburns myfilm.storyreel 1 --zoom-to 1.3 --pan-x -60
.venv/Scripts/storyreel fx myfilm.storyreel 1 --preset night
.venv/Scripts/storyreel transition myfilm.storyreel 1 2 dissolve --dur 1
.venv/Scripts/storyreel frame myfilm.storyreel --at 3.5 -o preview.png
.venv/Scripts/storyreel info myfilm.storyreel --clips
```

## Story to video: two styles

```
# animation: drawn scenery, animated characters, procedural ambience and music (fully offline)
.venv/Scripts/storyreel make --idea "a fox finds a lost cub" --length 60s --out fox.mp4

# realistic: real footage cut like a film, film grade, 2.39:1 letterbox, trailer sound design
.venv/Scripts/storyreel make --idea "a fox finds a lost cub" --style realistic --out fox_real.mp4
```

The two styles never mix: a realistic video uses only real footage and stops with a fix if a
scene has none. Footage sources, all free with commercial use allowed (checked in this order):

| source | setup | licence |
|---|---|---|
| your clips | `--footage-dir D:\clips` (name files by what they show: `forest_fox.mp4`) | yours |
| Pexels | free key: `setx PEXELS_API_KEY <key>` (pexels.com/api), new terminal | no credit needed |
| Pixabay | free key: `setx PIXABAY_API_KEY <key>` (pixabay.com/api/docs) | no credit needed |
| Wikimedia Commons | opt-in only: `--footage commons` + `setx STORYREEL_CONTACT <email>` | public domain / CC0 / CC BY; credits in `footage_credits.txt` |

Searches and downloads are cached (`%LOCALAPPDATA%\StoryReel\stock`), so a re-make works
offline (`--offline`). Pexels and Pixabay are curated and look best. Commons is not curated
(war, ritual and medical footage turn up for innocent searches), so it is never used
automatically; check every shot if you turn it on.

Footage is chosen by CLIP (a small open model, 146 MB, downloaded once): every candidate's
preview is scored against the narration, so "playful monkeys" gets monkeys playing, not a
monkey portrait. Drawn/CGI clips are never used. `--no-match` skips this (faster).

Sound in the realistic style: nature sound chosen from what each clip shows, music that changes
with the story's moods and builds to the climax (both crossfaded), a broadcast voice chain (EQ, compression, de-esser), a braam under the title, whooshes on cuts, a riser into the final
scene and an impact on it, hall reverb on effects, master glue compression, −14 LUFS.

Demo: `python examples/phase2_demo.py` (18 s animated scene, CLI only, offline).
Audio demo: `python examples/phase3_audio_demo.py` (60 s 4-track mix to -14 LUFS with stems).

Benchmarks: `python bench/bench_render.py --length 10min --kill-after 20`.

Requires ffmpeg >= 6 with ffprobe on PATH, or `STORYREEL_FFMPEG` / `STORYREEL_FFPROBE`.

Dev checks: `uv run pytest -q`, `uv run ruff check src tests`, `uv run mypy`.
