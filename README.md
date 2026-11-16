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

Demo: `python examples/phase2_demo.py` (18 s animated scene, CLI only, offline).
Audio demo: `python examples/phase3_audio_demo.py` (60 s 4-track mix to -14 LUFS with stems).

Benchmarks: `python bench/bench_render.py --length 10min --kill-after 20`.

Requires ffmpeg >= 6 with ffprobe on PATH, or `STORYREEL_FFMPEG` / `STORYREEL_FFPROBE`.

Dev checks: `uv run pytest -q`, `uv run ruff check src tests`, `uv run mypy`.
