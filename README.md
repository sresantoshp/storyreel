# StoryReel

Text-to-video studio for weak laptops (4-8 GB RAM, no discrete GPU). Everything streams
from disk through ffmpeg; renders are chunked, cached and resumable.

Status: Phase 1 (engine core) done. See `docs/ARCHITECTURE.md` and `docs/PROGRESS.md`.

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
```

Benchmarks: `python bench/bench_render.py --length 10min --kill-after 20`.

Requires ffmpeg >= 6 with ffprobe on PATH, or `STORYREEL_FFMPEG` / `STORYREEL_FFPROBE`.

Dev checks: `uv run pytest -q`, `uv run ruff check src tests`, `uv run mypy`.
