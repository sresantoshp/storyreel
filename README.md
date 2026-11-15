# StoryReel

Text-to-video studio for weak laptops (4-8 GB RAM, no discrete GPU). Everything streams
from disk through ffmpeg; renders are chunked, cached and resumable.

Status: architecture + environment tooling. See `docs/ARCHITECTURE.md` and `docs/PROGRESS.md`.

```
python -m pip install uv
python -m uv sync
.venv/Scripts/storyreel doctor     # Windows; .venv/bin/storyreel on Linux/macOS
.venv/Scripts/storyreel profile
```

Requires ffmpeg >= 6 with ffprobe on PATH, or `STORYREEL_FFMPEG` / `STORYREEL_FFPROBE`.

Dev checks: `uv run pytest -q`, `uv run ruff check src tests`, `uv run mypy`.
