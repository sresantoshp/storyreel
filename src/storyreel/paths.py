"""Per-user directories (config, data, logs). Created on demand."""

from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

_DIRS = PlatformDirs(appname="StoryReel", appauthor=False, ensure_exists=False)


def data_dir() -> Path:
    return Path(_DIRS.user_data_dir)


def log_dir() -> Path:
    return Path(_DIRS.user_log_dir)


def bin_dir() -> Path:
    """Where a bundled/downloaded ffmpeg is looked for first."""
    return data_dir() / "bin"


def profile_path() -> Path:
    return data_dir() / "hardware_profile.json"
