"""Grab single frames from video files through ffmpeg (small, RGB uint8)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

RGB = npt.NDArray[np.uint8]


def grab(ffmpeg: Path, path: Path, *, at_s: float | None = None, width: int = 320) -> RGB | None:
    """One frame `width` px wide: at `at_s`, or (None) a representative frame picked by
    ffmpeg's thumbnail filter from the first ~2 s. None if the file cannot be read."""
    seek = ["-ss", f"{at_s:.3f}"] if at_s is not None else []
    vf = f"scale={width}:-2" if at_s is not None else f"thumbnail=50,scale={width}:-2"
    argv = [str(ffmpeg), "-v", "error", *seek, "-i", str(path), "-frames:v", "1", "-vf", vf,
            "-f", "image2pipe", "-vcodec", "png", "-"]  # fmt: skip
    try:
        res = subprocess.run(argv, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    import cv2

    img = cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.uint8)
