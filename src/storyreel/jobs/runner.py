"""Run one ffmpeg job under a memory cap with heartbeat, timeout, and watchdog kill.

ffmpeg is started with `-progress pipe:1`, so it prints a key=value block about twice a
second. No output for `heartbeat_s` = hung → kill. stderr is drained continuously into a
bounded buffer (a full pipe would otherwise block ffmpeg forever).
"""

from __future__ import annotations

import collections
import contextlib
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psutil

from storyreel.errors import ExitCode, StoryReelError, ToolFailedError, ToolTimeoutError
from storyreel.jobs.limits import spawn
from storyreel.watchdog import MemState, Watchdog

_OOM_MARKERS = ("Cannot allocate memory", "Out of memory", "out of memory", "MemoryError")


class MemoryPressureError(StoryReelError):
    """Job stopped because memory ran out (watchdog CRITICAL or hard cap hit)."""

    exit_code = ExitCode.RESOURCE_EXHAUSTED


@dataclass(frozen=True)
class JobStats:
    wall_s: float
    peak_rss: int  # sampled working set of the child
    peak_commit: int | None  # Job Object peak (Windows only)
    stderr_tail: str = ""  # only when run with keep_stderr=True


def run_ffmpeg(
    argv: Sequence[str],
    *,
    limit_bytes: int | None,
    watchdog: Watchdog | None = None,
    heartbeat_s: float = 60.0,
    timeout_s: float = 6 * 3600,
    on_progress: Callable[[dict[str, str]], None] | None = None,
    keep_stderr: bool = False,
) -> JobStats:
    t0 = time.monotonic()
    try:
        p, limiter = spawn(
            argv,
            limit_bytes=limit_bytes,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise StoryReelError(f"Could not start {argv[0]}.") from exc
    assert p.stdout is not None and p.stderr is not None
    err_tail: collections.deque[str] = collections.deque(maxlen=200 if keep_stderr else 50)
    last_beat = [time.monotonic()]

    def read_progress() -> None:
        assert p.stdout is not None
        block: dict[str, str] = {}
        for raw in p.stdout:
            last_beat[0] = time.monotonic()
            key, _, value = raw.decode("utf-8", "replace").strip().partition("=")
            block[key] = value
            if key == "progress":
                if on_progress:
                    on_progress(block)
                block = {}

    def drain_stderr() -> None:
        assert p.stderr is not None
        for raw in p.stderr:
            err_tail.append(raw.decode("utf-8", "replace").rstrip())

    readers = [
        threading.Thread(target=read_progress, daemon=True),
        threading.Thread(target=drain_stderr, daemon=True),
    ]
    for t in readers:
        t.start()
    peak_rss = 0
    ps = psutil.Process(p.pid)
    reason: str | None = None
    try:
        while True:
            try:
                p.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            with contextlib.suppress(psutil.NoSuchProcess):
                peak_rss = max(peak_rss, ps.memory_info().rss)
            now = time.monotonic()
            if watchdog is not None and watchdog.state is MemState.CRITICAL:
                reason = "memory"
            elif now - last_beat[0] > heartbeat_s:
                reason = "hung"
            elif now - t0 > timeout_s:
                reason = "timeout"
            if reason:
                p.kill()
                p.wait()
                break
    finally:
        for t in readers:
            t.join(timeout=5)
        peak_commit = limiter.peak_commit_bytes()
        limiter.close()

    stderr = "\n".join(err_tail)
    stats = JobStats(time.monotonic() - t0, peak_rss, peak_commit, stderr if keep_stderr else "")
    name = argv[0].replace("\\", "/").rsplit("/", 1)[-1]
    if reason == "memory":
        raise MemoryPressureError("Stopped a render step because memory was nearly exhausted.")
    if reason == "hung":
        raise ToolTimeoutError(
            f"{name} stopped responding for {heartbeat_s:.0f} s and was stopped."
        )
    if reason == "timeout":
        raise ToolTimeoutError(f"{name} exceeded its {timeout_s:.0f} s time limit.")
    if p.returncode != 0:
        hit_cap = (
            limit_bytes is not None
            and peak_commit is not None
            and (peak_commit >= 0.97 * limit_bytes)
        )
        if hit_cap or any(m in stderr for m in _OOM_MARKERS):
            raise MemoryPressureError(
                f"A render step needed more than the "
                f"{(limit_bytes or 0) / 2**20:.0f} MB memory cap."
            )
        raise ToolFailedError(
            f"{name} failed (exit code {p.returncode})", returncode=p.returncode, stderr_tail=stderr
        )
    return stats
