from __future__ import annotations

import sys
import time

import pytest

from storyreel.errors import ToolFailedError, ToolTimeoutError
from storyreel.jobs.runner import MemoryPressureError, run_ffmpeg
from storyreel.watchdog import MemState, Sample, Watchdog, next_state

GB = 2**30
CAP = 1 * GB


def s(own: float, avail: float = 0.5, swap: float = 0.1) -> Sample:
    return Sample(round(own * CAP), CAP, int(avail * 8 * GB), 8 * GB, swap)


def test_state_machine_thresholds_and_hysteresis() -> None:
    N, P, C = MemState.NORMAL, MemState.PRESSURE, MemState.CRITICAL
    assert next_state(N, s(0.50)) is N
    assert next_state(N, s(0.801)) is P
    assert next_state(P, s(0.75)) is P  # hysteresis: stays until < 0.70
    assert next_state(P, s(0.69)) is N
    assert next_state(N, s(0.951)) is C
    assert next_state(C, s(0.90)) is C  # stays until < 0.85
    assert next_state(C, s(0.84)) is P
    assert next_state(N, s(0.2, avail=0.08)) is P  # system low → pressure
    assert next_state(P, s(0.2, avail=0.12)) is P  # needs > 15 % to clear
    assert next_state(N, s(0.2, avail=0.04, swap=0.95)) is C  # truly out of memory
    assert next_state(N, s(0.2, avail=0.04, swap=0.30)) is P  # low RAM but swap fine


def test_watchdog_tracks_peak_and_notifies() -> None:
    seq = iter([s(0.1), s(0.85), s(0.3), s(0.3)])
    wd = Watchdog(CAP, sampler=lambda: next(seq))
    changes: list[tuple[MemState, MemState]] = []
    wd.on_change(lambda a, b, _: changes.append((a, b)))
    for _ in range(4):
        wd.tick()
    assert wd.peak_tree_rss == int(0.85 * CAP)
    assert changes == [(MemState.NORMAL, MemState.PRESSURE), (MemState.PRESSURE, MemState.NORMAL)]


PY = sys.executable


def test_runner_kills_hung_process() -> None:
    t0 = time.monotonic()
    with pytest.raises(ToolTimeoutError, match="stopped responding"):
        run_ffmpeg([PY, "-c", "import time; time.sleep(30)"], limit_bytes=None, heartbeat_s=1.0)
    assert time.monotonic() - t0 < 5


def test_runner_reports_failure_with_stderr_tail() -> None:
    with pytest.raises(ToolFailedError) as exc:
        run_ffmpeg(
            [PY, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"], limit_bytes=None
        )
    assert exc.value.returncode == 3 and "boom" in exc.value.stderr_tail


def test_runner_kills_on_watchdog_critical() -> None:
    wd = Watchdog(CAP, sampler=lambda: s(0.99))
    wd.tick()
    assert wd.state is MemState.CRITICAL
    with pytest.raises(MemoryPressureError):
        run_ffmpeg([PY, "-c", "import time; time.sleep(30)"], limit_bytes=None, watchdog=wd)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_job_object_caps_child_memory() -> None:
    """Child tries to grow to 600 MB under a 150 MB job cap: it must fail, we must survive."""
    code = (
        "import sys\nb=[]\n"
        "try:\n    [b.append(bytearray(10*2**20)) for _ in range(60)]\n"
        "except MemoryError:\n    print('MemoryError', file=sys.stderr); sys.exit(1)\n"
    )
    with pytest.raises(MemoryPressureError):
        run_ffmpeg([PY, "-c", code], limit_bytes=150 * 2**20)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_job_object_allows_work_under_cap() -> None:
    code = "b = bytearray(50 * 2**20); print('progress=end')"
    stats = run_ffmpeg([PY, "-c", code], limit_bytes=300 * 2**20)
    assert stats.peak_commit is not None and 50 * 2**20 < stats.peak_commit < 300 * 2**20
