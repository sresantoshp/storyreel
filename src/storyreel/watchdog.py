"""Memory watchdog: samples our process tree's RSS and system memory, keeps a state.

States and thresholds (fractions of the cap; see ARCHITECTURE.md section 5):

    NORMAL   -> PRESSURE  own >= 0.80  or  system available < 10 %
    PRESSURE -> NORMAL    own <  0.70  and system available > 15 %
    any      -> CRITICAL  own >= 0.95  or  (system available < 5 % and page file > 90 % used)
    CRITICAL -> PRESSURE  own <  0.85  and not the system condition

Progress guarantee: PRESSURE only stops *additional* jobs. One job always runs, because on
the target laptop "available < 10 %" is the normal state with a browser open (measured
0.3-0.9 GB of 7.6 GB), and refusing to run at all would be a hang, not safety.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import psutil

log = logging.getLogger(__name__)


class MemState(IntEnum):
    NORMAL = 0
    PRESSURE = 1
    CRITICAL = 2


@dataclass(frozen=True)
class Thresholds:
    pressure_enter: float = 0.80
    pressure_exit: float = 0.70
    critical_enter: float = 0.95
    critical_exit: float = 0.85
    sys_low: float = 0.10
    sys_ok: float = 0.15
    sys_crit: float = 0.05
    swap_crit: float = 0.90


@dataclass(frozen=True)
class Sample:
    tree_rss: int
    cap: int
    sys_available: int
    sys_total: int
    swap_used_frac: float

    @property
    def own(self) -> float:
        return self.tree_rss / self.cap

    @property
    def avail(self) -> float:
        return self.sys_available / self.sys_total


def next_state(state: MemState, s: Sample, th: Thresholds = Thresholds()) -> MemState:
    sys_critical = s.avail < th.sys_crit and s.swap_used_frac > th.swap_crit
    if s.own >= th.critical_enter or sys_critical:
        return MemState.CRITICAL
    if state is MemState.CRITICAL:
        return MemState.CRITICAL if s.own >= th.critical_exit else MemState.PRESSURE
    if s.own >= th.pressure_enter or s.avail < th.sys_low:
        return MemState.PRESSURE
    if state is MemState.PRESSURE and (s.own >= th.pressure_exit or s.avail <= th.sys_ok):
        return MemState.PRESSURE
    return MemState.NORMAL


def tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def take_sample(pid: int, cap: int) -> Sample:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return Sample(tree_rss(pid), cap, vm.available, vm.total, swap.percent / 100.0)


class Watchdog:
    """Background thread. Read `state` and `peak_tree_rss` from any thread."""

    def __init__(
        self,
        cap_bytes: int,
        *,
        interval_s: float = 0.5,
        pid: int | None = None,
        sampler: Callable[[], Sample] | None = None,
        thresholds: Thresholds = Thresholds(),
    ) -> None:
        self.cap = cap_bytes
        self._pid = os.getpid() if pid is None else pid
        self._sampler = sampler or (lambda: take_sample(self._pid, self.cap))
        self._interval = interval_s
        self._th = thresholds
        self._state = MemState.NORMAL
        self._peak = 0
        self._last: Sample | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="storyreel-watchdog", daemon=True)
        self._listeners: list[Callable[[MemState, MemState, Sample], None]] = []

    @property
    def state(self) -> MemState:
        with self._lock:
            return self._state

    @property
    def peak_tree_rss(self) -> int:
        with self._lock:
            return self._peak

    @property
    def last(self) -> Sample | None:
        with self._lock:
            return self._last

    def on_change(self, fn: Callable[[MemState, MemState, Sample], None]) -> None:
        self._listeners.append(fn)

    def tick(self) -> MemState:
        s = self._sampler()
        with self._lock:
            old = self._state
            self._state = next_state(old, s, self._th)
            self._peak = max(self._peak, s.tree_rss)
            self._last = s
            new = self._state
        if new is not old:
            log.warning(
                "memory state change",
                extra={
                    "from": old.name,
                    "to": new.name,
                    "tree_rss": s.tree_rss,
                    "cap": s.cap,
                    "sys_available": s.sys_available,
                },
            )
            for fn in self._listeners:
                fn(old, new, s)
        return new

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.tick()

    def start(self) -> Watchdog:
        self.tick()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
