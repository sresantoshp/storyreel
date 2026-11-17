"""Hard memory limits for child processes.

Windows: a Job Object per child with JOB_OBJECT_LIMIT_JOB_MEMORY (caps committed private
memory for the child and anything it spawns) + KILL_ON_JOB_CLOSE (no orphans). Allocations
past the cap fail inside the child; the OS never touches our process.
Linux: RLIMIT_AS = 3 x cap in the child before exec (address-space limit; imprecise, see
ARCHITECTURE.md section 5). macOS: no hard limit; the watchdog is the only protection.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any


class Limiter:
    """No-op limiter (macOS, or no cap requested)."""

    def peak_commit_bytes(self) -> int | None:
        return None

    def close(self) -> None:
        return None


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IO(ctypes.Structure):
        _fields_ = [
            (n, ctypes.c_ulonglong)
            for n in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JobObjectExtendedLimitInformation = 9
    _LIMIT_JOB_MEMORY = 0x00000200
    _LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _check(ok: object, what: str) -> None:
        if not ok:
            raise OSError(ctypes.get_last_error(), f"{what} failed")

    class JobObjectLimiter(Limiter):
        def __init__(self, pid: int, limit_bytes: int) -> None:
            self._job = _k32.CreateJobObjectW(None, None)
            _check(self._job, "CreateJobObjectW")
            info = _Extended()
            info.BasicLimitInformation.LimitFlags = _LIMIT_JOB_MEMORY | _LIMIT_KILL_ON_JOB_CLOSE
            info.JobMemoryLimit = limit_bytes
            _check(
                _k32.SetInformationJobObject(
                    self._job,
                    _JobObjectExtendedLimitInformation,
                    ctypes.byref(info),
                    ctypes.sizeof(info),
                ),
                "SetInformationJobObject",
            )
            proc = _k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            _check(proc, "OpenProcess")
            try:
                _check(_k32.AssignProcessToJobObject(self._job, proc), "AssignProcessToJobObject")
            finally:
                _k32.CloseHandle(proc)

        def peak_commit_bytes(self) -> int | None:
            if not self._job:
                return None
            info = _Extended()
            ok = _k32.QueryInformationJobObject(
                self._job,
                _JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
                None,
            )
            return int(info.PeakJobMemoryUsed) if ok else None

        def close(self) -> None:
            if self._job:
                _k32.CloseHandle(self._job)  # KILL_ON_JOB_CLOSE ends any survivors
                self._job = None


def _preexec(limit_bytes: int | None) -> Callable[[], None] | None:
    if sys.platform == "win32" or sys.platform == "darwin" or limit_bytes is None:
        return None
    import resource

    def apply() -> None:
        lim = 3 * limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

    return apply


def spawn(
    argv: Sequence[str], *, limit_bytes: int | None, **popen_kwargs: Any
) -> tuple[subprocess.Popen[bytes], Limiter]:
    """Start a child with a hard memory cap where the OS supports one."""
    capped = sys.platform == "win32" and limit_bytes is not None
    if capped:
        # Start suspended, join the job, then resume: a child that finishes instantly
        # cannot exit before it is capped (assigning an exited process fails with
        # ERROR_ACCESS_DENIED, measured on a loaded machine with a fast ffmpeg run).
        popen_kwargs["creationflags"] = popen_kwargs.get("creationflags", 0) | _CREATE_SUSPENDED
    p: subprocess.Popen[bytes] = subprocess.Popen(
        list(argv), preexec_fn=_preexec(limit_bytes), **popen_kwargs
    )
    limiter: Limiter = Limiter()
    if sys.platform == "win32" and limit_bytes is not None:
        try:
            limiter = JobObjectLimiter(p.pid, limit_bytes)
        except OSError:
            p.kill()
            p.wait()
            raise
        finally:
            _resume(p.pid)
    return p, limiter


_CREATE_SUSPENDED = 0x00000004
_PROCESS_SUSPEND_RESUME = 0x0800


def _resume(pid: int) -> None:
    """Resume every thread of a process started with CREATE_SUSPENDED (Windows only)."""
    import ctypes

    win_dll = getattr(ctypes, "WinDLL")  # noqa: B009 (absent on POSIX type stubs)
    k32 = win_dll("kernel32", use_last_error=True)
    ntdll = win_dll("ntdll")
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(_PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        raise OSError(getattr(ctypes, "get_last_error")(), "OpenProcess (resume) failed")  # noqa: B009
    try:
        ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
        status = ntdll.NtResumeProcess(ctypes.c_void_p(h))
        if status != 0:
            raise OSError(status, "NtResumeProcess failed")
    finally:
        k32.CloseHandle(ctypes.c_void_p(h))
