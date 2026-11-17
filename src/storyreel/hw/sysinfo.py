"""Raw facts about the machine: CPU, RAM, GPUs, disk speed, encoder speed.

Each probe has a timeout and degrades to "unknown" instead of failing the whole profile.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from storyreel.errors import StoryReelError
from storyreel.proc import run

log = logging.getLogger(__name__)

MB = 2**20
GB = 2**30


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vendor: str  # nvidia | amd | intel | apple | other
    vram_bytes: int | None  # None = unknown or shared system memory
    discrete: bool


@dataclass(frozen=True)
class DiskBench:
    path: str
    free_bytes: int
    write_mb_s: float | None
    bytes_written: int


@dataclass(frozen=True)
class EncodeBench:
    encoder: str
    width: int
    height: int
    frames: int
    wall_s: float
    fps: float
    peak_rss_bytes: int  # ffmpeg process, sampled every 50 ms


def _vendor(name: str) -> str:
    n = name.lower()
    for key, vendor in (
        ("nvidia", "nvidia"),
        ("geforce", "nvidia"),
        ("quadro", "nvidia"),
        ("radeon", "amd"),
        ("amd", "amd"),
        ("intel", "intel"),
        ("apple", "apple"),
    ):
        if key in n:
            return vendor
    return "other"


def _is_discrete(name: str, vendor: str) -> bool:
    n = name.lower()
    if vendor == "nvidia":
        return True
    if vendor == "intel":
        return "arc" in n and "graphics" not in n.replace("arc graphics", "")
    if vendor == "amd":
        # "Radeon Graphics" / "Radeon(TM) Graphics" are APU iGPUs; "RX 6600" etc. are discrete.
        return " rx " in f" {n} " or "pro w" in n
    return False


def cpu_model() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        if sys.platform == "darwin":
            return run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout_s=5).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, StoryReelError) as exc:
        log.warning("cpu model probe failed", extra={"error": str(exc)})
    return platform.processor() or "unknown"


def _nvidia_smi() -> list[GpuInfo]:
    try:
        out = run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            timeout_s=10,
        ).stdout
    except StoryReelError:
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, _, mem = line.rpartition(",")
        try:
            vram = int(float(mem.strip()) * MB)
        except ValueError:
            vram = None
        gpus.append(GpuInfo(name.strip(), "nvidia", vram, True))
    return gpus


def gpus() -> list[GpuInfo]:
    found: list[GpuInfo] = []
    try:
        if sys.platform == "win32":
            out = run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
                ],
                timeout_s=20,
            ).stdout
            data = json.loads(out) if out.strip() else []
            for item in data if isinstance(data, list) else [data]:
                name = str(item.get("Name") or "unknown")
                vendor = _vendor(name)
                # AdapterRAM is a uint32: it caps at 4 GB and reports shared memory for iGPUs.
                raw = item.get("AdapterRAM")
                vram = int(raw) if isinstance(raw, int) and raw > 0 else None
                discrete = _is_discrete(name, vendor)
                found.append(GpuInfo(name, vendor, vram if discrete else None, discrete))
        elif sys.platform == "darwin":
            out = run(["system_profiler", "SPDisplaysDataType", "-json"], timeout_s=20).stdout
            for item in json.loads(out).get("SPDisplaysDataType", []):
                name = str(item.get("sppci_model", "unknown"))
                found.append(GpuInfo(name, _vendor(name), None, False))
        else:
            out = run(["lspci", "-mm"], timeout_s=10, check=False).stdout
            for line in out.splitlines():
                if '"VGA' in line or '"3D controller"' in line or '"Display' in line:
                    name = line.split('"')[5] + " " + line.split('"')[7]
                    vendor = _vendor(name)
                    found.append(GpuInfo(name, vendor, None, _is_discrete(name, vendor)))
    except (StoryReelError, ValueError, IndexError, json.JSONDecodeError) as exc:
        log.warning("gpu probe failed", extra={"error": str(exc)})
    # nvidia-smi gives exact VRAM; prefer it over WMI's capped value.
    smi = _nvidia_smi()
    if smi:
        found = [g for g in found if g.vendor != "nvidia"] + smi
    return found


def disk_bench(directory: Path, *, total_bytes: int = 256 * MB) -> DiskBench:
    """Sequential write with fsync. Reads are skipped: the OS page cache makes them meaningless."""
    directory.mkdir(parents=True, exist_ok=True)
    free = psutil.disk_usage(str(directory)).free
    if free < total_bytes * 4:
        return DiskBench(str(directory), free, None, 0)
    block = os.urandom(4 * MB)
    target = directory / f".storyreel_diskbench_{os.getpid()}.tmp"
    written = 0
    try:
        t0 = time.perf_counter()
        with target.open("wb", buffering=0) as fh:
            while written < total_bytes:
                written += fh.write(block)
            os.fsync(fh.fileno())
        dt = time.perf_counter() - t0
    finally:
        target.unlink(missing_ok=True)
    return DiskBench(str(directory), free, written / MB / dt, written)


def run_with_peak(argv: list[str], timeout_s: float) -> tuple[int, str, float, int, bool]:
    """Run a child to completion; return (exit code, stderr, wall s, peak RSS bytes, timed out).

    The peak is exact, not sampled: POSIX takes ru_maxrss from wait4() on exit; Windows
    reads the process's own peak working-set counter, which keeps the maximum even if the
    process lives only a few milliseconds (sampling missed it on a fast CI machine).
    """
    t0 = time.perf_counter()
    with subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as p:
        killed = threading.Event()

        def _kill() -> None:
            killed.set()
            p.kill()

        timer = threading.Timer(timeout_s, _kill)
        timer.start()
        peak = 0
        try:
            if sys.platform == "win32":
                ps = psutil.Process(p.pid)
                while p.poll() is None:
                    with contextlib.suppress(psutil.Error):
                        peak = max(peak, int(ps.memory_info().peak_wset))
                    time.sleep(0.01)
            else:
                _, status, usage = os.wait4(p.pid, 0)
                p.returncode = os.waitstatus_to_exitcode(status)
                peak = int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
        finally:
            timer.cancel()
        wall = time.perf_counter() - t0
        stderr = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
    return int(p.returncode or 0), stderr, wall, peak, killed.is_set()


def encode_bench(
    ffmpeg: Path,
    *,
    encoder: str = "libx264",
    width: int = 1920,
    height: int = 1080,
    seconds: float = 4.0,
    fps: int = 30,
    timeout_s: float = 120.0,
) -> EncodeBench:
    """Encode synthetic 1080p video to /dev/null and sample ffmpeg's RSS.

    The source (testsrc2) is generated by the same ffmpeg process, so the fps number is a
    slight under-estimate of pure encode speed.
    """
    frames = int(seconds * fps)
    argv = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s={width}x{height}:r={fps}:d={seconds}",
        "-pix_fmt",
        "nv12" if encoder.endswith("_qsv") else "yuv420p",
        "-c:v",
        encoder,
    ]
    if encoder == "libx264":
        argv += ["-preset", "veryfast", "-crf", "20"]
    argv += ["-f", "null", "-"]

    rc, stderr, wall, peak, timed_out = run_with_peak(argv, timeout_s)
    if timed_out:
        raise StoryReelError(f"Encoder benchmark ({encoder}) took over {timeout_s:.0f} s")
    if rc != 0:
        raise StoryReelError(f"Encoder benchmark ({encoder}) failed: {stderr.strip()[-300:]}")
    return EncodeBench(encoder, width, height, frames, wall, frames / wall, peak)
