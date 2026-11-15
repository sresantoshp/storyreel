"""Hardware profile: collect facts, derive safe defaults, state honestly what the machine can do.

`recommend()` and `capabilities()` are pure functions of the profile so they can be unit-tested
with fake machines (4 GB laptop, 32 GB workstation, ...).
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from storyreel import __version__
from storyreel.hw import sysinfo
from storyreel.hw.sysinfo import GB, MB, DiskBench, EncodeBench, GpuInfo
from storyreel.media import ffbin


@dataclass(frozen=True)
class HardwareProfile:
    created_at: str
    storyreel_version: str
    os: str
    python: str
    cpu_model: str
    cores_physical: int
    cores_logical: int
    ram_total: int
    ram_available: int
    gpus: list[GpuInfo]
    disk: DiskBench | None
    ffmpeg_path: str | None
    ffmpeg_version: str | None
    hw_encoders: list[str] = field(default_factory=list)
    encode_bench: EncodeBench | None = None  # libx264 veryfast, CPU
    hw_encode_bench: EncodeBench | None = None  # first working hardware H.264 encoder


@dataclass(frozen=True)
class Recommendations:
    memory_cap_bytes: int
    ram_cache_bytes: int
    proxy_height: int
    heavy_workers: int  # default concurrency (spec: 1)
    max_heavy_workers: int  # ceiling the scheduler may scale to when RAM is actually free
    preview_quality: str  # full | half | quarter | proxy
    video_encoder: str
    max_layers_1080p: int


@dataclass(frozen=True)
class Capability:
    name: str
    ok: bool
    detail: str


# Machines sold as "8 GB" report 7.4-7.9 GB (firmware and iGPU reserve some). Thresholds use
# the class, not the marketing number.
CLASS_4GB = 3 * GB + GB // 2
CLASS_8GB = 7 * GB

# Per-layer cost in a 1080p path-B render (see ARCHITECTURE.md section 3):
# ring of 3 RGBA frames (24.9 MB) + ffmpeg decoder (~60 MB, estimate).
_LAYER_COST = 85 * MB
_RENDER_FIXED_COST = 33 * MB + 80 * MB  # float accumulator + python/numpy


def collect(
    *, disk_dir: Path, disk_bytes: int = 256 * MB, run_benchmarks: bool = True
) -> HardwareProfile:
    vm = psutil.virtual_memory()
    ff, fp = ffbin.locate()
    info: ffbin.FFmpegInfo | None = None
    bench: EncodeBench | None = None
    hw_bench: EncodeBench | None = None
    if ff is not None:
        info = ffbin.probe(ff, fp)
        if run_benchmarks:
            info = ffbin.verify_hw_encoders(info)
            if "libx264" in info.encoders:
                bench = sysinfo.encode_bench(ff)
            hw264 = [e for e in info.hw_encoders_working if e.startswith("h264_")]
            if hw264:
                hw_bench = sysinfo.encode_bench(ff, encoder=hw264[0])
    return HardwareProfile(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        storyreel_version=__version__,
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python=sys.version.split()[0],
        cpu_model=sysinfo.cpu_model(),
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total=vm.total,
        ram_available=vm.available,
        gpus=sysinfo.gpus(),
        disk=sysinfo.disk_bench(disk_dir, total_bytes=disk_bytes) if run_benchmarks else None,
        ffmpeg_path=str(ff) if ff else None,
        ffmpeg_version=info.version_line if info else None,
        hw_encoders=list(info.hw_encoders_working) if info else [],
        encode_bench=bench,
        hw_encode_bench=hw_bench,
    )


def recommend(p: HardwareProfile) -> Recommendations:
    cap = min(3 * GB, p.ram_total // 2)
    small = p.ram_total < 6 * GB
    ram_cache = 128 * MB if small else 256 * MB
    proxy = 360 if (small or p.cores_physical <= 2) else 540
    # Default is one heavy job. The scheduler may go up to max_heavy when, at run time,
    # available RAM covers another full job budget (~1.5 GB) on top of a 2 GB reserve.
    by_ram = int(max(0, p.ram_total - 2 * GB) // int(1.5 * GB))
    max_heavy = max(1, min(p.cores_physical // 2, by_ram, 2))
    if p.cores_physical <= 2 or small:
        preview = "quarter"
    elif p.cores_physical <= 4:
        preview = "half"
    else:
        preview = "full"
    # A hardware encoder frees the CPU for compositing, but only pick it if it measured
    # at least as fast as libx264 veryfast (some iGPU encoders are slower).
    encoder = "libx264"
    hw, sw = p.hw_encode_bench, p.encode_bench
    if hw is not None and (sw is None or hw.fps >= sw.fps):
        encoder = hw.encoder
    encoder_rss = p.encode_bench.peak_rss_bytes if p.encode_bench else 300 * MB
    layers = int((cap - _RENDER_FIXED_COST - encoder_rss) // _LAYER_COST)
    return Recommendations(cap, ram_cache, proxy, 1, max_heavy, preview, encoder, max(1, layers))


def _eta(fps: float, video_seconds: float, out_fps: float = 30.0) -> str:
    wall = video_seconds * out_fps / fps
    return f"{wall / 60:.0f} min" if wall < 3600 else f"{wall / 3600:.1f} h"


def capabilities(p: HardwareProfile) -> list[Capability]:
    caps: list[Capability] = []
    nvidia = [g for g in p.gpus if g.vendor == "nvidia" and g.vram_bytes]
    best_vram = max((g.vram_bytes or 0 for g in nvidia), default=0)
    gpu_names = ", ".join(g.name for g in p.gpus) or "none detected"

    if best_vram >= 12 * GB:
        caps.append(
            Capability(
                "Local AI video generation", True, f"NVIDIA GPU with {best_vram / GB:.0f} GB VRAM"
            )
        )
    else:
        caps.append(
            Capability(
                "Local AI video generation",
                False,
                f"needs an NVIDIA GPU with >= 12 GB VRAM (have: {gpu_names}). "
                "Use cloud backend A or stock/procedural backend C.",
            )
        )

    if best_vram >= 4 * GB:
        caps.append(Capability("Local AI image generation", True, "NVIDIA GPU, CUDA"))
    else:
        caps.append(
            Capability(
                "Local AI image generation",
                False,
                "no CUDA GPU; CPU diffusion would take roughly 20-90 s per image (estimate, "
                "unmeasured). Off by default.",
            )
        )

    caps.append(
        Capability(
            "Offline voice-over (Piper TTS)",
            p.ram_total >= 2 * GB,
            "runs on CPU; speed measured in Phase 4",
        )
    )
    caps.append(
        Capability(
            "Auto captions (faster-whisper base int8)",
            p.ram_total >= 4 * GB,
            "CPU, processed in 30 s windows; speed measured in Phase 4",
        )
    )
    caps.append(
        Capability(
            "Local story-writing LLM (~3B, 4-bit)",
            p.ram_total >= CLASS_8GB,
            "needs ~2.5 GB RAM, slow on CPU; default offline mode uses a rule-based splitter",
        )
    )

    if p.hw_encoders:
        caps.append(Capability("Hardware video encoding", True, ", ".join(p.hw_encoders)))
    else:
        caps.append(Capability("Hardware video encoding", False, "none working; libx264 on CPU"))

    if p.encode_bench:
        fps = p.encode_bench.fps
        caps.append(
            Capability(
                "1080p render (libx264 veryfast)",
                True,
                f"{fps:.0f} fps encode-only; 10 min of video ~ {_eta(fps, 600)}, "
                f"1 h ~ {_eta(fps, 3600)} before effects/compositing",
            )
        )
        caps.append(
            Capability(
                "4K render",
                p.ram_total >= CLASS_8GB,
                f"~ {fps / 4:.0f} fps (4x the pixels, extrapolated); 1 h ~ {_eta(fps / 4, 3600)}",
            )
        )
    else:
        caps.append(Capability("Video rendering", False, "ffmpeg missing or benchmark skipped"))
    return caps


def to_json(p: HardwareProfile) -> dict[str, Any]:
    return {
        "profile": asdict(p),
        "recommendations": asdict(recommend(p)),
        "capabilities": [asdict(c) for c in capabilities(p)],
    }


def save(p: HardwareProfile, path: Path) -> None:
    """Atomic write: temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_json(p), indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
