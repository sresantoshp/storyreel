from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from conftest import needs_ffmpeg
from storyreel.hw import profile as hp
from storyreel.hw import sysinfo
from storyreel.hw.sysinfo import GB, MB, EncodeBench, GpuInfo
from storyreel.media import ffbin


def machine(ram_gb: float, cores: int, **kw: Any) -> hp.HardwareProfile:
    base = hp.HardwareProfile(
        created_at="t",
        storyreel_version="0",
        os="test",
        python="3.13",
        cpu_model="cpu",
        cores_physical=cores,
        cores_logical=cores * 2,
        ram_total=int(ram_gb * GB),
        ram_available=int(ram_gb * GB / 2),
        gpus=[],
        disk=None,
        ffmpeg_path=None,
        ffmpeg_version=None,
    )
    return replace(base, **kw)


def bench(encoder: str, fps: float, rss_mb: int = 400) -> EncodeBench:
    return EncodeBench(encoder, 1920, 1080, 120, 120 / fps, fps, rss_mb * MB)


def test_4gb_dual_core_gets_conservative_defaults() -> None:
    r = hp.recommend(machine(3.8, 2))
    assert r.memory_cap_bytes == int(3.8 * GB) // 2
    assert r.ram_cache_bytes == 128 * MB
    assert r.proxy_height == 360
    assert r.heavy_workers == 1
    assert r.max_heavy_workers == 1
    assert r.preview_quality == "quarter"


def test_8gb_class_laptop() -> None:
    r = hp.recommend(machine(7.6, 4))
    assert r.memory_cap_bytes == 3 * GB  # 50 % of 7.6 GB is above the 3 GB ceiling
    assert r.heavy_workers == 1  # spec: one heavy job by default
    assert r.proxy_height == 540
    assert r.preview_quality == "half"


def test_memory_cap_never_exceeds_3gb() -> None:
    assert hp.recommend(machine(64, 16)).memory_cap_bytes == 3 * GB


def test_hw_encoder_only_chosen_when_faster() -> None:
    sw = bench("libx264", 100)
    slow_hw = machine(16, 8, encode_bench=sw, hw_encode_bench=bench("h264_qsv", 70))
    fast_hw = machine(16, 8, encode_bench=sw, hw_encode_bench=bench("h264_qsv", 250))
    assert hp.recommend(slow_hw).video_encoder == "libx264"
    assert hp.recommend(fast_hw).video_encoder == "h264_qsv"


def test_layer_budget_uses_measured_encoder_rss() -> None:
    small_enc = hp.recommend(machine(16, 8, encode_bench=bench("libx264", 100, rss_mb=200)))
    big_enc = hp.recommend(machine(16, 8, encode_bench=bench("libx264", 100, rss_mb=1200)))
    assert small_enc.max_layers_1080p > big_enc.max_layers_1080p >= 1


def test_capabilities_are_honest_without_gpu() -> None:
    caps = {c.name: c for c in hp.capabilities(machine(7.6, 4, encode_bench=bench("libx264", 90)))}
    assert not caps["Local AI video generation"].ok
    assert not caps["Local AI image generation"].ok
    assert caps["4K render"].ok  # 7.6 GB reported = 8 GB-class machine
    assert caps["Local story-writing LLM (~3B, 4-bit)"].ok


def test_big_nvidia_gpu_enables_local_video() -> None:
    gpu = GpuInfo("NVIDIA GeForce RTX 4090", "nvidia", 24 * GB, True)
    caps = {c.name: c for c in hp.capabilities(machine(32, 16, gpus=[gpu]))}
    assert caps["Local AI video generation"].ok


def test_save_is_atomic_json(tmp_path: Path) -> None:
    out = tmp_path / "p.json"
    hp.save(machine(8, 4), out)
    assert out.read_text(encoding="utf-8").startswith("{")
    assert not out.with_suffix(".tmp").exists()


def test_disk_bench_writes_and_cleans_up(tmp_path: Path) -> None:
    res = sysinfo.disk_bench(tmp_path, total_bytes=8 * MB)
    assert res.bytes_written == 8 * MB
    assert res.write_mb_s is not None and res.write_mb_s > 0
    assert list(tmp_path.iterdir()) == []


@needs_ffmpeg
def test_encode_bench_reports_fps_and_rss() -> None:
    ff, _ = ffbin.locate()
    assert ff is not None
    b = sysinfo.encode_bench(ff, width=320, height=240, seconds=1.0)
    assert b.frames == 30
    assert b.fps > 0
    assert b.peak_rss_bytes > 1 * MB
