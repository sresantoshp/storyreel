"""Command-line entry point: `storyreel <command>`.

Only commands that actually work are registered. `make` arrives in Phase 4.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from storyreel import __version__, cli_audio, cli_edit, cli_project, doctor, logs, paths
from storyreel.errors import ExitCode, StoryReelError
from storyreel.hw import profile as hwprofile
from storyreel.hw.sysinfo import GB, MB

log = logging.getLogger("storyreel.cli")


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks = doctor.run_checks(test_hw=not args.no_hw_test)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2, default=str))
    else:
        print(doctor.format_report(checks))
    failed = any(c.status is doctor.Status.FAIL for c in checks)
    return int(ExitCode.ENV_CHECK_FAILED if failed else ExitCode.OK)


def _fmt_profile(p: hwprofile.HardwareProfile) -> str:
    r = hwprofile.recommend(p)
    out = [
        f"OS        {p.os}",
        f"CPU       {p.cpu_model}  ({p.cores_physical} cores / {p.cores_logical} threads)",
        f"RAM       {p.ram_total / GB:.1f} GB total, {p.ram_available / GB:.1f} GB available",
    ]
    for g in p.gpus or []:
        vram = f"{g.vram_bytes / GB:.1f} GB VRAM" if g.vram_bytes else "shared memory"
        out.append(f"GPU       {g.name}  ({'discrete' if g.discrete else 'integrated'}, {vram})")
    if p.disk:
        speed = (
            f"{p.disk.write_mb_s:.0f} MB/s seq write (fsync)" if p.disk.write_mb_s else "skipped"
        )
        out.append(f"Disk      {p.disk.free_bytes / GB:.0f} GB free, {speed}  [{p.disk.path}]")
    out.append(f"ffmpeg    {p.ffmpeg_version or 'NOT FOUND'}")
    out.append(f"HW enc    {', '.join(p.hw_encoders) or 'none working'}")
    for b in (p.encode_bench, p.hw_encode_bench):
        if b is not None:
            out.append(
                f"{b.encoder:<9} {b.width}x{b.height}: {b.fps:.1f} fps "
                f"({b.frames} frames in {b.wall_s:.2f} s), "
                f"ffmpeg peak RSS {b.peak_rss_bytes / MB:.0f} MB"
            )
    out += [
        "",
        "Safe defaults chosen for this machine:",
        f"  memory cap        {r.memory_cap_bytes / GB:.1f} GB",
        f"  RAM cache         {r.ram_cache_bytes / MB:.0f} MB",
        f"  proxy height      {r.proxy_height}p",
        f"  heavy workers     {r.heavy_workers} (scheduler may scale to {r.max_heavy_workers} "
        f"when RAM is free)",
        f"  preview quality   {r.preview_quality}",
        f"  video encoder     {r.video_encoder}",
        f"  max 1080p layers  {r.max_layers_1080p} (path-B compositor, within cap)",
        "",
        "What this machine can and cannot do:",
    ]
    for c in hwprofile.capabilities(p):
        out.append(f"  [{'yes' if c.ok else 'no '}] {c.name}: {c.detail}")
    return "\n".join(out)


def _cmd_profile(args: argparse.Namespace) -> int:
    disk_dir = Path(args.disk_dir) if args.disk_dir else paths.data_dir()
    p = hwprofile.collect(
        disk_dir=disk_dir,
        disk_bytes=(64 if args.quick else 256) * MB,
        run_benchmarks=not args.no_bench,
    )
    hwprofile.save(p, paths.profile_path())
    if args.json:
        print(json.dumps(hwprofile.to_json(p), indent=2, default=str))
    else:
        print(_fmt_profile(p))
        print(f"\nSaved to {paths.profile_path()}")
    return int(ExitCode.OK)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="storyreel", description="Text-to-video studio.")
    ap.add_argument("--version", action="version", version=f"storyreel {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug-level file logging")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check ffmpeg, codecs, disk, RAM, keys; print fixes")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.add_argument("--no-hw-test", action="store_true", help="skip test-encoding on GPU encoders")
    d.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("profile", help="measure this machine and pick safe defaults")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quick", action="store_true", help="64 MB disk test instead of 256 MB")
    p.add_argument("--no-bench", action="store_true", help="skip disk/encoder benchmarks")
    p.add_argument("--disk-dir", help="directory to benchmark (default: StoryReel data dir)")
    p.set_defaults(func=_cmd_profile)
    cli_project.register(sub)
    cli_edit.register(sub)
    cli_audio.register(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    # Legacy Windows consoles use cp1252; never crash on a character the console cannot show.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    log_path = logs.configure(
        paths.log_dir(), level=logging.DEBUG if args.verbose else logging.INFO
    )
    log.info("command start", extra={"command": args.command, "argv": sys.argv[1:]})
    try:
        rc: int = args.func(args)
    except StoryReelError as exc:
        log.exception("command failed", extra={"command": args.command})
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.fix:
            print(f"fix:   {exc.fix}", file=sys.stderr)
        return int(exc.exit_code)
    except KeyboardInterrupt:
        return int(ExitCode.INTERRUPTED)
    except Exception:
        log.exception("unexpected error", extra={"command": args.command})
        print(f"error: unexpected internal error. Details logged to {log_path}", file=sys.stderr)
        return int(ExitCode.ERROR)
    log.info("command end", extra={"command": args.command, "rc": rc})
    return rc
