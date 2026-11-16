"""ffmpeg argv builders.

Video path A: black base + one input per layer, each normalised (fps, fit-scale with an
explicit YUV matrix → BT.709, pad, yuv420p), delayed to its position, overlaid bottom-to-top.
Video path B: the worker (render/worker.py) encodes compositor output with the same
`encoder_args`, so chunks from both paths concatenate with `-c copy` and match in colour.
Audio chunks are built by storyreel.audio.graph; the mux here applies the final audio
filter (downmix, loudness gain, true-peak limiter) and the one lossy encode.
"""

from __future__ import annotations

from pathlib import Path

from storyreel.audio import spec as aspec
from storyreel.render.plan import ChunkPlan, RenderSettings
from storyreel.timebase import sec_str

COMMON = ["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-nostats", "-progress", "pipe:1"]
COLOR_TAGS = [
    "-colorspace",
    "bt709",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-color_range",
    "tv",
]


def encoder_args(s: RenderSettings, *, low_mem: bool = False) -> list[str]:
    gop = max(1, round(2 * s.rate.fps))
    args = [
        "-c:v",
        s.vcodec,
        "-preset",
        s.preset,
        "-crf",
        str(s.crf),
        "-g",
        str(gop),
        "-pix_fmt",
        s.pix_fmt,
        *COLOR_TAGS,
    ]
    if low_mem:
        args += ["-threads", "2", "-x264-params", "rc-lookahead=10", "-filter_threads", "1"]
    return args


def rgb_encoder_args(s: RenderSettings, *, low_mem: bool = False) -> list[str]:
    """For rgb24 input (path B): convert with the BT.709 matrix into limited-range YUV."""
    vf = f"scale=out_color_matrix=bt709:out_range=tv,format={s.pix_fmt}"
    return ["-vf", vf, *encoder_args(s, low_mem=low_mem)]


def video_argv(
    ffmpeg: Path, plan: ChunkPlan, s: RenderSettings, out: Path, *, low_mem: bool = False
) -> list[str]:
    fps = str(s.rate)
    w, h = s.width, s.height
    dur = plan.frames * s.rate.den / s.rate.num
    argv = [
        str(ffmpeg),
        *COMMON,
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={w}x{h}:r={fps}:d={dur:.6f}",
    ]
    chains: list[str] = []
    for i, layer in enumerate(plan.vlayers, start=1):
        ldur = layer.frames * s.rate.den / s.rate.num
        # one extra frame of input so rounding in the demuxer never leaves us a frame short
        in_dur = f"{ldur + s.rate.den / s.rate.num:.6f}"
        if layer.kind == "video":
            argv += ["-ss", sec_str(layer.src_in), "-t", in_dur, "-i", str(layer.path)]
        elif layer.kind == "image":
            argv += ["-loop", "1", "-framerate", fps, "-t", in_dur, "-i", str(layer.path)]
        else:
            argv += ["-f", "lavfi", "-i", f"color=c=0x{layer.color}:s={w}x{h}:r={fps}:d={in_dur}"]
        at = layer.at * s.rate.den / s.rate.num
        # swscale does not change the YUV matrix in a yuv->yuv scale, so convert explicitly
        # (matrix only, like path B's decoder; primaries and transfer are left alone)
        cs = {"bt601": "bt470bg", "bt2020": "bt2020ncl"}.get(layer.matrix)
        convert = (f"colorspace=space=bt709:ispace={cs}:primaries=bt709:iprimaries=bt709:"
                   f"trc=bt709:itrc=bt709:range=tv:irange=tv," if cs else "")  # fmt: skip
        chains.append(
            f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},{convert}"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=bicubic"
            f":out_color_matrix=bt709:out_range=tv,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p,"
            f"trim=end_frame={layer.frames},setpts=PTS+{at:.6f}/TB[l{i}]"
        )
    prev = "0:v"
    for i in range(1, len(plan.vlayers) + 1):
        chains.append(f"[{prev}][l{i}]overlay=eof_action=pass:repeatlast=0[o{i}]")
        prev = f"o{i}"
    chains.append(f"[{prev}]format={s.pix_fmt}[vout]")
    argv += ["-filter_complex", ";".join(chains), "-map", "[vout]", "-frames:v", str(plan.frames)]
    argv += encoder_args(s, low_mem=low_mem)
    argv += ["-f", "matroska", str(out)]
    return argv


def concat_list(entries: list[tuple[Path, float]]) -> str:
    """ffmpeg concat-demuxer script with exact per-file durations."""
    lines = ["ffconcat version 1.0"]
    for path, dur in entries:
        p = path.resolve().as_posix().replace("'", "'\\''")
        lines += [f"file '{p}'", f"duration {dur:.6f}"]
    return "\n".join(lines) + "\n"


def container(out: Path) -> str:
    return {".mkv": "matroska", ".mov": "mov", ".webm": "webm"}.get(out.suffix.lower(), "mp4")


def audio_codec_args(s: RenderSettings, channels: int) -> list[str]:
    codec = aspec.CODECS[s.acodec]
    args = ["-c:a", codec.encoder]
    bitrate = s.abitrate or codec.bitrate.get(channels, "")
    if codec.lossy and bitrate:
        args += ["-b:a", bitrate]
    if s.acodec == "opus":
        args += ["-ar", "48000"]  # Opus runs at 48 kHz only
        if channels > 2:
            args += ["-mapping_family", "1"]
    else:
        args += ["-ar", str(s.sample_rate)]
    if s.acodec == "flac":
        args += ["-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
    return args


def mux_argv(
    ffmpeg: Path,
    vlist: Path,
    alist: Path,
    s: RenderSettings,
    out: Path,
    *,
    audio_filter: str = "",
    channels: int | None = None,
) -> list[str]:
    fmt = container(out)
    argv = [
        str(ffmpeg),
        *COMMON,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(vlist),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(alist),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    if audio_filter:
        argv += ["-af", audio_filter]
    argv += audio_codec_args(s, channels or s.channels)
    if fmt == "mp4":
        argv += ["-movflags", "+faststart"]
    argv += ["-f", fmt, str(out)]
    return argv


def stem_argv(
    ffmpeg: Path, alist: Path, stream: int, gain_db: float, fmt: str, out: Path
) -> list[str]:
    """One bus stem from the chunk PCM: same gain as the master, no limiter."""
    codec = (
        ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", "-f", "flac"]
        if fmt == "flac"
        else ["-c:a", "pcm_f32le", "-f", "wav"]
    )
    return [
        str(ffmpeg),
        *COMMON,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(alist),
        "-map",
        f"0:a:{stream}",
        "-af",
        f"volume={gain_db:.3f}dB",
        *codec,
        str(out),
    ]
