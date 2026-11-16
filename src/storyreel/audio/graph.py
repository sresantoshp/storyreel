"""Per-chunk audio graph: clips → buses → master + stems, one ffmpeg process.

Window = [s0 - preroll, s1). Everything is rendered over the window and the pre-roll is
trimmed at the end, so stateful filters (EQ/IIR state, compressor envelopes, FFT denoise,
reverb tails) are warm at the chunk's first sample and tails from the previous chunk carry
over. Gain automation (fades, volume keyframes, ducking) is precomputed by numpy as raw
f32 files and multiplied in with `amultiply`; it depends only on timeline positions, so it
is identical whatever the chunking.

Per clip:  decode → resample → layout routing → trim → [areverse] → [atempo] → exact
           length → [x clip envelope] → gain → [balance pan] → delay to its window
           position → pad to the window → [clip effects, with 1 s extra pad for tails] →
           exact window length
Per bus:   silence base + clips → amix (sum, no normalisation) → bus effects → bus gain →
           [x duck envelope] → split into stem + master send
Master:    sum of buses → master effects → master gain → trim pre-roll
Output:    one matroska file, pcm_f32le; stream 0 = master, streams 1..4 = stems in
           spec.BUSES order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from storyreel.audio import envelope, spec
from storyreel.render.plan import ALayer, ChunkPlan, atempo_chain
from storyreel.timebase import sec_str

COMMON = ["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-nostats", "-progress", "pipe:1"]


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _num(x: float) -> str:
    return f"{x:.6g}"


def pan_filter(lay: spec.Layout, pan: float) -> str:
    """Balance pan: left-side channels x min(1, 1-p), right-side x min(1, 1+p); centre and
    LFE untouched. p = 0 is unity (no level change)."""
    g_l, g_r = min(1.0, 1.0 - pan), min(1.0, 1.0 + pan)
    parts = []
    for i, ch in enumerate(lay.channels):
        g = g_l if ch in spec.LEFT else g_r if ch in spec.RIGHT else 1.0
        parts.append(f"c{i}={_num(g)}*c{i}")
    return f"pan={lay.name}|" + "|".join(parts)


def to_layout(src_channels: int, lay: spec.Layout) -> list[str]:
    """Bring a source into the project layout without inventing channels: in 5.1/7.1 a
    mono source goes to FC only and a stereo source to FL/FR only (no fake surround, no
    LFE). Other cases use ffmpeg's standard rematrix (e.g. mono into stereo = -3 dB per
    side, 5.1 source into stereo = ITU downmix)."""
    if lay.count > 2 and src_channels in (1, 2):
        src = "mono" if src_channels == 1 else "stereo"
        route = "FC=c0" if src_channels == 1 else "FL=c0|FR=c1"
        return [f"aformat=sample_fmts=fltp:channel_layouts={src}", f"pan={lay.name}|{route}"]
    return [f"aformat=sample_fmts=fltp:channel_layouts={lay.name}"]


def upmix_mono(lay: spec.Layout) -> str:
    """Copy a mono gain envelope to every channel (pure channel mapping)."""
    return f"pan={lay.name}|" + "|".join(f"c{i}=c0" for i in range(lay.count))


def downmix_filter(src_channels: int, dst_channels: int) -> str:
    """ITU-R BS.775 style downmix to stereo: Lo = FL + 0.707 FC + 0.707 (BL + SL);
    LFE dropped. 7.1 → 5.1 folds the side channels into the backs at 0.707."""
    src = spec.layout(src_channels)
    if dst_channels == src_channels:
        return ""
    if dst_channels == 2:
        k = 0.7071
        terms_l = ["FL", f"{k}*FC", f"{k}*BL"] + ([f"{k}*SL"] if "SL" in src.channels else [])
        terms_r = ["FR", f"{k}*FC", f"{k}*BR"] + ([f"{k}*SR"] if "SR" in src.channels else [])
        if src_channels == 1:
            return "pan=stereo|FL=FC|FR=FC"
        if src_channels == 2:
            return ""
        return f"pan=stereo|FL={'+'.join(terms_l)}|FR={'+'.join(terms_r)}"
    if dst_channels == 6 and src_channels == 8:
        return "pan=5.1|FL=FL|FR=FR|FC=FC|LFE=LFE|BL=BL+0.7071*SL|BR=BR+0.7071*SR"
    raise ValueError(f"cannot downmix {src.name} to {dst_channels} channels")


def simple_fx(etype: str, p: dict[str, Any]) -> str:
    """ffmpeg filter for every effect except reverb (which needs an IR input)."""
    if etype == "eq":
        return f"equalizer=f={_num(p['freq'])}:t=q:w={_num(p['q'])}:g={_num(p['gain'])}"
    if etype == "highpass":
        return f"highpass=f={_num(p['freq'])}:p=2"
    if etype == "lowpass":
        return f"lowpass=f={_num(p['freq'])}:p=2"
    if etype == "compressor":
        thr = max(0.000976563, db_to_lin(p["threshold"]))
        mk = min(64.0, db_to_lin(p["makeup"]))
        return (
            f"acompressor=threshold={_num(thr)}:ratio={_num(p['ratio'])}"
            f":attack={_num(p['attack'])}:release={_num(p['release'])}"
            f":makeup={_num(mk)}:knee={_num(p['knee'])}"
        )
    if etype == "limiter":
        lim = max(0.0625, db_to_lin(p["ceiling"]))
        return (
            f"alimiter=limit={_num(lim)}:attack={_num(p['attack'])}"
            f":release={_num(p['release'])}:level=false:latency=true"
        )
    if etype == "deesser":
        return f"deesser=i={_num(p['intensity'])}:m={_num(p['max_reduction'])}:f={_num(p['freq'])}"
    if etype == "denoise":
        return f"afftdn=nr={_num(p['amount'])}:nf={_num(p['floor'])}"
    raise ValueError(f"unknown audio effect {etype!r}")


def effect_list(raw: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Effects with defaults filled in: [{"type": ..., "params": {...}}]."""
    items: list[dict[str, Any]] = json.loads(raw) if isinstance(raw, str) else raw
    out = []
    for e in items:
        params = {**spec.EFFECTS[e["type"]], **e.get("params", {})}
        out.append({"type": e["type"], "params": params})
    return out


@dataclass
class _Graph:
    inputs: list[list[str]] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    side: list[tuple[Path, envelope.F32]] = field(default_factory=list)
    n: int = 0

    def input(self, args: list[str]) -> int:
        self.inputs.append(args)
        return len(self.inputs) - 1

    def label(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def chain(self, src: str, filters: list[str], prefix: str = "x") -> str:
        if not filters:
            return src
        out = self.label(prefix)
        self.chains.append(f"[{src}]" + ",".join(filters) + f"[{out}]")
        return out


@dataclass(frozen=True)
class AudioContext:
    sample_rate: int
    layout: spec.Layout
    ir_path: Callable[[str], Path]  # room -> cached raw f32le IR in the project layout
    side_prefix: Path  # envelope files are written as <side_prefix>.env<N>.f32


def _envelope_input(g: _Graph, ctx: AudioContext, data: envelope.F32) -> str:
    path = ctx.side_prefix.with_name(ctx.side_prefix.name + f".env{len(g.side)}.f32")
    g.side.append((path, data))
    idx = g.input(
        ["-f", "f32le", "-ar", str(ctx.sample_rate), "-ch_layout", "mono", "-i", str(path)]
    )
    return g.chain(f"{idx}:a", ["aformat=sample_fmts=fltp", upmix_mono(ctx.layout)], "e")


def _apply_fx(g: _Graph, ctx: AudioContext, src: str, effects: list[dict[str, Any]]) -> str:
    """Effects chain. 1 s of silence is appended first so no filter reaches end-of-stream
    inside the window (block-based filters like afftdn flush their last partial block
    differently depending on where the stream started); callers trim to the window."""
    if not effects:
        return src
    pending: list[str] = []
    cur = g.chain(src, [f"apad=pad_len={ctx.sample_rate}"])
    for e in effects:
        if e["type"] != "reverb":
            pending.append(simple_fx(e["type"], e["params"]))
            continue
        cur = g.chain(cur, pending)
        pending = []
        room, mix = str(e["params"]["room"]), float(e["params"]["mix"])
        ir_idx = g.input(
            [
                "-f",
                "f32le",
                "-ar",
                str(ctx.sample_rate),
                "-ch_layout",
                ctx.layout.name,
                "-i",
                str(ctx.ir_path(room)),
            ]
        )
        dry, wet, wet_out, out = g.label("d"), g.label("w"), g.label("r"), g.label("x")
        g.chains.append(f"[{cur}]asplit=2[{dry}][{wet}]")
        g.chains.append(
            f"[{wet}][{ir_idx}:a]afir=irnorm=-1:irgain=1:maxir={_num(spec.reverb_tail_s(room) + 1)}"
            f"[{wet_out}]"
        )
        g.chains.append(
            f"[{dry}][{wet_out}]amix=inputs=2:duration=first:normalize=0"
            f":weights='{_num(1.0 - mix)} {_num(mix)}'[{out}]"
        )
        cur = out
    return g.chain(cur, pending)


def _layer(g: _Graph, ctx: AudioContext, layer: ALayer, window: int) -> str:
    sr, lay = ctx.sample_rate, ctx.layout
    speed = layer.speed[0] / layer.speed[1]
    src_samples = math.ceil(layer.samples * speed)
    idx = g.input(
        ["-ss", sec_str(layer.src_in), "-t", f"{src_samples / sr + 0.05:.6f}", "-i", layer.path]
    )
    steps = [f"aresample={sr}", *to_layout(layer.src_ch, lay)]
    steps += ["asetpts=PTS-STARTPTS", f"atrim=end_sample={src_samples}"]
    if layer.reverse:
        steps.append("areverse")  # bounded: one window of audio
    tempo = atempo_chain(layer.speed)
    if tempo:
        steps.append(tempo)
    steps += [f"apad=whole_len={layer.samples}", f"atrim=end_sample={layer.samples}"]
    cur = g.chain(f"{idx}:a", steps, "c")
    if layer.env:
        e = json.loads(layer.env)
        data = envelope.clip_envelope(
            e["off"], layer.samples, e["len"], sr, e["fi"], e["fo"], e["kf"]
        )
        env = _envelope_input(g, ctx, data)
        out = g.label("c")
        g.chains.append(f"[{cur}][{env}]amultiply[{out}]")
        cur = out
    post = [f"volume={layer.gain_db:.3f}dB"]
    if layer.pan:
        post.append(pan_filter(lay, layer.pan))
    # effects run on the window-aligned stream (frame grid anchored to the timeline)
    post += [f"adelay=delays={layer.at}S:all=1", f"apad=whole_len={window}"]
    cur = g.chain(cur, post, "c")
    cur = _apply_fx(g, ctx, cur, effect_list(layer.effects) if layer.effects else [])
    return g.chain(cur, [f"apad=whole_len={window}", f"atrim=end_sample={window}"], "l")


def _silence(g: _Graph, ctx: AudioContext, window: int) -> str:
    out = g.label("z")
    g.chains.append(
        f"anullsrc=r={ctx.sample_rate}:cl={ctx.layout.name},atrim=end_sample={window}[{out}]"
    )
    return out


def default_mix() -> dict[str, Any]:
    return {
        "buses": {
            b: {"gain_db": 0.0, "muted": False, "fx": [], **spec.BUS_PARAMS} for b in spec.ALL_BUSES
        },
        "speech": [],
    }


def build(
    ffmpeg: Path, plan: ChunkPlan, out: Path, ctx: AudioContext
) -> tuple[list[str], list[tuple[Path, envelope.F32]]]:
    """argv for one chunk, and the envelope files the caller must write before running."""
    g = _Graph()
    pre, n = plan.preroll, plan.samples
    window = pre + n
    mix = json.loads(plan.mix) if plan.mix else default_mix()
    buses: dict[str, dict[str, Any]] = mix["buses"]
    sends: list[str] = []
    stems: list[str] = []
    for bus in spec.BUSES:
        cfg = buses[bus]
        layers = [la for la in plan.alayers if la.bus == bus] if not cfg["muted"] else []
        labels = [_silence(g, ctx, window)] + [_layer(g, ctx, la, window) for la in layers]
        if len(labels) > 1:
            mixed = g.label("b")
            g.chains.append(
                "".join(f"[{x}]" for x in labels)
                + f"amix=inputs={len(labels)}:duration=first:normalize=0[{mixed}]"
            )
        else:
            mixed = labels[0]
        cur = mixed
        if layers:
            cur = _apply_fx(g, ctx, cur, effect_list(cfg["fx"]))
            if cfg["gain_db"]:
                cur = g.chain(cur, [f"volume={cfg['gain_db']:.3f}dB"], "b")
            if cfg.get("duck") and bus in spec.DUCKABLE and mix["speech"]:
                data = envelope.duck_envelope(
                    0,
                    window,
                    [(int(a), int(b)) for a, b in mix["speech"]],
                    ctx.sample_rate,
                    float(cfg["duck_db"]),
                    float(cfg["duck_attack"]),
                    float(cfg["duck_release"]),
                )
                env = _envelope_input(g, ctx, data)
                ducked = g.label("b")
                g.chains.append(f"[{cur}][{env}]amultiply[{ducked}]")
                cur = ducked
        cur = g.chain(cur, [f"apad=whole_len={window}", f"atrim=end_sample={window}"], "b")
        stem, send = g.label("s"), g.label("m")
        g.chains.append(f"[{cur}]asplit=2[{stem}][{send}]")
        stems.append(stem)
        sends.append(send)
    master = g.label("M")
    g.chains.append(
        "".join(f"[{x}]" for x in sends)
        + f"amix=inputs={len(sends)}:duration=first:normalize=0[{master}]"
    )
    mcfg = buses["master"]
    cur = _apply_fx(g, ctx, master, effect_list(mcfg["fx"]))
    trim = [
        f"atrim=start_sample={pre}",
        "asetpts=PTS-STARTPTS",
        f"apad=whole_len={n}",
        f"atrim=end_sample={n}",
    ]
    mgain = ["volume=0"] if mcfg["muted"] else [f"volume={mcfg['gain_db']:.3f}dB"]
    final = g.chain(cur, [*mgain, *trim], "out")
    outs = [final] + [g.chain(st, trim, "stem") for st in stems]

    argv = [str(ffmpeg), *COMMON]
    for args in g.inputs:
        argv += args
    argv += ["-filter_complex", ";".join(g.chains)]
    for o in outs:
        argv += ["-map", f"[{o}]"]
    argv += ["-c:a", "pcm_f32le", "-f", "matroska", str(out)]
    return argv, g.side
