"""Names, parameters, ranges and defaults for the audio engine. Pure data, no numpy.

The editor validates against this module; the graph builder reads defaults from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# -- buses ------------------------------------------------------------------------------
BUSES: tuple[str, ...] = ("dialogue", "music", "fx", "ambience")  # stem order
ALL_BUSES: tuple[str, ...] = (*BUSES, "master")
DUCKABLE = frozenset({"music", "fx", "ambience"})  # ducked under the dialogue bus

BUS_PARAMS: dict[str, Any] = {
    "duck": False,
    "duck_db": -12.0,  # gain while dialogue plays
    "duck_attack": 0.25,  # s: ramp down, finishing when the dialogue clip starts
    "duck_release": 0.6,  # s: ramp up after the dialogue clip ends
}
BUS_RANGES: dict[str, tuple[float, float]] = {
    "gain_db": (-60.0, 24.0),
    "duck_db": (-60.0, 0.0),
    "duck_attack": (0.0, 5.0),
    "duck_release": (0.0, 10.0),
}

# -- per-clip audio ----------------------------------------------------------------------
CLIP_PROPS: dict[str, Any] = {"pan": 0.0, "fade_in": 0.0, "fade_out": 0.0}
CLIP_RANGES: dict[str, tuple[float, float]] = {
    "pan": (-1.0, 1.0),
    "fade_in": (0.0, 3600.0),
    "fade_out": (0.0, 3600.0),
}
KEYFRAMABLE = frozenset({"volume"})  # dB, added to the clip gain
VOLUME_RANGE = (-96.0, 24.0)

# -- effects -----------------------------------------------------------------------------
EFFECTS: dict[str, dict[str, Any]] = {
    "eq": {"freq": 1000.0, "gain": 0.0, "q": 1.0},  # one peaking band; stack for more
    "highpass": {"freq": 80.0},
    "lowpass": {"freq": 12000.0},
    "compressor": {
        "threshold": -18.0,  # dBFS
        "ratio": 3.0,
        "attack": 10.0,  # ms
        "release": 150.0,  # ms
        "makeup": 0.0,  # dB
        "knee": 2.8,
    },
    "limiter": {"ceiling": -1.0, "attack": 5.0, "release": 50.0},  # dBFS, ms, ms
    "deesser": {"intensity": 0.5, "max_reduction": 0.5, "freq": 0.5},
    "denoise": {"amount": 12.0, "floor": -50.0},  # dB reduction, dBFS noise floor
    "reverb": {"room": "hall", "mix": 0.25},
}
EFFECT_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("eq", "freq"): (20.0, 20000.0),
    ("eq", "gain"): (-24.0, 24.0),
    ("eq", "q"): (0.1, 20.0),
    ("highpass", "freq"): (10.0, 20000.0),
    ("lowpass", "freq"): (100.0, 22000.0),
    ("compressor", "threshold"): (-60.0, 0.0),
    ("compressor", "ratio"): (1.0, 20.0),
    ("compressor", "attack"): (0.01, 2000.0),
    ("compressor", "release"): (0.01, 9000.0),
    ("compressor", "makeup"): (0.0, 36.0),
    ("compressor", "knee"): (1.0, 8.0),
    ("limiter", "ceiling"): (-24.0, 0.0),
    ("limiter", "attack"): (0.1, 80.0),
    ("limiter", "release"): (1.0, 8000.0),
    ("deesser", "intensity"): (0.0, 1.0),
    ("deesser", "max_reduction"): (0.0, 1.0),
    ("deesser", "freq"): (0.0, 1.0),
    ("denoise", "amount"): (0.01, 97.0),
    ("denoise", "floor"): (-80.0, -20.0),
    ("reverb", "mix"): (0.0, 1.0),
}


@dataclass(frozen=True)
class Room:
    """Procedural impulse response recipe (see audio/ir.py)."""

    rt60: float  # s, decay to -60 dB at low/mid frequencies
    hf_ratio: float  # high-band RT60 = rt60 * hf_ratio (air/wall absorption)
    predelay: float  # s
    early: tuple[tuple[float, float], ...]  # (delay s, gain) discrete early reflections
    density: float = 1.0  # 1 = dense noise tail; < 1 = sparse, echoing (cave)


ROOMS: dict[str, Room] = {
    "small_room": Room(0.45, 0.6, 0.004, ((0.007, 0.5), (0.011, 0.4), (0.017, 0.3), (0.023, 0.25))),
    "hall": Room(1.9, 0.55, 0.022, ((0.025, 0.4), (0.037, 0.35), (0.051, 0.3), (0.068, 0.2))),
    "cathedral": Room(4.2, 0.45, 0.045, ((0.052, 0.35), (0.081, 0.3), (0.117, 0.25))),
    "cave": Room(
        3.0,
        0.35,
        0.06,
        ((0.09, 0.5), (0.18, 0.35), (0.27, 0.25), (0.36, 0.18), (0.45, 0.12)),
        density=0.25,
    ),
}

# Every effect has state (IIR filters, envelope followers, FFT frames, reverb tails), so a
# chunk is rendered from WARMUP_S (+ the longest reverb tail) before its first sample
# whenever any effect is in use, and that pre-roll is trimmed off.
WARMUP_S = 2.0


def reverb_tail_s(room: str) -> float:
    r = ROOMS[room]
    return r.predelay + r.rt60 + max((d for d, _ in r.early), default=0.0)


# -- layouts -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    name: str  # ffmpeg channel layout name
    channels: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.channels)


LAYOUTS: dict[int, Layout] = {
    1: Layout("mono", ("FC",)),
    2: Layout("stereo", ("FL", "FR")),
    6: Layout("5.1", ("FL", "FR", "FC", "LFE", "BL", "BR")),
    8: Layout("7.1", ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR")),
}
LAYOUT_NAMES = {"mono": 1, "stereo": 2, "5.1": 6, "7.1": 8}
LEFT = frozenset({"FL", "BL", "SL"})
RIGHT = frozenset({"FR", "BR", "SR"})


def layout(channels: int) -> Layout:
    try:
        return LAYOUTS[channels]
    except KeyError:
        raise ValueError(f"unsupported channel count {channels}; use 1, 2, 6 or 8") from None


# -- loudness and codecs ---------------------------------------------------------------------
LOUDNESS_DEFAULTS: dict[str, Any] = {"target_lufs": -14.0, "true_peak": -1.0, "normalize": True}
LOUDNESS_PRESETS = {"youtube": -14.0, "broadcast": -23.0}
LOUDNESS_RANGES: dict[str, tuple[float, float]] = {
    "target_lufs": (-36.0, -5.0),
    "true_peak": (-9.0, 0.0),
}


@dataclass(frozen=True)
class Codec:
    encoder: str
    max_channels: int
    bitrate: dict[int, str]  # channels -> default bitrate ("" = lossless)
    lossy: bool = True


CODECS: dict[str, Codec] = {
    "aac": Codec("aac", 8, {1: "128k", 2: "192k", 6: "384k", 8: "512k"}),
    "ac3": Codec("ac3", 6, {1: "96k", 2: "192k", 6: "448k"}),
    "eac3": Codec("eac3", 6, {1: "96k", 2: "192k", 6: "384k"}),
    "opus": Codec("libopus", 8, {1: "96k", 2: "128k", 6: "256k", 8: "384k"}),
    "flac": Codec("flac", 8, {}, lossy=False),
}
