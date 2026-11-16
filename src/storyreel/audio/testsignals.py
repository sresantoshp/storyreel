"""Procedural test audio, generated offline by ffmpeg's lavfi sources (no downloads).

speech   - "talking" signal: a voiced harmonic source (f0 ~ 120-170 Hz with intonation),
           shaped by a 4-syllable/s envelope, phrases of 2.5 s separated by 0.8 s pauses,
           with short noise bursts for fricatives; band-limited 100 Hz-5 kHz. Not intelligible,
           but it has speech-like level dynamics, spectrum and pauses, so ducking, de-essing
           and loudness can be verified.
music    - bed: two alternating triads (A minor / F major, 2 s each) on detuned sines per
           channel, a soft bass pulse at 120 BPM and slow tremolo.
ambience - pink noise, low-passed at 4 kHz (room tone / wind).
fx       - a sharp click with a short decaying 1 kHz ping every 1.5 s.
"""

from __future__ import annotations

from pathlib import Path

from storyreel.proc import run

KINDS = ("speech", "music", "ambience", "fx")

_SPEECH = (
    "(sin(2*PI*140*t+4*sin(2*PI*0.7*t))"
    "+0.6*sin(2*PI*280*t+8*sin(2*PI*0.7*t))"
    "+0.45*sin(2*PI*420*t+12*sin(2*PI*0.7*t))"
    "+0.3*sin(2*PI*560*t+16*sin(2*PI*0.7*t))"
    "+0.25*sin(2*PI*840*t+24*sin(2*PI*0.7*t))"
    "+0.15*sin(2*PI*1260*t+36*sin(2*PI*0.7*t)))"
    "*pow(abs(sin(2*PI*2*t)),1.5)"
    "*lt(mod(t,3.3),2.5)*0.22"
    "+0.08*(2*random(0)-1)*gt(sin(2*PI*1.3*t),0.9)*lt(mod(t,3.3),2.5)"
)


def _music(detune: float) -> str:
    chord_a = (220.0, 261.63, 329.63)
    chord_f = (174.61, 220.0, 261.63)

    def triad(freqs: tuple[float, ...]) -> str:
        return "+".join(f"sin(2*PI*{f * detune:.3f}*t)" for f in freqs)

    return (
        f"(if(lt(mod(t,4),2),{triad(chord_a)},{triad(chord_f)}))*0.09"
        "*(0.85+0.15*sin(2*PI*0.25*t))"
        "+0.18*sin(2*PI*55*t)*exp(-6*mod(t,0.5))"
    )


def argv(ffmpeg: Path, kind: str, out: Path, duration: float, sample_rate: int) -> list[str]:
    d = f"{duration:.3f}"
    if kind == "speech":
        src = f"aevalsrc=exprs='{_SPEECH}':s={sample_rate}:d={d}"
        post = "highpass=f=100,lowpass=f=5000"
    elif kind == "music":
        src = f"aevalsrc=exprs='{_music(1.0)}|{_music(1.003)}':s={sample_rate}:d={d}"
        post = "anull"
    elif kind == "ambience":
        src = f"anoisesrc=color=pink:amplitude=0.25:seed=7:r={sample_rate}:d={d}"
        post = "lowpass=f=4000"
    elif kind == "fx":
        src = (
            "aevalsrc=exprs='0.5*sin(2*PI*1000*t)*exp(-25*mod(t,1.5))*lt(mod(t,1.5),0.4)'"
            f":s={sample_rate}:d={d}"
        )
        post = "anull"
    else:
        raise ValueError(f"kind must be one of {KINDS}")
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        src,
        "-af",
        post,
        "-c:a",
        "pcm_s16le",
        str(out),
    ]


def generate(
    ffmpeg: Path, kind: str, out: Path, duration: float = 20.0, sample_rate: int = 48_000
) -> Path:
    run(argv(ffmpeg, kind, out, duration, sample_rate), timeout_s=120 + duration)
    return out
