"""Audio commands: loudness, track, bus, afx, audio, gen-audio.

Every editing command is one undoable edit:
  storyreel track film.storyreel A2 bus=music
  storyreel bus film.storyreel music gain_db=-3 duck_db=-14
  storyreel afx film.storyreel 12 compressor threshold=-20 ratio=4     (clip 12)
  storyreel afx film.storyreel fx reverb room=cathedral mix=0.3         (bus "fx")
  storyreel set film.storyreel 12 pan=-0.3 fade_in=0.5 fade_out=2
  storyreel keyframe film.storyreel 12 volume 4.0 -12
  storyreel audio film.storyreel layout=5.1 target_lufs=broadcast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from storyreel.audio import loudness, testsignals
from storyreel.audio import spec as aspec
from storyreel.cli_edit import parse_kv
from storyreel.errors import ExitCode, StoryReelError
from storyreel.project import store
from storyreel.project.edits import Editor


def _ffmpeg() -> Path:
    from storyreel.cli_project import _tools

    return _tools()[0]


def cmd_loudness(a: argparse.Namespace) -> int:
    m = loudness.measure_file(_ffmpeg(), Path(a.file))
    if a.json:
        print(
            json.dumps(
                {"integrated_lufs": m.integrated, "lra_lu": m.lra, "true_peak_dbtp": m.true_peak}
            )
        )
    else:
        print(f"Integrated loudness  {m.integrated:6.1f} LUFS")
        print(f"Loudness range       {m.lra:6.1f} LU")
        print(f"True peak            {m.true_peak:6.1f} dBTP")
    return int(ExitCode.OK)


def _track_id(p: store.Project, ref: str) -> int:
    row = p.con.execute(
        "SELECT id FROM track WHERE name = ? OR (kind = 'audio' AND CAST(id AS TEXT) = ?)",
        (ref, ref),
    ).fetchone()
    if row is None:
        raise StoryReelError(f"No track named {ref!r}.", fix="See `storyreel info`.")
    return int(row["id"])


def cmd_track(a: argparse.Namespace) -> int:
    kv = parse_kv(a.settings)
    unknown = set(kv) - {"bus", "muted"}
    if unknown:
        raise StoryReelError(f"Track settings are bus= and muted=, not {sorted(unknown)}.")
    with store.open_project(Path(a.project)) as p:
        tid = _track_id(p, a.track)
        Editor(p.con).set_track(tid, bus=kv.get("bus"), muted=kv.get("muted"))
    print(f"Updated track {a.track}")
    return int(ExitCode.OK)


def _print_buses(p: store.Project) -> None:
    for b in p.con.execute("SELECT * FROM bus ORDER BY id"):
        params = {**aspec.BUS_PARAMS, **json.loads(b["params"])}
        duck = (
            f" duck {params['duck_db']:g} dB (attack {params['duck_attack']:g} s, "
            f"release {params['duck_release']:g} s)"
            if params["duck"] and b["name"] in aspec.DUCKABLE
            else ""
        )
        mute = " MUTED" if b["muted"] else ""
        print(f"  bus {b['name']:<9} gain {b['gain_db']:+.1f} dB{mute}{duck}")
        for e in p.con.execute(
            "SELECT * FROM bus_effect WHERE bus_id = ? ORDER BY idx", (b["id"],)
        ):
            print(f"      fx #{e['id']:<4} {e['type']} {e['params']}")


def cmd_bus(a: argparse.Namespace) -> int:
    with store.open_project(Path(a.project)) as p:
        if a.settings:
            Editor(p.con).set_bus(a.bus, **parse_kv(a.settings))
        _print_buses(p)
    return int(ExitCode.OK)


def cmd_afx(a: argparse.Namespace) -> int:
    with store.open_project(Path(a.project)) as p:
        ed = Editor(p.con)
        is_bus = a.target in aspec.ALL_BUSES
        if a.remove is not None:
            if is_bus:
                ed.remove_bus_effect(a.remove)
            else:
                ed.remove_effect(a.remove)
            print(f"Removed effect #{a.remove}")
            return int(ExitCode.OK)
        if not a.type:
            raise StoryReelError(
                "Give an audio effect type.", fix="Effects: " + ", ".join(aspec.EFFECTS)
            )
        params = parse_kv(a.params)
        if is_bus:
            eid = ed.add_bus_effect(a.target, a.type, **params)
            print(f"Added {a.type} #{eid} to bus {a.target}")
        else:
            if not a.target.isdigit():
                raise StoryReelError(
                    f"{a.target!r} is neither a clip id nor a bus.",
                    fix=f"Buses: {', '.join(aspec.ALL_BUSES)}.",
                )
            row = p.con.execute(
                "SELECT t.kind FROM clip c JOIN track t ON t.id = c.track_id WHERE c.id = ?",
                (int(a.target),),
            ).fetchone()
            if row is None or row["kind"] != "audio":
                raise StoryReelError(f"Clip {a.target} is not an audio clip.")
            eid = ed.add_effect(int(a.target), a.type, **params)
            print(f"Added {a.type} #{eid} to clip {a.target}")
    return int(ExitCode.OK)


def cmd_audio(a: argparse.Namespace) -> int:
    with store.open_project(Path(a.project)) as p:
        if a.settings:
            Editor(p.con).set_audio(**parse_kv(a.settings))
        s = p.settings()
        norm = (
            f"normalise to {s.target_lufs:g} LUFS, true peak <= {s.true_peak:g} dBTP"
            if s.normalize
            else "no loudness normalisation"
        )
        print(f"Audio: {aspec.layout(s.channels).name}, {s.sample_rate} Hz, {norm}")
    return int(ExitCode.OK)


def cmd_gen_audio(a: argparse.Namespace) -> int:
    out = testsignals.generate(_ffmpeg(), a.kind, Path(a.output), a.dur, a.sample_rate)
    print(f"Wrote {out} ({a.kind}, {a.dur:g} s)")
    return int(ExitCode.OK)


def register(sub: Any) -> None:
    lo = sub.add_parser("loudness", help="measure integrated loudness, LRA and true peak")
    lo.add_argument("file")
    lo.add_argument("--json", action="store_true")
    lo.set_defaults(func=cmd_loudness)

    tr = sub.add_parser("track", help="route an audio track to a bus / mute it: bus=music")
    tr.add_argument("project")
    tr.add_argument("track", help="track name, e.g. A2")
    tr.add_argument("settings", nargs="+", help="bus=dialogue|music|fx|ambience, muted=true")
    tr.set_defaults(func=cmd_track)

    bu = sub.add_parser("bus", help="show or set bus gain, mute and ducking")
    bu.add_argument("project")
    bu.add_argument("bus", choices=list(aspec.ALL_BUSES))
    bu.add_argument("settings", nargs="*", help="gain_db=-3 muted=false duck=true duck_db=-12 ...")
    bu.set_defaults(func=cmd_bus)

    fx = sub.add_parser("afx", help="add/remove an audio effect on a clip or a bus")
    fx.add_argument("project")
    fx.add_argument("target", help="audio clip id, or bus name (dialogue/music/fx/ambience/master)")
    fx.add_argument("type", nargs="?", choices=[*aspec.EFFECTS, None])
    fx.add_argument("params", nargs="*", help="key=value, e.g. room=hall mix=0.3")
    fx.add_argument("--remove", type=int, metavar="EFFECT_ID")
    fx.set_defaults(func=cmd_afx)

    au = sub.add_parser("audio", help="project audio: layout, loudness target, true peak")
    au.add_argument("project")
    au.add_argument(
        "settings",
        nargs="*",
        help="layout=stereo|5.1|7.1 target_lufs=-14|youtube|broadcast true_peak=-1 normalize=true",
    )
    au.set_defaults(func=cmd_audio)

    g = sub.add_parser("gen-audio", help="make procedural test audio offline (ffmpeg lavfi)")
    g.add_argument("kind", choices=list(testsignals.KINDS))
    g.add_argument("output")
    g.add_argument("--dur", type=float, default=20.0)
    g.add_argument("--sample-rate", type=int, default=48_000)
    g.set_defaults(func=cmd_gen_audio)
