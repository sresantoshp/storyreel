"""Editing commands: text, env, fx, kenburns, keyframe, transition, set, speed, frame,
undo, redo. Every command is one undoable edit (or one undo/redo step).

Values given as key=value are parsed as JSON when possible (numbers, true/false, lists),
otherwise taken as strings:  storyreel set film.storyreel 3 x=120 scale=1.2 reverse=true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from storyreel.errors import ExitCode, StoryReelError
from storyreel.project import store
from storyreel.project.edits import Editor
from storyreel.render import fxspec
from storyreel.timebase import FLICKS, seconds_to_flicks


def parse_kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise StoryReelError(f"Expected key=value, got {item!r}.")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _video_track(p: store.Project, ed: Editor, idx: int) -> int:
    while True:
        row = p.con.execute(
            "SELECT id FROM track WHERE kind = 'video' AND idx = ?", (idx,)
        ).fetchone()
        if row is not None:
            return int(row["id"])
        ed.add_track("video")


def _open(a: argparse.Namespace) -> tuple[store.Project, Editor]:
    p = store.open_project(Path(a.project))
    return p, Editor(p.con)


def _snap(p: store.Project, seconds: float) -> int:
    rate = p.settings().rate
    return rate.frame_to_flicks(rate.flicks_to_frame(seconds_to_flicks(seconds)))


def cmd_text(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        props: dict[str, Any] = {
            "text": a.text.replace("\\n", "\n"),
            "size": a.size,
            "anim": a.anim,
            "anim_unit": a.unit,
            "y": a.y,
        }
        props |= parse_kv(a.style)
        cid = ed.add_clip(
            _video_track(p, ed, a.track), _snap(p, a.at), _snap(p, a.dur), kind="text", props=props
        )
    print(f"Added text clip #{cid}")
    return int(ExitCode.OK)


def cmd_env(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        props = {"type": a.type, **parse_kv(a.params)}
        cid = ed.add_clip(
            _video_track(p, ed, a.track),
            _snap(p, a.at),
            _snap(p, a.dur),
            kind="generator",
            props=props,
        )
    print(f"Added {a.type} clip #{cid}")
    return int(ExitCode.OK)


def cmd_adjust(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        cid = ed.add_clip(
            _video_track(p, ed, a.track), _snap(p, a.at), _snap(p, a.dur), kind="adjust"
        )
        ids = ed.apply_color_preset(cid, a.preset) if a.preset else []
    print(f"Added adjustment clip #{cid}" + (f" with preset effects {ids}" if ids else ""))
    return int(ExitCode.OK)


def cmd_fx(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        if a.preset:
            ids = ed.apply_color_preset(a.clip, a.preset)
            print(f"Applied preset {a.preset}: effects {ids}")
        else:
            if not a.type:
                raise StoryReelError(
                    "Give an effect type or --preset.",
                    fix="Effects: " + ", ".join(sorted(fxspec.EFFECTS)),
                )
            eid = ed.add_effect(a.clip, a.type, **parse_kv(a.params))
            print(f"Added {a.type} effect #{eid} (animate with keyframe param fx{eid}.<name>)")
    return int(ExitCode.OK)


def cmd_kenburns(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        ed.ken_burns(
            a.clip,
            zoom_from=a.zoom_from,
            zoom_to=a.zoom_to,
            pan_x=a.pan_x,
            pan_y=a.pan_y,
            ease=a.ease,
        )
    print(f"Ken Burns on clip #{a.clip}")
    return int(ExitCode.OK)


def cmd_keyframe(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        interp = "hold" if a.ease == "hold" else ("linear" if a.ease == "linear" else "bezier")
        ease = a.ease if a.ease in fxspec.EASINGS else "linear"
        ed.add_keyframe(
            a.clip, a.param, seconds_to_flicks(a.time), a.value, interp=interp, ease=ease
        )
    print(f"Keyframe {a.param}={a.value} at {a.time:.3f} s on clip #{a.clip}")
    return int(ExitCode.OK)


def cmd_transition(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        tid = ed.add_transition(a.clip_a, a.clip_b, a.kind, seconds_to_flicks(a.dur))
    print(f"Added {a.kind} transition #{tid}")
    return int(ExitCode.OK)


def cmd_set(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        ed.set_props(a.clip, **parse_kv(a.props))
    print(f"Updated clip #{a.clip}")
    return int(ExitCode.OK)


def cmd_speed(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        ed.set_speed(a.clip, a.factor, keep_duration=a.keep_duration)
        dur = p.con.execute("SELECT duration FROM clip WHERE id = ?", (a.clip,)).fetchone()[0]
    print(f"Clip #{a.clip} now plays at {a.factor:g}x ({dur / FLICKS:.3f} s on the timeline)")
    return int(ExitCode.OK)


def cmd_undo(a: argparse.Namespace) -> int:
    p, ed = _open(a)
    with p:
        label = ed.redo() if a.command == "redo" else ed.undo()
    print(f"{a.command.capitalize()}: {label}" if label else f"Nothing to {a.command}.")
    return int(ExitCode.OK)


def cmd_frame(a: argparse.Namespace) -> int:
    import cv2

    from storyreel.cli_project import _tools
    from storyreel.render.plan import RenderSettings
    from storyreel.render.renderer import render_frame

    ff, _ = _tools()
    with store.open_project(Path(a.project)) as p:
        ps = p.settings()
        s = RenderSettings(ps.width, ps.height, ps.rate)
        img = render_frame(p, s, ff, ps.rate.flicks_to_frame(seconds_to_flicks(a.at)))
    out = Path(a.output)
    if not cv2.imwrite(str(out), img[..., ::-1]):
        raise StoryReelError(f"Could not write {out}.", fix="Use a .png or .jpg file name.")
    print(f"Wrote {out} ({img.shape[1]}x{img.shape[0]}, t={a.at:.3f} s)")
    return int(ExitCode.OK)


def register(sub: Any) -> None:
    t = sub.add_parser("text", help="add an animated text/title clip")
    t.add_argument("project")
    t.add_argument("text", help="use \\n for a new line; any script (Hindi, Tamil, ...)")
    t.add_argument("--at", type=float, default=0.0)
    t.add_argument("--dur", type=float, default=4.0)
    t.add_argument("--size", type=float, default=72.0)
    t.add_argument("--y", type=float, default=0.0, help="pixels from the centre (+ = down)")
    t.add_argument("--anim", default="none", choices=sorted(fxspec.TEXT_ANIMS))
    t.add_argument("--unit", default="letter", choices=["letter", "word"])
    t.add_argument("--track", type=int, default=2, help="video track index (created if needed)")
    t.add_argument(
        "style", nargs="*", help="style key=value, e.g. outline=3 color=#ffcc00 box=true"
    )
    t.set_defaults(func=cmd_text)

    e = sub.add_parser("env", help="add a procedural environment (rain, snow, fog, fire, ...)")
    e.add_argument("project")
    e.add_argument("type", choices=sorted(fxspec.GENERATORS))
    e.add_argument("--at", type=float, default=0.0)
    e.add_argument("--dur", type=float, default=5.0)
    e.add_argument("--track", type=int, default=1)
    e.add_argument("params", nargs="*", help="key=value, e.g. count=1200 opacity=0.4")
    e.set_defaults(func=cmd_env)

    ad = sub.add_parser("adjust", help="add an adjustment clip (effects apply to everything below)")
    ad.add_argument("project")
    ad.add_argument("--at", type=float, default=0.0)
    ad.add_argument("--dur", type=float, default=5.0)
    ad.add_argument("--track", type=int, default=3)
    ad.add_argument("--preset", choices=sorted(fxspec.COLOR_PRESETS))
    ad.set_defaults(func=cmd_adjust)

    f = sub.add_parser("fx", help="add an effect or a colour preset to a clip")
    f.add_argument("project")
    f.add_argument("clip", type=int)
    f.add_argument("type", nargs="?", choices=[*sorted(fxspec.EFFECTS), None])
    f.add_argument("params", nargs="*")
    f.add_argument("--preset", choices=sorted(fxspec.COLOR_PRESETS))
    f.set_defaults(func=cmd_fx)

    k = sub.add_parser("kenburns", help="slow zoom/pan over a clip")
    k.add_argument("project")
    k.add_argument("clip", type=int)
    k.add_argument("--zoom-from", type=float, default=1.0)
    k.add_argument("--zoom-to", type=float, default=1.2)
    k.add_argument("--pan-x", type=float, default=0.0)
    k.add_argument("--pan-y", type=float, default=0.0)
    k.add_argument("--ease", default="ease_in_out", choices=sorted(fxspec.EASINGS))
    k.set_defaults(func=cmd_kenburns)

    kf = sub.add_parser(
        "keyframe", help="animate a parameter: x y scale rotation opacity speed fx<N>.<p>"
    )
    kf.add_argument("project")
    kf.add_argument("clip", type=int)
    kf.add_argument("param")
    kf.add_argument("time", type=float, help="seconds from the clip start")
    kf.add_argument("value", type=float)
    kf.add_argument("--ease", default="linear", choices=[*sorted(fxspec.EASINGS), "hold"])
    kf.set_defaults(func=cmd_keyframe)

    tr = sub.add_parser("transition", help="transition at the cut between two adjacent clips")
    tr.add_argument("project")
    tr.add_argument("clip_a", type=int)
    tr.add_argument("clip_b", type=int)
    tr.add_argument("kind", choices=sorted(fxspec.TRANSITIONS))
    tr.add_argument("--dur", type=float, default=1.0)
    tr.set_defaults(func=cmd_transition)

    st = sub.add_parser(
        "set",
        help="set clip properties: x y scale rotation opacity crop_* reverse freeze blend, style",
    )
    st.add_argument("project")
    st.add_argument("clip", type=int)
    st.add_argument("props", nargs="+")
    st.set_defaults(func=cmd_set)

    sp = sub.add_parser("speed", help="constant speed 0.1x-16x")
    sp.add_argument("project")
    sp.add_argument("clip", type=int)
    sp.add_argument("factor", type=float)
    sp.add_argument("--keep-duration", action="store_true")
    sp.set_defaults(func=cmd_speed)

    fr = sub.add_parser("frame", help="render one frame to an image (preview)")
    fr.add_argument("project")
    fr.add_argument("--at", type=float, required=True)
    fr.add_argument("-o", "--output", default="frame.png")
    fr.set_defaults(func=cmd_frame)

    for name in ("undo", "redo"):
        u = sub.add_parser(name, help=f"{name} the last edit")
        u.add_argument("project")
        u.set_defaults(func=cmd_undo)


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use: storyreel <command>")
