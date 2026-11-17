"""Timeline edits as undoable commands.

A command is a list of row operations (insert / update / delete) on whitelisted tables.
The inverse is derived mechanically, so every edit is undoable without full-copy snapshots.
Each command (and each undo/redo) is one SQLite transaction. Compound edits (Ken Burns,
color presets, deleting a clip with its keyframes) are one command, so one undo step.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from fractions import Fraction
from typing import Any

from storyreel.audio import spec as aspec
from storyreel.errors import StoryReelError
from storyreel.render import fxspec
from storyreel.timebase import FLICKS

UNDO_KEEP = 200  # spec asks for 50; commands are tiny, keep more
_TABLES = frozenset(
    {"track", "clip", "keyframe", "effect", "transition", "bus", "bus_effect", "project"}
)
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
VISUAL_KINDS = frozenset({"color", "text", "generator", "adjust"})

Row = dict[str, Any]
Op = dict[str, Any]


class EditError(StoryReelError):
    pass


def _cols(con: sqlite3.Connection, table: str) -> set[str]:
    if table not in _TABLES:
        raise EditError(f"internal: table {table!r} is not editable")
    return {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}


def _apply(con: sqlite3.Connection, op: Op) -> None:
    table = op["table"]
    cols = _cols(con, table)
    kind = op["op"]
    if kind == "insert":
        row: Row = op["row"]
        if not set(row) <= cols:
            raise EditError(f"internal: unknown columns {set(row) - cols}")
        names = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        con.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", tuple(row.values()))
    elif kind == "delete":
        con.execute(f"DELETE FROM {table} WHERE id = ?", (op["row"]["id"],))
    elif kind == "update":
        after: Row = op["after"]
        if not set(after) <= cols:
            raise EditError(f"internal: unknown columns {set(after) - cols}")
        sets = ", ".join(f"{c} = ?" for c in after)
        con.execute(f"UPDATE {table} SET {sets} WHERE id = ?", (*after.values(), op["id"]))
    else:
        raise EditError(f"internal: bad op {kind!r}")


def _inverse(op: Op) -> Op:
    kind = op["op"]
    if kind == "insert":
        return {"op": "delete", "table": op["table"], "row": op["row"]}
    if kind == "delete":
        return {"op": "insert", "table": op["table"], "row": op["row"]}
    return {
        "op": "update",
        "table": op["table"],
        "id": op["id"],
        "before": op["after"],
        "after": op["before"],
    }


def _check_value(name: str, value: Any, default: Any) -> None:
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int | float):
        ok = isinstance(value, int | float) and not isinstance(value, bool)
    elif isinstance(default, str):
        ok = isinstance(value, str)
    else:
        ok = isinstance(value, list)
    if not ok:
        raise EditError(f"Parameter {name} must be a {type(default).__name__}.")
    if isinstance(default, str) and default.startswith("#"):
        many = "," in default  # a list of colours, e.g. flower_colors
        parts = str(value).split(",") if many else [str(value)]
        if not all(_HEX.match(c.strip()) for c in parts):
            example = "#ffffff,#ffd84a" if many else "#1a2b3c"
            raise EditError(f"Parameter {name} must be a color like {example}.")


def _check_range(name: str, value: Any, rng: tuple[float, float] | None) -> None:
    if rng is not None and isinstance(value, int | float) and not isinstance(value, bool):
        lo, hi = rng
        if not lo <= float(value) <= hi:
            raise EditError(f"{name} must be between {lo:g} and {hi:g} (got {value:g}).")


def _check_audio_effect(etype: str, params: dict[str, Any]) -> None:
    spec = aspec.EFFECTS.get(etype)
    if spec is None:
        raise EditError(f"Unknown audio effect {etype!r}. Available: {sorted(aspec.EFFECTS)}.")
    for k, v in params.items():
        if k not in spec:
            raise EditError(f"Audio effect {etype} has no parameter {k!r}.")
        _check_value(k, v, spec[k])
        _check_range(f"{etype}.{k}", v, aspec.EFFECT_RANGES.get((etype, k)))
    room = params.get("room")
    if etype == "reverb" and room is not None and room not in aspec.ROOMS:
        raise EditError(f"Reverb room must be one of {sorted(aspec.ROOMS)}.")


class Editor:
    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con
        self._pending_ids: dict[str, int] = {}

    # -- command machinery -------------------------------------------------------------
    def _commit(self, label: str, ops: list[Op]) -> None:
        con = self.con
        self._pending_ids.clear()
        con.execute("BEGIN IMMEDIATE")
        try:
            for op in ops:
                _apply(con, op)
            con.execute("DELETE FROM command WHERE undone = 1")  # new edit clears redo
            con.execute(
                "INSERT INTO command(label, ops, created_at) VALUES (?, ?, ?)",
                (label, json.dumps(ops), time.time()),
            )
            con.execute(
                "DELETE FROM command WHERE id <= (SELECT max(id) FROM command) - ?", (UNDO_KEEP,)
            )
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise

    def undo(self) -> str | None:
        """Undo the newest command. Returns its label, or None if nothing to undo."""
        return self._step(undo=True)

    def redo(self) -> str | None:
        return self._step(undo=False)

    def _step(self, *, undo: bool) -> str | None:
        con = self.con
        con.execute("BEGIN IMMEDIATE")
        try:
            row = con.execute(
                "SELECT id, label, ops FROM command WHERE undone = 0 ORDER BY id DESC LIMIT 1"
                if undo
                else "SELECT id, label, ops FROM command WHERE undone = 1 ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                con.execute("ROLLBACK")
                return None
            ops: list[Op] = json.loads(row["ops"])
            seq = [_inverse(o) for o in reversed(ops)] if undo else ops
            for op in seq:
                _apply(con, op)
            con.execute("UPDATE command SET undone = ? WHERE id = ?", (1 if undo else 0, row["id"]))
            con.execute("COMMIT")
            return str(row["label"])
        except BaseException:
            con.execute("ROLLBACK")
            raise

    # -- helpers -----------------------------------------------------------------------
    def _row(self, table: str, row_id: int) -> Row:
        _cols(self.con, table)
        r = self.con.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        if r is None:
            raise EditError(f"{table} {row_id} does not exist.")
        return dict(r)

    def _next_id(self, table: str) -> int:
        """Next id, accounting for inserts already queued in the command being built."""
        _cols(self.con, table)
        db_next = int(
            self.con.execute(f"SELECT coalesce(max(id), 0) + 1 FROM {table}").fetchone()[0]
        )
        nid = max(db_next, self._pending_ids.get(table, 0) + 1)
        self._pending_ids[table] = nid
        return nid

    def _check_free(self, track_id: int, start: int, duration: int, ignore: int | None) -> None:
        hit = self.con.execute(
            "SELECT id FROM clip WHERE track_id = ? AND id IS NOT ? "
            "AND start < ? AND start + duration > ? LIMIT 1",
            (track_id, ignore, start + duration, start),
        ).fetchone()
        if hit is not None:
            raise EditError(f"That position overlaps clip {hit['id']} on the same track.")

    def _check_media(self, track: Row, clip: Row) -> None:
        if clip["kind"] in VISUAL_KINDS:
            if track["kind"] != "video":
                raise EditError(f"{clip['kind'].capitalize()} clips go on video tracks.")
            return
        asset = self.con.execute("SELECT * FROM asset WHERE id = ?", (clip["asset_id"],)).fetchone()
        if asset is None:
            raise EditError(f"Asset {clip['asset_id']} does not exist.")
        if asset["status"] != "ok":
            raise EditError(f"Asset {asset['name']} is {asset['status']}: {asset['error'] or ''}")
        if track["kind"] == "video" and not asset["has_video"]:
            raise EditError(f"{asset['name']} has no picture; put it on an audio track.")
        if track["kind"] == "audio" and not asset["has_audio"]:
            raise EditError(f"{asset['name']} has no sound; put it on a video track.")
        speed = Fraction(clip["speed_num"], clip["speed_den"])
        if not Fraction(1, 10) <= speed <= 16:
            raise EditError("Speed must be between 0.1x and 16x.")
        props = json.loads(clip["props"]) if isinstance(clip["props"], str) else clip["props"]
        span = 0 if props.get("freeze") else int(clip["duration"] * speed)
        if asset["duration"] is not None and clip["src_in"] + span > asset["duration"]:
            raise EditError(
                f"Clip needs {span / FLICKS:.3f} s of source from {clip['src_in'] / FLICKS:.3f} s, "
                f"but {asset['name']} is {asset['duration'] / FLICKS:.3f} s long."
            )

    def _clip_children_ops(self, clip_id: int) -> list[Op]:
        ops: list[Op] = []
        for table, where in (
            ("keyframe", "clip_id = ?"),
            ("effect", "clip_id = ?"),
            ("transition", "clip_a = ? OR clip_b = ?"),
        ):
            args = (clip_id,) * where.count("?")
            for r in self.con.execute(f"SELECT * FROM {table} WHERE {where}", args):
                ops.append({"op": "delete", "table": table, "row": dict(r)})
        return ops

    # -- tracks and clips --------------------------------------------------------------
    def add_track(self, kind: str, name: str | None = None) -> int:
        idx = int(
            self.con.execute(
                "SELECT coalesce(max(idx), -1) + 1 FROM track WHERE kind = ?", (kind,)
            ).fetchone()[0]
        )
        tid = self._next_id("track")
        name = name or f"{'V' if kind == 'video' else 'A'}{idx + 1}"
        row = {"id": tid, "kind": kind, "idx": idx, "name": name, "muted": 0}
        self._commit(f"Add track {name}", [{"op": "insert", "table": "track", "row": row}])
        return tid

    def add_clip(
        self,
        track_id: int,
        start: int,
        duration: int,
        *,
        asset_id: int | None = None,
        src_in: int = 0,
        color: str | None = None,
        gain_db: float = 0.0,
        kind: str | None = None,
        props: dict[str, Any] | None = None,
    ) -> int:
        """Add a clip. kind defaults to 'color' if color is given, else 'media'.

        text:      props = {"text": "...", ...TEXT_DEFAULTS}
        generator: props = {"type": "rain", ...GENERATORS[type]}
        adjust:    no props; effects added to it apply to everything below.
        """
        track = self._row("track", track_id)
        kind = kind or ("color" if color is not None else "media")
        props = dict(props or {})
        if color is not None:
            if not _HEX.match(color):
                raise EditError("Color must look like #1a2b3c.")
            props["color"] = color
        self._validate_props(kind, props, audio=track["kind"] == "audio", duration=duration)
        if kind == "media" and asset_id is None:
            raise EditError("A media clip needs an asset.")
        if kind != "media" and asset_id is not None:
            raise EditError(f"A {kind} clip cannot use an asset.")
        if duration <= 0 or start < 0 or src_in < 0:
            raise EditError("Start and source-in must be >= 0 and duration > 0.")
        row: Row = {
            "id": self._next_id("clip"),
            "track_id": track_id,
            "kind": kind,
            "asset_id": asset_id,
            "start": start,
            "duration": duration,
            "src_in": src_in,
            "speed_num": 1,
            "speed_den": 1,
            "gain_db": gain_db,
            "props": json.dumps(props),
        }
        self._check_media(track, row)
        self._check_free(track_id, start, duration, None)
        self._commit(
            f"Add {kind} clip to {track['name']}", [{"op": "insert", "table": "clip", "row": row}]
        )
        return int(row["id"])

    def _validate_props(
        self, kind: str, props: dict[str, Any], *, audio: bool = False, duration: int = 0
    ) -> None:
        allowed: dict[str, Any] = {
            k: v for k, v in fxspec.TRANSFORM_DEFAULTS.items() if k != "speed"
        }
        allowed |= {"reverse": False, "freeze": False, "blend": False, "color": "#000000"}
        if audio:
            allowed |= aspec.CLIP_PROPS
            for k, rng in aspec.CLIP_RANGES.items():
                if k in props and props[k] is not None:
                    _check_value(k, props[k], aspec.CLIP_PROPS[k])
                    _check_range(k, props[k], rng)
            fades = float(props.get("fade_in") or 0.0) + float(props.get("fade_out") or 0.0)
            if duration and fades > duration / FLICKS + 1e-9:
                raise EditError(
                    f"Fade in + fade out ({fades:g} s) is longer than the clip "
                    f"({duration / FLICKS:.3f} s)."
                )
        if kind == "text":
            allowed |= fxspec.TEXT_DEFAULTS
            if not str(props.get("text", "")).strip():
                raise EditError("A text clip needs some text.")
            if props.get("anim", "none") not in fxspec.TEXT_ANIMS:
                raise EditError(f"Text animation must be one of {sorted(fxspec.TEXT_ANIMS)}.")
        if kind == "generator":
            gtype = props.get("type")
            if gtype not in fxspec.GENERATORS:
                raise EditError(f"Generator type must be one of {sorted(fxspec.GENERATORS)}.")
            allowed |= {"type": ""} | fxspec.GENERATORS[gtype]
        for k, v in props.items():
            if k not in allowed:
                raise EditError(f"Unknown property {k!r} for a {kind} clip.")
            if v is not None:
                _check_value(k, v, allowed[k])

    def update_clip(self, clip_id: int, label: str, **changes: Any) -> None:
        """Move/trim/regain/retime a clip.

        Allowed keys: track_id, start, duration, src_in, gain_db, speed_num, speed_den.
        """
        allowed = {"track_id", "start", "duration", "src_in", "gain_db", "speed_num", "speed_den"}
        if not set(changes) <= allowed:
            raise EditError(f"Cannot change {set(changes) - allowed} here.")
        before_full = self._row("clip", clip_id)
        after_full = {**before_full, **changes}
        if after_full["duration"] <= 0 or after_full["start"] < 0 or after_full["src_in"] < 0:
            raise EditError("Start and source-in must be >= 0 and duration > 0.")
        track = self._row("track", after_full["track_id"])
        self._check_media(track, after_full)
        self._check_free(track["id"], after_full["start"], after_full["duration"], clip_id)
        before = {k: before_full[k] for k in changes}
        op = {"op": "update", "table": "clip", "id": clip_id, "before": before, "after": changes}
        self._commit(label, [op])

    def move_clip(self, clip_id: int, start: int, track_id: int | None = None) -> None:
        changes: dict[str, Any] = {"start": start}
        if track_id is not None:
            changes["track_id"] = track_id
        self.update_clip(clip_id, "Move clip", **changes)

    def set_speed(
        self, clip_id: int, speed: Fraction | float, *, keep_duration: bool = False
    ) -> None:
        """Constant speed. By default the clip's timeline length changes so the same source
        span plays faster/slower (like most editors); keep_duration keeps the length."""
        f = Fraction(speed).limit_denominator(1000)
        clip = self._row("clip", clip_id)
        old = Fraction(clip["speed_num"], clip["speed_den"])
        changes: dict[str, Any] = {"speed_num": f.numerator, "speed_den": f.denominator}
        if not keep_duration:
            changes["duration"] = int(clip["duration"] * old / f)
        self.update_clip(clip_id, f"Speed {float(f):g}x", **changes)

    def set_props(self, clip_id: int, label: str = "Change properties", **props: Any) -> None:
        """Merge props into the clip's JSON props. A value of None removes the key."""
        clip = self._row("clip", clip_id)
        old = json.loads(clip["props"])
        new = {**old, **props}
        new = {k: v for k, v in new.items() if v is not None}
        track = self._row("track", clip["track_id"])
        self._validate_props(
            clip["kind"], new, audio=track["kind"] == "audio", duration=clip["duration"]
        )
        trial = {**clip, "props": json.dumps(new)}
        self._check_media(track, trial)
        op = {
            "op": "update",
            "table": "clip",
            "id": clip_id,
            "before": {"props": clip["props"]},
            "after": {"props": json.dumps(new)},
        }
        self._commit(label, [op])

    def delete_clip(self, clip_id: int) -> None:
        row = self._row("clip", clip_id)
        ops = self._clip_children_ops(clip_id)
        ops.append({"op": "delete", "table": "clip", "row": row})
        self._commit("Delete clip", ops)

    # -- keyframes ---------------------------------------------------------------------
    def _effect_types(self, clip_id: int) -> dict[int, str]:
        rows = self.con.execute("SELECT id, type FROM effect WHERE clip_id = ?", (clip_id,))
        return {int(r["id"]): str(r["type"]) for r in rows}

    def _keyframe_op(
        self, clip: Row, param: str, t: int, value: float, interp: str, ease: str
    ) -> Op:
        audio = self._row("track", clip["track_id"])["kind"] == "audio"
        if not fxspec.keyframable_param(param, self._effect_types(clip["id"]), audio=audio):
            hint = " Audio clips animate `volume` (dB)." if audio else ""
            raise EditError(f"{param!r} cannot be animated on this clip.{hint}")
        if param == "volume":
            _check_range("volume", value, aspec.VOLUME_RANGE)
        if interp not in ("hold", "linear", "bezier"):
            raise EditError("Interpolation must be hold, linear or bezier.")
        if ease not in fxspec.EASINGS:
            raise EditError(f"Easing must be one of {sorted(fxspec.EASINGS)}.")
        if not 0 <= t <= clip["duration"]:
            raise EditError("Keyframe time must be within the clip.")
        h = fxspec.EASINGS[ease]
        existing = self.con.execute(
            "SELECT * FROM keyframe WHERE clip_id = ? AND param = ? AND t = ?",
            (clip["id"], param, t),
        ).fetchone()
        vals = {
            "value": float(value),
            "interp": interp,
            "h1x": h[0],
            "h1y": h[1],
            "h2x": h[2],
            "h2y": h[3],
        }
        if existing is not None:
            before = {k: existing[k] for k in vals}
            return {
                "op": "update",
                "table": "keyframe",
                "id": existing["id"],
                "before": before,
                "after": vals,
            }
        row = {
            "id": self._next_id("keyframe"),
            "clip_id": clip["id"],
            "param": param,
            "t": t,
            **vals,
        }
        return {"op": "insert", "table": "keyframe", "row": row}

    def add_keyframe(
        self,
        clip_id: int,
        param: str,
        t: int,
        value: float,
        *,
        interp: str = "linear",
        ease: str = "ease_in_out",
    ) -> None:
        clip = self._row("clip", clip_id)
        self._commit(f"Keyframe {param}", [self._keyframe_op(clip, param, t, value, interp, ease)])

    def remove_keyframe(self, keyframe_id: int) -> None:
        row = self._row("keyframe", keyframe_id)
        self._commit(
            f"Remove keyframe {row['param']}", [{"op": "delete", "table": "keyframe", "row": row}]
        )

    def ken_burns(
        self,
        clip_id: int,
        *,
        zoom_from: float = 1.0,
        zoom_to: float = 1.2,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
        ease: str = "ease_in_out",
    ) -> None:
        """Slow zoom + pan over the whole clip, as one undo step."""
        clip = self._row("clip", clip_id)
        end = clip["duration"]
        interp = "linear" if ease == "linear" else "bezier"
        ops = [
            self._keyframe_op(clip, "scale", 0, zoom_from, interp, ease),
            self._keyframe_op(clip, "scale", end, zoom_to, interp, ease),
        ]
        if pan_x or pan_y:
            ops += [
                self._keyframe_op(clip, "x", 0, 0.0, interp, ease),
                self._keyframe_op(clip, "x", end, pan_x, interp, ease),
                self._keyframe_op(clip, "y", 0, 0.0, interp, ease),
                self._keyframe_op(clip, "y", end, pan_y, interp, ease),
            ]
        self._commit("Ken Burns", ops)

    # -- effects -----------------------------------------------------------------------
    def _effect_row(self, clip_id: int, etype: str, params: dict[str, Any], idx: int) -> Row:
        clip = self._row("clip", clip_id)
        if self._row("track", clip["track_id"])["kind"] == "audio":
            if etype in fxspec.EFFECTS:
                raise EditError(f"{etype} is a video effect; this clip is on an audio track.")
            _check_audio_effect(etype, params)
        else:
            spec = fxspec.EFFECTS.get(etype)
            if spec is None:
                hint = " (that is an audio effect)" if etype in aspec.EFFECTS else ""
                raise EditError(
                    f"Unknown video effect {etype!r}{hint}. Available: {sorted(fxspec.EFFECTS)}."
                )
            for k, v in params.items():
                if k not in spec:
                    raise EditError(f"Effect {etype} has no parameter {k!r}.")
                _check_value(k, v, spec[k])
        return {
            "id": self._next_id("effect"),
            "clip_id": clip_id,
            "idx": idx,
            "type": etype,
            "params": json.dumps(params),
            "enabled": 1,
        }

    def _next_effect_idx(self, clip_id: int) -> int:
        return int(
            self.con.execute(
                "SELECT coalesce(max(idx), -1) + 1 FROM effect WHERE clip_id = ?", (clip_id,)
            ).fetchone()[0]
        )

    def add_effect(self, clip_id: int, etype: str, **params: Any) -> int:
        self._row("clip", clip_id)
        row = self._effect_row(clip_id, etype, params, self._next_effect_idx(clip_id))
        self._commit(f"Add {etype}", [{"op": "insert", "table": "effect", "row": row}])
        return int(row["id"])

    def apply_color_preset(self, clip_id: int, preset: str) -> list[int]:
        if preset not in fxspec.COLOR_PRESETS:
            raise EditError(
                f"Unknown preset {preset!r}. Available: {sorted(fxspec.COLOR_PRESETS)}."
            )
        self._row("clip", clip_id)
        idx = self._next_effect_idx(clip_id)
        rows = [
            self._effect_row(clip_id, t, p, idx + i)
            for i, (t, p) in enumerate(fxspec.COLOR_PRESETS[preset])
        ]
        self._commit(
            f"Preset {preset}", [{"op": "insert", "table": "effect", "row": r} for r in rows]
        )
        return [int(r["id"]) for r in rows]

    def update_effect(self, effect_id: int, **params: Any) -> None:
        eff = self._row("effect", effect_id)
        merged = {**json.loads(eff["params"]), **params}
        self._effect_row(eff["clip_id"], eff["type"], merged, eff["idx"])  # validation only
        self._commit(
            f"Change {eff['type']}",
            [
                {
                    "op": "update",
                    "table": "effect",
                    "id": effect_id,
                    "before": {"params": eff["params"]},
                    "after": {"params": json.dumps(merged)},
                }
            ],
        )

    def remove_effect(self, effect_id: int) -> None:
        eff = self._row("effect", effect_id)
        ops = [
            {"op": "delete", "table": "keyframe", "row": dict(r)}
            for r in self.con.execute(
                "SELECT * FROM keyframe WHERE clip_id = ? AND param LIKE ?",
                (eff["clip_id"], f"fx{effect_id}.%"),
            )
        ]
        ops.append({"op": "delete", "table": "effect", "row": eff})
        self._commit(f"Remove {eff['type']}", ops)

    # -- transitions -------------------------------------------------------------------
    def add_transition(self, clip_a: int, clip_b: int, kind: str, duration: int) -> int:
        """Transition centred on the cut between A (ends) and B (starts) on one track.

        Frames outside a clip's source range are held (freeze) instead of failing.
        """
        if kind not in fxspec.TRANSITIONS:
            raise EditError(f"Transition must be one of {sorted(fxspec.TRANSITIONS)}.")
        a, b = self._row("clip", clip_a), self._row("clip", clip_b)
        if a["track_id"] != b["track_id"]:
            raise EditError("A transition joins two clips on the same track.")
        if a["start"] + a["duration"] != b["start"]:
            raise EditError("Clip B must start exactly where clip A ends.")
        if not 0 < duration <= min(a["duration"], b["duration"]):
            raise EditError("Transition must be shorter than both clips.")
        taken = self.con.execute(
            "SELECT id FROM transition WHERE clip_a = ? OR clip_b = ?", (clip_a, clip_b)
        ).fetchone()
        if taken is not None:
            raise EditError("One of these clips already has a transition on that side.")
        row: Row = {
            "id": self._next_id("transition"),
            "clip_a": clip_a,
            "clip_b": clip_b,
            "kind": kind,
            "duration": duration,
            "params": "{}",
        }
        self._commit(f"Add {kind}", [{"op": "insert", "table": "transition", "row": row}])
        return int(row["id"])

    # -- audio: tracks, buses, project settings ---------------------------------------
    def _bus(self, name: str) -> Row:
        r = self.con.execute("SELECT * FROM bus WHERE name = ?", (name,)).fetchone()
        if r is None:
            raise EditError(f"Bus must be one of {list(aspec.ALL_BUSES)}.")
        return dict(r)

    def set_track(
        self, track_id: int, *, bus: str | None = None, muted: bool | None = None
    ) -> None:
        track = self._row("track", track_id)
        after: Row = {}
        if bus is not None:
            if track["kind"] != "audio":
                raise EditError("Only audio tracks are routed to a bus.")
            if bus not in aspec.BUSES:
                raise EditError(
                    f"A track goes to one of {list(aspec.BUSES)} (master is the sum of them)."
                )
            after["bus"] = bus
        if muted is not None:
            after["muted"] = int(muted)
        if not after:
            raise EditError("Nothing to change: give bus= or muted=.")
        before = {k: track[k] for k in after}
        op = {"op": "update", "table": "track", "id": track_id, "before": before, "after": after}
        self._commit(f"Track {track['name']}", [op])

    def set_bus(self, name: str, **changes: Any) -> None:
        """gain_db, muted, duck, duck_db, duck_attack, duck_release."""
        bus = self._bus(name)
        params = json.loads(bus["params"])
        cols: Row = {}
        for k, v in changes.items():
            if k == "gain_db":
                _check_value(k, v, 0.0)
                _check_range(k, v, aspec.BUS_RANGES[k])
                cols[k] = float(v)
            elif k == "muted":
                _check_value(k, v, False)
                cols[k] = int(v)
            elif k in aspec.BUS_PARAMS:
                if name not in aspec.DUCKABLE:
                    raise EditError(f"Only {sorted(aspec.DUCKABLE)} buses duck under dialogue.")
                _check_value(k, v, aspec.BUS_PARAMS[k])
                _check_range(k, v, aspec.BUS_RANGES.get(k))
                params[k] = v
            else:
                raise EditError(
                    f"Bus setting must be one of gain_db, muted, {', '.join(aspec.BUS_PARAMS)}."
                )
        if json.dumps(params) != bus["params"]:
            cols["params"] = json.dumps(params)
        if not cols:
            return
        before = {k: bus[k] for k in cols}
        op = {"op": "update", "table": "bus", "id": bus["id"], "before": before, "after": cols}
        self._commit(f"Bus {name}", [op])

    def add_bus_effect(self, name: str, etype: str, **params: Any) -> int:
        bus = self._bus(name)
        _check_audio_effect(etype, params)
        idx = int(
            self.con.execute(
                "SELECT coalesce(max(idx), -1) + 1 FROM bus_effect WHERE bus_id = ?", (bus["id"],)
            ).fetchone()[0]
        )
        row: Row = {
            "id": self._next_id("bus_effect"),
            "bus_id": bus["id"],
            "idx": idx,
            "type": etype,
            "params": json.dumps(params),
            "enabled": 1,
        }
        self._commit(
            f"Add {etype} to {name}", [{"op": "insert", "table": "bus_effect", "row": row}]
        )
        return int(row["id"])

    def remove_bus_effect(self, effect_id: int) -> None:
        row = self._row("bus_effect", effect_id)
        self._commit(f"Remove {row['type']}", [{"op": "delete", "table": "bus_effect", "row": row}])

    def set_audio(self, **changes: Any) -> None:
        """Project audio: layout (mono/stereo/5.1/7.1), target_lufs, true_peak, normalize."""
        proj = self._row("project", 1)
        audio = json.loads(proj["audio"])
        cols: Row = {}
        for k, v in changes.items():
            if k == "layout":
                if isinstance(v, int | float) and not isinstance(v, bool):
                    v = f"{v:g}"  # `layout=5.1` arrives from the CLI as a number
                if v not in aspec.LAYOUT_NAMES:
                    raise EditError(f"Layout must be one of {list(aspec.LAYOUT_NAMES)}.")
                cols["channels"] = aspec.LAYOUT_NAMES[v]
            elif k == "target_lufs" and isinstance(v, str):
                if v not in aspec.LOUDNESS_PRESETS:
                    raise EditError(
                        f"target_lufs is a number or one of {sorted(aspec.LOUDNESS_PRESETS)}."
                    )
                audio[k] = aspec.LOUDNESS_PRESETS[v]
            elif k in aspec.LOUDNESS_DEFAULTS:
                _check_value(k, v, aspec.LOUDNESS_DEFAULTS[k])
                _check_range(k, v, aspec.LOUDNESS_RANGES.get(k))
                audio[k] = v
            else:
                raise EditError(
                    f"Audio setting must be one of layout, {', '.join(aspec.LOUDNESS_DEFAULTS)}."
                )
        if json.dumps(audio) != proj["audio"]:
            cols["audio"] = json.dumps(audio)
        if not cols:
            return
        before = {k: proj[k] for k in cols}
        op = {"op": "update", "table": "project", "id": 1, "before": before, "after": cols}
        self._commit("Audio settings", [op])
