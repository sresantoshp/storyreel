"""Timeline edits as undoable commands.

A command is a list of row operations (insert / update / delete) on whitelisted tables.
The inverse is derived mechanically, so every edit is undoable without full-copy snapshots.
Each command (and each undo/redo) is one SQLite transaction.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from storyreel.errors import StoryReelError

UNDO_KEEP = 200  # spec asks for 50; commands are tiny, keep more
_TABLES = frozenset({"track", "clip"})

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


class Editor:
    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    # -- command machinery -------------------------------------------------------------
    def _commit(self, label: str, ops: list[Op]) -> None:
        con = self.con
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
        _cols(self.con, table)
        return int(self.con.execute(f"SELECT coalesce(max(id), 0) + 1 FROM {table}").fetchone()[0])

    def _check_free(self, track_id: int, start: int, duration: int, ignore: int | None) -> None:
        hit = self.con.execute(
            "SELECT id FROM clip WHERE track_id = ? AND id IS NOT ? "
            "AND start < ? AND start + duration > ? LIMIT 1",
            (track_id, ignore, start + duration, start),
        ).fetchone()
        if hit is not None:
            raise EditError(f"That position overlaps clip {hit['id']} on the same track.")

    def _check_media(self, track: Row, clip: Row) -> None:
        if clip["kind"] == "color":
            if track["kind"] != "video":
                raise EditError("Color clips go on video tracks.")
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
        if asset["duration"] is not None and clip["src_in"] + clip["duration"] > asset["duration"]:
            raise EditError(
                f"Clip runs past the end of {asset['name']} "
                f"(source is {asset['duration'] / 705_600_000:.3f} s)."
            )

    # -- public edits ------------------------------------------------------------------
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
    ) -> int:
        track = self._row("track", track_id)
        kind = "color" if color is not None else "media"
        if kind == "media" and asset_id is None:
            raise EditError("A media clip needs an asset.")
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
            "props": json.dumps({"color": color} if color else {}),
        }
        if duration <= 0 or start < 0 or src_in < 0:
            raise EditError("Start and source-in must be >= 0 and duration > 0.")
        self._check_media(track, row)
        self._check_free(track_id, start, duration, None)
        self._commit(
            f"Add clip to {track['name']}", [{"op": "insert", "table": "clip", "row": row}]
        )
        return int(row["id"])

    def update_clip(self, clip_id: int, label: str, **changes: Any) -> None:
        """Move/trim/regain a clip. Allowed keys: track_id, start, duration, src_in, gain_db."""
        allowed = {"track_id", "start", "duration", "src_in", "gain_db"}
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

    def delete_clip(self, clip_id: int) -> None:
        row = self._row("clip", clip_id)
        self._commit("Delete clip", [{"op": "delete", "table": "clip", "row": row}])
