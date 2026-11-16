from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from pathlib import Path

import pytest

from storyreel.project import store
from storyreel.project.edits import EditError, Editor
from storyreel.timebase import FLICKS, Rate

SETTINGS = store.ProjectSettings("t", 640, 360, Rate(30))


def _add_color_asset_free_clip(ed: Editor, track: int, i: int) -> int:
    return ed.add_clip(track, i * FLICKS, FLICKS, color="#102030")


def test_create_open_and_lock(tmp_path: Path) -> None:
    root = tmp_path / "p.storyreel"
    with store.create(root, SETTINGS) as p:
        assert p.settings().rate == Rate(30)
        with pytest.raises(store.ProjectLockedError):
            store.open_project(root)
    assert not (root / "project.lock").exists()
    with store.open_project(root) as p:
        assert not p.recovery.recovered


def test_stale_lock_and_partial_files_are_recovered(tmp_path: Path) -> None:
    root = tmp_path / "p.storyreel"
    with store.create(root, SETTINGS) as p:
        p.con.execute(
            "INSERT INTO job(kind, state, spec, created_at) VALUES ('render', 'running', '{}', 0)"
        )
    # Simulate a kill -9: lock left behind by a dead PID, a half-written chunk.
    dead_pid = 2**22 + 12345
    (root / "project.lock").write_text(json.dumps({"pid": dead_pid, "host": socket.gethostname()}))
    (root / "cache" / "chunks" / "abc.mkv.part").write_bytes(b"x" * 100)
    with store.open_project(root) as p:
        r = p.recovery
        assert r.stale_lock_pid == dead_pid
        assert r.part_files_removed == 1
        assert r.jobs_requeued == 1
        assert p.con.execute("SELECT state FROM job").fetchone()[0] == "queued"


def test_edits_overlap_and_validation(tmp_path: Path) -> None:
    with store.create(tmp_path / "p", SETTINGS) as p:
        ed = Editor(p.con)
        v1 = p.con.execute("SELECT id FROM track WHERE kind='video'").fetchone()[0]
        ed.add_clip(v1, 0, 2 * FLICKS, color="#000000")
        with pytest.raises(EditError, match="overlaps"):
            ed.add_clip(v1, FLICKS, 2 * FLICKS, color="#000000")
        with pytest.raises(EditError):
            ed.add_clip(v1, 5 * FLICKS, 0, color="#000000")
        with pytest.raises(EditError, match="needs an asset"):
            ed.add_clip(v1, 5 * FLICKS, FLICKS)


def test_undo_redo_roundtrip(tmp_path: Path) -> None:
    with store.create(tmp_path / "p", SETTINGS) as p:
        ed = Editor(p.con)
        v1 = p.con.execute("SELECT id FROM track WHERE kind='video'").fetchone()[0]
        cid = ed.add_clip(v1, 0, FLICKS, color="#000000")
        ed.move_clip(cid, 3 * FLICKS)
        ed.delete_clip(cid)
        assert p.con.execute("SELECT count(*) FROM clip").fetchone()[0] == 0
        assert ed.undo() == "Delete clip"
        assert p.con.execute("SELECT start FROM clip").fetchone()[0] == 3 * FLICKS
        assert ed.undo() == "Move clip"
        assert p.con.execute("SELECT start FROM clip").fetchone()[0] == 0
        assert ed.redo() == "Move clip"
        assert p.con.execute("SELECT start FROM clip").fetchone()[0] == 3 * FLICKS
        ed.move_clip(cid, 9 * FLICKS)  # new edit clears the redo stack
        assert ed.redo() is None


def test_50_undos_each_under_50ms_and_survive_reopen(tmp_path: Path) -> None:
    root = tmp_path / "p"
    with store.create(root, SETTINGS) as p:
        ed = Editor(p.con)
        v1 = p.con.execute("SELECT id FROM track WHERE kind='video'").fetchone()[0]
        for i in range(60):
            _add_color_asset_free_clip(ed, v1, i)
    with store.open_project(root) as p:  # undo history is persisted
        ed = Editor(p.con)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            assert ed.undo() is not None
            times.append(time.perf_counter() - t0)
        assert p.con.execute("SELECT count(*) FROM clip").fetchone()[0] == 10
    worst_ms = max(times) * 1000
    print(f"undo: worst {worst_ms:.1f} ms, mean {sum(times) / len(times) * 1000:.1f} ms")
    assert worst_ms < 50


def test_committed_edit_survives_hard_kill(tmp_path: Path) -> None:
    """Each edit is a committed WAL transaction: a fresh connection sees it without close()."""
    root = tmp_path / "p"
    p = store.create(root, SETTINGS)
    ed = Editor(p.con)
    v1 = p.con.execute("SELECT id FROM track WHERE kind='video'").fetchone()[0]
    ed.add_clip(v1, 0, FLICKS, color="#000000")
    # No close(): simulate a crash by reading through an independent connection.
    other = sqlite3.connect(root / "project.db")
    assert other.execute("SELECT count(*) FROM clip").fetchone()[0] == 1
    other.close()
    p.con.close()
    (root / "project.lock").unlink()


def test_lock_survives_pid_reuse(tmp_path: Path) -> None:
    """A crashed holder whose PID now belongs to another live process must count as stale.

    Simulated with this test process's own PID but a different start time, which is exactly
    what Windows PID reuse looks like (found by CI on windows-latest).
    """
    root = tmp_path / "p.storyreel"
    with store.create(root, SETTINGS):
        pass
    reused = {"pid": os.getpid(), "host": socket.gethostname(), "started": 12345.0}
    (root / "project.lock").write_text(json.dumps(reused))
    with store.open_project(root) as p:  # must not raise "open in another process"
        assert p.recovery.stale_lock_pid == os.getpid()


def test_lock_holder_alive_rules() -> None:
    import psutil

    me = os.getpid()
    started = psutil.Process(me).create_time()
    assert store.lock_holder_alive(me, started)
    assert not store.lock_holder_alive(me, started - 60)  # same PID, different process
    assert store.lock_holder_alive(me, None)  # old lock format: trust pid_exists
    assert not store.lock_holder_alive(-1, None)
