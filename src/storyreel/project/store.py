"""Project folder: open/create, lock file, crash recovery, SQLite connection.

Layout:  <name>.storyreel/{project.db, project.lock, proxies/, cache/chunks/, renders/, logs/}

Durability: WAL + synchronous=FULL. Each edit is one transaction, fsynced on commit, so a
power cut loses only an edit that had not finished committing.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import psutil

from storyreel.audio.spec import LOUDNESS_DEFAULTS
from storyreel.errors import StoryReelError
from storyreel.project.schema import MIGRATIONS, SCHEMA_VERSION
from storyreel.timebase import Rate

log = logging.getLogger(__name__)

DB_NAME = "project.db"
LOCK_NAME = "project.lock"


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    width: int
    height: int
    rate: Rate
    sample_rate: int = 48_000
    channels: int = 2
    target_lufs: float = -14.0  # loudness target of the final mix (YouTube); -23 broadcast
    true_peak: float = -1.0  # dBTP ceiling
    normalize: bool = True  # two-pass loudness normalisation in the final encode


@dataclass(frozen=True)
class RecoveryReport:
    stale_lock_pid: int | None
    part_files_removed: int
    jobs_requeued: int

    @property
    def recovered(self) -> bool:
        return self.stale_lock_pid is not None or self.jobs_requeued > 0


class ProjectLockedError(StoryReelError):
    pass


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, isolation_level=None, timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def _migrate(con: sqlite3.Connection) -> None:
    current = int(con.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise StoryReelError(
            f"Project was created by a newer StoryReel (schema {current} > {SCHEMA_VERSION}).",
            fix="Update StoryReel.",
        )
    for version in range(current, SCHEMA_VERSION):
        con.execute("BEGIN IMMEDIATE")
        for stmt in _split_sql(MIGRATIONS[version]):
            con.execute(stmt)
        con.execute(f"PRAGMA user_version={version + 1}")
        con.execute("COMMIT")


def _split_sql(script: str) -> list[str]:
    """Split a migration into statements; sqlite3.complete_statement handles triggers."""
    out: list[str] = []
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        raise StoryReelError("internal: incomplete SQL statement in migration")
    return out


class Project:
    """An open project. Use as a context manager so the lock is always released."""

    def __init__(self, root: Path, con: sqlite3.Connection, recovery: RecoveryReport) -> None:
        self.root = root
        self.con = con
        self.recovery = recovery

    # -- paths -------------------------------------------------------------------------
    @property
    def chunks_dir(self) -> Path:
        return self.root / "cache" / "chunks"

    @property
    def proxies_dir(self) -> Path:
        return self.root / "proxies"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    # -- settings ----------------------------------------------------------------------
    def settings(self) -> ProjectSettings:
        r = self.con.execute("SELECT * FROM project WHERE id = 1").fetchone()
        audio = {**LOUDNESS_DEFAULTS, **json.loads(r["audio"])}
        return ProjectSettings(
            r["name"],
            r["width"],
            r["height"],
            Rate(r["fps_num"], r["fps_den"]),
            r["sample_rate"],
            r["channels"],
            float(audio["target_lufs"]),
            float(audio["true_peak"]),
            bool(audio["normalize"]),
        )

    # -- lifecycle ---------------------------------------------------------------------
    def close(self) -> None:
        self.con.close()
        (self.root / LOCK_NAME).unlink(missing_ok=True)

    def __enter__(self) -> Project:
        return self

    def __exit__(
        self,
        et: type[BaseException] | None,
        ev: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _process_started(pid: int) -> float | None:
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def lock_holder_alive(pid: int, started: float | None) -> bool:
    """True only if `pid` is the same process that wrote the lock.

    PIDs are reused (quickly on Windows: seen in CI seconds after a kill), so a bare
    pid_exists() check can mistake an unrelated process for the lock holder and lock the
    user out of their project after a crash. The process start time disambiguates.
    """
    if pid <= 0 or not psutil.pid_exists(pid):
        return False
    if started is None:  # lock written by an older StoryReel: best effort
        return True
    now = _process_started(pid)
    return now is not None and abs(now - started) < 1.0


def _acquire_lock(root: Path) -> int | None:
    """Create the lock file. Returns the PID of a stale lock that was taken over, if any."""
    lock = root / LOCK_NAME
    payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t": time.time(),
                          "started": _process_started(os.getpid())})  # fmt: skip
    stale: int | None = None
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                info = json.loads(lock.read_text(encoding="utf-8"))
                pid, host = int(info["pid"]), str(info["host"])
                started = info.get("started")
                started = float(started) if started is not None else None
            except (OSError, ValueError, KeyError, TypeError):
                pid, host, started = -1, socket.gethostname(), None  # unreadable: stale
            if host == socket.gethostname() and lock_holder_alive(pid, started):
                if pid == os.getpid():
                    raise ProjectLockedError(
                        f"Project {root.name} is already open in this process."
                    ) from None
                raise ProjectLockedError(
                    f"Project {root.name} is open in another StoryReel process (PID {pid}).",
                    fix="Close the other window, or wait for its render to finish.",
                ) from None
            if host != socket.gethostname():
                raise ProjectLockedError(
                    f"Project {root.name} is locked by computer '{host}'.",
                    fix=f"If that computer is not using it, delete {lock}.",
                ) from None
            lock.unlink(missing_ok=True)
            stale = pid
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        return stale
    raise ProjectLockedError(f"Could not lock project {root.name}.")


def _recover(root: Path, con: sqlite3.Connection, stale_pid: int | None) -> RecoveryReport:
    parts = 0
    for p in root.rglob("*.part"):
        p.unlink(missing_ok=True)
        parts += 1
    cur = con.execute("UPDATE job SET state = 'queued' WHERE state = 'running'")
    report = RecoveryReport(stale_pid, parts, cur.rowcount)
    if report.recovered or parts:
        log.warning("project recovered", extra={"project": str(root), **report.__dict__})
    return report


def create(root: Path, settings: ProjectSettings) -> Project:
    if (root / DB_NAME).exists():
        raise StoryReelError(f"A project already exists at {root}.", fix="Pick another folder.")
    for sub in ("proxies", "cache/chunks", "renders", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    _acquire_lock(root)
    con = connect(root / DB_NAME)
    _migrate(con)
    audio = {
        "target_lufs": settings.target_lufs,
        "true_peak": settings.true_peak,
        "normalize": settings.normalize,
    }
    con.execute(
        "INSERT INTO project VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            settings.name,
            settings.width,
            settings.height,
            settings.rate.num,
            settings.rate.den,
            settings.sample_rate,
            settings.channels,
            time.time(),
            json.dumps(audio),
        ),
    )
    con.execute("INSERT INTO track(kind, idx, name) VALUES ('video', 0, 'V1'), ('audio', 0, 'A1')")
    return Project(root, con, RecoveryReport(None, 0, 0))


def open_project(root: Path) -> Project:
    if not (root / DB_NAME).is_file():
        raise StoryReelError(
            f"No StoryReel project at {root}.", fix="Create one with `storyreel new`."
        )
    stale = _acquire_lock(root)
    try:
        con = connect(root / DB_NAME)
        _migrate(con)
        recovery = _recover(root, con, stale)
    except BaseException:
        (root / LOCK_NAME).unlink(missing_ok=True)
        raise
    return Project(root, con, recovery)
