"""SQLite schema. Version tracked with PRAGMA user_version; migrations are append-only.

All times are int64 flicks.
"""

from __future__ import annotations

MIGRATIONS: list[str] = [
    # 1: Phase 1 core
    """
    CREATE TABLE project (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        name TEXT NOT NULL,
        width INTEGER NOT NULL, height INTEGER NOT NULL,
        fps_num INTEGER NOT NULL, fps_den INTEGER NOT NULL,
        sample_rate INTEGER NOT NULL, channels INTEGER NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE TABLE asset (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('video', 'image', 'audio')),
        path TEXT NOT NULL,
        name TEXT NOT NULL,
        is_generated INTEGER NOT NULL DEFAULT 0,
        fingerprint TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        duration INTEGER,                  -- flicks; NULL for still images
        width INTEGER, height INTEGER,
        fps_num INTEGER, fps_den INTEGER,
        vcodec TEXT, acodec TEXT,
        has_video INTEGER NOT NULL, has_audio INTEGER NOT NULL,
        sample_rate INTEGER, channels INTEGER,
        license TEXT NOT NULL DEFAULT 'unknown',
        source_url TEXT, tags TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'missing', 'corrupt')),
        error TEXT,
        imported_at REAL NOT NULL,
        UNIQUE (fingerprint, path)
    );
    CREATE VIRTUAL TABLE asset_fts USING fts5(name, tags, license, source_url,
        content='asset', content_rowid='id');
    CREATE TRIGGER asset_ai AFTER INSERT ON asset BEGIN
        INSERT INTO asset_fts(rowid, name, tags, license, source_url)
        VALUES (new.id, new.name, new.tags, new.license, coalesce(new.source_url, ''));
    END;
    CREATE TRIGGER asset_ad AFTER DELETE ON asset BEGIN
        INSERT INTO asset_fts(asset_fts, rowid, name, tags, license, source_url)
        VALUES ('delete', old.id, old.name, old.tags, old.license, coalesce(old.source_url, ''));
    END;
    CREATE TRIGGER asset_au AFTER UPDATE ON asset BEGIN
        INSERT INTO asset_fts(asset_fts, rowid, name, tags, license, source_url)
        VALUES ('delete', old.id, old.name, old.tags, old.license, coalesce(old.source_url, ''));
        INSERT INTO asset_fts(rowid, name, tags, license, source_url)
        VALUES (new.id, new.name, new.tags, new.license, coalesce(new.source_url, ''));
    END;
    CREATE TABLE proxy (
        asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,               -- video540 | video360 | peaks | thumbs
        path TEXT NOT NULL,
        params_hash TEXT NOT NULL,
        status TEXT NOT NULL,             -- ok | failed
        error TEXT,
        PRIMARY KEY (asset_id, kind)
    );
    CREATE TABLE track (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('video', 'audio')),
        idx INTEGER NOT NULL,             -- video: higher = on top
        name TEXT NOT NULL,
        muted INTEGER NOT NULL DEFAULT 0,
        UNIQUE (kind, idx)
    );
    CREATE TABLE clip (
        id INTEGER PRIMARY KEY,
        track_id INTEGER NOT NULL REFERENCES track(id),
        kind TEXT NOT NULL CHECK (kind IN ('media', 'color')),
        asset_id INTEGER REFERENCES asset(id),
        start INTEGER NOT NULL CHECK (start >= 0),
        duration INTEGER NOT NULL CHECK (duration > 0),
        src_in INTEGER NOT NULL DEFAULT 0 CHECK (src_in >= 0),
        speed_num INTEGER NOT NULL DEFAULT 1,
        speed_den INTEGER NOT NULL DEFAULT 1,
        gain_db REAL NOT NULL DEFAULT 0,
        props TEXT NOT NULL DEFAULT '{}'  -- JSON; e.g. {"color": "#202040"}
    );
    CREATE INDEX clip_track_start ON clip(track_id, start);
    CREATE TABLE command (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        ops TEXT NOT NULL,                -- JSON list of row operations
        undone INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    CREATE TABLE job (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,              -- queued | running | done | failed | paused_memory
        spec TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        progress REAL NOT NULL DEFAULT 0,
        peak_rss INTEGER,
        created_at REAL NOT NULL, started_at REAL, finished_at REAL
    );
    CREATE TABLE cache_entry (
        key TEXT PRIMARY KEY,
        kind TEXT NOT NULL,               -- vchunk | achunk
        path TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        created_at REAL NOT NULL,
        last_used_at REAL NOT NULL,
        pinned INTEGER NOT NULL DEFAULT 0
    );
    """,
    # 2: Phase 2 compositor: more clip kinds, keyframes, effects, transitions
    """
    CREATE TABLE clip_v2 (
        id INTEGER PRIMARY KEY,
        track_id INTEGER NOT NULL REFERENCES track(id),
        kind TEXT NOT NULL CHECK (kind IN ('media', 'color', 'text', 'generator', 'adjust')),
        asset_id INTEGER REFERENCES asset(id),
        start INTEGER NOT NULL CHECK (start >= 0),
        duration INTEGER NOT NULL CHECK (duration > 0),
        src_in INTEGER NOT NULL DEFAULT 0 CHECK (src_in >= 0),
        speed_num INTEGER NOT NULL DEFAULT 1,
        speed_den INTEGER NOT NULL DEFAULT 1,
        gain_db REAL NOT NULL DEFAULT 0,
        props TEXT NOT NULL DEFAULT '{}'  -- JSON: transform statics, text, generator, flags
    );
    INSERT INTO clip_v2 SELECT * FROM clip;
    DROP TABLE clip;
    ALTER TABLE clip_v2 RENAME TO clip;
    CREATE INDEX clip_track_start ON clip(track_id, start);
    ALTER TABLE asset ADD COLUMN color_space TEXT;
    ALTER TABLE asset ADD COLUMN pix_fmt TEXT;
    CREATE TABLE keyframe (
        id INTEGER PRIMARY KEY,
        clip_id INTEGER NOT NULL REFERENCES clip(id),
        param TEXT NOT NULL,              -- x, y, scale, rotation, opacity, speed, fx<id>.<name>
        t INTEGER NOT NULL,               -- flicks from clip start
        value REAL NOT NULL,
        interp TEXT NOT NULL DEFAULT 'linear' CHECK (interp IN ('hold', 'linear', 'bezier')),
        h1x REAL NOT NULL DEFAULT 0.42, h1y REAL NOT NULL DEFAULT 0,
        h2x REAL NOT NULL DEFAULT 0.58, h2y REAL NOT NULL DEFAULT 1,
        UNIQUE (clip_id, param, t)
    );
    CREATE TABLE effect (
        id INTEGER PRIMARY KEY,
        clip_id INTEGER NOT NULL REFERENCES clip(id),
        idx INTEGER NOT NULL,
        type TEXT NOT NULL,
        params TEXT NOT NULL DEFAULT '{}',
        enabled INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE transition (
        id INTEGER PRIMARY KEY,
        clip_a INTEGER NOT NULL REFERENCES clip(id),
        clip_b INTEGER NOT NULL REFERENCES clip(id),
        kind TEXT NOT NULL,
        duration INTEGER NOT NULL CHECK (duration > 0),
        params TEXT NOT NULL DEFAULT '{}',
        UNIQUE (clip_a), UNIQUE (clip_b)
    );
    """,
    # 3: Phase 3 audio engine: buses, bus effect chains, project audio settings
    """
    ALTER TABLE track ADD COLUMN bus TEXT NOT NULL DEFAULT 'dialogue'
        CHECK (bus IN ('dialogue', 'music', 'fx', 'ambience'));
    CREATE TABLE bus (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
            CHECK (name IN ('dialogue', 'music', 'fx', 'ambience', 'master')),
        gain_db REAL NOT NULL DEFAULT 0,
        muted INTEGER NOT NULL DEFAULT 0,
        params TEXT NOT NULL DEFAULT '{}'  -- JSON: duck, duck_db, duck_attack, duck_release
    );
    INSERT INTO bus(id, name, params) VALUES
        (1, 'dialogue', '{}'),
        (2, 'music', '{"duck": true}'),
        (3, 'fx', '{}'),
        (4, 'ambience', '{"duck": true}'),
        (5, 'master', '{}');
    CREATE TABLE bus_effect (
        id INTEGER PRIMARY KEY,
        bus_id INTEGER NOT NULL REFERENCES bus(id),
        idx INTEGER NOT NULL,
        type TEXT NOT NULL,
        params TEXT NOT NULL DEFAULT '{}',
        enabled INTEGER NOT NULL DEFAULT 1
    );
    ALTER TABLE project ADD COLUMN audio TEXT NOT NULL DEFAULT '{}';  -- loudness settings
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)
