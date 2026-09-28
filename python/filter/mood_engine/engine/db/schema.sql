CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    is_provisional  INTEGER NOT NULL DEFAULT 0,
    notes           TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    last_seen_at    TEXT,
    baseline        TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BLOB    NOT NULL,
    dim         INTEGER NOT NULL,
    source      TEXT    NOT NULL,
    quality     REAL    NOT NULL DEFAULT 0,
    image_path  TEXT,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);

CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    frame_count  INTEGER NOT NULL DEFAULT 0,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id       INTEGER REFERENCES persons(id) ON DELETE CASCADE,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    ts              TEXT    NOT NULL,
    emotion         TEXT    NOT NULL,
    emotion_conf    REAL    NOT NULL,
    valence         REAL    NOT NULL,
    arousal         REAL    NOT NULL,
    probs           TEXT    NOT NULL,
    age             REAL,
    gender          TEXT,
    yaw             REAL,
    pitch           REAL,
    roll            REAL,
    gaze_x          REAL,
    gaze_y          REAL,
    attention       REAL,
    liveness        REAL,
    is_spoof        INTEGER NOT NULL DEFAULT 0,
    match_score     REAL,
    det_score       REAL,
    snapshot_path   TEXT,
    duchenne        REAL,
    mood_state      TEXT,
    smile_type      TEXT,
    compound        TEXT,
    engagement      REAL,
    fatigue         REAL,
    tension         REAL,
    volatility      REAL,
    blink_rate      REAL,
    expressiveness  REAL
);
CREATE INDEX IF NOT EXISTS idx_obs_person_ts ON observations(person_id, ts);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts);

CREATE TABLE IF NOT EXISTS session_reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    person_id    INTEGER REFERENCES persons(id) ON DELETE CASCADE,
    person_name  TEXT,
    started_at   TEXT NOT NULL,
    ended_at     TEXT NOT NULL,
    duration_s   REAL NOT NULL,
    samples      INTEGER NOT NULL,
    state        TEXT NOT NULL,
    confidence   REAL NOT NULL,
    dominance    REAL,
    valence      REAL,
    arousal      REAL,
    stability    REAL,
    engagement   REAL,
    fatigue      REAL,
    tension      REAL,
    emotion_mix  TEXT,
    notes        TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_person ON session_reports(person_id, ended_at);
CREATE INDEX IF NOT EXISTS idx_reports_ended ON session_reports(ended_at);
