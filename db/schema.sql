-- Aegis-Sphere Database Schema
-- Tables: sessions, evidence_items, onco_cases, overrides, vram_logs

CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    patient_id    TEXT NOT NULL,
    created_at    REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'INITIALIZED',
    degradation   TEXT,
    staging       TEXT,
    transcript    TEXT,
    clinical_frame_json TEXT,
    updated_at    REAL
);

CREATE TABLE IF NOT EXISTS evidence_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    modality      TEXT NOT NULL,
    model         TEXT NOT NULL,
    status        TEXT NOT NULL,
    finding       TEXT,
    confidence    REAL,
    nba           TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS onco_cases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    oncocase_json TEXT NOT NULL,
    degradation   TEXT NOT NULL,
    staging       TEXT NOT NULL,
    nba_json      TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS debate_outputs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    pass_number   INTEGER NOT NULL,
    persona       TEXT NOT NULL,
    output_text   TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS overrides (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    clinician_id  TEXT,
    field         TEXT NOT NULL,
    old_value     TEXT,
    new_value     TEXT NOT NULL,
    reason        TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vram_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    timestamp     REAL NOT NULL,
    elapsed_s     REAL NOT NULL,
    phase         TEXT NOT NULL,
    allocated_mb  REAL NOT NULL,
    reserved_mb   REAL NOT NULL,
    model_active  TEXT
);

CREATE INDEX IF NOT EXISTS idx_evidence_session ON evidence_items(session_id);
CREATE INDEX IF NOT EXISTS idx_onco_session ON onco_cases(session_id);
CREATE INDEX IF NOT EXISTS idx_debate_session ON debate_outputs(session_id);
CREATE INDEX IF NOT EXISTS idx_vram_session ON vram_logs(session_id);
