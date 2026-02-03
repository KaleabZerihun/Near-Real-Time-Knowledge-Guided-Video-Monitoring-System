PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at    TEXT NOT NULL DEFAULT (datetime('now')),
  ended_at      TEXT,
  mode          TEXT NOT NULL CHECK (mode IN ('vad', 'vad+kg')),
  model_version TEXT,
  notes         TEXT
);

CREATE TABLE IF NOT EXISTS detections (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      INTEGER,
  occurred_at TEXT NOT NULL,
  camera_id   TEXT NOT NULL,
  event_type  TEXT NOT NULL,
  vad_score   REAL,
  kg_context  TEXT,
  decision    TEXT NOT NULL DEFAULT 'logged'
              CHECK (decision IN ('ignored','logged','alerted')),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_time
  ON detections(occurred_at);

CREATE INDEX IF NOT EXISTS idx_detections_camera_time
  ON detections(camera_id, occurred_at);

CREATE INDEX IF NOT EXISTS idx_detections_event_type
  ON detections(event_type);

CREATE TABLE IF NOT EXISTS alerts (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  detection_id INTEGER NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  severity     TEXT NOT NULL CHECK (severity IN ('low','medium','high')),
  status       TEXT NOT NULL DEFAULT 'new'
               CHECK (status IN ('new','acknowledged','resolved')),
  channel      TEXT NOT NULL DEFAULT 'dashboard',
  FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alerts_status_time
  ON alerts(status, created_at);

CREATE TABLE IF NOT EXISTS system_metrics (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id          INTEGER,
  recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),
  inference_ms    REAL,
  fps             REAL,
  queue_depth     INTEGER,
  detections_cnt  INTEGER DEFAULT 0,
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_time
  ON system_metrics(recorded_at);