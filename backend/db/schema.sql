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

CREATE TABLE IF NOT EXISTS frame_batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id INTEGER NOT NULL UNIQUE,
  stream_id TEXT NOT NULL,
  ts_start REAL NOT NULL,
  ts_end REAL NOT NULL,
  fps REAL,
  frames_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vad_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id INTEGER NOT NULL UNIQUE,
  stream_id TEXT NOT NULL,
  ts_start REAL NOT NULL,
  ts_end REAL NOT NULL,
  label TEXT,
  confidence REAL,
  extra_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  FOREIGN KEY (clip_id) REFERENCES frame_batches(clip_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id INTEGER NOT NULL UNIQUE,
  stream_id TEXT NOT NULL,
  ts_start REAL NOT NULL,
  ts_end REAL NOT NULL,
  label TEXT,
  confidence REAL,
  frames_json TEXT NOT NULL,
  vad_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  FOREIGN KEY (clip_id) REFERENCES frame_batches(clip_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events_archive (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id INTEGER NOT NULL UNIQUE,
  stream_id TEXT NOT NULL,
  ts_start REAL NOT NULL,
  ts_end REAL NOT NULL,
  label TEXT,
  confidence REAL,
  frames_json TEXT NOT NULL,
  vad_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  archived_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_archive_stream_id ON events_archive(stream_id);
CREATE INDEX IF NOT EXISTS idx_events_archive_created_at ON events_archive(created_at);

CREATE INDEX IF NOT EXISTS idx_frame_batches_stream_time ON frame_batches(stream_id, ts_start);
CREATE INDEX IF NOT EXISTS idx_vad_predictions_stream_time ON vad_predictions(stream_id, ts_start);
CREATE INDEX IF NOT EXISTS idx_events_stream_time ON events(stream_id, ts_start);

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

-- Per-clip performance metrics: one row per processed clip/batch
CREATE TABLE IF NOT EXISTS clip_metrics (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id           INTEGER NOT NULL,
  stream_id         TEXT    NOT NULL,
  recorded_at       REAL    NOT NULL,   -- unix timestamp
  vad_inference_ms  REAL    NOT NULL,   -- time spent inside VAD.predict()
  kg_inference_ms   REAL    NOT NULL,   -- time spent inside KG.augment()
  db_write_ms       REAL    NOT NULL,   -- time spent persisting to DB
  e2e_latency_ms    REAL    NOT NULL,   -- clip ts_start → result ready
  label             TEXT,               -- "normal" | "anomaly"
  confidence        REAL,               -- VAD confidence 0-1
  capture_fps       REAL,               -- camera capture FPS at time of clip
  selected_fps      REAL,               -- frame-selector output FPS
  queue_depth       INTEGER,            -- batch queue depth at time of clip
  dropped_frames    INTEGER,            -- cumulative dropped frames
  dropped_batches   INTEGER,            -- cumulative dropped batches
  is_anomaly        INTEGER NOT NULL DEFAULT 0  -- 1 if label == "anomaly"
);

CREATE INDEX IF NOT EXISTS idx_clip_metrics_recorded_at
  ON clip_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_clip_metrics_clip_id
  ON clip_metrics(clip_id);