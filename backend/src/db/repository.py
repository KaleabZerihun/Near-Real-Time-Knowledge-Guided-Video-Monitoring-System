from __future__ import annotations
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

def _default_db_url_like_main_py() -> str:
    backend_dir = Path(__file__).resolve().parents[2]
    default_db_path = (backend_dir / "data" / "app.db").resolve()
    return f"sqlite:///{default_db_path.as_posix()}"

def _get_sqlite_path() -> Path:
    database_url = os.getenv("DATABASE_URL") or _default_db_url_like_main_py()

    if not database_url.startswith("sqlite:"):
        raise RuntimeError(f"DATABASE_URL is not sqlite: {database_url}")

    if database_url.startswith("sqlite:////"):
        db_path_str = database_url.replace("sqlite:////", "/", 1)
    else:
        db_path_str = database_url.replace("sqlite:///", "", 1)

    return Path(db_path_str).resolve()

def _connect() -> sqlite3.Connection:
    db_path = _get_sqlite_path()
    if not db_path.exists():
        raise RuntimeError(f"DB file not found: {db_path}. Did you run init_db()?")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def upsert_frame_batch(
    *,
    clip_id: int,
    stream_id: str,
    ts_start: float,
    ts_end: float,
    fps: Optional[float],
    frames: list[dict[str, Any]],
) -> None:
    now = time.time()
    frames_json = _dumps(frames)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO frame_batches (clip_id, stream_id, ts_start, ts_end, fps, frames_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
              stream_id   = excluded.stream_id,
              ts_start    = excluded.ts_start,
              ts_end      = excluded.ts_end,
              fps         = excluded.fps,
              frames_json = excluded.frames_json,
              created_at  = excluded.created_at
            """,
            (clip_id, stream_id, ts_start, ts_end, fps, frames_json, now),
        )
        conn.commit()

def get_frame_batch(clip_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT clip_id, stream_id, ts_start, ts_end, fps, frames_json, created_at
            FROM frame_batches
            WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()

    if not row:
        return None
    return {
        "clip_id": row[0],
        "stream_id": row[1],
        "ts_start": row[2],
        "ts_end": row[3],
        "fps": row[4],
        "frames": json.loads(row[5]),
        "created_at": row[6],
    }

def upsert_vad_prediction(
    *,
    clip_id: int,
    stream_id: str,
    ts_start: float,
    ts_end: float,
    label: Optional[str],
    confidence: Optional[float],
    extra: dict[str, Any],
) -> None:
    now = time.time()
    extra_json = _dumps(extra)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO vad_predictions (clip_id, stream_id, ts_start, ts_end, label, confidence, extra_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
              stream_id   = excluded.stream_id,
              ts_start    = excluded.ts_start,
              ts_end      = excluded.ts_end,
              label       = excluded.label,
              confidence  = excluded.confidence,
              extra_json  = excluded.extra_json,
              created_at  = excluded.created_at
            """,
            (clip_id, stream_id, ts_start, ts_end, label, confidence, extra_json, now),
        )
        conn.commit()

def get_vad_prediction(clip_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT clip_id, stream_id, ts_start, ts_end, label, confidence, extra_json, created_at
            FROM vad_predictions
            WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()

    if not row:
        return None
    return {
        "clip_id": row[0],
        "stream_id": row[1],
        "ts_start": row[2],
        "ts_end": row[3],
        "label": row[4],
        "confidence": row[5],
        "extra": json.loads(row[6]),
        "created_at": row[7],
    }

def insert_event_if_missing(
    *,
    clip_id: int,
    stream_id: str,
    ts_start: float,
    ts_end: float,
    label: Optional[str],
    confidence: Optional[float],
    frames: list[dict[str, Any]],
    vad: dict[str, Any],
) -> Optional[int]:
    now = time.time()
    frames_json = _dumps(frames)
    vad_json = _dumps(vad)

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO events
              (clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, now),
        )
        conn.commit()

        if cur.rowcount == 0:
            return None

        return cur.lastrowid

def get_event_by_clip_id(clip_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at
            FROM events
            WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()

    if not row:
        return None
    return {
        "id": row[0],
        "clip_id": row[1],
        "stream_id": row[2],
        "ts_start": row[3],
        "ts_end": row[4],
        "label": row[5],
        "confidence": row[6],
        "frames": json.loads(row[7]),
        "vad": json.loads(row[8]),
        "created_at": row[9],
    }

def get_recent_events(limit: int = 5) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at
            FROM events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    events = []
    for row in rows:
        events.append({
            "id": row[0],
            "clip_id": row[1],
            "stream_id": row[2],
            "ts_start": row[3],
            "ts_end": row[4],
            "label": row[5],
            "confidence": row[6],
            "frames": json.loads(row[7]),
            "vad": json.loads(row[8]),
            "created_at": row[9],
        })
    return events