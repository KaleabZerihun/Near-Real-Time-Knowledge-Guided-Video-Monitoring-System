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


def _loads(raw: Optional[str], default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default

def create_run(
    *,
    mode: str,
    model_version: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (mode, model_version, notes)
            VALUES (?, ?, ?)
            """,
            (mode, model_version, notes),
        )
        conn.commit()
        return cur.lastrowid


def finish_run(run_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE runs
            SET ended_at = datetime('now')
            WHERE id = ?
            """,
            (run_id,),
        )
        conn.commit()

def get_run(run_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, started_at, ended_at, mode, model_version, notes
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "started_at": row[1],
        "ended_at": row[2],
        "mode": row[3],
        "model_version": row[4],
        "notes": row[5],
    }

def insert_detection(
    *,
    run_id: Optional[int],
    occurred_at: str,
    camera_id: str,
    event_type: str,
    vad_score: Optional[float] = None,
    kg_context: Optional[Any] = None,
    decision: str = "logged",
) -> int:
    if isinstance(kg_context, (dict, list)):
        kg_context_db = _dumps(kg_context)
    else:
        kg_context_db = kg_context

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO detections
                (run_id, occurred_at, camera_id, event_type, vad_score, kg_context, decision)
            VALUES
                (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, occurred_at, camera_id, event_type, vad_score, kg_context_db, decision),
        )
        conn.commit()
        return cur.lastrowid


def create_detection(
    *,
    run_id: Optional[int],
    occurred_at: str,
    camera_id: str,
    event_type: str,
    vad_score: Optional[float] = None,
    kg_context: Optional[Any] = None,
    decision: str = "logged",
) -> int:
    return insert_detection(
        run_id=run_id,
        occurred_at=occurred_at,
        camera_id=camera_id,
        event_type=event_type,
        vad_score=vad_score,
        kg_context=kg_context,
        decision=decision,
    )


def get_recent_detections(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, run_id, occurred_at, camera_id, event_type, vad_score, kg_context, decision
            FROM detections
            ORDER BY occurred_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for row in rows:
        kg_context_value = row[6]

        try:
            if isinstance(kg_context_value, str) and kg_context_value.strip().startswith(("{", "[")):
                kg_context_value = json.loads(kg_context_value)
        except Exception:
            pass

        result.append(
            {
                "id": row[0],
                "run_id": row[1],
                "occurred_at": row[2],
                "camera_id": row[3],
                "event_type": row[4],
                "vad_score": row[5],
                "kg_context": kg_context_value,
                "decision": row[7],
            }
        )

    return result

def insert_alert(
    *,
    detection_id: int,
    severity: str,
    status: str = "new",
    channel: str = "dashboard",
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO alerts (detection_id, severity, status, channel)
            VALUES (?, ?, ?, ?)
            """,
            (detection_id, severity, status, channel),
        )
        conn.commit()
        return cur.lastrowid


def get_recent_alerts(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, detection_id, created_at, severity, status, channel
            FROM alerts
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row[0],
            "detection_id": row[1],
            "created_at": row[2],
            "severity": row[3],
            "status": row[4],
            "channel": row[5],
        }
        for row in rows
    ]

def insert_system_metric(
    *,
    run_id: Optional[int],
    inference_ms: Optional[float] = None,
    fps: Optional[float] = None,
    queue_depth: Optional[int] = None,
    detections_cnt: int = 0,
    recorded_at: Optional[str] = None,
) -> int:
    with _connect() as conn:
        if recorded_at is None:
            cur = conn.execute(
                """
                INSERT INTO system_metrics
                    (run_id, inference_ms, fps, queue_depth, detections_cnt)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                (run_id, inference_ms, fps, queue_depth, detections_cnt),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO system_metrics
                    (run_id, recorded_at, inference_ms, fps, queue_depth, detections_cnt)
                VALUES
                    (?, ?, ?, ?, ?, ?)
                """,
                (run_id, recorded_at, inference_ms, fps, queue_depth, detections_cnt),
            )

        conn.commit()
        return cur.lastrowid


def get_recent_system_metrics(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, run_id, recorded_at, inference_ms, fps, queue_depth, detections_cnt
            FROM system_metrics
            ORDER BY recorded_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row[0],
            "run_id": row[1],
            "recorded_at": row[2],
            "inference_ms": row[3],
            "fps": row[4],
            "queue_depth": row[5],
            "detections_cnt": row[6],
        }
        for row in rows
    ]

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
            INSERT INTO frame_batches
                (clip_id, stream_id, ts_start, ts_end, fps, frames_json, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?)
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
        "frames": _loads(row[5], []),
        "created_at": row[6],
    }


def delete_frame_batch(clip_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            """
            DELETE FROM frame_batches
            WHERE clip_id = ?
            """,
            (clip_id,),
        )
        conn.commit()
        return cur.rowcount > 0

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
            INSERT INTO vad_predictions
                (clip_id, stream_id, ts_start, ts_end, label, confidence, extra_json, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
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
        "extra": _loads(row[6], {}),
        "created_at": row[7],
    }


def delete_vad_prediction(clip_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            """
            DELETE FROM vad_predictions
            WHERE clip_id = ?
            """,
            (clip_id,),
        )
        conn.commit()
        return cur.rowcount > 0

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


def upsert_event(
    *,
    clip_id: int,
    stream_id: str,
    ts_start: float,
    ts_end: float,
    label: Optional[str],
    confidence: Optional[float],
    frames: list[dict[str, Any]],
    vad: dict[str, Any],
) -> None:
    now = time.time()
    frames_json = _dumps(frames)
    vad_json = _dumps(vad)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO events
                (clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
                stream_id   = excluded.stream_id,
                ts_start    = excluded.ts_start,
                ts_end      = excluded.ts_end,
                label       = excluded.label,
                confidence  = excluded.confidence,
                frames_json = excluded.frames_json,
                vad_json    = excluded.vad_json,
                created_at  = excluded.created_at
            """,
            (clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, now),
        )
        conn.commit()


def archive_event_by_clip_id(clip_id: int) -> bool:
    archived_at = time.time()

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at
            FROM events
            WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()

        if not row:
            return False

        conn.execute(
            """
            INSERT OR REPLACE INTO events_archive
                (clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at, archived_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row[0], row[1], row[2], row[3], row[4],
                row[5], row[6], row[7], row[8], archived_at,
            ),
        )

        conn.execute(
            """
            DELETE FROM events
            WHERE clip_id = ?
            """,
            (clip_id,),
        )

        conn.commit()
        return True


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
        "frames": _loads(row[7], []),
        "vad": _loads(row[8], {}),
        "created_at": row[9],
    }


def get_event_by_id(event_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at
            FROM events
            WHERE id = ?
            """,
            (event_id,),
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
        "frames": _loads(row[7], []),
        "vad": _loads(row[8], {}),
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

    return [
        {
            "id": row[0],
            "clip_id": row[1],
            "stream_id": row[2],
            "ts_start": row[3],
            "ts_end": row[4],
            "label": row[5],
            "confidence": row[6],
            "frames": _loads(row[7], []),
            "vad": _loads(row[8], {}),
            "created_at": row[9],
        }
        for row in rows
    ]


def get_events_by_stream(stream_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at
            FROM events
            WHERE stream_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (stream_id, limit),
        ).fetchall()

    return [
        {
            "id": row[0],
            "clip_id": row[1],
            "stream_id": row[2],
            "ts_start": row[3],
            "ts_end": row[4],
            "label": row[5],
            "confidence": row[6],
            "frames": _loads(row[7], []),
            "vad": _loads(row[8], {}),
            "created_at": row[9],
        }
        for row in rows
    ]


def delete_event_by_clip_id(clip_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            """
            DELETE FROM events
            WHERE clip_id = ?
            """,
            (clip_id,),
        )
        conn.commit()
        return cur.rowcount > 0
    
def list_events(limit: int = 100) -> list[dict[str, Any]]:
    return get_recent_events(limit=limit)

def count_runs() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
    return int(row[0])


def count_detections() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM detections").fetchone()
    return int(row[0])


def count_alerts() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
    return int(row[0])


def count_system_metrics() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()
    return int(row[0])

def count_events() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    return int(row[0])


def count_frame_batches() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM frame_batches").fetchone()
    return int(row[0])


def count_vad_predictions() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM vad_predictions").fetchone()
    return int(row[0])