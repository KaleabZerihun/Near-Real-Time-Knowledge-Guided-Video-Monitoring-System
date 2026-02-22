from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from src.metrics.models import ClipMetrics


def _get_db_path() -> Path:
    database_url = os.getenv("DATABASE_URL", "")
    if database_url.startswith("sqlite:////"):
        return Path(database_url.replace("sqlite:////", "/", 1)).resolve()
    if database_url.startswith("sqlite:///"):
        return Path(database_url.replace("sqlite:///", "", 1)).resolve()
    # Fallback: same default as main.py
    backend_dir = Path(__file__).resolve().parents[2]
    return (backend_dir / "data" / "app.db").resolve()


def insert_clip_metrics(m: ClipMetrics) -> None:
    """Persist a ClipMetrics record to the clip_metrics table."""
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO clip_metrics (
                clip_id, stream_id, recorded_at,
                vad_inference_ms, kg_inference_ms, db_write_ms, e2e_latency_ms,
                label, confidence,
                capture_fps, selected_fps, queue_depth,
                dropped_frames, dropped_batches, is_anomaly
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.clip_id, m.stream_id, m.recorded_at,
                m.vad_inference_ms, m.kg_inference_ms, m.db_write_ms, m.e2e_latency_ms,
                m.label, m.confidence,
                m.capture_fps, m.selected_fps, m.queue_depth,
                m.dropped_frames, m.dropped_batches,
                1 if m.is_anomaly else 0,
            ),
        )
        conn.commit()


def list_clip_metrics(limit: int = 300) -> list[dict]:
    """Fetch the most recent `limit` rows from clip_metrics, newest first."""
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM clip_metrics
            ORDER BY recorded_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
