from __future__ import annotations
import json
import os
import sqlite3
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.events.schemas import FramesIngest, VadIngest
from src.events import service as events_service

router = APIRouter()

def _default_db_url_like_main_py() -> str:
    backend_dir = Path(__file__).resolve().parents[2]
    default_db_path = (backend_dir / "data" / "app.db").resolve()
    return f"sqlite:///{default_db_path.as_posix()}"

def _get_sqlite_path() -> Path:
    # Resolve sqlite DB file path from DATABASE_URL.
    database_url = os.getenv("DATABASE_URL") or _default_db_url_like_main_py()

    if not database_url.startswith("sqlite:"):
        raise HTTPException(status_code=500, detail="DATABASE_URL is not sqlite")

    if database_url.startswith("sqlite:////"):
        db_path_str = database_url.replace("sqlite:////", "/", 1)
    else:
        db_path_str = database_url.replace("sqlite:///", "", 1)

    return Path(db_path_str).resolve()

@router.get("/db/health")
def db_health():
    # Checks that the sqlite file exists and we can query it.
    db_path = _get_sqlite_path()

    if not db_path.exists():
        raise HTTPException(status_code=500, detail=f"DB file not found: {db_path}")

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("SELECT 1;")
        return {"status": "ok", "db_path": str(db_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB health check failed: {e}")

@router.post("/db/test-insert")
def db_test_insert():
    # Inserts one run + one detection to prove writes work.
    db_path = _get_sqlite_path()

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")

            cur = conn.execute(
                "INSERT INTO runs (mode, model_version, notes) VALUES (?, ?, ?)",
                ("vad", "test", "test insert from /db/test-insert"),
            )
            run_id = cur.lastrowid

            conn.execute(
                """
                INSERT INTO detections
                  (run_id, occurred_at, camera_id, event_type, vad_score, kg_context, decision)
                VALUES
                  (?, datetime('now'), ?, ?, ?, ?, ?)
                """,
                (run_id, "cam0", "fall", 0.95, '{"rule":"demo"}', "logged"),
            )

            conn.commit()

        return {"inserted": True, "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {e}")

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/ingest/frames")
def ingest_frames(payload: FramesIngest):
    event = events_service.ingest_frames(
        stream_id=payload.stream_id,
        clip_id=payload.clip_id,
        ts_start=payload.ts_start,
        ts_end=payload.ts_end,
        fps=payload.fps,
        frames=[f.model_dump() for f in payload.frames],
    )
    return {
        "stored": True,
        "clip_id": payload.clip_id,
        "event_built": event is not None,
        "event": event,
    }

@router.post("/ingest/predictions")
def ingest_predictions(payload: VadIngest):
    event = events_service.ingest_prediction(
        stream_id=payload.stream_id,
        clip_id=payload.clip_id,
        ts_start=payload.ts_start,
        ts_end=payload.ts_end,
        label=payload.label,
        confidence=payload.confidence,
        extra=payload.extra,
    )
    return {
        "stored": True,
        "clip_id": payload.clip_id,
        "event_built": event is not None,
        "event": event,
    }

@router.get("/frame-selector/metrics")
def frame_selector_metrics(request: Request):
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    m = runner.metrics()
    return {
        "capture_fps_est": m.capture_fps_est,
        "selected_fps_est": m.selected_fps_est,
        "ring_size": m.ring_size,
        "batch_queue_size": m.batch_queue_size,
        "dropped_frames": m.dropped_frames,
        "dropped_batches": m.dropped_batches,
    }

@router.get("/pipeline/latest")
def pipeline_latest(request: Request):
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    latest = runner.latest()
    if latest is None:
        return {"status": "no_result_yet"}

    vad = latest["vad"]
    return {
        "clip_id": vad["clip_id"],
        "ts_start": vad["ts_start"],
        "ts_end": vad["ts_end"],
        "label": vad["label"],
        "confidence": vad["confidence"],
        "top_caption": vad.get("top_caption"),
        "kg_validated": latest.get("kg_validated"),
        "explanation": latest.get("explanation"),
        "rules_fired": latest.get("rules_fired"),
        "extra": vad.get("extra"),
        "event_id": latest.get("event_id"),
        "updated_at": latest.get("updated_at"),
    }
@router.get("/pipeline/history")
def pipeline_history(request: Request, limit: int = Query(300, ge=1, le=2000)):
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    # Overtime performance data for graph: x=time, y=confidence
    return {"points": runner.confidence_history(limit=limit)}


@router.get("/events/recent")
def recent_events(limit: int = Query(5, ge=1, le=100)):
    from src.db import repository
    try:
        events = repository.get_recent_events(limit=limit)
        return {"events": events}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/alerts")
def alerts(request: Request):
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    return {"alerts": runner.get_alerts()}

@router.get("/pipeline/stream")
def pipeline_stream(request: Request):
    # Server-Sent Events: pushes new VAD results as they arrive.
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    def gen():
        last_id = -1
        while True:
            payload = runner.latest()
            if payload is not None and payload.get("event_id") != last_id:
                last_id = payload.get("event_id")
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream")

# MJPEG stream of the latest frame
@router.get("/video/mjpeg")
def video_mjpeg(request: Request): 
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    boundary = "frame"

    def gen():
        while True:
            frame = runner.latest_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(0.03)
                continue

            data = jpg.tobytes()
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                + data + b"\r\n"
            )
            time.sleep(0.03)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )
@router.get("/events")
def get_events():
    return list_events_from_db()

@router.post("/dev/inject_alert")
def inject_alert(
    request: Request,
    severity: str = Query("warning"),
    message: str = Query("test"),
    source: str = Query("dev"),
):
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    alert = {
        "id": 9999,
        "ts": time.time(),
        "source": source,
        "severity": severity,
        "message": message,
        "data": {},
    }

    runner.logger._append(alert) if hasattr(runner.logger, "_append") else runner.logger._alerts.append(alert)
    with runner._lock:
        runner._event_id += 1
        runner._latest_payload = {
            "event_id": runner._event_id,
            "updated_at": time.time(),
            "vad": {
                "clip_id": None,
                "ts_start": 0,
                "ts_end": 0,
                "label": "manual",
                "confidence": 0.0,
                "top_caption": None,
                "extra": {},
            },
            "kg_validated": True,
            "explanation": "manual injection",
            "rules_fired": [],
            "alerts": [alert],
        }

    return {"status": "ok", "alert": alert}