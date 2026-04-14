from __future__ import annotations
import json
import os
import sqlite3
import time
import uuid
import threading
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from importlib.util import module_from_spec, spec_from_file_location

from src.events.schemas import FramesIngest, VadIngest
from src.events import service as events_service
from src.db import repository

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


def _load_rtvad_textencoder_module(rtvad_root: Path):
    textencoder_path = rtvad_root / "textencoder.py"
    if not textencoder_path.exists():
        raise FileNotFoundError(f"RT-VAD textencoder not found at {textencoder_path}")

    spec = spec_from_file_location("rtvad_textencoder", str(textencoder_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import textencoder from {textencoder_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_custom_memory_file(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"custom_anomalies": []}, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_custom_memory(path: Path):
    _ensure_custom_memory_file(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("custom_anomalies", []) if isinstance(data, dict) else []


def _save_custom_memory(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"custom_anomalies": items}, f, ensure_ascii=False, indent=2)


def _build_custom_anomaly_async(runner, rtvad_root: Path, text: str, custom_path: Path):
    """
    Background thread worker to build custom anomaly embedding without blocking the pipeline.
    """
    try:
        items = _load_custom_memory(custom_path)
        
        # Check again in case another request added it
        if any(item.get("text", "").strip().lower() == text.lower() for item in items):
            print(f"[CUSTOM-ANOMALY] Already exists (concurrent add): {text}")
            return
        
        textencoder = _load_rtvad_textencoder_module(rtvad_root)
        textencoder.CHECKPOINT_PATH = rtvad_root / ".checkpoints" / "imagebind_huge.pth"
        model, ModalityType, imagebind_data = textencoder.load_imagebind(
            textencoder.torch.device(runner.vad.device if runner.vad.device else "cpu")
        )
        emb, _ = textencoder.build_custom_anomaly_embedding(
            text=text,
            model=model,
            ModalityType=ModalityType,
            imagebind_data=imagebind_data,
            device=textencoder.torch.device(runner.vad.device if runner.vad.device else "cpu"),
            alpha=0.95,
            apply_prompt_template=False,
        )
        
        items.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "category": "user_defined_anomaly",
            "label": 1,
            "embedding": emb.tolist(),
            "created_at": time.time(),
        })
        
        _save_custom_memory(custom_path, items)
        runner.vad.reload_memory()
        print(f"[CUSTOM-ANOMALY] Successfully added: {text}")
    except Exception as e:
        print(f"[CUSTOM-ANOMALY] Error building embedding for '{text}': {e}")


@router.post("/pipeline/custom-anomaly")
async def pipeline_custom_anomaly(request: Request):
    try:
        runner = request.app.state.runner
        if not runner:
            return JSONResponse({"error": "Pipeline is not running"}, status_code=503)

        payload = await request.json()
        text = payload.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "Text is required"}, status_code=400)

        rtvad_root = Path(runner.vad.rtvad_root)
        custom_path = rtvad_root / "custom_anomaly_memory.json"
        _ensure_custom_memory_file(custom_path)
        items = _load_custom_memory(custom_path)

        if any(item.get("text", "").strip().lower() == text.lower() for item in items):
            return JSONResponse({"error": "This custom anomaly already exists"}, status_code=409)

        # Return immediately and build embedding in background thread
        thread = threading.Thread(
            target=_build_custom_anomaly_async,
            args=(runner, rtvad_root, text, custom_path),
            daemon=True
        )
        thread.start()

        return JSONResponse({"added": True, "text": text, "status": "building_embedding"}, status_code=202)
    except Exception as e:
        print(f"[CUSTOM-ANOMALY] Route error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
@router.get("/pipeline/perf")
def pipeline_perf(request: Request):
    """Live performance metrics: speed, inference time, E2E latency, throughput."""
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)
    return runner.get_perf()

@router.get("/pipeline/history")
def pipeline_history(limit: int = 300):
    rows = repository.get_recent_detections(limit=limit)

    points = [
        {
            "t": row["occurred_at"],
            "confidence": row["vad_score"],
            "label": row["event_type"],
            "decision": row["decision"],
            "camera_id": row["camera_id"],
        }
        for row in rows
        if row["vad_score"] is not None
    ]

    points.reverse()  # optional: oldest -> newest
    return {"points": points}


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
def get_events(limit: int = Query(100, ge=1, le=500)):
    try:
        return {"events": repository.get_recent_events(limit=limit)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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

@router.get("/detections/recent")
def recent_detections(limit: int = Query(20, ge=1, le=500)):
    try:
        return {"detections": repository.get_recent_detections(limit=limit)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/metrics/recent")
def recent_metrics(limit: int = Query(50, ge=1, le=500)):
    try:
        return {"metrics": repository.get_recent_system_metrics(limit=limit)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
@router.get("/alerts/recent")
def recent_alerts(limit: int = Query(20, ge=1, le=500)):
    try:
        return {"alerts": repository.get_recent_alerts(limit=limit)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
@router.get("/db/summary")
def db_summary():
    try:
        return {
            "runs": repository.count_runs(),
            "detections": repository.count_detections(),
            "alerts": repository.count_alerts(),
            "system_metrics": repository.count_system_metrics(),
            "events": repository.count_events(),
            "frame_batches": repository.count_frame_batches(),
            "vad_predictions": repository.count_vad_predictions(),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)