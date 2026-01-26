from __future__ import annotations

import json
import time
import cv2
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


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
        "top_caption": vad["top_caption"],
        "kg_validated": latest["kg_validated"],
        "explanation": latest["explanation"],
        "rules_fired": latest["rules_fired"],
        "extra": vad["extra"],
        "event_id": latest["event_id"],
        "updated_at": latest["updated_at"],
    }


@router.get("/pipeline/stream")
def pipeline_stream(request: Request):
    """
    Server-Sent Events: pushes new VAD results as they arrive.
    """
    runner = request.app.state.runner
    if not runner:
        return JSONResponse({"error": "runner not started"}, status_code=503)

    def gen():
        last_id = -1
        while True:
            payload = runner.latest()
            if payload is not None and payload["event_id"] != last_id:
                last_id = payload["event_id"]
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/video/mjpeg")
def video_mjpeg(request: Request):
    """
    MJPEG stream of the latest frame (with VAD overlay if available).
    Open in browser: /video/mjpeg
    """
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

            # Encode as JPEG
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
