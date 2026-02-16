from __future__ import annotations
import time
import threading
from typing import Optional, Any
from src.db import repository
from src.events.builder import build_event_payload

_lock = threading.Lock()
_pending_created_at: dict[int, float] = {}

PENDING_TTL_SECONDS = 180

def _cleanup_pending() -> None:
    now = time.time()
    expired = [cid for cid, created in _pending_created_at.items() if now - created > PENDING_TTL_SECONDS]
    for cid in expired:
        _pending_created_at.pop(cid, None)

def ingest_frames(
    *,
    stream_id: str,
    clip_id: int,
    ts_start: float,
    ts_end: float,
    fps: Optional[float],
    frames: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    repository.upsert_frame_batch(
        clip_id=clip_id,
        stream_id=stream_id,
        ts_start=ts_start,
        ts_end=ts_end,
        fps=fps,
        frames=frames,
    )
    with _lock:
        _pending_created_at.setdefault(clip_id, time.time())
        _cleanup_pending()
    return try_assemble_event(clip_id)

def ingest_prediction(
    *,
    stream_id: str,
    clip_id: int,
    ts_start: float,
    ts_end: float,
    label: Optional[str],
    confidence: Optional[float],
    extra: dict[str, Any],
) -> Optional[dict[str, Any]]:
    repository.upsert_vad_prediction(
        clip_id=clip_id,
        stream_id=stream_id,
        ts_start=ts_start,
        ts_end=ts_end,
        label=label,
        confidence=confidence,
        extra=extra,
    )
    with _lock:
        _pending_created_at.setdefault(clip_id, time.time())
        _cleanup_pending()

    return try_assemble_event(clip_id)

def try_assemble_event(clip_id: int) -> Optional[dict[str, Any]]:
    frames = repository.get_frame_batch(clip_id)
    vad = repository.get_vad_prediction(clip_id)

    if not frames or not vad:
        return None

    existing = repository.get_event_by_clip_id(clip_id)
    if existing:
        with _lock:
            _pending_created_at.pop(clip_id, None)
        return existing

    payload = build_event_payload(frames_batch=frames, vad_pred=vad)

    repository.insert_event_if_missing(
        clip_id=payload["clip_id"],
        stream_id=payload["stream_id"],
        ts_start=payload["ts_start"],
        ts_end=payload["ts_end"],
        label=payload["label"],
        confidence=payload["confidence"],
        frames=payload["frames"],
        vad=payload["vad"],
    )
    saved = repository.get_event_by_clip_id(clip_id)
    with _lock:
        _pending_created_at.pop(clip_id, None)
    return saved