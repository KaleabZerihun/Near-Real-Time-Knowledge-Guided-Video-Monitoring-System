from __future__ import annotations
from typing import Any, Optional

def build_event_payload(
    *,
    frames_batch: dict[str, Any],
    vad_pred: dict[str, Any],
) -> dict[str, Any]:
    frames_clip_id = frames_batch.get("clip_id")
    vad_clip_id = vad_pred.get("clip_id")

    if frames_clip_id is None or vad_clip_id is None:
        raise ValueError("Missing clip_id in frames_batch or vad_pred")

    if int(frames_clip_id) != int(vad_clip_id):
        raise ValueError(f"clip_id mismatch: frames={frames_clip_id} vad={vad_clip_id}")

    stream_id = frames_batch.get("stream_id")
    if not stream_id:
        raise ValueError("Missing stream_id in frames_batch")

    vad_stream_id = vad_pred.get("stream_id")
    vad_out = dict(vad_pred)  
    if vad_stream_id and vad_stream_id != stream_id:
        extra = dict(vad_out.get("extra") or {})
        extra["stream_id_mismatch"] = {
            "frames_stream_id": stream_id,
            "vad_stream_id": vad_stream_id,
        }
        vad_out["extra"] = extra

    ts_start_f = frames_batch.get("ts_start")
    ts_end_f = frames_batch.get("ts_end")
    ts_start_v = vad_out.get("ts_start")
    ts_end_v = vad_out.get("ts_end")

    if ts_start_f is None or ts_end_f is None or ts_start_v is None or ts_end_v is None:
        raise ValueError("Missing ts_start/ts_end in frames_batch or vad_pred")

    ts_start = float(min(ts_start_f, ts_start_v))
    ts_end = float(max(ts_end_f, ts_end_v))

    label: Optional[str] = vad_out.get("label")
    confidence: Optional[float] = vad_out.get("confidence")

    frames = frames_batch.get("frames")
    if frames is None:
        raise ValueError("Missing frames in frames_batch")

    return {
        "stream_id": stream_id,
        "clip_id": int(frames_clip_id),
        "ts_start": ts_start,
        "ts_end": ts_end,
        "label": label,
        "confidence": confidence,
        "frames": frames,
        "vad": vad_out,
    }
