from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ClipMetrics:
    """Performance data captured for a single processed clip."""

    clip_id: int
    stream_id: str
    recorded_at: float          # unix timestamp when the clip finished processing

    # Per-stage timing (milliseconds)
    vad_inference_ms: float     # time spent inside VAD.predict()
    kg_inference_ms: float      # time spent inside KG.augment()
    db_write_ms: float          # time spent writing frames + prediction to DB

    # End-to-end latency: clip ts_start → result ready (ms)
    e2e_latency_ms: float

    # VAD result
    label: Optional[str]
    confidence: Optional[float]
    is_anomaly: bool

    # Frame-selector health snapshot at time of clip
    capture_fps: Optional[float]
    selected_fps: Optional[float]
    queue_depth: Optional[int]
    dropped_frames: Optional[int]
    dropped_batches: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricsSummary:
    """Aggregate statistics computed over the last N clips."""

    window_clips: int           # how many clips are in the window
    since: float                # unix timestamp of oldest clip in window

    # VAD inference timing
    mean_vad_ms: float
    max_vad_ms: float
    p95_vad_ms: float

    # End-to-end latency
    mean_e2e_ms: float
    max_e2e_ms: float
    p95_e2e_ms: float

    # Throughput
    throughput_clips_per_sec: float   # clips processed per second

    # Detection quality
    anomaly_rate: float               # fraction of clips labelled anomaly
    mean_confidence: float            # mean VAD confidence across all clips

    # Frame selector health (latest snapshot)
    capture_fps: Optional[float]
    selected_fps: Optional[float]
    dropped_frames_total: Optional[int]
    dropped_batches_total: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)
