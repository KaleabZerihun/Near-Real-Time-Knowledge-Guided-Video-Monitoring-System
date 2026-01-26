from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    frame_bgr: np.ndarray
    source_id: str = "webcam0"


@dataclass(frozen=True)
class ClipBatch:
    clip_id: int
    frames: List[FramePacket]
    ts_start: float
    ts_end: float


@dataclass(frozen=True)
class FrameSelectorMetrics:
    capture_fps_est: float
    selected_fps_est: float
    ring_size: int
    batch_queue_size: int
    dropped_frames: int
    dropped_batches: int
