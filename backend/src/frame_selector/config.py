from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class FrameSelectorConfig:
    # Central configuration for capture/selection/batching. Used in Steps 2–4.
    # Video source: 0 for default webcam, a filepath like "data/demo.mp4",
    # or "browser_upload" for frames streamed from the frontend.
    source: int | str = 0
    source_id: str = "webcam0"

    # Selection / sampling
    select_every: int = 2

    # Preprocessing
    resize_hw: Tuple[int, int] = (224, 224)  # (H, W)

    capture_hw: Tuple[int, int] = (640, 480)  # (W, H)
    capture_fps: float = 30.0
    capture_fourcc: str = "MJPG"
    capture_backend: str = "auto"

    # Batching
    clip_len: int = 16
    stride: int = 8  # overlap control

    # Buffering / backpressure
    frame_ring_maxlen: int = 256
    max_batches: int = 8
    drop_policy: str = "drop_oldest"
