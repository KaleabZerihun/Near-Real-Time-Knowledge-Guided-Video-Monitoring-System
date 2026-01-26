from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FrameSelectorConfig:
    """
    Central configuration for capture/selection/batching.
    Used in Steps 2–4.
    """
    # Video source: 0 for default webcam, or a filepath like "data/demo.mp4"
    source: int | str = 0
    source_id: str = "webcam0"

    # Selection / sampling
    target_fps: float = 8.0

    # Preprocessing
    resize_hw: Tuple[int, int] = (224, 224)  # (H, W)

    # Batching
    clip_len: int = 16
    stride: int = 8  # overlap control

    # Buffering / backpressure
    frame_ring_maxlen: int = 256
    max_batches: int = 8
    drop_policy: str = "drop_oldest"
