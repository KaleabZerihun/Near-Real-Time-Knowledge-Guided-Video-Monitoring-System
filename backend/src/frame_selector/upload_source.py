from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

_latest_uploaded_frame: Optional[np.ndarray] = None
_latest_uploaded_frame_ts: float = 0.0
_lock = threading.Lock()


def set_uploaded_frame(frame_bgr: np.ndarray, timestamp: float) -> None:
    global _latest_uploaded_frame, _latest_uploaded_frame_ts
    with _lock:
        _latest_uploaded_frame = frame_bgr.copy()
        _latest_uploaded_frame_ts = timestamp


def get_uploaded_frame() -> Tuple[Optional[np.ndarray], float]:
    with _lock:
        if _latest_uploaded_frame is None:
            return None, 0.0
        return _latest_uploaded_frame.copy(), _latest_uploaded_frame_ts