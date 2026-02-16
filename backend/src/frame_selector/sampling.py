from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SamplingState:
    last_selected_ts: float = 0.0

class FrameSampler:
    # Time-based sampler. Selects frames at approximately target_fps by enforcing a minimum time delta.
    def __init__(self, target_fps: float):
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.target_fps = float(target_fps)
        self.min_dt = 1.0 / self.target_fps
        self.state = SamplingState()
    
    # Return True if enough time has passed since last selected frame.
    def should_select(self, ts: float) -> bool:
        if self.state.last_selected_ts == 0.0:
            self.state.last_selected_ts = ts
            return True

        if (ts - self.state.last_selected_ts) >= self.min_dt:
            self.state.last_selected_ts = ts
            return True

        return False
