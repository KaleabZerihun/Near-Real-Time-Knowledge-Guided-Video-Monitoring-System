from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SamplingState:
    frame_count: int = 0

class FrameSampler:
    # Time-based sampler. Selects frames at approximately select_every by enforcing a minimum time delta.
    def __init__(self, select_every: int = 2):
        if select_every <= 0:
            raise ValueError("select_every must be > 0")
        self.select_every = int(select_every)
        #self.min_dt = 1.0 / self.select_every
        self.state = SamplingState()
    
    # Return True if enough time has passed since last selected frame.
    def should_select(self) -> bool:
        self.state.frame_count += 1
        return self.state.frame_count % self.select_every == 0