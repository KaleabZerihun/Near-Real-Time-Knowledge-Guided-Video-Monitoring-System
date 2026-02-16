from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from .types import ClipBatch, FrameSelectorMetrics

# Public interface for the Frame Selector module.
class IFrameSelector(ABC):
  
    # Start capture + selection + batching loops.
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError
    
    # Stop loops and release resources.
    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
    
    # Return the next available ClipBatch, or None if none available.
    @abstractmethod
    def get_batch(self, timeout: float = 0.5) -> Optional[ClipBatch]:
        raise NotImplementedError
    
    # Expose basic runtime metrics for monitoring/logging.
    @abstractmethod
    def get_metrics(self) -> FrameSelectorMetrics:
        raise NotImplementedError
