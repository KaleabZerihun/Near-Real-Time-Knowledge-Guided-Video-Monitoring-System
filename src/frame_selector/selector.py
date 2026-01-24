from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .types import ClipBatch, FrameSelectorMetrics


class IFrameSelector(ABC):
    """
    Public interface for the Frame Selector module.
    Downstream modules (VAD, logger, dashboard) should depend on this interface,
    not on OpenCV internals.
    """

    @abstractmethod
    def start(self) -> None:
        """Start capture + selection + batching loops."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop loops and release resources."""
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, timeout: float = 0.5) -> Optional[ClipBatch]:
        """Return the next available ClipBatch, or None if none available."""
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self) -> FrameSelectorMetrics:
        """Expose basic runtime metrics for monitoring/logging."""
        raise NotImplementedError
