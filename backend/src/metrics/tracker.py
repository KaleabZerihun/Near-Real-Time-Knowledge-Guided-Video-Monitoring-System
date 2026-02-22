from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from src.metrics.models import ClipMetrics, MetricsSummary


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


class MetricsTracker:
    """
    Thread-safe in-memory rolling window of per-clip performance metrics.

    Keeps the last `maxlen` ClipMetrics records and can produce an
    aggregate MetricsSummary on demand.
    """

    def __init__(self, maxlen: int = 600) -> None:
        self._lock = threading.Lock()
        self._history: deque[ClipMetrics] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, m: ClipMetrics) -> None:
        """Append a new clip's metrics to the rolling window."""
        with self._lock:
            self._history.append(m)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def history(self, limit: int = 300) -> list[ClipMetrics]:
        """Return the most recent `limit` ClipMetrics, oldest first."""
        with self._lock:
            pts = list(self._history)
        return pts[-limit:]

    def summary(self, last_n: int = 300) -> Optional[MetricsSummary]:
        """
        Aggregate the most recent `last_n` clips into a MetricsSummary.
        Returns None if no data has been recorded yet.
        """
        with self._lock:
            pts = list(self._history)[-last_n:]

        if not pts:
            return None

        vad_ms   = sorted(m.vad_inference_ms for m in pts)
        e2e_ms   = sorted(m.e2e_latency_ms   for m in pts)
        confs    = [m.confidence for m in pts if m.confidence is not None]
        anomalies = sum(1 for m in pts if m.is_anomaly)

        # Throughput: clips / elapsed wall-clock seconds
        oldest = pts[0].recorded_at
        newest = pts[-1].recorded_at
        elapsed = max(newest - oldest, 1e-6)
        throughput = len(pts) / elapsed

        last = pts[-1]

        return MetricsSummary(
            window_clips=len(pts),
            since=oldest,
            mean_vad_ms=sum(vad_ms) / len(vad_ms),
            max_vad_ms=max(vad_ms),
            p95_vad_ms=_percentile(vad_ms, 95),
            mean_e2e_ms=sum(e2e_ms) / len(e2e_ms),
            max_e2e_ms=max(e2e_ms),
            p95_e2e_ms=_percentile(e2e_ms, 95),
            throughput_clips_per_sec=throughput,
            anomaly_rate=anomalies / len(pts),
            mean_confidence=sum(confs) / len(confs) if confs else 0.0,
            capture_fps=last.capture_fps,
            selected_fps=last.selected_fps,
            dropped_frames_total=last.dropped_frames,
            dropped_batches_total=last.dropped_batches,
        )
