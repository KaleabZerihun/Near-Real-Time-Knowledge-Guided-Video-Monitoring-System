from __future__ import annotations
import time
import threading
from typing import Optional

import cv2

from .sampling import FrameSampler
from .buffer import FrameRingBuffer, DropOldestBatchQueue
from .config import FrameSelectorConfig
from .types import FramePacket, ClipBatch, FrameSelectorMetrics
from .selector import IFrameSelector

class FrameSelector(IFrameSelector):
    # Frame Selector runtime:
    #   - Captures frames from an OpenCV source
    #   - Resizes frames
    #   - Applies frame selection (target_fps sampling;)
    #   - Buffers selected frames in a bounded ring buffer (backpressure)
    #   - Builds ClipBatch outputs using clip_len and stride
    #   - Enqueues batches in a bounded queue with drop-oldest policy (backpressure)
    #   - Tracks metrics + logs queue depth/drop counters periodically

    def __init__(self, cfg: FrameSelectorConfig):
        self.cfg = cfg

        # logging cadence (seconds)
        self._last_log_ts = 0.0
        self._log_every_sec = 2.0

        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()

        # backpressure structures
        self._ring = FrameRingBuffer(maxlen=cfg.frame_ring_maxlen)
        self._batch_q = DropOldestBatchQueue(maxsize=cfg.max_batches)

        # ids
        self._frame_id = 0
        self._clip_id = 0

        # threads
        self._capture_thread: Optional[threading.Thread] = None
        self._batch_thread: Optional[threading.Thread] = None

        # frame selection sampler (time-based downsampling)
        self._sampler = FrameSampler(target_fps=self.cfg.target_fps)

        # counters
        self._dropped_batches = 0  # counts batches dropped by queue policy OR failed pushes

        # Separate FPS estimates
        self._capture_fps_est = 0.0   # how fast we READ from camera
        self._selected_fps_est = 0.0  # how fast we ACCEPT into ring after sampling

        # Capture FPS tracking (counts every frame read)
        self._cap_fps_last_ts = 0.0
        self._cap_fps_frames = 0

        # Selected FPS tracking (counts only selected frames)
        self._sel_fps_last_ts = 0.0
        self._sel_fps_frames = 0

        # Batch emission tracking
        self._last_emitted_start_frame_id: Optional[int] = None

    # ------------------ public API ------------------

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.cfg.source or 0)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.cfg.source}")

        self._stop.clear()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True)

        self._capture_thread.start()
        self._batch_thread.start()

    def stop(self) -> None:
        self._stop.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._batch_thread:
            self._batch_thread.join(timeout=1.0)

        if self._cap:
            self._cap.release()
            self._cap = None

    def get_batch(self, timeout: float = 0.5) -> Optional[ClipBatch]:
        return self._batch_q.pop(timeout=timeout)

    def get_metrics(self) -> FrameSelectorMetrics:
        return FrameSelectorMetrics(
            capture_fps_est=self._capture_fps_est,
            selected_fps_est=self._selected_fps_est,
            ring_size=len(self._ring),
            batch_queue_size=self._batch_q.qsize(),
            dropped_frames=self._ring.dropped_count(),  
            dropped_batches=self._dropped_batches,
        )

    # ------------------ internal helpers ------------------

    def _maybe_log_status(self) -> None:
        now = time.time()
        if self._last_log_ts == 0.0 or (now - self._last_log_ts) >= self._log_every_sec:
            m = self.get_metrics()
            print(
                f"[FrameSelector] ring={m.ring_size}/{self.cfg.frame_ring_maxlen} "
                f"q={m.batch_queue_size}/{self.cfg.max_batches} "
                f"cap_fps~{m.capture_fps_est:.1f} sel_fps~{m.selected_fps_est:.1f} "
                f"dropped_frames={m.dropped_frames} dropped_batches={m.dropped_batches}"
            )
            self._last_log_ts = now

    # ------------------ internal loops ------------------

    def _capture_loop(self) -> None:
        assert self._cap is not None

        # buffering reduction to reduce latency
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Initialize FPS trackers
        self._cap_fps_last_ts = 0.0
        self._cap_fps_frames = 0
        self._sel_fps_last_ts = 0.0
        self._sel_fps_frames = 0

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            ts = time.time()

            # resize (H, W) -> OpenCV wants (W, H)
            H, W = self.cfg.resize_hw
            frame_rs = cv2.resize(frame, (W, H))

            now = time.time()

            # Capture FPS (EVERY frame read)
            self._cap_fps_frames += 1
            if self._cap_fps_last_ts == 0.0:
                self._cap_fps_last_ts = now
            elif now - self._cap_fps_last_ts >= 1.0:
                dt = now - self._cap_fps_last_ts
                self._capture_fps_est = self._cap_fps_frames / dt
                self._cap_fps_frames = 0
                self._cap_fps_last_ts = now

            # Frame Selection — only push selected frames to ring
            if not self._sampler.should_select(ts):
                self._maybe_log_status()
                continue

            # Selected FPS (ONLY selected frames)
            self._sel_fps_frames += 1
            if self._sel_fps_last_ts == 0.0:
                self._sel_fps_last_ts = now
            elif now - self._sel_fps_last_ts >= 1.0:
                dt = now - self._sel_fps_last_ts
                self._selected_fps_est = self._sel_fps_frames / dt
                self._sel_fps_frames = 0
                self._sel_fps_last_ts = now

            # Create packet + push to ring buffer (bounded, drop-oldest)
            pkt = FramePacket(
                frame_id=self._frame_id,
                timestamp=ts,
                frame_bgr=frame_rs,
                source_id=self.cfg.source_id,
            )
            self._frame_id += 1
            self._ring.push(pkt)

            self._maybe_log_status()

    def _batch_loop(self) -> None:
        # Build clip batches from the most recent frames in the ring buffer.
        # Sliding window:
        #   - take last clip_len frames
        #   - only emit a new batch if start frame advanced by at least stride
        
        clip_len = self.cfg.clip_len
        stride = self.cfg.stride

        while not self._stop.is_set():
            ring_list = self._ring.snapshot()

            if len(ring_list) < clip_len:
                time.sleep(0.005)
                continue

            window = ring_list[-clip_len:]
            start_id = window[0].frame_id

            # enforce stride progression to avoid duplicate batches
            if self._last_emitted_start_frame_id is not None:
                if start_id - self._last_emitted_start_frame_id < stride:
                    time.sleep(0.005)
                    continue

            batch = ClipBatch(
                clip_id=self._clip_id,
                frames=window,
                ts_start=window[0].timestamp,
                ts_end=window[-1].timestamp,
            )
            self._clip_id += 1

            # bounded queue push with drop-oldest backpressure
            pushed, dropped_oldest = self._batch_q.push(batch)

            # count drops (drop-oldest policy) as "dropped batch"
            if dropped_oldest:
                self._dropped_batches += 1

            if not pushed:
                self._dropped_batches += 1
            else:
                self._last_emitted_start_frame_id = start_id

            self._maybe_log_status()
