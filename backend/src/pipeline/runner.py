from __future__ import annotations

import threading
import time
from typing import Optional, Dict, Any, Tuple

import cv2
from collections import deque

from src.frame_selector.runtime_selector import FrameSelector
from src.frame_selector.config import FrameSelectorConfig
from src.vad.flashback_vad import FlashbackVAD, VADOutput
from src.pipeline.kg_stub import DummyAugmentor
from src.logger.logger import InMemoryLogger
from src.events import service as events_service
from src.metrics.models import ClipMetrics
from src.metrics.tracker import MetricsTracker
from src.metrics import persistence as metrics_persistence


class PipelineRunner:
    def __init__(self, cfg: FrameSelectorConfig, thesis_root: str, logger: InMemoryLogger | None = None):
        #create the frame selector, VAD and KG object
        self.selector = FrameSelector(cfg)
        self.vad = FlashbackVAD(thesis_root=thesis_root)
        self.kg = DummyAugmentor()
        self.logger = logger or InMemoryLogger()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._latest_frame_bgr = None
        self._event_id = 0
        # Store (timestamp, confidence) points for overtime performance graph
        # Keep it bounded so it doesn't grow forever.
        #overtime performance
        self._confidence_history: deque[Tuple[float, float]] = deque(maxlen=600)
        self._metrics_tracker = MetricsTracker(maxlen=600)
    # run the frame selecor
    def start(self) -> None:
        self.selector.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.selector.stop()

    def metrics(self):
        return self.selector.get_metrics()

    def get_alerts(self):
        return self.logger.get_alerts()

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_payload

    def latest_frame(self):
        with self._lock:
            return None if self._latest_frame_bgr is None else self._latest_frame_bgr.copy()

    def confidence_history(self, limit: int = 300) -> list[dict]:
        with self._lock:
            pts = list(self._confidence_history)
        pts = pts[-limit:]
        return [{"t": t, "confidence": c} for t, c in pts]

    def performance_summary(self, last_n: int = 300) -> Optional[Dict[str, Any]]:
        s = self._metrics_tracker.summary(last_n=last_n)
        return s.to_dict() if s else None

    def performance_history(self, limit: int = 300) -> list[Dict[str, Any]]:
        return [m.to_dict() for m in self._metrics_tracker.history(limit=limit)]

    def _overlay(self, frame_bgr, vad_out: VADOutput | None):
        if frame_bgr is None:
            return None
        return frame_bgr.copy()


        # out = frame_bgr.copy()
        # if vad_out is None:
        #     cv2.putText(out, "VAD: waiting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        #     return out

        # label = vad_out.label.upper()
        # conf = vad_out.confidence
        # color = (0, 255, 0) if label == "NORMAL" else (0, 0, 255)

        # cv2.putText(out, f"VAD: {label}  conf={conf:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # if vad_out.top_caption:
        #     cv2.putText(out, f"Caption: {vad_out.top_caption[:70]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

        # return out
    #runs the VAD in a loop
    def _loop(self) -> None:
        last_vad: VADOutput | None = None

        while not self._stop.is_set():
            batch = self.selector.get_batch(timeout=0.5)
            if batch is None:
                # Even before first batch: show a “waiting” frame (latest ring frame if exists)
                try:
                    ring_list = self.selector._ring.snapshot()  #type: ignore
                    if ring_list:
                        frame = ring_list[-1].frame_bgr
                        with self._lock:
                            self._latest_frame_bgr = self._overlay(frame, last_vad)
                except Exception:
                    pass

                time.sleep(0.01)
                continue

            stream_id = batch.frames[0].source_id if batch.frames else "webcam0"

            # ---- Stage 1: VAD (timed) ----
            t0 = time.perf_counter()
            vad_out: VADOutput = self.vad.predict(batch)
            vad_ms = (time.perf_counter() - t0) * 1000
            last_vad = vad_out

            # ---- Stage 2: KG augmentor (timed) ----
            t1 = time.perf_counter()
            kg_out = self.kg.augment(vad_out)
            kg_ms = (time.perf_counter() - t1) * 1000

            # ---- Stage 3: Logger ----
            alerts = self.logger.handle(vad_out, kg_out)

            now_ts = time.time()
            payload = {
                "event_id": self._event_id,
                "updated_at": now_ts,
                "vad": {
                    "clip_id": vad_out.clip_id,
                    "ts_start": vad_out.ts_start,
                    "ts_end": vad_out.ts_end,
                    "label": vad_out.label,
                    "confidence": vad_out.confidence,
                    "top_caption": vad_out.top_caption,
                    "extra": vad_out.extra,
                },
                "kg_validated": kg_out.kg_validated,
                "explanation": kg_out.explanation,
                "rules_fired": kg_out.rules_fired,
                "alerts": [a.to_dict() for a in alerts],
            }

            # ---- Stage 4: DB persistence (timed) ----
            db_ms = 0.0
            try:
                frame_refs = [
                    {"key": f"clip{batch.clip_id}_f{fp.frame_id}", "ts": fp.timestamp, "index": i}
                    for i, fp in enumerate(batch.frames)
                ]
                t2 = time.perf_counter()
                events_service.ingest_frames(
                    stream_id=stream_id,
                    clip_id=batch.clip_id,
                    ts_start=batch.ts_start,
                    ts_end=batch.ts_end,
                    fps=None,
                    frames=frame_refs,
                )
                events_service.ingest_prediction(
                    stream_id=stream_id,
                    clip_id=batch.clip_id,
                    ts_start=vad_out.ts_start,
                    ts_end=vad_out.ts_end,
                    label=vad_out.label,
                    confidence=vad_out.confidence,
                    extra=vad_out.extra or {},
                )
                db_ms = (time.perf_counter() - t2) * 1000
            except Exception as e:
                print(f"[WARN] DB persistence failed for clip {batch.clip_id}: {e}")

            # ---- Stage 5: Record per-clip performance metrics ----
            sel_metrics = self.selector.get_metrics()
            e2e_ms = (now_ts - batch.ts_start) * 1000
            clip_m = ClipMetrics(
                clip_id=batch.clip_id,
                stream_id=stream_id,
                recorded_at=now_ts,
                vad_inference_ms=vad_ms,
                kg_inference_ms=kg_ms,
                db_write_ms=db_ms,
                e2e_latency_ms=e2e_ms,
                label=vad_out.label,
                confidence=vad_out.confidence,
                is_anomaly=(vad_out.label or "").lower() == "anomaly",
                capture_fps=sel_metrics.capture_fps_est,
                selected_fps=sel_metrics.selected_fps_est,
                queue_depth=sel_metrics.batch_queue_size,
                dropped_frames=sel_metrics.dropped_frames,
                dropped_batches=sel_metrics.dropped_batches,
            )
            self._metrics_tracker.record(clip_m)
            try:
                metrics_persistence.insert_clip_metrics(clip_m)
            except Exception as e:
                print(f"[WARN] Metrics persistence failed for clip {batch.clip_id}: {e}")

            # Use a frame from the batch for the video stream
            mid = len(batch.frames) // 2
            frame = batch.frames[mid].frame_bgr

            with self._lock:
                self._confidence_history.append((now_ts, float(vad_out.confidence)))
                self._latest_payload = payload
                self._latest_frame_bgr = self._overlay(frame, vad_out)
                self._event_id += 1
