from __future__ import annotations

import threading
import time
from typing import Optional, Dict, Any, Tuple

import cv2
from src.db import repository
from datetime import datetime, timezone
from collections import deque

from src.frame_selector.runtime_selector import FrameSelector
from src.frame_selector.config import FrameSelectorConfig
from src.vad.flashback_vad import FlashbackVAD, VADOutput
from src.pipeline.kg_stub import DummyAugmentor
from src.logger.logger import InMemoryLogger


class PipelineRunner:
    def __init__(self, cfg: FrameSelectorConfig, rtvad_root: str, logger: InMemoryLogger | None = None):
        #create the frame selector, VAD and KG object
        self.selector = FrameSelector(cfg)
        self.vad = FlashbackVAD(rtvad_root=rtvad_root)
        self.kg = DummyAugmentor()
        self.logger = logger or InMemoryLogger()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._run_id: int | None = None
        self._lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._latest_frame_bgr = None
        self._event_id = 0
        # Store (timestamp, confidence) points for overtime performance graph
        # Keep it bounded so it doesn't grow forever.
        #overtime performance
        self._confidence_history: deque[Tuple[float, float]] = deque(maxlen=600)
    # run the frame selecor
    def start(self) -> None:
        self._run_id = repository.create_run(
            mode="vad+kg",
            model_version="flashback_vad",
            notes="PipelineRunner session"
        )
        self.selector.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.selector.stop()

        if self._run_id is not None:
            repository.finish_run(self._run_id)
            self._run_id = None

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

            # ---- Sponsor VAD runs here ----
            vad_out: VADOutput = self.vad.predict(batch)
            last_vad = vad_out
#####################################################################################
            # KG stub (Sprint 1)
            kg_out = self.kg.augment(vad_out)

            #create alerts based on VAD / KG outputs (in-memory logger)
            alerts = self.logger.handle(vad_out, kg_out)

            payload = {
                "event_id": self._event_id,
                "updated_at": time.time(),
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
            
            try:
                clip_id = int(vad_out.clip_id) if vad_out.clip_id is not None else None

                if clip_id is not None:
                    frames_data = []
                    for f in batch.frames:
                        frames_data.append({
                            "ts": getattr(f, "ts", None),
                        })

                    repository.upsert_frame_batch(
                        clip_id=clip_id,
                        stream_id="webcam0",
                        ts_start=float(vad_out.ts_start),
                        ts_end=float(vad_out.ts_end),
                        fps=None,
                        frames=frames_data,
                    )

                    repository.upsert_vad_prediction(
                        clip_id=clip_id,
                        stream_id="webcam0",
                        ts_start=float(vad_out.ts_start),
                        ts_end=float(vad_out.ts_end),
                        label=vad_out.label,
                        confidence=float(vad_out.confidence) if vad_out.confidence is not None else None,
                        extra=vad_out.extra or {},
                    )

                    repository.insert_event_if_missing(
                        clip_id=clip_id,
                        stream_id="webcam0",
                        ts_start=float(vad_out.ts_start),
                        ts_end=float(vad_out.ts_end),
                        label=vad_out.label,
                        confidence=float(vad_out.confidence) if vad_out.confidence is not None else None,
                        frames=frames_data,
                        vad={
                            "clip_id": vad_out.clip_id,
                            "ts_start": vad_out.ts_start,
                            "ts_end": vad_out.ts_end,
                            "label": vad_out.label,
                            "confidence": vad_out.confidence,
                            "top_caption": vad_out.top_caption,
                            "extra": vad_out.extra,
                        },
                    )

                detection_id = repository.insert_detection(
                    run_id=self._run_id,
                    occurred_at=datetime.now(timezone.utc).isoformat(),
                    camera_id="webcam0",
                    event_type=vad_out.label or "unknown",
                    vad_score=float(vad_out.confidence) if vad_out.confidence is not None else None,
                    kg_context={
                        "kg_validated": kg_out.kg_validated,
                        "explanation": kg_out.explanation,
                        "rules_fired": kg_out.rules_fired,
                    },
                    decision="alerted" if alerts else "logged",
                )

                for alert in alerts:
                    severity = "medium"
                    if hasattr(alert, "severity"):
                        severity = str(alert.severity).lower()
                        if severity not in {"low", "medium", "high"}:
                            severity = "medium"

                    repository.insert_alert(
                        detection_id=detection_id,
                        severity=severity,
                        status="new",
                        channel="dashboard",
                    )

                metrics = self.selector.get_metrics()
                repository.insert_system_metric(
                    run_id=self._run_id,
                    inference_ms=None,
                    fps=getattr(metrics, "selected_fps_est", None),
                    queue_depth=getattr(metrics, "batch_queue_size", None),
                    detections_cnt=1 if vad_out.label and vad_out.label.lower() != "normal" else 0,
                )

            except Exception as e:
                print(f"[WARN] failed to persist pipeline output: {e}")

            # Use a frame from the batch for the video stream, to get vadz
            mid = len(batch.frames) // 2
            frame = batch.frames[mid].frame_bgr
            now_ts = time.time()


            with self._lock:
                self._confidence_history.append((now_ts, float(vad_out.confidence)))
                self._latest_payload = payload
                self._latest_frame_bgr = self._overlay(frame, vad_out)
                self._event_id += 1
