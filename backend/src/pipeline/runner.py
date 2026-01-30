from __future__ import annotations

import threading
import time
from typing import Optional, Dict, Any

import cv2

from src.frame_selector.runtime_selector import FrameSelector
from src.frame_selector.config import FrameSelectorConfig
from src.vad.flashback_vad import FlashbackVAD, VADOutput
from src.pipeline.kg_stub import DummyAugmentor


class PipelineRunner:
    def __init__(self, cfg: FrameSelectorConfig, thesis_root: str):
        #create the frame selector, VAD and KG object
        self.selector = FrameSelector(cfg)
        self.vad = FlashbackVAD(thesis_root=thesis_root)
        self.kg = DummyAugmentor()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._latest_frame_bgr = None
        self._event_id = 0
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

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_payload

    def latest_frame(self):
        with self._lock:
            return None if self._latest_frame_bgr is None else self._latest_frame_bgr.copy()

    def _overlay(self, frame_bgr, vad_out: VADOutput | None):
        if frame_bgr is None:
            return None

        out = frame_bgr.copy()
        if vad_out is None:
            cv2.putText(out, "VAD: waiting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            return out

        label = vad_out.label.upper()
        conf = vad_out.confidence
        color = (0, 255, 0) if label == "NORMAL" else (0, 0, 255)

        cv2.putText(out, f"VAD: {label}  conf={conf:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if vad_out.top_caption:
            cv2.putText(out, f"Caption: {vad_out.top_caption[:70]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

        return out
#runs the VAD in a loop
    def _loop(self) -> None:
        last_vad: VADOutput | None = None

        while not self._stop.is_set():
            batch = self.selector.get_batch(timeout=0.5)
            if batch is None:
                # Even before first batch: show a “waiting” frame (latest ring frame if exists)
                try:
                    ring_list = self.selector._ring.snapshot()  # type: ignore
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
            }

            # Use a frame from the batch for the video stream, to get vadz
            mid = len(batch.frames) // 2
            frame = batch.frames[mid].frame_bgr

            with self._lock:
                self._latest_payload = payload
                self._latest_frame_bgr = self._overlay(frame, vad_out)
                self._event_id += 1
