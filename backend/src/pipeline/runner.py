from __future__ import annotations
import threading
import time
from typing import Optional, Dict, Any
import cv2
from src.frame_selector.runtime_selector import FrameSelector
from src.frame_selector.config import FrameSelectorConfig
from src.vad.flashback_vad import FlashbackVAD, VADOutput
from src.pipeline.kg_stub import DummyAugmentor
from src.logger.logger import InMemoryLogger
from src.events import service as events_service

class PipelineRunner:
    def __init__(self, cfg: FrameSelectorConfig, thesis_root: str, logger: InMemoryLogger | None = None):
        # create the frame selector, VAD and KG object
        self.selector = FrameSelector(cfg)
        self.vad = FlashbackVAD(thesis_root=thesis_root)
        self.kg = DummyAugmentor()
        self.logger = logger or InMemoryLogger()
        self.cfg = cfg
        self.stream_id = getattr(cfg, "source_id", None) or "stream0"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None
        self._latest_frame_bgr = None
        self._event_id = 0

    # run the frame selector
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

    def _overlay(self, frame_bgr, vad_out: VADOutput | None):
        if frame_bgr is None:
            return None

        out = frame_bgr.copy()
        if vad_out is None:
            cv2.putText(out, "VAD: waiting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            return out

        label = (vad_out.label or "").upper()
        conf = float(vad_out.confidence or 0.0)
        color = (0, 255, 0) if label == "NORMAL" else (0, 0, 255)

        cv2.putText(out, f"VAD: {label}  conf={conf:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if getattr(vad_out, "top_caption", None):
            cv2.putText(
                out,
                f"Caption: {vad_out.top_caption[:70]}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 30, 30),
                2,
            )
        return out

    # convert batch frames into small JSON refs (not raw images)
    def _frames_to_refs(self, batch) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for i, fp in enumerate(getattr(batch, "frames", []) or []):
            # Try to extract useful fields if they exist; otherwise keep minimal ref
            ts = getattr(fp, "ts", None)
            if ts is None:
                ts = getattr(fp, "timestamp", None)
            if ts is None:
                ts = getattr(fp, "t", None)
            idx = getattr(fp, "index", None)
            if idx is None:
                idx = getattr(fp, "idx", None)
            if idx is None:
                idx = i
            key = f"clip_{getattr(batch, 'clip_id', None) or 'unknown'}_frame_{idx}"
            refs.append(
                {
                    "key": key,
                    "index": int(idx) if idx is not None else i,
                    "ts": float(ts) if ts is not None else None,
                }
            )
        return refs
    
    # runs the VAD in a loop
    def _loop(self) -> None:
        last_vad: VADOutput | None = None

        while not self._stop.is_set():
            batch = self.selector.get_batch(timeout=0.5)
            if batch is None:
                try:
                    ring_list = self.selector._ring.snapshot()  
                    if ring_list:
                        frame = ring_list[-1].frame_bgr
                        with self._lock:
                            self._latest_frame_bgr = self._overlay(frame, last_vad)
                except Exception:
                    pass
                time.sleep(0.01)
                continue

            vad_out: VADOutput = self.vad.predict(batch)
            last_vad = vad_out

            kg_out = self.kg.augment(vad_out)

            alerts = self.logger.handle(vad_out, kg_out)

            # persist frames + prediction and auto-build event
            built_event: dict[str, Any] | None = None
            try:
                # store frames (small references only)
                frames_refs = self._frames_to_refs(batch)
                built_event = events_service.ingest_frames(
                    stream_id=self.stream_id,
                    clip_id=int(vad_out.clip_id),
                    ts_start=float(vad_out.ts_start),
                    ts_end=float(vad_out.ts_end),
                    fps=getattr(batch, "fps", None),
                    frames=frames_refs,
                )

                # store prediction
                built_event = events_service.ingest_prediction(
                    stream_id=self.stream_id,
                    clip_id=int(vad_out.clip_id),
                    ts_start=float(vad_out.ts_start),
                    ts_end=float(vad_out.ts_end),
                    label=vad_out.label,
                    confidence=float(vad_out.confidence),
                    extra=vad_out.extra or {},
                ) or built_event
            except Exception as e:
                print("[WARN] event ingest/build failed:", e)

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
                "db_event": built_event,
            }

            # Use a frame from the batch for the video stream, to get overlay
            mid = len(batch.frames) // 2 if getattr(batch, "frames", None) else 0
            frame = batch.frames[mid].frame_bgr if getattr(batch, "frames", None) else None

            with self._lock:
                self._latest_payload = payload
                self._latest_frame_bgr = self._overlay(frame, vad_out) if frame is not None else None
                self._event_id += 1