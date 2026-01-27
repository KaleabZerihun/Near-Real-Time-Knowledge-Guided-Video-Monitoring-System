from __future__ import annotations

import os
import time
import cv2

from src.frame_selector.config import FrameSelectorConfig
from src.frame_selector.runtime_selector import FrameSelector
from src.vad.baseline_vad import BaselineVAD


def main():
    """
    Test script for Baseline VAD model.
    
    This script demonstrates the baseline anomaly detection model
    that can be compared against FlashbackVAD.
    """
    
    # --- Frame Selector config (same as FlashbackVAD for fair comparison) ---
    cfg = FrameSelectorConfig(
        source=0,
        source_id="webcam0",
        target_fps=8.0,
        resize_hw=(224, 224),
        clip_len=16,
        stride=8,
        frame_ring_maxlen=256,
        max_batches=8,
    )

    selector = FrameSelector(cfg)
    baseline_vad = BaselineVAD(
        anomaly_threshold=0.5,
        calibration_samples=50,  # First 50 frames assumed normal
    )

    print("\n=== RUNNING BASELINE VAD PIPELINE ===")
    print("Model: Pre-trained ResNet18 + Distance-based scoring")
    print("Press 'q' on the webcam window to quit.")
    print("\nCalibration Phase: First ~50 frames will be used to learn 'normal' behavior.\n")

    selector.start()

    window_name = "Baseline VAD Demo | Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_print = 0.0

    try:
        while True:
            # 1) Get next batch from frame selector
            batch = selector.get_batch(timeout=0.5)

            # Show metrics periodically
            now = time.time()
            if now - last_print >= 2.0:
                m = selector.get_metrics()
                print(
                    f"[METRICS] ring={m.ring_size}/{cfg.frame_ring_maxlen} "
                    f"q={m.batch_queue_size}/{cfg.max_batches} "
                    f"cap_fps~{m.capture_fps_est:.1f} sel_fps~{m.selected_fps_est:.1f}"
                )
                last_print = now

            if batch is None:
                # Allow quitting even during waiting
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # 2) Get middle frame for preview
            mid = len(batch.frames) // 2
            frame = batch.frames[mid].frame_bgr  # 224x224
            preview = cv2.resize(frame, (640, 640))  # Enlarge for better visibility

            # 3) Run baseline VAD
            try:
                out = baseline_vad.predict(batch)
            except Exception as e:
                print(f"[VAD ERROR] {e}")
                cv2.imshow(window_name, preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # 4) Overlay VAD output on preview
            label_text = f"{out.label.upper()}"
            score_text = f"Score: {out.confidence:.3f}"
            status = "CALIBRATING..." if not out.extra.get("is_calibrated", False) else "ACTIVE"
            
            # Color coding
            color = (0, 255, 0) if out.label == "normal" else (0, 0, 255)
            
            # Draw overlays
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(preview, label_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(preview, score_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(preview, out.top_caption, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show clip info
            cv2.putText(preview, f"Clip ID: {out.clip_id}", (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 5) Display
            cv2.imshow(window_name, preview)

            # 6) Print to console
            print(f"[VAD] clip={out.clip_id:4d} {out.label.upper():8s} score={out.confidence:.3f} | {out.top_caption}")

            # Quit key
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        selector.stop()
        cv2.destroyAllWindows()
        print("\nStopped cleanly.")


if __name__ == "__main__":
    main()
