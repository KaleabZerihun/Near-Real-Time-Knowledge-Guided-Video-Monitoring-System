from __future__ import annotations

import os
import time
import cv2

from src.frame_selector.config import FrameSelectorConfig
from src.frame_selector.runtime_selector import FrameSelector
from src.vad.flashback_vad import FlashbackVAD


def find_rtvad_root() -> str:
    # scripts/ is inside backend/, so RT-VAD is at project root: ../RT-VAD
    here = os.path.dirname(__file__)
    rtvad = os.path.abspath(os.path.join(here, "..", "..", "RT-VAD"))
    return rtvad


def main():
    rtvad_root = find_rtvad_root()

    # --- Darik FrameSelector config (reuse as-is) ---
    cfg = FrameSelectorConfig(
        source=0,
        source_id="webcam0",
        target_fps=8.0,         # drop to 4.0 if slow
        resize_hw=(224, 224),
        clip_len=16,
        stride=8,
        frame_ring_maxlen=256,
        max_batches=8,
    )

    selector = FrameSelector(cfg)
    vad = FlashbackVAD(
        rtvad_root=rtvad_root,
        top_k=10,
        anomaly_threshold=0.5,
    )

    print("\n=== RUNNING LIVE PIPELINE ===")
    print(f"RT-VAD root: {rtvad_root}")
    print("Press 'q' on the webcam window to quit.\n")

    selector.start()

    window_name = "LIVE (Darik selector → Sponsor VAD)  |  press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_print = 0.0

    try:
        while True:
            # 1) Get next batch from Darik’s selector
            batch = selector.get_batch(timeout=0.5)

            # Show metrics periodically (Darik output)
            now = time.time()
            if now - last_print >= 1.0:
                m = selector.get_metrics()
                print(
                    f"[METRICS] ring={m.ring_size}/{cfg.frame_ring_maxlen} "
                    f"q={m.batch_queue_size}/{cfg.max_batches} "
                    f"cap_fps~{m.capture_fps_est:.1f} sel_fps~{m.selected_fps_est:.1f} "
                    f"dropped_frames={m.dropped_frames} dropped_batches={m.dropped_batches}"
                )
                last_print = now

            if batch is None:
                # still allow quitting
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # 2) Preview: show the SAME frame you feed into VAD (middle frame)
            mid = len(batch.frames) // 2
            frame = batch.frames[mid].frame_bgr  # already 224x224
            preview = frame.copy()

            # 3) Run sponsor VAD (via wrapper)
            try:
                out = vad.predict(batch)
            except Exception as e:
                print(f"[VAD ERROR] {e}")
                # still show webcam
                cv2.imshow(window_name, preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # 4) Overlay VAD output on the preview
            text1 = f"clip={out.clip_id}  label={out.label}  score={out.confidence:.3f}"
            cap = out.top_caption.replace("Normal:", "").replace("Anomalous:", "").strip()
            text2 = f"top: {cap[:60]}"

            color = (0, 255, 0) if out.label == "normal" else (0, 0, 255)
            cv2.putText(preview, text1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(preview, text2, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # 5) Show webcam window
            cv2.imshow(window_name, preview)

            # 6) Print VAD outputs (this is what you wanted)
            print(f"[VAD] clip={out.clip_id} {out.label.upper()} score={out.confidence:.3f} | {cap}")

            # Quit key
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        selector.stop()
        cv2.destroyAllWindows()
        print("\nStopped cleanly.")


if __name__ == "__main__":
    main()
