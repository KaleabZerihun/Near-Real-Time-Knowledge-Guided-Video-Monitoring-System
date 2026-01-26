import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.frame_selector.config import FrameSelectorConfig
from src.frame_selector.runtime_selector import FrameSelector


def main():
    cfg = FrameSelectorConfig(
        source=0,
        source_id="webcam0",

        #frame selection target
        target_fps=8.0,

        #preprocessing
        resize_hw=(224, 224),

        #batching
        clip_len=16,
        stride=8,

        #buffering
        frame_ring_maxlen=256,
        max_batches=8,
    )

    selector = FrameSelector(cfg)
    selector.start()

    print("Running FrameSelector... press Ctrl+C to stop.")
    try:
        while True:
            batch = selector.get_batch(timeout=1.0)
            m = selector.get_metrics()

            if batch is None:
                print(
                    f"[no batch] ring={m.ring_size} q={m.batch_queue_size} "
                    f"selected_fps~{m.selected_fps_est:.1f} capture_fps~{m.capture_fps_est:.1f}"
                )
                continue

            print(
                f"[batch #{batch.clip_id}] frames={len(batch.frames)} "
                f"ts={batch.ts_start:.3f}->{batch.ts_end:.3f} "
                f"ring={m.ring_size} q={m.batch_queue_size} "
                f"selected_fps~{m.selected_fps_est:.1f} capture_fps~{m.capture_fps_est:.1f}"
            )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        selector.stop()


if __name__ == "__main__":
    main()
