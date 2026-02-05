import sys
import os
import time

print("STARTING demo_runner.py", flush=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.frame_selector.config import FrameSelectorConfig
from src.frame_selector.runtime_selector import FrameSelector


def main():
    cfg = FrameSelectorConfig(
        source=0,
        source_id="webcam0",
        resize_hw=(224, 224),
        target_fps=8.0,
        clip_len=16,
        stride=8,
        frame_ring_maxlen=256,
        max_batches=8,
    )

    print(
        f"Config: target_fps={cfg.target_fps}, clip_len={cfg.clip_len}, stride={cfg.stride}, "
        f"ring_max={cfg.frame_ring_maxlen}, max_batches={cfg.max_batches}",
        flush=True
    )
    # expected batch cadence (approx): stride frames at target_fps seconds
    expected_batch_sec = cfg.stride / cfg.target_fps
    print(f"Expected: about 1 batch every ~{expected_batch_sec:.2f}s (once warmed up).", flush=True)

    print("About to start FrameSelector (opening camera)...", flush=True)
    selector = FrameSelector(cfg)
    selector.start()
    print("FrameSelector started.", flush=True)

    print("Demo step: expecting batches with clip_len frames; Ctrl+C to stop.", flush=True)

    got = 0
    t0 = time.time()

    # Warm-up: give it a moment to accumulate frames for the first batch
    warmup_deadline = t0 + 2.0

    try:
        while True:
            batch = selector.get_batch(timeout=1.0)
            m = selector.get_metrics()

            # requirement #3: queue doesn't exceed limit
            assert m.batch_queue_size <= cfg.max_batches, "Queue exceeded max_batches!"

            if batch is None:
                # During warm-up, allow silence without failing
                if time.time() < warmup_deadline:
                    print(
                        f"[warmup] ring={m.ring_size} q={m.batch_queue_size} "
                        f"sel_fps~{m.selected_fps_est:.1f} cap_fps~{m.capture_fps_est:.1f}",
                        flush=True
                    )
                    continue

                print(
                    f"[no batch] ring={m.ring_size} q={m.batch_queue_size} "
                    f"sel_fps~{m.selected_fps_est:.1f} cap_fps~{m.capture_fps_est:.1f}",
                    flush=True
                )
                continue

            # requirement #2: batches always have clip_len frames
            assert len(batch.frames) == cfg.clip_len, (
                f"Batch had {len(batch.frames)} frames, expected {cfg.clip_len}"
            )

            got += 1
            print(
                f"[OK batch #{batch.clip_id}] frames={len(batch.frames)} "
                f"ts={batch.ts_start:.3f}->{batch.ts_end:.3f} "
                f"ring={m.ring_size} q={m.batch_queue_size} "
                f"sel_fps~{m.selected_fps_est:.1f} cap_fps~{m.capture_fps_est:.1f}",
                flush=True
            )

            # requirement #1: “batches are coming”
            if got >= 3:
                print("Demo PASS: batches are coming, clip_len correct, queue bounded.", flush=True)
                break

            # fail if nothing meaningful happens within 10 seconds after start
            if time.time() - t0 > 10:
                raise RuntimeError(
                    "Demo FAIL: did not receive at least 3 batches within 10 seconds.\n"
                    "Tips:\n"
                    "  - Make sure the webcam isn't locked (close Zoom/Teams/Discord)\n"
                    "  - Try source=1 instead of source=0\n"
                    "  - Confirm Step 0 webcam test still works\n"
                )

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...", flush=True)
    finally:
        selector.stop()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
