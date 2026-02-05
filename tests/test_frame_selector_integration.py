import time
import cv2
import pytest

from tests.conftest import FakeVideoCapture
from src.frame_selector.config import FrameSelectorConfig
from src.frame_selector.runtime_selector import FrameSelector


@pytest.mark.timeout(10)
def test_frame_selector_batches_and_limits(monkeypatch):
    # monkeypatch cv2.VideoCapture to return fake camera
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args, **_kwargs: FakeVideoCapture(fps=30))

    cfg = FrameSelectorConfig(
        source=0,
        source_id="fakecam0",
        resize_hw=(64, 64),
        target_fps=8.0,
        clip_len=16,
        stride=8,
        frame_ring_maxlen=64,
        max_batches=2,
    )

    selector = FrameSelector(cfg)
    selector.start()

    try:
        # wait for at least 2 batches
        batches = []
        t0 = time.time()
        while len(batches) < 2 and (time.time() - t0) < 6.0:
            b = selector.get_batch(timeout=1.0)
            if b is not None:
                batches.append(b)

        assert len(batches) >= 2, "Expected at least 2 batches to arrive"

        # batches always have clip_len frames
        for b in batches:
            assert len(b.frames) == cfg.clip_len

        # queue never exceeds limit (best effort: check via metrics)
        m = selector.get_metrics()
        assert m.batch_queue_size <= cfg.max_batches

    finally:
        selector.stop()
