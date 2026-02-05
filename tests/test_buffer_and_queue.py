import time
import numpy as np

from src.frame_selector.buffer import FrameRingBuffer, DropOldestBatchQueue
from src.frame_selector.types import FramePacket, ClipBatch


def test_ring_buffer_caps_and_counts_drops():
    ring = FrameRingBuffer(maxlen=3)

    for i in range(5):
        pkt = FramePacket(
            frame_id=i,
            timestamp=time.time(),
            frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
            source_id="test",
        )
        ring.push(pkt)

    assert len(ring) == 3
    # pushed 5 into maxlen 3 => dropped 2
    assert ring.dropped_count() == 2


def test_batch_queue_never_exceeds_limit_and_drops_oldest():
    q = DropOldestBatchQueue(maxsize=2)

    def make_batch(cid: int) -> ClipBatch:
        dummy = FramePacket(0, time.time(), np.zeros((2, 2, 3), dtype=np.uint8), "test")
        return ClipBatch(clip_id=cid, frames=[dummy], ts_start=dummy.timestamp, ts_end=dummy.timestamp)

    # Push 3 items into maxsize=2 => should drop oldest once
    pushed1, dropped1 = q.push(make_batch(1))
    pushed2, dropped2 = q.push(make_batch(2))
    pushed3, dropped3 = q.push(make_batch(3))

    assert pushed1 and pushed2 and pushed3
    assert dropped1 is False and dropped2 is False
    assert dropped3 is True

    # Queue size must never exceed 2
    assert q.qsize() == 2

    # Oldest should have been dropped, so remaining ids should be 2 and 3
    b_a = q.pop(timeout=0.1)
    b_b = q.pop(timeout=0.1)
    ids = sorted([b_a.clip_id, b_b.clip_id])
    assert ids == [2, 3]
