import pytest
import threading
import time
from src.frame_selector.buffer import FrameRingBuffer, DropOldestBatchQueue
from src.frame_selector.types import FramePacket, ClipBatch
import numpy as np


@pytest.mark.unit
class TestFrameRingBuffer:
    """Tests for FrameRingBuffer thread-safe ring buffer."""

    def test_buffer_creation(self):
        """Test creating a buffer with valid size."""
        buf = FrameRingBuffer(maxlen=10)
        assert len(buf) == 0
        assert buf.dropped_count() == 0

    def test_buffer_invalid_size(self):
        """Test that invalid sizes are rejected."""
        with pytest.raises(ValueError, match="maxlen must be > 0"):
            FrameRingBuffer(maxlen=0)
        
        with pytest.raises(ValueError):
            FrameRingBuffer(maxlen=-5)

    def test_push_single_frame(self):
        """Test pushing a single frame."""
        buf = FrameRingBuffer(maxlen=10)
        frame = FramePacket(
            frame_id=1,
            timestamp=0.0,
            frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
            source_id="test"
        )
        buf.push(frame)
        assert len(buf) == 1
        assert buf.dropped_count() == 0

    def test_push_multiple_frames(self):
        """Test pushing multiple frames."""
        buf = FrameRingBuffer(maxlen=5)
        for i in range(3):
            frame = FramePacket(
                frame_id=i,
                timestamp=float(i),
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id="test"
            )
            buf.push(frame)
        
        assert len(buf) == 3
        assert buf.dropped_count() == 0

    def test_buffer_overflow_drops_oldest(self):
        """Test that buffer drops oldest frame when full."""
        buf = FrameRingBuffer(maxlen=3)
        
        for i in range(3):
            frame = FramePacket(
                frame_id=i,
                timestamp=float(i),
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id="test"
            )
            buf.push(frame)
        
        assert len(buf) == 3
        assert buf.dropped_count() == 0
        
        frame = FramePacket(
            frame_id=3,
            timestamp=3.0,
            frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
            source_id="test"
        )
        buf.push(frame)
        
        assert len(buf) == 3
        assert buf.dropped_count() == 1

    def test_snapshot_returns_list(self):
        """Test that snapshot returns a list copy."""
        buf = FrameRingBuffer(maxlen=5)
        frames = []
        for i in range(3):
            frame = FramePacket(
                frame_id=i,
                timestamp=float(i),
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id="test"
            )
            frames.append(frame)
            buf.push(frame)
        
        snapshot = buf.snapshot()
        assert len(snapshot) == 3
        assert isinstance(snapshot, list)
        assert snapshot[0].frame_id == 0
        assert snapshot[2].frame_id == 2

    def test_thread_safety_push(self):
        """Test thread-safe concurrent pushes."""
        buf = FrameRingBuffer(maxlen=100)
        errors = []
        
        def push_frames(thread_id, count):
            try:
                for i in range(count):
                    frame = FramePacket(
                        frame_id=thread_id * 1000 + i,
                        timestamp=float(thread_id * 1000 + i),
                        frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                        source_id=f"thread_{thread_id}"
                    )
                    buf.push(frame)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=push_frames, args=(i, 10))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(buf) > 0

    def test_thread_safety_snapshot_and_len(self):
        """Test thread-safe concurrent reads."""
        buf = FrameRingBuffer(maxlen=50)
        
        for i in range(20):
            frame = FramePacket(
                frame_id=i,
                timestamp=float(i),
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id="test"
            )
            buf.push(frame)
        
        sizes = []
        errors = []
        
        def read_concurrent():
            try:
                sizes.append(len(buf))
                snapshot = buf.snapshot()
                sizes.append(len(snapshot))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=read_concurrent) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert all(s >= 0 for s in sizes)


@pytest.mark.unit
class TestDropOldestBatchQueue:
    """Tests for DropOldestBatchQueue."""

    def test_queue_creation(self):
        """Test creating a batch queue."""
        q = DropOldestBatchQueue(maxsize=5)
        assert q.qsize() == 0

    def test_queue_invalid_size(self):
        """Test that invalid sizes are rejected."""
        with pytest.raises(ValueError, match="maxsize must be > 0"):
            DropOldestBatchQueue(maxsize=0)

    def test_push_single_batch(self):
        """Test pushing a single batch."""
        q = DropOldestBatchQueue(maxsize=5)
        batch = ClipBatch(
            clip_id=1,
            frames=[],
            ts_start=0.0,
            ts_end=1.0
        )
        success, dropped = q.push(batch)
        assert success is True
        assert dropped is False
        assert q.qsize() == 1

    def test_push_multiple_batches(self):
        """Test pushing multiple batches."""
        q = DropOldestBatchQueue(maxsize=5)
        for i in range(3):
            batch = ClipBatch(
                clip_id=i,
                frames=[],
                ts_start=float(i),
                ts_end=float(i + 1)
            )
            success, dropped = q.push(batch)
            assert success is True
            assert dropped is False
        
        assert q.qsize() == 3

    def test_queue_full_drops_oldest(self):
        """Test that queue drops oldest batch when full."""
        q = DropOldestBatchQueue(maxsize=3)
        
        for i in range(3):
            batch = ClipBatch(
                clip_id=i,
                frames=[],
                ts_start=float(i),
                ts_end=float(i + 1)
            )
            success, dropped = q.push(batch)
            assert success is True
            assert dropped is False
        
        batch = ClipBatch(
            clip_id=3,
            frames=[],
            ts_start=3.0,
            ts_end=4.0
        )
        success, dropped = q.push(batch)
        assert success is True
        assert dropped is True
        assert q.qsize() <= 3

    def test_pop_from_queue(self):
        """Test popping batches from queue."""
        q = DropOldestBatchQueue(maxsize=5)

        batch_ids = []
        for i in range(3):
            batch = ClipBatch(
                clip_id=i,
                frames=[],
                ts_start=float(i),
                ts_end=float(i + 1)
            )
            batch_ids.append(i)
            q.push(batch)

        for expected_id in batch_ids:
            batch = q.pop(timeout=0.1)
            assert batch is not None
            assert batch.clip_id == expected_id

    def test_pop_timeout(self):
        """Test that pop returns None on timeout when empty."""
        q = DropOldestBatchQueue(maxsize=5)
        batch = q.pop(timeout=0.1)
        assert batch is None

    def test_fifo_ordering(self):
        """Test that queue maintains FIFO ordering."""
        q = DropOldestBatchQueue(maxsize=10)

        for i in range(5):
            batch = ClipBatch(
                clip_id=i,
                frames=[],
                ts_start=float(i),
                ts_end=float(i + 1)
            )
            q.push(batch)
        
        for expected_id in range(5):
            batch = q.pop(timeout=0.1)
            assert batch is not None
            assert batch.clip_id == expected_id

    def test_concurrent_push_pop(self):
        """Test concurrent push and pop operations."""
        q = DropOldestBatchQueue(maxsize=10)
        results = []
        errors = []
        
        def producer(start_id, count):
            try:
                for i in range(count):
                    batch = ClipBatch(
                        clip_id=start_id + i,
                        frames=[],
                        ts_start=float(start_id + i),
                        ts_end=float(start_id + i + 1)
                    )
                    q.push(batch)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def consumer(count):
            try:
                for _ in range(count):
                    batch = q.pop(timeout=1.0)
                    if batch is not None:
                        results.append(batch.clip_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        prod_thread = threading.Thread(target=producer, args=(0, 5))
        cons_thread = threading.Thread(target=consumer, args=(5,))
        
        prod_thread.start()
        cons_thread.start()
        
        prod_thread.join(timeout=5.0)
        cons_thread.join(timeout=5.0)
        
        assert len(errors) == 0
        assert len(results) > 0
