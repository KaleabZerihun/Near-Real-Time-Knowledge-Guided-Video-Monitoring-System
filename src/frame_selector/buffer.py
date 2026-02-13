from __future__ import annotations
from collections import deque
from threading import Lock
from typing import Deque, List, Tuple
import queue

from .types import FramePacket, ClipBatch

class FrameRingBuffer:
    # Thread-safe bounded ring buffer for FramePacket. 
    # Drops oldest frames when full. Counts drops.
    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        self._buf: Deque[FramePacket] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._dropped_frames = 0

    def push(self, pkt: FramePacket) -> None:
        with self._lock:
            # If deque is full, append() will drop one from the left automatically.
            if len(self._buf) == self._buf.maxlen:
                self._dropped_frames += 1
            self._buf.append(pkt)

    def snapshot(self) -> List[FramePacket]:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_frames


class DropOldestBatchQueue:
    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._q: "queue.Queue[ClipBatch]" = queue.Queue(maxsize=maxsize)

    def push(self, batch: ClipBatch) -> tuple[bool, bool]:
        dropped_oldest = False

        if self._q.full():
            try:
                _ = self._q.get_nowait()
                dropped_oldest = True
            except queue.Empty:
                pass

        try:
            self._q.put_nowait(batch)
            return True, dropped_oldest
        except queue.Full:
            return False, dropped_oldest

    def pop(self, timeout: float = 0.5):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()
