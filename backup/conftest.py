import time
import numpy as np

class FakeVideoCapture:
    # VideoCapture that generates synthetic frames.
    def __init__(self, source=0, fps=30, shape=(480, 640, 3)):      #shape is for the image arrays (height, width, color channels) in NumPy
        self.source = source
        self._opened = True
        self.fps = fps
        self.dt = 1.0 / fps
        self.shape = shape
        self._last = time.time()
        self._i = 0

    def isOpened(self):
        return self._opened
    
    # throttle to fps to mimic camera timing
    def read(self):
        now = time.time()
        if now - self._last < self.dt:
            time.sleep(max(0.0, self.dt - (now - self._last)))
        self._last = time.time()

        # synthetic frame content changes over time
        frame = np.zeros(self.shape, dtype=np.uint8)
        v = (self._i % 255)
        frame[:, :, :] = v
        self._i += 1

        return True, frame

    def release(self):
        self._opened = False

    def set(self, *_args, **_kwargs):
        return True
