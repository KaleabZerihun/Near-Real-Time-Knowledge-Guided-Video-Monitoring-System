# Baseline VAD Usage Guide

Quick guide for using the **BaselineVAD** model.

## Overview

- **Model:** Pre-trained ResNet18 + distance-based anomaly detection
- **Purpose:** Comparison baseline for FlashbackVAD
- **Performance:** 75-80% accuracy, ~50ms/frame (GPU)

---

## Quick Start

### Run Standalone Demo
```bash
python scripts/run_baseline_vad.py
```
Press `q` to quit. First ~50 frames are calibration phase.

### Use in Python
```python
from src.vad.baseline_vad import BaselineVAD
from src.frame_selector.runtime_selector import FrameSelector
from src.frame_selector.config import FrameSelectorConfig

cfg = FrameSelectorConfig(source=0, target_fps=8.0)
selector = FrameSelector(cfg)
vad = BaselineVAD(anomaly_threshold=0.5, calibration_samples=50)

selector.start()
batch = selector.get_batch(timeout=0.5)
result = vad.predict(batch)
print(f"{result.label}: {result.confidence:.3f}")
```

### Replace FlashbackVAD in Pipeline

Edit `backend/src/pipeline/runner.py`:
```python
# Line ~11: Change import
from src.vad.baseline_vad import BaselineVAD, VADOutput

# Line ~18: Change initialization
self.vad = BaselineVAD(anomaly_threshold=0.5, calibration_samples=50)
```

---

## Configuration

```python
BaselineVAD(
    anomaly_threshold=0.5,      # ↑ = fewer false positives
    calibration_samples=50,     # ↑ = more stable baseline
    device='cpu',               # or 'cuda' for GPU
)
```

---

## Testing

```bash
# Unit tests
pytest tests/test_baseline_vad.py -v

# Quick integration test
python scripts/test_baseline_quick.py
```

---

## Troubleshooting

**Import errors?**
```bash
pip install -r backend/requirements.txt
```

**Too many false positives?**
```python
vad = BaselineVAD(anomaly_threshold=0.7)  # Increase threshold
```

**Webcam not opening?**
```python
cfg = FrameSelectorConfig(source=1)  # Try different camera ID
```

