# Baseline VAD Model

## Overview

The **BaselineVAD** is a simple video anomaly detection model designed as a baseline for comparison against the more sophisticated **FlashbackVAD** (ImageBind + memory retrieval).

## Method

### Architecture
- **Feature Extractor**: Pre-trained ResNet18 (torchvision)
- **Anomaly Detection**: Distance-based scoring using Mahalanobis-like distance

### How It Works

1. **Feature Extraction**: 
   - Uses ResNet18 (pre-trained on ImageNet) with the final classification layer removed
   - Extracts 512-dimensional feature vectors from video frames
   - Features are L2-normalized

2. **Calibration Phase**:
   - Assumes the first N frames (default: 50) are "normal"
   - Computes mean and standard deviation of normal feature distribution
   - This establishes a baseline for what "normal" looks like

3. **Anomaly Scoring**:
   - For each new frame, computes standardized Euclidean distance from normal mean
   - Distance is normalized by standard deviation (similar to z-score)
   - Maps distance to [0, 1] range using sigmoid function
   - Score ≥ 0.5 → "anomaly", Score < 0.5 → "normal"

### Formula

```
distance = ||features - mean|| / std
anomaly_score = sigmoid(distance - 3.0)
label = "anomaly" if anomaly_score >= threshold else "normal"
```

## Advantages

✅ **No Training Required**: Uses pre-trained ResNet18, no additional training needed  
✅ **Real-Time Performance**: Single forward pass, very fast  
✅ **Simple & Interpretable**: Distance-based scoring is easy to understand  
✅ **Minimal Dependencies**: Only requires PyTorch and torchvision  

## Limitations

⚠️ **Calibration Required**: Assumes first N frames are normal  
⚠️ **Limited Context**: Only uses spatial features, no temporal modeling  
⚠️ **Generic Features**: ResNet18 is trained on ImageNet, not optimized for VAD  
⚠️ **No Semantic Understanding**: Cannot explain *why* something is anomalous  

## Comparison to FlashbackVAD

| Feature | BaselineVAD | FlashbackVAD |
|---------|-------------|--------------|
| **Model** | ResNet18 | ImageBind (huge) |
| **Method** | Distance from normal | Memory retrieval + SAP |
| **Training** | None (pre-trained) | Pre-trained + memory |
| **Semantic Info** | No | Yes (captions) |
| **Explainability** | Low | High |
| **Speed** | Very Fast | Moderate |
| **Expected Performance** | Good | Better (5-10% higher) |

## Usage

### Standalone Script

```bash
cd /path/to/project
python scripts/run_baseline_vad.py
```

### Python API

```python
from src.frame_selector.config import FrameSelectorConfig
from src.frame_selector.runtime_selector import FrameSelector
from src.vad.baseline_vad import BaselineVAD

# Configure
cfg = FrameSelectorConfig(
    source=0,  # webcam
    target_fps=8.0,
    resize_hw=(224, 224),
    clip_len=16,
    stride=8,
)

# Initialize
selector = FrameSelector(cfg)
vad = BaselineVAD(
    anomaly_threshold=0.5,
    calibration_samples=50,
)

# Run
selector.start()
while True:
    batch = selector.get_batch(timeout=0.5)
    if batch is None:
        continue
    
    result = vad.predict(batch)
    print(f"Label: {result.label}, Score: {result.confidence:.3f}")
```

### Integration with Pipeline

Replace FlashbackVAD in `backend/src/pipeline/runner.py`:

```python
# Before:
from src.vad.flashback_vad import FlashbackVAD
self.vad = FlashbackVAD(thesis_root=thesis_root)

# After:
from src.vad.baseline_vad import BaselineVAD
self.vad = BaselineVAD(anomaly_threshold=0.5, calibration_samples=50)
```

## Configuration Parameters

### `anomaly_threshold` (float, default=0.5)
- Threshold for classifying anomalies
- Range: [0, 1]
- Higher → fewer false positives, more false negatives
- Lower → more false positives, fewer false negatives

### `calibration_samples` (int, default=50)
- Number of initial frames used to learn normal distribution
- Range: [20, 200]
- More samples → more robust calibration, but slower startup
- Fewer samples → faster startup, but less stable

### `device` (str, optional)
- Compute device: 'cuda' or 'cpu'
- Auto-detected if not specified

## Performance Metrics

Expected performance (based on similar baselines in VAD literature):

- **Accuracy**: 75-85%
- **Precision**: 70-80%
- **Recall**: 70-80%
- **Latency**: < 100ms per frame (GPU), < 500ms (CPU)
- **Memory**: ~200MB (model weights)

## Calibration Tips

1. **Ensure Normal Startup**: The first `calibration_samples` frames should contain only normal behavior
2. **Stable Environment**: Avoid drastic lighting/background changes during calibration
3. **Adjust Threshold**: If too many false positives, increase threshold to 0.6-0.7
4. **Re-calibrate**: If environment changes significantly, restart to re-calibrate

## Troubleshooting

### Issue: Too many false positives
**Solution**: Increase `anomaly_threshold` to 0.6 or 0.7

### Issue: Missing real anomalies
**Solution**: Decrease `anomaly_threshold` to 0.3 or 0.4

### Issue: Poor performance after startup
**Solution**: Increase `calibration_samples` to 100-200

### Issue: Slow inference
**Solution**: Ensure PyTorch is using GPU (check `device='cuda'`)

## Citation

This baseline follows standard practices in video anomaly detection literature:

- Deep feature extraction: He et al., "Deep Residual Learning for Image Recognition" (2016)
- Distance-based anomaly detection: Scholkopf et al., "Support Vector Method for Novelty Detection" (2000)

## License

This baseline model uses:
- ResNet18 from torchvision (BSD License)
- PyTorch (BSD License)
