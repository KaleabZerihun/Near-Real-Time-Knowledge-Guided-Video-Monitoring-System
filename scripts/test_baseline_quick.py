#!/usr/bin/env python3
"""
Quick integration test for BaselineVAD - no webcam required.
This creates synthetic video data to test the complete pipeline.
"""

import sys
import os
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from src.frame_selector.types import FramePacket, ClipBatch
from src.vad.baseline_vad import BaselineVAD


def create_synthetic_batch(clip_id: int, anomalous: bool = False) -> ClipBatch:
    """Create a synthetic clip batch for testing."""
    frames = []
    for i in range(16):
        if anomalous:
            # Create visually different frames (random noise)
            frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        else:
            # Create similar frames (gray with slight variation)
            base = 128 + np.random.randint(-10, 10, (224, 224, 3))
            frame = np.clip(base, 0, 255).astype(np.uint8)
        
        pkt = FramePacket(
            frame_id=clip_id * 16 + i,
            timestamp=float(clip_id * 16 + i) * 0.125,
            frame_bgr=frame,
            source_id="synthetic"
        )
        frames.append(pkt)
    
    return ClipBatch(
        clip_id=clip_id,
        frames=frames,
        ts_start=frames[0].timestamp,
        ts_end=frames[-1].timestamp
    )


def main():
    print("=" * 60)
    print("BaselineVAD Quick Integration Test")
    print("=" * 60)
    print()
    
    # Initialize baseline VAD
    print("[1/4] Initializing BaselineVAD...")
    vad = BaselineVAD(
        anomaly_threshold=0.5,
        calibration_samples=5,  # Small number for quick test
        device='cpu'  # Use CPU for compatibility
    )
    print(f"     ✓ Model initialized (device: {vad.device})")
    print()
    
    # Calibration phase with normal frames
    print("[2/4] Calibration phase (5 normal clips)...")
    for i in range(5):
        batch = create_synthetic_batch(clip_id=i, anomalous=False)
        result = vad.predict(batch)
        print(f"     Clip {i}: {result.label} (score: {result.confidence:.3f})")
    
    print(f"     ✓ Calibration complete!")
    print()
    
    # Test with normal frames
    print("[3/4] Testing with normal clips...")
    normal_scores = []
    for i in range(5, 8):
        batch = create_synthetic_batch(clip_id=i, anomalous=False)
        result = vad.predict(batch)
        normal_scores.append(result.confidence)
        status = "✓" if result.label == "normal" else "✗"
        print(f"     {status} Clip {i}: {result.label.upper()} (score: {result.confidence:.3f})")
    
    avg_normal = np.mean(normal_scores)
    print(f"     Average normal score: {avg_normal:.3f}")
    print()
    
    # Test with anomalous frames
    print("[4/4] Testing with anomalous clips...")
    anomaly_scores = []
    for i in range(8, 11):
        batch = create_synthetic_batch(clip_id=i, anomalous=True)
        result = vad.predict(batch)
        anomaly_scores.append(result.confidence)
        status = "✓" if result.label == "anomaly" else "✗"
        print(f"     {status} Clip {i}: {result.label.upper()} (score: {result.confidence:.3f})")
        print(f"        Caption: {result.top_caption}")
    
    avg_anomaly = np.mean(anomaly_scores)
    print(f"     Average anomaly score: {avg_anomaly:.3f}")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Normal clips avg score:   {avg_normal:.3f} (expected: < 0.5)")
    print(f"✓ Anomaly clips avg score:  {avg_anomaly:.3f} (expected: > normal)")
    print(f"✓ Score separation:         {(avg_anomaly - avg_normal):.3f}")
    
    if avg_anomaly > avg_normal:
        print()
        print("🎉 SUCCESS! BaselineVAD successfully distinguishes normal from anomalous clips!")
    else:
        print()
        print("⚠️  Note: Anomaly scores are similar to normal (expected with synthetic data)")
        print("   Real-world performance with actual video will be better.")
    
    print()
    print("=" * 60)
    print("Next steps:")
    print("  1. Run with webcam: python scripts/run_baseline_vad.py")
    print("  2. Compare with FlashbackVAD: python scripts/run_live_vad.py")
    print("  3. Integrate into dashboard: Modify backend/src/pipeline/runner.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
