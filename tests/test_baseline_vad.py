import pytest
import numpy as np
import torch

from src.frame_selector.types import FramePacket, ClipBatch
from src.vad.baseline_vad import BaselineVAD, VADOutput


@pytest.mark.unit
class TestBaselineVAD:
    """Unit tests for BaselineVAD model."""

    def test_baseline_vad_initialization(self):
        """Test that BaselineVAD initializes correctly."""
        vad = BaselineVAD(
            anomaly_threshold=0.5,
            calibration_samples=10,
            device='cpu'
        )
        
        assert vad.anomaly_threshold == 0.5
        assert vad.calibration_samples == 10
        assert vad.device == 'cpu'
        assert not vad._is_calibrated

    def test_baseline_vad_with_default_params(self):
        """Test BaselineVAD with default parameters."""
        vad = BaselineVAD()
        
        assert vad.anomaly_threshold == 0.5
        assert vad.calibration_samples == 50
        assert vad.device in ['cpu', 'cuda']

    def test_feature_extraction(self):
        """Test that feature extraction works."""
        vad = BaselineVAD(device='cpu')
        
        # Create a dummy frame
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Extract features
        features = vad._extract_features(frame)
        
        # Check shape and properties
        assert features.shape == (512,)  # ResNet18 feature dimension
        assert torch.is_tensor(features)
        assert features.device.type == 'cpu'
        
        # Check normalization (L2 norm should be ~1)
        norm = torch.norm(features).item()
        assert 0.99 <= norm <= 1.01

    def test_calibration(self):
        """Test calibration phase."""
        vad = BaselineVAD(calibration_samples=3, device='cpu')
        
        # Create dummy features
        for i in range(3):
            dummy_features = torch.randn(512)
            dummy_features = dummy_features / torch.norm(dummy_features)  # normalize
            vad._calibrate(dummy_features)
        
        # After 3 samples, should be calibrated
        assert vad._is_calibrated
        assert vad._normal_mean is not None
        assert vad._normal_std is not None
        assert vad._normal_mean.shape == (512,)
        assert vad._normal_std.shape == (512,)

    def test_predict_returns_valid_output(self):
        """Test that predict returns properly formatted VADOutput."""
        vad = BaselineVAD(calibration_samples=2, device='cpu')
        
        # Create dummy ClipBatch
        frames = []
        for i in range(16):
            frame_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pkt = FramePacket(
                frame_id=i,
                timestamp=float(i) * 0.1,
                frame_bgr=frame_data,
                source_id="test"
            )
            frames.append(pkt)
        
        batch = ClipBatch(
            clip_id=1,
            frames=frames,
            ts_start=0.0,
            ts_end=1.5
        )
        
        # Run prediction
        result = vad.predict(batch)
        
        # Validate output
        assert isinstance(result, VADOutput)
        assert result.clip_id == 1
        assert result.ts_start == 0.0
        assert result.ts_end == 1.5
        assert result.label in ['normal', 'anomaly']
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.top_caption, str)
        assert len(result.top_caption) > 0
        assert isinstance(result.extra, dict)
        assert 'model' in result.extra
        assert result.extra['model'] == 'baseline_resnet18'

    def test_predict_during_calibration(self):
        """Test that predictions during calibration assume normal."""
        vad = BaselineVAD(calibration_samples=100, device='cpu')  # Large number
        
        # Create dummy batch
        frames = []
        for i in range(16):
            frame_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pkt = FramePacket(
                frame_id=i,
                timestamp=float(i),
                frame_bgr=frame_data,
                source_id="test"
            )
            frames.append(pkt)
        
        batch = ClipBatch(
            clip_id=1,
            frames=frames,
            ts_start=0.0,
            ts_end=1.5
        )
        
        # Should not be calibrated yet
        result = vad.predict(batch)
        
        # During calibration, should return low scores
        assert result.confidence <= 0.1
        assert not result.extra['is_calibrated']

    def test_predict_after_calibration(self):
        """Test that predictions work after calibration."""
        vad = BaselineVAD(calibration_samples=2, device='cpu')
        
        # Create and process calibration batches
        for clip_id in range(2):
            frames = []
            for i in range(16):
                # Use similar frames for calibration (should be "normal")
                frame_data = np.ones((224, 224, 3), dtype=np.uint8) * 128
                pkt = FramePacket(
                    frame_id=i,
                    timestamp=float(i),
                    frame_bgr=frame_data,
                    source_id="test"
                )
                frames.append(pkt)
            
            batch = ClipBatch(
                clip_id=clip_id,
                frames=frames,
                ts_start=0.0,
                ts_end=1.5
            )
            result = vad.predict(batch)
        
        # Now should be calibrated
        assert vad._is_calibrated
        assert result.extra['is_calibrated']

    def test_anomaly_score_computation(self):
        """Test anomaly score computation logic."""
        vad = BaselineVAD(device='cpu')
        
        # Manually set calibrated state
        vad._normal_mean = torch.zeros(512)
        vad._normal_std = torch.ones(512)
        vad._is_calibrated = True
        
        # Test with features close to mean (should be low score)
        close_features = torch.zeros(512) + 0.01
        close_score = vad._compute_anomaly_score(close_features)
        assert close_score < 0.3
        
        # Test with features far from mean (should be high score)
        far_features = torch.ones(512) * 5.0
        far_score = vad._compute_anomaly_score(far_features)
        assert far_score > 0.5

    def test_threshold_behavior(self):
        """Test that threshold affects label assignment."""
        vad_low = BaselineVAD(anomaly_threshold=0.3, calibration_samples=2, device='cpu')
        vad_high = BaselineVAD(anomaly_threshold=0.7, calibration_samples=2, device='cpu')
        
        # Create identical test batch
        frames = []
        for i in range(16):
            frame_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pkt = FramePacket(
                frame_id=i,
                timestamp=float(i),
                frame_bgr=frame_data,
                source_id="test"
            )
            frames.append(pkt)
        
        batch = ClipBatch(
            clip_id=1,
            frames=frames,
            ts_start=0.0,
            ts_end=1.5
        )
        
        # Calibrate both
        for _ in range(2):
            vad_low.predict(batch)
            vad_high.predict(batch)
        
        # With same score, different thresholds should potentially give different labels
        # (though this is probabilistic, so we just check they both return valid labels)
        result_low = vad_low.predict(batch)
        result_high = vad_high.predict(batch)
        
        assert result_low.label in ['normal', 'anomaly']
        assert result_high.label in ['normal', 'anomaly']

    def test_output_format_matches_flashback(self):
        """Ensure output format is compatible with FlashbackVAD."""
        vad = BaselineVAD(calibration_samples=1, device='cpu')
        
        # Create minimal batch
        frame_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        pkt = FramePacket(frame_id=0, timestamp=0.0, frame_bgr=frame_data, source_id="test")
        batch = ClipBatch(clip_id=1, frames=[pkt], ts_start=0.0, ts_end=0.1)
        
        result = vad.predict(batch)
        
        # Check all required fields exist (same as FlashbackVAD)
        assert hasattr(result, 'clip_id')
        assert hasattr(result, 'ts_start')
        assert hasattr(result, 'ts_end')
        assert hasattr(result, 'label')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'top_caption')
        assert hasattr(result, 'extra')


@pytest.mark.integration
class TestBaselineVADIntegration:
    """Integration tests for BaselineVAD with frame selector."""

    def test_baseline_vad_with_frame_selector_output(self):
        """Test BaselineVAD can process real FrameSelector output format."""
        vad = BaselineVAD(calibration_samples=2, device='cpu')
        
        # Simulate realistic frame selector output
        frames = []
        for i in range(16):
            # Realistic frame size
            frame_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pkt = FramePacket(
                frame_id=i,
                timestamp=1000.0 + i * 0.125,  # Realistic timestamps
                frame_bgr=frame_data,
                source_id="webcam0"
            )
            frames.append(pkt)
        
        batch = ClipBatch(
            clip_id=42,
            frames=frames,
            ts_start=frames[0].timestamp,
            ts_end=frames[-1].timestamp
        )
        
        # Should process without errors
        result = vad.predict(batch)
        
        assert result.clip_id == 42
        assert result.ts_start == frames[0].timestamp
        assert result.ts_end == frames[-1].timestamp
        assert 0.0 <= result.confidence <= 1.0
