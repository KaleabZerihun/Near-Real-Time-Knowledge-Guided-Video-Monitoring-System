import pytest
import numpy as np
from src.frame_selector.config import FrameSelectorConfig
from src.frame_selector.types import FramePacket, ClipBatch


@pytest.mark.unit
class TestFrameSelectorConfig:
    """Tests for FrameSelectorConfig."""

    def test_default_config(self):
        """Test creating config with all defaults."""
        cfg = FrameSelectorConfig()
        assert cfg.source == 0
        assert cfg.source_id == "webcam0"
        assert cfg.target_fps == 8.0
        assert cfg.resize_hw == (224, 224)
        assert cfg.clip_len == 16
        assert cfg.stride == 8
        assert cfg.frame_ring_maxlen == 256
        assert cfg.max_batches == 8
        assert cfg.drop_policy == "drop_oldest"

    def test_custom_config(self):
        """Test creating config with custom values."""
        cfg = FrameSelectorConfig(
            source="video.mp4",
            source_id="file_001",
            target_fps=15.0,
            resize_hw=(480, 480),
            clip_len=32,
            stride=16,
            frame_ring_maxlen=512,
            max_batches=16
        )
        assert cfg.source == "video.mp4"
        assert cfg.source_id == "file_001"
        assert cfg.target_fps == 15.0
        assert cfg.resize_hw == (480, 480)
        assert cfg.clip_len == 32
        assert cfg.stride == 16
        assert cfg.frame_ring_maxlen == 512
        assert cfg.max_batches == 16

    def test_config_immutable(self):
        """Test that config is frozen (immutable)."""
        cfg = FrameSelectorConfig()
        with pytest.raises(AttributeError):
            cfg.target_fps = 10.0

    def test_config_with_webcam_source(self):
        """Test config with integer webcam source."""
        cfg = FrameSelectorConfig(source=1)
        assert cfg.source == 1

    def test_config_with_file_source(self):
        """Test config with file path source."""
        cfg = FrameSelectorConfig(
            source="/path/to/video.mp4",
            source_id="demo"
        )
        assert cfg.source == "/path/to/video.mp4"
        assert cfg.source_id == "demo"

    def test_config_with_rtsp_source(self):
        """Test config with RTSP stream source."""
        cfg = FrameSelectorConfig(
            source="rtsp://camera.local:554/stream",
            source_id="ip_camera"
        )
        assert isinstance(cfg.source, str)
        assert "rtsp" in cfg.source


@pytest.mark.unit
class TestFramePacket:
    """Tests for FramePacket type."""

    def test_create_frame_packet(self):
        """Test creating a frame packet."""
        frame_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        pkt = FramePacket(
            frame_id=1,
            timestamp=0.0,
            frame_bgr=frame_data,
            source_id="webcam0"
        )
        assert pkt.frame_id == 1
        assert pkt.timestamp == 0.0
        assert pkt.source_id == "webcam0"
        assert pkt.frame_bgr.shape == (224, 224, 3)

    def test_frame_packet_default_source(self):
        """Test frame packet with default source."""
        pkt = FramePacket(
            frame_id=1,
            timestamp=0.0,
            frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8)
        )
        assert pkt.source_id == "webcam0"

    def test_frame_packet_immutable(self):
        """Test that frame packet is frozen."""
        pkt = FramePacket(
            frame_id=1,
            timestamp=0.0,
            frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
            source_id="test"
        )
        with pytest.raises(AttributeError):
            pkt.frame_id = 2

    def test_frame_packet_large_array(self):
        """Test frame packet with large frame array."""
        large_frame = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
        pkt = FramePacket(
            frame_id=100,
            timestamp=100.5,
            frame_bgr=large_frame,
            source_id="hd_camera"
        )
        assert pkt.frame_bgr.shape == (1920, 1080, 3)


@pytest.mark.unit
class TestClipBatch:
    """Tests for ClipBatch type."""

    def test_create_empty_clip(self):
        """Test creating a clip with no frames."""
        clip = ClipBatch(
            clip_id=1,
            frames=[],
            ts_start=0.0,
            ts_end=0.5
        )
        assert clip.clip_id == 1
        assert len(clip.frames) == 0
        assert clip.ts_start == 0.0
        assert clip.ts_end == 0.5

    def test_create_clip_with_frames(self):
        """Test creating a clip with frames."""
        frames = []
        for i in range(16):
            frame = FramePacket(
                frame_id=i,
                timestamp=float(i) * 0.1,
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id="test"
            )
            frames.append(frame)
        
        clip = ClipBatch(
            clip_id=1,
            frames=frames,
            ts_start=0.0,
            ts_end=1.5
        )
        assert clip.clip_id == 1
        assert len(clip.frames) == 16
        assert clip.frames[0].frame_id == 0
        assert clip.frames[15].frame_id == 15

    def test_clip_batch_immutable(self):
        """Test that clip batch is frozen."""
        clip = ClipBatch(
            clip_id=1,
            frames=[],
            ts_start=0.0,
            ts_end=1.0
        )
        with pytest.raises(AttributeError):
            clip.clip_id = 2

    def test_clip_timestamp_ordering(self):
        """Test that clip maintains frame timestamp ordering."""
        frames = []
        for i in range(10):
            frame = FramePacket(
                frame_id=i,
                timestamp=float(i) * 0.1,
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id="test"
            )
            frames.append(frame)
        
        clip = ClipBatch(
            clip_id=5,
            frames=frames,
            ts_start=frames[0].timestamp,
            ts_end=frames[-1].timestamp
        )
        
        for i in range(len(clip.frames) - 1):
            assert clip.frames[i].timestamp <= clip.frames[i + 1].timestamp

    def test_clip_time_span(self):
        """Test that clip time span is consistent."""
        frames = []
        start_time = 100.0
        for i in range(8):
            frame = FramePacket(
                frame_id=i,
                timestamp=start_time + i * 0.125,
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id="test"
            )
            frames.append(frame)
        
        clip = ClipBatch(
            clip_id=1,
            frames=frames,
            ts_start=start_time,
            ts_end=start_time + 1.0
        )
        
        assert clip.ts_start == start_time
        assert clip.ts_end == start_time + 1.0
        assert clip.ts_end - clip.ts_start == pytest.approx(1.0)

    def test_clip_with_multiple_sources(self):
        """Test clip can contain frames from different sources."""
        frames = []
        for i in range(16):
            source = "cam_01" if i < 8 else "cam_02"
            frame = FramePacket(
                frame_id=i,
                timestamp=float(i) * 0.1,
                frame_bgr=np.zeros((224, 224, 3), dtype=np.uint8),
                source_id=source
            )
            frames.append(frame)
        
        clip = ClipBatch(
            clip_id=1,
            frames=frames,
            ts_start=0.0,
            ts_end=1.5
        )
        
        sources = set(f.source_id for f in clip.frames)
        assert sources == {"cam_01", "cam_02"}
