from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from src.frame_selector.types import ClipBatch


@dataclass(frozen=True)
class VADOutput:
    """
    Standardized output format for all VAD models.
    Must match the format used by FlashbackVAD.
    """
    clip_id: int
    ts_start: float
    ts_end: float
    label: str               # "normal" or "anomaly"
    confidence: float        # anomaly score (0..1)
    top_caption: str
    extra: Dict[str, Any]


class BaselineVAD:
    """
    Baseline Video Anomaly Detection model for comparison against FlashbackVAD.
    
    Method:
    - Uses pre-trained ResNet18 for feature extraction
    - Computes anomaly score based on Mahalanobis distance from "normal" distribution
    - Simple, fast, and requires no training
    
    Design:
    - Assumes first N frames are "normal" for calibration
    - Computes mean and covariance of normal features
    - Anomaly score = normalized distance from normal distribution
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.5,
        calibration_samples: int = 50,
        device: Optional[str] = None,
    ):
        """
        Args:
            anomaly_threshold: Threshold above which predictions are labeled "anomaly"
            calibration_samples: Number of initial frames to use for normal distribution
            device: 'cuda' or 'cpu'
        """
        self.anomaly_threshold = float(anomaly_threshold)
        self.calibration_samples = int(calibration_samples)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained ResNet18 (lightweight, fast)
        self._model = models.resnet18(pretrained=True)
        # Remove final classification layer to get features
        self._model = torch.nn.Sequential(*list(self._model.children())[:-1])
        self._model.eval()
        self._model.to(self.device)

        # Image preprocessing (ImageNet normalization)
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Normal distribution statistics (computed during calibration)
        self._normal_mean: Optional[torch.Tensor] = None
        self._normal_std: Optional[torch.Tensor] = None
        self._is_calibrated = False
        self._calibration_features = []

        print(f"[BaselineVAD] Initialized. Device: {self.device}")
        print(f"[BaselineVAD] Calibration mode: will use first {self.calibration_samples} samples")

    def _extract_features(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Extract feature vector from a single frame.
        
        Args:
            frame_bgr: BGR frame (H, W, 3) uint8
            
        Returns:
            Feature tensor (512,) for ResNet18
        """
        # BGR -> RGB
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        
        # Apply ImageNet preprocessing
        frame_tensor = self._transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.inference_mode():
            features = self._model(frame_tensor)
            features = features.squeeze()  # (512,)
            features = F.normalize(features, p=2, dim=-1)
        
        return features

    def _calibrate(self, features: torch.Tensor) -> None:
        """
        Accumulate features for calibration. Once enough samples are collected,
        compute mean and std of normal distribution.
        
        Args:
            features: Feature vector from current frame
        """
        if self._is_calibrated:
            return
        
        self._calibration_features.append(features.cpu())
        
        if len(self._calibration_features) >= self.calibration_samples:
            # Compute statistics
            all_features = torch.stack(self._calibration_features)  # (N, 512)
            self._normal_mean = all_features.mean(dim=0).to(self.device)
            self._normal_std = all_features.std(dim=0).to(self.device) + 1e-6  # avoid div by 0
            
            self._is_calibrated = True
            self._calibration_features = []  # free memory
            
            print(f"[BaselineVAD] Calibration complete using {self.calibration_samples} samples")

    def _compute_anomaly_score(self, features: torch.Tensor) -> float:
        """
        Compute anomaly score based on distance from normal distribution.
        
        Uses normalized Euclidean distance (similar to z-score):
        score = ||features - mean|| / std
        
        Then applies sigmoid to map to [0, 1] range.
        
        Args:
            features: Feature vector from current frame
            
        Returns:
            Anomaly score in [0, 1]
        """
        if not self._is_calibrated:
            # During calibration, assume everything is normal
            return 0.0
        
        # Compute standardized distance
        diff = features - self._normal_mean
        normalized_diff = diff / self._normal_std
        distance = torch.norm(normalized_diff).item()
        
        # Map distance to [0, 1] using sigmoid
        # distance ~ 0 → score ~ 0 (normal)
        # distance >> 1 → score ~ 1 (anomaly)
        score = torch.sigmoid(torch.tensor(distance - 3.0)).item()  # shift for better calibration
        
        return float(score)

    def predict(self, batch: ClipBatch) -> VADOutput:
        """
        Process a ClipBatch and return anomaly prediction.
        
        Strategy: Use middle frame of the clip for simplicity (same as FlashbackVAD).
        
        Args:
            batch: ClipBatch containing frames
            
        Returns:
            VADOutput with anomaly prediction
        """
        # Use middle frame (consistent with FlashbackVAD)
        mid = len(batch.frames) // 2
        frame_bgr = batch.frames[mid].frame_bgr  # (H, W, 3) BGR

        # Extract features
        features = self._extract_features(frame_bgr)
        
        # Calibrate if needed
        if not self._is_calibrated:
            self._calibrate(features)
        
        # Compute anomaly score
        anomaly_score = self._compute_anomaly_score(features)
        
        # Determine label
        label = "anomaly" if anomaly_score >= self.anomaly_threshold else "normal"
        
        # Generate simple caption
        if label == "anomaly":
            caption = f"Baseline: Unusual activity detected (distance-based)"
        else:
            caption = f"Baseline: Normal activity"
        
        return VADOutput(
            clip_id=batch.clip_id,
            ts_start=batch.ts_start,
            ts_end=batch.ts_end,
            label=label,
            confidence=float(anomaly_score),
            top_caption=caption,
            extra={
                "model": "baseline_resnet18",
                "threshold": self.anomaly_threshold,
                "is_calibrated": self._is_calibrated,
                "device": self.device,
            },
        )
