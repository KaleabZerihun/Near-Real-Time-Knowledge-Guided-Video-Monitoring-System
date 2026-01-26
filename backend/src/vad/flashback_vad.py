from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from src.frame_selector.types import ClipBatch


@dataclass(frozen=True)
class VADOutput:
    clip_id: int
    ts_start: float
    ts_end: float
    label: str               # "normal" or "anomaly"
    confidence: float        # anomaly score (0..1)
    top_caption: str
    extra: Dict[str, Any]


class FlashbackVAD:
    """
    Wrapper around sponsor Flashback/ImageBind VAD.
    - DOES NOT modify sponsor folder
    - Reads memory files from: THESIS/src/memory/
    """

    def __init__(
        self,
        thesis_root: str,
        top_k: int = 10,
        anomaly_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.thesis_root = os.path.abspath(thesis_root)
        self.top_k = int(top_k)
        self.anomaly_threshold = float(anomaly_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[torch.nn.Module] = None
        self._text_emb: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        self._captions: Optional[List[str]] = None

        self._load_memory()
        self._load_model()

    def _memory_paths(self) -> Tuple[str, str]:
        mem_dir = os.path.join(self.thesis_root, "src", "memory")
        emb_path = os.path.join(mem_dir, "flashback_text_embeddings_SAP.npy")
        cap_path = os.path.join(mem_dir, "flashback_captions.txt")
        return emb_path, cap_path

    def _load_memory(self) -> None:
        emb_path, cap_path = self._memory_paths()

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
        if not os.path.exists(cap_path):
            raise FileNotFoundError(f"Missing captions file: {cap_path}")

        emb_np = np.load(emb_path)
        if emb_np.ndim != 2:
            raise ValueError(f"Embeddings must be 2D (N,D). Got shape={emb_np.shape}")

        text_emb = torch.tensor(emb_np, dtype=torch.float32, device=self.device)
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        n_total = text_emb.shape[0]
        half = n_total // 2
        labels = torch.zeros(n_total, device=self.device)
        labels[half:] = 1.0  # sponsor logic: first half normal, second half anomalous

        with open(cap_path, "r", encoding="utf-8") as f:
            captions = [line.strip() for line in f.readlines() if line.strip()]

        # If captions mismatch embeddings, still run but be safe
        if len(captions) != n_total:
            print(
                f"[WARN] captions({len(captions)}) != embeddings({n_total}). "
                "Top caption lookup may be wrong."
            )

        self._text_emb = text_emb
        self._labels = labels
        self._captions = captions

        # clamp top_k
        if self.top_k > n_total:
            self.top_k = n_total

        print(
            f"[VAD] Memory loaded. device={self.device} "
            f"embeddings={n_total} top_k={self.top_k}"
        )

    def _load_model(self) -> None:
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval().to(self.device)

        # speed on CUDA
        if self.device == "cuda":
            model.half()

        self._model = model
        print("[VAD] ImageBind model loaded.")

    def predict(self, batch: ClipBatch) -> VADOutput:
        """
        Convert ClipBatch -> one VAD decision.
        Sprint-friendly: uses the middle frame (fast).
        """

        assert self._model is not None
        assert self._text_emb is not None
        assert self._labels is not None
        assert self._captions is not None

        mid = len(batch.frames) // 2
        frame_bgr = batch.frames[mid].frame_bgr  # (H,W,3) BGR

        # BGR -> RGB float [0,1]
        rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

        dtype = torch.half if (self.device == "cuda") else torch.float32
        frame_tensor = frame_tensor.to(self.device, dtype=dtype)

        with torch.inference_mode():
            out = self._model({ModalityType.VISION: frame_tensor})
            vid_emb = out[ModalityType.VISION]
            vid_emb = F.normalize(vid_emb, p=2, dim=-1)

            scores = torch.matmul(vid_emb, self._text_emb.T)  # (1,N)
            topk_vals, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)

            weights = F.softmax(topk_vals, dim=-1)[0]       # (K,)
            selected_labels = self._labels[topk_idx[0]]     # (K,)

            anomaly_score = torch.sum(weights * selected_labels).item()

            # caption (safe)
            best_caption = ""
            idx0 = int(topk_idx[0][0].item())
            if 0 <= idx0 < len(self._captions):
                best_caption = self._captions[idx0]

        label = "anomaly" if anomaly_score >= self.anomaly_threshold else "normal"

        return VADOutput(
            clip_id=batch.clip_id,
            ts_start=batch.ts_start,
            ts_end=batch.ts_end,
            label=label,
            confidence=float(anomaly_score),
            top_caption=best_caption,
            extra={
                "top_k": self.top_k,
                "threshold": self.anomaly_threshold,
                "device": self.device,
            },
        )
