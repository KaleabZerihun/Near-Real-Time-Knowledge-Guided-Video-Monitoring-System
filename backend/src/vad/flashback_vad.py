from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from ..frame_selector.types import ClipBatch

@dataclass(frozen=True)
class VADOutput:
    clip_id: int
    ts_start: float
    ts_end: float
    label: str               # "normal" or "anomaly"
    confidence: float       
    top_caption: str
    extra: Dict[str, Any]

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}
_MEMORY_CACHE: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, List[str]]] = {}

class FlashbackVAD:

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
        self._model = self._get_or_load_model()
        self._text_emb, self._labels, self._captions = self._get_or_load_memory()

        n_total = self._text_emb.shape[0]
        if self.top_k > n_total:
            self.top_k = n_total

        print(
            f"[VAD] Ready. device={self.device} embeddings={n_total} "
            f"top_k={self.top_k} threshold={self.anomaly_threshold}"
        )

    # ---------------- paths / loading ----------------
    def _memory_paths(self) -> Tuple[str, str]:
        mem_dir = os.path.join(self.thesis_root, "src", "memory")
        emb_path = os.path.join(mem_dir, "flashback_text_embeddings_SAP.npy")
        cap_path = os.path.join(mem_dir, "flashback_captions.txt")
        return emb_path, cap_path

    def _get_or_load_model(self) -> torch.nn.Module:
        key = self.device
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval().to(self.device)

        # CUDA(Compute Unified Device Architecture) 
        #       - a NVIDIA platform that let's the program use the GPU instead of the CPU for heavy computation.
        if self.device == "cuda":
            model.half()

        _MODEL_CACHE[key] = model
        print("[VAD] ImageBind model loaded (cached).")
        return model

    def _get_or_load_memory(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        emb_path, cap_path = self._memory_paths()
        cache_key = (emb_path, self.device)

        if cache_key in _MEMORY_CACHE:
            return _MEMORY_CACHE[cache_key]

        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Missing embeddings file: {emb_path}\n"
                "This file must NOT be committed to GitHub (too large).\n"
                "Each teammate must place it at: THESIS/src/memory/flashback_text_embeddings_SAP.npy"
            )
        if not os.path.exists(cap_path):
            raise FileNotFoundError(
                f"Missing captions file: {cap_path}\n"
                "Place it at: THESIS/src/memory/flashback_captions.txt"
            )

        emb_np = np.load(emb_path)
        if emb_np.ndim != 2:
            raise ValueError(f"Embeddings must be 2D (N,D). Got {emb_np.shape}")

        text_emb = torch.tensor(emb_np, dtype=torch.float32, device=self.device)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        if self.device == "cuda":
            text_emb = text_emb.half()

        n_total = text_emb.shape[0]
        half = n_total // 2

        y_labels = torch.zeros(n_total, device=self.device)
        y_labels[half:] = 1.0

        with open(cap_path, "r", encoding="utf-8") as f:
            captions = [c.strip() for c in f.readlines() if c.strip()]

        if len(captions) != n_total:
            print(
                f"[WARN] captions({len(captions)}) != embeddings({n_total}). "
                "Top caption lookup may be mismatched."
            )

        _MEMORY_CACHE[cache_key] = (text_emb, y_labels, captions)
        print("[VAD] Memory loaded (cached).")
        return text_emb, y_labels, captions

    def _compute_anomaly_score(self, video_emb: torch.Tensor) -> Tuple[float, str, Dict[str, Any]]:
      
        scores = torch.matmul(video_emb, self._text_emb.T)  # (1, N)
        topk_vals, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)

        weights = F.softmax(topk_vals, dim=-1)              # (1, K)
        selected_labels = self._labels[topk_idx[0]]         # (K,)
        anomaly_score = torch.sum(weights[0] * selected_labels).item()

        idx0 = int(topk_idx[0][0].item())
        top_caption = self._captions[idx0] if 0 <= idx0 < len(self._captions) else ""

        debug = {
            "topk_indices": [int(i.item()) for i in topk_idx[0]],
            "topk_values": [float(v.item()) for v in topk_vals[0]],
        }

        return float(anomaly_score), top_caption, debug

    def predict(self, batch: ClipBatch) -> VADOutput:

        mid = len(batch.frames) // 2
        frame_bgr = batch.frames[mid].frame_bgr  # (H,W,3) BGR (already resized by selector)

        model_input = cv2.resize(frame_bgr, (224, 224))
        rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        dtype = torch.half if self.device == "cuda" else torch.float32
        frame_tensor = frame_tensor.to(self.device, dtype=dtype)

        with torch.inference_mode():
            out = self._model({ModalityType.VISION: frame_tensor})
            video_emb = out[ModalityType.VISION]
            video_emb = F.normalize(video_emb, p=2, dim=-1)

        raw_score, top_caption, dbg = self._compute_anomaly_score(video_emb)

        label = "anomaly" if raw_score >= self.anomaly_threshold else "normal"

        return VADOutput(
            clip_id=batch.clip_id,
            ts_start=batch.ts_start,
            ts_end=batch.ts_end,
            label=label,
            confidence=float(raw_score),
            top_caption=top_caption,
            extra={
                "device": self.device,
                "top_k": self.top_k,
                "threshold": self.anomaly_threshold,
                **dbg,
            },
        )