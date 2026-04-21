from __future__ import annotations
import os
import json
from typing import Optional, Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from ..frame_selector.types import ClipBatch
from .types import VADOutput

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}
_MEMORY_CACHE: Dict[Tuple[str, str, str, str, str], Tuple[torch.Tensor, torch.Tensor, List[str]]] = {}

class FlashbackVAD:

    def __init__(
        self,
        rtvad_root: str,
        top_k: int = 10,
        anomaly_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.rtvad_root = os.path.abspath(rtvad_root)
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
    def _memory_paths(self) -> Tuple[str, str, str, str]:
        emb_dir = os.path.join(self.rtvad_root, "embeddings", "stage1")
        emb_normal_path = os.path.join(emb_dir, "embeddings_normal.npy")
        emb_anomalous_path = os.path.join(emb_dir, "embeddings_anomalous.npy")
        meta_path = os.path.join(emb_dir, "memory_meta.json")
        custom_path = os.path.join(self.rtvad_root, "custom_anomaly_memory.json")
        return emb_normal_path, emb_anomalous_path, meta_path, custom_path

    def _cache_key(self) -> Tuple[str, str, str, str, str]:
        emb_normal_path, emb_anomalous_path, meta_path, custom_path = self._memory_paths()
        return (emb_normal_path, emb_anomalous_path, meta_path, custom_path, self.device)

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
        emb_normal_path, emb_anomalous_path, meta_path, custom_path = self._memory_paths()
        cache_key = self._cache_key()

        if cache_key in _MEMORY_CACHE:
            return _MEMORY_CACHE[cache_key]

        if not os.path.exists(emb_normal_path):
            raise FileNotFoundError(
                f"Missing normal embeddings file: {emb_normal_path}\n"
                "Run the memory generation steps in RT-VAD directory."
            )
        if not os.path.exists(emb_anomalous_path):
            raise FileNotFoundError(
                f"Missing anomalous embeddings file: {emb_anomalous_path}\n"
                "Run the memory generation steps in RT-VAD directory."
            )
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing memory meta file: {meta_path}\n"
                "Run the memory generation steps in RT-VAD directory."
            )

        emb_normal_np = np.load(emb_normal_path)
        emb_anomalous_np = np.load(emb_anomalous_path)
        emb_np = np.concatenate([emb_normal_np, emb_anomalous_np], axis=0)

        if emb_np.ndim != 2:
            raise ValueError(f"Embeddings must be 2D (N,D). Got {emb_np.shape}")

        text_emb = torch.tensor(emb_np, dtype=torch.float32, device=self.device)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        if self.device == "cuda":
            text_emb = text_emb.half()

        n_normal = emb_normal_np.shape[0]
        n_anomalous = emb_anomalous_np.shape[0]
        n_total = n_normal + n_anomalous

        y_labels = torch.zeros(n_total, device=self.device)
        y_labels[n_normal:] = 1.0

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        captions_normal = meta.get("captions_normal", [])
        captions_anomalous = meta.get("captions_anomalous", [])
        captions = captions_normal + captions_anomalous

        if os.path.exists(custom_path):
            try:
                with open(custom_path, "r", encoding="utf-8") as f:
                    custom_data = json.load(f)
                custom_items = custom_data.get("custom_anomalies", []) if isinstance(custom_data, dict) else []
            except Exception:
                custom_items = []

            if isinstance(custom_items, list) and custom_items:
                custom_embs = np.array(
                    [item.get("embedding") for item in custom_items if item.get("embedding") is not None],
                    dtype=np.float32,
                )
                if custom_embs.ndim == 1:
                    custom_embs = custom_embs.reshape(1, -1)

                if custom_embs.ndim == 2 and custom_embs.shape[1] == text_emb.shape[1]:
                    custom_text_emb = torch.tensor(custom_embs, dtype=torch.float32, device=self.device)
                    custom_text_emb = F.normalize(custom_text_emb, p=2, dim=-1)
                    if self.device == "cuda":
                        custom_text_emb = custom_text_emb.half()

                    text_emb = torch.cat([text_emb, custom_text_emb], dim=0)
                    custom_labels = torch.ones(custom_text_emb.shape[0], device=self.device)
                    y_labels = torch.cat([y_labels, custom_labels], dim=0)
                    captions += [item.get("text", "") for item in custom_items]
                else:
                    print(
                        f"[WARN] Skipping custom anomaly memory load; expected emb dim {text_emb.shape[1]}, got {custom_embs.shape}"
                    )

        if len(captions) != y_labels.shape[0]:
            print(
                f"[WARN] captions({len(captions)}) != embeddings({y_labels.shape[0]}). "
                "Top caption lookup may be mismatched."
            )

        _MEMORY_CACHE[cache_key] = (text_emb, y_labels, captions)
        print("[VAD] Memory loaded (cached).")
        return text_emb, y_labels, captions

    def reload_memory(self) -> None:
        cache_key = self._cache_key()
        _MEMORY_CACHE.pop(cache_key, None)
        self._text_emb, self._labels, self._captions = self._get_or_load_memory()

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