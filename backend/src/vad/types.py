from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class VADOutput:
    clip_id: int
    ts_start: float
    ts_end: float
    label: str
    confidence: float
    top_caption: str
    extra: Dict[str, Any]