from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field

class FrameRef(BaseModel):
    key: str
    ts: Optional[float] = None
    index: Optional[int] = None

class FramesIngest(BaseModel):
    stream_id: str = Field(..., examples=["cam0"])
    clip_id: int = Field(..., examples=["123"])
    ts_start: float
    ts_end: float
    fps: Optional[float] = None
    frames: list[FrameRef]

class VadIngest(BaseModel):
    stream_id: str = Field(..., examples=["cam0"])
    clip_id: int = Field(..., examples=["123"])
    ts_start: float
    ts_end: float
    label: Optional[str] = None
    confidence: Optional[float] = None
    extra: dict[str, Any] = Field(default_factory=dict)
