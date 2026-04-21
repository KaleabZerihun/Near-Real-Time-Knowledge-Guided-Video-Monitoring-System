from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Any

#can add augmentor logic here
@dataclass(frozen=True)
class KGOutput:
    vad: Any
    kg_validated: bool
    explanation: str
    rules_fired: Optional[List[str]] = None


class DummyAugmentor:
    """
    KG stub (Sprint 1). Always returns "validated".
    Replace this later with the sponsor KG / your real KG rules.
    """
    def augment(self, vad: Any) -> KGOutput:
        return KGOutput(
            vad=vad,
            kg_validated=True,
            explanation="KG not integrated yet (stub).",
            rules_fired=[],
        )
