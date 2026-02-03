from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..vad.flashback_vad import VADOutput
from ..pipeline.kg_stub import KGOutput


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    id: int
    ts: float
    source: str
    severity: Severity
    message: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "source": self.source,
            "severity": self.severity.value,
            "message": self.message,
            "data": self.data,
        }


class ILogger:
    """minimal logger interface for alerts."""

    def handle(self, vad: VADOutput, kg: KGOutput) -> List[Alert]:
        raise NotImplementedError

    def get_alerts(self, since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryLogger(ILogger):
    """simple in-memory alert logger.

    - keeps a rolling list of recent alerts
    - creates alerts when VAD or KG indicate anomalies or rule firings
    """

    def __init__(self, maxlen: int = 256) -> None:
        self._alerts: List[Alert] = []
        self._next_id = 1
        self._maxlen = int(maxlen)

    def _append(self, alert: Alert) -> None:
        self._alerts.append(alert)
        if len(self._alerts) > self._maxlen:
            self._alerts = self._alerts[-self._maxlen :]

    def handle(self, vad: VADOutput, kg: KGOutput) -> List[Alert]:
        """inspect VAD and KG outputs and create alerts when relevant.
        returns the list of newly created Alert objects for this event.
        """
        created: List[Alert] = []

        try:
            label = vad.label.lower()
            if label == "anomaly":
                severity = Severity.CRITICAL if vad.confidence >= 0.8 else Severity.WARNING
                msg = f"VAD anomaly detected (score={vad.confidence:.3f})"
                if vad.top_caption:
                    msg = f"{msg}: {vad.top_caption}"

                a = Alert(
                    id=self._next_id,
                    ts=time.time(),
                    source="vad",
                    severity=severity,
                    message=msg,
                    data={
                        "clip_id": vad.clip_id,
                        "ts_start": vad.ts_start,
                        "ts_end": vad.ts_end,
                        "confidence": vad.confidence,
                        "top_caption": vad.top_caption,
                    },
                )
                self._next_id += 1
                self._append(a)
                created.append(a)
        except Exception:
            pass

        try:
            if not kg.kg_validated:
                a = Alert(
                    id=self._next_id,
                    ts=time.time(),
                    source="kg",
                    severity=Severity.WARNING,
                    message=f"KG validation failed: {kg.explanation}",
                    data={
                        "explanation": kg.explanation,
                    },
                )
                self._next_id += 1
                self._append(a)
                created.append(a)

            if kg.rules_fired:
                for r in kg.rules_fired:
                    a = Alert(
                        id=self._next_id,
                        ts=time.time(),
                        source="kg",
                        severity=Severity.INFO,
                        message=f"KG rule fired: {r}",
                        data={"rule": r},
                    )
                    self._next_id += 1
                    self._append(a)
                    created.append(a)
        except Exception:
            pass

        return created

    def get_alerts(self, since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        if since_ts is None:
            return [a.to_dict() for a in self._alerts]
        return [a.to_dict() for a in self._alerts if a.ts >= since_ts]
