import pytest
from backend.src.logger import InMemoryLogger
from backend.src.pipeline.kg_stub import KGOutput
from backend.src.vad.types import VADOutput
import time

@pytest.mark.unit
def test_vad_anomaly_alert():
    l = InMemoryLogger()

    vad = VADOutput(clip_id=1, ts_start=0.0, ts_end=1.0, label="anomaly", confidence=0.92, top_caption="person running", extra={})
    kg = KGOutput(vad=vad, kg_validated=True, explanation="ok", rules_fired=[])

    alerts = l.handle(vad, kg)

    assert len(alerts) == 1
    a = alerts[0].to_dict()

    assert a["source"] == "vad"
    assert a["severity"] == "critical"

@pytest.mark.unit
def test_kg_rule_and_validation_alerts():
    l = InMemoryLogger()

    vad = VADOutput(clip_id=2, ts_start=0.0, ts_end=1.0, label="normal", confidence=0.1, top_caption="", extra={})
    kg = KGOutput(vad=vad, kg_validated=False, explanation="mismatch", rules_fired=["rule1"])

    alerts = l.handle(vad, kg)

    assert any(a for a in alerts if a.source == "kg" and "validation" in a.message.lower())
    assert any(a for a in alerts if a.source == "kg" and "rule fired" in a.message.lower())

@pytest.mark.unit
def test_get_alerts_since_ts():
    l = InMemoryLogger()

    vad = VADOutput(clip_id=3, ts_start=0.0, ts_end=1.0, label="anomaly", confidence=0.6, top_caption="odd", extra={})
    kg = KGOutput(vad=vad, kg_validated=True, explanation="ok", rules_fired=[])

    _ = l.handle(vad, kg)

    now = time.time()
    # request alerts since a far-future timestamp (now + 1e9 seconds) to ensure none are returned
    res = l.get_alerts(since_ts=1e9 + now)

    assert res == []