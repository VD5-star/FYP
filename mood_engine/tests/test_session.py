from __future__ import annotations

import pytest

from engine.core.session import SESSION_STATES, SessionAnalyser


def feed(analyser: SessionAnalyser, count: int, *, person: int | None = 1,
         valence: float = 0.0, arousal: float = 0.4,
         state: str = "neutral", emotion: str = "neutral",
         confidence: float = 0.8, start: float = 0.0, step: float = 0.11,
         units: dict[str, float] | None = None) -> list:
    units = units or {"engagement": 0.5, "fatigue": 0.2, "tension": 0.2}
    reports = []
    for i in range(count):
        report = analyser.observe(
            person_id=person, person_name=f"P{person}",
            valence=valence, arousal=arousal, mood_state=state,
            emotion=emotion, confidence=confidence, units=units,
            now=start + i * step)
        if report is not None:
            reports.append(report)
    return reports


def test_thin_window_reports_nothing():
    analyser = SessionAnalyser(period_s=30, min_samples=60)
    feed(analyser, 20)
    assert analyser.flush(now=40.0) is None


def test_period_closes_on_time():
    analyser = SessionAnalyser(period_s=30, min_samples=50)
    reports = feed(analyser, 400, step=0.11)
    assert len(reports) >= 1
    assert reports[0].duration_s == pytest.approx(30.0, abs=1.0)


def test_person_change_closes_the_window():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    feed(analyser, 100, person=1, valence=-0.5, state="negative",
         emotion="sad")
    report = analyser.observe(
        person_id=2, person_name="P2", valence=0.5, arousal=0.5,
        mood_state="positive", emotion="happy", confidence=0.8, now=11.0)
    assert report is not None
    assert report.person_id == 1
    assert report.valence < 0


def test_previous_person_does_not_leak_into_the_next():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    feed(analyser, 120, person=1, valence=-0.7, state="negative",
         emotion="sad")
    analyser.observe(person_id=2, person_name="P2", valence=0.6, arousal=0.5,
                     mood_state="positive", emotion="happy", confidence=0.8,
                     now=14.0)
    feed(analyser, 120, person=2, valence=0.6, state="positive",
         emotion="happy", start=15.0)
    report = analyser.flush(now=60.0)
    assert report is not None
    assert report.person_id == 2
    assert report.valence > 0.4, "previous person's readings leaked"
    assert "sad" not in report.emotion_mix


def test_swinging_person_is_restless_not_neutral():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    for i in range(160):
        positive = (i // 10) % 2 == 0
        analyser.observe(
            person_id=5, person_name="P5",
            valence=0.7 if positive else -0.7, arousal=0.6,
            mood_state="positive" if positive else "negative",
            emotion="happy" if positive else "sad",
            confidence=0.8,
            units={"engagement": 0.5, "fatigue": 0.2, "tension": 0.3},
            now=i * 0.11)
    report = analyser.flush(now=30.0)
    assert report is not None
    assert report.state == "restless"
    assert report.stability < 0.3
    assert "mood shifted repeatedly" in report.notes


def test_low_confidence_frames_do_not_vote():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    feed(analyser, 100, valence=0.5, state="positive", emotion="happy")
    feed(analyser, 100, valence=-0.9, state="negative", emotion="sad",
         confidence=0.1, start=12.0)
    report = analyser.flush(now=30.0)
    assert report is not None
    assert report.valence > 0.3
    assert "sad" not in report.emotion_mix


def test_sustained_state_wins_over_derived_rules():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    feed(analyser, 120, valence=-0.5, arousal=0.3, state="negative",
         emotion="sad")
    report = analyser.flush(now=20.0)
    assert report is not None
    assert report.state == "low"
    assert report.dominance > 0.9


def test_tiredness_is_distinguished_from_sadness():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    feed(analyser, 120, valence=0.0, arousal=0.2, state="neutral",
         emotion="neutral",
         units={"engagement": 0.2, "fatigue": 0.8, "tension": 0.1})
    report = analyser.flush(now=20.0)
    assert report is not None
    assert report.state == "tired"
    assert "signs of tiredness" in report.notes


def test_every_state_has_a_bilingual_label():
    for key, (english, arabic) in SESSION_STATES.items():
        assert english and arabic, f"{key} is missing a label"
        assert english != arabic, f"{key} was never translated"
        assert any("\u0600" <= ch <= "\u06ff" for ch in arabic), key


def test_report_is_serialisable():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    feed(analyser, 120, valence=0.5, state="positive", emotion="happy")
    report = analyser.flush(now=20.0)
    assert report is not None
    payload = report.to_dict()
    assert payload["state"] == "positive"
    assert payload["label_en"] and payload["label_ar"]
    assert isinstance(payload["emotion_mix"], dict)
    assert isinstance(payload["notes"], list)


def test_progress_reports_the_slower_of_time_and_evidence():
    analyser = SessionAnalyser(period_s=100, min_samples=100)
    feed(analyser, 10, step=1.0)
    progress = analyser.progress(now=10.0)
    assert progress["progress"] == pytest.approx(0.1, abs=0.02)
    assert progress["will_report"] is False


def test_flush_restarts_cleanly():
    analyser = SessionAnalyser(period_s=600, min_samples=50)
    feed(analyser, 120, valence=0.5, state="positive", emotion="happy")
    assert analyser.flush(now=20.0) is not None
    assert analyser.flush(now=21.0) is None, "evidence was not cleared"
