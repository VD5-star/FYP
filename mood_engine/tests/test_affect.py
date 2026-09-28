from __future__ import annotations

from dataclasses import dataclass

import pytest

from conftest import load_image
from engine.core.affect import AffectAnalyzer


@dataclass
class FakeEmotion:
    label: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.3
    probs: dict | None = None
    action_units: dict | None = None

    def __post_init__(self):
        self.probs = self.probs or {"neutral": 1.0}
        self.action_units = self.action_units or {}


@dataclass
class FakePose:
    attention: float = 0.8
    eye_openness: float = 0.4
    pitch: float = 0.0
    is_blinking: bool = False


def test_genuine_smile_scores_above_social_smile():
    analyzer = AffectAnalyzer()

    social = analyzer.update(
        FakeEmotion(label="happy", probs={"happy": 0.9}),
        FakePose(),
        {"smile": 0.05, "eye_open": 0.34},
    )

    analyzer.reset()
    genuine = analyzer.update(
        FakeEmotion(label="happy", probs={"happy": 0.9}),
        FakePose(),
        {"smile": 0.05, "eye_open": 0.19},
    )

    assert genuine.duchenne > social.duchenne
    assert genuine.smile_type == "genuine"
    assert social.smile_type == "social"


def test_no_smile_reports_none():
    analyzer = AffectAnalyzer()
    reading = analyzer.update(
        FakeEmotion(label="neutral", probs={"neutral": 0.9}),
        FakePose(), {"smile": 0.0, "eye_open": 0.30})
    assert reading.smile_type == "none"
    assert reading.duchenne == 0.0


def test_compound_emotion_is_named_for_a_real_mixture():
    analyzer = AffectAnalyzer()
    reading = analyzer.update(
        FakeEmotion(label="sad", probs={"sad": 0.45, "fear": 0.35,
                                        "neutral": 0.20}),
        FakePose(), {})
    assert reading.compound == "anxious"
    assert reading.compound_ar


def test_dominant_emotion_reports_no_compound():
    analyzer = AffectAnalyzer()
    reading = analyzer.update(
        FakeEmotion(label="happy", probs={"happy": 0.95, "neutral": 0.05}),
        FakePose(), {})
    assert reading.compound is None


def test_mixture_lists_meaningful_components_only():
    analyzer = AffectAnalyzer()
    reading = analyzer.update(
        FakeEmotion(probs={"happy": 0.50, "surprise": 0.30, "sad": 0.15,
                           "fear": 0.05}),
        FakePose(), {})
    names = [name for name, _ in reading.mixture]
    assert "happy" in names and "surprise" in names
    assert "fear" not in names, "negligible components should be dropped"


def test_engagement_needs_more_than_a_blank_stare():
    attentive_blank = AffectAnalyzer()
    blank = attentive_blank.update(
        FakeEmotion(probs={"neutral": 1.0}), FakePose(attention=0.95), {})

    attentive_lively = AffectAnalyzer()
    lively = attentive_lively.update(
        FakeEmotion(label="happy", probs={"happy": 0.9, "neutral": 0.1}),
        FakePose(attention=0.95), {})

    assert lively.engagement > blank.engagement


def test_fatigue_rises_with_closed_eyes_and_head_droop():
    alert = AffectAnalyzer()
    tired = AffectAnalyzer()
    for _ in range(30):
        alert.update(FakeEmotion(), FakePose(eye_openness=0.55, pitch=0.0), {})
        tired.update(FakeEmotion(), FakePose(eye_openness=0.15, pitch=-28.0), {})

    a = alert.update(FakeEmotion(), FakePose(eye_openness=0.55), {})
    t = tired.update(FakeEmotion(), FakePose(eye_openness=0.15, pitch=-28.0), {})
    assert t.fatigue > a.fatigue
    assert 0.0 <= t.fatigue <= 1.0


def test_tension_rises_with_knitted_brow_and_negative_affect():
    calm = AffectAnalyzer().update(
        FakeEmotion(probs={"neutral": 1.0}), FakePose(),
        {"brow_knit": 1.10, "mouth_open": 0.30})
    tense = AffectAnalyzer().update(
        FakeEmotion(label="anger", probs={"anger": 0.7, "fear": 0.3}),
        FakePose(), {"brow_knit": 0.62, "mouth_open": 0.12})
    assert tense.tension > calm.tension


def test_blink_rate_counts_only_transitions():
    analyzer = AffectAnalyzer()
    for _ in range(10):
        analyzer.update(FakeEmotion(), FakePose(is_blinking=True), {})
    reading = analyzer.update(FakeEmotion(), FakePose(is_blinking=True), {})
    assert len(analyzer._blinks) == 1
    assert reading.blink_rate >= 0.0


def test_volatility_separates_steady_from_swinging_affect():
    steady = AffectAnalyzer()
    swinging = AffectAnalyzer()
    for i in range(40):
        steady.update(FakeEmotion(valence=0.5, arousal=0.5), FakePose(), {})
        sign = 1 if i % 2 == 0 else -1
        swinging.update(FakeEmotion(valence=0.8 * sign, arousal=0.5 + 0.4 * sign),
                        FakePose(), {})

    a = steady.update(FakeEmotion(valence=0.5, arousal=0.5), FakePose(), {})
    b = swinging.update(FakeEmotion(valence=-0.8, arousal=0.1), FakePose(), {})
    assert b.volatility > a.volatility


def test_transition_is_reported_once_per_change():
    analyzer = AffectAnalyzer()
    analyzer.update(FakeEmotion(label="neutral"), FakePose(), {})
    changed = analyzer.update(FakeEmotion(label="happy"), FakePose(), {})
    same = analyzer.update(FakeEmotion(label="happy"), FakePose(), {})

    assert changed.transition == "neutral -> happy"
    assert same.transition is None


def test_summary_reports_session_shape():
    analyzer = AffectAnalyzer()
    for _ in range(12):
        analyzer.update(FakeEmotion(label="happy", valence=0.7), FakePose(), {})
    for _ in range(4):
        analyzer.update(FakeEmotion(label="sad", valence=-0.5), FakePose(), {})

    summary = analyzer.summary()
    assert summary["samples"] == 16
    assert summary["dominant"] == "happy"
    assert summary["distinct_emotions"] == 2
    assert 0.0 <= summary["stability"] <= 1.0
    assert summary["valence_range"] > 0


def test_reset_clears_all_state():
    analyzer = AffectAnalyzer()
    for _ in range(10):
        analyzer.update(FakeEmotion(label="anger"), FakePose(), {})
    analyzer.reset()
    assert analyzer.summary()["samples"] == 0


def test_engine_emits_affect_for_a_real_face(engine):
    result = engine.analyse_frame(load_image("obama_a"), persist=False)
    assert result.ok
    assert result.affect is not None

    payload = result.to_dict()["affect"]
    for key in ("duchenne", "smile_type", "engagement", "fatigue", "tension",
                "volatility", "expressiveness"):
        assert key in payload

    for key in ("duchenne", "engagement", "fatigue", "tension", "volatility",
                "expressiveness"):
        assert 0.0 <= payload[key] <= 1.0, f"{key} out of range: {payload[key]}"


def test_affect_is_persisted(engine):
    engine.start_session("affect-test")
    engine.analyse_frame(load_image("obama_a"), persist=True)

    row = engine.db.recent_observations(limit=1)[0]
    assert "engagement" in row.keys()
    assert row["smile_type"] is not None
