from __future__ import annotations

import numpy as np
import pytest

from engine.config import EMOTIONS
from engine.core.calibration import TOTAL_FRAMES
from engine.core.baseline import (BaselineTracker, MoodSummary,
                                  PersonBaseline)

NEUTRAL = EMOTIONS.index("neutral")
HAPPY = EMOTIONS.index("happy")
ANGER = EMOTIONS.index("anger")


def probs(**weights) -> np.ndarray:
    vector = np.full(len(EMOTIONS), 0.02)
    for name, value in weights.items():
        vector[EMOTIONS.index(name)] = value
    return vector / vector.sum()


def calibrate(tracker: BaselineTracker, resting: np.ndarray,
              person_id: int = 1, frames: int | None = None) -> None:
    for _ in range(frames if frames is not None else PersonBaseline.REQUIRED):
        tracker.adjust(resting, {}, person_id)


def test_baseline_needs_enough_samples_before_it_is_trusted():
    tracker = BaselineTracker()
    reading = probs(anger=0.6, neutral=0.2)

    adjusted, baseline = tracker.adjust(reading, {}, 1)
    assert not baseline.ready
    assert np.allclose(adjusted, reading), (
        "readings must pass through untouched until calibration completes")


def test_baseline_forms_from_ordinary_frames():
    tracker = BaselineTracker()
    angry_resting_face = probs(anger=0.55, neutral=0.08)

    calibrate(tracker, angry_resting_face)
    assert tracker.baseline_for(1).ready


def test_extreme_frames_stop_shifting_an_established_baseline():
    tracker = BaselineTracker()
    resting = probs(neutral=0.35, happy=0.25, sad=0.15)
    calibrate(tracker, resting)

    settled = tracker.baseline_for(1).probs.copy()
    samples_before = tracker.baseline_for(1).samples

    laughing = probs(happy=0.97)
    for _ in range(60):
        tracker.adjust(laughing, {}, 1)

    assert tracker.baseline_for(1).samples == samples_before, (
        "extreme frames should not be recorded once calibrated")
    assert np.allclose(tracker.baseline_for(1).probs, settled), (
        "a long laugh must not become this person's resting face")


def test_progress_is_reported_while_calibrating():
    tracker = BaselineTracker()
    resting = probs(neutral=0.3, anger=0.3)

    for _ in range(10):
        tracker.adjust(resting, {}, 1)
    baseline = tracker.baseline_for(1)

    assert 0.0 < baseline.progress < 1.0
    calibrate(tracker, resting)
    assert tracker.baseline_for(1).progress == 1.0


def test_a_persons_resting_bias_is_suppressed():
    tracker = BaselineTracker()
    resting = probs(anger=0.50, neutral=0.10, sad=0.15)
    calibrate(tracker, resting)

    adjusted, _ = tracker.adjust(resting, {}, 1)
    assert adjusted[ANGER] < resting[ANGER], (
        "the person's habitual anger reading must be reduced")


def test_a_real_expression_still_registers():
    tracker = BaselineTracker()
    resting = probs(anger=0.45, neutral=0.12)
    calibrate(tracker, resting)

    smiling = probs(happy=0.70, neutral=0.10)
    adjusted, _ = tracker.adjust(smiling, {}, 1)

    assert EMOTIONS[int(np.argmax(adjusted))] == "happy"
    assert adjusted[HAPPY] > 0.3


def test_an_unusual_emotion_is_amplified():
    tracker = BaselineTracker()
    resting = probs(neutral=0.4, happy=0.3, anger=0.2)
    calibrate(tracker, resting)

    surprised = probs(surprise=0.35, neutral=0.3, happy=0.2)
    adjusted, _ = tracker.adjust(surprised, {}, 1)
    raw_rank = list(np.argsort(surprised)[::-1]).index(
        EMOTIONS.index("surprise"))
    adj_rank = list(np.argsort(adjusted)[::-1]).index(
        EMOTIONS.index("surprise"))
    assert adj_rank <= raw_rank, "a rare class should rank at least as high"


def test_output_stays_a_valid_distribution():
    tracker = BaselineTracker()
    resting = probs(anger=0.5, neutral=0.05)
    calibrate(tracker, resting)

    for reading in (probs(happy=0.8), probs(fear=0.4, sad=0.4),
                    probs(neutral=0.9), resting):
        adjusted, _ = tracker.adjust(reading, {}, 1)
        assert np.all(adjusted >= 0)
        assert np.isclose(adjusted.sum(), 1.0)
        assert np.all(np.isfinite(adjusted))


def test_baselines_are_kept_separate_per_person():
    tracker = BaselineTracker()
    angry_face = probs(anger=0.55, neutral=0.10)
    happy_face = probs(happy=0.55, neutral=0.10)

    calibrate(tracker, angry_face, person_id=1)
    calibrate(tracker, happy_face, person_id=2)

    one = tracker.baseline_for(1).probs
    two = tracker.baseline_for(2).probs
    assert one[ANGER] > two[ANGER]
    assert two[HAPPY] > one[HAPPY]


def test_a_baseline_survives_a_round_trip():
    tracker = BaselineTracker()
    calibrate(tracker, probs(anger=0.45, neutral=0.20, sad=0.15))

    payload = tracker.export(1)
    assert payload and payload["samples"] >= PersonBaseline.REQUIRED

    restored = BaselineTracker()
    restored.load(1, payload)
    assert restored.baseline_for(1).ready
    assert np.allclose(restored.baseline_for(1).probs,
                       tracker.baseline_for(1).probs, atol=1e-4)


def test_forget_clears_calibration():
    tracker = BaselineTracker()
    calibrate(tracker, probs(anger=0.45, neutral=0.25, sad=0.15))
    assert tracker.baseline_for(1).ready

    tracker.forget(1)
    assert not tracker.baseline_for(1).ready


def test_mood_needs_a_few_frames_before_committing():
    tracker = BaselineTracker()
    summary = tracker.update_mood(0.8, 0.7, 0.1, 0.1, 1)
    assert summary.state == "neutral"
    assert summary.confidence == 0.0


def test_sustained_positive_affect_reads_positive():
    tracker = BaselineTracker()
    for _ in range(40):
        summary = tracker.update_mood(0.65, 0.60, 0.1, 0.05, 1)
    assert summary.state in ("positive", "content")
    assert summary.valence > 0.5


def test_calm_positive_affect_reads_content():
    tracker = BaselineTracker()
    for _ in range(40):
        summary = tracker.update_mood(0.45, 0.20, 0.1, 0.05, 1)
    assert summary.state == "content"


def test_sustained_negative_affect_reads_negative_or_tense():
    tracker = BaselineTracker()
    for _ in range(40):
        summary = tracker.update_mood(-0.55, 0.30, 0.2, 0.05, 1)
    assert summary.state in ("negative", "withdrawn", "tense")


def test_high_tension_reads_tense():
    tracker = BaselineTracker()
    for _ in range(40):
        summary = tracker.update_mood(-0.45, 0.65, 0.8, 0.1, 1)
    assert summary.state == "tense"


def test_swinging_affect_reads_volatile():
    tracker = BaselineTracker()
    for i in range(60):
        sign = 1 if i % 2 == 0 else -1
        summary = tracker.update_mood(0.75 * sign, 0.5, 0.2, 0.8, 1)
    assert summary.state == "volatile"
    assert summary.stability < 0.5


def test_mood_window_resets_when_the_person_changes():
    tracker = BaselineTracker()
    for _ in range(40):
        tracker.update_mood(0.8, 0.7, 0.1, 0.05, 1)

    summary = tracker.update_mood(-0.5, 0.3, 0.2, 0.1, 2)
    assert summary.samples == 1, "the window should restart for person 2"


def test_summary_serialises_cleanly():
    tracker = BaselineTracker()
    for _ in range(30):
        summary = tracker.update_mood(0.5, 0.5, 0.2, 0.1, 1)

    payload = summary.to_dict()
    for key in ("state", "state_en", "state_ar", "confidence", "valence",
                "energy", "stability", "calibrated", "baseline_progress"):
        assert key in payload
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["state_ar"], "Arabic label must be present"


def test_engine_reports_mood(engine):
    from conftest import load_image

    result = engine.analyse_frame(load_image("obama_a"), persist=False)
    assert result.ok
    assert result.mood is not None
    assert result.mood.to_dict()["state"] in {
        "positive", "content", "neutral", "withdrawn", "tense", "negative",
        "volatile"}


def test_engine_persists_and_reloads_a_baseline(engine):
    from conftest import load_image

    engine.start_session("baseline-test")
    image = load_image("obama_a")

    engine.enrol_image("Baseline_Subject", image)

    for _ in range(TOTAL_FRAMES + 40):
        engine.analyse_frame(image, persist=False)

    person_id = engine._current_person
    assert person_id is not None, "the enrolled person should be recognised"

    assert engine.save_baselines() >= 1
    stored = engine.db.load_baseline(person_id)
    assert stored and stored["samples"] >= PersonBaseline.REQUIRED

    fresh = BaselineTracker()
    fresh.load(person_id, stored)
    reloaded = fresh.baseline_for(person_id)
    assert reloaded.ready
    assert reloaded.age_s < 60, "a reloaded baseline lost its timestamp"
