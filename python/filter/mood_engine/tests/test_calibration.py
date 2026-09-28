from __future__ import annotations

import time

import numpy as np
import pytest

from engine.core.baseline import PersonBaseline
from engine.core.calibration import (CALIBRATION_STEPS, REST_FRAMES,
                                     TOTAL_FRAMES, CalibrationManager,
                                     CalibrationSession, rest_frames_only)


def good(session: CalibrationSession, count: int) -> None:
    for _ in range(count):
        session.offer(has_face=True, quality=0.9, yaw=0.0, pitch=0.0)


def test_calibration_can_actually_finish():
    assert PersonBaseline.REQUIRED <= REST_FRAMES, (
        f"baseline needs {PersonBaseline.REQUIRED} resting frames but guided "
        f"calibration only collects {REST_FRAMES}")


def test_completing_every_step_completes_the_session():
    session = CalibrationSession(person_id=1)
    for step in CALIBRATION_STEPS:
        good(session, step["frames"])
    assert session.complete
    assert session.total_accepted == TOTAL_FRAMES
    assert session.progress == pytest.approx(1.0)


def test_a_step_cannot_be_skipped_by_waiting():
    session = CalibrationSession(person_id=1)
    for _ in range(5000):
        session.offer(has_face=False, quality=0.0)
    assert not session.complete
    assert session.total_accepted == 0
    assert session.step_index == 0


def test_poor_frames_are_refused_with_a_reason():
    session = CalibrationSession(person_id=1)
    assert session.offer(has_face=False, quality=0.9) == "no_face"
    assert session.offer(has_face=True, quality=0.05) == "low_quality"
    assert session.offer(has_face=True, quality=0.9, yaw=70) == "extreme_angle"
    assert session.offer(has_face=True, quality=0.9) is None


def test_the_turn_step_accepts_the_angles_others_refuse():
    session = CalibrationSession(person_id=1)
    for step in CALIBRATION_STEPS:
        if step["key"] == "turn":
            break
        good(session, step["frames"])

    assert session.step is not None and session.step["key"] == "turn"
    assert session.offer(has_face=True, quality=0.9, yaw=45.0) is None


def test_a_stalled_session_explains_itself():
    session = CalibrationSession(person_id=1)
    for _ in range(100):
        session.offer(has_face=False, quality=0.0)
    assert session.hint() == "no_face"


def test_no_hint_before_there_is_evidence():
    session = CalibrationSession(person_id=1)
    session.offer(has_face=False, quality=0.0)
    assert session.hint() is None


def test_only_resting_steps_feed_the_baseline():
    probs = np.zeros(7)
    session = CalibrationSession(person_id=1)

    assert session.step["key"] == "rest"
    assert rest_frames_only(session, probs) is True

    good(session, CALIBRATION_STEPS[0]["frames"])
    assert session.step["key"] == "smile"
    assert rest_frames_only(session, probs) is False

    good(session, CALIBRATION_STEPS[1]["frames"])
    assert session.step["key"] == "brows"
    assert rest_frames_only(session, probs) is False


def test_normal_operation_always_learns():
    assert rest_frames_only(None, np.zeros(7)) is True


def test_a_fresh_person_needs_calibration():
    manager = CalibrationManager()
    assert manager.needed_for(None) is True
    assert manager.needed_for(PersonBaseline()) is True


def test_a_calibrated_person_is_left_alone():
    manager = CalibrationManager()
    baseline = PersonBaseline(samples=PersonBaseline.REQUIRED)
    assert manager.needed_for(baseline) is False


def test_a_stale_baseline_is_recalibrated():
    manager = CalibrationManager()
    old = PersonBaseline(samples=PersonBaseline.REQUIRED)
    old.updated_at = time.time() - PersonBaseline.MAX_AGE_S - 60
    assert old.expired is True
    assert manager.needed_for(old) is True


def test_an_unformed_baseline_is_not_called_stale():
    partial = PersonBaseline(samples=5)
    partial.updated_at = time.time() - PersonBaseline.MAX_AGE_S - 60
    assert partial.expired is False


def test_skipping_is_remembered():
    manager = CalibrationManager()
    manager.begin(7)
    manager.skip()
    assert manager.ensure(7, "x", PersonBaseline()) is None


def test_a_new_person_restarts_calibration():
    manager = CalibrationManager()
    first = manager.ensure(1, "A", PersonBaseline())
    assert first is not None
    good(first, 40)

    assert manager.ensure(2, "B", PersonBaseline()) is first

    for _ in range(CalibrationManager.IDENTITY_SWITCH_FRAMES):
        second = manager.ensure(2, "B", PersonBaseline())

    assert second is not None
    assert second is not first
    assert second.total_accepted == 0


def test_a_recognition_gap_does_not_restart_calibration():
    manager = CalibrationManager()
    first = manager.ensure(1, "A", PersonBaseline())
    assert first is not None
    good(first, 40)

    for _ in range(CalibrationManager.IDENTITY_SWITCH_FRAMES * 2):
        session = manager.ensure(None, None, PersonBaseline())

    assert session is first
    assert session.total_accepted == 40


def test_disabled_manager_never_calibrates():
    manager = CalibrationManager(enabled=False)
    assert manager.needed_for(PersonBaseline()) is False
    assert manager.ensure(1, "x", PersonBaseline()) is None


def test_resetting_a_baseline_forces_recalibration():
    baseline = PersonBaseline(samples=PersonBaseline.REQUIRED)
    assert baseline.ready
    baseline.reset()
    assert not baseline.ready
    assert baseline.samples == 0
    assert float(baseline.probs.sum()) == 0.0


def test_a_reloaded_baseline_keeps_its_age():
    original = PersonBaseline(samples=PersonBaseline.REQUIRED)
    original.updated_at = time.time() - 3 * 24 * 3600
    restored = PersonBaseline.from_dict(original.to_dict())
    assert restored.age_s > 2 * 24 * 3600


def test_every_step_is_bilingual():
    for step in CALIBRATION_STEPS:
        assert step["en"] and step["ar"]
        assert any("\u0600" <= c <= "\u06ff" for c in step["ar"]), step["key"]


def test_progress_payload_is_complete():
    session = CalibrationSession(person_id=3, person_name="X")
    good(session, 20)
    payload = session.to_dict()
    for key in ("active", "step_index", "step_count", "step_en", "step_ar",
                "step_progress", "progress", "accepted", "required",
                "complete"):
        assert key in payload, key
    assert payload["required"] == TOTAL_FRAMES
