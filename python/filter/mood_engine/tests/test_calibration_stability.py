from __future__ import annotations

import numpy as np
import pytest

from engine.core.calibration import (CALIBRATION_STEPS, CalibrationManager,
                                     CalibrationSession)


class _Baseline:
    ready = False
    expired = False


def _drive(manager: CalibrationManager, identities: list[int | None]):
    baseline = _Baseline()
    restarts = 0
    previous = 0
    peak = 0

    for person_id in identities:
        session = manager.ensure(person_id, "someone", baseline)
        if session is None:
            continue
        session.offer(has_face=True, quality=0.8, yaw=0.0, pitch=0.0)
        accepted = session.total_accepted
        if accepted < previous:
            restarts += 1
        previous = accepted
        peak = max(peak, accepted)

    return restarts, previous, peak


class TestIdentityFlickerDoesNotResetProgress:
    def test_stable_identity_accumulates(self) -> None:
        restarts, final, _ = _drive(CalibrationManager(), [1] * 300)
        assert restarts == 0
        assert final == 300

    def test_dropout_to_unknown_keeps_progress(self) -> None:
        identities = [None if i % 7 == 0 else 1 for i in range(300)]

        restarts, final, _ = _drive(CalibrationManager(), identities)

        assert restarts == 0
        assert final >= 290

    def test_brief_wrong_match_keeps_progress(self) -> None:
        identities = [2 if i % 11 == 0 else 1 for i in range(300)]

        restarts, final, _ = _drive(CalibrationManager(), identities)

        assert restarts == 0, "a one-frame mismatch must not restart"
        assert final >= 250

    def test_alternating_noise_keeps_progress(self) -> None:
        identities = []
        for i in range(300):
            identities.append(None if i % 5 == 0
                              else (3 if i % 13 == 0 else 1))

        restarts, final, _ = _drive(CalibrationManager(), identities)

        assert restarts == 0
        assert final >= 200

    def test_a_real_subject_change_does_restart(self) -> None:
        identities = [1] * 150 + [2] * 150

        restarts, final, peak = _drive(CalibrationManager(), identities)

        assert restarts == 1, "a sustained new identity must start over"
        assert final < peak

    def test_switch_threshold_is_sane(self) -> None:
        assert 20 <= CalibrationManager.IDENTITY_SWITCH_FRAMES <= 120


class TestCalibrationCanStillFinish:
    def test_a_cooperative_person_completes(self) -> None:
        manager = CalibrationManager()
        session = manager.ensure(1, "someone", _Baseline())
        assert session is not None

        for _ in range(2000):
            if session.complete:
                break
            step = session.step
            yaw = 40.0 if step and step["key"] == "turn" else 0.0
            session.offer(has_face=True, quality=0.8, yaw=yaw, pitch=0.0)

        assert session.complete
        assert session.step_index == len(CALIBRATION_STEPS)

    def test_flicker_does_not_prevent_completion(self) -> None:
        manager = CalibrationManager()
        baseline = _Baseline()
        session = None

        for i in range(4000):
            person_id = None if i % 9 == 0 else (2 if i % 23 == 0 else 1)
            session = manager.ensure(person_id, "someone", baseline)
            if session is None or session.complete:
                break
            step = session.step
            yaw = 40.0 if step and step["key"] == "turn" else 0.0
            session.offer(has_face=True, quality=0.8, yaw=yaw, pitch=0.0)

        assert session is not None
        assert session.complete, "a person who moves must still be able to finish"


class TestRejectionsDoNotResetEither:
    @pytest.mark.parametrize("kwargs", [
        {"has_face": False, "quality": 0.9},
        {"has_face": True, "quality": 0.1},
        {"has_face": True, "quality": 0.9, "yaw": 80.0},
    ])
    def test_a_rejected_frame_keeps_earlier_progress(self, kwargs) -> None:
        session = CalibrationSession(person_id=1)
        for _ in range(40):
            session.offer(has_face=True, quality=0.8, yaw=0.0, pitch=0.0)
        before = session.total_accepted

        session.offer(**kwargs)

        assert session.total_accepted == before
