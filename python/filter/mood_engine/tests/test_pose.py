from __future__ import annotations

import numpy as np
import pytest

from conftest import load_image
from engine.core.attributes import PoseEstimator


@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0), (45.0, 45.0), (181.0, -179.0), (-171.0, -171.0),
    (360.0, 0.0), (-190.0, 170.0), (179.9, 179.9),
    (180.0, -180.0), (540.0, -180.0),
])
def test_angle_wrapping(angle, expected):
    assert PoseEstimator._wrap_180(angle) == pytest.approx(expected, abs=1e-6)


def test_wrapping_is_idempotent():
    for a in (-179.0, -90.0, 0.0, 37.5, 179.0):
        assert PoseEstimator._wrap_180(PoseEstimator._wrap_180(a)) == \
            pytest.approx(a, abs=1e-9)


def test_upright_face_is_not_reported_upside_down(engine):
    for key in ("obama_a", "obama_b", "biden"):
        r = engine.analyse_frame(load_image(key), persist=False)
        if not r.ok:
            continue
        assert abs(r.pose.roll) <= 90.0, (
            f"{key}: roll {r.pose.roll:.1f} indicates an inverted head")


def test_pose_angles_stay_in_range(engine):
    for key in ("obama_a", "obama_b", "biden", "stranger"):
        r = engine.analyse_frame(load_image(key), persist=False)
        if not r.ok:
            continue
        assert -180.0 <= r.pose.yaw < 180.0
        assert -180.0 <= r.pose.pitch < 180.0
        assert -180.0 <= r.pose.roll < 180.0


def test_frontal_face_has_plausible_angles(engine):
    for key in ("obama_a", "obama_b", "biden"):
        r = engine.analyse_frame(load_image(key), persist=False)
        if not r.ok:
            continue
        assert abs(r.pose.pitch) <= 90.0, (
            f"{key}: pitch {r.pose.pitch:.1f} is implausible for a portrait")
        assert abs(r.pose.yaw) <= 90.0, (
            f"{key}: yaw {r.pose.yaw:.1f} is implausible for a portrait")


def test_attention_is_bounded(engine):
    for key in ("obama_a", "biden"):
        r = engine.analyse_frame(load_image(key), persist=False)
        if r.ok:
            assert 0.0 <= r.pose.attention <= 1.0


def test_liveness_signals_are_bounded(engine):
    r = engine.analyse_frame(load_image("obama_a"), persist=False)
    assert r.spoof is not None
    for name, value in r.spoof.signals.items():
        assert 0.0 <= value <= 1.0, f"signal {name} out of range: {value}"
    assert 0.0 <= r.spoof.liveness <= 1.0


def test_blink_counter_starts_at_zero(engine):
    assert engine.pose.blink_count == 0
