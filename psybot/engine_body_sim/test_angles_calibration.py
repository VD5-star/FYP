from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

import angles as A
import calibration as C
import movement as m
from pose_sim import IDX
from test_movement import knee_angle, make_pose

_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)

def test_straight_limb_is_180_degrees() -> None:
    """The convention the whole system depends on.

    Interior angles, where 180 is straight. LearnOpenCV's widely-copied
    thresholds are angles to the vertical instead, and mixing the conventions
    produces thresholds wrong by 90 degrees.
    """
    pts, vis = make_pose(knee_bend=0.0)
    angle = A.joint_angle(pts, vis, "leftHip", "leftKnee", "leftAnkle")
    check(angle is not None and abs(angle.degrees - 180.0) < 1.0,
          f"a straight leg measured {angle.degrees if angle else None}")


def test_bent_limb_is_acute() -> None:
    pts, vis = make_pose(knee_bend=0.95)
    angle = A.joint_angle(pts, vis, "leftHip", "leftKnee", "leftAnkle")
    check(angle is not None and 60 < angle.degrees < 120,
          f"a deep squat measured {angle.degrees if angle else None}")


def test_confidence_is_the_weakest_landmark() -> None:
    """A mean would let two good landmarks disguise one the model is guessing.

    The guessed landmark moves the angle just as much as the others do.
    """
    pts, vis = make_pose()
    vis = vis.copy()
    vis[IDX["leftAnkle"]] = 0.2
    angle = A.joint_angle(pts, vis, "leftHip", "leftKnee", "leftAnkle")
    check(angle is not None and abs(angle.confidence - 0.2) < 1e-6,
          f"confidence was {angle.confidence if angle else None}, expected 0.2")
    check(angle is not None and not angle.reliable,
          "an angle with a 0.2-confidence landmark was marked reliable")
    check(angle is not None and angle.or_none() is None,
          "an unreliable angle was returned as a usable number")


def test_unreliable_angles_reach_the_counter_as_none() -> None:
    """The integration that makes occlusion harmless.

    Measured in test_movement: 15% occlusion has no effect on counting, because
    None is treated as absence of information rather than as a state. That only
    holds if the angle layer actually produces None.
    """
    counter = m.RepCounter(m.SQUAT)
    t = 0.0
    for i in range(60):
        pts, vis = make_pose(knee_bend=0.0)
        vis = vis.copy()
        if i % 3 == 0:
            vis[IDX["leftKnee"]] = 0.1
        angle = A.joint_angle(pts, vis, "leftHip", "leftKnee", "leftAnkle")
        counter.update(angle.or_none() if angle else None, t)
        t += 1 / 30
    check(counter.count == 0, "occluded standing produced a repetition")
    check(counter.state is m.RepState.UP, "state was lost during occlusion")


def test_all_angles_covers_the_tracked_joints() -> None:
    pts, vis = make_pose()
    out = A.all_angles(pts, vis)
    for joint in ("leftKnee", "rightKnee", "leftHip", "leftElbow"):
        check(joint in out, f"{joint} missing from all_angles")
    check("leftAnkle" not in out,
          "ankles should be excluded - foot landmarks are the least reliable")


def test_side_tracker_holds_its_side_through_brief_occlusion() -> None:
    """The defect this prevents.

    Choosing the more confident side per frame sounds right and is wrong: left
    and right differ by several degrees even in symmetric movement, so switching
    mid-descent looks exactly like the user moving.
    """
    tracker = A.SideTracker()
    pts, vis = make_pose()

    for _ in range(10):
        tracker.update(A.all_angles(pts, vis), "Knee")
    chosen = tracker.side
    check(chosen is not None, "no side was chosen")

    dipped = vis.copy()
    dipped[IDX[f"{chosen}Knee"]] = 0.3
    for _ in range(3):
        tracker.update(A.all_angles(pts, dipped), "Knee")
    check(tracker.side == chosen,
          "a 3-frame confidence dip caused a side switch")


def test_side_tracker_switches_when_persistently_worse() -> None:
    tracker = A.SideTracker()
    pts, vis = make_pose()
    for _ in range(10):
        tracker.update(A.all_angles(pts, vis), "Knee")
    chosen = tracker.side

    occluded = vis.copy()
    for name in ("Hip", "Knee", "Ankle"):
        occluded[IDX[f"{chosen}{name}"]] = 0.2
    for _ in range(10):
        tracker.update(A.all_angles(pts, occluded), "Knee")
    check(tracker.side != chosen,
          "a persistently occluded side was never given up")


def test_symmetry_detects_favouring_one_leg() -> None:
    pts, vis = make_pose(knee_bend=0.5)
    balanced = A.symmetry(A.all_angles(pts, vis), "Knee")
    check(balanced is not None and balanced < 5.0,
          f"a symmetric pose measured {balanced} degrees of asymmetry")

    lopsided = pts.copy()
    lopsided[IDX["leftKnee"]] = lopsided[IDX["leftKnee"]] + np.array([60.0, 0.0])
    uneven = A.symmetry(A.all_angles(lopsided, vis), "Knee")
    check(uneven is not None and uneven > 10.0,
          f"a lopsided pose measured only {uneven} degrees of asymmetry")

def _sweep(calibrator: C.RangeCalibrator, low_bend: float, high_bend: float,
           cycles: int = 3, frames: int = 30) -> None:
    for _ in range(cycles):
        for i in range(frames):
            phase = (math.sin(2 * math.pi * i / frames - math.pi / 2) + 1) / 2
            bend = low_bend + (high_bend - low_bend) * phase
            pts, _ = make_pose(knee_bend=bend)
            calibrator.add(knee_angle(pts))


def test_calibration_brackets_the_users_range() -> None:
    calibrator = C.RangeCalibrator("squat")
    _sweep(calibrator, 0.0, 0.95)
    result = calibrator.result()

    check(result.ok, f"calibration failed: {result.reason}")
    if result.config:
        check(result.observed_min < result.config.down_below,
              "the down threshold is below the user's deepest point")
        check(result.config.up_above < result.observed_max,
              "the up threshold is above the user's highest point")
        check(result.config.dead_zone >= 20.0,
              f"dead zone {result.config.dead_zone:.0f} is inside the noise")


def test_calibration_helps_a_limited_user() -> None:
    """The case this feature exists for.

    Someone whose deepest squat is 115 degrees never crosses the default 100
    degree threshold, so **nothing ever counts** and the app silently reports
    they did nothing. For a mental-health application that is worse than
    useless.
    """
    limited_low, limited_high = 0.0, 0.70

    default_counter = m.RepCounter(m.SQUAT)
    t = 0.0
    for _ in range(3):
        for i in range(30):
            phase = (math.sin(2 * math.pi * i / 30 - math.pi / 2) + 1) / 2
            pts, _ = make_pose(knee_bend=limited_low +
                               (limited_high - limited_low) * phase)
            default_counter.update(knee_angle(pts), t)
            t += 1 / 30
    check(default_counter.count == 0,
          "fixture is wrong: this movement should not clear the default")

    calibrator = C.RangeCalibrator("squat")
    _sweep(calibrator, limited_low, limited_high)
    result = calibrator.result()
    check(result.ok, f"calibration failed for a limited user: {result.reason}")

    if result.config:
        counter = m.RepCounter(result.config)
        t = 0.0
        for _ in range(3):
            for i in range(30):
                phase = (math.sin(2 * math.pi * i / 30 - math.pi / 2) + 1) / 2
                pts, _ = make_pose(knee_bend=limited_low +
                                   (limited_high - limited_low) * phase)
                counter.update(knee_angle(pts), t)
                t += 1 / 30
        check(counter.count >= 2,
              f"after calibration only {counter.count} of 3 reps counted")


def test_calibration_refuses_too_small_a_range() -> None:
    """Failing loudly beats shipping a threshold inside the noise."""
    calibrator = C.RangeCalibrator("squat")
    _sweep(calibrator, 0.0, 0.08)
    result = calibrator.result()
    check(not result.ok, "a near-zero range of motion was accepted")
    check("range of motion" in result.reason,
          f"unhelpful reason: {result.reason!r}")


def test_calibration_refuses_too_few_samples() -> None:
    calibrator = C.RangeCalibrator("squat")
    for bend in (0.0, 0.9):
        pts, _ = make_pose(knee_bend=bend)
        calibrator.add(knee_angle(pts))
    result = calibrator.result()
    check(not result.ok, "calibration accepted two samples")
    check("usable frames" in result.reason,
          f"unhelpful reason: {result.reason!r}")


def test_calibration_ignores_unreliable_frames() -> None:
    calibrator = C.RangeCalibrator("squat")
    for _ in range(50):
        calibrator.add(None)
    check(calibrator.count == 0, "None was stored as a sample")
    check(not calibrator.result().ok, "50 unusable frames passed calibration")


def test_calibration_survives_a_single_bad_frame() -> None:
    """Percentiles rather than min/max.

    A calibration error is permanent in a way a per-frame error is not: one bad
    frame at the bottom of a squat would otherwise set the threshold for every
    future session.
    """
    clean = C.RangeCalibrator("squat")
    _sweep(clean, 0.0, 0.95)
    baseline = clean.result()

    spiked = C.RangeCalibrator("squat")
    _sweep(spiked, 0.0, 0.95)
    spiked.add(5.0)      # an impossible angle from a limb swap
    spiked.add(179.9)
    result = spiked.result()

    check(result.ok, "one bad frame broke calibration entirely")
    if result.config and baseline.config:
        drift = abs(result.config.down_below - baseline.config.down_below)
        check(drift < 8.0,
              f"a single spike moved the threshold by {drift:.1f} degrees")


def test_profile_round_trips() -> None:
    calibrator = C.RangeCalibrator("squat")
    _sweep(calibrator, 0.0, 0.95)
    result = calibrator.result()

    profile = C.UserProfile()
    check(profile.store("squat", result), "a valid calibration was rejected")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.json"
        profile.save(path)
        loaded = C.UserProfile.load(path)
        restored = loaded.config_for("squat", m.SQUAT)
        check(result.config is not None
              and abs(restored.down_below - result.config.down_below) < 1e-6,
              "thresholds did not survive a save/load round trip")


def test_profile_falls_back_for_unknown_exercise() -> None:
    profile = C.UserProfile()
    config = profile.config_for("pushup", m.PUSHUP)
    check(config is m.PUSHUP, "an uncalibrated exercise did not use the default")


def test_corrupt_profile_does_not_crash() -> None:
    """A corrupt profile means an uncalibrated user, not a crash.

    That is a state the rest of the system already handles.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text("{ not json at all", encoding="utf-8")
        profile = C.UserProfile.load(path)
        check(profile.config_for("squat", m.SQUAT) is m.SQUAT,
              "a corrupt profile did not fall back to defaults")

        path.write_text('{"exercises": {"squat": {"down_below": 150}}}',
                        encoding="utf-8")
        profile = C.UserProfile.load(path)
        check(profile.config_for("squat", m.SQUAT) is m.SQUAT,
              "a profile missing up_above was trusted")


def test_stored_profile_that_no_longer_validates_is_rejected() -> None:
    """Rules can tighten between versions. Old profiles must not bypass them."""
    profile = C.UserProfile(exercises={
        "squat": {"down_below": 100.0, "up_above": 105.0,
                  "observed_min": 90.0, "observed_max": 115.0},
    })
    config = profile.config_for("squat", m.SQUAT)
    check(config is m.SQUAT,
          "a stored 5 degree dead zone was accepted despite the 20 degree rule")


def test_calibrated_counter_end_to_end() -> None:
    calibrator = C.RangeCalibrator("squat")
    _sweep(calibrator, 0.0, 0.95)
    result = calibrator.result()
    check(result.ok and result.config is not None, "calibration failed")

    if result.config:
        session = m.MovementSession(rep_config=result.config)
        t = 0.0
        for _ in range(4):
            for i in range(36):
                phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2
                pts, vis = make_pose(knee_bend=0.95 * phase)
                angle = A.joint_angle(pts, vis, "leftHip", "leftKnee",
                                      "leftAnkle")
                session.update(pts, vis, angle.or_none() if angle else None, t)
                t += 1 / 30
        check(session.rep_counter.count == 4,
              f"calibrated session counted {session.rep_counter.count} of 4")
        check(session.rep_counter.partial_count == 0,
              f"calibrated session produced "
              f"{session.rep_counter.partial_count} false partials")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            _failures.append(f"{test.__name__} raised {exc!r}")

    print(f"\n  {_passes} checks passed, {len(_failures)} failed")
    for failure in _failures:
        print(f"    FAIL: {failure}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
