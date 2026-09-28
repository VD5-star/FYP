from __future__ import annotations

import math
import sys

import numpy as np

import angles as A
import movement as m
import reliability as rel
from pose_sim import IDX
from test_movement import knee_angle, make_pose
from velocity_reps import VelocityRepCounter

_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)


def squat_angles(depth: float, cycles: int = 5, frames: int = 36,
                 noise: float = 0.0, seed: int = 0):
    """Yields (angle, timestamp) for a smooth squat cycle."""
    rng = np.random.default_rng(seed)
    t = 0.0
    for _ in range(cycles):
        for i in range(frames):
            phase = (math.sin(2 * math.pi * i / frames - math.pi / 2) + 1) / 2
            pts, _ = make_pose(knee_bend=depth * phase)
            if noise:
                pts = pts + rng.normal(0, noise, pts.shape)
            yield knee_angle(pts), t
            t += 1 / 30

def test_counts_without_any_calibration() -> None:
    """The headline claim, and the reason this was built.

    Measured against the threshold counter on the same movement, with no
    calibration for either:

        user bottoming at 71 deg    threshold 5/5   velocity 5/5
        user bottoming at 115 deg   threshold 0/5   velocity 5/5
        user bottoming at 138 deg   threshold 0/5   velocity 5/5

    A position threshold has to know what the user's range means. A velocity
    reversal asks only "did they turn around", which is the same question for
    everyone.
    """
    for depth, label in ((0.95, 'deep'), (0.70, 'moderate'), (0.55, 'shallow')):
        counter = VelocityRepCounter()
        for angle, t in squat_angles(depth):
            counter.update(angle, t)
        check(counter.count == 5,
              f'{label} squatter counted {counter.count} of 5 without '
              'calibration')


def test_threshold_counter_needs_calibration_where_this_does_not() -> None:
    """Pins the comparison, so a regression in either is visible."""
    threshold = m.RepCounter(m.SQUAT)
    velocity = VelocityRepCounter()
    for angle, t in squat_angles(0.70):
        threshold.update(angle, t)
        velocity.update(angle, t)

    check(threshold.count == 0,
          'fixture is wrong: the default threshold should miss this user')
    check(velocity.count == 5,
          f'velocity counted {velocity.count} of 5 for a user the threshold '
          'cannot see')


def test_no_phantom_reps_when_still() -> None:
    """A defect this caught: the first reversal had nothing to measure against.

    `travelled` was left at zero when there was no previous reversal, so the
    minimum-travel check was skipped and the first reversal was always
    accepted. A person standing perfectly still scored exactly one phantom
    repetition in every condition - the signature of a fixed off-by-one.
    """
    for noise in (3.0, 6.0, 10.0, 15.0):
        rng = np.random.default_rng(1)
        counter = VelocityRepCounter()
        t = 0.0
        for _ in range(300):
            pts, _ = make_pose(knee_bend=0.0)
            pts = pts + rng.normal(0, noise, pts.shape)
            counter.update(knee_angle(pts), t)
            t += 1 / 30
        check(counter.count == 0,
              f'standing still with {noise:.0f}px noise produced '
              f'{counter.count} repetitions')


def test_everyday_movement_counts_nothing() -> None:
    """False positives matter more than misses.

    A missed repetition is a nuisance; a phantom one means reporting movement
    the user did not make.
    """
    cases = {
        'walking in place':
            lambda i: 0.25 + 0.12 * math.sin(2 * math.pi * i / 20),
        'sitting down once':
            lambda i: min(0.95, i / 40) if i < 120 else 0.95,
        'standing, arms moving':
            lambda i: 0.0,
    }
    for label, bend_of in cases.items():
        rng = np.random.default_rng(2)
        counter = VelocityRepCounter()
        t = 0.0
        for i in range(300):
            pts, _ = make_pose(knee_bend=bend_of(i))
            pts = pts + rng.normal(0, 3, pts.shape)
            counter.update(knee_angle(pts), t)
            t += 1 / 30
        check(counter.count == 0,
              f'{label} produced {counter.count} phantom repetitions')


def test_resting_end_is_inferred_not_assumed() -> None:
    """A squat starts extended, a pull-up starts flexed.

    The detector reads which from the direction of first movement, so it needs
    no per-exercise configuration. An earlier version waited for a full
    reversal to learn this and lost the first repetition as a result - five
    squats scored four.
    """
    descending = VelocityRepCounter()
    for angle, t in squat_angles(0.95, cycles=3):
        descending.update(angle, t)
    check(descending.count == 3,
          f'movement starting from rest-high counted {descending.count} of 3')

    rising = VelocityRepCounter()
    t = 0.0
    for _ in range(3):
        for i in range(36):
            phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2
            pts, _ = make_pose(knee_bend=0.95 * (1 - phase))
            rising.update(knee_angle(pts), t)
            t += 1 / 30
    check(rising.count >= 2,
          f'movement starting from rest-low counted {rising.count}')


def test_missing_angle_does_not_fabricate_movement() -> None:
    counter = VelocityRepCounter()
    t = 0.0
    for _ in range(60):
        counter.update(None, t)
        t += 1 / 30
    check(counter.count == 0, 'null angles produced repetitions')


def test_a_single_nan_does_not_silently_stop_counting() -> None:
    """The defect this pins, found by auditing degenerate input.

    An EMA poisoned with NaN stays NaN for ever, because
    ``NaN + alpha * (x - NaN)`` is NaN. Every later sign comparison against it
    is False, so no direction is ever established and **the counter silently
    stops counting**.

    Measured before the fix: one NaN landmark mid-session took counting from 7
    to 4 out of 7, and the three lost repetitions were every one that followed.
    Silent failure, which is worse than raising.

    NaN reaches the counter from real sources - a landmark placed exactly at a
    joint centre gives a zero-length vector.
    """
    for nan_frame in (18, 40, 55):
        counter = VelocityRepCounter()
        for i in range(108):
            phase = (math.sin(2 * math.pi * (i % 36) / 36 - math.pi / 2) + 1) / 2
            angle = float('nan') if i == nan_frame else 180 - 100 * phase
            counter.update(angle, i / 30)
        check(counter.count == 3,
              f'a NaN at frame {nan_frame} left {counter.count} of 3 '
              'repetitions')


def test_a_nan_landmark_does_not_poison_the_session() -> None:
    """End to end: one bad landmark must not cost every *later* repetition.


    Seven repetitions are performed; one of them has a NaN wrist throughout.

    The repetition containing the fault is **correctly not counted**: the
    reliability gates see a landmark that has vanished and refuse those frames,
    which is exactly their job. Declining is a first-class outcome here.

    What must not happen is the *silent* failure this test was written for -
    before the fix, the NaN poisoned the velocity filter permanently and every
    one of the three repetitions that followed was lost too, giving four.

    So six is correct and seven would be wrong: it would mean the engine had
    measured a repetition through a frame where it could not see the body.
    """
    session = m.MovementSession()
    t = 0.0

    def cycle(inject_nan: bool) -> None:
        nonlocal t
        for i in range(36):
            phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2
            pts, vis = make_pose(knee_bend=0.95 * phase)
            if inject_nan:
                pts = pts.copy()
                pts[IDX['leftWrist']] = np.nan
            session.update(pts, vis, knee_angle(pts), t)
            t += 1 / 30

    for _ in range(3):
        cycle(False)
    before = session.velocity_counter.count
    cycle(True)
    for _ in range(3):
        cycle(False)

    check(before == 3, f'setup counted {before} of 3 clean repetitions')
    check(session.velocity_counter.count == 6,
          f'a NaN landmark cost later repetitions: counted '
          f'{session.velocity_counter.count}, expected 6 (the faulty '
          'repetition is correctly refused, the three after it are not)')


def test_degenerate_input_does_not_raise() -> None:
    """Every entry point must survive nonsense rather than crash a session."""
    cases = {
        'all zeros': (np.zeros((33, 2)), np.ones(33)),
        'all NaN': (np.full((33, 2), np.nan), np.ones(33)),
        'all infinite': (np.full((33, 2), np.inf), np.ones(33)),
        'no confidence': (np.random.default_rng(0).random((33, 2)) * 100,
                          np.zeros(33)),
        'absurd scale': (np.full((33, 2), 1e12), np.ones(33)),
    }
    for label, (pts, vis) in cases.items():
        try:
            m.torso_length(pts, vis)
            m.centre_of_mass(pts, vis)
            m.viewpoint_offset_degrees(pts, vis)
            A.all_angles(pts, vis)
            m.MovementSession().update(pts, vis, None, 0.0)
        except Exception as exc:  # noqa: BLE001
            check(False, f'{label} raised {type(exc).__name__}: {exc}')
        else:
            check(True, '')


def test_gap_in_the_stream_does_not_fabricate_velocity() -> None:
    """Frames either side of a gap are not adjacent, so their difference is
    not a velocity."""
    velocity = rel.SignedVelocity()
    velocity.update(170.0, 0.0)
    velocity.update(169.0, 1 / 30)
    rate = velocity.update(90.0, 1 / 30 + 0.5)
    check(rate is None,
          f'a 500ms gap produced a velocity of {rate}')

def test_bone_length_catches_a_snapped_landmark() -> None:
    """The failure velocity gating misses.

    When a knee landmark snaps onto the other leg the jump may be small - the
    legs are close together - but the thigh segment changes length immediately
    and unmistakably.
    """
    monitor = rel.BoneLengthMonitor()
    pts, vis = make_pose()
    torso = m.torso_length(pts, vis)

    for _ in range(30):
        suspect = monitor.update(pts, vis, torso)
        check(not suspect, 'a steady pose was flagged as impossible')

    broken = pts.copy()
    broken[IDX['leftKnee']] = broken[IDX['leftKnee']] + np.array([0.0, 180.0])
    suspect = monitor.update(broken, vis, torso)
    check('leftKnee' in suspect,
          f'a lengthened femur was not detected, got {suspect}')


def test_bone_length_tolerates_normal_movement() -> None:
    """Foreshortening is correct behaviour, not a fault.

    A limb pointing toward the camera projects shorter, which is why the
    tolerance is generous and the reference is a running median rather than a
    single calibration frame.
    """
    monitor = rel.BoneLengthMonitor()
    false_alarms = 0
    t = 0.0
    for cycle in range(4):
        for i in range(36):
            phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2
            pts, vis = make_pose(knee_bend=0.95 * phase)
            torso = m.torso_length(pts, vis)
            suspect = monitor.update(pts, vis, torso)
            if cycle > 0 and suspect:
                false_alarms += 1
            t += 1 / 30
    check(false_alarms == 0,
          f'normal squatting raised {false_alarms} false bone-length alarms')


def test_velocity_gate_catches_a_teleport() -> None:
    monitor = rel.ReliabilityMonitor()
    pts, vis = make_pose()
    torso = m.torso_length(pts, vis)
    t = 0.0
    for _ in range(20):
        monitor.update(pts, vis, torso, t)
        t += 1 / 30

    jumped = pts.copy()
    jumped[IDX['leftWrist']] = jumped[IDX['leftWrist']] + np.array([500.0, 0.0])
    report = monitor.update(jumped, vis, torso, t)
    check(not report.trustworthy, 'a 500px teleport was trusted')
    check(any('fast' in r for r in report.reasons),
          f'unhelpful reasons: {report.reasons}')


def test_reliability_reacts_faster_than_the_visibility_score() -> None:
    """The measured justification for this whole module.

    MediaPipe smooths visibility with `alpha: 0.1` - confirmed by reading
    pose_landmark_filtering.pbtxt. An EMA with that coefficient takes 7 frames
    (233 ms) to fall halfway and 22 frames (733 ms) to fall 90% of the way.

    These checks compare consecutive frames, so they react in one.
    """
    alpha = 0.1
    value = 1.0
    frames_to_halfway = 0
    while value > 0.5:
        value += alpha * (0.0 - value)
        frames_to_halfway += 1
    check(frames_to_halfway >= 6,
          'the visibility lag this module exists to avoid has changed')

    monitor = rel.ReliabilityMonitor()
    pts, vis = make_pose()
    torso = m.torso_length(pts, vis)
    for i in range(20):
        monitor.update(pts, vis, torso, i / 30)

    jumped = pts.copy()
    jumped[IDX['leftWrist']] = jumped[IDX['leftWrist']] + np.array([500.0, 0.0])
    report = monitor.update(jumped, vis, torso, 20 / 30)
    check(not report.trustworthy,
          'the fast gate failed to react in the frame the fault appeared')


def test_3d_angle_matches_2d_when_facing_the_camera() -> None:
    """Neither is ground truth; their agreement is the signal.

    Square to the camera there is no foreshortening, so they should agree.
    """
    world = np.zeros((33, 3))
    world[IDX['leftHip']] = (0.0, 0.0, 0.0)
    world[IDX['leftKnee']] = (0.0, 0.4, 0.0)
    world[IDX['leftAnkle']] = (0.0, 0.8, 0.0)
    straight = rel.angle_3d(world, 'leftHip', 'leftKnee', 'leftAnkle')
    check(straight is not None and abs(straight - 180) < 1,
          f'a straight leg measured {straight} in 3D')

    check(rel.angle_disagreement(180.0, straight) < 1.0,
          'agreeing angles reported a disagreement')


def test_disagreement_flags_a_foreshortened_limb() -> None:
    monitor = rel.ReliabilityMonitor()
    pts, vis = make_pose()
    torso = m.torso_length(pts, vis)
    report = monitor.update(pts, vis, torso, 0.0,
                            angle_2d=175.0, angle_3d_value=110.0)
    check(not report.trustworthy, 'a 65 degree disagreement was trusted')
    check(any('disagree' in r for r in report.reasons),
          f'unhelpful reasons: {report.reasons}')


def test_velocity_is_scale_free() -> None:
    """Torso-relative speed means one threshold works at any distance."""
    speeds = []
    for scale in (0.6, 1.0, 1.6):
        tracker = rel.VelocityTracker()
        t = 0.0
        result = None
        for i in range(10):
            pts, vis = make_pose(knee_bend=0.05 * i, scale=scale)
            torso = m.torso_length(pts, vis)
            result = tracker.update(pts, torso, t)
            t += 1 / 30
        speeds.append(float(np.max(result)) if result is not None else 0.0)

    spread = max(speeds) - min(speeds)
    check(spread < max(speeds) * 0.25,
          f'speeds varied with distance: {[round(s, 2) for s in speeds]}')


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            _failures.append(f'{test.__name__} raised {exc!r}')

    print(f'\n  {_passes} checks passed, {len(_failures)} failed')
    for failure in _failures:
        print(f'    FAIL: {failure}')
    return 1 if _failures else 0


if __name__ == '__main__':
    sys.exit(main())
