from __future__ import annotations

import math
import sys

import numpy as np

import angles as ang_mod
import hands as hands_mod
import occlusion as occ_mod
import skeleton as sk
from angles import IDX
from test_movement import make_pose

_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)


def calibrated(frames: int = 60, **pose_kwargs):
    """A calibrator fed a steady pose, plus that pose."""
    pts, vis = make_pose(**pose_kwargs)
    cal = sk.SkeletonCalibrator()
    for _ in range(frames):
        cal.update(pts, vis)
    return cal, pts, vis

def test_model_becomes_ready() -> None:
    cal, _, _ = calibrated()
    check(cal.model.ready,
          f'model not ready after 60 frames: {len(cal.model.lengths)} bones')


def test_model_is_not_ready_immediately() -> None:
    """Enforcing a model built from two frames would enforce noise."""
    cal, pts, vis = calibrated(frames=3)
    check(not cal.model.ready,
          'the model claimed to be ready after only three frames')


def test_lengths_are_relative_to_torso() -> None:
    """A pixel length is only right at the distance it was measured at."""
    near, _, _ = calibrated(scale=1.0)
    far, _, _ = calibrated(scale=0.5)
    for bone in near.model.lengths:
        if bone not in far.model.lengths:
            continue
        a, b = near.model.lengths[bone], far.model.lengths[bone]
        check(abs(a - b) < 0.02,
              f'{bone} measured {a:.3f} near and {b:.3f} far - the length is '
              f'not scale-invariant')


def test_symmetry_is_enforced() -> None:
    """Left and right errors are independent, so averaging is free accuracy."""
    cal, _, _ = calibrated()
    for left, right in sk.MIRRORED:
        if left in cal.model.lengths and right in cal.model.lengths:
            check(abs(cal.model.lengths[left]
                      - cal.model.lengths[right]) < 1e-9,
                  f'{left} and {right} were not averaged')


def test_calibration_ignores_unreliable_landmarks() -> None:
    """A guessed landmark must not become part of the body model."""
    pts, vis = make_pose()
    blind = vis.copy()
    blind[IDX['leftWrist']] = 0.1
    cal = sk.SkeletonCalibrator()
    for _ in range(60):
        cal.update(pts, blind)
    check('leftWrist' not in cal.model.lengths,
          'a bone was calibrated from a landmark the model was guessing at')


def test_calibration_refuses_without_a_torso() -> None:
    """No torso means no scale, so nothing measured can be normalised."""
    pts, vis = make_pose()
    blind = vis.copy()
    blind[IDX['leftHip']] = 0.1
    cal = sk.SkeletonCalibrator()
    for _ in range(60):
        cal.update(pts, blind)
    check(not cal.model.lengths,
          'bones were calibrated from a frame with no reliable torso')

def test_applying_the_model_changes_no_angle() -> None:
    """The whole justification, as a test.

    Zero, not 'small'. The constraint preserves directions exactly.
    """
    cal, _, vis = calibrated()
    worst = 0.0
    for bend in (0.0, 0.3, 0.6, 0.9):
        for raise_ in (0.0, 0.5, 1.0):
            pts, v = make_pose(knee_bend=bend, arm_raise=raise_)
            fixed = sk.apply_model(pts, cal.model)
            before = ang_mod.all_angles(pts, v)
            after = ang_mod.all_angles(fixed, v)
            for joint, a in before.items():
                b = after.get(joint)
                if a is None or b is None:
                    continue
                worst = max(worst, abs(a.degrees - b.degrees))
    check(worst < 1e-4,
          f'applying the body model changed a joint angle by {worst:.6f} '
          f'degrees - directions are being measured from moved parents again')


def test_bone_lengths_become_constant() -> None:
    """The point of the exercise: a forearm stops changing length.

    Measured against the *torso of the same frame*, because that is the scale
    every bone is rebuilt from. Bone length is fixed as a multiple of torso
    length, so any residual here would mean the constraint is not being
    applied - whereas residual measured in pixels would just be the torso
    itself moving, which is a separate and documented matter.
    """
    cal, _, vis = calibrated()
    lengths: dict[str, list[float]] = {}
    rng = np.random.default_rng(7)
    for _ in range(40):
        pts, v = make_pose(knee_bend=float(rng.uniform(0, 0.9)),
                           arm_raise=float(rng.uniform(0, 1)))
        pts = pts + rng.normal(0, 2.0, pts.shape)
        fixed = sk.apply_model(pts, cal.model)
        torso = sk.torso_length(pts)  # the scale apply_model itself used
        for child, parent in sk.TREE:
            if parent is None or child not in cal.model.lengths:
                continue
            d = float(np.linalg.norm(fixed[IDX[child]] - fixed[IDX[parent]]))
            lengths.setdefault(child, []).append(d / torso)

    for bone, values in lengths.items():
        arr = np.array(values)
        variation = arr.std() / arr.mean() * 100
        check(variation < 0.01,
              f'{bone} still varied by {variation:.3f}% after the constraint')


def test_residual_variation_is_the_torso_not_the_bones() -> None:
    """Where the remaining variation lives, stated as a test.

    After the constraint, bone length measured in *pixels* still varies - but
    only because the torso it is a multiple of varies. Measured on real
    footage, the two are equal to two decimal places. This pins that
    relationship, so a future change that reintroduces independent bone noise
    is caught.
    """
    cal, _, _ = calibrated()
    rng = np.random.default_rng(11)
    torsos: list[float] = []
    bones: list[float] = []
    for _ in range(60):
        pts, _ = make_pose(knee_bend=float(rng.uniform(0, 0.9)))
        pts = pts + rng.normal(0, 2.0, pts.shape)
        fixed = sk.apply_model(pts, cal.model)
        torsos.append(sk.torso_length(pts))
        bones.append(float(np.linalg.norm(
            fixed[IDX['leftKnee']] - fixed[IDX['leftHip']])))

    t = np.array(torsos)
    b = np.array(bones)
    torso_var = t.std() / t.mean() * 100
    bone_var = b.std() / b.mean() * 100
    check(abs(torso_var - bone_var) < 0.01,
          f'bone variation {bone_var:.3f}% does not match torso variation '
          f'{torso_var:.3f}% - bones are varying independently again')


def test_apply_is_a_no_op_without_a_model() -> None:
    pts, _ = make_pose()
    out = sk.apply_model(pts, sk.BodyModel())
    check(np.allclose(out, pts),
          'an unready model altered the landmarks')


def test_hands_and_feet_follow_their_limb() -> None:
    """Otherwise a corrected wrist leaves its hand behind."""
    cal, _, _ = calibrated()
    pts, _ = make_pose(arm_raise=0.7)
    moved = pts.copy()
    moved[IDX['leftWrist']] += np.array([40.0, 25.0])
    fixed = sk.apply_model(moved, cal.model)
    wrist_shift = fixed[IDX['leftWrist']] - moved[IDX['leftWrist']]
    thumb_shift = fixed[IDX['leftThumb']] - moved[IDX['leftThumb']]
    check(np.allclose(wrist_shift, thumb_shift, atol=1e-9),
          'the hand did not move with its wrist')

FRAME_W, FRAME_H = 1280, 720

STEADY_FRAMES = occ_mod.FRAMES_BEFORE_TRUSTED + 10


def steady_tracker(frames: int = STEADY_FRAMES):
    pts, vis = make_pose()
    cal = sk.SkeletonCalibrator()
    tracker = occ_mod.OcclusionTracker()
    t = 0.0
    for _ in range(frames):
        cal.update(pts, vis)
        tracker.update(pts, vis, cal.model, t, FRAME_W, FRAME_H)
        t += 1 / 30
    return tracker, cal, pts, vis, t


def test_a_steady_pose_is_never_inferred() -> None:
    tracker, cal, pts, vis, t = steady_tracker()
    report = tracker.update(pts, vis, cal.model, t, FRAME_W, FRAME_H)
    check(not report.inferred.any(),
          'a perfectly steady pose was reported as inferred')


def test_a_confident_wrong_wrist_is_caught() -> None:
    """The measured defect: 149 px out of place at 0.97 confidence.

    The model's own visibility cannot be relied on, so the bone-length test
    must catch this without any help from it.
    """
    tracker, cal, pts, vis, t = steady_tracker()
    broken = pts.copy()
    broken[IDX['leftWrist']] += np.array([150.0, 0.0])
    report = tracker.update(broken, vis, cal.model, t + 1 / 30, FRAME_W, FRAME_H)
    check(report.inferred[IDX['leftWrist']],
          'a wrist displaced by a full torso length at high confidence was '
          'accepted as measured')
    check(np.allclose(report.points[IDX['leftWrist']], pts[IDX['leftWrist']]),
          'the wrist was not held at its last trusted position')


def test_confidence_decays_while_held() -> None:
    tracker, cal, pts, vis, t = steady_tracker()
    broken = pts.copy()
    broken[IDX['leftWrist']] += np.array([150.0, 0.0])
    seen = []
    for i in range(1, 8):
        report = tracker.update(broken, vis, cal.model, t + i * 0.2, FRAME_W, FRAME_H)
        seen.append(report.confidence[IDX['leftWrist']])
    check(all(b <= a + 1e-9 for a, b in zip(seen, seen[1:])),
          f'confidence did not decay monotonically while held: {seen}')
    check(seen[-1] < 0.5,
          f'confidence still {seen[-1]:.2f} after 1.4 s of occlusion')


def test_a_long_occlusion_is_abandoned() -> None:
    """A limb hidden for seconds could be anywhere."""
    tracker, cal, pts, vis, t = steady_tracker()
    broken = pts.copy()
    broken[IDX['leftWrist']] += np.array([150.0, 0.0])
    for i in range(1, 20):
        report = tracker.update(broken, vis, cal.model, t + i * 0.15, FRAME_W, FRAME_H)
    check(report.discarded[IDX['leftWrist']],
          'a wrist hidden for nearly 3 seconds was still being asserted')
    check(not report.usable(IDX['leftWrist']),
          'a discarded landmark was still reported usable')
    check(not report.drawable(IDX['leftWrist']),
          'a landmark hidden for seconds would still be drawn')


def test_recovery_when_the_limb_reappears() -> None:
    """Holding must not become permanent."""
    tracker, cal, pts, vis, t = steady_tracker()
    broken = pts.copy()
    broken[IDX['leftWrist']] += np.array([150.0, 0.0])
    tracker.update(broken, vis, cal.model, t + 1 / 30, FRAME_W, FRAME_H)
    report = tracker.update(pts, vis, cal.model, t + 2 / 30, FRAME_W, FRAME_H)
    check(not report.inferred[IDX['leftWrist']],
          'the wrist stayed held after reappearing in the right place')


def test_a_stream_gap_clears_held_state() -> None:
    """Judging motion across a gap uses a velocity nobody moved at."""
    tracker, cal, pts, vis, t = steady_tracker()
    report = tracker.update(pts, vis, cal.model, t + 5.0, FRAME_W, FRAME_H)
    check(not report.inferred.any(),
          'state survived a five-second gap in the stream')

def hand_pose(extended: dict[str, bool]) -> np.ndarray:
    """21 hand landmarks with the named fingers extended or folded.

    Built radially from the wrist so the test does not depend on an upright
    hand - which is the same property `finger_extended` relies on.
    """
    pts = np.zeros((21, 2), dtype=np.float64)
    pts[0] = (0.0, 0.0)
    for i, (name, (tip, middle, base)) in enumerate(
            hands_mod.FINGER_JOINTS.items()):
        angle = math.radians(-140 + i * 22)
        direction = np.array([math.cos(angle), math.sin(angle)])
        pts[base] = direction * 30
        if extended.get(name, False):
            pts[middle] = direction * 55
            pts[tip] = direction * 80
        else:
            pts[middle] = direction * 45
            pts[tip] = direction * 25
    return pts


def test_reads_an_open_hand() -> None:
    states = hands_mod.read_fingers(hand_pose({n: True for n in
                                               hands_mod.FINGER_JOINTS}))
    check(all(states.values()), f'an open hand read as {states}')
    check(hands_mod.finger_numbers(states) == [1, 2, 3, 4, 5],
          f'open hand numbered {hands_mod.finger_numbers(states)}')


def test_reads_a_closed_hand() -> None:
    states = hands_mod.read_fingers(hand_pose({}))
    check(not any(states.values()), f'a fist read as {states}')
    check(hands_mod.finger_numbers(states) == [], 'a fist reported numbers')


def test_the_stop_gesture() -> None:
    """Fingers 1 and 5 up, 2, 3 and 4 down."""
    states = hands_mod.read_fingers(
        hand_pose({'pinky': True, 'thumb': True}))
    check(hands_mod.finger_numbers(states) == [1, 5],
          f'shaka numbered {hands_mod.finger_numbers(states)}')
    check(hands_mod.is_stop_gesture(states), 'the stop gesture was not read')


def test_similar_gestures_are_not_the_stop_gesture() -> None:
    """A command that fires on the wrong shape is worse than no command."""
    for extended, label in (
        ({'pinky': True}, 'little finger alone'),
        ({'thumb': True}, 'thumb alone'),
        ({'pinky': True, 'thumb': True, 'index': True}, 'plus index'),
        ({n: True for n in hands_mod.FINGER_JOINTS}, 'open hand'),
        ({}, 'fist'),
        ({'index': True, 'middle': True}, 'victory'),
    ):
        states = hands_mod.read_fingers(hand_pose(extended))
        check(not hands_mod.is_stop_gesture(states),
              f'{label} was read as the stop gesture')


def test_finger_reading_is_rotation_invariant() -> None:
    """A hand is rarely upright, and a y-coordinate test would fail here."""
    pts = hand_pose({'pinky': True, 'thumb': True})
    for degrees in (0, 45, 90, 180, 270):
        r = math.radians(degrees)
        rot = np.array([[math.cos(r), -math.sin(r)],
                        [math.sin(r), math.cos(r)]])
        turned = pts @ rot.T
        states = hands_mod.read_fingers(turned)
        check(hands_mod.is_stop_gesture(states),
              f'the stop gesture was lost when the hand was rotated '
              f'{degrees} degrees')

def test_gesture_requires_the_full_hold() -> None:
    """A single frame must not close the program mid-exercise."""
    gesture = hands_mod.HeldGesture(seconds=1.5)
    state = gesture.update(True, 0.0)
    check(not state.fired, 'the gesture fired on its first frame')
    for t in (0.3, 0.6, 0.9, 1.2):
        state = gesture.update(True, t)
        check(not state.fired, f'the gesture fired early, at {t} s')
    state = gesture.update(True, 1.55)
    check(state.fired, 'the gesture never fired after a full hold')


def test_gesture_fires_only_once() -> None:
    gesture = hands_mod.HeldGesture(seconds=1.0)
    for t in (0.0, 0.5, 1.1, 1.4, 2.0):
        state = gesture.update(True, t)
    fires = 0
    gesture = hands_mod.HeldGesture(seconds=1.0)
    for t in (0.0, 0.5, 1.1, 1.4, 2.0):
        if gesture.update(True, t).fired:
            fires += 1
    check(fires == 1, f'the gesture fired {fires} times while held')


def test_releasing_the_gesture_resets_it() -> None:
    gesture = hands_mod.HeldGesture(seconds=1.5, grace_seconds=0.4)
    gesture.update(True, 0.0)
    gesture.update(True, 0.5)
    gesture.update(False, 1.0)
    state = gesture.update(False, 1.6)
    check(state.progress == 0.0,
          f'progress survived the hand opening: {state.progress}')


def test_a_dropped_sample_does_not_reset_the_hold() -> None:
    """The reader samples every few frames and will miss one."""
    gesture = hands_mod.HeldGesture(seconds=1.5, grace_seconds=0.4)
    gesture.update(True, 0.0)
    gesture.update(True, 0.4)
    gesture.update(False, 0.5)      # one missed sample
    gesture.update(True, 0.6)
    state = gesture.update(True, 1.55)
    check(state.fired,
          'a single missed sample prevented the gesture from completing')


def test_progress_is_reported_for_the_countdown() -> None:
    """The ring is what lets the user abandon the command deliberately."""
    gesture = hands_mod.HeldGesture(seconds=1.0)
    gesture.update(True, 0.0)
    state = gesture.update(True, 0.5)
    check(0.4 < state.progress < 0.6,
          f'progress at half a hold was {state.progress:.2f}')


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
