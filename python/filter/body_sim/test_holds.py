from __future__ import annotations

import math
import sys

import numpy as np

import movement as m
from holds import EXERCISES, HoldDetector
from test_movement import knee_angle, make_pose

_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)


def drive(bend_of, seconds: float = 8.0, noise: float = 3.0,
          seed: int = 0, detector: HoldDetector | None = None):
    """Runs a detector over a movement, returning it and any completed holds."""
    rng = np.random.default_rng(seed)
    detector = detector or HoldDetector()
    events = []
    t = 0.0
    for i in range(int(seconds * 30)):
        pts, vis = make_pose(knee_bend=bend_of(i))
        pts = pts + rng.normal(0, noise, pts.shape)
        state = detector.update(pts, vis, m.torso_length(pts, vis), t,
                                angle=knee_angle(pts))
        if state.completed:
            events.append(state.completed)
        t += 1 / 30
    final = detector.finish(t)
    if final:
        events.append(final)
    return detector, events


def test_a_still_position_is_detected_as_a_hold() -> None:
    _, events = drive(lambda i: 0.4)
    check(len(events) == 1, f'a still 8s position gave {len(events)} holds')
    check(events and abs(events[0].duration - 8.0) < 0.5,
          f'duration was {events[0].duration if events else None}, expected ~8')


def test_continuous_movement_is_not_a_hold() -> None:
    """The defect this pins.

    A first threshold of 2.0 torso lengths per second was reasoned from the
    noise floor rather than measured, and a person doing continuous full squats
    was recorded as holding a position for eight unbroken seconds.

    Measuring the real speeds settled it: standing still peaks at 0.85 and a
    normal squat medians at 1.27. The gap is far narrower than intuition
    suggests.
    """
    _, events = drive(lambda i: (math.sin(2 * math.pi * i / 40) + 1) / 2 * 0.9)
    check(not events,
          f'continuous squatting produced {len(events)} phantom holds')


def test_a_small_wobble_does_not_break_a_hold() -> None:
    """Real holds include small corrections. A detector that abandons on the
    first one is unusable."""
    _, events = drive(
        lambda i: 0.4 + 0.01 * math.sin(2 * math.pi * i / 25))
    check(len(events) == 1, f'a wobbling hold gave {len(events)} holds')
    check(events and events[0].duration > 7.0,
          f'a wobble truncated the hold to '
          f'{events[0].duration if events else None}s')


def test_a_hold_ends_when_movement_starts() -> None:
    _, events = drive(
        lambda i: 0.4 if i < 150 else (math.sin(2 * math.pi * i / 40) + 1) / 2 * 0.9)
    check(len(events) == 1, f'expected one hold, got {len(events)}')
    check(events and 4.0 < events[0].duration < 5.6,
          f'the hold measured {events[0].duration if events else None}s, '
          'expected about 5')


def test_a_brief_pause_is_not_a_hold() -> None:
    """Below the minimum duration it is a pause, not a hold."""
    detector = HoldDetector(min_duration=3.0)
    _, events = drive(
        lambda i: 0.4 if i < 45 else (math.sin(2 * math.pi * i / 40) + 1) / 2 * 0.9,
        detector=detector)
    check(not events,
          f'a 1.5s pause was reported as a hold: {events}')


def test_steadiness_distinguishes_calm_from_trembling() -> None:
    """Thirty seconds of trembling is a different event from thirty seconds of
    calm, and the engine can report which without interpreting it."""
    _, calm = drive(lambda i: 0.4, noise=1.0, seed=1)
    _, shaky = drive(lambda i: 0.4, noise=6.0, seed=1)

    check(calm and shaky, 'both conditions should produce a hold')
    if calm and shaky:
        check(calm[0].steadiness > shaky[0].steadiness,
              f'a calm hold ({calm[0].steadiness:.2f}) did not score steadier '
              f'than a shaky one ({shaky[0].steadiness:.2f})')


def test_lost_tracking_does_not_abandon_a_hold() -> None:
    """A single dropped frame is not evidence that the person moved."""
    detector = HoldDetector()
    rng = np.random.default_rng(0)
    t = 0.0
    for i in range(240):
        pts, vis = make_pose(knee_bend=0.4)
        pts = pts + rng.normal(0, 3, pts.shape)
        if i % 40 == 0:
            state = detector.update(pts, vis, None, t, angle=None)
            check(state.holding or i < 40,
                  f'an unmeasurable frame at {i} abandoned the hold')
        else:
            detector.update(pts, vis, m.torso_length(pts, vis), t,
                            angle=knee_angle(pts))
        t += 1 / 30


def test_mean_angle_is_reported() -> None:
    _, events = drive(lambda i: 0.4)
    check(events and events[0].mean_angle is not None,
          'no mean angle was reported')
    if events and events[0].mean_angle is not None:
        expected = knee_angle(make_pose(knee_bend=0.4)[0])
        check(abs(events[0].mean_angle - expected) < 8,
              f'mean angle {events[0].mean_angle:.0f} differs from the held '
              f'angle {expected:.0f}')


def test_total_accumulates_across_holds() -> None:
    detector = HoldDetector()

    def pattern(i: int) -> float:
        if i < 120:
            return 0.4
        if i < 180:
            return (math.sin(2 * math.pi * i / 40) + 1) / 2 * 0.9
        return 0.4

    _, events = drive(pattern, seconds=12, detector=detector)
    check(len(events) == 2, f'expected two holds, got {len(events)}')
    check(detector.total_seconds > 6.0,
          f'total held time was {detector.total_seconds:.1f}s')

def test_every_preset_is_coherent() -> None:
    for name, exercise in EXERCISES.items():
        check(exercise.kind in ('reps', 'hold'),
              f'{name} has unknown kind {exercise.kind!r}')
        check(bool(exercise.description),
              f'{name} has no camera guidance, which users need')
        if exercise.kind == 'reps':
            try:
                config = exercise.config()
                check(config.dead_zone >= 20,
                      f'{name} has a {config.dead_zone:.0f} degree dead zone')
            except ValueError as exc:
                check(False, f'{name} produced an invalid config: {exc}')


def test_presets_only_track_reliable_joints() -> None:
    """Every preset tracks a large joint in the camera plane.

    Movements depending on rotation, small joints or depth are absent - not
    because they are unimportant, but because a single phone camera cannot
    measure them honestly.
    """
    allowed = {'Knee', 'Hip', 'Elbow', 'Shoulder'}
    for name, exercise in EXERCISES.items():
        check(exercise.joint in allowed,
              f'{name} tracks {exercise.joint}, which is not reliably visible')


def test_rep_presets_work_with_the_counter() -> None:
    for name, exercise in EXERCISES.items():
        if exercise.kind != 'reps':
            continue
        counter = m.RepCounter(exercise.config(), inverted=exercise.inverted)
        check(counter.state is m.RepState.BETWEEN,
              f'{name} did not initialise cleanly')


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
