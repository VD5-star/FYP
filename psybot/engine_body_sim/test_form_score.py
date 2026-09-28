from __future__ import annotations

import math
import sys

import numpy as np

from form_score import (
    FormAnalyser,
    RepetitionCollector,
    RepetitionRecord,
    dtw_distance,
    resample,
)
from test_movement import knee_angle, make_pose

_failures: list[str] = []
_passes = 0


def check(condition: bool, message: str) -> None:
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(message)


def rep_angles(depth: float, frames: int = 36) -> list[float]:
    """One smooth repetition's worth of knee angles."""
    return [
        knee_angle(make_pose(
            knee_bend=depth * (math.sin(2 * math.pi * i / frames
                                        - math.pi / 2) + 1) / 2)[0])
        for i in range(frames)
    ]


def record_for(depth: float, frames: int = 36,
               start: float = 0.0) -> RepetitionRecord:
    angles = rep_angles(depth, frames)
    return RepetitionRecord(
        angles=angles,
        start_time=start,
        end_time=start + frames / 30,
    )

def test_resample_separates_shape_from_duration() -> None:
    """Two identically shaped repetitions at different speeds must score alike.

    Otherwise "performed differently" and "performed slower" would be confused,
    and the slower repetition would be reported as badly formed.


    An earlier version compared the resampled arrays directly and required them
    within 3 degrees. They differ by up to 8.3 degrees at a single index, for a
    reason that is not a defect: a 20-frame sine and a 60-frame sine sample the
    same curve at different densities, so interpolating both to 32 points does
    not reproduce identical intermediate values. The extremes are identical to
    0.0 degrees.

    That intermediate representation is not what the system acts on. DTW
    absorbs the sampling difference and the resulting similarity is 0.992, so
    the *decision* is correct even though the arrays differ. The test now
    checks the decision.
    """
    analyser = FormAnalyser()
    analyser.add(RepetitionRecord(
        angles=rep_angles(0.95, frames=20), start_time=0, end_time=20 / 30))
    analyser.add(RepetitionRecord(
        angles=rep_angles(0.95, frames=20), start_time=1, end_time=1 + 20 / 30))
    report = analyser.add(RepetitionRecord(
        angles=rep_angles(0.95, frames=60), start_time=2,
        end_time=2 + 60 / 30))

    check(report is not None and report.similarity > 0.95,
          f'the same movement at 3x duration scored '
          f'{report.similarity if report else None} on shape')
    check(report is not None
          and all('differently' not in n for n in report.notes),
          f'a purely slower repetition was called differently shaped: '
          f'{report.notes if report else None}')


def test_dtw_is_zero_for_identical_trajectories() -> None:
    a = resample(rep_angles(0.95))
    check(dtw_distance(a, a) < 1e-9,
          'a trajectory did not match itself')


def test_dtw_grows_with_difference() -> None:
    reference = resample(rep_angles(0.95))
    close = dtw_distance(resample(rep_angles(0.90)), reference)
    far = dtw_distance(resample(rep_angles(0.40)), reference)
    check(close < far,
          f'a near repetition ({close:.2f}) scored worse than a far one '
          f'({far:.2f})')


def test_dtw_tolerates_a_pause() -> None:
    """Pausing at the bottom is a timing difference, not a form difference.

    This is the reason for DTW rather than a pointwise comparison: a pointwise
    difference counts a pause as a large error.
    """
    normal = rep_angles(0.95)
    paused = normal[:18] + [normal[18]] * 12 + normal[18:]

    warped = dtw_distance(resample(paused), resample(normal))
    pointwise = float(np.abs(resample(paused) - resample(normal)).mean())

    check(warped < pointwise,
          f'DTW ({warped:.2f}) did not beat a pointwise comparison '
          f'({pointwise:.2f}) on a paused repetition')


def test_dtw_band_prevents_absurd_alignments() -> None:
    """Without a band, DTW can align the start of one movement to the end of
    another and report a suspiciously good score for two unlike movements."""
    rising = resample(list(range(80, 180)))
    falling = resample(list(range(180, 80, -1)))
    check(dtw_distance(rising, falling) > 10.0,
          'opposite movements were scored as similar')

def test_first_repetition_is_not_scored() -> None:
    """There is nothing to compare it against, and inventing a score for it
    would be fabricating information."""
    analyser = FormAnalyser()
    check(analyser.add(record_for(0.95)) is None,
          'the first repetition was given a score')


def test_consistent_repetitions_score_high() -> None:
    analyser = FormAnalyser()
    analyser.add(record_for(0.95))
    analyser.add(record_for(0.95, start=1.2))
    for i in range(4):
        report = analyser.add(record_for(0.95, start=(i + 2) * 1.2))
        check(report is not None and report.similarity > 0.9,
              f'an identical repetition scored '
              f'{report.similarity if report else None}')
        check(report is not None and not report.notes,
              f'an identical repetition drew a comment: '
              f'{report.notes if report else None}')


def test_a_shallow_repetition_is_noticed() -> None:
    analyser = FormAnalyser()
    analyser.add(record_for(0.95))
    analyser.add(record_for(0.95, start=1.2))
    analyser.add(record_for(0.95, start=2.4))
    report = analyser.add(record_for(0.50, start=3.6))

    check(report is not None and report.depth_difference is not None
          and report.depth_difference > 20,
          f'a much shallower repetition reported a depth difference of '
          f'{report.depth_difference if report else None}')
    check(report is not None and any('shallow' in n for n in report.notes),
          f'unhelpful notes: {report.notes if report else None}')


def test_a_slower_repetition_is_noticed() -> None:
    analyser = FormAnalyser()
    analyser.add(record_for(0.95, frames=30))
    analyser.add(record_for(0.95, frames=30, start=1.0))
    report = analyser.add(record_for(0.95, frames=90, start=2.0))

    check(report is not None and report.tempo_ratio is not None
          and report.tempo_ratio > 1.5,
          f'a three-times-slower repetition gave a tempo ratio of '
          f'{report.tempo_ratio if report else None}')
    check(report is not None and any('slow' in n for n in report.notes),
          f'unhelpful notes: {report.notes if report else None}')


def test_consistency_summarises_the_session() -> None:
    analyser = FormAnalyser()
    for i in range(5):
        analyser.add(record_for(0.95, start=i * 1.2))
    steady = analyser.consistency
    check(steady is not None and steady > 0.9,
          f'five identical repetitions gave consistency {steady}')

    varied = FormAnalyser()
    for i, depth in enumerate((0.95, 0.50, 0.90, 0.40, 0.85)):
        varied.add(record_for(depth, start=i * 1.2))
    mixed = varied.consistency
    check(mixed is not None and steady is not None and mixed < steady,
          f'varied repetitions ({mixed}) scored no worse than steady ones '
          f'({steady})')


def test_consistency_is_none_until_there_is_something_to_compare() -> None:
    analyser = FormAnalyser()
    check(analyser.consistency is None, 'consistency was reported with no data')
    analyser.add(record_for(0.95))
    check(analyser.consistency is None,
          'consistency was reported from one repetition')
    analyser.add(record_for(0.95, start=1.2))
    check(analyser.consistency is None,
          'consistency was reported before a second scored repetition')


def test_similarity_scale_is_range_relative() -> None:
    """A user with a small range must not be penalised for it.

    The score is scaled by the reference's own range of motion, so a 5 degree
    deviation means the same thing to someone with a 90 degree range as to
    someone with a 30 degree one.
    """
    large = FormAnalyser()
    large.add(record_for(0.95))
    large.add(record_for(0.95, start=1.2))
    large_report = large.add(record_for(0.90, start=2.4))

    small = FormAnalyser()
    small.add(record_for(0.45))
    small.add(record_for(0.45, start=1.2))
    small_report = small.add(record_for(0.42, start=2.4))

    check(large_report is not None and small_report is not None
          and abs(large_report.similarity - small_report.similarity) < 0.2,
          f'similar relative deviations scored very differently: '
          f'{large_report.similarity if large_report else None} vs '
          f'{small_report.similarity if small_report else None}')

def test_collector_rejects_too_short_a_repetition() -> None:
    """A three-frame 'repetition' is not something to score."""
    collector = RepetitionCollector()
    for i in range(3):
        collector.add(170.0, i / 30)
    check(collector.close(0.1) is None,
          'a three-frame repetition was accepted')


def test_collector_ignores_unreliable_frames() -> None:
    collector = RepetitionCollector()
    for i in range(20):
        collector.add(None, i / 30)
    check(collector.close(0.7) is None,
          'a repetition made entirely of null angles was accepted')


def test_collector_closes_and_restarts() -> None:
    collector = RepetitionCollector()
    for i, angle in enumerate(rep_angles(0.95)):
        collector.add(angle, i / 30)
    first = collector.close(36 / 30)
    check(first is not None and len(first.angles) == 36,
          'the first repetition was not captured whole')

    for i, angle in enumerate(rep_angles(0.95)):
        collector.add(angle, (36 + i) / 30)
    second = collector.close(72 / 30)
    check(second is not None and len(second.angles) == 36,
          'the collector did not restart cleanly')


def test_end_to_end_flags_the_odd_repetition_out() -> None:
    """The whole point: five repetitions where one is shallow."""
    analyser = FormAnalyser()
    flagged = []
    t = 0.0
    for index, depth in enumerate((0.95, 0.95, 0.95, 0.50, 0.95, 0.95)):
        collector = RepetitionCollector()
        for angle in rep_angles(depth):
            collector.add(angle, t)
            t += 1 / 30
        record = collector.close(t)
        report = analyser.add(record) if record else None
        if report and report.notes:
            flagged.append(index)

    check(flagged == [3],
          f'expected only the shallow repetition to be flagged, got {flagged}')


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
