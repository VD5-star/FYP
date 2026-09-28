import 'dart:math' as math;

import 'package:engine_body/src/core/form_score.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  List<double> repAngles(double depth, {int frames = 36}) => [
        for (var i = 0; i < frames; i++)
          180 -
              (180 - depth) *
                  (math.sin(2 * math.pi * i / frames - math.pi / 2) + 1) /
                  2,
      ];

  RepetitionRecord recordFor(
    double depth, {
    int frames = 36,
    double startSeconds = 0,
  }) {
    final start = Duration(milliseconds: (startSeconds * 1000).round());
    return RepetitionRecord(
      angles: repAngles(depth, frames: frames),
      startTime: start,
      endTime: start + Duration(milliseconds: (frames / 30 * 1000).round()),
    );
  }

  group('dtwDistance', () {
    test('is zero for identical trajectories', () {
      final a = resampleTrajectory(repAngles(80));
      expect(dtwDistance(a, a), lessThan(1e-9));
    });

    test('grows with difference', () {
      final reference = resampleTrajectory(repAngles(80));
      final close = dtwDistance(resampleTrajectory(repAngles(90)), reference);
      final far = dtwDistance(resampleTrajectory(repAngles(150)), reference);
      expect(close, lessThan(far));
    });

    test('tolerates a pause', () {
      final normal = repAngles(80);
      final paused = [
        ...normal.take(18),
        ...List<double>.filled(12, normal[18]),
        ...normal.skip(18),
      ];

      final warped = dtwDistance(
        resampleTrajectory(paused),
        resampleTrajectory(normal),
      );

      final a = resampleTrajectory(paused);
      final b = resampleTrajectory(normal);
      var pointwise = 0.0;
      for (var i = 0; i < a.length; i++) {
        pointwise += (a[i] - b[i]).abs();
      }
      pointwise /= a.length;

      expect(warped, lessThan(pointwise));
    });

    test('does not align opposite movements', () {
      final rising = resampleTrajectory([for (var i = 80; i < 180; i++) i * 1.0]);
      final falling =
          resampleTrajectory([for (var i = 180; i > 80; i--) i * 1.0]);
      expect(dtwDistance(rising, falling), greaterThan(10));
    });
  });

  group('resampleTrajectory', () {
    test('preserves the extremes', () {
      final resampled = resampleTrajectory(repAngles(80, frames: 20));
      expect(resampled.first, closeTo(180, 1));
      expect(resampled.reduce(math.min), closeTo(80, 1));
    });

    test('handles degenerate input rather than throwing', () {
      expect(resampleTrajectory(const []), hasLength(32));
      expect(resampleTrajectory(const [150.0]), everyElement(150.0));
    });
  });

  group('FormAnalyser', () {
    test('does not score before a reference exists', () {
      final analyser = FormAnalyser();
      expect(analyser.add(recordFor(80)), isNull);
      expect(analyser.add(recordFor(80, startSeconds: 1.2)), isNull);
    });

    test('consistent repetitions score high and draw no comment', () {
      final analyser = FormAnalyser();
      analyser.add(recordFor(80));
      analyser.add(recordFor(80, startSeconds: 1.2));

      for (var i = 0; i < 4; i++) {
        final report = analyser.add(recordFor(80, startSeconds: (i + 2) * 1.2));
        expect(report, isNotNull);
        expect(report!.similarity, greaterThan(0.9));
        expect(
          report.notes,
          isEmpty,
          reason: 'identical repetitions must pass silently',
        );
      }
    });

    test('a shallow repetition is noticed', () {
      final analyser = FormAnalyser();
      analyser.add(recordFor(80));
      analyser.add(recordFor(80, startSeconds: 1.2));
      analyser.add(recordFor(80, startSeconds: 2.4));
      final report = analyser.add(recordFor(140, startSeconds: 3.6));

      expect(report, isNotNull);
      expect(report!.depthDifference, greaterThan(20));
      expect(report.notes.join(), contains('shallow'));
    });

    test('a slower repetition is noticed but not called misshapen', () {
      final analyser = FormAnalyser();
      analyser.add(recordFor(80, frames: 30));
      analyser.add(recordFor(80, frames: 30, startSeconds: 1));
      final report = analyser.add(
        recordFor(80, frames: 90, startSeconds: 2),
      );

      expect(report, isNotNull);
      expect(report!.tempoRatio, greaterThan(1.5));
      expect(report.notes.join(), contains('slow'));
      expect(report.similarity, greaterThan(0.9));
      expect(report.notes.join(), isNot(contains('differently')));
    });

    test('the score scale is relative to the user range', () {
      final large = FormAnalyser();
      large.add(recordFor(80));
      large.add(recordFor(80, startSeconds: 1.2));
      final largeReport = large.add(recordFor(90, startSeconds: 2.4));

      final small = FormAnalyser();
      small.add(recordFor(140));
      small.add(recordFor(140, startSeconds: 1.2));
      final smallReport = small.add(recordFor(143, startSeconds: 2.4));

      expect(
        (largeReport!.similarity - smallReport!.similarity).abs(),
        lessThan(0.2),
      );
    });

    test('consistency summarises the session', () {
      final steady = FormAnalyser();
      for (var i = 0; i < 6; i++) {
        steady.add(recordFor(80, startSeconds: i * 1.2));
      }

      final varied = FormAnalyser();
      const depths = [80.0, 80.0, 140.0, 90.0, 150.0, 95.0];
      for (var i = 0; i < depths.length; i++) {
        varied.add(recordFor(depths[i], startSeconds: i * 1.2));
      }

      expect(steady.consistency, greaterThan(0.9));
      expect(varied.consistency, lessThan(steady.consistency!));
    });

    test('consistency is null until there is something to compare', () {
      final analyser = FormAnalyser();
      expect(analyser.consistency, isNull);
      analyser.add(recordFor(80));
      expect(analyser.consistency, isNull);
    });
  });

  group('RepetitionCollector', () {
    test('rejects a repetition too short to describe a movement', () {
      final collector = RepetitionCollector();
      for (var i = 0; i < 3; i++) {
        collector.add(170, Duration(milliseconds: i * 33));
      }
      expect(collector.close(const Duration(milliseconds: 100)), isNull);
    });

    test('ignores unreliable frames', () {
      final collector = RepetitionCollector();
      for (var i = 0; i < 20; i++) {
        collector.add(null, Duration(milliseconds: i * 33));
      }
      expect(collector.close(const Duration(milliseconds: 700)), isNull);
    });

    test('closes and restarts cleanly', () {
      final collector = RepetitionCollector();
      final angles = repAngles(80);
      for (var i = 0; i < angles.length; i++) {
        collector.add(angles[i], Duration(milliseconds: i * 33));
      }
      final first = collector.close(const Duration(milliseconds: 1200));
      expect(first, isNotNull);
      expect(first!.angles, hasLength(36));

      for (var i = 0; i < angles.length; i++) {
        collector.add(angles[i], Duration(milliseconds: 1200 + i * 33));
      }
      final second = collector.close(const Duration(milliseconds: 2400));
      expect(second, isNotNull);
      expect(second!.angles, hasLength(36));
    });
  });

  test('end to end: only the odd repetition out is flagged', () {
    final analyser = FormAnalyser();
    final flagged = <int>[];
    const depths = [80.0, 80.0, 80.0, 140.0, 80.0, 80.0];

    for (var index = 0; index < depths.length; index++) {
      final collector = RepetitionCollector();
      final angles = repAngles(depths[index]);
      for (var i = 0; i < angles.length; i++) {
        collector.add(
          angles[i],
          Duration(milliseconds: (index * 36 + i) * 33),
        );
      }
      final record = collector.close(
        Duration(milliseconds: (index + 1) * 36 * 33),
      );
      final report = record == null ? null : analyser.add(record);
      if (report != null && report.notes.isNotEmpty) flagged.add(index);
    }

    expect(flagged, [3]);
  });
}
