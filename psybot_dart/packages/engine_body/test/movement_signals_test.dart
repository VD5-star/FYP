import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/pose_fixtures.dart';

void main() {
  final t0 = DateTime(2026, 1, 1);
  DateTime at(int frameIndex, {int fps = 15}) =>
      t0.add(Duration(microseconds: (1000000 ~/ fps) * frameIndex));

  group('MovementAnalyser', () {
    test('a single frame yields no signal, not a zero signal', () {
      final analyser = MovementAnalyser();
      analyser.add(PoseFixtures.frame(PoseFixtures.standing(), at: at(0)));
      expect(analyser.signals.sampleCount, 0);
      expect(analyser.signals, same(MovementSignals.unavailable));
    });

    test('an unchanging subject produces near-zero movement energy', () {
      final analyser = MovementAnalyser();
      for (final frame in PoseFixtures.stillSequence(frames: 20)) {
        analyser.add(frame);
      }

      final signals = analyser.signals;
      expect(signals.sampleCount, greaterThan(1));
      expect(signals.movementEnergy, lessThan(1e-6));
      expect(signals.stillness, closeTo(1.0, 1e-4));
    });

    test('a moving subject produces more energy than a still one', () {
      final still = MovementAnalyser(smooth: false);
      for (final frame in PoseFixtures.stillSequence(frames: 20)) {
        still.add(frame);
      }

      final moving = MovementAnalyser(smooth: false);
      for (var i = 0; i < 20; i++) {
        final pose = i.isEven
            ? PoseFixtures.standing()
            : PoseFixtures.leftArmRaised();
        moving.add(PoseFixtures.frame(pose, at: at(i)));
      }

      expect(
        moving.signals.movementEnergy,
        greaterThan(still.signals.movementEnergy),
      );
      expect(moving.signals.stillness, lessThan(still.signals.stillness));
    });

    test('energy is a speed, so it is unchanged by frame rate', () {
      double energyAt(int fps) {
        final analyser = MovementAnalyser(smooth: false);
        for (var i = 0; i < fps; i++) {
          final seconds = i / fps;
          final pose = _wristAt(0.09 + 0.1 * seconds);
          analyser.add(
            PoseFixtures.frame(
              pose,
              at: t0.add(Duration(microseconds: (seconds * 1e6).round())),
            ),
          );
        }
        return analyser.signals.movementEnergy;
      }

      expect(energyAt(30), closeTo(energyAt(10), 1e-6));
    });

    test('movement energy is unaffected by the subject stepping back', () {
      double energyAtScale(double scale) {
        final analyser = MovementAnalyser(smooth: false);
        for (var i = 0; i < 10; i++) {
          final pose = i.isEven
              ? PoseFixtures.standing(scale: scale)
              : PoseFixtures.leftArmRaised(scale: scale);
          analyser.add(PoseFixtures.frame(pose, at: at(i)));
        }
        return analyser.signals.movementEnergy;
      }

      expect(energyAtScale(0.5), closeTo(energyAtScale(1.5), 1e-6));
    });

    test('frames without a usable torso are ignored', () {
      final analyser = MovementAnalyser();
      for (var i = 0; i < 10; i++) {
        analyser.add(
          PoseFixtures.frame(PoseFixtures.standing(confidence: 0.2), at: at(i)),
        );
      }
      expect(analyser.hasTracking, isFalse);
      expect(analyser.signals.sampleCount, 0);
    });

    test('the window discards samples older than its duration', () {
      final analyser = MovementAnalyser(window: const Duration(seconds: 1));
      for (final frame in PoseFixtures.stillSequence(frames: 45)) {
        analyser.add(frame);
      }
      expect(analyser.signals.sampleCount, lessThanOrEqualTo(16));
    });

    test('reports the worst framing in the window, not the most common', () {
      final analyser = MovementAnalyser();
      for (var i = 0; i < 10; i++) {
        final pose = PoseFixtures.standing(legsVisible: i != 7);
        analyser.add(PoseFixtures.frame(pose, at: at(i)));
      }
      expect(analyser.signals.framing, Framing.upperBodyOnly);
    });

    test('reset clears all state', () {
      final analyser = MovementAnalyser();
      for (final frame in PoseFixtures.stillSequence(frames: 10)) {
        analyser.add(frame);
      }
      expect(analyser.hasTracking, isTrue);
      analyser.reset();
      expect(analyser.hasTracking, isFalse);
      expect(analyser.signals.sampleCount, 0);
    });

    test('signals carry a confidence a consumer can weight by', () {
      final analyser = MovementAnalyser();
      for (final frame in PoseFixtures.stillSequence(frames: 10)) {
        analyser.add(frame);
      }
      expect(analyser.signals.confidence, greaterThan(0.5));
    });
  });

  group('OneEuroFilter', () {
    test('passes the first sample through unchanged', () {
      final filter = OneEuroFilter();
      expect(filter.filter(0.42, t0), 0.42);
    });

    test('reduces jitter around a constant value', () {
      final filter = OneEuroFilter();
      var jittered = 0.0;
      var filtered = 0.0;
      const truth = 0.5;

      for (var i = 0; i < 60; i++) {
        final noisy = truth + (i.isEven ? 0.01 : -0.01);
        final out = filter.filter(noisy, at(i));
        if (i > 10) {
          jittered += (noisy - truth).abs();
          filtered += (out - truth).abs();
        }
      }
      expect(filtered, lessThan(jittered));
    });

    test('does not smooth across a tracking gap after reset', () {
      final filter = OneEuroFilter();
      filter.filter(0.1, at(0));
      filter.reset();
      expect(filter.filter(0.9, at(30)), 0.9);
    });

    test('survives duplicate timestamps without producing infinity', () {
      final filter = OneEuroFilter();
      filter.filter(0.1, t0);
      final out = filter.filter(0.2, t0);
      expect(out.isFinite, isTrue);
    });
  });
}

List<BodyLandmark> _wristAt(double dx) {
  final base = PoseFixtures.standing();
  return [
    for (final l in base)
      if (l.type == LandmarkType.leftWrist) l.copyWith(x: 0.5 + dx) else l,
  ];
}
