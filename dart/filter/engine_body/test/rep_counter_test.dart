import 'package:engine_body/src/core/rep_counter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame, {int fps = 30}) =>
      Duration(microseconds: (frame * 1000000 / fps).round());

  List<RepResult> feed(
    RepCounter counter,
    double angle,
    int frames,
    int startFrame, {
    int fps = 30,
  }) {
    return [
      for (var i = 0; i < frames; i++)
        counter.update(angle, at(startFrame + i, fps: fps)),
    ];
  }

  group('HysteresisConfig', () {
    test('rejects a dead zone inside the measured noise band', () {
      expect(
        () => HysteresisConfig(name: 'bad', downBelow: 100, upAbove: 110),
        throwsArgumentError,
      );
    });

    test('rejects reversed thresholds', () {
      expect(
        () => HysteresisConfig(name: 'bad', downBelow: 160, upAbove: 100),
        throwsArgumentError,
      );
    });
  });

  group('RepCounter', () {
    test('counts a clean squat', () {
      final counter = RepCounter(squatConfig);
      feed(counter, 175, 20, 0);
      expect(counter.state, RepState.up, reason: 'standing should be UP');

      feed(counter, 70, 20, 20);
      expect(counter.state, RepState.down, reason: 'deep squat should be DOWN');

      final results = feed(counter, 175, 20, 40);
      expect(counter.count, 1);
      expect(
        results.any((r) => r.completed),
        isTrue,
        reason: 'a frame should report the repetition completing',
      );
    });

    test('jitter cannot produce a phantom repetition', () {
      final counter = RepCounter(squatConfig);
      for (var i = 0; i < 200; i++) {
        counter.update(175 + (i.isEven ? -15.0 : 15.0), at(i));
      }
      expect(counter.count, 0);
    });

    test('a single-frame spike is ignored', () {
      final counter = RepCounter(squatConfig);
      for (var i = 0; i < 120; i++) {
        counter.update(i == 60 ? 30.0 : 175.0, at(i));
      }
      expect(counter.count, 0);
      expect(counter.state, RepState.up);
    });

    test('reports a partial rather than staying silent', () {
      final counter = RepCounter(squatConfig);
      feed(counter, 175, 20, 0);
      feed(counter, 130, 20, 20);
      feed(counter, 175, 20, 40);

      expect(counter.partialCount, 1);
      expect(counter.count, 0, reason: 'a partial must not count as full');
    });

    test('a bottom that is touched but not held does not count', () {
      final counter = RepCounter(squatConfig);
      feed(counter, 175, 60, 0, fps: 300);
      feed(counter, 60, 6, 60, fps: 300);
      feed(counter, 175, 60, 66, fps: 300);
      expect(counter.count, 0);
    });

    test('a single fast movement is not reported', () {
      final counter = RepCounter(squatConfig);
      feed(counter, 175, 60, 0, fps: 300);
      feed(counter, 60, 6, 60, fps: 300);
      feed(counter, 175, 60, 66, fps: 300);
      expect(counter.partialCount, 0);
    });

    test('repeated bouncing is reported', () {
      final counter = RepCounter(squatConfig);
      var frame = 0;
      feed(counter, 175, 60, frame, fps: 300);
      frame += 60;
      for (var i = 0; i < 3; i++) {
        feed(counter, 60, 6, frame, fps: 300);
        frame += 6;
        feed(counter, 175, 6, frame, fps: 300);
        frame += 6;
      }
      feed(counter, 175, 30, frame, fps: 300);
      expect(counter.count, 0);
      expect(counter.partialCount, 1);
    });

    test('clean repetitions produce no false partials', () {
      final counter = RepCounter(squatConfig);
      var frame = 0;
      for (var rep = 0; rep < 5; rep++) {
        feed(counter, 175, 15, frame);
        frame += 15;
        feed(counter, 70, 15, frame);
        frame += 15;
      }
      feed(counter, 175, 15, frame);

      expect(counter.count, 5);
      expect(
        counter.partialCount,
        0,
        reason: 'perfect movement must produce zero complaints',
      );
    });

    test('a missing angle is not a state', () {
      final counter = RepCounter(squatConfig);
      feed(counter, 175, 20, 0);
      final before = counter.state;
      for (var i = 0; i < 30; i++) {
        counter.update(null, at(20 + i));
      }
      expect(counter.state, before);
      expect(counter.count, 0);
    });

    test('an inverted exercise counts with the same machinery', () {
      final counter = RepCounter(armRaiseConfig, inverted: true);
      feed(counter, 20, 20, 0);
      feed(counter, 165, 20, 20);
      feed(counter, 20, 20, 40);
      expect(counter.count, 1);
    });

    test('reset keeps completed counts by default', () {
      final counter = RepCounter(squatConfig);
      feed(counter, 175, 20, 0);
      feed(counter, 70, 20, 20);
      feed(counter, 175, 20, 40);
      expect(counter.count, 1);

      counter.reset();
      expect(counter.count, 1);
      expect(counter.state, RepState.between);

      counter.reset(keepCounts: false);
      expect(counter.count, 0);
    });
  });

  group('RangeCalibrator', () {
    void sweep(
      RangeCalibrator calibrator,
      double low,
      double high, {
      int cycles = 3,
      int frames = 30,
    }) {
      for (var c = 0; c < cycles; c++) {
        for (var i = 0; i < frames; i++) {
          final phase = i / (frames - 1);
          final t = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
          calibrator.add(high - (high - low) * t);
        }
      }
    }

    test('brackets the range it observed', () {
      final calibrator = RangeCalibrator('squat');
      sweep(calibrator, 70, 175);
      final config = calibrator.result();

      expect(config, isNotNull, reason: calibrator.failureReason ?? '');
      expect(calibrator.observedMin!, lessThan(config!.downBelow));
      expect(config.upAbove, lessThan(calibrator.observedMax!));
      expect(config.deadZone, greaterThanOrEqualTo(20));
    });

    test('makes a limited user countable', () {
      final counter = RepCounter(squatConfig);
      var frame = 0;
      for (var rep = 0; rep < 3; rep++) {
        feed(counter, 175, 15, frame);
        frame += 15;
        feed(counter, 115, 15, frame);
        frame += 15;
      }
      expect(counter.count, 0, reason: 'default should not see this movement');

      final calibrator = RangeCalibrator('squat');
      sweep(calibrator, 115, 175);
      final config = calibrator.result();
      expect(config, isNotNull, reason: calibrator.failureReason ?? '');

      final calibrated = RepCounter(config!);
      frame = 0;
      for (var rep = 0; rep < 3; rep++) {
        feed(calibrated, 175, 15, frame);
        frame += 15;
        feed(calibrated, 115, 15, frame);
        frame += 15;
      }
      feed(calibrated, 175, 15, frame);
      expect(calibrated.count, greaterThanOrEqualTo(2));
    });

    test('refuses a range too small to threshold safely', () {
      final calibrator = RangeCalibrator('squat');
      sweep(calibrator, 165, 175);
      expect(calibrator.result(), isNull);
      expect(calibrator.failureReason, contains('range of motion'));
    });

    test('refuses too few samples', () {
      final calibrator = RangeCalibrator('squat');
      calibrator.add(175);
      calibrator.add(70);
      expect(calibrator.result(), isNull);
      expect(calibrator.failureReason, contains('usable frames'));
    });

    test('ignores unreliable frames rather than storing them', () {
      final calibrator = RangeCalibrator('squat');
      for (var i = 0; i < 50; i++) {
        calibrator.add(null);
      }
      expect(calibrator.sampleCount, 0);
      expect(calibrator.result(), isNull);
    });

    test('survives a single spike', () {
      final clean = RangeCalibrator('squat');
      sweep(clean, 70, 175);
      final baseline = clean.result()!;

      final spiked = RangeCalibrator('squat');
      sweep(spiked, 70, 175);
      spiked.add(5);
      spiked.add(179.9);
      final config = spiked.result();

      expect(config, isNotNull);
      expect(
        (config!.downBelow - baseline.downBelow).abs(),
        lessThan(8),
        reason: 'a single spike must not move the threshold far',
      );
    });
  });
}
