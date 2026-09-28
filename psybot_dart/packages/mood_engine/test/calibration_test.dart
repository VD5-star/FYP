import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

void good(CalibrationSession session, int count) {
  for (var i = 0; i < count; i++) {
    session.offer(hasFace: true, quality: 0.9);
  }
}

void main() {
  test('calibration can actually finish', () {
    expect(PersonBaseline.required, lessThanOrEqualTo(restFrames),
        reason: 'baseline needs ${PersonBaseline.required} resting frames '
            'but calibration only collects $restFrames');
    expect(() => assertCalibrationCanFinish(), returnsNormally);
  });

  test('totals match the python source', () {
    expect(totalFrames, 1000);
    expect(restFrames, 510);
  });

  test('completing every step completes the session', () {
    final session = CalibrationSession(personId: 1);
    for (final step in calibrationSteps) {
      good(session, step.frames);
    }
    expect(session.complete, isTrue);
    expect(session.totalAccepted, totalFrames);
    expect(session.progress, closeTo(1.0, 1e-12));
  });

  test('a step cannot be skipped by waiting', () {
    final session = CalibrationSession(personId: 1);
    for (var i = 0; i < 5000; i++) {
      session.offer(hasFace: false, quality: 0.0);
    }
    expect(session.complete, isFalse);
    expect(session.totalAccepted, 0);
    expect(session.stepIndex, 0);
  });

  test('poor frames are refused with a reason', () {
    final session = CalibrationSession(personId: 1);
    expect(session.offer(hasFace: false, quality: 0.9), 'no_face');
    expect(session.offer(hasFace: true, quality: 0.05), 'low_quality');
    expect(session.offer(hasFace: true, quality: 0.9, yaw: 70),
        'extreme_angle');
    expect(session.offer(hasFace: true, quality: 0.9), isNull);
  });

  test('the turn step accepts the angles others refuse', () {
    final session = CalibrationSession(personId: 1);
    for (final step in calibrationSteps) {
      if (step.key == 'turn') break;
      good(session, step.frames);
    }

    expect(session.step, isNotNull);
    expect(session.step!.key, 'turn');
    expect(session.offer(hasFace: true, quality: 0.9, yaw: 45.0), isNull);
  });

  test('a stalled session explains itself', () {
    final session = CalibrationSession(personId: 1);
    for (var i = 0; i < 100; i++) {
      session.offer(hasFace: false, quality: 0.0);
    }
    expect(session.hint(), 'no_face');
  });

  test('no hint before there is evidence', () {
    final session = CalibrationSession(personId: 1);
    session.offer(hasFace: false, quality: 0.0);
    expect(session.hint(), isNull);
  });

  test('only resting steps feed the baseline', () {
    final probs = List<double>.filled(7, 0.0);
    final session = CalibrationSession(personId: 1);

    expect(session.step!.key, 'rest');
    expect(restFramesOnly(session, probs), isTrue);

    good(session, calibrationSteps[0].frames);
    expect(session.step!.key, 'smile');
    expect(restFramesOnly(session, probs), isFalse);

    good(session, calibrationSteps[1].frames);
    expect(session.step!.key, 'brows');
    expect(restFramesOnly(session, probs), isFalse);
  });

  test('normal operation always learns', () {
    expect(restFramesOnly(null, List<double>.filled(7, 0.0)), isTrue);
  });

  test('a fresh person needs calibration', () {
    final manager = CalibrationManager();
    expect(manager.neededFor(null), isTrue);
    expect(manager.neededFor(PersonBaseline()), isTrue);
  });

  test('a calibrated person is left alone', () {
    final manager = CalibrationManager();
    final baseline = PersonBaseline(samples: PersonBaseline.required);
    expect(manager.neededFor(baseline), isFalse);
  });

  test('a stale baseline is recalibrated', () {
    final manager = CalibrationManager();
    final old = PersonBaseline(samples: PersonBaseline.required)
      ..updatedAt = DateTime.now().millisecondsSinceEpoch / 1000.0 -
          PersonBaseline.maxAgeS -
          60;
    expect(old.expired, isTrue);
    expect(manager.neededFor(old), isTrue);
  });

  test('an unformed baseline is not called stale', () {
    final partial = PersonBaseline(samples: 5)
      ..updatedAt = DateTime.now().millisecondsSinceEpoch / 1000.0 -
          PersonBaseline.maxAgeS -
          60;
    expect(partial.expired, isFalse);
  });

  test('skipping is remembered', () {
    final manager = CalibrationManager()
      ..begin(7)
      ..skip();
    expect(manager.ensure(7, 'x', PersonBaseline()), isNull);
  });

  test('a new person restarts calibration', () {
    final manager = CalibrationManager();
    final first = manager.ensure(1, 'A', PersonBaseline());
    expect(first, isNotNull);
    good(first!, 40);

    expect(manager.ensure(2, 'B', PersonBaseline()), same(first));

    CalibrationSession? second;
    for (var i = 0; i < CalibrationManager.identitySwitchFrames; i++) {
      second = manager.ensure(2, 'B', PersonBaseline());
    }

    expect(second, isNotNull);
    expect(second, isNot(same(first)));
    expect(second!.totalAccepted, 0);
  });

  test('a recognition gap does not restart calibration', () {
    final manager = CalibrationManager();
    final first = manager.ensure(1, 'A', PersonBaseline());
    expect(first, isNotNull);
    good(first!, 40);

    CalibrationSession? session;
    for (var i = 0;
        i < CalibrationManager.identitySwitchFrames * 2;
        i++) {
      session = manager.ensure(null, null, PersonBaseline());
    }

    expect(session, same(first));
    expect(session!.totalAccepted, 40);
  });

  test('disabled manager never calibrates', () {
    final manager = CalibrationManager(enabled: false);
    expect(manager.neededFor(PersonBaseline()), isFalse);
    expect(manager.ensure(1, 'x', PersonBaseline()), isNull);
  });

  test('resetting a baseline forces recalibration', () {
    final baseline = PersonBaseline(samples: PersonBaseline.required);
    expect(baseline.ready, isTrue);
    baseline.reset();
    expect(baseline.ready, isFalse);
    expect(baseline.samples, 0);
    expect(sumOf(baseline.probs), 0.0);
  });

  test('a reloaded baseline keeps its age', () {
    final original = PersonBaseline(samples: PersonBaseline.required)
      ..updatedAt =
          DateTime.now().millisecondsSinceEpoch / 1000.0 - 3 * 24 * 3600;
    final restored = PersonBaseline.fromMap(original.toMap());
    expect(restored.ageS, greaterThan(2 * 24 * 3600));
  });

  test('every step is bilingual', () {
    for (final step in calibrationSteps) {
      expect(step.en, isNotEmpty);
      expect(step.ar, isNotEmpty);
      expect(
        step.ar.runes.any((int r) => r >= 0x600 && r <= 0x6ff),
        isTrue,
        reason: step.key,
      );
    }
  });

  test('progress payload is complete', () {
    final session = CalibrationSession(personId: 3, personName: 'X');
    good(session, 20);
    final payload = session.toMap();
    for (final key in const <String>[
      'active',
      'step_index',
      'step_count',
      'step_en',
      'step_ar',
      'step_progress',
      'progress',
      'accepted',
      'required',
      'complete',
    ]) {
      expect(payload.containsKey(key), isTrue, reason: key);
    }
    expect(payload['required'], totalFrames);
  });
}
