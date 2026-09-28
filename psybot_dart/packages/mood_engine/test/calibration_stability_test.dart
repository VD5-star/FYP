import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

(int, int, int) drive(CalibrationManager manager, List<int?> identities) {
  final baseline = PersonBaseline();
  var restarts = 0;
  var previous = 0;
  var peak = 0;

  for (final personId in identities) {
    final session = manager.ensure(personId, 'someone', baseline);
    if (session == null) continue;
    session.offer(hasFace: true, quality: 0.8);
    final accepted = session.totalAccepted;
    if (accepted < previous) restarts += 1;
    previous = accepted;
    if (accepted > peak) peak = accepted;
  }

  return (restarts, previous, peak);
}

void main() {
  group('identity flicker does not reset progress', () {
    test('stable identity accumulates', () {
      final result = drive(CalibrationManager(), List<int?>.filled(300, 1));
      expect(result.$1, 0);
      expect(result.$2, 300);
    });

    test('dropout to unknown keeps progress', () {
      final identities = <int?>[
        for (var i = 0; i < 300; i++) i % 7 == 0 ? null : 1,
      ];

      final result = drive(CalibrationManager(), identities);

      expect(result.$1, 0);
      expect(result.$2, greaterThanOrEqualTo(290));
    });

    test('brief wrong match keeps progress', () {
      final identities = <int?>[
        for (var i = 0; i < 300; i++) i % 11 == 0 ? 2 : 1,
      ];

      final result = drive(CalibrationManager(), identities);

      expect(result.$1, 0, reason: 'a one-frame mismatch must not restart');
      expect(result.$2, greaterThanOrEqualTo(250));
    });

    test('alternating noise keeps progress', () {
      final identities = <int?>[
        for (var i = 0; i < 300; i++)
          if (i % 5 == 0) null else (i % 13 == 0 ? 3 : 1),
      ];

      final result = drive(CalibrationManager(), identities);

      expect(result.$1, 0);
      expect(result.$2, greaterThanOrEqualTo(200));
    });

    test('a real subject change does restart', () {
      final identities = <int?>[
        ...List<int?>.filled(150, 1),
        ...List<int?>.filled(150, 2),
      ];

      final result = drive(CalibrationManager(), identities);

      expect(result.$1, 1, reason: 'a sustained new identity must start over');
      expect(result.$2, lessThan(result.$3));
      expect(result.$2, 106);
      expect(result.$3, 194);
    });

    test('switch threshold is sane', () {
      expect(CalibrationManager.identitySwitchFrames,
          inInclusiveRange(20, 120));
    });
  });

  group('calibration can still finish', () {
    test('a cooperative person completes', () {
      final manager = CalibrationManager();
      final session = manager.ensure(1, 'someone', PersonBaseline());
      expect(session, isNotNull);

      for (var i = 0; i < 2000; i++) {
        if (session!.complete) break;
        final step = session.step;
        final yaw = step != null && step.key == 'turn' ? 40.0 : 0.0;
        session.offer(hasFace: true, quality: 0.8, yaw: yaw);
      }

      expect(session!.complete, isTrue);
      expect(session.stepIndex, calibrationSteps.length);
    });

    test('flicker does not prevent completion', () {
      final manager = CalibrationManager();
      final baseline = PersonBaseline();
      CalibrationSession? session;

      for (var i = 0; i < 4000; i++) {
        final personId = i % 9 == 0 ? null : (i % 23 == 0 ? 2 : 1);
        session = manager.ensure(personId, 'someone', baseline);
        if (session == null || session.complete) break;
        final step = session.step;
        final yaw = step != null && step.key == 'turn' ? 40.0 : 0.0;
        session.offer(hasFace: true, quality: 0.8, yaw: yaw);
      }

      expect(session, isNotNull);
      expect(session!.complete, isTrue,
          reason: 'a person who moves must still be able to finish');
    });
  });

  group('rejections do not reset either', () {
    final cases = <String, Map<String, Object>>{
      'no face': <String, Object>{'hasFace': false, 'quality': 0.9},
      'low quality': <String, Object>{'hasFace': true, 'quality': 0.1},
      'extreme angle': <String, Object>{
        'hasFace': true,
        'quality': 0.9,
        'yaw': 80.0,
      },
    };

    for (final entry in cases.entries) {
      test('a rejected frame keeps earlier progress: ${entry.key}', () {
        final session = CalibrationSession(personId: 1);
        for (var i = 0; i < 40; i++) {
          session.offer(hasFace: true, quality: 0.8);
        }
        final before = session.totalAccepted;

        session.offer(
          hasFace: entry.value['hasFace']! as bool,
          quality: entry.value['quality']! as double,
          yaw: (entry.value['yaw'] as double?) ?? 0.0,
        );

        expect(session.totalAccepted, before);
      });
    }
  });
}
