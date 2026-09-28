import 'dart:math' as math;

import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  List<HandPoint> handPose(Set<Finger> extended) {
    final points = List<HandPoint>.filled(21, const HandPoint(0, 0));
    var i = 0;
    for (final finger in Finger.values) {
      final joints = fingerJoints[finger]!;
      final angle = (-140 + i * 22) * math.pi / 180.0;
      final dx = math.cos(angle);
      final dy = math.sin(angle);

      points[joints.base] = HandPoint(dx * 30, dy * 30);
      if (extended.contains(finger)) {
        points[joints.middle] = HandPoint(dx * 55, dy * 55);
        points[joints.tip] = HandPoint(dx * 80, dy * 80);
      } else {
        points[joints.middle] = HandPoint(dx * 45, dy * 45);
        points[joints.tip] = HandPoint(dx * 25, dy * 25);
      }
      i++;
    }
    return points;
  }

  List<HandPoint> rotated(List<HandPoint> points, double degrees) {
    final r = degrees * math.pi / 180.0;
    final c = math.cos(r);
    final s = math.sin(r);
    return [
      for (final p in points) HandPoint(p.x * c - p.y * s, p.x * s + p.y * c),
    ];
  }

  group('finger reading', () {
    test('reads an open hand', () {
      final states = readFingers(handPose(Finger.values.toSet()));
      expect(states.values.every((up) => up), isTrue);
      expect(fingerNumbersOf(states), [1, 2, 3, 4, 5]);
    });

    test('reads a closed hand', () {
      final states = readFingers(handPose(const {}));
      expect(states.values.any((up) => up), isFalse);
      expect(fingerNumbersOf(states), isEmpty);
    });

    test('reads the stop gesture', () {
      final states = readFingers(handPose({Finger.pinky, Finger.thumb}));
      expect(fingerNumbersOf(states), [1, 5]);
      expect(isStopGesture(states), isTrue);
    });

    test('similar gestures are not the stop gesture', () {
      const cases = <(Set<Finger>, String)>[
        ({Finger.pinky}, 'little finger alone'),
        ({Finger.thumb}, 'thumb alone'),
        (
          {Finger.pinky, Finger.thumb, Finger.indexFinger},
          'stop plus index',
        ),
        (<Finger>{}, 'fist'),
        ({Finger.indexFinger, Finger.middle}, 'victory'),
      ];

      for (final (extended, label) in cases) {
        expect(
          isStopGesture(readFingers(handPose(extended))),
          isFalse,
          reason: '$label was read as the stop gesture',
        );
      }

      expect(
        isStopGesture(readFingers(handPose(Finger.values.toSet()))),
        isFalse,
        reason: 'an open hand was read as the stop gesture',
      );
    });

    test('finger reading is rotation invariant', () {
      final pose = handPose({Finger.pinky, Finger.thumb});
      for (final degrees in [0.0, 45.0, 90.0, 180.0, 270.0]) {
        expect(
          isStopGesture(readFingers(rotated(pose, degrees))),
          isTrue,
          reason: 'lost at $degrees degrees',
        );
      }
    });

    test('a reading exposes numbers and the stop flag together', () {
      final reading = HandReading.fromPoints(
        handPose({Finger.pinky, Finger.thumb}),
        handedness: 'Left',
      );
      expect(reading, isNotNull);
      expect(reading!.handedness, 'Left');
      expect(reading.numbers, [1, 5]);
      expect(reading.isStop, isTrue);
    });

    test('a short landmark list is refused rather than guessed at', () {
      expect(HandReading.fromPoints(const [HandPoint(0, 0)]), isNull);
    });

    test('decodes what a native hand detector would report', () {
      final pose = handPose(Finger.values.toSet());
      final reading = HandReading.fromMap({
        'handedness': 'Right',
        'landmarks': [
          for (final p in pose) {'x': p.x, 'y': p.y},
        ],
      });

      expect(reading, isNotNull);
      expect(reading!.handedness, 'Right');
      expect(reading.numbers, [1, 2, 3, 4, 5]);
    });

    test('the bone list covers every finger', () {
      expect(handBones, hasLength(23));
      for (final joints in fingerJoints.values) {
        expect(joints.tip, lessThan(21));
        expect(joints.base, lessThan(21));
      }
    });
  });

  group('HeldGesture', () {
    Duration seconds(double s) =>
        Duration(microseconds: (s * 1000000).round());

    test('requires the full hold', () {
      final gesture = HeldGesture();
      expect(gesture.update(true, Duration.zero).fired, isFalse);
      for (final t in [0.3, 0.6, 0.9, 1.2]) {
        expect(
          gesture.update(true, seconds(t)).fired,
          isFalse,
          reason: 'fired early at $t s',
        );
      }
      expect(gesture.update(true, seconds(1.55)).fired, isTrue);
    });

    test('fires only once while held', () {
      final gesture = HeldGesture(hold: const Duration(seconds: 1));
      var fires = 0;
      for (final t in [0.0, 0.5, 1.1, 1.4, 2.0]) {
        if (gesture.update(true, seconds(t)).fired) fires++;
      }
      expect(fires, 1);
    });

    test('releasing the gesture resets it', () {
      final gesture = HeldGesture();
      gesture.update(true, Duration.zero);
      gesture.update(true, seconds(0.5));
      gesture.update(false, seconds(1.0));
      expect(gesture.update(false, seconds(1.6)).progress, 0.0);
    });

    test('a dropped sample does not reset the hold', () {
      final gesture = HeldGesture();
      gesture.update(true, Duration.zero);
      gesture.update(true, seconds(0.4));
      gesture.update(false, seconds(0.5));
      gesture.update(true, seconds(0.6));
      expect(gesture.update(true, seconds(1.55)).fired, isTrue);
    });

    test('progress is reported for the countdown', () {
      final gesture = HeldGesture(hold: const Duration(seconds: 1));
      gesture.update(true, Duration.zero);
      final state = gesture.update(true, seconds(0.5));
      expect(state.progress, greaterThan(0.4));
      expect(state.progress, lessThan(0.6));
    });

    test('reset clears a hold in progress', () {
      final gesture = HeldGesture();
      gesture.update(true, Duration.zero);
      gesture.update(true, seconds(1.0));
      gesture.reset();
      expect(gesture.update(true, seconds(1.1)).progress, 0.0);
    });
  });

  group('HandSampler', () {
    test('samples every Nth frame', () {
      final sampler = HandSampler(everyNFrames: 4);
      final due = [for (var i = 0; i < 12; i++) sampler.tick()];
      expect(due, [
        true, false, false, false,
        true, false, false, false,
        true, false, false, false,
      ]);
    });

    test('a zero interval still samples every frame', () {
      final sampler = HandSampler(everyNFrames: 0);
      expect([for (var i = 0; i < 4; i++) sampler.tick()],
          [true, true, true, true]);
    });
  });
}
