import 'dart:math' as math;

import 'package:engine_body/src/core/body_frame.dart';
import 'package:engine_body/src/core/body_landmark.dart';
import 'package:engine_body/src/core/framing.dart';
import 'package:engine_body/src/core/hold_detector.dart';
import 'package:engine_body/src/core/landmark_type.dart';
import 'package:engine_body/src/core/reliability.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame) =>
      Duration(microseconds: (frame * 1000000 / 30).round());

  BodyFrame poseAt(double bend, {double noise = 0, math.Random? random}) {
    const femur = 0.20;
    const tibia = 0.18;
    const torso = 0.20;
    const ankleY = 0.88;

    final kneeAngle = math.pi * (180 - bend * 105) / 180;
    final shinLean = bend * 0.45;
    final kneeY = ankleY - tibia * math.cos(shinLean);
    final kneeForward = tibia * math.sin(shinLean);
    final thighLean = math.pi - kneeAngle - shinLean;
    final hipY = kneeY - femur * math.cos(thighLean);
    final hipBack = kneeForward - femur * math.sin(thighLean);
    final shoulderY = hipY - torso;

    double jitter() =>
        noise == 0 ? 0 : ((random?.nextDouble() ?? 0.5) - 0.5) * 2 * noise;

    final positions = <LandmarkType, (double, double)>{
      LandmarkType.nose: (0.50, shoulderY - 0.13),
      LandmarkType.leftShoulder: (0.58, shoulderY),
      LandmarkType.rightShoulder: (0.42, shoulderY),
      LandmarkType.leftElbow: (0.63, shoulderY + 0.10),
      LandmarkType.rightElbow: (0.37, shoulderY + 0.10),
      LandmarkType.leftWrist: (0.66, shoulderY + 0.20),
      LandmarkType.rightWrist: (0.34, shoulderY + 0.20),
      LandmarkType.leftHip: (0.555 + hipBack, hipY),
      LandmarkType.rightHip: (0.445 + hipBack, hipY),
      LandmarkType.leftKnee: (0.555 + kneeForward, kneeY),
      LandmarkType.rightKnee: (0.445 + kneeForward, kneeY),
      LandmarkType.leftAnkle: (0.555, ankleY),
      LandmarkType.rightAnkle: (0.445, ankleY),
    };

    return BodyFrame(
      timestamp: DateTime.fromMillisecondsSinceEpoch(0),
      landmarks: [
        for (final entry in positions.entries)
          BodyLandmark(
            type: entry.key,
            x: entry.value.$1 + jitter(),
            y: entry.value.$2 + jitter(),
            z: 0,
            likelihood: 1,
            inFrameLikelihood: 1,
          ),
      ],
      framing: FramingReport.noSubject,
      isMirrored: false,
    );
  }

  List<HoldEvent> drive(
    double Function(int frame) bendOf, {
    double seconds = 8,
    double noise = 0,
    int seed = 1,
    HoldDetector? detector,
  }) {
    final random = math.Random(seed);
    final d = detector ?? HoldDetector();
    final events = <HoldEvent>[];
    final frames = (seconds * 30).round();

    for (var i = 0; i < frames; i++) {
      final frame = poseAt(bendOf(i), noise: noise, random: random);
      final state = d.update(frame, torsoLength(frame), at(i));
      if (state.completed != null) events.add(state.completed!);
    }
    final last = d.finish(at(frames));
    if (last != null) events.add(last);
    return events;
  }

  double squatting(int i) =>
      (math.sin(2 * math.pi * i / 40) + 1) / 2 * 0.9;

  group('HoldDetector', () {
    test('a still position is detected as a hold', () {
      final events = drive((_) => 0.4);
      expect(events, hasLength(1));
      expect(
        events.first.duration.inMilliseconds,
        closeTo(8000, 600),
      );
    });

    test('continuous movement is not a hold', () {
      expect(drive(squatting), isEmpty);
    });

    test('a small wobble does not break a hold', () {
      final events = drive(
        (i) => 0.4 + 0.004 * math.sin(2 * math.pi * i / 25),
      );
      expect(events, hasLength(1));
      expect(events.first.duration.inSeconds, greaterThanOrEqualTo(7));
    });

    test('a hold ends when movement starts', () {
      final events = drive((i) => i < 150 ? 0.4 : squatting(i));
      expect(events, hasLength(1));
      expect(events.first.duration.inMilliseconds, inInclusiveRange(4000, 5600));
    });

    test('a brief pause is not a hold', () {
      final events = drive((i) => i < 45 ? 0.4 : squatting(i));
      expect(events, isEmpty);
    });

    test('works across camera noise levels', () {
      for (final noise in [0.0, 0.005, 0.01, 0.02]) {
        final held = drive((_) => 0.4, noise: noise);
        expect(
          held,
          hasLength(1),
          reason: 'a still person was not held at noise $noise',
        );

        final moving = drive(squatting, noise: noise);
        expect(
          moving,
          isEmpty,
          reason: 'squatting was held at noise $noise',
        );
      }
    });

    test('steadiness distinguishes calm from trembling', () {
      final calm = drive((_) => 0.4, noise: 0.002);
      final shaky = drive((_) => 0.4, noise: 0.02);
      expect(calm, isNotEmpty);
      expect(shaky, isNotEmpty);
      expect(calm.first.steadiness, greaterThan(shaky.first.steadiness));
    });

    test('lost tracking does not abandon a hold', () {
      final detector = HoldDetector();
      for (var i = 0; i < 120; i++) {
        final frame = poseAt(0.4);
        if (i % 40 == 0 && i > 0) {
          final state = detector.update(frame, null, at(i));
          expect(state.holding, isTrue);
        } else {
          detector.update(frame, torsoLength(frame), at(i));
        }
      }
    });

    test('the mean angle is reported', () {
      final events = drive((_) => 0.4);
      expect(events.first.meanAngle, isNull,
          reason: 'no angle was supplied, so none should be invented');

      final detector = HoldDetector();
      for (var i = 0; i < 240; i++) {
        final frame = poseAt(0.4);
        detector.update(frame, torsoLength(frame), at(i), angle: 138);
      }
      final held = detector.finish(at(240));
      expect(held!.meanAngle, closeTo(138, 0.1));
    });

    test('total accumulates across holds', () {
      final detector = HoldDetector();
      drive(
        (i) {
          if (i < 120) return 0.4;
          if (i < 180) return squatting(i);
          return 0.4;
        },
        seconds: 12,
        detector: detector,
      );
      expect(detector.holds.length, 2);
      expect(detector.totalHeld.inSeconds, greaterThan(6));
    });
  });

  group('exercise presets', () {
    test('every preset is coherent', () {
      for (final entry in exercises.entries) {
        expect(entry.value.description, isNotEmpty,
            reason: '${entry.key} has no camera guidance, which users need');
        if (entry.value.kind == ExerciseKind.reps) {
          final config = entry.value.config();
          expect(config.deadZone, greaterThanOrEqualTo(20),
              reason: '${entry.key} has too narrow a dead zone');
        }
      }
    });

    test('presets only track reliably visible joints', () {
      const allowed = {'Knee', 'Hip', 'Elbow', 'Shoulder'};
      for (final entry in exercises.entries) {
        expect(allowed, contains(entry.value.joint),
            reason: '${entry.key} tracks an unreliable joint');
      }
    });
  });
}
