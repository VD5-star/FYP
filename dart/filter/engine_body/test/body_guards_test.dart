import 'dart:math' as math;

import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame, {int fps = 30}) =>
      Duration(microseconds: (frame * 1000000 / fps).round());

  BodyFrame poseAt(
    double bend, {
    double riseMetres = 0,
    double scale = 1,
    double z = 0,
    Map<LandmarkType, double> likelihoods = const {},
    Map<LandmarkType, (double, double)> overrides = const {},
  }) {
    const femur = 0.20;
    const tibia = 0.18;
    const torso = 0.20;
    const ankleY = 0.88;
    const pixelsPerMetre = 0.4;

    final kneeAngle = math.pi * (180 - bend * 105) / 180;
    final shinLean = bend * 0.45;
    final kneeY = ankleY - tibia * math.cos(shinLean);
    final kneeForward = tibia * math.sin(shinLean);
    final thighLean = math.pi - kneeAngle - shinLean;
    final hipY = kneeY - femur * math.cos(thighLean);
    final hipBack = kneeForward - femur * math.sin(thighLean);
    final shoulderY = hipY - torso;
    final lift = riseMetres * pixelsPerMetre;

    final base = <LandmarkType, (double, double)>{
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
      LandmarkType.leftFootIndex: (0.575, ankleY + 0.03),
      LandmarkType.rightFootIndex: (0.425, ankleY + 0.03),
    };

    final positions = <LandmarkType, (double, double)>{};
    base.forEach((type, value) {
      final scaled = (
        0.5 + (value.$1 - 0.5) * scale,
        0.5 + (value.$2 - 0.5) * scale - lift,
      );
      positions[type] = overrides[type] ?? scaled;
    });

    return BodyFrame(
      timestamp: DateTime.fromMillisecondsSinceEpoch(0),
      landmarks: [
        for (final entry in positions.entries)
          BodyLandmark(
            type: entry.key,
            x: entry.value.$1,
            y: entry.value.$2,
            z: entry.key == LandmarkType.leftShoulder
                ? z
                : entry.key == LandmarkType.rightShoulder
                    ? -z
                    : 0,
            likelihood: likelihoods[entry.key] ?? 1.0,
            inFrameLikelihood: likelihoods[entry.key] ?? 1.0,
          ),
      ],
      framing: FramingReport.noSubject,
      isMirrored: false,
    );
  }

  group('centreOfMass', () {
    test('sits inside the trunk', () {
      final frame = poseAt(0);
      final com = centreOfMass(frame);
      expect(com, isNotNull);

      final shoulderY = frame[LandmarkType.leftShoulder]!.y;
      final kneeY = frame[LandmarkType.leftKnee]!.y;
      expect(com!.y, greaterThan(shoulderY));
      expect(com.y, lessThan(kneeY));
      expect(com.x, closeTo(0.5, 0.05),
          reason: 'a symmetric pose should give a centred mass');
    });

    test('refuses when too much of the body is unseen', () {
      final frame = poseAt(0, likelihoods: {
        for (final type in [
          LandmarkType.leftHip,
          LandmarkType.rightHip,
          LandmarkType.leftKnee,
          LandmarkType.rightKnee,
          LandmarkType.leftShoulder,
          LandmarkType.rightShoulder,
        ])
          type: 0.1,
      });
      expect(centreOfMass(frame), isNull);
    });

    test('is steadier than any single landmark', () {
      final random = math.Random(42);
      final comX = <double>[];
      final wristX = <double>[];

      for (var i = 0; i < 300; i++) {
        final base = poseAt(0);
        final jittered = BodyFrame(
          timestamp: base.timestamp,
          landmarks: [
            for (final l in base.landmarks)
              BodyLandmark(
                type: l.type,
                x: l.x + (random.nextDouble() - 0.5) * 0.02,
                y: l.y + (random.nextDouble() - 0.5) * 0.02,
                z: l.z,
                likelihood: l.likelihood,
                inFrameLikelihood: l.inFrameLikelihood,
              ),
          ],
          framing: base.framing,
          isMirrored: false,
        );
        final com = centreOfMass(jittered);
        if (com != null) {
          comX.add(com.x);
          wristX.add(jittered[LandmarkType.leftWrist]!.x);
        }
      }

      expect(_stdDev(comX), lessThan(_stdDev(wristX) * 0.6));
    });
  });

  group('viewpoint', () {
    test('3D yaw reads square, angled and profile correctly', () {
      expect(torsoYawDegrees(poseAt(0, z: 0.0001))!, lessThan(5));
      expect(torsoYawDegrees(poseAt(0, z: 0.08))!, inInclusiveRange(35, 55));
      expect(torsoYawDegrees(poseAt(0, z: 10))!, greaterThan(80));
    });

    test('falls back to asymmetry when depth is unavailable', () {
      final frame = poseAt(0);
      expect(torsoYawDegrees(frame), isNull,
          reason: 'all-zero depth is absent data, not a square subject');
      expect(viewpointYawDegrees(frame), isNotNull,
          reason: 'the fallback must still produce an estimate');
    });

    test('a front-facing subject is accepted', () {
      expect(viewpointYawDegrees(poseAt(0))!, lessThan(35));
    });

    test('shoulder ratio collapses in profile', () {
      final front = shoulderTorsoRatio(poseAt(0))!;
      final turned = shoulderTorsoRatio(poseAt(0, overrides: {
        LandmarkType.leftShoulder: (0.51, 0.28),
        LandmarkType.rightShoulder: (0.49, 0.28),
      }))!;
      expect(turned, lessThan(front * 0.5));
    });
  });

  group('SubjectSwitchDetector', () {
    test('a real squat does not trigger a switch', () {
      final detector = SubjectSwitchDetector();
      var switches = 0;
      for (var i = 0; i < 72; i++) {
        final phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2;
        if (detector(poseAt(0.95 * phase), at(i))) switches++;
      }
      expect(switches, 0);
    });

    test('an abrupt size change is detected', () {
      final detector = SubjectSwitchDetector();
      for (var i = 0; i < 10; i++) {
        expect(detector(poseAt(0), at(i)), isFalse);
      }
      expect(detector(poseAt(0, scale: 1.6), at(10)), isTrue);
    });

    test('a gap in the stream is not a switch', () {
      final detector = SubjectSwitchDetector();
      for (var i = 0; i < 10; i++) {
        detector(poseAt(0), at(i));
      }
      expect(
        detector(poseAt(0, scale: 1.6), at(10) + const Duration(seconds: 1)),
        isFalse,
      );
    });

    test('an explosive jump is not a switch', () {
      final detector = SubjectSwitchDetector();
      const fps = 30;
      const g = 9.81;

      List<double> ease(double from, double to, int frames) => [
            for (var i = 1; i <= frames; i++)
              from + (to - from) * (1 - math.cos(math.pi * i / frames)) / 2,
          ];

      final sequence = <(double, double)>[
        for (var i = 0; i < 8; i++) (0.0, 0.0),
        for (final b in ease(0, 0.55, 10)) (b, 0.0),
        for (final b in ease(0.55, 0, 4)) (b, 0.0),
      ];
      final takeOff = math.sqrt(2 * g * 0.40);
      final flightFrames = (2 * takeOff / g * fps).round();
      for (var i = 1; i <= flightFrames; i++) {
        final s = i / fps;
        sequence.add((0.0, math.max(0, takeOff * s - 0.5 * g * s * s)));
      }
      sequence
        ..addAll([for (final b in ease(0, 0.38, 3)) (b, 0.0)])
        ..addAll([for (final b in ease(0.38, 0, 10)) (b, 0.0)])
        ..addAll([for (var i = 0; i < 8; i++) (0.0, 0.0)]);

      final fired = <int>[];
      for (var i = 0; i < sequence.length; i++) {
        final (bend, airborne) = sequence[i];
        if (detector(poseAt(bend, riseMetres: airborne), at(i))) fired.add(i);
      }
      expect(fired, isEmpty,
          reason: 'an explosive 40 cm jump was reported as a subject switch');
    });

    test('a modest build change is still detected', () {
      for (final scale in <double>[0.85, 1.15]) {
        final detector = SubjectSwitchDetector();
        for (var i = 0; i < 10; i++) {
          detector(poseAt(0), at(i));
        }
        expect(detector(poseAt(0, scale: scale), at(10)), isTrue,
            reason: 'a ${((scale - 1).abs() * 100).round()}% build change '
                'was missed');
      }
    });
  });

  group('LandmarkMedianFilter', () {
    test('rejects a spike', () {
      final filter = LandmarkMedianFilter();
      final clean = poseAt(0);
      Map<LandmarkType, ({double x, double y})>? out;

      for (var i = 0; i < 12; i++) {
        final frame = i == 8
            ? poseAt(0, overrides: {LandmarkType.leftWrist: (0.95, 0.95)})
            : clean;
        out = filter(frame, at(i));
      }

      expect(
        out![LandmarkType.leftWrist]!.x,
        closeTo(clean[LandmarkType.leftWrist]!.x, 0.01),
      );
    });

    test('resets across a gap', () {
      final filter = LandmarkMedianFilter();
      for (var i = 0; i < 10; i++) {
        filter(poseAt(0), at(i));
      }
      final moved = poseAt(0.9);
      final out = filter(moved, at(10) + const Duration(milliseconds: 500));
      expect(
        out[LandmarkType.leftKnee]!.x,
        closeTo(moved[LandmarkType.leftKnee]!.x, 1e-9),
        reason: 'state from before a long gap must not blend in',
      );
    });
  });

  group('JumpDetector', () {
    JumpEvent? simulate({
      required double peakMetres,
      bool tiptoe = false,
      int fps = 60,
    }) {
      final detector = JumpDetector();
      JumpEvent? event;
      var frame = 0;

      void feed(double rise) {
        final result = detector.update(
          poseAt(0, riseMetres: rise),
          at(frame, fps: fps),
        );
        event ??= result;
        frame++;
      }

      for (var i = 0; i < fps; i++) {
        feed(0);
      }

      if (tiptoe) {
        final n = (0.5 * fps).round();
        for (var i = 0; i < n; i++) {
          feed(peakMetres * math.sin(math.pi * i / n));
        }
      } else {
        final v0 = math.sqrt(2 * gravity * peakMetres);
        final flight = 2 * v0 / gravity;
        for (var i = 0; i <= (flight * fps).round(); i++) {
          final dt = i / fps;
          feed(math.max(0, v0 * dt - 0.5 * gravity * dt * dt));
        }
      }

      for (var i = 0; i < (0.5 * fps).round(); i++) {
        feed(0);
      }
      return event;
    }

    test('detects a real jump', () {
      final event = simulate(peakMetres: 0.30);
      expect(event, isNotNull);
      expect(event!.fittedGravityRatio, closeTo(1.0, 0.25),
          reason: 'free fall is genuinely parabolic');
    });

    test('rejects rising onto tiptoes', () {
      for (final peak in [0.05, 0.10, 0.15, 0.20]) {
        expect(
          simulate(peakMetres: peak, tiptoe: true),
          isNull,
          reason: 'a ${(peak * 100).round()}cm tiptoe rise was called a jump',
        );
      }
    });

    test('ignores postural sway', () {
      final detector = JumpDetector();
      final random = math.Random(7);
      for (var i = 0; i < 200; i++) {
        final event = detector.update(
          poseAt(0, riseMetres: (random.nextDouble() - 0.5) * 0.01),
          at(i, fps: 60),
        );
        expect(event, isNull);
      }
    });

    test('losing the subject aborts the jump', () {
      final detector = JumpDetector();
      for (var i = 0; i < 60; i++) {
        detector.update(poseAt(0), at(i, fps: 60));
      }
      for (var i = 0; i < 10; i++) {
        detector.update(poseAt(0, riseMetres: i * 0.03), at(60 + i, fps: 60));
      }
      for (var i = 0; i < 10; i++) {
        final blank = BodyFrame(
          timestamp: DateTime.fromMillisecondsSinceEpoch(0),
          landmarks: const [],
          framing: FramingReport.noSubject,
          isMirrored: false,
        );
        expect(detector.update(blank, at(70 + i, fps: 60)), isNull);
      }
    });
  });

  group('ExerciseSession integration', () {
    test('declines when the subject is turned away', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      final update = session.update(poseAt(0, z: 10), Duration.zero);
      expect(update.usable, isFalse);
      expect(update.reason, contains('face the camera'));
    });

    test('declines and holds counts when the subject changes', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      var frame = 0;
      for (var rep = 0; rep < 3; rep++) {
        for (var i = 0; i < 36; i++) {
          final phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2;
          session.update(poseAt(0.95 * phase), at(frame));
          frame++;
        }
      }
      final before = session.repCount;
      expect(before, greaterThan(0));

      final update = session.update(poseAt(0, scale: 1.6), at(frame));
      expect(update.subjectChanged, isTrue);
      expect(update.usable, isFalse);
      expect(update.repCount, before,
          reason: 'completed repetitions must survive a subject switch');
    });

    test('a normal set triggers neither guard', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      var refused = 0;
      var frame = 0;
      for (var rep = 0; rep < 5; rep++) {
        for (var i = 0; i < 36; i++) {
          final phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2;
          if (!session.update(poseAt(0.95 * phase), at(frame)).usable) {
            refused++;
          }
          frame++;
        }
      }
      expect(refused, 0);
      expect(session.repCount, 5);
    });
  });
}

double _stdDev(List<double> values) {
  if (values.length < 2) return 0;
  final mean = values.reduce((a, b) => a + b) / values.length;
  final variance = values
          .map((v) => (v - mean) * (v - mean))
          .reduce((a, b) => a + b) /
      values.length;
  return math.sqrt(variance);
}
