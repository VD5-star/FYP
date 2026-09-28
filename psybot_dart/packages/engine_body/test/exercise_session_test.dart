import 'dart:math' as math;

import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame) =>
      Duration(microseconds: (frame * 1000000 / 30).round());

  BodyFrame poseAt(double bend, {Map<LandmarkType, double> likelihoods = const {}}) {
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
            x: entry.value.$1,
            y: entry.value.$2,
            z: 0,
            likelihood: likelihoods[entry.key] ?? 1.0,
            inFrameLikelihood: likelihoods[entry.key] ?? 1.0,
          ),
      ],
      framing: FramingReport.noSubject,
      isMirrored: false,
    );
  }

  List<ExerciseUpdate> squat(
    ExerciseSession session,
    double depth, {
    int cycles = 5,
    int frames = 36,
    int startFrame = 0,
  }) {
    final updates = <ExerciseUpdate>[];
    var frame = startFrame;
    for (var c = 0; c < cycles; c++) {
      for (var i = 0; i < frames; i++) {
        final phase =
            (math.sin(2 * math.pi * i / frames - math.pi / 2) + 1) / 2;
        updates.add(session.update(poseAt(depth * phase), at(frame)));
        frame++;
      }
    }
    return updates;
  }

  group('ExerciseSession, counted exercises', () {
    test('counts a deep squatter with both counters', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      squat(session, 0.95);
      expect(session.repCount, 5);
      expect(session.depthCount, 5);
    });

    test('counts a limited user the depth threshold cannot see', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      squat(session, 0.70);

      expect(session.repCount, 5, reason: 'velocity counting should see these');
      expect(session.depthCount, 0,
          reason: 'the depth threshold should not, which is the point');
    });

    test('the gap between counts is the feedback', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      squat(session, 0.70);
      expect(session.repCount, greaterThan(session.depthCount));
    });

    test('standing still counts nothing', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      for (var i = 0; i < 300; i++) {
        session.update(poseAt(0), at(i));
      }
      expect(session.repCount, 0);
      expect(session.depthCount, 0);
    });

    test('an unreliable frame is refused without losing counts', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      squat(session, 0.95);
      final before = session.repCount;
      expect(before, 5);

      final jumped = BodyFrame(
        timestamp: DateTime.fromMillisecondsSinceEpoch(0),
        landmarks: [
          for (final l in poseAt(0).landmarks)
            BodyLandmark(
              type: l.type,
              x: l.x + 0.25,
              y: l.y,
              z: l.z,
              likelihood: l.likelihood,
              inFrameLikelihood: l.inFrameLikelihood,
            ),
        ],
        framing: FramingReport.noSubject,
        isMirrored: false,
      );

      final update = session.update(jumped, at(180));
      expect(update.usable, isFalse);
      expect(update.reason, isNotEmpty,
          reason: 'a refusal must say why, so the user can fix it');
      expect(update.repCount, before, reason: 'counts must survive');
    });

    test('form scoring flags the repetition that differs', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      final flagged = <int>[];
      var frame = 0;

      const depths = [0.95, 0.95, 0.95, 0.55, 0.95, 0.95];
      for (final depth in depths) {
        final updates = squat(session, depth, cycles: 1, startFrame: frame);
        frame += 36;
        for (final update in updates) {
          if (update.form != null && update.form!.notes.isNotEmpty) {
            flagged.add(session.repCount);
          }
        }
      }

      expect(flagged, isNotEmpty,
          reason: 'a much shallower repetition should draw a comment');
    });

    test('a calibrated config is used when supplied', () {
      final calibrated = HysteresisConfig(
        name: 'squat',
        downBelow: 120,
        upAbove: 160,
      );
      final session = ExerciseSession(
        exercise: exercises['squat']!,
        calibrated: calibrated,
      );
      squat(session, 0.70);
      expect(session.depthCount, greaterThan(0),
          reason: 'calibrated thresholds should now see this user');
    });

    test('reset clears in-progress state but can keep counts', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      squat(session, 0.95);
      expect(session.repCount, 5);

      session.reset(keepCounts: true);
      expect(session.repCount, 5);

      session.reset();
      expect(session.repCount, 0);
    });
  });

  group('ExerciseSession, timed exercises', () {
    test('times a held position', () {
      final session = ExerciseSession(exercise: exercises['plank']!);
      for (var i = 0; i < 240; i++) {
        session.update(poseAt(0.4), at(i));
      }
      final held = session.finish(at(240));
      expect(held, isNotNull);
      expect(held!.duration.inSeconds, greaterThanOrEqualTo(7));
    });

    test('does not count continuous movement as a hold', () {
      final session = ExerciseSession(exercise: exercises['plank']!);
      for (var i = 0; i < 240; i++) {
        final bend = (math.sin(2 * math.pi * i / 40) + 1) / 2 * 0.9;
        session.update(poseAt(bend), at(i));
      }
      session.finish(at(240));
      expect(session.holds, isEmpty);
    });

    test('reports elapsed time while holding', () {
      final session = ExerciseSession(exercise: exercises['stretch']!);
      ExerciseUpdate? last;
      for (var i = 0; i < 120; i++) {
        last = session.update(poseAt(0.4), at(i));
      }
      expect(last!.holding, isTrue);
      expect(last.elapsed.inSeconds, greaterThanOrEqualTo(3));
    });

    test('a timed exercise does not count repetitions', () {
      final session = ExerciseSession(exercise: exercises['plank']!);
      for (var i = 0; i < 120; i++) {
        session.update(poseAt(0.4), at(i));
      }
      expect(session.repCount, 0);
      expect(session.depthCount, 0);
    });
  });

  group('every preset runs without error', () {
    test('each exercise processes frames and counts nothing when still', () {
      for (final entry in exercises.entries) {
        final session = ExerciseSession(exercise: entry.value);
        for (var i = 0; i < 90; i++) {
          final update = session.update(poseAt(0.3), at(i));
          expect(update, isNotNull);
        }
        expect(session.repCount, 0,
            reason: '${entry.key} counted a repetition from a still pose');
      }
    });
  });
}
