import 'package:engine_body/src/core/body_frame.dart';
import 'package:engine_body/src/core/body_landmark.dart';
import 'package:engine_body/src/core/framing.dart';
import 'package:engine_body/src/core/joint_angle.dart';
import 'package:engine_body/src/core/landmark_type.dart';
import 'package:engine_body/src/core/rep_counter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  BodyFrame frameWith(
    Map<LandmarkType, (double, double)> positions, {
    Map<LandmarkType, double> likelihoods = const {},
  }) {
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

  group('JointAngle', () {
    test('a straight limb is 180 degrees', () {
      final frame = frameWith({
        LandmarkType.leftHip: (0.5, 0.4),
        LandmarkType.leftKnee: (0.5, 0.6),
        LandmarkType.leftAnkle: (0.5, 0.8),
      });
      final angle = frame.angleAt(
        LandmarkType.leftHip,
        LandmarkType.leftKnee,
        LandmarkType.leftAnkle,
      );
      expect(angle, isNotNull);
      expect(angle!.degrees, closeTo(180, 0.5));
    });

    test('a right angle measures 90 degrees', () {
      final frame = frameWith({
        LandmarkType.leftHip: (0.5, 0.4),
        LandmarkType.leftKnee: (0.5, 0.6),
        LandmarkType.leftAnkle: (0.7, 0.6),
      });
      final angle = frame.angleAt(
        LandmarkType.leftHip,
        LandmarkType.leftKnee,
        LandmarkType.leftAnkle,
      );
      expect(angle!.degrees, closeTo(90, 0.5));
    });

    test('confidence is the weakest landmark, not the mean', () {
      final frame = frameWith(
        {
          LandmarkType.leftHip: (0.5, 0.4),
          LandmarkType.leftKnee: (0.5, 0.6),
          LandmarkType.leftAnkle: (0.5, 0.8),
        },
        likelihoods: {LandmarkType.leftAnkle: 0.2},
      );
      final angle = frame.angleAt(
        LandmarkType.leftHip,
        LandmarkType.leftKnee,
        LandmarkType.leftAnkle,
      )!;

      expect(angle.confidence, closeTo(0.2, 1e-9));
      expect(angle.isReliable, isFalse);
      expect(
        angle.orNull,
        isNull,
        reason: 'an unreliable angle must not be usable as a number',
      );
    });

    test('a missing landmark gives null, not a guess', () {
      final frame = frameWith({
        LandmarkType.leftHip: (0.5, 0.4),
        LandmarkType.leftKnee: (0.5, 0.6),
      });
      expect(
        frame.angleAt(
          LandmarkType.leftHip,
          LandmarkType.leftKnee,
          LandmarkType.leftAnkle,
        ),
        isNull,
      );
    });

    test('coincident points give null rather than a meaningless angle', () {
      final frame = frameWith({
        LandmarkType.leftHip: (0.5, 0.6),
        LandmarkType.leftKnee: (0.5, 0.6),
        LandmarkType.leftAnkle: (0.5, 0.8),
      });
      expect(
        frame.angleAt(
          LandmarkType.leftHip,
          LandmarkType.leftKnee,
          LandmarkType.leftAnkle,
        ),
        isNull,
      );
    });

    test('unreliable angles reach the counter as null', () {
      final counter = RepCounter(squatConfig);
      for (var i = 0; i < 60; i++) {
        final occluded = i % 3 == 0;
        final frame = frameWith(
          {
            LandmarkType.leftHip: (0.5, 0.4),
            LandmarkType.leftKnee: (0.5, 0.6),
            LandmarkType.leftAnkle: (0.5, 0.8),
          },
          likelihoods: occluded ? {LandmarkType.leftKnee: 0.1} : const {},
        );
        final angle = frame.angleAt(
          LandmarkType.leftHip,
          LandmarkType.leftKnee,
          LandmarkType.leftAnkle,
        );
        counter.update(
          angle?.orNull,
          Duration(microseconds: (i * 1000000 / 30).round()),
        );
      }
      expect(counter.count, 0);
      expect(counter.state, RepState.up);
    });
  });

  group('allAngles', () {
    test('omits joints whose landmarks are missing', () {
      final frame = frameWith({
        LandmarkType.leftHip: (0.5, 0.4),
        LandmarkType.leftKnee: (0.5, 0.6),
        LandmarkType.leftAnkle: (0.5, 0.8),
      });
      final angles = frame.allAngles;
      expect(angles.keys, contains('leftKnee'));
      expect(angles.keys, isNot(contains('rightKnee')));
    });

    test('ankles are not tracked', () {
      expect(trackedJoints.keys, isNot(contains('leftAnkle')));
    });
  });

  group('SideTracker', () {
    Map<String, JointAngle> sides(double leftConf, double rightConf) => {
      'leftKnee': JointAngle(degrees: 120, confidence: leftConf),
      'rightKnee': JointAngle(degrees: 128, confidence: rightConf),
    };

    test('holds its side through a brief confidence dip', () {
      final tracker = SideTracker();
      for (var i = 0; i < 10; i++) {
        tracker.update(sides(0.9, 0.8), 'Knee');
      }
      final chosen = tracker.side;
      expect(chosen, 'left');

      for (var i = 0; i < 3; i++) {
        tracker.update(sides(0.3, 0.9), 'Knee');
      }
      expect(
        tracker.side,
        chosen,
        reason: 'a 3-frame dip is shorter than patience',
      );
    });

    test('switches when the other side is persistently better', () {
      final tracker = SideTracker();
      for (var i = 0; i < 10; i++) {
        tracker.update(sides(0.9, 0.8), 'Knee');
      }
      for (var i = 0; i < 10; i++) {
        tracker.update(sides(0.2, 0.9), 'Knee');
      }
      expect(tracker.side, 'right');
    });

    test('uses whichever side is available when only one is', () {
      final tracker = SideTracker();
      final result = tracker.update({
        'rightKnee': const JointAngle(degrees: 130, confidence: 0.9),
      }, 'Knee');
      expect(result, isNotNull);
      expect(tracker.side, 'right');
    });

    test('returns null when neither side is available', () {
      final tracker = SideTracker();
      expect(tracker.update(const {}, 'Knee'), isNull);
    });
  });

  group('jointSymmetry', () {
    test('reports the left-right difference', () {
      final asymmetry = jointSymmetry({
        'leftKnee': const JointAngle(degrees: 120, confidence: 0.9),
        'rightKnee': const JointAngle(degrees: 150, confidence: 0.9),
      }, 'Knee');
      expect(asymmetry, closeTo(30, 1e-9));
    });

    test('refuses when either side is unreliable', () {
      expect(
        jointSymmetry({
          'leftKnee': const JointAngle(degrees: 120, confidence: 0.2),
          'rightKnee': const JointAngle(degrees: 150, confidence: 0.9),
        }, 'Knee'),
        isNull,
      );
    });
  });
}
