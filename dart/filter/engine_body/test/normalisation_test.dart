import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/pose_fixtures.dart';

void main() {
  final at = DateTime(2026, 1, 1);

  group('TorsoFrame', () {
    test('normalised coordinates do not change with distance from camera', () {
      final near = PoseFixtures.frame(
        PoseFixtures.standing(scale: 1.4),
        at: at,
      );
      final mid = PoseFixtures.frame(PoseFixtures.standing(), at: at);
      final far = PoseFixtures.frame(
        PoseFixtures.standing(scale: 0.5),
        at: at,
      );

      final wrists = [near, mid, far]
          .map((f) => f.normalised![LandmarkType.leftWrist]!)
          .toList();

      for (final wrist in wrists.skip(1)) {
        expect(wrist.x, closeTo(wrists.first.x, 1e-9));
        expect(wrist.y, closeTo(wrists.first.y, 1e-9));
      }
    });

    test('normalised coordinates do not change with position in frame', () {
      final centre = PoseFixtures.frame(PoseFixtures.standing(), at: at);
      final offset = PoseFixtures.frame(
        PoseFixtures.standing(centreX: 0.25, centreY: 0.4),
        at: at,
      );

      final a = centre.normalised![LandmarkType.leftWrist]!;
      final b = offset.normalised![LandmarkType.leftWrist]!;
      expect(b.x, closeTo(a.x, 1e-9));
      expect(b.y, closeTo(a.y, 1e-9));
    });

    test('origin is the mid-hip point and shoulders are one unit up', () {
      final frame = PoseFixtures.frame(PoseFixtures.standing(), at: at);
      final points = frame.normalised!;

      final midHipY =
          (points[LandmarkType.leftHip]!.y + points[LandmarkType.rightHip]!.y) /
          2;
      final midHipX =
          (points[LandmarkType.leftHip]!.x + points[LandmarkType.rightHip]!.x) /
          2;
      expect(midHipX, closeTo(0, 1e-9));
      expect(midHipY, closeTo(0, 1e-9));

      final midShoulderY =
          (points[LandmarkType.leftShoulder]!.y +
              points[LandmarkType.rightShoulder]!.y) /
          2;
      expect(midShoulderY, closeTo(1.0, 1e-9));
    });

    test('y is positive upwards, unlike image coordinates', () {
      final frame = PoseFixtures.frame(PoseFixtures.standing(), at: at);
      final points = frame.normalised!;
      expect(points[LandmarkType.nose]!.y, greaterThan(0));
      expect(points[LandmarkType.leftAnkle]!.y, lessThan(0));
    });

    test('refuses to build a frame from an unreliable torso', () {
      final landmarks = PoseFixtures.standing(confidence: 0.2);
      final frame = PoseFixtures.frame(landmarks, at: at);
      expect(frame.torso, isNull);
      expect(frame.normalised, isNull);
    });

    test('refuses a collapsed torso instead of dividing by a tiny scale', () {
      final landmarks = PoseFixtures.standing(scale: 0.01);
      final frame = PoseFixtures.frame(landmarks, at: at);
      expect(frame.torso, isNull);
    });

    test('reports trunk lean in degrees', () {
      final upright = PoseFixtures.frame(PoseFixtures.standing(), at: at);
      expect(upright.torso!.tiltDegrees, closeTo(0, 0.5));
    });
  });

  group('joint angles', () {
    test('an arm held straight out reads ~180 at the elbow', () {
      final frame = PoseFixtures.frame(PoseFixtures.leftArmRaised(), at: at);
      expect(frame.leftElbowAngle, closeTo(180, 1.0));
    });

    test('an arm raised sideways opens the shoulder past 90', () {
      final frame = PoseFixtures.frame(PoseFixtures.leftArmRaised(), at: at);
      final angle = frame.leftShoulderAngle!;
      expect(angle, greaterThan(90));
      expect(angle, lessThan(110));

      final lowered = PoseFixtures.frame(PoseFixtures.standing(), at: at);
      expect(angle, greaterThan(lowered.leftShoulderAngle! + 40));
    });

    test('angles are invariant to distance and position', () {
      final near = PoseFixtures.frame(
        PoseFixtures.leftArmRaised(scale: 1.5),
        at: at,
      );
      final far = PoseFixtures.frame(
        PoseFixtures.leftArmRaised(scale: 0.4, centreX: 0.2, centreY: 0.7),
        at: at,
      );
      expect(near.leftElbowAngle!, closeTo(far.leftElbowAngle!, 1e-6));
      expect(near.leftShoulderAngle!, closeTo(far.leftShoulderAngle!, 1e-6));
    });

    test('returns null rather than a number when a landmark is unreliable', () {
      final frame = PoseFixtures.frame(
        PoseFixtures.standing(legsVisible: false),
        at: at,
      );
      expect(frame.leftKneeAngle, isNull);
      expect(frame.leftElbowAngle, isNotNull);
    });
  });
}
