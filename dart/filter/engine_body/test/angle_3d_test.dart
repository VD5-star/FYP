import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame) =>
      Duration(microseconds: (frame * 1000000 / 30).round());

  BodyFrame worldFrame(Map<LandmarkType, (double, double, double)> points) =>
      BodyFrame(
        timestamp: DateTime.fromMillisecondsSinceEpoch(0),
        landmarks: [
          for (final entry in points.entries)
            BodyLandmark(
              type: entry.key,
              x: entry.value.$1,
              y: entry.value.$2,
              z: entry.value.$3,
              likelihood: 1,
              inFrameLikelihood: 1,
            ),
        ],
        framing: FramingReport.noSubject,
        isMirrored: false,
      );

  group('angle3d', () {
    test('a straight leg measures 180 degrees', () {
      final frame = worldFrame({
        LandmarkType.leftHip: (0, 0, 0),
        LandmarkType.leftKnee: (0, 0.4, 0),
        LandmarkType.leftAnkle: (0, 0.8, 0),
      });

      final straight = angle3d(
        frame,
        LandmarkType.leftHip,
        LandmarkType.leftKnee,
        LandmarkType.leftAnkle,
      );
      expect(straight, isNotNull);
      expect(straight!, closeTo(180, 1e-9));
      expect(angleDisagreement(180, straight)!, lessThan(1.0));
    });

    test('a right angle in depth is measured, not flattened', () {
      final frame = worldFrame({
        LandmarkType.leftHip: (0, 0, 0.4),
        LandmarkType.leftKnee: (0, 0, 0),
        LandmarkType.leftAnkle: (0, 0.4, 0),
      });

      expect(
        angle3d(
          frame,
          LandmarkType.leftHip,
          LandmarkType.leftKnee,
          LandmarkType.leftAnkle,
        )!,
        closeTo(90, 1e-9),
      );
    });

    test('a degenerate limb gives no angle rather than a wrong one', () {
      final frame = worldFrame({
        LandmarkType.leftHip: (0, 0, 0),
        LandmarkType.leftKnee: (0, 0, 0),
        LandmarkType.leftAnkle: (0, 0.4, 0),
      });

      expect(
        angle3d(
          frame,
          LandmarkType.leftHip,
          LandmarkType.leftKnee,
          LandmarkType.leftAnkle,
        ),
        isNull,
      );
    });

    test('a missing landmark gives no angle', () {
      final frame = worldFrame({
        LandmarkType.leftHip: (0, 0, 0),
        LandmarkType.leftKnee: (0, 0.4, 0),
      });

      expect(
        angle3d(
          frame,
          LandmarkType.leftHip,
          LandmarkType.leftKnee,
          LandmarkType.leftAnkle,
        ),
        isNull,
      );
    });
  });

  group('angleDisagreement', () {
    test('reports the gap between the two estimates', () {
      expect(angleDisagreement(175, 110), closeTo(65, 1e-9));
      expect(angleDisagreement(110, 175), closeTo(65, 1e-9));
    });

    test('is null when either estimate is missing', () {
      expect(angleDisagreement(null, 110), isNull);
      expect(angleDisagreement(175, null), isNull);
    });

    test('refuses non-finite input rather than propagating it', () {
      expect(angleDisagreement(double.nan, 110), isNull);
      expect(angleDisagreement(175, double.infinity), isNull);
    });
  });

  group('ReliabilityMonitor disagreement gate', () {
    test('flags a foreshortened limb', () {
      final monitor = ReliabilityMonitor();
      final frame = SyntheticPose.frame();

      final report = monitor.update(
        frame,
        torsoLength(frame),
        Duration.zero,
        angle2d: 175,
        angle3dValue: 110,
      );

      expect(report.trustworthy, isFalse);
      expect(report.reasons.join(), contains('disagree'));
      expect(report.disagreement, closeTo(65, 1e-9));
    });

    test('tolerates the published limits of agreement', () {
      final monitor = ReliabilityMonitor();
      final frame = SyntheticPose.frame();

      final report = monitor.update(
        frame,
        torsoLength(frame),
        Duration.zero,
        angle2d: 175,
        angle3dValue: 155,
      );

      expect(report.trustworthy, isTrue);
      expect(report.disagreement, closeTo(20, 1e-9));
    });

    test('the threshold sits at 30 degrees, as measured', () {
      expect(ReliabilityMonitor().maxDisagreement, 30.0);

      final monitor = ReliabilityMonitor();
      final frame = SyntheticPose.frame();
      final torso = torsoLength(frame);

      expect(
        monitor
            .update(frame, torso, at(0), angle2d: 180, angle3dValue: 150.5)
            .trustworthy,
        isTrue,
      );
      expect(
        monitor
            .update(frame, torso, at(1), angle2d: 180, angle3dValue: 149.5)
            .trustworthy,
        isFalse,
      );
    });

    test('says nothing when no 3D estimate is available', () {
      final monitor = ReliabilityMonitor();
      final frame = SyntheticPose.frame();

      final report =
          monitor.update(frame, torsoLength(frame), Duration.zero,
              angle2d: 175);

      expect(report.trustworthy, isTrue);
      expect(report.disagreement, isNull);
    });
  });
}
