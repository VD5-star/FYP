import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/pose_fixtures.dart';

void main() {
  final t0 = DateTime(2026, 1, 1);
  DateTime at(int i) => t0.add(Duration(milliseconds: 66 * i));

  group('PoseSmoother', () {
    test('passes the first frame through unchanged', () {
      final smoother = PoseSmoother();
      final frame = PoseFixtures.frame(PoseFixtures.standing(), at: at(0));
      final out = smoother.smooth(frame);

      for (var i = 0; i < frame.landmarks.length; i++) {
        expect(out.landmarks[i].x, frame.landmarks[i].x);
        expect(out.landmarks[i].y, frame.landmarks[i].y);
      }
    });

    test('reduces jitter on a motionless subject', () {
      final smoother = PoseSmoother();
      const truth = 0.5;
      var rawError = 0.0;
      var smoothedError = 0.0;

      for (var i = 0; i < 40; i++) {
        final jitter = (i.isEven ? 0.012 : -0.012);
        final landmarks = [
          for (final l in PoseFixtures.standing())
            if (l.type == LandmarkType.leftWrist)
              l.copyWith(x: truth + jitter)
            else
              l,
        ];

        final out = smoother.smooth(
          PoseFixtures.frame(landmarks, at: at(i)),
        );
        final wrist = out.landmarks.firstWhere(
          (l) => l.type == LandmarkType.leftWrist,
        );

        if (i > 8) {
          rawError += jitter.abs();
          smoothedError += (wrist.x - truth).abs();
        }
      }

      expect(smoothedError, lessThan(rawError * 0.7));
    });

    test('still follows a real movement rather than freezing it', () {
      final smoother = PoseSmoother();
      const start = 0.30;
      const end = 0.70;

      double? last;
      for (var i = 0; i < 25; i++) {
        final x = start + (end - start) * (i / 24);
        final landmarks = [
          for (final l in PoseFixtures.standing())
            if (l.type == LandmarkType.leftWrist) l.copyWith(x: x) else l,
        ];
        final out = smoother.smooth(PoseFixtures.frame(landmarks, at: at(i)));
        last = out.landmarks
            .firstWhere((l) => l.type == LandmarkType.leftWrist)
            .x;
      }

      expect(last, greaterThan(0.62));
      expect(last, lessThanOrEqualTo(end + 1e-9));
    });

    test('does not smooth across a tracking gap after reset', () {
      final smoother = PoseSmoother();
      smoother.smooth(PoseFixtures.frame(PoseFixtures.standing(), at: at(0)));
      smoother.smooth(PoseFixtures.frame(PoseFixtures.standing(), at: at(1)));
      smoother.reset();

      final moved = [
        for (final l in PoseFixtures.standing())
          if (l.type == LandmarkType.leftWrist) l.copyWith(x: 0.9) else l,
      ];
      final out = smoother.smooth(PoseFixtures.frame(moved, at: at(40)));
      expect(
        out.landmarks.firstWhere((l) => l.type == LandmarkType.leftWrist).x,
        0.9,
      );
    });

    test('leaves low-confidence landmarks untouched', () {
      final smoother = PoseSmoother(confidenceThreshold: 0.5);
      for (var i = 0; i < 5; i++) {
        final frame = PoseFixtures.frame(
          PoseFixtures.standing(legsVisible: false),
          at: at(i),
        );
        final out = smoother.smooth(frame);
        final ankle = out.landmarks.firstWhere(
          (l) => l.type == LandmarkType.leftAnkle,
        );
        final source = frame.landmarks.firstWhere(
          (l) => l.type == LandmarkType.leftAnkle,
        );
        expect(ankle.x, source.x);
      }
    });

    test('preserves confidence, framing and mirroring', () {
      final smoother = PoseSmoother();
      final frame = PoseFixtures.frame(
        PoseFixtures.standing(legsVisible: false),
        at: at(0),
      );
      final out = smoother.smooth(frame);

      expect(out.framing.framing, frame.framing.framing);
      expect(out.isMirrored, frame.isMirrored);
      expect(out.landmarks.length, frame.landmarks.length);
      for (var i = 0; i < frame.landmarks.length; i++) {
        expect(
          out.landmarks[i].inFrameLikelihood,
          frame.landmarks[i].inFrameLikelihood,
        );
      }
    });

    test('handles an empty frame without throwing', () {
      final smoother = PoseSmoother();
      expect(smoother.smooth(BodyFrame.empty).landmarks, isEmpty);
    });
  });

  group('PreviewInfo', () {
    test('a quarter-turned sensor swaps the displayed axes', () {
      const upright = PreviewInfo(
        textureId: 1,
        width: 640,
        height: 480,
        rotation: 0,
        isMirrored: false,
      );
      const turned = PreviewInfo(
        textureId: 1,
        width: 640,
        height: 480,
        rotation: 90,
        isMirrored: false,
      );

      expect(upright.displayWidth, 640);
      expect(upright.displayHeight, 480);
      expect(upright.isQuarterTurned, isFalse);
      expect(upright.displayAspectRatio, closeTo(640 / 480, 1e-9));

      expect(turned.displayWidth, 480);
      expect(turned.displayHeight, 640);
      expect(turned.isQuarterTurned, isTrue);
      expect(turned.displayAspectRatio, closeTo(480 / 640, 1e-9));
    });

    test('180 degrees is upside down, not turned on its side', () {
      const info = PreviewInfo(
        textureId: 1,
        width: 640,
        height: 480,
        rotation: 180,
        isMirrored: false,
      );
      expect(info.isQuarterTurned, isFalse);
      expect(info.displayWidth, 640);
      expect(info.displayHeight, 480);
    });

    test('270 degrees swaps the axes like 90 does', () {
      const info = PreviewInfo(
        textureId: 1,
        width: 1280,
        height: 960,
        rotation: 270,
        isMirrored: true,
      );
      expect(info.isQuarterTurned, isTrue);
      expect(info.displayWidth, 960);
      expect(info.displayHeight, 1280);
    });

    test('decodes what the native layer reports', () {
      final info = PreviewInfo.fromMap({
        'textureId': 7,
        'previewWidth': 1280,
        'previewHeight': 720,
        'rotation': 270,
        'isMirrored': true,
      });

      expect(info, isNotNull);
      expect(info!.textureId, 7);
      expect(info.rotation, 270);
      expect(info.isMirrored, isTrue);
    });

    test('returns null rather than a fake preview when there is none', () {
      expect(PreviewInfo.fromMap(null), isNull);
      expect(PreviewInfo.fromMap({'previewWidth': 640}), isNull);
    });
  });
}
