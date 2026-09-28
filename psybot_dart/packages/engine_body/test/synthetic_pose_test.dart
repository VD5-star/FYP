import 'dart:math' as math;

import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame, {int fps = 30}) =>
      Duration(microseconds: (frame * 1000000 / fps).round());

  group('SyntheticPose geometry', () {
    test('knee angle matches the Python generator exactly', () {
      const expected = <(double, double)>[
        (0.00, 180.000),
        (0.08, 171.600),
        (0.25, 153.750),
        (0.40, 138.000),
        (0.55, 122.250),
        (0.70, 106.500),
        (0.95, 80.250),
      ];

      for (final (bend, degrees) in expected) {
        final frame = SyntheticPose.frame(kneeBend: bend);
        expect(
          SyntheticPose.kneeAngleOf(frame),
          closeTo(degrees, 1e-6),
          reason: 'bend $bend should give $degrees',
        );
      }
    });

    test('a straight leg is exactly 180 degrees', () {
      final frame = SyntheticPose.frame(kneeBend: 0);
      final angle = frame.angleAt(
        LandmarkType.leftHip,
        LandmarkType.leftKnee,
        LandmarkType.leftAnkle,
      );
      expect(angle, isNotNull);
      expect(angle!.degrees, closeTo(180, 1e-6));
    });

    test('a deep squat is acute', () {
      final frame = SyntheticPose.frame(kneeBend: 0.95);
      final angle = frame.angleAt(
        LandmarkType.leftHip,
        LandmarkType.leftKnee,
        LandmarkType.leftAnkle,
      );
      expect(angle!.degrees, greaterThan(60));
      expect(angle.degrees, lessThan(120));
    });

    test('shoulder angle tracks the arm raise, as in Python', () {
      const expected = <(double, double)>[
        (0.0, 7.125016),
        (0.5, 52.125016),
        (1.0, 97.125016),
      ];

      for (final (raise, degrees) in expected) {
        final angles = SyntheticPose.frame(armRaise: raise).allAngles;
        expect(
          angles['leftShoulder']!.degrees,
          closeTo(degrees, 1e-5),
          reason: 'raise $raise',
        );
      }
    });

    test('the elbow stays straight whatever the arm does', () {
      for (final raise in [0.0, 0.5, 1.0]) {
        final angles = SyntheticPose.frame(armRaise: raise).allAngles;
        expect(angles['leftElbow']!.degrees, closeTo(180, 1e-5));
      }
    });

    test('the hip angle matches Python at each depth', () {
      const expected = <(double, double)>[
        (0.00, 172.874984),
        (0.40, 122.247745),
        (0.95, 69.030249),
      ];

      for (final (bend, degrees) in expected) {
        final angles = SyntheticPose.frame(kneeBend: bend).allAngles;
        expect(angles['leftHip']!.degrees, closeTo(degrees, 1e-5));
      }
    });

    test('left and right knees are symmetric', () {
      for (final bend in [0.0, 0.4, 0.95]) {
        final angles = SyntheticPose.frame(kneeBend: bend).allAngles;
        expect(
          angles['leftKnee']!.degrees,
          closeTo(angles['rightKnee']!.degrees, 1e-9),
        );
      }
    });

    test('every landmark is produced', () {
      final frame = SyntheticPose.frame();
      expect(frame.landmarks, hasLength(LandmarkType.values.length));
      for (final type in LandmarkType.values) {
        expect(frame[type], isNotNull, reason: '$type missing');
      }
    });

    test('the torso is one fifth of the normalised frame', () {
      expect(torsoLength(SyntheticPose.frame()), closeTo(0.2, 1e-9));
    });

    test('a symmetric pose gives a centred centre of mass', () {
      final com = centreOfMass(SyntheticPose.frame(centreX: 0.5));
      expect(com, isNotNull);
      expect(com!.x, closeTo(0.5, 0.02));
    });

    test('the centre of mass matches Python once rescaled', () {
      final com = centreOfMass(SyntheticPose.frame(centreX: 0.36))!;
      expect(com.x * syntheticUnit, closeTo(360.0, 0.02 * syntheticUnit));
      expect(com.y * syntheticUnit, closeTo(502.017585, 1e-3));
    });

    test('the centre of mass sits inside the trunk', () {
      final frame = SyntheticPose.frame();
      final com = centreOfMass(frame)!;
      final shoulderY = (frame[LandmarkType.leftShoulder]!.y +
              frame[LandmarkType.rightShoulder]!.y) /
          2;
      expect(com.y, greaterThan(shoulderY));
      expect(com.y, lessThan(frame[LandmarkType.leftKnee]!.y));
    });

    test('the centre of mass refuses a mostly unseen body', () {
      final frame = SyntheticPose.frame(likelihoods: {
        LandmarkType.leftHip: 0.1,
        LandmarkType.rightHip: 0.1,
        LandmarkType.leftKnee: 0.1,
        LandmarkType.rightKnee: 0.1,
        LandmarkType.leftAnkle: 0.1,
        LandmarkType.rightAnkle: 0.1,
        LandmarkType.leftShoulder: 0.1,
        LandmarkType.rightShoulder: 0.1,
      });
      expect(centreOfMass(frame), isNull);
    });

    test('a front-facing pose reads as square to the camera', () {
      expect(viewpointYawDegrees(SyntheticPose.frame()), closeTo(0, 1e-9));
    });

    test('the shoulder to torso ratio matches Python', () {
      expect(shoulderTorsoRatio(SyntheticPose.frame()), closeTo(0.8, 1e-9));
    });

    test('scaling leaves every joint angle untouched', () {
      final reference = SyntheticPose.frame(kneeBend: 0.5).allAngles;
      for (final scale in [0.6, 1.6]) {
        final scaled =
            SyntheticPose.frame(kneeBend: 0.5, scale: scale).allAngles;
        for (final joint in reference.keys) {
          expect(
            scaled[joint]!.degrees,
            closeTo(reference[joint]!.degrees, 1e-9),
            reason: '$joint changed at scale $scale',
          );
        }
      }
    });
  });

  group('GaussianNoise', () {
    test('is centred near zero with unit spread', () {
      final noise = GaussianNoise(42);
      final samples = [for (var i = 0; i < 20000; i++) noise.next()];
      final mean = samples.reduce((a, b) => a + b) / samples.length;
      final variance = samples
              .map((v) => (v - mean) * (v - mean))
              .reduce((a, b) => a + b) /
          samples.length;

      expect(mean, closeTo(0, 0.05));
      expect(math.sqrt(variance), closeTo(1, 0.05));
    });

    test('the same seed replays the same sequence', () {
      final a = [for (var i = 0; i < 50; i++) GaussianNoise(7).next()];
      final b = [for (var i = 0; i < 50; i++) GaussianNoise(7).next()];
      expect(a, b);
    });
  });

  group('SyntheticPoseSource', () {
    test('a squat drives the velocity counter to the right count', () {
      for (final depth in [0.95, 0.70, 0.55]) {
        final source = SyntheticPoseSource(depth: depth);
        final counter = VelocityRepCounter();
        for (var i = 0; i < 5 * 36; i++) {
          final frame = source.next();
          counter.update(SyntheticPose.kneeAngleOf(frame), at(i));
        }
        expect(counter.count, 5, reason: 'depth $depth');
      }
    });

    test('a shallow movement counts nothing', () {
      final source = SyntheticPoseSource(depth: 0.08);
      final counter = VelocityRepCounter();
      for (var i = 0; i < 5 * 36; i++) {
        counter.update(SyntheticPose.kneeAngleOf(source.next()), at(i));
      }
      expect(counter.count, 0);
    });

    test('standing still produces no repetitions up to 6px of noise', () {
      for (final noise in [0.0, 3.0, 6.0]) {
        for (var seed = 0; seed < 20; seed++) {
          final source = SyntheticPoseSource(
            movement: SyntheticMovement.still,
            noise: noise,
            seed: seed,
          );
          final counter = VelocityRepCounter();
          for (var i = 0; i < 300; i++) {
            counter.update(SyntheticPose.kneeAngleOf(source.next()), at(i));
          }
          expect(counter.count, 0, reason: 'noise $noise seed $seed');
        }
      }
    });

    test('everyday movement counts nothing when tracking is clean', () {
      for (final movement in [
        SyntheticMovement.walkInPlace,
        SyntheticMovement.sitDownOnce,
        SyntheticMovement.armRaise,
      ]) {
        final source = SyntheticPoseSource(movement: movement);
        final counter = VelocityRepCounter();
        for (var i = 0; i < 300; i++) {
          counter.update(SyntheticPose.kneeAngleOf(source.next()), at(i));
        }
        expect(counter.count, 0, reason: '$movement');
      }
    });

    test('walking in place is robust to camera noise across seeds', () {
      for (var seed = 0; seed < 20; seed++) {
        final source = SyntheticPoseSource(
          movement: SyntheticMovement.walkInPlace,
          noise: 3,
          seed: seed,
        );
        final counter = VelocityRepCounter();
        for (var i = 0; i < 300; i++) {
          counter.update(SyntheticPose.kneeAngleOf(source.next()), at(i));
        }
        expect(counter.count, 0, reason: 'seed $seed');
      }
    });

    test('timestamps advance at the stated frame rate', () {
      final source = SyntheticPoseSource(fps: 30);
      source.take(31);
      expect(source.elapsedAt(30), const Duration(seconds: 1));
    });

    test('reset replays an identical sequence', () {
      final source = SyntheticPoseSource(noise: 3, seed: 5);
      final first = source.take(20).map(SyntheticPose.kneeAngleOf).toList();
      source.reset();
      final second = source.take(20).map(SyntheticPose.kneeAngleOf).toList();
      expect(second, first);
    });

    test('streams the number of frames asked for', () async {
      final source = SyntheticPoseSource();
      expect(await source.stream(frames: 12).length, 12);
    });

    test('drives a whole exercise session with no camera', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      final source = SyntheticPoseSource(depth: 0.95);

      var completed = 0;
      for (var i = 0; i < 5 * 36; i++) {
        final update = session.update(source.next(), at(i));
        if (update.repCompleted) completed++;
      }

      expect(completed, 5);
      expect(session.repCount, 5);
    });
  });
}
