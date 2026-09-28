import 'dart:math' as math;

import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  BodyFrame poseAt(
    double bend, {
    double scale = 1,
    double armRaise = 0,
    Map<LandmarkType, double> likelihoods = const {},
    Map<LandmarkType, (double, double)> overrides = const {},
  }) {
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

    final armAngle = math.pi * (90 - armRaise * 90) / 180;
    const armLength = 0.11;

    final base = <LandmarkType, (double, double)>{
      LandmarkType.nose: (0.50, shoulderY - 0.13),
      LandmarkType.leftShoulder: (0.58, shoulderY),
      LandmarkType.rightShoulder: (0.42, shoulderY),
      LandmarkType.leftElbow: (
        0.58 + armLength * math.cos(armAngle) * 0.55,
        shoulderY + armLength * math.sin(armAngle) * 0.55,
      ),
      LandmarkType.rightElbow: (
        0.42 - armLength * math.cos(armAngle) * 0.55,
        shoulderY + armLength * math.sin(armAngle) * 0.55,
      ),
      LandmarkType.leftWrist: (
        0.58 + armLength * math.cos(armAngle) * 1.30,
        shoulderY + armLength * math.sin(armAngle) * 1.30,
      ),
      LandmarkType.rightWrist: (
        0.42 - armLength * math.cos(armAngle) * 1.30,
        shoulderY + armLength * math.sin(armAngle) * 1.30,
      ),
      LandmarkType.leftHip: (0.555 + hipBack, hipY),
      LandmarkType.rightHip: (0.445 + hipBack, hipY),
      LandmarkType.leftKnee: (0.555 + kneeForward, kneeY),
      LandmarkType.rightKnee: (0.445 + kneeForward, kneeY),
      LandmarkType.leftAnkle: (0.555, ankleY),
      LandmarkType.rightAnkle: (0.445, ankleY),
      LandmarkType.leftFootIndex: (0.575, ankleY + 0.03),
      LandmarkType.rightFootIndex: (0.425, ankleY + 0.03),
    };

    return BodyFrame(
      timestamp: DateTime.fromMillisecondsSinceEpoch(0),
      framing: const FramingReport(
        framing: Framing.full,
        advice: FramingAdvice.none,
        torsoQuality: 1,
        upperBodyQuality: 1,
        lowerBodyQuality: 1,
      ),
      isMirrored: false,
      landmarks: <BodyLandmark>[
        for (final entry in base.entries)
          BodyLandmark(
            type: entry.key,
            x: overrides[entry.key]?.$1 ??
                (0.5 + (entry.value.$1 - 0.5) * scale),
            y: overrides[entry.key]?.$2 ??
                (0.5 + (entry.value.$2 - 0.5) * scale),
            z: 0,
            likelihood: likelihoods[entry.key] ?? 1.0,
            inFrameLikelihood: likelihoods[entry.key] ?? 1.0,
          ),
      ],
    );
  }

  SkeletonCalibrator calibrated({int frames = 60, double scale = 1}) {
    final cal = SkeletonCalibrator();
    for (var i = 0; i < frames; i++) {
      cal.update(poseAt(0, scale: scale));
    }
    return cal;
  }

  double angleAt(
    BodyFrame frame,
    LandmarkType vertex,
    LandmarkType a,
    LandmarkType b,
  ) {
    final v = frame[vertex]!;
    final p = frame[a]!;
    final q = frame[b]!;
    final ax = p.x - v.x;
    final ay = p.y - v.y;
    final bx = q.x - v.x;
    final by = q.y - v.y;
    final na = math.sqrt(ax * ax + ay * ay);
    final nb = math.sqrt(bx * bx + by * by);
    final cos = ((ax * bx + ay * by) / (na * nb)).clamp(-1.0, 1.0);
    return math.acos(cos) * 180 / math.pi;
  }

  double boneLength(BodyFrame f, LandmarkType a, LandmarkType b) {
    final p = f[a]!;
    final q = f[b]!;
    return math.sqrt(math.pow(p.x - q.x, 2) + math.pow(p.y - q.y, 2));
  }

  group('SkeletonCalibrator', () {
    test('becomes ready after about two seconds', () {
      expect(calibrated().model.ready, isTrue);
    });

    test('is not ready immediately', () {
      expect(calibrated(frames: 3).model.ready, isFalse);
    });

    test('lengths are relative to torso, not pixels', () {
      final near = calibrated().model;
      final far = calibrated(scale: 0.5).model;
      for (final entry in near.lengths.entries) {
        final other = far.lengths[entry.key];
        if (other == null) continue;
        expect((entry.value - other).abs(), lessThan(0.02),
            reason: '${entry.key} is not scale-invariant');
      }
    });

    test('left and right share a length', () {
      final model = calibrated().model;
      for (final (a, b) in mirroredBones) {
        final la = model.lengths[a];
        final lb = model.lengths[b];
        if (la == null || lb == null) continue;
        expect(la, closeTo(lb, 1e-12));
      }
    });

    test('a landmark the model is guessing at is not calibrated from', () {
      final cal = SkeletonCalibrator();
      for (var i = 0; i < 60; i++) {
        cal.update(poseAt(0, likelihoods: {LandmarkType.leftWrist: 0.1}));
      }
      expect(cal.model.lengths.containsKey(LandmarkType.leftWrist), isFalse);
    });

    test('a frame with no reliable torso is refused', () {
      final cal = SkeletonCalibrator();
      for (var i = 0; i < 60; i++) {
        cal.update(poseAt(0, likelihoods: {LandmarkType.leftHip: 0.1}));
      }
      expect(cal.model.lengths, isEmpty);
    });
  });

  group('applyBodyModel', () {
    test('changes no joint angle', () {
      final model = calibrated().model;
      var worst = 0.0;
      for (final bend in <double>[0, 0.3, 0.6, 0.9]) {
        for (final raise in <double>[0, 0.5, 1.0]) {
          final frame = poseAt(bend, armRaise: raise);
          final fixed = applyBodyModel(frame, model);
          for (final (v, a, b) in <(LandmarkType, LandmarkType, LandmarkType)>[
            (LandmarkType.leftKnee, LandmarkType.leftHip,
                LandmarkType.leftAnkle),
            (LandmarkType.rightKnee, LandmarkType.rightHip,
                LandmarkType.rightAnkle),
            (LandmarkType.leftElbow, LandmarkType.leftShoulder,
                LandmarkType.leftWrist),
            (LandmarkType.rightElbow, LandmarkType.rightShoulder,
                LandmarkType.rightWrist),
          ]) {
            final before = angleAt(frame, v, a, b);
            final after = angleAt(fixed, v, a, b);
            worst = math.max(worst, (before - after).abs());
          }
        }
      }
      expect(worst, lessThan(1e-4),
          reason: 'directions are being measured from moved parents again');
    });

    test('bone lengths become a fixed multiple of the torso', () {
      final model = calibrated().model;
      final ratios = <double>[];
      for (final bend in <double>[0, 0.2, 0.4, 0.6, 0.8]) {
        final frame = poseAt(bend);
        final fixed = applyBodyModel(frame, model);
        final torso = torsoScale(frame)!;
        ratios.add(
          boneLength(fixed, LandmarkType.leftKnee, LandmarkType.leftHip) /
              torso,
        );
      }
      for (final r in ratios) {
        expect(r, closeTo(ratios.first, 1e-9),
            reason: 'the femur is still changing length');
      }
    });

    test('an unready model leaves the frame alone', () {
      final frame = poseAt(0);
      final out = applyBodyModel(frame, const BodyModel());
      expect(identical(out, frame), isTrue);
    });

    test('a corrected wrist carries its hand with it', () {
      final model = calibrated().model;
      final frame = poseAt(0, armRaise: 0.7, overrides: {
        LandmarkType.leftWrist: (0.95, 0.30),
      });
      final fixed = applyBodyModel(frame, model);
      final wristShift = fixed[LandmarkType.leftWrist]!.x -
          frame[LandmarkType.leftWrist]!.x;
      final ankleShift = fixed[LandmarkType.leftAnkle]!.x -
          frame[LandmarkType.leftAnkle]!.x;
      final footShift = fixed[LandmarkType.leftFootIndex]!.x -
          frame[LandmarkType.leftFootIndex]!.x;
      expect(wristShift.abs(), greaterThan(0),
          reason: 'the displaced wrist was not corrected at all');
      expect(footShift, closeTo(ankleShift, 1e-12),
          reason: 'the foot did not move with its ankle');
    });

    test('framing is carried through, not recomputed', () {
      final model = calibrated().model;
      final frame = poseAt(0);
      expect(applyBodyModel(frame, model).framing, frame.framing);
    });
  });

  group('torsoScale', () {
    test('is positive for a normal pose', () {
      expect(torsoScale(poseAt(0)), greaterThan(0));
    });

    test('scales with the subject', () {
      final near = torsoScale(poseAt(0))!;
      final far = torsoScale(poseAt(0, scale: 0.5))!;
      expect(far, closeTo(near / 2, 1e-9));
    });
  });
}
