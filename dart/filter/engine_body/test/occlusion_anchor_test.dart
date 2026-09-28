import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  BodyLandmark put(
    LandmarkType type,
    double x,
    double y, {
    double likelihood = 0.95,
  }) =>
      BodyLandmark(
        type: type,
        x: x,
        y: y,
        z: 0,
        likelihood: likelihood,
        inFrameLikelihood: likelihood,
      );

  BodyFrame pose({
    Map<LandmarkType, double> likelihoods = const {},
    Map<LandmarkType, (double, double)> moved = const {},
    double shiftX = 0,
    double scale = 1,
  }) {
    const base = <LandmarkType, (double, double)>{
      LandmarkType.nose: (0.50, 0.18),
      LandmarkType.leftShoulder: (0.58, 0.30),
      LandmarkType.rightShoulder: (0.42, 0.30),
      LandmarkType.leftElbow: (0.62, 0.42),
      LandmarkType.rightElbow: (0.38, 0.42),
      LandmarkType.leftWrist: (0.65, 0.54),
      LandmarkType.rightWrist: (0.35, 0.54),
      LandmarkType.leftHip: (0.55, 0.55),
      LandmarkType.rightHip: (0.45, 0.55),
      LandmarkType.leftKnee: (0.55, 0.72),
      LandmarkType.rightKnee: (0.45, 0.72),
      LandmarkType.leftAnkle: (0.55, 0.90),
      LandmarkType.rightAnkle: (0.45, 0.90),
    };
    const centreX = 0.5;
    const centreY = 0.55;
    return BodyFrame(
      timestamp: DateTime(2026),
      landmarks: [
        for (final e in base.entries)
          put(
            e.key,
            moved[e.key]?.$1 ??
                (centreX + (e.value.$1 - centreX) * scale + shiftX),
            moved[e.key]?.$2 ?? (centreY + (e.value.$2 - centreY) * scale),
            likelihood: likelihoods[e.key] ?? 0.95,
          ),
      ],
      framing: FramingReport.noSubject,
      isMirrored: false,
    );
  }

  Duration at(int frame) =>
      Duration(microseconds: (frame * 1000000 / 30).round());

  group('off-frame detection', () {
    test('catches a landmark past each edge', () {
      for (final entry in {
        'left': (-0.05, 0.5),
        'right': (1.05, 0.5),
        'top': (0.5, -0.05),
        'bottom': (0.5, 1.05),
      }.entries) {
        expect(
          isOutsideFrame(
              put(LandmarkType.nose, entry.value.$1, entry.value.$2)),
          isTrue,
          reason: 'a landmark past the ${entry.key} edge was not detected',
        );
      }
      expect(isOutsideFrame(put(LandmarkType.nose, 0.5, 0.5)), isFalse);
    });

    test('refuses a non-finite coordinate', () {
      expect(isOutsideFrame(put(LandmarkType.nose, double.nan, 0.5)), isTrue);
      expect(
          isOutsideFrame(put(LandmarkType.nose, 0.5, double.infinity)), isTrue);
    });

    test('the margin is small enough to reject a genuine escape', () {
      expect(edgeMargin, lessThan(0.01),
          reason: 'a generous margin measurably admitted landmarks that were '
              'genuinely outside the picture');
      expect(edgeMargin, greaterThan(0.00085),
          reason: 'the margin must clear the measured noise floor');
    });

    test('an off-frame landmark is discarded whatever its confidence', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < 40; i++) {
        final f = pose();
        tracker.update(f, cal.update(f), at(i));
      }

      final gone = pose(
        moved: {LandmarkType.leftWrist: (1.06, 0.54)},
        likelihoods: {LandmarkType.leftWrist: 0.82},
      );
      final report = tracker.update(gone, cal.model, at(41));

      expect(report.verdictFor(LandmarkType.leftWrist),
          LandmarkVerdict.discarded);
      expect(report.isDrawable(LandmarkType.leftWrist), isFalse);
      expect(report.isUsable(LandmarkType.leftWrist), isFalse);
    });
  });

  group('the seen-before rule', () {
    test('a limb seen once is not held when it disappears', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      final f = pose();
      tracker.update(f, cal.update(f), at(0));

      final hidden = pose(likelihoods: {LandmarkType.leftWrist: 0.1});
      final report = tracker.update(hidden, cal.model, at(1));
      expect(report.verdictFor(LandmarkType.leftWrist),
          LandmarkVerdict.discarded,
          reason: 'a position was invented for a limb seen once');
    });

    test('a limb seen enough times is held', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < defaultFramesBeforeTrusted + 5; i++) {
        final f = pose();
        tracker.update(f, cal.update(f), at(i));
      }

      final hidden = pose(likelihoods: {LandmarkType.leftWrist: 0.1});
      final report =
          tracker.update(hidden, cal.model, at(defaultFramesBeforeTrusted + 6));
      expect(report.verdictFor(LandmarkType.leftWrist), LandmarkVerdict.held);
      expect(report.isDrawable(LandmarkType.leftWrist), isTrue);
    });

    test('doubtful frames do not earn trust', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < defaultFramesBeforeTrusted * 3; i++) {
        final f = pose(likelihoods: {LandmarkType.leftWrist: 0.1});
        tracker.update(f, cal.update(pose()), at(i));
      }
      final report = tracker.update(
        pose(likelihoods: {LandmarkType.leftWrist: 0.1}),
        cal.model,
        at(defaultFramesBeforeTrusted * 3 + 1),
      );
      expect(report.verdictFor(LandmarkType.leftWrist),
          LandmarkVerdict.discarded);
    });

    test('leaving the frame erases the memory', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < defaultFramesBeforeTrusted + 5; i++) {
        final f = pose();
        tracker.update(f, cal.update(f), at(i));
      }

      tracker.update(
        pose(moved: {LandmarkType.leftWrist: (1.2, 0.54)}),
        cal.model,
        at(defaultFramesBeforeTrusted + 6),
      );

      final report = tracker.update(
        pose(likelihoods: {LandmarkType.leftWrist: 0.1}),
        cal.model,
        at(defaultFramesBeforeTrusted + 7),
      );
      expect(report.verdictFor(LandmarkType.leftWrist),
          LandmarkVerdict.discarded,
          reason: 'a limb was held at a position from before it left frame');
    });
  });

  group('holding', () {
    test('a held limb follows the body as it changes apparent size', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < defaultFramesBeforeTrusted + 5; i++) {
        final f = pose();
        tracker.update(f, cal.update(f), at(i));
      }

      final seenAt = pose();
      final wristBefore = seenAt[LandmarkType.leftWrist]!;
      final elbowBefore = seenAt[LandmarkType.leftElbow]!;
      final torsoBefore = torsoScale(seenAt)!;
      final ratioBefore = ((wristBefore.x - elbowBefore.x).abs() +
              (wristBefore.y - elbowBefore.y).abs()) /
          torsoBefore;

      final bigger = pose(
        scale: 1.5,
        likelihoods: {LandmarkType.leftWrist: 0.1},
      );
      final report =
          tracker.update(bigger, cal.model, at(defaultFramesBeforeTrusted + 6));
      expect(report.verdictFor(LandmarkType.leftWrist), LandmarkVerdict.held);

      final held = report.landmarks
          .firstWhere((l) => l.type == LandmarkType.leftWrist);
      final elbowNow = report.landmarks
          .firstWhere((l) => l.type == LandmarkType.leftElbow);
      final torsoNow = torsoScale(report.applyTo(bigger))!;
      final ratioNow =
          ((held.x - elbowNow.x).abs() + (held.y - elbowNow.y).abs()) /
              torsoNow;

      expect((ratioNow - ratioBefore).abs() / ratioBefore, lessThan(0.2),
          reason: 'the held forearm did not follow the body as it grew');
    });

    test('a hold expires rather than being asserted forever', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < defaultFramesBeforeTrusted + 5; i++) {
        final f = pose();
        tracker.update(f, cal.update(f), at(i));
      }

      final hidden = pose(likelihoods: {LandmarkType.leftWrist: 0.1});
      var verdict = LandmarkVerdict.seen;
      for (var i = 0; i < 60; i++) {
        final report = tracker.update(
          hidden,
          cal.model,
          at(defaultFramesBeforeTrusted + 6) + Duration(milliseconds: i * 40),
        );
        verdict = report.verdictFor(LandmarkType.leftWrist);
      }
      expect(verdict, LandmarkVerdict.discarded,
          reason: 'a wrist hidden for over two seconds was still asserted');
    });

    test('a stream gap clears held state', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < defaultFramesBeforeTrusted + 5; i++) {
        final f = pose();
        tracker.update(f, cal.update(f), at(i));
      }
      final report = tracker.update(
        pose(likelihoods: {LandmarkType.leftWrist: 0.1}),
        cal.model,
        const Duration(seconds: 30),
      );
      expect(report.verdictFor(LandmarkType.leftWrist),
          LandmarkVerdict.discarded);
    });
  });

  group('joint anchor', () {
    test('holds a still joint against noise', () {
      final anchor = JointAnchor();
      final base = pose();
      anchor.update(base);

      var worst = 0.0;
      for (var i = 0; i < 30; i++) {
        final jitter = (i.isEven ? 1 : -1) * 0.0008;
        final noisy = pose(moved: {
          LandmarkType.leftWrist: (0.65 + jitter, 0.54 + jitter),
        });
        final out = anchor.update(noisy);
        final w = out[LandmarkType.leftWrist]!;
        final d = (w.x - 0.65).abs();
        if (d > worst) worst = d;
      }
      expect(worst, lessThan(0.0008),
          reason: 'the anchor did not damp noise at all');
    });

    test('follows real movement without lag', () {
      final anchor = JointAnchor();
      anchor.update(pose());
      final moved =
          pose(moved: {LandmarkType.leftWrist: (0.65 + 0.12, 0.54)});
      final out = anchor.update(moved);
      expect((out[LandmarkType.leftWrist]!.x - 0.77).abs(), lessThan(0.001),
          reason: 'a large movement was held back');
    });

    test('never pushes a landmark out of frame', () {
      final anchor = JointAnchor();
      final edge = pose(moved: {LandmarkType.leftWrist: (0.9999, 0.54)});
      anchor.update(edge);
      for (var i = 0; i < 20; i++) {
        final out = anchor.update(edge);
        final w = out[LandmarkType.leftWrist]!;
        expect(w.x, lessThanOrEqualTo(1.0));
        expect(w.x, greaterThanOrEqualTo(0.0));
      }
    });

    test('does not drag a genuinely off-frame landmark back inside', () {
      final anchor = JointAnchor();
      anchor.update(pose());
      final gone = pose(moved: {LandmarkType.leftWrist: (1.4, 0.54)});
      final out = anchor.update(gone);
      expect(out[LandmarkType.leftWrist]!.x, greaterThan(1.0));
    });

    test('leaves discarded landmarks alone', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      final anchor = JointAnchor();
      final f = pose();
      final report = tracker.update(f, cal.update(f), at(0));
      final out = anchor.update(report.applyTo(f), report: report);
      expect(out.landmarks.length, f.landmarks.length);
    });
  });

  group('angle rate guard', () {
    test('rejects a change no joint can make', () {
      final guard = AngleRateGuard();
      expect(guard.check('leftKnee', 90, Duration.zero), isTrue);
      expect(
        guard.check('leftKnee', 244, const Duration(milliseconds: 40)),
        isFalse,
      );
    });

    test('allows genuinely fast movement', () {
      final guard = AngleRateGuard();
      var rejected = 0;
      var angle = 60.0;
      for (var k = 0; k < 40; k++) {
        final phase = k % 14;
        angle = 60.0 + 15.0 * (phase <= 7 ? phase : 14 - phase);
        if (!guard.check('leftKnee', angle, at(k))) rejected++;
      }
      expect(rejected, 0);
    });

    test('does not drag its reference with a rejected value', () {
      final guard = AngleRateGuard();
      guard.check('leftKnee', 90, Duration.zero);
      expect(guard.check('leftKnee', 250, const Duration(milliseconds: 40)),
          isFalse);
      expect(guard.check('leftKnee', 92, const Duration(milliseconds: 80)),
          isTrue,
          reason: 'a rejected value was recorded as the new reference');
    });

    test('forgives a detection gap', () {
      final guard = AngleRateGuard();
      guard.check('leftKnee', 90, Duration.zero);
      expect(
          guard.check('leftKnee', 170, const Duration(seconds: 3)), isTrue);
    });

    test('survives two frames sharing a timestamp', () {
      final guard = AngleRateGuard();
      guard.check('leftKnee', 90, Duration.zero);
      expect(guard.check('leftKnee', 170, Duration.zero), isTrue);
    });

    test('tracks joints separately', () {
      final guard = AngleRateGuard();
      guard.check('leftKnee', 90, Duration.zero);
      expect(
          guard.check('rightKnee', 30, const Duration(milliseconds: 40)),
          isTrue);
    });
  });

  group('calibration gates', () {
    test('accepts a low-confidence torso that is in frame', () {
      final cal = SkeletonCalibrator();
      BodyModel model = const BodyModel();
      for (var i = 0; i < 40; i++) {
        model = cal.update(pose(likelihoods: {
          LandmarkType.leftShoulder: 0.35,
          LandmarkType.rightShoulder: 0.35,
          LandmarkType.leftHip: 0.35,
          LandmarkType.rightHip: 0.35,
        }));
      }
      expect(model.ready, isTrue);
    });

    test('rejects a torso outside the frame', () {
      final cal = SkeletonCalibrator();
      BodyModel model = const BodyModel();
      for (var i = 0; i < 40; i++) {
        model = cal.update(pose(moved: {
          LandmarkType.leftHip: (0.55, 1.34),
          LandmarkType.rightHip: (0.45, 1.34),
        }));
      }
      expect(model.ready, isFalse);
    });

    test('the learn window matches the measurement', () {
      expect(SkeletonCalibrator().frames, 30);
      expect(SkeletonCalibrator().refreshEvery, 300,
          reason: 'faster refreshes measured much worse: 19.5% at 90 frames');
    });
  });

  group('wired into the session', () {
    test('an off-frame joint stops the measurement rather than faking it', () {
      final session = ExerciseSession(exercise: exercises['squat']!);
      for (var i = 0; i < 40; i++) {
        session.update(pose(), at(i));
      }
      final gone = pose(moved: {
        LandmarkType.leftHip: (1.3, 0.55),
        LandmarkType.rightHip: (1.3, 0.55),
        LandmarkType.leftKnee: (1.3, 0.72),
        LandmarkType.rightKnee: (1.3, 0.72),
        LandmarkType.leftAnkle: (1.3, 0.90),
        LandmarkType.rightAnkle: (1.3, 0.90),
      });
      final update = session.update(gone, at(41));
      expect(update.usable, isFalse);
      expect(update.reason, isNotEmpty,
          reason: 'a refusal must say why, so the user can act');
    });

    test('an established limb is forgotten when the subject changes', () {
      final tracker = OcclusionTracker();
      final cal = SkeletonCalibrator();
      for (var i = 0; i < defaultFramesBeforeTrusted + 5; i++) {
        final f = pose();
        tracker.update(f, cal.update(f), at(i));
      }

      final beforeReset = tracker.update(
        pose(likelihoods: {LandmarkType.leftWrist: 0.1}),
        cal.model,
        at(defaultFramesBeforeTrusted + 6),
      );
      expect(beforeReset.verdictFor(LandmarkType.leftWrist),
          LandmarkVerdict.held);

      tracker.reset();

      final afterReset = tracker.update(
        pose(likelihoods: {LandmarkType.leftWrist: 0.1}),
        cal.model,
        at(defaultFramesBeforeTrusted + 7),
      );
      expect(afterReset.verdictFor(LandmarkType.leftWrist),
          LandmarkVerdict.discarded,
          reason: "a new subject inherited the previous one's trusted limb");
    });
  });
}
