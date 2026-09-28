import 'dart:math' as math;

import 'package:engine_body/src/core/body_frame.dart';
import 'package:engine_body/src/core/body_landmark.dart';
import 'package:engine_body/src/core/framing.dart';
import 'package:engine_body/src/core/landmark_type.dart';
import 'package:engine_body/src/core/reliability.dart';
import 'package:engine_body/src/core/rep_counter.dart';
import 'package:engine_body/src/core/velocity_reps.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame, {int fps = 30}) =>
      Duration(microseconds: (frame * 1000000 / fps).round());

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

  double kneeAngleOf(BodyFrame frame) {
    final h = frame[LandmarkType.leftHip]!;
    final k = frame[LandmarkType.leftKnee]!;
    final a = frame[LandmarkType.leftAnkle]!;
    final v1x = h.x - k.x;
    final v1y = h.y - k.y;
    final v2x = a.x - k.x;
    final v2y = a.y - k.y;
    final n1 = math.sqrt(v1x * v1x + v1y * v1y);
    final n2 = math.sqrt(v2x * v2x + v2y * v2y);
    final cosine = (v1x * v2x + v1y * v2y) / (n1 * n2);
    return math.acos(cosine.clamp(-1.0, 1.0)) * 180 / math.pi;
  }

  void squat(
    void Function(double angle, Duration t) feed,
    double depth, {
    int cycles = 5,
    int frames = 36,
  }) {
    var frame = 0;
    for (var c = 0; c < cycles; c++) {
      for (var i = 0; i < frames; i++) {
        final phase =
            (math.sin(2 * math.pi * i / frames - math.pi / 2) + 1) / 2;
        feed(kneeAngleOf(poseAt(depth * phase)), at(frame));
        frame++;
      }
    }
  }

  group('VelocityRepCounter', () {
    test('counts without any calibration, where the threshold cannot', () {
      for (final depth in [0.95, 0.70, 0.55]) {
        final counter = VelocityRepCounter();
        squat((angle, t) => counter.update(angle, t), depth);
        expect(
          counter.count,
          5,
          reason: 'depth $depth counted ${counter.count} of 5',
        );
      }
    });

    test('the threshold counter misses what this one catches', () {
      final threshold = RepCounter(squatConfig);
      final velocity = VelocityRepCounter();
      squat((angle, t) {
        threshold.update(angle, t);
        velocity.update(angle, t);
      }, 0.70);

      expect(
        threshold.count,
        0,
        reason: 'fixture is wrong: the default should miss this user',
      );
      expect(velocity.count, 5);
    });

    test('standing still produces no phantom repetitions', () {
      final counter = VelocityRepCounter();
      final random = math.Random(1);
      for (var i = 0; i < 300; i++) {
        final jitter = (random.nextDouble() - 0.5) * 6;
        counter.update(178 + jitter, at(i));
      }
      expect(counter.count, 0);
    });

    test('shallow movement is not counted', () {
      final counter = VelocityRepCounter();
      squat((angle, t) => counter.update(angle, t), 0.08);
      expect(counter.count, 0);
    });

    test('the resting end is inferred, not assumed', () {
      final descending = VelocityRepCounter();
      squat((angle, t) => descending.update(angle, t), 0.95, cycles: 3);
      expect(descending.count, 3);
      expect(descending.restingExtreme, isTrue,
          reason: 'a squat rests at the high extreme');
    });

    test('a null angle does not fabricate movement', () {
      final counter = VelocityRepCounter();
      for (var i = 0; i < 60; i++) {
        counter.update(null, at(i));
      }
      expect(counter.count, 0);
    });

    test('a single NaN does not silently stop the counter', () {
      for (final nanFrame in [18, 40, 55]) {
        final counter = VelocityRepCounter();
        for (var i = 0; i < 108; i++) {
          final phase =
              (math.sin(2 * math.pi * (i % 36) / 36 - math.pi / 2) + 1) / 2;
          final angle = i == nanFrame ? double.nan : 180 - 100 * phase;
          counter.update(angle, at(i));
        }
        expect(
          counter.count,
          3,
          reason: 'a NaN at frame $nanFrame cost repetitions',
        );
      }
    });

    test('infinities are refused like NaN', () {
      final counter = VelocityRepCounter();
      for (var i = 0; i < 108; i++) {
        final phase =
            (math.sin(2 * math.pi * (i % 36) / 36 - math.pi / 2) + 1) / 2;
        final angle = i == 30 ? double.infinity : 180 - 100 * phase;
        counter.update(angle, at(i));
      }
      expect(counter.count, 3);
    });

    test('a gap in the stream does not fabricate velocity', () {
      final velocity = SignedVelocity();
      velocity.update(170, Duration.zero);
      velocity.update(169, at(1));
      final rate = velocity.update(
        90,
        at(1) + const Duration(milliseconds: 500),
      );
      expect(rate, isNull);
    });
  });

  group('BoneLengthMonitor', () {
    test('catches a snapped landmark', () {
      final monitor = BoneLengthMonitor();
      final frame = poseAt(0);
      final torso = torsoLength(frame);

      for (var i = 0; i < 30; i++) {
        expect(monitor.update(frame, torso), isEmpty);
      }

      final broken = BodyFrame(
        timestamp: frame.timestamp,
        landmarks: [
          for (final l in frame.landmarks)
            if (l.type == LandmarkType.leftKnee)
              BodyLandmark(
                type: l.type,
                x: l.x,
                y: l.y + 0.25,
                z: l.z,
                likelihood: l.likelihood,
                inFrameLikelihood: l.inFrameLikelihood,
              )
            else
              l,
        ],
        framing: frame.framing,
        isMirrored: false,
      );

      expect(
        monitor.update(broken, torso),
        contains(LandmarkType.leftKnee),
      );
    });

    test('tolerates normal movement', () {
      final monitor = BoneLengthMonitor();
      var falseAlarms = 0;
      for (var cycle = 0; cycle < 4; cycle++) {
        for (var i = 0; i < 36; i++) {
          final phase = (math.sin(2 * math.pi * i / 36 - math.pi / 2) + 1) / 2;
          final frame = poseAt(0.95 * phase);
          final suspect = monitor.update(frame, torsoLength(frame));
          if (cycle > 0 && suspect.isNotEmpty) falseAlarms++;
        }
      }
      expect(falseAlarms, 0);
    });

    test('ignores landmarks the model is guessing at', () {
      final monitor = BoneLengthMonitor();
      final frame = poseAt(0, likelihoods: {LandmarkType.leftKnee: 0.1});
      final torso = torsoLength(frame);
      for (var i = 0; i < 40; i++) {
        monitor.update(frame, torso);
      }
      expect(monitor.update(frame, torso), isEmpty);
    });
  });

  group('ReliabilityMonitor', () {
    test('catches a teleport in one frame', () {
      final monitor = ReliabilityMonitor();
      final frame = poseAt(0);
      final torso = torsoLength(frame);
      for (var i = 0; i < 20; i++) {
        monitor.update(frame, torso, at(i));
      }

      final jumped = BodyFrame(
        timestamp: frame.timestamp,
        landmarks: [
          for (final l in frame.landmarks)
            if (l.type == LandmarkType.leftWrist)
              BodyLandmark(
                type: l.type,
                x: l.x + 0.5,
                y: l.y,
                z: l.z,
                likelihood: l.likelihood,
                inFrameLikelihood: l.inFrameLikelihood,
              )
            else
              l,
        ],
        framing: frame.framing,
        isMirrored: false,
      );

      final report = monitor.update(jumped, torso, at(20));
      expect(report.trustworthy, isFalse);
      expect(report.reasons.join(), contains('fast'));
    });

    test('a steady pose is trusted', () {
      final monitor = ReliabilityMonitor();
      final frame = poseAt(0);
      final torso = torsoLength(frame);
      var distrusted = 0;
      for (var i = 0; i < 60; i++) {
        if (!monitor.update(frame, torso, at(i)).trustworthy) distrusted++;
      }
      expect(distrusted, 0);
    });

    test('reacts faster than a smoothed confidence score would', () {
      var value = 1.0;
      var frames = 0;
      while (value > 0.5) {
        value += 0.1 * (0 - value);
        frames++;
      }
      expect(
        frames,
        greaterThanOrEqualTo(6),
        reason: 'the lag this file exists to avoid has changed',
      );
    });
  });

  group('torsoLength', () {
    test('is positive for a normal pose', () {
      expect(torsoLength(poseAt(0)), greaterThan(0));
    });

    test('refuses when a required landmark is unreliable', () {
      final frame = poseAt(0, likelihoods: {LandmarkType.leftHip: 0.2});
      expect(torsoLength(frame), isNull);
    });
  });
}
