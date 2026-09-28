import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/pose_fixtures.dart';

void main() {
  const gate = FramingGate();

  group('FramingGate', () {
    test('a fully visible subject is full framing', () {
      final report = gate.assess(PoseFixtures.standing());
      expect(report.framing, Framing.full);
      expect(report.advice, FramingAdvice.none);
      expect(report.framing.hasTorso, isTrue);
      expect(report.framing.hasLegs, isTrue);
    });

    test('a seated subject with legs out of frame is upperBodyOnly', () {
      final report = gate.assess(PoseFixtures.standing(legsVisible: false));
      expect(report.framing, Framing.upperBodyOnly);
      expect(report.framing.hasTorso, isTrue);
      expect(report.framing.hasLegs, isFalse);
      expect(report.advice, FramingAdvice.moveBack);
    });

    test('low-confidence legs do not silently count as visible', () {
      final landmarks = PoseFixtures.standing(legsVisible: false);
      final ankle = landmarks.firstWhere(
        (l) => l.type == LandmarkType.leftAnkle,
      );
      expect(ankle.likelihood, lessThan(0.5), reason: 'fixture sanity check');

      final report = gate.assess(landmarks);
      expect(report.lowerBodyQuality, lessThan(0.5));
      expect(report.framing.hasLegs, isFalse);
    });

    test('an unreadable torso is partial, not upperBodyOnly', () {
      final report = gate.assess(PoseFixtures.standing(confidence: 0.25));
      expect(report.framing, Framing.partial);
      expect(report.framing.hasTorso, isFalse);
    });

    test('an empty landmark list is noSubject', () {
      final report = gate.assess([]);
      expect(report.framing, Framing.noSubject);
      expect(report.advice, FramingAdvice.stepIntoView);
      expect(report.torsoQuality, 0);
    });

    test('quality is not diluted by parts a consumer does not use', () {
      final report = gate.assess(PoseFixtures.standing(legsVisible: false));
      expect(report.qualityFor(needsLegs: false), greaterThan(0.7));
      expect(report.qualityFor(needsLegs: true), lessThan(0.3));
    });

    test('confidence combines placement and in-frame likelihood', () {
      final landmarks = [
        for (final l in PoseFixtures.standing())
          BodyLandmark(
            type: l.type,
            x: l.x,
            y: l.y,
            z: l.z,
            likelihood: 0.95,
            inFrameLikelihood: 0.1,
          ),
      ];
      final report = gate.assess(landmarks);
      expect(report.torsoQuality, lessThan(0.2));
      expect(report.framing, Framing.partial);
    });
  });

  group('BodyLandmark.isReliable', () {
    test('requires both confidence values to clear the threshold', () {
      const confident = BodyLandmark(
        type: LandmarkType.leftWrist,
        x: 0.5,
        y: 0.5,
        z: 0,
        likelihood: 0.9,
        inFrameLikelihood: 0.2,
      );
      expect(confident.isReliable(), isFalse);
    });
  });
}
