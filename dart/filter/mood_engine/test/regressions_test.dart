import 'dart:math' as math;

import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

List<double> normalised(List<double> v) {
  final total = v.fold<double>(0.0, (double a, double b) => a + b);
  if (total == 0) return v;
  return <double>[for (final x in v) x / total];
}

double gauss(math.Random r) {
  final u1 = math.max(1e-12, r.nextDouble());
  final u2 = r.nextDouble();
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
}

BaselineTracker readyBaseline(List<double> resting) {
  final tracker = BaselineTracker();
  final baseline = tracker.baselineFor(1);
  final probs = normalised(resting);
  final r = math.Random(7);
  for (var i = 0; i < PersonBaseline.required + 5; i++) {
    final noisy = <double>[
      for (final p in probs) math.max(1e-4, p + gauss(r) * 0.015)
    ];
    baseline.observe(normalised(noisy), <String, double>{});
  }
  expect(baseline.ready, isTrue);
  return tracker;
}

void main() {
  group('the baseline never invents an emotion', () {
    const resting = <double>[0.55, 0.12, 0.13, 0.05, 0.04, 0.06, 0.05];

    test('a resting face still reads neutral', () {
      final tracker = readyBaseline(resting);
      final probs = normalised(
          <double>[0.42, 0.08, 0.15, 0.06, 0.07, 0.12, 0.10]);
      final (List<double> out, PersonBaseline _) =
          tracker.adjust(probs, <String, double>{}, 1, learn: false);
      expect(emotions[argMax(out)], 'neutral');
    });

    test('neutral is never inflated or shaved', () {
      final tracker = readyBaseline(resting);
      final probs = normalised(
          <double>[0.42, 0.08, 0.15, 0.06, 0.07, 0.12, 0.10]);
      final (List<double> out, PersonBaseline _) =
          tracker.adjust(probs, <String, double>{}, 1, learn: false);
      final i = emotions.indexOf('neutral');
      expect(out[i], closeTo(probs[i], 1e-9));
    });

    test('a real expression still wins', () {
      final cases = <String, List<double>>{
        'happy': <double>[0.15, 0.62, 0.05, 0.07, 0.03, 0.04, 0.04],
        'anger': <double>[0.12, 0.03, 0.10, 0.05, 0.08, 0.12, 0.50],
        'sad': <double>[0.30, 0.05, 0.38, 0.05, 0.07, 0.08, 0.07],
        'surprise': <double>[0.10, 0.06, 0.05, 0.55, 0.10, 0.07, 0.07],
      };
      cases.forEach((String name, List<double> reading) {
        final tracker = readyBaseline(resting);
        final (List<double> out, PersonBaseline _) = tracker.adjust(
            normalised(reading), <String, double>{}, 1,
            learn: false);
        expect(emotions[argMax(out)], name, reason: name);
      });
    });

    test('what comes out is still a distribution', () {
      final tracker = readyBaseline(resting);
      final r = math.Random(1);
      for (var i = 0; i < 50; i++) {
        final probs = normalised(
            <double>[for (var k = 0; k < emotions.length; k++) r.nextDouble()]);
        final (List<double> out, PersonBaseline _) =
            tracker.adjust(probs, <String, double>{}, 1, learn: false);
        final sum = out.fold<double>(0.0, (double a, double b) => a + b);
        expect(sum, closeTo(1.0, 1e-6));
        for (final v in out) {
          expect(v, greaterThanOrEqualTo(0.0));
          expect(v.isFinite, isTrue);
        }
      }
    });

    test('it cannot amplify without limit', () {
      final tracker = readyBaseline(
          <double>[0.70, 0.24, 0.02, 0.01, 0.01, 0.01, 0.01]);
      final flat = <double>[
        for (var i = 0; i < emotions.length; i++) 1.0 / emotions.length
      ];
      final (List<double> out, PersonBaseline _) =
          tracker.adjust(flat, <String, double>{}, 1, learn: false);
      final hi = out.reduce(math.max);
      final lo = out.reduce(math.min);
      expect(hi / lo, lessThan(8.0));
    });
  });

  group('boxes come back in the frame they came from', () {
    test('the detector input is capped but not tiny', () {
      expect(defaultEngineConfig.detection.detectMaxWidth, greaterThan(0));
      expect(defaultEngineConfig.detection.detectMaxWidth,
          greaterThanOrEqualTo(480));
    });

    test('rescaling maps a box back to full size', () {
      final face = DetectedFace(
        bbox: const BoundingBox(10.0, 20.0, 60.0, 90.0),
        detScore: 0.9,
        keypoints: const <Point2>[Point2(10.0, 20.0), Point2(30.0, 25.0)],
      );
      FaceDetector.rescaleFaces(<DetectedFace>[face], 2.0, 1280, 720);
      final r = face;
      expect(r.bbox.x1, closeTo(20.0, 1e-6));
      expect(r.bbox.y1, closeTo(40.0, 1e-6));
      expect(r.bbox.x2, closeTo(120.0, 1e-6));
      expect(r.bbox.y2, closeTo(180.0, 1e-6));
      expect(r.keypoints.first.x, closeTo(20.0, 1e-6));
    });

    test('rescaling never runs off the frame', () {
      final face = DetectedFace(
        bbox: const BoundingBox(10.0, 20.0, 400.0, 400.0),
        detScore: 0.9,
        keypoints: const <Point2>[Point2(10.0, 20.0)],
      );
      FaceDetector.rescaleFaces(<DetectedFace>[face], 2.0, 640, 480);
      final r = face;
      expect(r.bbox.x2, lessThanOrEqualTo(639.0));
      expect(r.bbox.y2, lessThanOrEqualTo(479.0));
      expect(r.bbox.x1, greaterThanOrEqualTo(0.0));
      expect(r.bbox.y1, greaterThanOrEqualTo(0.0));
    });
  });
}
