import 'dart:convert';
import 'dart:math' as math;

import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

List<double> unit(List<double> v) {
  var n = 0.0;
  for (final x in v) {
    n += x * x;
  }
  n = math.sqrt(n);
  if (n == 0) return v;
  return <double>[for (final x in v) x / n];
}

DetectedFace faceAt({
  double x1 = 100,
  double y1 = 100,
  double x2 = 300,
  double y2 = 360,
  double score = 0.95,
  int seed = 1,
}) {
  final r = math.Random(seed);
  return DetectedFace(
    bbox: BoundingBox(x1, y1, x2, y2),
    detScore: score,
    embedding: unit(<double>[for (var i = 0; i < 512; i++) r.nextDouble() - 0.5]),
    keypoints: <Point2>[
      Point2(x1 + (x2 - x1) * 0.35, y1 + (y2 - y1) * 0.38),
      Point2(x1 + (x2 - x1) * 0.65, y1 + (y2 - y1) * 0.38),
      Point2(x1 + (x2 - x1) * 0.50, y1 + (y2 - y1) * 0.55),
      Point2(x1 + (x2 - x1) * 0.38, y1 + (y2 - y1) * 0.75),
      Point2(x1 + (x2 - x1) * 0.62, y1 + (y2 - y1) * 0.75),
    ],
  );
}

const Map<String, double> liveSignals = <String, double>{
  'depth': 1.0,
  'texture': 1.0,
  'colour': 1.0,
  'moire': 1.0,
  'motion': 1.0,
};

FrameInput frameWith(List<DetectedFace> faces, {double? now}) => FrameInput(
      width: 640,
      height: 480,
      faces: faces,
      sharpness: 120.0,
      spoofSignals: liveSignals,
      now: now,
    );

Future<MoodEngine> makeEngine({bool autoEnrol = true}) async {
  final db = InMemoryMoodStore();
  await db.open();
  return MoodEngine(
    db: db,
    emotionModel: FakeEmotionModel(),
    autoEnrolUnknown: autoEnrol,
  );
}

void main() {
  group('a whole frame goes through the engine', () {
    test('a face gives a usable reading', () async {
      final engine = await makeEngine();
      final out = await engine.analyseFrame(frameWith(<DetectedFace>[faceAt()]),
          persist: false);
      expect(out.ok, isTrue);
      expect(out.faceCount, 1);
      expect(out.emotion, isNotNull);
      expect(emotions.contains(out.emotion!.label), isTrue);
    });

    test('no face is handled without complaint', () async {
      final engine = await makeEngine();
      final out = await engine.analyseFrame(frameWith(<DetectedFace>[]),
          persist: false);
      expect(out.ok, isFalse);
      expect(out.message, 'no_face');
      expect(out.faceCount, 0);
    });

    test('the result survives being turned into json', () async {
      final engine = await makeEngine();
      final out = await engine.analyseFrame(frameWith(<DetectedFace>[faceAt()]),
          persist: false);
      final text = jsonEncode(out.toMap());
      expect(text, isNotEmpty);
      final back = jsonDecode(text) as Map<String, Object?>;
      expect(back['ok'], isTrue);
    });

    test('the nearest face is the one that gets read', () async {
      final engine = await makeEngine();
      final near = faceAt(x1: 80, y1: 60, x2: 380, y2: 440, seed: 2);
      final far = faceAt(x1: 10, y1: 10, x2: 70, y2: 90, seed: 3);
      final out = await engine.analyseFrame(
          frameWith(<DetectedFace>[far, near]),
          persist: false);
      expect(out.ok, isTrue);
      expect(out.faceCount, 2);
      expect(out.bbox, isNotNull);
      final b = out.bbox!;
      final w = (b[2] - b[0]).toDouble();
      final h = (b[3] - b[1]).toDouble();
      expect(w * h, closeTo(near.bbox.area, 2.0));
    });

    test('readings pile up over many frames', () async {
      final engine = await makeEngine();
      for (var i = 0; i < 12; i++) {
        await engine.analyseFrame(
            frameWith(<DetectedFace>[faceAt()], now: 1000.0 + i / 30.0),
            persist: false);
      }
      expect(engine.frameIndex, 12);
      final summary = engine.affect.summary();
      expect(summary['samples'], greaterThan(0));
    });
  });

  group('an unknown face can be picked up on its own', () {
    test('auto enrol makes a provisional person', () async {
      final engine = await makeEngine();
      await engine.startSession('auto');
      await engine.analyseFrame(frameWith(<DetectedFace>[faceAt(seed: 11)]),
          persist: false);
      final people = await engine.listPersons();
      expect(people, isNotEmpty);
      expect(people.first['is_provisional'], 1);
    });

    test('auto enrol can be switched off', () async {
      final engine = await makeEngine(autoEnrol: false);
      await engine.startSession('manual');
      await engine.analyseFrame(frameWith(<DetectedFace>[faceAt(seed: 12)]),
          persist: false);
      expect(await engine.listPersons(), isEmpty);
    });
  });

  group('what the engine reports stays sane', () {
    test('pose and attention stay in range', () async {
      final engine = await makeEngine();
      for (var i = 0; i < 6; i++) {
        final out = await engine.analyseFrame(
            frameWith(<DetectedFace>[faceAt(seed: 20 + i)]),
            persist: false);
        final pose = out.pose;
        if (pose == null) continue;
        expect(pose.attention, greaterThanOrEqualTo(0.0));
        expect(pose.attention, lessThanOrEqualTo(1.0));
        for (final a in <double>[pose.yaw, pose.pitch, pose.roll]) {
          expect(a, greaterThanOrEqualTo(-180.0));
          expect(a, lessThanOrEqualTo(180.0));
        }
      }
    });

    test('the emotion probabilities always form a distribution', () async {
      final engine = await makeEngine();
      for (var i = 0; i < 8; i++) {
        final out = await engine.analyseFrame(
            frameWith(<DetectedFace>[faceAt(seed: 30 + i)]),
            persist: false);
        final e = out.emotion;
        if (e == null) continue;
        final sum = e.probs.values.fold<double>(0.0, (double a, double b) => a + b);
        expect(sum, closeTo(1.0, 1e-6));
        for (final v in e.probs.values) {
          expect(v, greaterThanOrEqualTo(0.0));
        }
      }
    });

    test('nothing is written until a session asks for it', () async {
      final engine = await makeEngine();
      await engine.analyseFrame(frameWith(<DetectedFace>[faceAt()]),
          persist: false);
      final stats = await engine.db.stats();
      expect(stats['observations'], 0);
    });
  });
}
