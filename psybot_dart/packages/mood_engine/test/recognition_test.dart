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

List<double> seededEmbedding(int seed, {int size = 512}) {
  final r = math.Random(seed);
  return unit(<double>[for (var i = 0; i < size; i++) r.nextDouble() - 0.5]);
}

List<double> nudged(List<double> base, int seed, double amount) {
  final r = math.Random(seed);
  return unit(<double>[
    for (final x in base) x + (r.nextDouble() - 0.5) * amount
  ]);
}

void main() {
  group('embeddings behave like directions', () {
    test('a normalised embedding has unit length', () {
      final e = seededEmbedding(1);
      expect(e.length, 512);
      var n = 0.0;
      for (final x in e) {
        n += x * x;
      }
      expect((math.sqrt(n) - 1.0).abs(), lessThan(1e-4));
    });

    test('an embedding matches itself exactly', () {
      final e = seededEmbedding(2);
      expect(cosineSimilarity(e, e), closeTo(1.0, 1e-9));
      expect(cosineDistance(e, e), closeTo(0.0, 1e-9));
    });

    test('the same face scores far above a different one', () {
      final base = seededEmbedding(3);
      final same = nudged(base, 4, 0.10);
      final other = seededEmbedding(99);
      final near = cosineSimilarity(base, same);
      final far = cosineSimilarity(base, other);
      expect(near, greaterThan(far));
      expect(near - far, greaterThan(0.25),
          reason: 'separation margin is too narrow: $near vs $far');
    });

    test('similarity never leaves its range', () {
      for (var s = 0; s < 40; s++) {
        final a = seededEmbedding(s);
        final b = seededEmbedding(s + 500);
        final v = cosineSimilarity(a, b);
        expect(v, greaterThanOrEqualTo(-1.0 - 1e-9));
        expect(v, lessThanOrEqualTo(1.0 + 1e-9));
      }
    });
  });

  group('a face the engine has not met becomes provisional', () {
    late InMemoryMoodStore db;
    late FaceRecognizer rec;
    late MoodEngine engine;

    setUp(() async {
      db = InMemoryMoodStore();
      await db.open();
      rec = FaceRecognizer(db);
      engine = MoodEngine(db: db);
    });

    test('the first unknown face is named Unknown-1', () async {
      final (int pid, String name) =
          await rec.enrolProvisional(seededEmbedding(7), quality: 0.8);
      expect(name, 'Unknown-1');
      final person = await db.getPerson(pid);
      expect(person, isNotNull);
      expect(person!['is_provisional'], 1);
    });

    test('naming it makes it a real person', () async {
      final (int pid, String _) =
          await rec.enrolProvisional(seededEmbedding(8), quality: 0.8);
      await engine.renamePerson(pid, 'Obama');
      final person = await db.getPerson(pid);
      expect(person!['name'], 'Obama');
      expect(person['is_provisional'], 0);
    });

    test('naming it into an existing person merges the two', () async {
      final int existing = await db.createPerson('Obama');
      await db.addEmbedding(existing, seededEmbedding(10), quality: 0.9);

      final (int pid, String _) =
          await rec.enrolProvisional(seededEmbedding(11), quality: 0.8);
      await engine.renamePerson(pid, 'Obama');

      expect(await db.getPerson(pid), isNull,
          reason: 'the provisional record should be gone');
      expect(await db.countEmbeddings(existing), 2);
    });

    test('provisional numbering keeps climbing', () async {
      final (int _, String a) =
          await rec.enrolProvisional(seededEmbedding(20), quality: 0.8);
      final (int _, String b) =
          await rec.enrolProvisional(seededEmbedding(21), quality: 0.8);
      expect(a, 'Unknown-1');
      expect(b, 'Unknown-2');
    });
  });
}
