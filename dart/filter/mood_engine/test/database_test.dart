import 'package:mood_engine/mood_engine.dart';
import 'package:mood_engine/sqflite_store.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'package:test/test.dart';

typedef StoreFactory = Future<MoodStore> Function();

List<double> pseudoVector(int seed, int size, {double scale = 1.0}) {
  var state = seed * 2654435761 + 12345;
  return <double>[
    for (var i = 0; i < size; i++)
      () {
        state = (state * 1103515245 + 12345) & 0x7fffffff;
        return ((state / 0x7fffffff) * 2.0 - 1.0) * scale;
      }(),
  ];
}

void main() {
  sqfliteFfiInit();

  final factories = <String, StoreFactory>{
    'in-memory': () async {
      final store = InMemoryMoodStore();
      await store.open();
      return store;
    },
    'sqflite-ffi': () async {
      final database = await databaseFactoryFfi.openDatabase(
        inMemoryDatabasePath,
      );
      final store = SqfliteMoodStore(database);
      await store.open();
      return store;
    },
  };

  for (final entry in factories.entries) {
    group(entry.key, () {
      late MoodStore db;

      setUp(() async => db = await entry.value());
      tearDown(() async => db.close());

      test('person crud', () async {
        final pid = await db.createPerson('Ahmed');
        expect((await db.getPerson(pid))!['name'], 'Ahmed');
        expect(await db.getOrCreatePerson('Ahmed'), pid);

        await db.renamePerson(pid, 'Ahmed Ali');
        final person = (await db.getPerson(pid))!;
        expect(person['name'], 'Ahmed Ali');
        expect(person['is_provisional'], 0);

        await db.deletePerson(pid);
        expect(await db.getPerson(pid), isNull);
      });

      test('embedding cache is normalised', () async {
        final pid = await db.createPerson('Sara');
        for (var i = 0; i < 5; i++) {
          await db.addEmbedding(pid, pseudoVector(i, 512, scale: 7.0));
        }

        final matrix = db.embeddingMatrix!;
        expect(matrix.length, 5);
        expect(matrix.first.length, 512);
        for (final row in matrix) {
          expect(vectorNorm(row), closeTo(1.0, 1e-5),
              reason: 'cache vectors must be unit norm');
        }
        expect(db.embeddingPersonIds, List<int>.filled(5, pid));
      });

      test('provisional naming fills gaps', () async {
        expect(await db.nextProvisionalName(), 'Unknown-1');
        await db.createPerson('Unknown-1', provisional: true);
        await db.createPerson('Unknown-3', provisional: true);
        expect(await db.nextProvisionalName(), 'Unknown-2');
      });

      test('prune keeps highest quality', () async {
        final pid = await db.createPerson('Omar');
        var seed = 0;
        for (final q in <double>[0.1, 0.9, 0.5, 0.7, 0.3]) {
          await db.addEmbedding(
            pid,
            pseudoVector(seed++, 512),
            quality: q,
          );
        }

        await db.pruneEmbeddings(pid, 2);
        expect(await db.countEmbeddings(pid), 2);
      });

      test('merge persons moves all data', () async {
        final a = await db.createPerson('Unknown-1', provisional: true);
        final b = await db.createPerson('Khalid');
        await db.addEmbedding(a, pseudoVector(2, 512));
        await db.addObservation(<String, Object?>{
          'person_id': a,
          'emotion': 'sad',
          'emotion_conf': 0.5,
          'valence': -0.5,
          'arousal': 0.3,
          'probs': <String, double>{'sad': 0.5},
        });

        await db.mergePersons(a, b);
        expect(await db.getPerson(a), isNull);
        expect(await db.countEmbeddings(b), 1);
        expect((await db.recentObservations(personId: b)).length, 1);
      });

      test('cascade delete removes children', () async {
        final pid = await db.createPerson('Nora');
        await db.addEmbedding(pid, List<double>.filled(512, 1.0));
        await db.addObservation(<String, Object?>{
          'person_id': pid,
          'emotion': 'happy',
          'emotion_conf': 0.8,
          'valence': 0.7,
          'arousal': 0.5,
          'probs': <String, double>{'happy': 0.8},
        });

        await db.deletePersonData(pid);
        expect(await db.countEmbeddings(), 0);
        expect((await db.stats())['observations'], 0);
      });

      test('wipe all clears everything', () async {
        final pid = await db.createPerson('Layla');
        await db.addEmbedding(pid, List<double>.filled(512, 1.0));
        final sid = await db.startSession();
        await db.addObservation(<String, Object?>{
          'person_id': pid,
          'session_id': sid,
          'emotion': 'fear',
          'emotion_conf': 0.6,
          'valence': -0.6,
          'arousal': 0.8,
          'probs': <String, double>{'fear': 0.6},
        });

        final removed = await db.wipeAll();
        expect(removed['persons'], 1);
        expect(await db.stats(), <String, int>{
          'persons': 0,
          'provisional': 0,
          'embeddings': 0,
          'observations': 0,
          'sessions': 0,
        });
        expect(db.embeddingMatrix, isNull);
      });

      test('observation summary', () async {
        final pid = await db.createPerson('Yousef');
        for (final pair in <MapEntry<String, int>>[
          const MapEntry<String, int>('happy', 3),
          const MapEntry<String, int>('sad', 1),
        ]) {
          for (var i = 0; i < pair.value; i++) {
            await db.addObservation(<String, Object?>{
              'person_id': pid,
              'emotion': pair.key,
              'emotion_conf': 0.8,
              'valence': 0.0,
              'arousal': 0.5,
              'probs': <String, double>{pair.key: 0.8},
            });
          }
        }

        final summary = <String, Object?>{
          for (final row in await db.emotionSummary())
            row['emotion']! as String: row['n'],
        };
        expect(summary, <String, Object?>{'happy': 3, 'sad': 1});
      });

      test('session reports round trip', () async {
        final personId = await db.createPerson('Reported');
        await db.saveSessionReport(<String, Object?>{
          'person_id': personId,
          'person_name': 'Reported',
          'started_at': 1700000000.0,
          'ended_at': 1700000300.0,
          'duration_s': 300.0,
          'samples': 480,
          'state': 'tense',
          'confidence': 0.72,
          'dominance': 0.61,
          'valence': -0.18,
          'arousal': 0.66,
          'stability': 0.44,
          'engagement': 0.55,
          'fatigue': 0.21,
          'tension': 0.71,
          'emotion_mix': <String, double>{'anger': 0.4, 'neutral': 0.6},
          'notes': <String>['sustained tension'],
        });
        final rows = await db.sessionReports();
        expect(rows.length, 1);
        final row = rows.first;
        expect(row['state'], 'tense');
        expect(row['samples'], 480);
        expect(row['confidence'] as double, closeTo(0.72, 1e-9));
        expect(row['emotion_mix'],
            <String, double>{'anger': 0.4, 'neutral': 0.6});
        expect(row['notes'], <String>['sustained tension']);
      });

      test('session reports filter by person', () async {
        final a = await db.createPerson('A');
        final b = await db.createPerson('B');
        for (final pair in <MapEntry<int, String>>[
          MapEntry<int, String>(a, 'low'),
          MapEntry<int, String>(b, 'positive'),
          MapEntry<int, String>(a, 'calm'),
        ]) {
          await db.saveSessionReport(<String, Object?>{
            'person_id': pair.key,
            'person_name': 'x',
            'started_at': 1.0,
            'ended_at': 2.0,
            'duration_s': 1.0,
            'samples': 90,
            'state': pair.value,
            'confidence': 0.5,
          });
        }
        expect((await db.sessionReports(personId: a)).length, 2);
        expect((await db.sessionReports(personId: b)).length, 1);
        expect((await db.sessionReports()).length, 3);
      });

      test('deleting a person removes their reports', () async {
        final personId = await db.createPerson('Temporary');
        await db.saveSessionReport(<String, Object?>{
          'person_id': personId,
          'person_name': 'Temporary',
          'started_at': 1.0,
          'ended_at': 2.0,
          'duration_s': 1.0,
          'samples': 90,
          'state': 'low',
          'confidence': 0.5,
        });
        expect(await db.sessionReports(personId: personId), isNotEmpty);
        await db.deletePerson(personId);
        expect(await db.sessionReports(personId: personId), isEmpty);
      });

      test('baselines round trip through storage', () async {
        final personId = await db.createPerson('Based');
        final baseline = PersonBaseline(samples: PersonBaseline.required);
        await db.saveBaseline(personId, baseline.toMap());

        final stored = await db.loadBaseline(personId);
        expect(stored, isNotNull);
        expect(stored!['samples'], PersonBaseline.required);

        final all = await db.allBaselines();
        expect(all.containsKey(personId), isTrue);
      });

      test('observations carry the affect columns', () async {
        final personId = await db.createPerson('Affected');
        final sid = await db.startSession('t');
        await db.addObservation(<String, Object?>{
          'person_id': personId,
          'session_id': sid,
          'emotion': 'happy',
          'emotion_conf': 0.9,
          'valence': 0.8,
          'arousal': 0.6,
          'probs': <String, double>{'happy': 0.9},
          'smile_type': 'genuine',
          'engagement': 0.7,
          'mood_state': 'positive',
        });

        final row = (await db.recentObservations(limit: 1)).first;
        expect(row.containsKey('engagement'), isTrue);
        expect(row['smile_type'], isNotNull);
        expect(row['mood_state'], 'positive');
        expect(row['session_id'], sid);
      });

      test('bulk observations insert', () async {
        final personId = await db.createPerson('Bulk');
        final inserted = await db.addObservationsBulk(<Map<String, Object?>>[
          for (var i = 0; i < 4; i++)
            <String, Object?>{
              'person_id': personId,
              'emotion': 'neutral',
              'emotion_conf': 0.5,
              'valence': 0.0,
              'arousal': 0.3,
              'probs': <String, double>{'neutral': 0.5},
            },
        ]);
        expect(inserted, 4);
        expect((await db.stats())['observations'], 4);
      });

      test('sessions can be opened and closed', () async {
        final sid = await db.startSession('note');
        await db.endSession(sid, frameCount: 12);
        expect((await db.stats())['sessions'], 1);
      });

      test('list persons reports child counts', () async {
        final pid = await db.createPerson('Counted');
        await db.addEmbedding(pid, List<double>.filled(512, 0.5));
        final persons = await db.listPersons();
        expect(persons.length, 1);
        expect(persons.first['embedding_count'], 1);
        expect(persons.first['observation_count'], 0);
      });
    });
  }

  test('observation columns match the python tuple', () {
    expect(observationColumns.length, 31);
    expect(observationColumns.first, 'person_id');
    expect(observationColumns.last, 'mood_state');
    expect(observationColumns, contains('duchenne'));
    expect(observationColumns, contains('expressiveness'));
  });

  test('schema version is three', () {
    expect(schemaVersion, '3');
  });
}
