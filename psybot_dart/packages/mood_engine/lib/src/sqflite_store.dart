import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:sqflite_common/sqlite_api.dart';

import 'database.dart';

const List<List<String>> addedColumns = <List<String>>[
  <String>['observations', 'duchenne', 'REAL'],
  <String>['observations', 'smile_type', 'TEXT'],
  <String>['observations', 'compound', 'TEXT'],
  <String>['observations', 'engagement', 'REAL'],
  <String>['observations', 'fatigue', 'REAL'],
  <String>['observations', 'tension', 'REAL'],
  <String>['observations', 'volatility', 'REAL'],
  <String>['observations', 'blink_rate', 'REAL'],
  <String>['observations', 'expressiveness', 'REAL'],
  <String>['observations', 'mood_state', 'TEXT'],
  <String>['persons', 'baseline', 'TEXT'],
];

Uint8List vectorToBlob(List<double> vector) {
  final data = Float32List.fromList(vector);
  return data.buffer.asUint8List(
    data.offsetInBytes,
    data.lengthInBytes,
  );
}

List<double> blobToVector(Uint8List blob) {
  final copy = Uint8List.fromList(blob);
  return copy.buffer.asFloat32List().toList();
}

class SqfliteMoodStore implements MoodStore {
  SqfliteMoodStore(this.database);

  final Database database;

  List<List<double>>? _matrix;
  List<int> _embPersonIds = <int>[];

  @override
  Future<void> open() async {
    await database.execute('PRAGMA foreign_keys = ON');
    for (final statement in schemaStatements) {
      await database.execute(statement);
    }
    for (final column in addedColumns) {
      final existing = await database.rawQuery(
        'PRAGMA table_info(${column[0]})',
      );
      final names = <String>{
        for (final row in existing) row['name']! as String,
      };
      if (!names.contains(column[1])) {
        await database.execute(
          'ALTER TABLE ${column[0]} ADD COLUMN ${column[1]} ${column[2]}',
        );
      }
    }
    await database.rawInsert(
      'INSERT INTO meta(key, value) VALUES(?, ?) '
      'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
      <Object?>['schema_version', schemaVersion],
    );
    await refreshCache();
  }

  @override
  Future<void> close() => database.close();

  @override
  Future<int> createPerson(
    String name, {
    bool provisional = false,
    String? notes,
  }) async {
    final now = utcnow();
    return database.rawInsert(
      'INSERT INTO persons(name, is_provisional, notes, created_at, '
      'updated_at) VALUES(?,?,?,?,?)',
      <Object?>[name, provisional ? 1 : 0, notes, now, now],
    );
  }

  @override
  Future<Map<String, Object?>?> getPerson(int personId) async {
    final rows = await database.rawQuery(
      'SELECT * FROM persons WHERE id = ?',
      <Object?>[personId],
    );
    return rows.isEmpty ? null : Map<String, Object?>.from(rows.first);
  }

  @override
  Future<Map<String, Object?>?> getPersonByName(String name) async {
    final rows = await database.rawQuery(
      'SELECT * FROM persons WHERE name = ?',
      <Object?>[name],
    );
    return rows.isEmpty ? null : Map<String, Object?>.from(rows.first);
  }

  @override
  Future<int> getOrCreatePerson(
    String name, {
    bool provisional = false,
  }) async {
    final found = await getPersonByName(name);
    if (found != null) return found['id']! as int;
    return createPerson(name, provisional: provisional);
  }

  @override
  Future<List<Map<String, Object?>>> listPersons() async {
    final rows = await database.rawQuery(
      'SELECT p.*, '
      '  (SELECT COUNT(*) FROM embeddings e WHERE e.person_id = p.id) '
      '     AS embedding_count, '
      '  (SELECT COUNT(*) FROM observations o WHERE o.person_id = p.id) '
      '     AS observation_count '
      'FROM persons p ORDER BY p.is_provisional, p.name',
    );
    return <Map<String, Object?>>[
      for (final row in rows) Map<String, Object?>.from(row),
    ];
  }

  @override
  Future<void> renamePerson(int personId, String newName) async {
    await database.rawUpdate(
      'UPDATE persons SET name = ?, is_provisional = 0, '
      'updated_at = ? WHERE id = ?',
      <Object?>[newName, utcnow(), personId],
    );
  }

  @override
  Future<void> mergePersons(int sourceId, int targetId) async {
    if (sourceId == targetId) return;
    await database.rawUpdate(
      'UPDATE embeddings SET person_id = ? WHERE person_id = ?',
      <Object?>[targetId, sourceId],
    );
    await database.rawUpdate(
      'UPDATE observations SET person_id = ? WHERE person_id = ?',
      <Object?>[targetId, sourceId],
    );
    await database.rawDelete(
      'DELETE FROM persons WHERE id = ?',
      <Object?>[sourceId],
    );
    await refreshCache();
  }

  @override
  Future<void> touchPerson(int personId) async {
    await database.rawUpdate(
      'UPDATE persons SET last_seen_at = ? WHERE id = ?',
      <Object?>[utcnow(), personId],
    );
  }

  @override
  Future<void> saveBaseline(
    int personId,
    Map<String, Object?> payload,
  ) async {
    await database.rawUpdate(
      'UPDATE persons SET baseline = ? WHERE id = ?',
      <Object?>[jsonEncode(payload), personId],
    );
  }

  @override
  Future<Map<String, Object?>?> loadBaseline(int personId) async {
    final rows = await database.rawQuery(
      'SELECT baseline FROM persons WHERE id = ?',
      <Object?>[personId],
    );
    if (rows.isEmpty || rows.first['baseline'] == null) return null;
    try {
      return jsonDecode(rows.first['baseline']! as String)
          as Map<String, Object?>;
    } on Object {
      return null;
    }
  }

  @override
  Future<Map<int, Map<String, Object?>>> allBaselines() async {
    final rows = await database.rawQuery(
      'SELECT id, baseline FROM persons WHERE baseline IS NOT NULL',
    );
    final out = <int, Map<String, Object?>>{};
    for (final row in rows) {
      try {
        out[row['id']! as int] =
            jsonDecode(row['baseline']! as String) as Map<String, Object?>;
      } on Object {
        continue;
      }
    }
    return out;
  }

  @override
  Future<int> saveSessionReport(
    Map<String, Object?> report, {
    int? sessionId,
  }) async {
    return database.rawInsert(
      'INSERT INTO session_reports('
      ' session_id, person_id, person_name, started_at, ended_at,'
      ' duration_s, samples, state, confidence, dominance, valence,'
      ' arousal, stability, engagement, fatigue, tension,'
      ' emotion_mix, notes)'
      ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
      <Object?>[
        sessionId,
        report['person_id'],
        report['person_name'],
        stampOf(report['started_at']),
        stampOf(report['ended_at']),
        report['duration_s'] ?? 0.0,
        report['samples'] ?? 0,
        report['state'] ?? 'neutral',
        report['confidence'] ?? 0.0,
        report['dominance'],
        report['valence'],
        report['arousal'],
        report['stability'],
        report['engagement'],
        report['fatigue'],
        report['tension'],
        jsonEncode(report['emotion_mix'] ?? <String, double>{}),
        jsonEncode(report['notes'] ?? <String>[]),
      ],
    );
  }

  @override
  Future<List<Map<String, Object?>>> sessionReports({
    int? personId,
    int limit = 50,
  }) async {
    final sql = 'SELECT * FROM session_reports'
        '${personId != null ? ' WHERE person_id = ?' : ''}'
        ' ORDER BY ended_at DESC LIMIT ?';
    final args = personId != null
        ? <Object?>[personId, limit]
        : <Object?>[limit];
    final rows = await database.rawQuery(sql, args);
    final out = <Map<String, Object?>>[];
    for (final row in rows) {
      final item = Map<String, Object?>.from(row);
      for (final key in const <String>['emotion_mix', 'notes']) {
        final raw = item[key];
        try {
          item[key] = raw == null ? null : jsonDecode(raw as String);
        } on Object {
          item[key] = null;
        }
      }
      out.add(item);
    }
    return out;
  }

  @override
  Future<String> nextProvisionalName() async {
    final rows = await database.rawQuery(
      "SELECT name FROM persons WHERE name LIKE 'Unknown-%'",
    );
    final used = <int>{};
    for (final row in rows) {
      final tail = (row['name']! as String).split('-').last;
      final parsed = int.tryParse(tail);
      if (parsed != null) used.add(parsed);
    }
    var n = 1;
    while (used.contains(n)) {
      n += 1;
    }
    return 'Unknown-$n';
  }

  @override
  Future<void> deletePerson(int personId) async {
    await database.rawDelete(
      'DELETE FROM persons WHERE id = ?',
      <Object?>[personId],
    );
    await refreshCache();
  }

  @override
  Future<int> addEmbedding(
    int personId,
    List<double> vector, {
    String source = 'camera',
    double quality = 0.0,
    String? imagePath,
    bool refreshCache = true,
  }) async {
    final id = await database.rawInsert(
      'INSERT INTO embeddings(person_id, vector, dim, source, '
      'quality, image_path, created_at) VALUES(?,?,?,?,?,?,?)',
      <Object?>[
        personId,
        vectorToBlob(vector),
        vector.length,
        source,
        quality,
        imagePath,
        utcnow(),
      ],
    );
    if (refreshCache) await this.refreshCache();
    return id;
  }

  @override
  Future<int> pruneEmbeddings(int personId, int keep) async {
    final removed = await database.rawDelete(
      'DELETE FROM embeddings WHERE id IN ('
      '  SELECT id FROM embeddings WHERE person_id = ? '
      '  ORDER BY quality DESC, id DESC LIMIT -1 OFFSET ?)',
      <Object?>[personId, keep],
    );
    if (removed > 0) await refreshCache();
    return removed;
  }

  @override
  Future<int> countEmbeddings([int? personId]) async {
    final rows = personId == null
        ? await database.rawQuery('SELECT COUNT(*) n FROM embeddings')
        : await database.rawQuery(
            'SELECT COUNT(*) n FROM embeddings WHERE person_id = ?',
            <Object?>[personId],
          );
    return (rows.first['n']! as num).toInt();
  }

  @override
  List<List<double>>? get embeddingMatrix => _matrix;

  @override
  List<int> get embeddingPersonIds => _embPersonIds;

  @override
  Future<void> refreshCache() async {
    final rows = await database.rawQuery(
      'SELECT id, person_id, vector FROM embeddings ORDER BY id',
    );
    if (rows.isEmpty) {
      _matrix = null;
      _embPersonIds = <int>[];
      return;
    }
    final decoded = <(int, List<double>)>[
      for (final row in rows)
        (row['person_id']! as int, blobToVector(row['vector']! as Uint8List)),
    ];
    var dim = 0;
    for (final entry in decoded) {
      if (entry.$2.length > dim) dim = entry.$2.length;
    }
    final kept = <(int, List<double>)>[
      for (final entry in decoded)
        if (entry.$2.length == dim) entry,
    ];
    _matrix = <List<double>>[
      for (final entry in kept) _normalise(entry.$2),
    ];
    _embPersonIds = <int>[for (final entry in kept) entry.$1];
  }

  static List<double> _normalise(List<double> vector) {
    var total = 0.0;
    for (final value in vector) {
      total += value * value;
    }
    var norm = math.sqrt(total);
    if (norm == 0.0) norm = 1.0;
    return <double>[for (final value in vector) value / norm];
  }

  @override
  Future<int> startSession([String? note]) => database.rawInsert(
    'INSERT INTO sessions(started_at, note) VALUES(?,?)',
    <Object?>[utcnow(), note],
  );

  @override
  Future<void> endSession(int sessionId, {int frameCount = 0}) async {
    await database.rawUpdate(
      'UPDATE sessions SET ended_at = ?, frame_count = ? WHERE id = ?',
      <Object?>[utcnow(), frameCount, sessionId],
    );
  }

  @override
  Future<int> addObservation(Map<String, Object?> values) async {
    final row = Map<String, Object?>.from(values);
    final probs = row['probs'];
    if (probs is Map) row['probs'] = jsonEncode(probs);
    row.putIfAbsent('ts', utcnow);
    row.putIfAbsent('is_spoof', () => 0);
    final placeholders = List<String>.filled(
      observationColumns.length,
      '?',
    ).join(',');
    return database.rawInsert(
      'INSERT INTO observations(${observationColumns.join(',')}) '
      'VALUES($placeholders)',
      <Object?>[for (final column in observationColumns) row[column]],
    );
  }

  @override
  Future<int> addObservationsBulk(List<Map<String, Object?>> rows) async {
    if (rows.isEmpty) return 0;
    final batch = database.batch();
    final placeholders = List<String>.filled(
      observationColumns.length,
      '?',
    ).join(',');
    for (final values in rows) {
      final row = Map<String, Object?>.from(values);
      final probs = row['probs'];
      if (probs is Map) row['probs'] = jsonEncode(probs);
      row.putIfAbsent('ts', utcnow);
      row.putIfAbsent('is_spoof', () => 0);
      batch.rawInsert(
        'INSERT INTO observations(${observationColumns.join(',')}) '
        'VALUES($placeholders)',
        <Object?>[for (final column in observationColumns) row[column]],
      );
    }
    await batch.commit(noResult: true);
    return rows.length;
  }

  @override
  Future<List<Map<String, Object?>>> recentObservations({
    int? personId,
    int limit = 100,
  }) async {
    final rows = personId == null
        ? await database.rawQuery(
            'SELECT o.*, p.name AS person_name FROM observations o '
            'LEFT JOIN persons p ON p.id = o.person_id '
            'ORDER BY o.id DESC LIMIT ?',
            <Object?>[limit],
          )
        : await database.rawQuery(
            'SELECT o.*, p.name AS person_name FROM observations o '
            'LEFT JOIN persons p ON p.id = o.person_id '
            'WHERE o.person_id = ? ORDER BY o.id DESC LIMIT ?',
            <Object?>[personId, limit],
          );
    return <Map<String, Object?>>[
      for (final row in rows) Map<String, Object?>.from(row),
    ];
  }

  @override
  Future<List<Map<String, Object?>>> emotionSummary({
    int? personId,
    String? since,
  }) async {
    var sql = 'SELECT emotion, COUNT(*) AS n, AVG(emotion_conf) AS avg_conf, '
        'AVG(valence) AS avg_valence, AVG(arousal) AS avg_arousal '
        'FROM observations WHERE 1=1';
    final args = <Object?>[];
    if (personId != null) {
      sql += ' AND person_id = ?';
      args.add(personId);
    }
    if (since != null) {
      sql += ' AND ts >= ?';
      args.add(since);
    }
    sql += ' GROUP BY emotion ORDER BY n DESC';
    final rows = await database.rawQuery(sql, args);
    return <Map<String, Object?>>[
      for (final row in rows) Map<String, Object?>.from(row),
    ];
  }

  Future<int> _count(String sql) async {
    final rows = await database.rawQuery(sql);
    return (rows.first['n']! as num).toInt();
  }

  @override
  Future<Map<String, int>> wipeAll() async {
    final counts = <String, int>{
      'persons': await _count('SELECT COUNT(*) n FROM persons'),
      'embeddings': await _count('SELECT COUNT(*) n FROM embeddings'),
      'observations': await _count('SELECT COUNT(*) n FROM observations'),
      'sessions': await _count('SELECT COUNT(*) n FROM sessions'),
    };
    await database.execute('DELETE FROM observations');
    await database.execute('DELETE FROM embeddings');
    await database.execute('DELETE FROM sessions');
    await database.execute('DELETE FROM persons');
    await database.execute('DELETE FROM sqlite_sequence');
    await database.execute('VACUUM');
    await refreshCache();
    return counts;
  }

  @override
  Future<Map<String, int>> deletePersonData(int personId) async {
    final rows = await database.rawQuery(
      'SELECT COUNT(*) n FROM observations WHERE person_id = ?',
      <Object?>[personId],
    );
    final counts = <String, int>{
      'embeddings': await countEmbeddings(personId),
      'observations': (rows.first['n']! as num).toInt(),
    };
    await deletePerson(personId);
    return counts;
  }

  @override
  Future<Map<String, int>> stats() async => <String, int>{
    'persons': await _count('SELECT COUNT(*) n FROM persons'),
    'provisional': await _count(
      'SELECT COUNT(*) n FROM persons WHERE is_provisional = 1',
    ),
    'embeddings': await _count('SELECT COUNT(*) n FROM embeddings'),
    'observations': await _count('SELECT COUNT(*) n FROM observations'),
    'sessions': await _count('SELECT COUNT(*) n FROM sessions'),
  };
}
