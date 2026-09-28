import 'dart:convert';

const String schemaVersion = '3';

const List<String> observationColumns = <String>[
  'person_id',
  'session_id',
  'ts',
  'emotion',
  'emotion_conf',
  'valence',
  'arousal',
  'probs',
  'age',
  'gender',
  'yaw',
  'pitch',
  'roll',
  'gaze_x',
  'gaze_y',
  'attention',
  'liveness',
  'is_spoof',
  'match_score',
  'det_score',
  'snapshot_path',
  'duchenne',
  'smile_type',
  'compound',
  'engagement',
  'fatigue',
  'tension',
  'volatility',
  'blink_rate',
  'expressiveness',
  'mood_state',
];

const List<String> schemaStatements = <String>[
  '''
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
)''',
  '''
CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    is_provisional  INTEGER NOT NULL DEFAULT 0,
    notes           TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    last_seen_at    TEXT,
    baseline        TEXT
)''',
  '''
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BLOB    NOT NULL,
    dim         INTEGER NOT NULL,
    source      TEXT    NOT NULL,
    quality     REAL    NOT NULL DEFAULT 0,
    image_path  TEXT,
    created_at  TEXT    NOT NULL
)''',
  'CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id)',
  '''
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    frame_count  INTEGER NOT NULL DEFAULT 0,
    note         TEXT
)''',
  '''
CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id       INTEGER REFERENCES persons(id) ON DELETE CASCADE,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    ts              TEXT    NOT NULL,
    emotion         TEXT    NOT NULL,
    emotion_conf    REAL    NOT NULL,
    valence         REAL    NOT NULL,
    arousal         REAL    NOT NULL,
    probs           TEXT    NOT NULL,
    age             REAL,
    gender          TEXT,
    yaw             REAL,
    pitch           REAL,
    roll            REAL,
    gaze_x          REAL,
    gaze_y          REAL,
    attention       REAL,
    liveness        REAL,
    is_spoof        INTEGER NOT NULL DEFAULT 0,
    match_score     REAL,
    det_score       REAL,
    snapshot_path   TEXT,
    duchenne        REAL,
    mood_state      TEXT,
    smile_type      TEXT,
    compound        TEXT,
    engagement      REAL,
    fatigue         REAL,
    tension         REAL,
    volatility      REAL,
    blink_rate      REAL,
    expressiveness  REAL
)''',
  'CREATE INDEX IF NOT EXISTS idx_obs_person_ts ON observations(person_id, ts)',
  'CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts)',
  '''
CREATE TABLE IF NOT EXISTS session_reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    person_id    INTEGER REFERENCES persons(id) ON DELETE CASCADE,
    person_name  TEXT,
    started_at   TEXT NOT NULL,
    ended_at     TEXT NOT NULL,
    duration_s   REAL NOT NULL,
    samples      INTEGER NOT NULL,
    state        TEXT NOT NULL,
    confidence   REAL NOT NULL,
    dominance    REAL,
    valence      REAL,
    arousal      REAL,
    stability    REAL,
    engagement   REAL,
    fatigue      REAL,
    tension      REAL,
    emotion_mix  TEXT,
    notes        TEXT
)''',
  'CREATE INDEX IF NOT EXISTS idx_reports_person '
      'ON session_reports(person_id, ended_at)',
  'CREATE INDEX IF NOT EXISTS idx_reports_ended ON session_reports(ended_at)',
];

String utcnow() {
  final now = DateTime.now().toUtc();
  final iso = now.toIso8601String();
  final trimmed = iso.substring(0, 19);
  return '$trimmed+00:00';
}

String stampOf(Object? value) {
  final seconds = (value as num?)?.toDouble();
  if (seconds == null || !seconds.isFinite) {
    return DateTime.now().toUtc().toIso8601String();
  }
  return DateTime.fromMillisecondsSinceEpoch(
    (seconds * 1000).round(),
    isUtc: true,
  ).toIso8601String();
}

abstract class MoodStore {
  Future<void> open();

  Future<void> close();

  Future<int> createPerson(
    String name, {
    bool provisional = false,
    String? notes,
  });

  Future<Map<String, Object?>?> getPerson(int personId);

  Future<Map<String, Object?>?> getPersonByName(String name);

  Future<int> getOrCreatePerson(String name, {bool provisional = false});

  Future<List<Map<String, Object?>>> listPersons();

  Future<void> renamePerson(int personId, String newName);

  Future<void> mergePersons(int sourceId, int targetId);

  Future<void> touchPerson(int personId);

  Future<void> saveBaseline(int personId, Map<String, Object?> payload);

  Future<Map<String, Object?>?> loadBaseline(int personId);

  Future<Map<int, Map<String, Object?>>> allBaselines();

  Future<int> saveSessionReport(
    Map<String, Object?> report, {
    int? sessionId,
  });

  Future<List<Map<String, Object?>>> sessionReports({
    int? personId,
    int limit = 50,
  });

  Future<String> nextProvisionalName();

  Future<void> deletePerson(int personId);

  Future<int> addEmbedding(
    int personId,
    List<double> vector, {
    String source = 'camera',
    double quality = 0.0,
    String? imagePath,
    bool refreshCache = true,
  });

  Future<int> pruneEmbeddings(int personId, int keep);

  Future<int> countEmbeddings([int? personId]);

  List<List<double>>? get embeddingMatrix;

  List<int> get embeddingPersonIds;

  Future<void> refreshCache();

  Future<int> startSession([String? note]);

  Future<void> endSession(int sessionId, {int frameCount = 0});

  Future<int> addObservation(Map<String, Object?> values);

  Future<int> addObservationsBulk(List<Map<String, Object?>> rows);

  Future<List<Map<String, Object?>>> recentObservations({
    int? personId,
    int limit = 100,
  });

  Future<List<Map<String, Object?>>> emotionSummary({
    int? personId,
    String? since,
  });

  Future<Map<String, int>> wipeAll();

  Future<Map<String, int>> deletePersonData(int personId);

  Future<Map<String, int>> stats();
}

class InMemoryMoodStore implements MoodStore {
  final List<Map<String, Object?>> _persons = <Map<String, Object?>>[];
  final List<Map<String, Object?>> _embeddings = <Map<String, Object?>>[];
  final List<Map<String, Object?>> _observations = <Map<String, Object?>>[];
  final List<Map<String, Object?>> _sessions = <Map<String, Object?>>[];
  final List<Map<String, Object?>> _reports = <Map<String, Object?>>[];

  int _personSeq = 0;
  int _embeddingSeq = 0;
  int _observationSeq = 0;
  int _sessionSeq = 0;
  int _reportSeq = 0;

  List<List<double>>? _matrix;
  List<int> _embPersonIds = <int>[];

  @override
  Future<void> open() async {}

  @override
  Future<void> close() async {}

  @override
  Future<int> createPerson(
    String name, {
    bool provisional = false,
    String? notes,
  }) async {
    if (_persons.any((Map<String, Object?> p) => p['name'] == name)) {
      throw StateError('UNIQUE constraint failed: persons.name');
    }
    final now = utcnow();
    _personSeq += 1;
    _persons.add(<String, Object?>{
      'id': _personSeq,
      'name': name,
      'is_provisional': provisional ? 1 : 0,
      'notes': notes,
      'created_at': now,
      'updated_at': now,
      'last_seen_at': null,
      'baseline': null,
    });
    return _personSeq;
  }

  Map<String, Object?>? _person(int personId) {
    for (final person in _persons) {
      if (person['id'] == personId) return person;
    }
    return null;
  }

  @override
  Future<Map<String, Object?>?> getPerson(int personId) async {
    final person = _person(personId);
    return person == null ? null : Map<String, Object?>.from(person);
  }

  @override
  Future<Map<String, Object?>?> getPersonByName(String name) async {
    for (final person in _persons) {
      if (person['name'] == name) return Map<String, Object?>.from(person);
    }
    return null;
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
    final out = <Map<String, Object?>>[
      for (final person in _persons)
        <String, Object?>{
          ...person,
          'embedding_count': _embeddings
              .where((Map<String, Object?> e) => e['person_id'] == person['id'])
              .length,
          'observation_count': _observations
              .where((Map<String, Object?> o) => o['person_id'] == person['id'])
              .length,
        },
    ];
    out.sort((Map<String, Object?> a, Map<String, Object?> b) {
      final byProvisional = (a['is_provisional']! as int)
          .compareTo(b['is_provisional']! as int);
      if (byProvisional != 0) return byProvisional;
      return (a['name']! as String).compareTo(b['name']! as String);
    });
    return out;
  }

  @override
  Future<void> renamePerson(int personId, String newName) async {
    final person = _person(personId);
    if (person == null) return;
    person['name'] = newName;
    person['is_provisional'] = 0;
    person['updated_at'] = utcnow();
  }

  @override
  Future<void> mergePersons(int sourceId, int targetId) async {
    if (sourceId == targetId) return;
    for (final embedding in _embeddings) {
      if (embedding['person_id'] == sourceId) {
        embedding['person_id'] = targetId;
      }
    }
    for (final observation in _observations) {
      if (observation['person_id'] == sourceId) {
        observation['person_id'] = targetId;
      }
    }
    _persons.removeWhere((Map<String, Object?> p) => p['id'] == sourceId);
    await refreshCache();
  }

  @override
  Future<void> touchPerson(int personId) async {
    _person(personId)?['last_seen_at'] = utcnow();
  }

  @override
  Future<void> saveBaseline(
    int personId,
    Map<String, Object?> payload,
  ) async {
    _person(personId)?['baseline'] = jsonEncode(payload);
  }

  @override
  Future<Map<String, Object?>?> loadBaseline(int personId) async {
    final raw = _person(personId)?['baseline'];
    if (raw == null) return null;
    try {
      return jsonDecode(raw as String) as Map<String, Object?>;
    } on Object {
      return null;
    }
  }

  @override
  Future<Map<int, Map<String, Object?>>> allBaselines() async {
    final out = <int, Map<String, Object?>>{};
    for (final person in _persons) {
      final raw = person['baseline'];
      if (raw == null) continue;
      try {
        out[person['id']! as int] =
            jsonDecode(raw as String) as Map<String, Object?>;
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
    _reportSeq += 1;
    _reports.add(<String, Object?>{
      'id': _reportSeq,
      'session_id': sessionId,
      'person_id': report['person_id'],
      'person_name': report['person_name'],
      'started_at': stampOf(report['started_at']),
      'ended_at': stampOf(report['ended_at']),
      'duration_s': report['duration_s'] ?? 0.0,
      'samples': report['samples'] ?? 0,
      'state': report['state'] ?? 'neutral',
      'confidence': report['confidence'] ?? 0.0,
      'dominance': report['dominance'],
      'valence': report['valence'],
      'arousal': report['arousal'],
      'stability': report['stability'],
      'engagement': report['engagement'],
      'fatigue': report['fatigue'],
      'tension': report['tension'],
      'emotion_mix': jsonEncode(report['emotion_mix'] ?? <String, double>{}),
      'notes': jsonEncode(report['notes'] ?? <String>[]),
    });
    return _reportSeq;
  }

  @override
  Future<List<Map<String, Object?>>> sessionReports({
    int? personId,
    int limit = 50,
  }) async {
    final rows = <Map<String, Object?>>[
      for (final report in _reports)
        if (personId == null || report['person_id'] == personId)
          Map<String, Object?>.from(report),
    ];
    rows.sort((Map<String, Object?> a, Map<String, Object?> b) =>
        (b['ended_at']! as String).compareTo(a['ended_at']! as String));
    final limited = rows.length > limit ? rows.sublist(0, limit) : rows;
    for (final row in limited) {
      for (final key in const <String>['emotion_mix', 'notes']) {
        final raw = row[key];
        try {
          row[key] = raw == null ? null : jsonDecode(raw as String);
        } on Object {
          row[key] = null;
        }
      }
    }
    return limited;
  }

  @override
  Future<String> nextProvisionalName() async {
    final used = <int>{};
    for (final person in _persons) {
      final name = person['name']! as String;
      if (!name.startsWith('Unknown-')) continue;
      final tail = name.split('-').last;
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
    _persons.removeWhere((Map<String, Object?> p) => p['id'] == personId);
    _embeddings.removeWhere(
      (Map<String, Object?> e) => e['person_id'] == personId,
    );
    _observations.removeWhere(
      (Map<String, Object?> o) => o['person_id'] == personId,
    );
    _reports.removeWhere((Map<String, Object?> r) => r['person_id'] == personId);
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
    _embeddingSeq += 1;
    _embeddings.add(<String, Object?>{
      'id': _embeddingSeq,
      'person_id': personId,
      'vector': List<double>.from(vector),
      'dim': vector.length,
      'source': source,
      'quality': quality,
      'image_path': imagePath,
      'created_at': utcnow(),
    });
    if (refreshCache) await this.refreshCache();
    return _embeddingSeq;
  }

  @override
  Future<int> pruneEmbeddings(int personId, int keep) async {
    final mine = <Map<String, Object?>>[
      for (final embedding in _embeddings)
        if (embedding['person_id'] == personId) embedding,
    ];
    mine.sort((Map<String, Object?> a, Map<String, Object?> b) {
      final byQuality =
          (b['quality']! as num).compareTo(a['quality']! as num);
      if (byQuality != 0) return byQuality;
      return (b['id']! as int).compareTo(a['id']! as int);
    });
    if (mine.length <= keep) return 0;
    final doomed = mine.sublist(keep);
    for (final embedding in doomed) {
      _embeddings.remove(embedding);
    }
    if (doomed.isNotEmpty) await refreshCache();
    return doomed.length;
  }

  @override
  Future<int> countEmbeddings([int? personId]) async {
    if (personId == null) return _embeddings.length;
    return _embeddings
        .where((Map<String, Object?> e) => e['person_id'] == personId)
        .length;
  }

  @override
  List<List<double>>? get embeddingMatrix => _matrix;

  @override
  List<int> get embeddingPersonIds => _embPersonIds;

  @override
  Future<void> refreshCache() async {
    if (_embeddings.isEmpty) {
      _matrix = null;
      _embPersonIds = <int>[];
      return;
    }
    final sorted = List<Map<String, Object?>>.from(_embeddings)
      ..sort((Map<String, Object?> a, Map<String, Object?> b) =>
          (a['id']! as int).compareTo(b['id']! as int));
    var dim = 0;
    for (final embedding in sorted) {
      final size = (embedding['vector']! as List<double>).length;
      if (size > dim) dim = size;
    }
    final kept = <Map<String, Object?>>[
      for (final embedding in sorted)
        if ((embedding['vector']! as List<double>).length == dim) embedding,
    ];
    _matrix = <List<double>>[
      for (final embedding in kept)
        _normalise(embedding['vector']! as List<double>),
    ];
    _embPersonIds = <int>[
      for (final embedding in kept) embedding['person_id']! as int,
    ];
  }

  static List<double> _normalise(List<double> vector) {
    var total = 0.0;
    for (final value in vector) {
      total += value * value;
    }
    var norm = total <= 0 ? 1.0 : _sqrt(total);
    if (norm == 0.0) norm = 1.0;
    return <double>[for (final value in vector) value / norm];
  }

  static double _sqrt(double value) {
    var guess = value;
    var previous = 0.0;
    while ((guess - previous).abs() > 1e-15 * (guess.abs() + 1)) {
      previous = guess;
      guess = 0.5 * (guess + value / guess);
    }
    return guess;
  }

  @override
  Future<int> startSession([String? note]) async {
    _sessionSeq += 1;
    _sessions.add(<String, Object?>{
      'id': _sessionSeq,
      'started_at': utcnow(),
      'ended_at': null,
      'frame_count': 0,
      'note': note,
    });
    return _sessionSeq;
  }

  @override
  Future<void> endSession(int sessionId, {int frameCount = 0}) async {
    for (final session in _sessions) {
      if (session['id'] == sessionId) {
        session['ended_at'] = utcnow();
        session['frame_count'] = frameCount;
      }
    }
  }

  @override
  Future<int> addObservation(Map<String, Object?> values) async {
    final row = Map<String, Object?>.from(values);
    final probs = row['probs'];
    if (probs is Map) row['probs'] = jsonEncode(probs);
    row.putIfAbsent('ts', utcnow);
    row.putIfAbsent('is_spoof', () => 0);
    _observationSeq += 1;
    final stored = <String, Object?>{'id': _observationSeq};
    for (final column in observationColumns) {
      stored[column] = row[column];
    }
    _observations.add(stored);
    return _observationSeq;
  }

  @override
  Future<int> addObservationsBulk(List<Map<String, Object?>> rows) async {
    for (final row in rows) {
      await addObservation(row);
    }
    return rows.length;
  }

  @override
  Future<List<Map<String, Object?>>> recentObservations({
    int? personId,
    int limit = 100,
  }) async {
    final rows = <Map<String, Object?>>[
      for (final observation in _observations)
        if (personId == null || observation['person_id'] == personId)
          <String, Object?>{
            ...observation,
            'person_name': observation['person_id'] == null
                ? null
                : _person(observation['person_id']! as int)?['name'],
          },
    ];
    rows.sort((Map<String, Object?> a, Map<String, Object?> b) =>
        (b['id']! as int).compareTo(a['id']! as int));
    return rows.length > limit ? rows.sublist(0, limit) : rows;
  }

  @override
  Future<List<Map<String, Object?>>> emotionSummary({
    int? personId,
    String? since,
  }) async {
    final grouped = <String, List<Map<String, Object?>>>{};
    for (final observation in _observations) {
      if (personId != null && observation['person_id'] != personId) continue;
      final ts = observation['ts'] as String?;
      if (since != null && (ts == null || ts.compareTo(since) < 0)) continue;
      grouped
          .putIfAbsent(observation['emotion']! as String, () => [])
          .add(observation);
    }
    final out = <Map<String, Object?>>[
      for (final entry in grouped.entries)
        <String, Object?>{
          'emotion': entry.key,
          'n': entry.value.length,
          'avg_conf': _average(entry.value, 'emotion_conf'),
          'avg_valence': _average(entry.value, 'valence'),
          'avg_arousal': _average(entry.value, 'arousal'),
        },
    ];
    out.sort((Map<String, Object?> a, Map<String, Object?> b) =>
        (b['n']! as int).compareTo(a['n']! as int));
    return out;
  }

  static double _average(List<Map<String, Object?>> rows, String key) {
    var total = 0.0;
    var count = 0;
    for (final row in rows) {
      final value = (row[key] as num?)?.toDouble();
      if (value == null) continue;
      total += value;
      count++;
    }
    return count == 0 ? 0.0 : total / count;
  }

  @override
  Future<Map<String, int>> wipeAll() async {
    final counts = <String, int>{
      'persons': _persons.length,
      'embeddings': _embeddings.length,
      'observations': _observations.length,
      'sessions': _sessions.length,
    };
    _observations.clear();
    _embeddings.clear();
    _sessions.clear();
    _persons.clear();
    _reports.clear();
    _personSeq = 0;
    _embeddingSeq = 0;
    _observationSeq = 0;
    _sessionSeq = 0;
    _reportSeq = 0;
    await refreshCache();
    return counts;
  }

  @override
  Future<Map<String, int>> deletePersonData(int personId) async {
    final counts = <String, int>{
      'embeddings': await countEmbeddings(personId),
      'observations': _observations
          .where((Map<String, Object?> o) => o['person_id'] == personId)
          .length,
    };
    await deletePerson(personId);
    return counts;
  }

  @override
  Future<Map<String, int>> stats() async => <String, int>{
    'persons': _persons.length,
    'provisional': _persons
        .where((Map<String, Object?> p) => p['is_provisional'] == 1)
        .length,
    'embeddings': _embeddings.length,
    'observations': _observations.length,
    'sessions': _sessions.length,
  };
}
