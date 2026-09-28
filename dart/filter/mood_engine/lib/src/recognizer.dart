import 'dart:math' as math;

import 'config.dart';
import 'database.dart';
import 'maths.dart';

class MatchResult {
  const MatchResult({
    required this.personId,
    required this.name,
    required this.score,
    required this.isNew,
    required this.isUncertain,
    this.runnerUp,
  });

  final int? personId;
  final String? name;
  final double score;
  final bool isNew;
  final bool isUncertain;
  final MapEntry<String, double>? runnerUp;

  bool get matched => personId != null && !isNew;
}

abstract class FaceEmbeddingModel {
  List<double> embed(Object image);

  int get dimensions;

  String get name;
}

class FakeFaceEmbeddingModel implements FaceEmbeddingModel {
  FakeFaceEmbeddingModel({this.dimensions = 512, List<double>? vector})
    : _vector = vector;

  List<double>? _vector;

  @override
  final int dimensions;

  @override
  String get name => 'fake';

  void setVector(List<double> vector) => _vector = vector;

  @override
  List<double> embed(Object image) =>
      _vector ?? List<double>.filled(dimensions, 0.0);
}

List<double> unitNormalise(List<double> vector) {
  final norm = vectorNorm(vector);
  if (norm == 0.0) return vector;
  return <double>[for (final value in vector) value / norm];
}

double vectorNorm(List<double> vector) {
  var total = 0.0;
  for (final value in vector) {
    total += value * value;
  }
  return math.sqrt(total);
}

double dotProduct(List<double> a, List<double> b) {
  var total = 0.0;
  for (var i = 0; i < a.length; i++) {
    total += a[i] * b[i];
  }
  return total;
}

double cosineSimilarity(List<double> a, List<double> b) {
  final na = vectorNorm(a);
  final nb = vectorNorm(b);
  if (na == 0.0 || nb == 0.0) return 0.0;
  return dotProduct(a, b) / (na * nb);
}

double cosineDistance(List<double> a, List<double> b) =>
    1.0 - cosineSimilarity(a, b);

class FaceRecognizer {
  FaceRecognizer(this.db, {RecognitionConfig? config})
    : config = config ?? const RecognitionConfig();

  final MoodStore db;
  final RecognitionConfig config;

  Future<MatchResult> match(List<double> embedding, {int topK = 3}) async {
    final matrix = db.embeddingMatrix;
    if (matrix == null || matrix.isEmpty) {
      return const MatchResult(
        personId: null,
        name: null,
        score: 0.0,
        isNew: true,
        isUncertain: false,
      );
    }

    final norm = vectorNorm(embedding);
    if (norm == 0.0) {
      return const MatchResult(
        personId: null,
        name: null,
        score: 0.0,
        isNew: true,
        isUncertain: false,
      );
    }
    final query = <double>[for (final value in embedding) value / norm];

    if (query.length != matrix.first.length) {
      return const MatchResult(
        personId: null,
        name: null,
        score: 0.0,
        isNew: true,
        isUncertain: false,
      );
    }

    final personIds = db.embeddingPersonIds;
    final grouped = <int, List<double>>{};
    for (var i = 0; i < matrix.length; i++) {
      grouped
          .putIfAbsent(personIds[i], () => <double>[])
          .add(dotProduct(matrix[i], query));
    }

    final best = <int, double>{};
    for (final entry in grouped.entries) {
      final sorted = List<double>.from(entry.value)..sort();
      final k = math.min(topK, sorted.length);
      best[entry.key] = mean(sorted.sublist(sorted.length - k));
    }

    final ranked = rankedDescending(best);
    final topId = ranked.first.key;
    final topScore = ranked.first.value;

    MapEntry<String, double>? runnerUp;
    if (ranked.length > 1) {
      final second = await db.getPerson(ranked[1].key);
      if (second != null) {
        runnerUp = MapEntry<String, double>(
          second['name'].toString(),
          ranked[1].value,
        );
      }
    }

    final isNew = topScore < config.newThreshold;
    final isUncertain = !isNew && topScore < config.matchThreshold;

    if (isNew) {
      return MatchResult(
        personId: null,
        name: null,
        score: topScore,
        isNew: true,
        isUncertain: false,
        runnerUp: runnerUp,
      );
    }

    final person = await db.getPerson(topId);
    return MatchResult(
      personId: topId,
      name: person?['name']?.toString(),
      score: topScore,
      isNew: false,
      isUncertain: isUncertain,
      runnerUp: runnerUp,
    );
  }

  Future<int> enrol(
    String name,
    List<double> embedding, {
    String source = 'camera',
    double quality = 0.0,
    String? imagePath,
    bool provisional = false,
  }) async {
    final personId = await db.getOrCreatePerson(name, provisional: provisional);
    await db.addEmbedding(
      personId,
      embedding,
      source: source,
      quality: quality,
      imagePath: imagePath,
    );
    await _enforceLimit(personId);
    return personId;
  }

  Future<(int, String)> enrolProvisional(
    List<double> embedding, {
    double quality = 0.0,
    String? imagePath,
  }) async {
    final name = await db.nextProvisionalName();
    final personId = await db.createPerson(name, provisional: true);
    await db.addEmbedding(
      personId,
      embedding,
      source: 'auto',
      quality: quality,
      imagePath: imagePath,
    );
    return (personId, name);
  }

  Future<bool> maybeAutoEnrol(
    int personId,
    List<double> embedding,
    double score,
    double quality,
  ) async {
    if (score < config.autoEnrolLow || score > config.autoEnrolHigh) {
      return false;
    }
    if (quality < 0.55) return false;
    if (await db.countEmbeddings(personId) >=
        config.maxEmbeddingsPerPerson) {
      return false;
    }
    await db.addEmbedding(
      personId,
      embedding,
      source: 'auto',
      quality: quality,
    );
    return true;
  }

  Future<void> _enforceLimit(int personId) async {
    final limit = config.maxEmbeddingsPerPerson;
    if (await db.countEmbeddings(personId) > limit) {
      await db.pruneEmbeddings(personId, limit);
    }
  }

  Future<List<MatchResult>> identifyAll(List<List<double>> embeddings) async {
    final out = <MatchResult>[];
    for (final embedding in embeddings) {
      out.add(await match(embedding));
    }
    return out;
  }

  double similarity(List<double> a, List<double> b) => cosineSimilarity(a, b);
}
