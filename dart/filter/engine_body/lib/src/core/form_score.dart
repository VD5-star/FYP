import 'dart:math' as math;

class RepetitionRecord {
  const RepetitionRecord({
    required this.angles,
    required this.startTime,
    required this.endTime,
  });

  final List<double> angles;
  final Duration startTime;
  final Duration endTime;

  Duration get duration => endTime - startTime;

  double? get depth =>
      angles.isEmpty ? null : angles.reduce((a, b) => a < b ? a : b);

  double? get rangeOfMotion {
    if (angles.isEmpty) return null;
    final lo = angles.reduce((a, b) => a < b ? a : b);
    final hi = angles.reduce((a, b) => a > b ? a : b);
    return hi - lo;
  }
}

class FormReport {
  const FormReport({
    required this.similarity,
    this.depthDifference,
    this.tempoRatio,
    this.notes = const [],
  });

  final double similarity;

  final double? depthDifference;

  final double? tempoRatio;

  final List<String> notes;
}

List<double> resampleTrajectory(List<double> sequence, {int length = 32}) {
  if (sequence.isEmpty) return List<double>.filled(length, 0);
  if (sequence.length == 1) {
    return List<double>.filled(length, sequence.first);
  }

  final out = List<double>.filled(length, 0);
  for (var i = 0; i < length; i++) {
    final position = i * (sequence.length - 1) / (length - 1);
    final lower = position.floor();
    final upper = math.min(lower + 1, sequence.length - 1);
    final weight = position - lower;
    out[i] = sequence[lower] * (1 - weight) + sequence[upper] * weight;
  }
  return out;
}

double dtwDistance(List<double> a, List<double> b, {int? band}) {
  final n = a.length;
  final m = b.length;
  if (n == 0 || m == 0) return double.infinity;

  final width = band ?? math.max(4, math.max(n, m) ~/ 4);

  final cost = List<List<double>>.generate(
    n + 1,
    (_) => List<double>.filled(m + 1, double.infinity),
  );
  cost[0][0] = 0;

  for (var i = 1; i <= n; i++) {
    final lo = math.max(1, i - width);
    final hi = math.min(m, i + width);
    for (var j = lo; j <= hi; j++) {
      final d = (a[i - 1] - b[j - 1]).abs();
      final best = math.min(
        cost[i - 1][j],
        math.min(cost[i][j - 1], cost[i - 1][j - 1]),
      );
      cost[i][j] = d + best;
    }
  }

  final result = cost[n][m];
  if (!result.isFinite) return double.infinity;
  return result / (n + m);
}

class FormAnalyser {
  FormAnalyser({RepetitionRecord? reference, this.resampleLength = 32})
      : _reference = reference,
        _skippedFirst = reference != null {
    if (reference != null) {
      _referenceShape =
          resampleTrajectory(reference.angles, length: resampleLength);
    }
  }

  final int resampleLength;

  static const notableDifference = 0.25;

  static const notableDepth = 20.0;

  static const discardFirst = true;

  RepetitionRecord? _reference;
  List<double>? _referenceShape;
  bool _skippedFirst;
  final List<RepetitionRecord> history = [];

  RepetitionRecord? get reference => _reference;

  FormReport? add(RepetitionRecord record) {
    if (record.angles.isEmpty) return null;
    history.add(record);

    if (_reference == null) {
      if (discardFirst && !_skippedFirst) {
        _skippedFirst = true;
        return null;
      }
      _reference = record;
      _referenceShape =
          resampleTrajectory(record.angles, length: resampleLength);
      return null;
    }

    final shape = resampleTrajectory(record.angles, length: resampleLength);
    final distance = dtwDistance(shape, _referenceShape!);

    final scale = math.max(_reference!.rangeOfMotion ?? 1.0, 1e-6);
    final similarity = (1.0 - distance / scale).clamp(0.0, 1.0);

    final notes = <String>[];

    double? depthDifference;
    final depth = record.depth;
    final referenceDepth = _reference!.depth;
    if (depth != null && referenceDepth != null) {
      depthDifference = depth - referenceDepth;
      if (depthDifference > notableDepth) {
        notes.add('shallower than your first repetition');
      } else if (depthDifference < -notableDepth) {
        notes.add('deeper than your first repetition');
      }
    }

    double? tempoRatio;
    final referenceMicros = _reference!.duration.inMicroseconds;
    if (referenceMicros > 0) {
      tempoRatio = record.duration.inMicroseconds / referenceMicros;
      if (tempoRatio > 1.5) {
        notes.add('slower than your first repetition');
      } else if (tempoRatio < 0.67) {
        notes.add('faster than your first repetition');
      }
    }

    if (similarity < (1.0 - notableDifference) && notes.isEmpty) {
      notes.add('moved differently from your first repetition');
    }

    return FormReport(
      similarity: similarity,
      depthDifference: depthDifference,
      tempoRatio: tempoRatio,
      notes: notes,
    );
  }

  double? get consistency {
    if (history.length < 2) return null;
    final referenceShape = _referenceShape;
    if (referenceShape == null) return null;

    final scored = discardFirst ? history.skip(1) : history;
    if (scored.length < 2) return null;

    final scale = math.max(_reference?.rangeOfMotion ?? 1.0, 1e-6);
    final scores = <double>[];
    for (final record in scored.skip(1)) {
      final shape =
          resampleTrajectory(record.angles, length: resampleLength);
      scores.add((1.0 - dtwDistance(shape, referenceShape) / scale)
          .clamp(0.0, 1.0));
    }
    if (scores.isEmpty) return null;
    return scores.reduce((a, b) => a + b) / scores.length;
  }

  void reset() {
    _reference = null;
    _referenceShape = null;
    _skippedFirst = false;
    history.clear();
  }
}

class RepetitionCollector {
  final List<double> _angles = [];
  Duration? _start;

  void add(double? angle, Duration timestamp) {
    if (angle == null) return;
    _start ??= timestamp;
    _angles.add(angle);
  }

  RepetitionRecord? close(Duration timestamp) {
    final start = _start;
    if (_angles.length < 8 || start == null) {
      _angles.clear();
      _start = null;
      return null;
    }

    final record = RepetitionRecord(
      angles: List<double>.from(_angles),
      startTime: start,
      endTime: timestamp,
    );
    _angles.clear();
    _start = timestamp;
    return record;
  }

  void reset() {
    _angles.clear();
    _start = null;
  }
}
