import 'dart:collection';
import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';
import 'framing.dart';
import 'landmark_type.dart';
import 'smoothing.dart';

class MovementSignals {
  const MovementSignals({
    required this.movementEnergy,
    required this.stillness,
    required this.upperBodyEnergy,
    required this.lowerBodyEnergy,
    required this.postureOpenness,
    required this.trunkLeanDegrees,
    required this.confidence,
    required this.framing,
    required this.sampleCount,
  });

  static const unavailable = MovementSignals(
    movementEnergy: 0,
    stillness: 0,
    upperBodyEnergy: 0,
    lowerBodyEnergy: 0,
    postureOpenness: 0,
    trunkLeanDegrees: 0,
    confidence: 0,
    framing: Framing.noSubject,
    sampleCount: 0,
  );

  final double movementEnergy;

  final double stillness;

  final double upperBodyEnergy;

  final double lowerBodyEnergy;

  final double postureOpenness;

  final double trunkLeanDegrees;

  final double confidence;

  final Framing framing;

  final int sampleCount;

  static const stillnessReference = 0.5;

  Map<String, Object?> toMap() => {
    'movementEnergy': movementEnergy,
    'stillness': stillness,
    'upperBodyEnergy': upperBodyEnergy,
    'lowerBodyEnergy': lowerBodyEnergy,
    'postureOpenness': postureOpenness,
    'trunkLeanDegrees': trunkLeanDegrees,
    'confidence': confidence,
    'framing': framing.name,
    'sampleCount': sampleCount,
  };

  @override
  String toString() =>
      'MovementSignals(energy: ${movementEnergy.toStringAsFixed(3)}, '
      'stillness: ${stillness.toStringAsFixed(2)}, '
      'confidence: ${confidence.toStringAsFixed(2)}, '
      'n=$sampleCount)';
}

class MovementAnalyser {
  MovementAnalyser({
    this.window = const Duration(seconds: 2),
    this.confidenceThreshold = 0.5,
    this.smooth = true,
  });

  final Duration window;

  final double confidenceThreshold;

  final bool smooth;

  final Queue<_Sample> _samples = Queue<_Sample>();
  final Map<LandmarkType, OneEuroPoint> _filters = {};

  _Sample? _previous;

  void add(BodyFrame frame) {
    final normalised = frame.normalised;
    final torso = frame.torso;
    if (normalised == null || torso == null) {
      _resetFilters();
      _previous = null;
      return;
    }

    final points = <LandmarkType, Vec2>{};
    for (final entry in normalised.entries) {
      final landmark = frame[entry.key];
      if (landmark == null || !landmark.isReliable(confidenceThreshold)) {
        continue;
      }

      var point = entry.value;
      if (smooth) {
        final filter = _filters.putIfAbsent(entry.key, OneEuroPoint.new);
        final (x, y) = filter.filter(point.x, point.y, frame.timestamp);
        point = Vec2(x, y);
      }
      points[entry.key] = point;
    }

    final sample = _Sample(
      timestamp: frame.timestamp,
      points: points,
      trunkLeanDegrees: torso.tiltDegrees,
      framing: frame.framing,
    );

    _samples.addLast(sample);
    _previous = sample;
    _evict(frame.timestamp);
  }

  MovementSignals get signals {
    if (_samples.length < 2) return MovementSignals.unavailable;

    final samples = _samples.toList(growable: false);

    var totalDistance = 0.0;
    var totalCount = 0;
    var upperDistance = 0.0;
    var upperCount = 0;
    var lowerDistance = 0.0;
    var lowerCount = 0;

    for (var i = 1; i < samples.length; i++) {
      final previous = samples[i - 1];
      final current = samples[i];
      final dt =
          current.timestamp.difference(previous.timestamp).inMicroseconds / 1e6;
      if (dt <= 0) continue;

      for (final entry in current.points.entries) {
        final before = previous.points[entry.key];
        if (before == null) continue;

        final speed = entry.value.distanceTo(before) / dt;

        totalDistance += speed;
        totalCount++;

        if (upperBodyLandmarks.contains(entry.key)) {
          upperDistance += speed;
          upperCount++;
        } else if (lowerBodyLandmarks.contains(entry.key)) {
          lowerDistance += speed;
          lowerCount++;
        }
      }
    }

    if (totalCount == 0) return MovementSignals.unavailable;

    final energy = totalDistance / totalCount;

    var leanSum = 0.0;
    var confidenceSum = 0.0;
    var worstFraming = Framing.full;
    for (final sample in samples) {
      leanSum += sample.trunkLeanDegrees;
      confidenceSum += sample.framing.qualityFor(
        needsLegs: sample.framing.framing.hasLegs,
      );
      if (sample.framing.framing.index > worstFraming.index) {
        worstFraming = sample.framing.framing;
      }
    }

    return MovementSignals(
      movementEnergy: energy,
      stillness: _stillnessFrom(energy),
      upperBodyEnergy: upperCount == 0 ? 0 : upperDistance / upperCount,
      lowerBodyEnergy: lowerCount == 0 ? 0 : lowerDistance / lowerCount,
      postureOpenness: _openness(samples.last),
      trunkLeanDegrees: leanSum / samples.length,
      confidence: confidenceSum / samples.length,
      framing: worstFraming,
      sampleCount: samples.length,
    );
  }

  void _evict(DateTime now) {
    final cutoff = now.subtract(window);
    while (_samples.isNotEmpty && _samples.first.timestamp.isBefore(cutoff)) {
      _samples.removeFirst();
    }
  }

  void _resetFilters() {
    for (final filter in _filters.values) {
      filter.reset();
    }
  }

  void reset() {
    _samples.clear();
    _resetFilters();
    _filters.clear();
    _previous = null;
  }

  bool get hasTracking => _previous != null;

  static double _stillnessFrom(double energy) =>
      math.exp(-energy / MovementSignals.stillnessReference);

  static double _openness(_Sample sample) {
    final left = sample.points[LandmarkType.leftWrist];
    final right = sample.points[LandmarkType.rightWrist];
    if (left == null && right == null) return 0;

    final distances = <double>[
      if (left != null) left.x.abs(),
      if (right != null) right.x.abs(),
    ];
    return distances.reduce((a, b) => a + b) / distances.length;
  }
}

class _Sample {
  const _Sample({
    required this.timestamp,
    required this.points,
    required this.trunkLeanDegrees,
    required this.framing,
  });

  final DateTime timestamp;
  final Map<LandmarkType, Vec2> points;
  final double trunkLeanDegrees;
  final FramingReport framing;
}
