import 'dart:math' as math;

import 'landmark_type.dart';

class BodyLandmark {
  const BodyLandmark({
    required this.type,
    required this.x,
    required this.y,
    required this.z,
    required this.likelihood,
    required this.inFrameLikelihood,
  });

  final LandmarkType type;

  final double x;

  final double y;

  final double z;

  final double likelihood;

  final double inFrameLikelihood;

  bool isReliable([double threshold = 0.5]) =>
      likelihood >= threshold && inFrameLikelihood >= threshold;

  double distanceTo(BodyLandmark other) =>
      math.sqrt(math.pow(x - other.x, 2) + math.pow(y - other.y, 2));

  BodyLandmark copyWith({double? x, double? y, double? z}) => BodyLandmark(
    type: type,
    x: x ?? this.x,
    y: y ?? this.y,
    z: z ?? this.z,
    likelihood: likelihood,
    inFrameLikelihood: inFrameLikelihood,
  );

  static BodyLandmark? fromMap(Map<Object?, Object?> map) {
    final type = LandmarkType.fromId(_int(map['type']));
    if (type == null) return null;
    return BodyLandmark(
      type: type,
      x: _double(map['x']),
      y: _double(map['y']),
      z: _double(map['z']),
      likelihood: _double(map['likelihood']),
      inFrameLikelihood: _double(map['inFrameLikelihood']),
    );
  }

  Map<String, Object?> toMap() => {
    'type': type.id,
    'x': x,
    'y': y,
    'z': z,
    'likelihood': likelihood,
    'inFrameLikelihood': inFrameLikelihood,
  };

  static int _int(Object? v) => v is num ? v.toInt() : -1;
  static double _double(Object? v) => v is num ? v.toDouble() : 0.0;

  @override
  String toString() =>
      'BodyLandmark(${type.name}, '
      '${x.toStringAsFixed(3)}, ${y.toStringAsFixed(3)}, '
      'p=${likelihood.toStringAsFixed(2)})';
}

class Vec2 {
  const Vec2(this.x, this.y);

  final double x;
  final double y;

  static const zero = Vec2(0, 0);

  double get length => math.sqrt(x * x + y * y);

  Vec2 operator -(Vec2 other) => Vec2(x - other.x, y - other.y);
  Vec2 operator +(Vec2 other) => Vec2(x + other.x, y + other.y);
  Vec2 operator *(double s) => Vec2(x * s, y * s);

  double distanceTo(Vec2 other) => (this - other).length;

  @override
  String toString() => '(${x.toStringAsFixed(3)}, ${y.toStringAsFixed(3)})';
}
