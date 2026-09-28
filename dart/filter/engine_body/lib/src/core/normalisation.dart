import 'dart:math' as math;

import 'body_landmark.dart';
import 'landmark_type.dart';

class TorsoFrame {
  const TorsoFrame._({
    required this.origin,
    required this.scale,
    required this.tiltRadians,
    required this.shoulderWidth,
  });

  final Vec2 origin;

  final double scale;

  final double tiltRadians;

  final double shoulderWidth;

  double get tiltDegrees => tiltRadians * 180 / math.pi;

  static TorsoFrame? from(
    Map<LandmarkType, BodyLandmark> byType, {
    double threshold = 0.5,
  }) {
    final ls = byType[LandmarkType.leftShoulder];
    final rs = byType[LandmarkType.rightShoulder];
    final lh = byType[LandmarkType.leftHip];
    final rh = byType[LandmarkType.rightHip];

    if (ls == null || rs == null || lh == null || rh == null) return null;
    if (!ls.isReliable(threshold) ||
        !rs.isReliable(threshold) ||
        !lh.isReliable(threshold) ||
        !rh.isReliable(threshold)) {
      return null;
    }

    final midHip = Vec2((lh.x + rh.x) / 2, (lh.y + rh.y) / 2);
    final midShoulder = Vec2((ls.x + rs.x) / 2, (ls.y + rs.y) / 2);

    final spine = Vec2(
      midShoulder.x - midHip.x,
      -(midShoulder.y - midHip.y),
    );
    final scale = spine.length;

    if (scale < _minimumTorsoExtent) return null;

    final tilt = math.atan2(spine.x, spine.y);

    final shoulderWidth = ls.distanceTo(rs) / scale;

    return TorsoFrame._(
      origin: midHip,
      scale: scale,
      tiltRadians: tilt,
      shoulderWidth: shoulderWidth,
    );
  }

  static const _minimumTorsoExtent = 0.03;

  Vec2 normalise(BodyLandmark landmark) => Vec2(
    (landmark.x - origin.x) / scale,
    -(landmark.y - origin.y) / scale,
  );

  Map<LandmarkType, Vec2> normaliseAll(Iterable<BodyLandmark> landmarks) => {
    for (final l in landmarks) l.type: normalise(l),
  };

  @override
  String toString() =>
      'TorsoFrame(origin: $origin, scale: ${scale.toStringAsFixed(3)}, '
      'tilt: ${tiltDegrees.toStringAsFixed(1)}deg)';
}

double? angleAt(Vec2 vertex, Vec2 a, Vec2 b) {
  final v1 = a - vertex;
  final v2 = b - vertex;
  final l1 = v1.length;
  final l2 = v2.length;
  if (l1 < 1e-6 || l2 < 1e-6) return null;

  final cosine = (v1.x * v2.x + v1.y * v2.y) / (l1 * l2);
  return math.acos(cosine.clamp(-1.0, 1.0)) * 180 / math.pi;
}
