import 'dart:math' as math;

import 'body_frame.dart';
import 'landmark_type.dart';

class JointAngle {
  const JointAngle({required this.degrees, required this.confidence});

  final double degrees;

  final double confidence;

  static const reliableThreshold = 0.5;

  bool get isReliable => confidence >= reliableThreshold;

  double? get orNull => isReliable ? degrees : null;

  @override
  String toString() =>
      'JointAngle(${degrees.toStringAsFixed(1)}°, '
      'conf ${confidence.toStringAsFixed(2)})';
}

class JointDefinition {
  const JointDefinition(this.from, this.vertex, this.to);

  final LandmarkType from;
  final LandmarkType vertex;
  final LandmarkType to;
}

const trackedJoints = <String, JointDefinition>{
  'leftKnee': JointDefinition(
    LandmarkType.leftHip,
    LandmarkType.leftKnee,
    LandmarkType.leftAnkle,
  ),
  'rightKnee': JointDefinition(
    LandmarkType.rightHip,
    LandmarkType.rightKnee,
    LandmarkType.rightAnkle,
  ),
  'leftHip': JointDefinition(
    LandmarkType.leftShoulder,
    LandmarkType.leftHip,
    LandmarkType.leftKnee,
  ),
  'rightHip': JointDefinition(
    LandmarkType.rightShoulder,
    LandmarkType.rightHip,
    LandmarkType.rightKnee,
  ),
  'leftElbow': JointDefinition(
    LandmarkType.leftShoulder,
    LandmarkType.leftElbow,
    LandmarkType.leftWrist,
  ),
  'rightElbow': JointDefinition(
    LandmarkType.rightShoulder,
    LandmarkType.rightElbow,
    LandmarkType.rightWrist,
  ),
  'leftShoulder': JointDefinition(
    LandmarkType.leftElbow,
    LandmarkType.leftShoulder,
    LandmarkType.leftHip,
  ),
  'rightShoulder': JointDefinition(
    LandmarkType.rightElbow,
    LandmarkType.rightShoulder,
    LandmarkType.rightHip,
  ),
};

extension JointAngles on BodyFrame {
  JointAngle? angleAt(
    LandmarkType from,
    LandmarkType vertex,
    LandmarkType to,
  ) {
    final a = this[from];
    final v = this[vertex];
    final c = this[to];
    if (a == null || v == null || c == null) return null;

    final v1x = a.x - v.x;
    final v1y = a.y - v.y;
    final v2x = c.x - v.x;
    final v2y = c.y - v.y;

    final n1 = math.sqrt(v1x * v1x + v1y * v1y);
    final n2 = math.sqrt(v2x * v2x + v2y * v2y);
    if (!n1.isFinite || !n2.isFinite || n1 < 1e-9 || n2 < 1e-9) return null;

    final cosine = (v1x * v2x + v1y * v2y) / (n1 * n2);
    if (!cosine.isFinite) return null;
    final degrees = math.acos(cosine.clamp(-1.0, 1.0)) * 180 / math.pi;

    final confidence = math.min(
      a.likelihood,
      math.min(v.likelihood, c.likelihood),
    );

    return JointAngle(degrees: degrees, confidence: confidence);
  }

  Map<String, JointAngle> get allAngles {
    final out = <String, JointAngle>{};
    for (final entry in trackedJoints.entries) {
      final angle = angleAt(
        entry.value.from,
        entry.value.vertex,
        entry.value.to,
      );
      if (angle != null) out[entry.key] = angle;
    }
    return out;
  }
}

class SideTracker {
  SideTracker({this.margin = 0.25, this.patience = 5});

  final double margin;

  final int patience;

  String? _side;
  int _pressure = 0;

  String? get side => _side;

  JointAngle? update(Map<String, JointAngle> angles, String joint) {
    final left = angles['left$joint'];
    final right = angles['right$joint'];

    if (left == null && right == null) return null;
    if (left == null) {
      _side = 'right';
      _pressure = 0;
      return right;
    }
    if (right == null) {
      _side = 'left';
      _pressure = 0;
      return left;
    }

    if (_side == null) {
      _side = left.confidence >= right.confidence ? 'left' : 'right';
      _pressure = 0;
    }

    final current = _side == 'left' ? left : right;
    final other = _side == 'left' ? right : left;

    if (other.confidence > current.confidence + margin) {
      _pressure++;
      if (_pressure >= patience) {
        _side = _side == 'left' ? 'right' : 'left';
        _pressure = 0;
        return other;
      }
    } else {
      _pressure = 0;
    }

    return current;
  }

  void reset() {
    _side = null;
    _pressure = 0;
  }
}

const double maxAngleRate = 500.0;

class AngleRateGuard {
  AngleRateGuard({this.maxRate = maxAngleRate, this.resetGap = 0.5});

  final double maxRate;

  final double resetGap;

  final Map<String, ({Duration at, double degrees})> _last =
      <String, ({Duration at, double degrees})>{};

  bool check(String name, double degrees, Duration timestamp) {
    final previous = _last[name];
    if (previous != null) {
      final dt = (timestamp - previous.at).inMicroseconds / 1e6;
      if (dt > 0 && dt <= resetGap) {
        if ((degrees - previous.degrees).abs() / dt > maxRate) {
          return false;
        }
      }
    }
    _last[name] = (at: timestamp, degrees: degrees);
    return true;
  }

  Map<String, JointAngle> filter(
    Map<String, JointAngle> angles,
    Duration timestamp,
  ) {
    final out = <String, JointAngle>{};
    angles.forEach((name, angle) {
      if (check(name, angle.degrees, timestamp)) out[name] = angle;
    });
    return out;
  }

  void reset() => _last.clear();
}

double? jointSymmetry(Map<String, JointAngle> angles, String joint) {
  final left = angles['left$joint'];
  final right = angles['right$joint'];
  if (left == null || right == null) return null;
  if (!left.isReliable || !right.isReliable) return null;
  return (left.degrees - right.degrees).abs();
}
