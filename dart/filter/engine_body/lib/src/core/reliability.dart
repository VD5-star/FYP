import 'dart:collection';
import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';
import 'landmark_type.dart';

const rigidBones = <(LandmarkType, LandmarkType)>[
  (LandmarkType.leftShoulder, LandmarkType.leftElbow),
  (LandmarkType.leftElbow, LandmarkType.leftWrist),
  (LandmarkType.rightShoulder, LandmarkType.rightElbow),
  (LandmarkType.rightElbow, LandmarkType.rightWrist),
  (LandmarkType.leftHip, LandmarkType.leftKnee),
  (LandmarkType.leftKnee, LandmarkType.leftAnkle),
  (LandmarkType.rightHip, LandmarkType.rightKnee),
  (LandmarkType.rightKnee, LandmarkType.rightAnkle),
  (LandmarkType.leftShoulder, LandmarkType.rightShoulder),
  (LandmarkType.leftHip, LandmarkType.rightHip),
];

class BoneLengthMonitor {
  BoneLengthMonitor({
    this.tolerance = 0.35,
    this.window = 45,
    this.minSamples = 15,
  });

  final double tolerance;

  final int window;

  final int minSamples;

  final _history = HashMap<int, Queue<double>>();

  Set<LandmarkType> update(BodyFrame frame, double? torso) {
    final suspect = <LandmarkType>{};
    if (torso == null || torso <= 0) return suspect;

    for (var i = 0; i < rigidBones.length; i++) {
      final (a, b) = rigidBones[i];
      final first = frame[a];
      final second = frame[b];
      if (first == null || second == null) continue;

      if (first.likelihood < 0.5 || second.likelihood < 0.5) continue;

      final dx = first.x - second.x;
      final dy = first.y - second.y;
      final length = math.sqrt(dx * dx + dy * dy) / torso;

      final history = _history.putIfAbsent(i, Queue<double>.new);

      if (history.length >= minSamples) {
        final reference = _median(history);
        if (reference > 1e-9) {
          final deviation = (length - reference).abs() / reference;
          if (deviation > tolerance) {
            suspect
              ..add(a)
              ..add(b);
            continue;
          }
        }
      }

      history.addLast(length);
      while (history.length > window) {
        history.removeFirst();
      }
    }

    return suspect;
  }

  static double _median(Iterable<double> values) {
    final sorted = List<double>.from(values)..sort();
    if (sorted.isEmpty) return 0;
    final middle = sorted.length ~/ 2;
    if (sorted.length.isOdd) return sorted[middle];
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  void reset() => _history.clear();
}

class LandmarkVelocity {
  static const alpha = 0.17;

  static const gapReset = Duration(milliseconds: 100);

  Map<LandmarkType, ({double x, double y})>? _previous;
  Duration? _previousTime;
  Map<LandmarkType, double>? _speeds;

  Map<LandmarkType, double>? update(BodyFrame frame, double? torso,
      Duration timestamp) {
    if (torso == null || torso <= 0) {
      reset();
      return null;
    }

    final current = <LandmarkType, ({double x, double y})>{
      for (final l in frame.landmarks) l.type: (x: l.x, y: l.y),
    };

    final prev = _previous;
    final prevTime = _previousTime;
    if (prev == null || prevTime == null) {
      _previous = current;
      _previousTime = timestamp;
      return null;
    }

    final dt = timestamp - prevTime;
    if (dt <= Duration.zero || dt > gapReset) {
      _previous = current;
      _previousTime = timestamp;
      _speeds = null;
      return null;
    }

    final seconds = dt.inMicroseconds / 1e6;
    final updated = <LandmarkType, double>{};
    for (final entry in current.entries) {
      final before = prev[entry.key];
      if (before == null) continue;
      final dx = entry.value.x - before.x;
      final dy = entry.value.y - before.y;
      var raw = math.sqrt(dx * dx + dy * dy) / torso / seconds;
      if (!raw.isFinite) raw = 0;
      final previousSpeed = _speeds?[entry.key];
      updated[entry.key] =
          previousSpeed == null ? raw : previousSpeed + alpha * (raw - previousSpeed);
    }

    _previous = current;
    _previousTime = timestamp;
    _speeds = updated;
    return updated;
  }

  void reset() {
    _previous = null;
    _previousTime = null;
    _speeds = null;
  }
}

double? angle3d(BodyFrame frame, LandmarkType from, LandmarkType vertex,
    LandmarkType to) {
  final a = frame[from];
  final v = frame[vertex];
  final c = frame[to];
  if (a == null || v == null || c == null) return null;

  final v1x = a.x - v.x;
  final v1y = a.y - v.y;
  final v1z = a.z - v.z;
  final v2x = c.x - v.x;
  final v2y = c.y - v.y;
  final v2z = c.z - v.z;

  final n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
  final n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
  if (!n1.isFinite || !n2.isFinite || n1 < 1e-9 || n2 < 1e-9) return null;

  final cosine = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2);
  if (!cosine.isFinite) return null;
  return math.acos(cosine.clamp(-1.0, 1.0)) * 180 / math.pi;
}

double? angleDisagreement(double? angle2d, double? angle3dValue) {
  if (angle2d == null || angle3dValue == null) return null;
  if (!angle2d.isFinite || !angle3dValue.isFinite) return null;
  return (angle2d - angle3dValue).abs();
}

class TrustReport {
  const TrustReport({
    required this.trustworthy,
    this.reasons = const [],
    this.suspectLandmarks = const {},
    this.maxSpeed,
    this.disagreement,
  });

  final bool trustworthy;
  final List<String> reasons;
  final Set<LandmarkType> suspectLandmarks;
  final double? maxSpeed;

  final double? disagreement;
}

class ReliabilityMonitor {
  ReliabilityMonitor({this.maxSpeed = 12.0, this.maxDisagreement = 30.0});

  final double maxSpeed;

  final double maxDisagreement;

  final velocity = LandmarkVelocity();
  final bones = BoneLengthMonitor();

  TrustReport update(
    BodyFrame frame,
    double? torso,
    Duration timestamp, {
    double? angle2d,
    double? angle3dValue,
  }) {
    final reasons = <String>[];

    final speeds = velocity.update(frame, torso, timestamp);
    double? peak;
    if (speeds != null) {
      for (final entry in speeds.entries) {
        final landmark = frame[entry.key];
        if (landmark == null || landmark.likelihood < 0.5) continue;
        if (peak == null || entry.value > peak) peak = entry.value;
      }
      if (peak != null && peak > maxSpeed) {
        reasons.add('landmark moved impossibly fast');
      }
    }

    final suspect = bones.update(frame, torso);
    if (suspect.isNotEmpty) {
      reasons.add('limb length changed impossibly');
    }

    final disagreement = angleDisagreement(angle2d, angle3dValue);
    if (disagreement != null && disagreement > maxDisagreement) {
      reasons.add('2D and 3D angles disagree — check camera angle');
    }

    return TrustReport(
      trustworthy: reasons.isEmpty,
      reasons: reasons,
      suspectLandmarks: suspect,
      maxSpeed: peak,
      disagreement: disagreement,
    );
  }

  void reset() {
    velocity.reset();
    bones.reset();
  }
}

double? torsoLength(BodyFrame frame, {double threshold = 0.5}) {
  final ls = frame[LandmarkType.leftShoulder];
  final rs = frame[LandmarkType.rightShoulder];
  final lh = frame[LandmarkType.leftHip];
  final rh = frame[LandmarkType.rightHip];
  if (ls == null || rs == null || lh == null || rh == null) return null;

  for (final l in <BodyLandmark>[ls, rs, lh, rh]) {
    if (l.likelihood < threshold) return null;
  }

  final shoulderX = (ls.x + rs.x) / 2;
  final shoulderY = (ls.y + rs.y) / 2;
  final hipX = (lh.x + rh.x) / 2;
  final hipY = (lh.y + rh.y) / 2;

  final dx = shoulderX - hipX;
  final dy = shoulderY - hipY;
  final length = math.sqrt(dx * dx + dy * dy);
  return length > 1e-9 ? length : null;
}
