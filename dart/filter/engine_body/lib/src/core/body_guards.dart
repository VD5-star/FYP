import 'dart:collection';
import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';
import 'landmark_type.dart';

const gapResetThreshold = Duration(milliseconds: 100);

class LandmarkMedianFilter {
  LandmarkMedianFilter({this.width = 5})
      : assert(width.isOdd, 'median window must be odd');

  final int width;

  final _history = HashMap<LandmarkType, Queue<({double x, double y})>>();
  Duration? _lastTime;

  Map<LandmarkType, ({double x, double y})> call(
      BodyFrame frame, Duration timestamp) {
    final last = _lastTime;
    if (last != null && timestamp - last > gapResetThreshold) {
      _history.clear();
    }
    _lastTime = timestamp;

    final out = <LandmarkType, ({double x, double y})>{};
    for (final landmark in frame.landmarks) {
      final queue = _history.putIfAbsent(
          landmark.type, Queue<({double x, double y})>.new);
      queue.addLast((x: landmark.x, y: landmark.y));
      while (queue.length > width) {
        queue.removeFirst();
      }

      if (queue.length < width) {
        out[landmark.type] = (x: landmark.x, y: landmark.y);
      } else {
        out[landmark.type] = (
          x: _median(queue.map((p) => p.x)),
          y: _median(queue.map((p) => p.y)),
        );
      }
    }
    return out;
  }

  void reset() {
    _history.clear();
    _lastTime = null;
  }

  static double _median(Iterable<double> values) {
    final sorted = List<double>.from(values)..sort();
    final middle = sorted.length ~/ 2;
    if (sorted.length.isOdd) return sorted[middle];
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
}

double? torsoYawDegrees(BodyFrame frame) {
  final ls = frame[LandmarkType.leftShoulder];
  final rs = frame[LandmarkType.rightShoulder];
  if (ls == null || rs == null) return null;
  if (ls.likelihood < 0.5 || rs.likelihood < 0.5) return null;

  final dx = ls.x - rs.x;
  final dz = ls.z - rs.z;
  if (!dx.isFinite || !dz.isFinite) return null;
  if (dx.abs() < 1e-9 && dz.abs() < 1e-9) return null;

  if (ls.z == 0 && rs.z == 0) return null;

  return (math.atan2(dz, dx) * 180 / math.pi).abs();
}

double? noseAsymmetryYaw(BodyFrame frame) {
  final nose = frame[LandmarkType.nose];
  final ls = frame[LandmarkType.leftShoulder];
  final rs = frame[LandmarkType.rightShoulder];
  if (nose == null || ls == null || rs == null) return null;
  for (final l in <BodyLandmark>[nose, ls, rs]) {
    if (l.likelihood < 0.5) return null;
  }

  final left = (ls.x - nose.x).abs();
  final right = (rs.x - nose.x).abs();
  final total = left + right;
  if (!total.isFinite || total < 1e-9) return null;

  return (left - right).abs() / total * 90;
}

double? viewpointYawDegrees(BodyFrame frame) =>
    torsoYawDegrees(frame) ?? noseAsymmetryYaw(frame);

double? shoulderTorsoRatio(BodyFrame frame) {
  final ls = frame[LandmarkType.leftShoulder];
  final rs = frame[LandmarkType.rightShoulder];
  final lh = frame[LandmarkType.leftHip];
  final rh = frame[LandmarkType.rightHip];
  if (ls == null || rs == null || lh == null || rh == null) return null;
  for (final l in <BodyLandmark>[ls, rs, lh, rh]) {
    if (l.likelihood < 0.5) return null;
  }

  final width = math.sqrt(
      math.pow(ls.x - rs.x, 2) + math.pow(ls.y - rs.y, 2));
  final shoulderX = (ls.x + rs.x) / 2;
  final shoulderY = (ls.y + rs.y) / 2;
  final hipX = (lh.x + rh.x) / 2;
  final hipY = (lh.y + rh.y) / 2;
  final torso = math.sqrt(
      math.pow(shoulderX - hipX, 2) + math.pow(shoulderY - hipY, 2));
  if (!torso.isFinite || torso < 1e-9) return null;
  return width / torso;
}

class SubjectSwitchDetector {
  SubjectSwitchDetector({this.torsoRate = 4.0, this.hipRate = 20.0});

  final double torsoRate;

  final double hipRate;

  double? _torso;
  ({double x, double y})? _hip;
  Duration? _lastTime;

  bool call(BodyFrame frame, Duration timestamp) {
    final ls = frame[LandmarkType.leftShoulder];
    final rs = frame[LandmarkType.rightShoulder];
    final lh = frame[LandmarkType.leftHip];
    final rh = frame[LandmarkType.rightHip];
    if (ls == null || rs == null || lh == null || rh == null) return false;
    for (final l in <BodyLandmark>[ls, rs, lh, rh]) {
      if (l.likelihood < 0.5) return false;
    }

    final shoulderX = (ls.x + rs.x) / 2;
    final shoulderY = (ls.y + rs.y) / 2;
    final hipX = (lh.x + rh.x) / 2;
    final hipY = (lh.y + rh.y) / 2;
    final torso = math.sqrt(
        math.pow(shoulderX - hipX, 2) + math.pow(shoulderY - hipY, 2));
    if (!torso.isFinite || torso < 1e-9) return false;

    final last = _lastTime;
    final gapped = last != null && timestamp - last > gapResetThreshold * 5;
    final dt = last == null ? 0.0 : (timestamp - last).inMicroseconds / 1e6;
    _lastTime = timestamp;

    var switched = false;
    final previousTorso = _torso;
    final previousHip = _hip;
    if (previousTorso != null && previousHip != null && !gapped && dt > 1e-6) {
      final torsoChange = (torso - previousTorso).abs() / previousTorso / dt;
      final hipShift = math.sqrt(math.pow(hipX - previousHip.x, 2) +
              math.pow(hipY - previousHip.y, 2)) /
          torso /
          dt;
      switched = torsoChange > torsoRate || hipShift > hipRate;
    }

    _torso = torso;
    _hip = (x: hipX, y: hipY);
    return switched;
  }

  void reset() {
    _torso = null;
    _hip = null;
    _lastTime = null;
  }
}

const dempsterSegments = <(double, List<LandmarkType>)>[
  (0.430, [
    LandmarkType.leftShoulder,
    LandmarkType.rightShoulder,
    LandmarkType.leftHip,
    LandmarkType.rightHip,
  ]),
  (0.100, [LandmarkType.leftHip, LandmarkType.leftKnee]),
  (0.100, [LandmarkType.rightHip, LandmarkType.rightKnee]),
  (0.0465, [LandmarkType.leftKnee, LandmarkType.leftAnkle]),
  (0.0465, [LandmarkType.rightKnee, LandmarkType.rightAnkle]),
  (0.015, [LandmarkType.leftAnkle, LandmarkType.leftFootIndex]),
  (0.015, [LandmarkType.rightAnkle, LandmarkType.rightFootIndex]),
  (0.028, [LandmarkType.leftShoulder, LandmarkType.leftElbow]),
  (0.028, [LandmarkType.rightShoulder, LandmarkType.rightElbow]),
  (0.022, [LandmarkType.leftElbow, LandmarkType.leftWrist]),
  (0.022, [LandmarkType.rightElbow, LandmarkType.rightWrist]),
];

const minimumMassFraction = 0.60;

({double x, double y})? centreOfMass(BodyFrame frame,
    {double threshold = 0.5}) {
  var total = 0.0;
  var accX = 0.0;
  var accY = 0.0;

  for (final (mass, types) in dempsterSegments) {
    var sumX = 0.0;
    var sumY = 0.0;
    var usable = true;
    for (final type in types) {
      final landmark = frame[type];
      if (landmark == null || landmark.likelihood < threshold) {
        usable = false;
        break;
      }
      sumX += landmark.x;
      sumY += landmark.y;
    }
    if (!usable) continue;

    accX += mass * sumX / types.length;
    accY += mass * sumY / types.length;
    total += mass;
  }

  if (total < minimumMassFraction) return null;
  final x = accX / total;
  final y = accY / total;
  if (!x.isFinite || !y.isFinite) return null;
  return (x: x, y: y);
}
