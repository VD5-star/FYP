import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';
import 'landmark_type.dart';

class Bone {
  const Bone(this.child, this.parent);

  final LandmarkType child;

  final LandmarkType? parent;
}

const List<Bone> kinematicTree = <Bone>[
  Bone(LandmarkType.leftHip, null),
  Bone(LandmarkType.rightHip, null),
  Bone(LandmarkType.leftShoulder, LandmarkType.leftHip),
  Bone(LandmarkType.rightShoulder, LandmarkType.rightHip),
  Bone(LandmarkType.leftElbow, LandmarkType.leftShoulder),
  Bone(LandmarkType.rightElbow, LandmarkType.rightShoulder),
  Bone(LandmarkType.leftWrist, LandmarkType.leftElbow),
  Bone(LandmarkType.rightWrist, LandmarkType.rightElbow),
  Bone(LandmarkType.leftKnee, LandmarkType.leftHip),
  Bone(LandmarkType.rightKnee, LandmarkType.rightHip),
  Bone(LandmarkType.leftAnkle, LandmarkType.leftKnee),
  Bone(LandmarkType.rightAnkle, LandmarkType.rightKnee),
];

const List<(LandmarkType, LandmarkType)> mirroredBones =
    <(LandmarkType, LandmarkType)>[
  (LandmarkType.leftShoulder, LandmarkType.rightShoulder),
  (LandmarkType.leftElbow, LandmarkType.rightElbow),
  (LandmarkType.leftWrist, LandmarkType.rightWrist),
  (LandmarkType.leftKnee, LandmarkType.rightKnee),
  (LandmarkType.leftAnkle, LandmarkType.rightAnkle),
];

const double calibrationVisibility = 0.2;

const double torsoVisibility = 0.2;

const double edgeMargin = 0.002;

bool isOutsideFrame(BodyLandmark landmark, {double margin = edgeMargin}) {
  if (!landmark.x.isFinite || !landmark.y.isFinite) return true;
  return landmark.x < -margin ||
      landmark.x > 1 + margin ||
      landmark.y < -margin ||
      landmark.y > 1 + margin;
}

const int minimumBoneSamples = 10;

class BodyModel {
  const BodyModel({this.lengths = const {}, this.samples = 0});

  final Map<LandmarkType, double> lengths;

  final int samples;

  bool get ready => lengths.length >= 8;
}

double? torsoScale(BodyFrame frame) {
  final ls = frame[LandmarkType.leftShoulder];
  final rs = frame[LandmarkType.rightShoulder];
  final lh = frame[LandmarkType.leftHip];
  final rh = frame[LandmarkType.rightHip];
  if (ls == null || rs == null || lh == null || rh == null) return null;

  final sx = (ls.x + rs.x) / 2;
  final sy = (ls.y + rs.y) / 2;
  final hx = (lh.x + rh.x) / 2;
  final hy = (lh.y + rh.y) / 2;
  final d = math.sqrt(math.pow(sx - hx, 2) + math.pow(sy - hy, 2));
  if (!d.isFinite || d < 1e-9) return null;
  return d;
}

class SkeletonCalibrator {
  SkeletonCalibrator({this.frames = 30, this.refreshEvery = 300});

  final int frames;

  final int refreshEvery;

  final Map<LandmarkType, List<double>> _samples =
      <LandmarkType, List<double>>{};
  int _count = 0;
  int _sinceRefresh = 0;

  BodyModel _model = const BodyModel();
  BodyModel get model => _model;

  BodyModel update(BodyFrame frame) {
    if (_count >= frames) {
      _sinceRefresh++;
      if (_sinceRefresh < refreshEvery) return _model;
      _sinceRefresh = 0;
      _count = 0;
      _samples.clear();
    }

    final torso = torsoScale(frame);
    if (torso == null) return _model;

    for (final type in <LandmarkType>[
      LandmarkType.leftShoulder,
      LandmarkType.rightShoulder,
      LandmarkType.leftHip,
      LandmarkType.rightHip,
    ]) {
      final l = frame[type];
      if (l == null || l.likelihood < torsoVisibility) return _model;
      if (isOutsideFrame(l)) return _model;
    }

    for (final bone in kinematicTree) {
      final parent = bone.parent;
      if (parent == null) continue;
      final c = frame[bone.child];
      final p = frame[parent];
      if (c == null || p == null) continue;
      if (c.likelihood < calibrationVisibility ||
          p.likelihood < calibrationVisibility) {
        continue;
      }
      final d =
          math.sqrt(math.pow(c.x - p.x, 2) + math.pow(c.y - p.y, 2)) / torso;
      if (d.isFinite && d > 1e-9) {
        _samples.putIfAbsent(bone.child, () => <double>[]).add(d);
      }
    }

    _count++;
    _rebuild();
    return _model;
  }

  void _rebuild() {
    final lengths = <LandmarkType, double>{};
    _samples.forEach((type, values) {
      if (values.length < minimumBoneSamples) return;
      final sorted = List<double>.from(values)..sort();
      final mid = sorted.length ~/ 2;
      lengths[type] = sorted.length.isOdd
          ? sorted[mid]
          : (sorted[mid - 1] + sorted[mid]) / 2;
    });

    for (final (a, b) in mirroredBones) {
      final la = lengths[a];
      final lb = lengths[b];
      if (la != null && lb != null) {
        final mean = (la + lb) / 2;
        lengths[a] = mean;
        lengths[b] = mean;
      }
    }

    _model = BodyModel(lengths: lengths, samples: _count);
  }

  void reset() {
    _samples.clear();
    _count = 0;
    _sinceRefresh = 0;
    _model = const BodyModel();
  }
}

const Map<LandmarkType, LandmarkType> carriedLandmarks =
    <LandmarkType, LandmarkType>{
  LandmarkType.leftThumb: LandmarkType.leftWrist,
  LandmarkType.leftIndex: LandmarkType.leftWrist,
  LandmarkType.leftPinky: LandmarkType.leftWrist,
  LandmarkType.rightThumb: LandmarkType.rightWrist,
  LandmarkType.rightIndex: LandmarkType.rightWrist,
  LandmarkType.rightPinky: LandmarkType.rightWrist,
  LandmarkType.leftHeel: LandmarkType.leftAnkle,
  LandmarkType.leftFootIndex: LandmarkType.leftAnkle,
  LandmarkType.rightHeel: LandmarkType.rightAnkle,
  LandmarkType.rightFootIndex: LandmarkType.rightAnkle,
};

BodyFrame applyBodyModel(BodyFrame frame, BodyModel model) {
  if (!model.ready) return frame;
  final torso = torsoScale(frame);
  if (torso == null) return frame;

  final moved = <LandmarkType, ({double x, double y})>{};

  for (final bone in kinematicTree) {
    final parent = bone.parent;
    if (parent == null) continue;
    final length = model.lengths[bone.child];
    if (length == null) continue;

    final c = frame[bone.child];
    final p = frame[parent];
    if (c == null || p == null) continue;

    final dx = c.x - p.x;
    final dy = c.y - p.y;
    final norm = math.sqrt(dx * dx + dy * dy);
    if (!norm.isFinite || norm < 1e-9) continue;

    final base = moved[parent] ?? (x: p.x, y: p.y);
    final scale = length * torso / norm;
    moved[bone.child] = (x: base.x + dx * scale, y: base.y + dy * scale);
  }

  carriedLandmarks.forEach((type, parent) {
    final l = frame[type];
    final p = frame[parent];
    final corrected = moved[parent];
    if (l == null || p == null || corrected == null) return;
    moved[type] = (x: l.x + (corrected.x - p.x), y: l.y + (corrected.y - p.y));
  });

  if (moved.isEmpty) return frame;

  return BodyFrame(
    timestamp: frame.timestamp,
    landmarks: <BodyLandmark>[
      for (final l in frame.landmarks)
        if (moved[l.type] case final m?)
          BodyLandmark(
            type: l.type,
            x: m.x,
            y: m.y,
            z: l.z,
            likelihood: l.likelihood,
            inFrameLikelihood: l.inFrameLikelihood,
          )
        else
          l,
    ],
    framing: frame.framing,
    isMirrored: frame.isMirrored,
    trackingId: frame.trackingId,
  );
}
