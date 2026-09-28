library;

import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';
import 'landmark_type.dart';
import 'skeleton.dart';

const double defaultBoneTolerance = 0.35;

const double defaultTeleportRate = 45.0;

const double defaultMaxHoldSeconds = 1.5;

const int defaultFramesBeforeTrusted = 20;

const double believableConfidence = 0.5;

class _LandmarkState {
  _LandmarkState({
    required this.x,
    required this.y,
    required this.confidence,
    this.observations = 0,
    this.inferred = false,
    this.heldFor = 0.0,
    this.offsetX,
    this.offsetY,
  });

  final double x;
  final double y;
  final double confidence;

  final int observations;

  final bool inferred;

  final double heldFor;

  final double? offsetX;
  final double? offsetY;

  bool get trusted => observations >= defaultFramesBeforeTrusted;
}

enum LandmarkVerdict {
  seen,

  held,

  discarded,
}

class OcclusionReport {
  const OcclusionReport({
    required this.landmarks,
    required this.verdicts,
    this.believeThreshold = believableConfidence,
  });

  final List<BodyLandmark> landmarks;

  final Map<LandmarkType, LandmarkVerdict> verdicts;

  final double believeThreshold;

  LandmarkVerdict verdictFor(LandmarkType type) =>
      verdicts[type] ?? LandmarkVerdict.discarded;

  bool isDrawable(LandmarkType type) =>
      verdictFor(type) != LandmarkVerdict.discarded;

  bool isUsable(LandmarkType type) {
    if (!isDrawable(type)) return false;
    final l = landmarks.firstWhere(
      (l) => l.type == type,
      orElse: () => throw StateError('landmark $type missing from report'),
    );
    return l.likelihood >= believeThreshold;
  }

  bool get hasHeldLandmarks =>
      verdicts.values.any((v) => v == LandmarkVerdict.held);

  BodyFrame applyTo(BodyFrame frame) => BodyFrame(
        timestamp: frame.timestamp,
        landmarks: landmarks,
        framing: frame.framing,
        isMirrored: frame.isMirrored,
        trackingId: frame.trackingId,
      );
}

class OcclusionTracker {
  OcclusionTracker({
    this.boneTolerance = defaultBoneTolerance,
    this.teleportRate = defaultTeleportRate,
    this.maxHoldSeconds = defaultMaxHoldSeconds,
    this.framesBeforeTrusted = defaultFramesBeforeTrusted,
    this.believeThreshold = believableConfidence,
  });

  final double boneTolerance;
  final double teleportRate;
  final double maxHoldSeconds;
  final int framesBeforeTrusted;
  final double believeThreshold;

  final Map<LandmarkType, _LandmarkState> _state =
      <LandmarkType, _LandmarkState>{};
  Duration? _lastTimestamp;

  OcclusionReport update(BodyFrame frame, BodyModel model, Duration timestamp) {
    var dt = 0.0;
    final last = _lastTimestamp;
    if (last != null) {
      dt = (timestamp - last).inMicroseconds / 1e6;
      if (dt < 0) dt = 0;
      if (dt > 1.0) {
        _state.clear();
        dt = 0;
      }
    }
    _lastTimestamp = timestamp;

    final torso = torsoScale(frame);
    final suspect = _suspectLandmarks(frame, model, torso, dt);

    final lh = frame[LandmarkType.leftHip];
    final rh = frame[LandmarkType.rightHip];
    double? midHipX;
    double? midHipY;
    if (lh != null &&
        rh != null &&
        torso != null &&
        lh.x.isFinite &&
        lh.y.isFinite &&
        rh.x.isFinite &&
        rh.y.isFinite) {
      midHipX = (lh.x + rh.x) / 2;
      midHipY = (lh.y + rh.y) / 2;
    }
    final canRebase = midHipX != null && midHipY != null && torso != null;

    final out = <BodyLandmark>[];
    final verdicts = <LandmarkType, LandmarkVerdict>{};

    for (final landmark in frame.landmarks) {
      final type = landmark.type;

      if (isOutsideFrame(landmark)) {
        _state.remove(type);
        verdicts[type] = LandmarkVerdict.discarded;
        out.add(_withConfidence(landmark, 0));
        continue;
      }

      final state = _state[type];
      final believable = landmark.likelihood >= believeThreshold &&
          !suspect.contains(type);

      if (believable) {
        _state[type] = _LandmarkState(
          x: landmark.x,
          y: landmark.y,
          confidence: landmark.likelihood,
          observations: (state?.observations ?? 0) + 1,
          offsetX: canRebase ? (landmark.x - midHipX) / torso : null,
          offsetY: canRebase ? (landmark.y - midHipY) / torso : null,
        );
        verdicts[type] = LandmarkVerdict.seen;
        out.add(landmark);
        continue;
      }

      if (state == null || !state.trusted) {
        if (state != null) {
          _state[type] = _LandmarkState(
            x: state.x,
            y: state.y,
            confidence: state.confidence,
            observations: state.observations,
            inferred: true,
            heldFor: state.heldFor + dt,
            offsetX: state.offsetX,
            offsetY: state.offsetY,
          );
        }
        verdicts[type] = LandmarkVerdict.discarded;
        out.add(_withConfidence(landmark, 0));
        continue;
      }

      final held = state.heldFor + dt;
      var x = state.x;
      var y = state.y;
      if (canRebase && state.offsetX != null && state.offsetY != null) {
        final rx = midHipX + state.offsetX! * torso;
        final ry = midHipY + state.offsetY! * torso;
        final probe = _at(landmark, rx, ry, state.confidence);
        if (!isOutsideFrame(probe)) {
          x = rx;
          y = ry;
        }
      }

      final expired = held >= maxHoldSeconds;
      final confidence =
          expired ? 0.0 : state.confidence * (1 - held / maxHoldSeconds);
      verdicts[type] =
          expired ? LandmarkVerdict.discarded : LandmarkVerdict.held;
      out.add(_at(landmark, x, y, confidence));

      _state[type] = _LandmarkState(
        x: x,
        y: y,
        confidence: state.confidence,
        observations: state.observations,
        inferred: true,
        heldFor: held,
        offsetX: state.offsetX,
        offsetY: state.offsetY,
      );
    }

    return OcclusionReport(
      landmarks: out,
      verdicts: verdicts,
      believeThreshold: believeThreshold,
    );
  }

  Set<LandmarkType> _suspectLandmarks(
    BodyFrame frame,
    BodyModel model,
    double? torso,
    double dt,
  ) {
    final suspect = <LandmarkType>{};
    if (torso == null || torso < 1e-9) return suspect;

    if (model.ready) {
      for (final bone in kinematicTree) {
        final parent = bone.parent;
        if (parent == null) continue;
        final expectedRatio = model.lengths[bone.child];
        if (expectedRatio == null) continue;
        final c = frame[bone.child];
        final p = frame[parent];
        if (c == null || p == null) continue;
        final expected = expectedRatio * torso;
        if (expected < 1e-9) continue;
        final actual = _distance(c.x, c.y, p.x, p.y);
        if (!actual.isFinite) continue;
        if ((actual - expected).abs() / expected > boneTolerance) {
          suspect.add(bone.child);
        }
      }
    }

    if (dt > 1e-9) {
      final limit = teleportRate * torso * dt;
      _state.forEach((type, state) {
        if (state.inferred) return;
        final l = frame[type];
        if (l == null) return;
        if (_distance(l.x, l.y, state.x, state.y) > limit) suspect.add(type);
      });
    }

    return suspect;
  }

  void reset() {
    _state.clear();
    _lastTimestamp = null;
  }
}

double _distance(double ax, double ay, double bx, double by) =>
    math.sqrt(math.pow(ax - bx, 2) + math.pow(ay - by, 2));

BodyLandmark _withConfidence(BodyLandmark l, double confidence) =>
    BodyLandmark(
      type: l.type,
      x: l.x,
      y: l.y,
      z: l.z,
      likelihood: confidence,
      inFrameLikelihood: confidence,
    );

BodyLandmark _at(BodyLandmark l, double x, double y, double confidence) =>
    BodyLandmark(
      type: l.type,
      x: x,
      y: y,
      z: l.z,
      likelihood: confidence,
      inFrameLikelihood: confidence,
    );
