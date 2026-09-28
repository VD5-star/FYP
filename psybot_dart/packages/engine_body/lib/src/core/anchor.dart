library;

import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';
import 'landmark_type.dart';
import 'occlusion.dart';
import 'skeleton.dart';

const double defaultQuietThreshold = 0.005;

const double defaultReleaseThreshold = 0.015;

class JointAnchor {
  JointAnchor({
    this.quiet = defaultQuietThreshold,
    this.release = defaultReleaseThreshold,
  }) : assert(release > quiet, 'release threshold must exceed the quiet one');

  final double quiet;
  final double release;

  final Map<LandmarkType, ({double x, double y})> _held =
      <LandmarkType, ({double x, double y})>{};

  BodyFrame update(BodyFrame frame, {OcclusionReport? report}) {
    final torso = torsoScale(frame);
    if (torso == null || torso < 1e-9) return frame;

    final out = <BodyLandmark>[];
    for (final landmark in frame.landmarks) {
      final type = landmark.type;

      if (!landmark.x.isFinite || !landmark.y.isFinite) {
        _held.remove(type);
        out.add(landmark);
        continue;
      }

      if (report != null &&
          report.verdictFor(type) == LandmarkVerdict.discarded) {
        _held.remove(type);
        out.add(landmark);
        continue;
      }

      final previous = _held[type];
      if (previous == null) {
        _held[type] = (x: landmark.x, y: landmark.y);
        out.add(landmark);
        continue;
      }

      final step = math.sqrt(math.pow(landmark.x - previous.x, 2) +
              math.pow(landmark.y - previous.y, 2)) /
          torso;

      final blend = ((step - quiet) / (release - quiet)).clamp(0.0, 1.0);
      var x = previous.x + (landmark.x - previous.x) * blend;
      var y = previous.y + (landmark.y - previous.y) * blend;

      if (!x.isFinite || !y.isFinite) {
        x = landmark.x;
        y = landmark.y;
      }

      if (!isOutsideFrame(landmark)) {
        x = x.clamp(0.0, 1.0);
        y = y.clamp(0.0, 1.0);
      }

      _held[type] = (x: x, y: y);
      out.add(BodyLandmark(
        type: type,
        x: x,
        y: y,
        z: landmark.z,
        likelihood: landmark.likelihood,
        inFrameLikelihood: landmark.inFrameLikelihood,
      ));
    }

    return BodyFrame(
      timestamp: frame.timestamp,
      landmarks: out,
      framing: frame.framing,
      isMirrored: frame.isMirrored,
      trackingId: frame.trackingId,
    );
  }

  void reset() => _held.clear();
}
