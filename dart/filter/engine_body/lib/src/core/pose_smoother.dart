import 'body_frame.dart';
import 'body_landmark.dart';
import 'landmark_type.dart';
import 'smoothing.dart';

const Duration defaultSmootherGap = Duration(milliseconds: 100);

class PoseSmoother {
  PoseSmoother({
    this.minCutoff = 1.2,
    this.beta = 0.02,
    this.confidenceThreshold = 0.3,
    this.gapReset = defaultSmootherGap,
    this.dropFilterBelowThreshold = false,
  });

  final double minCutoff;

  final double beta;

  final double confidenceThreshold;

  final Duration gapReset;

  final bool dropFilterBelowThreshold;

  final Map<LandmarkType, OneEuroPoint> _filters = {};

  DateTime? _lastTimestamp;

  BodyFrame smooth(BodyFrame frame) {
    if (frame.landmarks.isEmpty) return frame;

    final last = _lastTimestamp;
    if (last != null && frame.timestamp.difference(last) > gapReset) {
      reset();
    }
    _lastTimestamp = frame.timestamp;

    final smoothed = <BodyLandmark>[];
    for (final landmark in frame.landmarks) {
      if (landmark.inFrameLikelihood < confidenceThreshold) {
        if (dropFilterBelowThreshold) {
          _filters.remove(landmark.type)?.reset();
        }
        smoothed.add(landmark);
        continue;
      }

      final filter = _filters.putIfAbsent(
        landmark.type,
        () => OneEuroPoint(minCutoff: minCutoff, beta: beta),
      );
      final (x, y) = filter.filter(landmark.x, landmark.y, frame.timestamp);
      smoothed.add(landmark.copyWith(x: x, y: y));
    }

    return BodyFrame(
      timestamp: frame.timestamp,
      landmarks: smoothed,
      framing: frame.framing,
      isMirrored: frame.isMirrored,
      trackingId: frame.trackingId,
    );
  }

  void reset() {
    for (final filter in _filters.values) {
      filter.reset();
    }
    _filters.clear();
    _lastTimestamp = null;
  }
}
