import 'dart:math' as math;

import 'config.dart';
import 'geometry.dart';
import 'maths.dart';

class PoseResult {
  PoseResult({
    this.yaw = 0.0,
    this.pitch = 0.0,
    this.roll = 0.0,
    this.gazeX = 0.0,
    this.gazeY = 0.0,
    this.attention = 0.0,
    this.eyeOpenness = 0.0,
    this.isBlinking = false,
  });

  double yaw;
  double pitch;
  double roll;
  double gazeX;
  double gazeY;
  double attention;
  double eyeOpenness;
  bool isBlinking;

  Map<String, Object?> toMap() => <String, Object?>{
    'yaw': roundTo(yaw, 3),
    'pitch': roundTo(pitch, 3),
    'roll': roundTo(roll, 3),
    'gaze_x': roundTo(gazeX, 3),
    'gaze_y': roundTo(gazeY, 3),
    'attention': roundTo(attention, 3),
    'eye_openness': roundTo(eyeOpenness, 3),
    'is_blinking': isBlinking,
  };
}

class SpoofResult {
  const SpoofResult({
    this.liveness = 1.0,
    this.isSpoof = false,
    this.signals = const <String, double>{},
  });

  final double liveness;
  final bool isSpoof;
  final Map<String, double> signals;

  Map<String, Object?> toMap() => <String, Object?>{
    'liveness': roundTo(liveness, 4),
    'is_spoof': isSpoof,
    'signals': <String, double>{
      for (final entry in signals.entries) entry.key: roundTo(entry.value, 4),
    },
  };
}

abstract class GazeEstimator {
  (double, double) estimate(List<Point2> landmarks, PoseResult pose);
}

class NoGazeEstimator implements GazeEstimator {
  const NoGazeEstimator();

  @override
  (double, double) estimate(List<Point2> landmarks, PoseResult pose) =>
      (0.0, 0.0);
}

class PoseEstimator {
  PoseEstimator({this.gaze = const NoGazeEstimator()});

  final GazeEstimator gaze;

  final List<double> _blinkHistory = <double>[];
  int blinkCount = 0;
  bool _wasClosed = false;

  PoseResult estimate({
    List<Point2>? landmarks,
    List<double>? pose,
  }) {
    final result = PoseResult();

    if (pose != null && pose.length >= 3) {
      result.pitch = pose[0];
      result.yaw = pose[1];
      result.roll = pose[2];
    }

    result.roll = wrap180(result.roll);
    result.pitch = wrap180(result.pitch);
    result.yaw = wrap180(result.yaw);
    if (result.roll.abs() > 90.0) {
      result.roll = wrap180(result.roll - 180.0);
      result.pitch = wrap180(-result.pitch);
      result.yaw = wrap180(-result.yaw);
    }

    if (landmarks != null && landmarks.length >= 106) {
      result.eyeOpenness = eyeOpennessOf(landmarks);
      final estimated = gaze.estimate(landmarks, result);
      result.gazeX = estimated.$1;
      result.gazeY = estimated.$2;
    }

    _blinkHistory.add(result.eyeOpenness);
    while (_blinkHistory.length > 90) {
      _blinkHistory.removeAt(0);
    }

    if (result.eyeOpenness < 0.18 && !_wasClosed) {
      _wasClosed = true;
      blinkCount += 1;
      result.isBlinking = true;
    } else if (result.eyeOpenness > 0.24) {
      _wasClosed = false;
    }

    result.attention = attentionOf(result);
    return result;
  }

  static double wrap180(double angle) {
    final wrapped = (angle + 180.0) % 360.0;
    return (wrapped < 0 ? wrapped + 360.0 : wrapped) - 180.0;
  }

  static double eyeOpennessOf(List<Point2> lm) {
    final left = <Point2>[for (final i in leftEyeIndices) lm[i]];
    final right = <Point2>[for (final i in rightEyeIndices) lm[i]];
    final lc = centroid(left);
    final rc = centroid(right);
    var iod = lc.distanceTo(rc);
    if (iod == 0.0) iod = 1.0;
    final h = (verticalSpread(left) + verticalSpread(right)) / 2.0;
    return clampd(h / iod / 0.35, 0.0, 1.5);
  }

  static double attentionOf(PoseResult p) {
    final headRaw =
        1.0 - (p.yaw.abs() / 40.0) * 0.6 - (p.pitch.abs() / 35.0) * 0.4;
    final head = headRaw > 0.0 ? headRaw : 0.0;
    final gazeRaw = 1.0 - (p.gazeX.abs() * 0.6 + p.gazeY.abs() * 0.4);
    final gaze = gazeRaw > 0.0 ? gazeRaw : 0.0;
    final eyes = p.eyeOpenness > 0.25 ? 1.0 : p.eyeOpenness / 0.25;
    return clampd(0.5 * head + 0.35 * gaze + 0.15 * eyes, 0.0, 1.0);
  }

  void reset() {
    _blinkHistory.clear();
    blinkCount = 0;
    _wasClosed = false;
  }
}

abstract class SpoofSignalSource {
  Map<String, double> signals();
}

class SpoofDetector {
  SpoofDetector({SpoofConfig? config})
    : config = config ?? const SpoofConfig();

  static const Map<String, double> weights = <String, double>{
    'depth': 0.28,
    'texture': 0.24,
    'colour': 0.14,
    'moire': 0.16,
    'motion': 0.18,
  };

  final SpoofConfig config;

  SpoofResult combine(Map<String, double> signals) {
    if (!config.enabled) {
      return const SpoofResult();
    }
    var live = 0.0;
    for (final entry in weights.entries) {
      live += (signals[entry.key] ?? 0.0) * entry.value;
    }
    live = clampd(live, 0.0, 1.0);
    return SpoofResult(
      liveness: live,
      isSpoof: live < config.threshold,
      signals: signals,
    );
  }

  static double depthFromLandmarks(List<double> depths, double faceHeight) {
    if (depths.isEmpty) return 0.6;
    final spread = populationStd(depths);
    final scale = faceHeight == 0.0 ? 1.0 : faceHeight;
    return clampd((spread / scale) / 0.045, 0.0, 1.0);
  }

  static double depthFromPose(double pitch, double yaw) => clampd(
    0.45 + (yaw.abs() / 30.0) * 0.35 + (pitch.abs() / 30.0) * 0.20,
    0.0,
    1.0,
  );

  static double textureSignal(double laplacianVariance, double highFrequency) =>
      clampd(
        0.6 * math.min(1.0, laplacianVariance / 260.0) +
            0.4 * math.min(1.0, highFrequency / 9.0),
        0.0,
        1.0,
      );

  static double colourSignal(double saturationMean, double valueStd) {
    final sOk = 1.0 - math.min(1.0, (saturationMean - 0.34).abs() / 0.34);
    final vOk = math.min(1.0, valueStd / 0.16);
    return clampd(0.5 * sOk + 0.5 * vOk, 0.0, 1.0);
  }

  static double moireSignal(double peakRatio, double energy) {
    var score = 1.0 - clampd((peakRatio - 2.6) / 3.0, 0.0, 1.0);
    score = 0.7 * score + 0.3 * clampd(energy / 0.65, 0.0, 1.0);
    return clampd(score, 0.0, 1.0);
  }

  static double motionSignal(List<double> frameDifferences) {
    if (frameDifferences.length < 3) return 0.7;
    return clampd(mean(frameDifferences) / 2.2, 0.0, 1.0);
  }
}
