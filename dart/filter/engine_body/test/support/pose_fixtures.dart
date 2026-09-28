import 'package:engine_body/engine_body.dart';

class PoseFixtures {
  static List<BodyLandmark> standing({
    double scale = 1.0,
    double centreX = 0.5,
    double centreY = 0.5,
    double confidence = 0.9,
    bool legsVisible = true,
  }) {
    const shoulderY = -0.20;
    const hipY = 0.0;
    const shoulderHalfWidth = 0.09;
    const hipHalfWidth = 0.06;

    BodyLandmark at(
      LandmarkType type,
      double dx,
      double dy, {
      double? likelihood,
    }) {
      final c = likelihood ?? confidence;
      return BodyLandmark(
        type: type,
        x: centreX + dx * scale,
        y: centreY + dy * scale,
        z: 0,
        likelihood: c,
        inFrameLikelihood: c,
      );
    }

    final legConfidence = legsVisible ? confidence : 0.1;

    return [
      at(LandmarkType.nose, 0, shoulderY - 0.13),
      at(LandmarkType.leftEye, 0.025, shoulderY - 0.15),
      at(LandmarkType.rightEye, -0.025, shoulderY - 0.15),
      at(LandmarkType.leftEar, 0.05, shoulderY - 0.145),
      at(LandmarkType.rightEar, -0.05, shoulderY - 0.145),
      at(LandmarkType.leftMouth, 0.02, shoulderY - 0.10),
      at(LandmarkType.rightMouth, -0.02, shoulderY - 0.10),
      at(LandmarkType.leftShoulder, shoulderHalfWidth, shoulderY),
      at(LandmarkType.rightShoulder, -shoulderHalfWidth, shoulderY),
      at(LandmarkType.leftElbow, shoulderHalfWidth + 0.01, shoulderY + 0.11),
      at(LandmarkType.rightElbow, -shoulderHalfWidth - 0.01, shoulderY + 0.11),
      at(LandmarkType.leftWrist, shoulderHalfWidth + 0.02, shoulderY + 0.22),
      at(LandmarkType.rightWrist, -shoulderHalfWidth - 0.02, shoulderY + 0.22),
      at(LandmarkType.leftThumb, shoulderHalfWidth + 0.025, shoulderY + 0.25),
      at(LandmarkType.rightThumb, -shoulderHalfWidth - 0.025, shoulderY + 0.25),
      at(LandmarkType.leftIndex, shoulderHalfWidth + 0.03, shoulderY + 0.26),
      at(LandmarkType.rightIndex, -shoulderHalfWidth - 0.03, shoulderY + 0.26),
      at(LandmarkType.leftPinky, shoulderHalfWidth + 0.02, shoulderY + 0.26),
      at(LandmarkType.rightPinky, -shoulderHalfWidth - 0.02, shoulderY + 0.26),
      at(LandmarkType.leftHip, hipHalfWidth, hipY),
      at(LandmarkType.rightHip, -hipHalfWidth, hipY),
      at(LandmarkType.leftKnee, hipHalfWidth, 0.22, likelihood: legConfidence),
      at(LandmarkType.rightKnee, -hipHalfWidth, 0.22, likelihood: legConfidence),
      at(LandmarkType.leftAnkle, hipHalfWidth, 0.44, likelihood: legConfidence),
      at(
        LandmarkType.rightAnkle,
        -hipHalfWidth,
        0.44,
        likelihood: legConfidence,
      ),
      at(LandmarkType.leftHeel, hipHalfWidth, 0.46, likelihood: legConfidence),
      at(
        LandmarkType.rightHeel,
        -hipHalfWidth,
        0.46,
        likelihood: legConfidence,
      ),
      at(
        LandmarkType.leftFootIndex,
        hipHalfWidth + 0.03,
        0.47,
        likelihood: legConfidence,
      ),
      at(
        LandmarkType.rightFootIndex,
        -hipHalfWidth - 0.03,
        0.47,
        likelihood: legConfidence,
      ),
      at(LandmarkType.leftEyeInner, 0.015, shoulderY - 0.15, likelihood: 0.6),
      at(LandmarkType.leftEyeOuter, 0.035, shoulderY - 0.15, likelihood: 0.6),
      at(LandmarkType.rightEyeInner, -0.015, shoulderY - 0.15, likelihood: 0.6),
      at(LandmarkType.rightEyeOuter, -0.035, shoulderY - 0.15, likelihood: 0.6),
    ];
  }

  static List<BodyLandmark> leftArmRaised({
    double scale = 1.0,
    double centreX = 0.5,
    double centreY = 0.5,
  }) {
    final base = standing(scale: scale, centreX: centreX, centreY: centreY);
    final byType = {for (final l in base) l.type: l};

    final shoulder = byType[LandmarkType.leftShoulder]!;
    byType[LandmarkType.leftElbow] = byType[LandmarkType.leftElbow]!.copyWith(
      x: shoulder.x + 0.11 * scale,
      y: shoulder.y,
    );
    byType[LandmarkType.leftWrist] = byType[LandmarkType.leftWrist]!.copyWith(
      x: shoulder.x + 0.22 * scale,
      y: shoulder.y,
    );
    return byType.values.toList();
  }

  static BodyFrame frame(
    List<BodyLandmark> landmarks, {
    required DateTime at,
    FramingGate gate = const FramingGate(),
  }) => BodyFrame(
    timestamp: at,
    landmarks: landmarks,
    framing: gate.assess(landmarks),
    isMirrored: false,
  );

  static List<BodyFrame> stillSequence({
    int frames = 30,
    int fps = 15,
    DateTime? start,
  }) {
    final t0 = start ?? DateTime(2026, 1, 1);
    final pose = standing();
    return [
      for (var i = 0; i < frames; i++)
        frame(pose, at: t0.add(Duration(microseconds: (1e6 ~/ fps) * i))),
    ];
  }
}
