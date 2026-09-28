import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';
import 'framing.dart';
import 'landmark_type.dart';

const double syntheticUnit = 1000.0;

const double syntheticTorsoPx = 200.0;

const double syntheticFemurPx = 200.0;

const double syntheticTibiaPx = 180.0;

const double syntheticAnklePx = 880.0;

const double syntheticArmPx = 110.0;

const double syntheticKneeRangeDegrees = 105.0;

const double syntheticShinLean = 0.45;

double syntheticKneeAngle(double kneeBend) =>
    180.0 - kneeBend * syntheticKneeRangeDegrees;

class GaussianNoise {
  GaussianNoise(this.seed) : _random = math.Random(seed);

  final int seed;

  math.Random _random;
  double? _spare;

  double next() {
    final spare = _spare;
    if (spare != null) {
      _spare = null;
      return spare;
    }
    double u, v, s;
    do {
      u = _random.nextDouble() * 2 - 1;
      v = _random.nextDouble() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s == 0);
    final f = math.sqrt(-2 * math.log(s) / s);
    _spare = v * f;
    return u * f;
  }

  void reset() {
    _random = math.Random(seed);
    _spare = null;
  }
}

class SyntheticPose {
  const SyntheticPose._();

  static Map<LandmarkType, ({double x, double y})> offsets({
    double kneeBend = 0,
    double armRaise = 0,
  }) {
    final kneeAngle = syntheticKneeAngle(kneeBend) * math.pi / 180.0;
    final shinLean = kneeBend * syntheticShinLean;

    final kneeY = syntheticAnklePx - syntheticTibiaPx * math.cos(shinLean);
    final kneeForward = syntheticTibiaPx * math.sin(shinLean);
    final thighLean = math.pi - kneeAngle - shinLean;
    final hipY = kneeY - syntheticFemurPx * math.cos(thighLean);
    final hipBack = kneeForward - syntheticFemurPx * math.sin(thighLean);
    final shoulderY = hipY - syntheticTorsoPx;

    final out = <LandmarkType, ({double x, double y})>{
      LandmarkType.nose: (x: 0, y: shoulderY - 130),
      LandmarkType.leftEye: (x: 22, y: shoulderY - 145),
      LandmarkType.rightEye: (x: -22, y: shoulderY - 145),
      LandmarkType.leftEyeInner: (x: 12, y: shoulderY - 145),
      LandmarkType.rightEyeInner: (x: -12, y: shoulderY - 145),
      LandmarkType.leftEyeOuter: (x: 32, y: shoulderY - 145),
      LandmarkType.rightEyeOuter: (x: -32, y: shoulderY - 145),
      LandmarkType.leftEar: (x: 46, y: shoulderY - 138),
      LandmarkType.rightEar: (x: -46, y: shoulderY - 138),
      LandmarkType.leftMouth: (x: 16, y: shoulderY - 100),
      LandmarkType.rightMouth: (x: -16, y: shoulderY - 100),
      LandmarkType.leftShoulder: (x: 80, y: shoulderY),
      LandmarkType.rightShoulder: (x: -80, y: shoulderY),
      LandmarkType.leftHip: (x: 55 + hipBack, y: hipY),
      LandmarkType.rightHip: (x: -55 + hipBack, y: hipY),
      LandmarkType.leftKnee: (x: 55 + kneeForward, y: kneeY),
      LandmarkType.rightKnee: (x: -55 + kneeForward, y: kneeY),
      LandmarkType.leftAnkle: (x: 55, y: syntheticAnklePx),
      LandmarkType.rightAnkle: (x: -55, y: syntheticAnklePx),
      LandmarkType.leftHeel: (x: 50, y: syntheticAnklePx + 16),
      LandmarkType.rightHeel: (x: -50, y: syntheticAnklePx + 16),
      LandmarkType.leftFootIndex: (x: 85, y: syntheticAnklePx + 12),
      LandmarkType.rightFootIndex: (x: -85, y: syntheticAnklePx + 12),
    };

    final armAngle = (90 - armRaise * 90) * math.pi / 180.0;
    const sides = <(LandmarkType, LandmarkType, LandmarkType, LandmarkType,
        LandmarkType, double)>[
      (
        LandmarkType.leftElbow,
        LandmarkType.leftWrist,
        LandmarkType.leftThumb,
        LandmarkType.leftIndex,
        LandmarkType.leftPinky,
        1.0,
      ),
      (
        LandmarkType.rightElbow,
        LandmarkType.rightWrist,
        LandmarkType.rightThumb,
        LandmarkType.rightIndex,
        LandmarkType.rightPinky,
        -1.0,
      ),
    ];

    for (final (elbow, wrist, thumb, index, pinky, sign) in sides) {
      final sx = 80 * sign;
      final ex = sx + sign * syntheticArmPx * math.cos(armAngle) * 0.55;
      final ey = shoulderY + syntheticArmPx * math.sin(armAngle) * 0.55;
      final wx = ex + sign * syntheticArmPx * math.cos(armAngle) * 0.75;
      final wy = ey + syntheticArmPx * math.sin(armAngle) * 0.75;

      out[elbow] = (x: ex, y: ey);
      out[wrist] = (x: wx, y: wy);
      out[thumb] = (x: wx + sign * 10, y: wy + 12);
      out[index] = (x: wx + sign * 16, y: wy + 16);
      out[pinky] = (x: wx + sign * 6, y: wy + 18);
    }

    return out;
  }

  static List<BodyLandmark> landmarks({
    double kneeBend = 0,
    double armRaise = 0,
    double rise = 0,
    double scale = 1,
    double centreX = 0.5,
    double confidence = 1,
    Map<LandmarkType, double> likelihoods = const {},
    double noise = 0,
    GaussianNoise? random,
  }) {
    final raw = offsets(kneeBend: kneeBend, armRaise: armRaise);
    final jitter = noise == 0 ? null : (random ?? GaussianNoise(0));

    final out = <BodyLandmark>[];
    for (final type in LandmarkType.values) {
      final point = raw[type];
      if (point == null) continue;

      var x = centreX + point.x * scale / syntheticUnit;
      var y = (point.y + rise) * scale / syntheticUnit;
      if (jitter != null) {
        x += jitter.next() * noise / syntheticUnit;
        y += jitter.next() * noise / syntheticUnit;
      }

      final p = likelihoods[type] ?? confidence;
      out.add(BodyLandmark(
        type: type,
        x: x,
        y: y,
        z: 0,
        likelihood: p,
        inFrameLikelihood: p,
      ));
    }
    return out;
  }

  static BodyFrame frame({
    double kneeBend = 0,
    double armRaise = 0,
    double rise = 0,
    double scale = 1,
    double centreX = 0.5,
    double confidence = 1,
    Map<LandmarkType, double> likelihoods = const {},
    double noise = 0,
    GaussianNoise? random,
    DateTime? at,
    FramingGate gate = const FramingGate(),
    bool isMirrored = false,
  }) {
    final points = landmarks(
      kneeBend: kneeBend,
      armRaise: armRaise,
      rise: rise,
      scale: scale,
      centreX: centreX,
      confidence: confidence,
      likelihoods: likelihoods,
      noise: noise,
      random: random,
    );
    return BodyFrame(
      timestamp: at ?? DateTime.fromMillisecondsSinceEpoch(0),
      landmarks: points,
      framing: gate.assess(points),
      isMirrored: isMirrored,
    );
  }

  static double kneeAngleOf(BodyFrame frame) {
    final h = frame[LandmarkType.leftHip];
    final k = frame[LandmarkType.leftKnee];
    final a = frame[LandmarkType.leftAnkle];
    if (h == null || k == null || a == null) return double.nan;

    final v1x = h.x - k.x;
    final v1y = h.y - k.y;
    final v2x = a.x - k.x;
    final v2y = a.y - k.y;
    final n1 = math.sqrt(v1x * v1x + v1y * v1y);
    final n2 = math.sqrt(v2x * v2x + v2y * v2y);
    if (n1 < 1e-12 || n2 < 1e-12) return double.nan;

    final cosine = (v1x * v2x + v1y * v2y) / (n1 * n2);
    return math.acos(cosine.clamp(-1.0, 1.0)) * 180 / math.pi;
  }
}

enum SyntheticMovement { still, squat, armRaise, walkInPlace, sitDownOnce }

class SyntheticPoseSource {
  SyntheticPoseSource({
    this.movement = SyntheticMovement.squat,
    this.fps = 30,
    this.framesPerCycle = 36,
    this.depth = 0.95,
    this.noise = 0,
    this.scale = 1,
    this.centreX = 0.5,
    int seed = 0,
    DateTime? startedAt,
  })  : _random = GaussianNoise(seed),
        _startedAt = startedAt ?? DateTime.fromMillisecondsSinceEpoch(0);

  final SyntheticMovement movement;

  final int fps;

  final int framesPerCycle;

  final double depth;

  final double noise;

  final double scale;

  final double centreX;

  final GaussianNoise _random;
  final DateTime _startedAt;

  int _frame = 0;

  int get frameIndex => _frame;

  Duration elapsedAt(int frame) =>
      Duration(microseconds: (frame * 1000000 / fps).round());

  Duration get elapsed => elapsedAt(_frame);

  double bendAt(int frame) {
    switch (movement) {
      case SyntheticMovement.still:
        return 0;
      case SyntheticMovement.squat:
        final phase = (math.sin(2 * math.pi * (frame % framesPerCycle) /
                    framesPerCycle -
                math.pi / 2) +
            1) /
            2;
        return depth * phase;
      case SyntheticMovement.armRaise:
        return 0;
      case SyntheticMovement.walkInPlace:
        return 0.25 + 0.12 * math.sin(2 * math.pi * frame / 20);
      case SyntheticMovement.sitDownOnce:
        return frame < 120 ? math.min(depth, frame / 40) : depth;
    }
  }

  double raiseAt(int frame) {
    if (movement != SyntheticMovement.armRaise) return 0;
    final phase = (math.sin(2 * math.pi * (frame % framesPerCycle) /
                framesPerCycle -
            math.pi / 2) +
        1) /
        2;
    return phase;
  }

  BodyFrame next() {
    final frame = frameAt(_frame);
    _frame++;
    return frame;
  }

  BodyFrame frameAt(int frame) => SyntheticPose.frame(
        kneeBend: bendAt(frame),
        armRaise: raiseAt(frame),
        scale: scale,
        centreX: centreX,
        noise: noise,
        random: _random,
        at: _startedAt.add(elapsedAt(frame)),
      );

  List<BodyFrame> take(int count) =>
      [for (var i = 0; i < count; i++) next()];

  Stream<BodyFrame> stream({int? frames}) async* {
    var emitted = 0;
    while (frames == null || emitted < frames) {
      yield next();
      emitted++;
    }
  }

  void reset() {
    _frame = 0;
    _random.reset();
  }
}
