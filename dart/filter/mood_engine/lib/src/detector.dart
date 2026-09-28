import 'dart:math' as math;

import 'config.dart';
import 'geometry.dart';
import 'maths.dart';

class DetectedFace {
  DetectedFace({
    required this.bbox,
    required this.detScore,
    this.embedding,
    List<Point2>? keypoints,
    this.landmarks2d,
    this.landmarks3d,
    this.age,
    this.gender,
    this.pose,
  }) : keypoints = keypoints ?? <Point2>[];

  BoundingBox bbox;
  double detScore;
  List<double>? embedding;
  List<Point2> keypoints;
  List<Point2>? landmarks2d;
  List<Point2>? landmarks3d;
  double? age;
  String? gender;
  List<double>? pose;

  double get width => bbox.width;

  double get height => bbox.height;

  double get area => bbox.area;

  Point2 get centre => bbox.centre;

  BoundingBox cropBox(int frameWidth, int frameHeight, {double margin = 0.2}) {
    final mx = width * margin;
    final my = height * margin;
    final x1 = math.max(0.0, bbox.x1 - mx);
    final y1 = math.max(0.0, bbox.y1 - my);
    final x2 = math.min(frameWidth.toDouble(), bbox.x2 + mx);
    final y2 = math.min(frameHeight.toDouble(), bbox.y2 + my);
    if (x2 <= x1 || y2 <= y1) return const BoundingBox(0, 0, 1, 1);
    return BoundingBox(x1, y1, x2, y2);
  }
}

abstract class FaceDetectionModel {
  List<DetectedFace> detect(Object frame, int width, int height);

  String get name;
}

class FakeFaceDetectionModel implements FaceDetectionModel {
  FakeFaceDetectionModel({List<DetectedFace>? faces})
    : _faces = faces ?? <DetectedFace>[];

  List<DetectedFace> _faces;

  int calls = 0;

  @override
  String get name => 'fake';

  void setFaces(List<DetectedFace> faces) => _faces = faces;

  @override
  List<DetectedFace> detect(Object frame, int width, int height) {
    calls += 1;
    return List<DetectedFace>.from(_faces);
  }
}

class FaceDetector {
  FaceDetector({
    DetectionConfig? config,
    TrackingConfig? tracking,
    this.model,
  }) : config = config ?? const DetectionConfig(),
       tracking = tracking ?? const TrackingConfig(),
       _identityEvery = (config ?? const DetectionConfig())
           .identityEveryNFrames;

  final DetectionConfig config;
  final TrackingConfig tracking;
  final FaceDetectionModel? model;

  int _identityEvery;
  int _identityCounter = 0;
  (List<double>, double?, String?)? _cachedIdentity;
  Point2? _lastCentre;
  int _missing = 0;

  int get identityEvery => _identityEvery;

  (List<double>, double?, String?)? get cachedIdentity => _cachedIdentity;

  void enableIdentityCache(int everyNFrames) {
    _identityEvery = everyNFrames < 1 ? 1 : everyNFrames;
    _identityCounter = 0;
    _cachedIdentity = null;
  }

  bool wantsIdentity() {
    if (_identityEvery <= 1 || _cachedIdentity == null) return true;
    return _identityCounter % _identityEvery == 0;
  }

  static void rescaleFaces(
    List<DetectedFace> faces,
    double scale,
    int width,
    int height,
  ) {
    if (scale == 1.0) return;
    for (final face in faces) {
      face.bbox = face.bbox
          .scaled(scale)
          .clampTo(width.toDouble(), height.toDouble());
      face.keypoints = <Point2>[
        for (final point in face.keypoints) point * scale,
      ];
      final lm2d = face.landmarks2d;
      if (lm2d != null && lm2d.isNotEmpty) {
        face.landmarks2d = <Point2>[for (final p in lm2d) p * scale];
      }
      final lm3d = face.landmarks3d;
      if (lm3d != null && lm3d.isNotEmpty) {
        face.landmarks3d = <Point2>[for (final p in lm3d) p * scale];
      }
    }
  }

  static void shiftFaces(List<DetectedFace> faces, double pad,
      int width, int height) {
    for (final face in faces) {
      face.bbox = face.bbox
          .shifted(-pad, -pad)
          .clampTo(width.toDouble(), height.toDouble());
      face.keypoints = <Point2>[
        for (final point in face.keypoints) Point2(point.x - pad, point.y - pad),
      ];
      final lm2d = face.landmarks2d;
      if (lm2d != null) {
        face.landmarks2d = <Point2>[
          for (final p in lm2d) Point2(p.x - pad, p.y - pad),
        ];
      }
    }
  }

  List<DetectedFace> filterDetections(
    List<DetectedFace> raw,
    int frameWidth,
    int frameHeight,
    int referenceHeight,
  ) {
    final faces = <DetectedFace>[];
    for (final face in raw) {
      if (face.detScore < config.minDetScore) continue;
      if (face.bbox.height < config.minFaceRatio * referenceHeight) continue;

      face.bbox = face.bbox.clampTo(
        frameWidth.toDouble(),
        frameHeight.toDouble(),
      );

      final embedding = face.embedding;
      if (embedding != null) {
        face.embedding = unitNormaliseVector(embedding);
      } else if (_cachedIdentity != null) {
        face.embedding = _cachedIdentity!.$1;
        face.age = _cachedIdentity!.$2;
        face.gender = _cachedIdentity!.$3;
      } else {
        continue;
      }
      faces.add(face);
    }

    if (faces.isNotEmpty) {
      final subject = faces.reduce(
        (DetectedFace a, DetectedFace b) => a.area >= b.area ? a : b,
      );
      if (subject.embedding != null) {
        _cachedIdentity =
            (subject.embedding!, subject.age, subject.gender);
      }
      _identityCounter += 1;
    } else {
      _identityCounter = 0;
      _cachedIdentity = null;
    }

    return faces;
  }

  static List<double> unitNormaliseVector(List<double> vector) {
    var total = 0.0;
    for (final value in vector) {
      total += value * value;
    }
    final norm = math.sqrt(total);
    if (norm <= 0) return vector;
    return <double>[for (final value in vector) value / norm];
  }

  DetectedFace? selectSubject(
    List<DetectedFace> faces,
    int frameWidth,
    int frameHeight,
  ) {
    if (faces.isEmpty) {
      _missing += 1;
      if (_missing > tracking.maxMissingFrames) {
        _lastCentre = null;
        _missing = 0;
      }
      return null;
    }

    _missing = 0;
    var largest = faces.reduce(
      (DetectedFace a, DetectedFace b) => a.area >= b.area ? a : b,
    );

    final last = _lastCentre;
    if (last != null && faces.length > 1) {
      final diag = math.sqrt(
        frameWidth * frameWidth + frameHeight * frameHeight,
      );
      final limit = tracking.maxCentreDrift * diag;
      final near = <DetectedFace>[
        for (final face in faces)
          if (face.centre.distanceTo(last) <= limit) face,
      ];
      if (near.isNotEmpty) {
        final tracked = near.reduce(
          (DetectedFace a, DetectedFace b) => a.area >= b.area ? a : b,
        );
        if (largest.area <= tracked.area * 1.35) largest = tracked;
      }
    }

    _lastCentre = largest.centre;
    return largest;
  }

  void resetTracking() {
    _lastCentre = null;
    _missing = 0;
    _identityCounter = 0;
    _cachedIdentity = null;
  }

  bool mayBeCroppedPortrait(int width, int height, double pixelStd) {
    if (height == 0 || width == 0) return false;
    final aspect = math.max(width / height, height / width);
    if (aspect > 1.7) return false;
    if (pixelStd < 12.0) return false;
    return true;
  }

  static double alignmentAngle(Point2 leftEye, Point2 rightEye) {
    final delta = rightEye - leftEye;
    return degreesOf(math.atan2(delta.y, delta.x));
  }

  static double alignmentScale(
    Point2 leftEye,
    Point2 rightEye,
    int size,
    double eyeRatio,
  ) {
    final delta = rightEye - leftEye;
    final distance = math.sqrt(delta.x * delta.x + delta.y * delta.y);
    if (distance < 1e-3) return 0.0;
    return eyeRatio * size / distance;
  }

  static double degreesOf(double radians) => radians * 180.0 / math.pi;

  static double qualityScore({
    required double faceHeight,
    required int frameHeight,
    required double detScore,
    List<double>? pose,
    double sharpness = 0.0,
  }) {
    final size = math.min(1.0, faceHeight / (0.45 * frameHeight));
    final conf = math.min(1.0, detScore);

    var frontality = 1.0;
    if (pose != null && pose.length >= 2) {
      final pitch = pose[0];
      final yaw = pose[1];
      frontality = math.max(
        0.0,
        1.0 - (yaw.abs() / 45.0) * 0.7 - (pitch.abs() / 45.0) * 0.3,
      );
    }

    final sharp = math.min(1.0, sharpness / 220.0);

    return clampd(
      0.30 * conf + 0.25 * size + 0.25 * frontality + 0.20 * sharp,
      0.0,
      1.0,
    );
  }
}
