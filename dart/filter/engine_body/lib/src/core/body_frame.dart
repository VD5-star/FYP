import 'body_landmark.dart';
import 'framing.dart';
import 'landmark_type.dart';
import 'normalisation.dart';

class BodyFrame {
  BodyFrame({
    required this.timestamp,
    required this.landmarks,
    required this.framing,
    required this.isMirrored,
    this.trackingId,
  }) : _byType = {for (final l in landmarks) l.type: l};

  final DateTime timestamp;

  final List<BodyLandmark> landmarks;

  final FramingReport framing;

  final bool isMirrored;

  final int? trackingId;

  final Map<LandmarkType, BodyLandmark> _byType;

  static final empty = BodyFrame(
    timestamp: DateTime.fromMillisecondsSinceEpoch(0),
    landmarks: const [],
    framing: FramingReport.noSubject,
    isMirrored: false,
  );

  BodyLandmark? operator [](LandmarkType type) => _byType[type];

  bool get hasSubject => framing.framing != Framing.noSubject;

  TorsoFrame? get torso => _torso ??= TorsoFrame.from(_byType);
  TorsoFrame? _torso;

  Map<LandmarkType, Vec2>? get normalised {
    final frame = torso;
    if (frame == null) return null;
    return _normalised ??= frame.normaliseAll(landmarks);
  }

  Map<LandmarkType, Vec2>? _normalised;

  double? jointAngle(
    LandmarkType vertex,
    LandmarkType a,
    LandmarkType b, {
    double threshold = 0.5,
  }) {
    final points = normalised;
    if (points == null) return null;

    for (final type in [vertex, a, b]) {
      final landmark = _byType[type];
      if (landmark == null || !landmark.isReliable(threshold)) return null;
    }

    final v = points[vertex];
    final pa = points[a];
    final pb = points[b];
    if (v == null || pa == null || pb == null) return null;

    return angleAt(v, pa, pb);
  }

  double? get leftElbowAngle => jointAngle(
    LandmarkType.leftElbow,
    LandmarkType.leftShoulder,
    LandmarkType.leftWrist,
  );

  double? get rightElbowAngle => jointAngle(
    LandmarkType.rightElbow,
    LandmarkType.rightShoulder,
    LandmarkType.rightWrist,
  );

  double? get leftShoulderAngle => jointAngle(
    LandmarkType.leftShoulder,
    LandmarkType.leftHip,
    LandmarkType.leftElbow,
  );

  double? get rightShoulderAngle => jointAngle(
    LandmarkType.rightShoulder,
    LandmarkType.rightHip,
    LandmarkType.rightElbow,
  );

  double? get leftKneeAngle => jointAngle(
    LandmarkType.leftKnee,
    LandmarkType.leftHip,
    LandmarkType.leftAnkle,
  );

  double? get rightKneeAngle => jointAngle(
    LandmarkType.rightKnee,
    LandmarkType.rightHip,
    LandmarkType.rightAnkle,
  );

  static BodyFrame fromMap(
    Map<Object?, Object?> map, {
    FramingGate gate = const FramingGate(),
  }) {
    final rawLandmarks = map['landmarks'];
    final landmarks = <BodyLandmark>[];
    if (rawLandmarks is List) {
      for (final entry in rawLandmarks) {
        if (entry is Map<Object?, Object?>) {
          final landmark = BodyLandmark.fromMap(entry);
          if (landmark != null) landmarks.add(landmark);
        }
      }
    }

    final micros = map['timestampMicros'];
    return BodyFrame(
      timestamp: micros is num
          ? DateTime.fromMicrosecondsSinceEpoch(micros.toInt())
          : DateTime.now(),
      landmarks: landmarks,
      framing: gate.assess(landmarks),
      isMirrored: map['isMirrored'] == true,
      trackingId: map['trackingId'] is num
          ? (map['trackingId']! as num).toInt()
          : null,
    );
  }

  Map<String, Object?> toMap() => {
    'timestampMicros': timestamp.microsecondsSinceEpoch,
    'isMirrored': isMirrored,
    if (trackingId != null) 'trackingId': trackingId,
    'landmarks': [for (final l in landmarks) l.toMap()],
  };

  @override
  String toString() =>
      'BodyFrame(${landmarks.length} landmarks, ${framing.framing.name})';
}
