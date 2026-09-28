enum PoseModel {
  fast,

  accurate,
}

enum CameraFacing {
  front,

  back,
}

class TrackerConfig {
  const TrackerConfig({
    this.model = PoseModel.fast,
    this.facing = CameraFacing.front,
    this.targetFps = 15,
    this.analysisWidth = 480,
  });

  final PoseModel model;
  final CameraFacing facing;

  final int targetFps;

  final int analysisWidth;

  Map<String, Object?> toMap() => {
    'model': model.name,
    'facing': facing.name,
    'targetFps': targetFps,
    'analysisWidth': analysisWidth,
  };

  TrackerConfig copyWith({
    PoseModel? model,
    CameraFacing? facing,
    int? targetFps,
    int? analysisWidth,
  }) => TrackerConfig(
    model: model ?? this.model,
    facing: facing ?? this.facing,
    targetFps: targetFps ?? this.targetFps,
    analysisWidth: analysisWidth ?? this.analysisWidth,
  );
}

class PreviewInfo {
  const PreviewInfo({
    required this.textureId,
    required this.width,
    required this.height,
    required this.rotation,
    required this.isMirrored,
  });

  final int textureId;

  final int width;
  final int height;

  final int rotation;

  final bool isMirrored;

  bool get isQuarterTurned => rotation == 90 || rotation == 270;

  int get displayWidth => isQuarterTurned ? height : width;

  int get displayHeight => isQuarterTurned ? width : height;

  double get displayAspectRatio =>
      displayHeight == 0 ? 1 : displayWidth / displayHeight;

  static PreviewInfo? fromMap(Map<Object?, Object?>? map) {
    if (map == null) return null;
    final id = map['textureId'];
    if (id is! num) return null;
    return PreviewInfo(
      textureId: id.toInt(),
      width: (map['previewWidth'] as num?)?.toInt() ?? 0,
      height: (map['previewHeight'] as num?)?.toInt() ?? 0,
      rotation: (map['rotation'] as num?)?.toInt() ?? 0,
      isMirrored: map['isMirrored'] == true,
    );
  }

  @override
  String toString() =>
      'PreviewInfo(texture: $textureId, ${width}x$height raw, '
      '${displayWidth}x$displayHeight upright, '
      'rotation: $rotation, mirrored: $isMirrored)';
}

enum TrackerError {
  cameraPermissionDenied,
  cameraUnavailable,
  modelUnavailable,
  platformUnsupported,
  unknown,
}

class BodyTrackerException implements Exception {
  const BodyTrackerException(this.error, this.message);

  final TrackerError error;
  final String message;

  @override
  String toString() => 'BodyTrackerException(${error.name}): $message';
}
