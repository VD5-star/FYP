import 'body_landmark.dart';
import 'landmark_type.dart';

enum Framing {
  full,

  upperBodyOnly,

  partial,

  noSubject;

  bool get hasTorso => this == full || this == upperBodyOnly;

  bool get hasLegs => this == full;
}

enum FramingAdvice {
  none,

  moveBack,

  stepIntoView,

  faceTheCamera,
}

class FramingReport {
  const FramingReport({
    required this.framing,
    required this.advice,
    required this.torsoQuality,
    required this.upperBodyQuality,
    required this.lowerBodyQuality,
  });

  static const noSubject = FramingReport(
    framing: Framing.noSubject,
    advice: FramingAdvice.stepIntoView,
    torsoQuality: 0,
    upperBodyQuality: 0,
    lowerBodyQuality: 0,
  );

  final Framing framing;
  final FramingAdvice advice;

  final double torsoQuality;

  final double upperBodyQuality;
  final double lowerBodyQuality;

  double qualityFor({bool needsLegs = false}) =>
      needsLegs ? lowerBodyQuality * upperBodyQuality : upperBodyQuality;

  Map<String, Object?> toMap() => {
    'framing': framing.name,
    'advice': advice.name,
    'torsoQuality': torsoQuality,
    'upperBodyQuality': upperBodyQuality,
    'lowerBodyQuality': lowerBodyQuality,
  };

  @override
  String toString() =>
      'FramingReport(${framing.name}, torso=${torsoQuality.toStringAsFixed(2)}, '
      'upper=${upperBodyQuality.toStringAsFixed(2)}, '
      'lower=${lowerBodyQuality.toStringAsFixed(2)})';
}

class FramingGate {
  const FramingGate({
    this.landmarkThreshold = 0.5,
    this.groupThreshold = 0.6,
  });

  final double landmarkThreshold;

  final double groupThreshold;

  FramingReport assess(List<BodyLandmark> landmarks) {
    if (landmarks.isEmpty) return FramingReport.noSubject;

    final byType = {for (final l in landmarks) l.type: l};

    final torso = _meanConfidence(byType, torsoLandmarks);
    final upper = _meanConfidence(byType, upperBodyLandmarks);
    final lower = _meanConfidence(byType, lowerBodyLandmarks);

    if (torso < groupThreshold) {
      return FramingReport(
        framing: Framing.partial,
        advice: upper > 0.3
            ? FramingAdvice.faceTheCamera
            : FramingAdvice.stepIntoView,
        torsoQuality: torso,
        upperBodyQuality: upper,
        lowerBodyQuality: lower,
      );
    }

    final hasLegs = lower >= groupThreshold;
    return FramingReport(
      framing: hasLegs ? Framing.full : Framing.upperBodyOnly,
      advice: hasLegs ? FramingAdvice.none : FramingAdvice.moveBack,
      torsoQuality: torso,
      upperBodyQuality: upper,
      lowerBodyQuality: lower,
    );
  }

  double _meanConfidence(
    Map<LandmarkType, BodyLandmark> byType,
    Set<LandmarkType> group,
  ) {
    if (group.isEmpty) return 0;
    var sum = 0.0;
    for (final type in group) {
      final l = byType[type];
      if (l == null) continue;
      sum += l.likelihood * l.inFrameLikelihood;
    }
    return sum / group.length;
  }
}
