import 'package:engine_body/engine_body.dart';

import 'angles.dart';
import 'bridge.dart';
import 'points.dart';
import 'pose_source.dart';

const List<List<String>> bones = <List<String>>[
  <String>['leftShoulder', 'leftElbow'],
  <String>['leftElbow', 'leftWrist'],
  <String>['rightShoulder', 'rightElbow'],
  <String>['rightElbow', 'rightWrist'],
  <String>['leftHip', 'leftKnee'],
  <String>['leftKnee', 'leftAnkle'],
  <String>['rightHip', 'rightKnee'],
  <String>['rightKnee', 'rightAnkle'],
];

const List<String> head = <String>[
  'nose',
  'leftEye',
  'rightEye',
  'leftEar',
  'rightEar',
];

const double bodyBelieveThreshold = 0.2;
const double drawThreshold = 0.15;
const double minTorsoUnits = 40.0 / 720.0;
const double smoothVisibility = 0.15;
const double smoothMinCutoff = 1.2;
const double smoothBetaPixels = 0.02;

class BodyTracker {
  BodyTracker({
    double believe = bodyBelieveThreshold,
    this.smoothing = true,
  }) : occlusion = OcclusionTracker(believeThreshold: believe);

  final SkeletonCalibrator skeleton = SkeletonCalibrator();
  final OcclusionTracker occlusion;
  final JointAnchor anchor = JointAnchor();
  final bool smoothing;

  PoseSmoother? _smoother;
  double _smootherUnit = 0.0;

  Points? points;
  List<double>? visibility;
  List<bool>? discarded;
  List<bool>? inferred;
  double? torso;

  Points? update(PoseFrame frame) {
    points = null;
    visibility = null;
    discarded = null;
    inferred = null;
    torso = null;

    final FrameScale scale = FrameScale(frame.width, frame.height);
    if (!scale.usable) return null;

    final Points iso = pixelsToIso(frame.points, scale);
    BodyFrame shaped =
        frameFromIso(iso, frame.visibility, scale, frame.timestamp);

    final BodyModel model = skeleton.update(shaped);
    shaped = applyBodyModel(shaped, model);

    final OcclusionReport report = occlusion.update(
      shaped,
      model,
      Duration(microseconds: (frame.timestamp * 1e6).round()),
    );

    BodyFrame held = anchor.update(report.applyTo(shaped), report: report);

    if (smoothing) {
      held = _smootherFor(scale).smooth(held);
    }

    final Points isoOut = isoFromFrame(held, scale, landmarkCount);
    final double torsoUnits = isoTorso(isoOut);

    final List<double> confidence =
        confidenceFromFrame(report.applyTo(held), landmarkCount);
    final List<bool> cut = List<bool>.filled(landmarkCount, false);
    final List<bool> guessed = List<bool>.filled(landmarkCount, false);
    for (final LandmarkType type in LandmarkType.values) {
      final int i = type.index;
      if (i >= landmarkCount) continue;
      final LandmarkVerdict verdict = report.verdictFor(type);
      cut[i] = verdict == LandmarkVerdict.discarded;
      guessed[i] = verdict == LandmarkVerdict.held;
    }

    points = isoToPixels(isoOut, scale);
    visibility = confidence;
    discarded = cut;
    inferred = guessed;
    torso = torsoUnits >= minTorsoUnits ? torsoUnits * scale.unit : null;
    return points;
  }

  PoseSmoother _smootherFor(FrameScale scale) {
    final double unit = scale.isoHeight <= 1e-9 ? 1.0 : scale.isoHeight;
    final PoseSmoother? existing = _smoother;
    if (existing != null && (_smootherUnit - unit).abs() < 1e-9) {
      return existing;
    }
    final PoseSmoother made = PoseSmoother(
      minCutoff: smoothMinCutoff,
      beta: smoothBetaPixels * scale.unit * unit,
      confidenceThreshold: smoothVisibility,
      dropFilterBelowThreshold: true,
    );
    _smoother = made;
    _smootherUnit = unit;
    return made;
  }

  List<List<double>> wrists() {
    final Points? p = points;
    final List<double>? vis = visibility;
    if (p == null || vis == null) return <List<double>>[];
    final List<List<double>> out = <List<double>>[];
    for (final String name in <String>['leftWrist', 'rightWrist']) {
      final int i = idx[name]!;
      if (discarded != null && discarded![i]) continue;
      if (vis[i] < drawThreshold) continue;
      out.add(<double>[p.xs[i], p.ys[i]]);
    }
    return out;
  }

  void reset() {
    skeleton.reset();
    occlusion.reset();
    anchor.reset();
    _smoother?.reset();
    _smoother = null;
    _smootherUnit = 0.0;
    points = null;
    visibility = null;
    discarded = null;
    inferred = null;
    torso = null;
  }
}
