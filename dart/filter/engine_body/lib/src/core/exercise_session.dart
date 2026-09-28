import 'anchor.dart';
import 'body_frame.dart';
import 'body_guards.dart';
import 'form_score.dart';
import 'hold_detector.dart';
import 'joint_angle.dart';
import 'jump_detector.dart';
import 'occlusion.dart';
import 'reliability.dart';
import 'rep_counter.dart';
import 'skeleton.dart';
import 'velocity_reps.dart';

class ExerciseSession {
  ExerciseSession({
    required this.exercise,
    HysteresisConfig? calibrated,
    this.checkReliability = true,
    this.maxYawDegrees = 35,
    this.detectJumps = true,
  })  : _repCounter = exercise.kind == ExerciseKind.reps
            ? RepCounter(calibrated ?? exercise.config(),
                inverted: exercise.inverted)
            : null,
        _velocityCounter =
            exercise.kind == ExerciseKind.reps ? VelocityRepCounter() : null,
        _holdDetector =
            exercise.kind == ExerciseKind.hold ? HoldDetector() : null;

  final Exercise exercise;

  final bool checkReliability;

  final double maxYawDegrees;

  final bool detectJumps;

  final RepCounter? _repCounter;
  final VelocityRepCounter? _velocityCounter;
  final HoldDetector? _holdDetector;

  final _sideTracker = SideTracker();

  final _skeleton = SkeletonCalibrator();

  final _occlusion = OcclusionTracker();

  final _anchor = JointAnchor();

  final _angleGuard = AngleRateGuard();

  final _reliability = ReliabilityMonitor();
  final _collector = RepetitionCollector();
  final _form = FormAnalyser();
  final _subjectSwitch = SubjectSwitchDetector();
  final _jumps = JumpDetector();

  List<JumpEvent> get jumps => _jumps.jumps;

  int get repCount => _velocityCounter?.count ?? 0;

  int get depthCount => _repCounter?.count ?? 0;

  int get partialCount => _repCounter?.partialCount ?? 0;

  List<HoldEvent> get holds => _holdDetector?.holds ?? const [];

  Duration get totalHeld => _holdDetector?.totalHeld ?? Duration.zero;

  double? get consistency => _form.consistency;

  String? get side => _sideTracker.side;

  static bool _hasDepth(BodyFrame frame) {
    for (final l in frame.landmarks) {
      if (l.z != 0 && l.z.isFinite) return true;
    }
    return false;
  }

  bool _jointVisible(OcclusionReport report, String joint) {
    for (final side in const ['left', 'right']) {
      final definition = trackedJoints['$side$joint'];
      if (definition == null) continue;
      if (report.isDrawable(definition.from) &&
          report.isDrawable(definition.vertex) &&
          report.isDrawable(definition.to)) {
        return true;
      }
    }
    return false;
  }

  ExerciseUpdate update(BodyFrame rawFrame, Duration timestamp) {
    final model = _skeleton.update(rawFrame);
    final modelled = applyBodyModel(rawFrame, model);
    final occlusionReport = _occlusion.update(modelled, model, timestamp);
    final frame = _anchor.update(
      occlusionReport.applyTo(modelled),
      report: occlusionReport,
    );

    final torso = torsoLength(frame);

    if (_subjectSwitch(frame, timestamp)) {
      _repCounter?.reset();
      _velocityCounter?.reset();
      _holdDetector?.reset();
      _reliability.reset();
      _collector.reset();
      _jumps.reset();
      _skeleton.reset();
      _occlusion.reset();
      _anchor.reset();
      _angleGuard.reset();
      return ExerciseUpdate(
        usable: false,
        reason: 'someone else came into view',
        repCount: repCount,
        depthCount: depthCount,
        partialCount: partialCount,
        subjectChanged: true,
      );
    }

    final yaw = viewpointYawDegrees(frame);
    if (yaw != null && yaw > maxYawDegrees) {
      return ExerciseUpdate(
        usable: false,
        reason: 'turn to face the camera',
        repCount: repCount,
        depthCount: depthCount,
        partialCount: partialCount,
        yawDegrees: yaw,
      );
    }

    if (checkReliability) {
      final side = _sideTracker.side;
      double? angle2d;
      double? angle3dValue;
      if (side != null && _hasDepth(rawFrame)) {
        final definition = trackedJoints['$side${exercise.joint}'];
        if (definition != null) {
          angle2d = frame
              .angleAt(definition.from, definition.vertex, definition.to)
              ?.orNull;
          angle3dValue = angle3d(
            rawFrame,
            definition.from,
            definition.vertex,
            definition.to,
          );
        }
      }

      final trust = _reliability.update(
        frame,
        torso,
        timestamp,
        angle2d: angle2d,
        angle3dValue: angle3dValue,
      );
      if (!trust.trustworthy) {
        return ExerciseUpdate(
          usable: false,
          reason: trust.reasons.join('; '),
          repCount: repCount,
          depthCount: depthCount,
          partialCount: partialCount,
          yawDegrees: yaw,
        );
      }
    }

    if (!_jointVisible(occlusionReport, exercise.joint)) {
      return ExerciseUpdate(
        usable: false,
        reason: 'step back — I cannot see your '
            '${exercise.joint.toLowerCase()}',
        repCount: repCount,
        depthCount: depthCount,
        partialCount: partialCount,
        yawDegrees: yaw,
      );
    }

    final jump = detectJumps ? _jumps.update(frame, timestamp) : null;

    final angle = _sideTracker.update(
      _angleGuard.filter(frame.allAngles, timestamp),
      exercise.joint,
    );
    final value = angle?.orNull;

    if (_holdDetector != null) {
      final state = _holdDetector.update(frame, torso, timestamp, angle: value);
      return ExerciseUpdate(
        usable: true,
        angle: value,
        holding: state.holding,
        elapsed: state.elapsed,
        completedHold: state.completed,
        reason: state.reason,
        yawDegrees: yaw,
        jump: jump,
        jumpCount: _jumps.count,
      );
    }

    final rep = _repCounter!.update(value, timestamp);
    _collector.add(value, timestamp);

    final velocity = _velocityCounter!.update(value, timestamp);
    FormReport? form;
    if (velocity.completed) {
      final record = _collector.close(timestamp);
      if (record != null) form = _form.add(record);
    }

    return ExerciseUpdate(
      usable: true,
      angle: value,
      state: rep.state,
      repCount: velocity.count,
      depthCount: rep.count,
      partialCount: rep.partialCount,
      repCompleted: velocity.completed,
      depthReached: rep.completed,
      tooShallow: rep.partial,
      form: form,
      yawDegrees: yaw,
      jump: jump,
      jumpCount: _jumps.count,
    );
  }

  HoldEvent? finish(Duration timestamp) => _holdDetector?.finish(timestamp);

  void reset({bool keepCounts = false}) {
    _repCounter?.reset(keepCounts: keepCounts);
    _velocityCounter?.reset(keepCounts: keepCounts);
    _holdDetector?.reset(keepHolds: keepCounts);
    _sideTracker.reset();
    _reliability.reset();
    _collector.reset();
    if (!keepCounts) _form.reset();
  }
}

class ExerciseUpdate {
  const ExerciseUpdate({
    required this.usable,
    this.reason = '',
    this.angle,
    this.state = RepState.between,
    this.repCount = 0,
    this.depthCount = 0,
    this.partialCount = 0,
    this.repCompleted = false,
    this.depthReached = false,
    this.tooShallow = false,
    this.form,
    this.holding = false,
    this.elapsed = Duration.zero,
    this.completedHold,
    this.yawDegrees,
    this.jump,
    this.jumpCount = 0,
    this.subjectChanged = false,
  });

  final bool usable;

  final String reason;

  final double? angle;
  final RepState state;

  final int repCount;

  final int depthCount;

  final int partialCount;

  final bool repCompleted;

  final bool depthReached;

  final bool tooShallow;

  final FormReport? form;

  final bool holding;
  final Duration elapsed;
  final HoldEvent? completedHold;

  final double? yawDegrees;

  final JumpEvent? jump;

  final int jumpCount;

  final bool subjectChanged;
}
