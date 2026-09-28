import 'dart:math' as math;

import 'body_frame.dart';
import 'landmark_type.dart';
import 'reliability.dart';
import 'rep_counter.dart';

class HoldEvent {
  const HoldEvent({
    required this.duration,
    required this.steadiness,
    this.meanAngle,
  });

  final Duration duration;

  final double? meanAngle;

  final double steadiness;
}

class HoldState {
  const HoldState({
    required this.holding,
    this.elapsed = Duration.zero,
    this.completed,
    this.reason = '',
  });

  final bool holding;

  final Duration elapsed;

  final HoldEvent? completed;

  final String reason;
}

class HoldDetector {
  HoldDetector({
    this.maxDrift = 0.12,
    this.minDuration = const Duration(seconds: 3),
    this.grace = const Duration(milliseconds: 500),
  });

  final double maxDrift;

  final Duration minDuration;

  final Duration grace;

  final _velocity = LandmarkVelocity();
  final List<HoldEvent> holds = [];

  Duration? _started;
  Map<LandmarkType, ({double x, double y})>? _anchor;
  Duration? _anchorTime;
  Duration? _lastStill;
  final List<double> _angles = [];
  final List<double> _drifts = [];

  HoldState update(BodyFrame frame, double? torso, Duration timestamp,
      {double? angle}) {
    _velocity.update(frame, torso, timestamp);

    if (torso == null || torso <= 0) {
      return HoldState(
        holding: _started != null,
        elapsed: _elapsed(timestamp),
        reason: 'waiting for tracking',
      );
    }

    final current = <LandmarkType, ({double x, double y})>{
      for (final l in frame.landmarks)
        if (l.likelihood >= 0.5) l.type: (x: l.x, y: l.y),
    };
    if (current.isEmpty) {
      return HoldState(
        holding: _started != null,
        elapsed: _elapsed(timestamp),
        reason: 'cannot see you clearly',
      );
    }

    _anchor ??= current;
    _anchorTime ??= timestamp;

    final drifts = <double>[];
    for (final entry in current.entries) {
      final anchored = _anchor![entry.key];
      if (anchored == null) continue;
      final dx = entry.value.x - anchored.x;
      final dy = entry.value.y - anchored.y;
      drifts.add(math.sqrt(dx * dx + dy * dy) / torso);
    }

    if (drifts.isEmpty) {
      return HoldState(
        holding: _started != null,
        elapsed: _elapsed(timestamp),
        reason: 'cannot see you clearly',
      );
    }

    final drift = _median(drifts);
    final still = drift <= maxDrift;

    if (still) {
      if (_started == null) {
        _started = _anchorTime;
        _angles.clear();
        _drifts.clear();
      }
      _lastStill = timestamp;
      if (angle != null) _angles.add(angle);
      _drifts.add(drift);
      return HoldState(holding: true, elapsed: _elapsed(timestamp));
    }

    if (_started == null) {
      _anchor = current;
      _anchorTime = timestamp;
      return const HoldState(holding: false, reason: 'moving');
    }

    final lastStill = _lastStill;
    if (lastStill != null && timestamp - lastStill <= grace) {
      return HoldState(holding: true, elapsed: _elapsed(timestamp));
    }

    final completed = _finish(lastStill ?? timestamp);
    _anchor = current;
    _anchorTime = timestamp;
    return HoldState(holding: false, completed: completed, reason: 'moving');
  }

  HoldEvent? finish(Duration timestamp) => _finish(timestamp);

  HoldEvent? _finish(Duration endTime) {
    final started = _started;
    _started = null;
    _anchor = null;
    _anchorTime = null;
    _lastStill = null;

    final angles = List<double>.from(_angles);
    final drifts = List<double>.from(_drifts);
    _angles.clear();
    _drifts.clear();

    if (started == null) return null;

    final duration = endTime - started;
    if (duration < minDuration) return null;

    final meanDrift = drifts.isEmpty
        ? 0.0
        : drifts.reduce((a, b) => a + b) / drifts.length;
    final steadiness = (1.0 - meanDrift / maxDrift).clamp(0.0, 1.0);

    final event = HoldEvent(
      duration: duration,
      meanAngle: angles.isEmpty
          ? null
          : angles.reduce((a, b) => a + b) / angles.length,
      steadiness: steadiness,
    );
    holds.add(event);
    return event;
  }

  Duration _elapsed(Duration timestamp) =>
      _started == null ? Duration.zero : timestamp - _started!;

  Duration get totalHeld => holds.fold(
        Duration.zero,
        (total, hold) => total + hold.duration,
      );

  void reset({bool keepHolds = true}) {
    _velocity.reset();
    _started = null;
    _anchor = null;
    _anchorTime = null;
    _lastStill = null;
    _angles.clear();
    _drifts.clear();
    if (!keepHolds) holds.clear();
  }

  static double _median(List<double> values) {
    final sorted = List<double>.from(values)..sort();
    final middle = sorted.length ~/ 2;
    if (sorted.length.isOdd) return sorted[middle];
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
}

enum ExerciseKind { reps, hold }

class Exercise {
  const Exercise({
    required this.name,
    required this.joint,
    required this.kind,
    required this.description,
    this.inverted = false,
    this.downBelow = 100,
    this.upAbove = 160,
  });

  final String name;

  final String joint;

  final ExerciseKind kind;

  final String description;

  final bool inverted;

  final double downBelow;
  final double upAbove;

  HysteresisConfig config() => HysteresisConfig(
        name: name,
        downBelow: downBelow,
        upAbove: upAbove,
      );
}

const exercises = <String, Exercise>{
  'squat': Exercise(
    name: 'squat',
    joint: 'Knee',
    kind: ExerciseKind.reps,
    downBelow: 100,
    upAbove: 160,
    description: 'stand side-on or facing the camera',
  ),
  'pushup': Exercise(
    name: 'pushup',
    joint: 'Elbow',
    kind: ExerciseKind.reps,
    downBelow: 100,
    upAbove: 155,
    description: 'place the phone at floor level, side-on',
  ),
  'armraise': Exercise(
    name: 'armraise',
    joint: 'Shoulder',
    kind: ExerciseKind.reps,
    inverted: true,
    downBelow: 60,
    upAbove: 140,
    description: 'face the camera',
  ),
  'situp': Exercise(
    name: 'situp',
    joint: 'Hip',
    kind: ExerciseKind.reps,
    downBelow: 80,
    upAbove: 140,
    description: 'lie side-on to the camera',
  ),
  'plank': Exercise(
    name: 'plank',
    joint: 'Hip',
    kind: ExerciseKind.hold,
    description: 'side-on, whole body in frame',
  ),
  'stretch': Exercise(
    name: 'stretch',
    joint: 'Knee',
    kind: ExerciseKind.hold,
    description: 'hold the position still',
  ),
  'balance': Exercise(
    name: 'balance',
    joint: 'Knee',
    kind: ExerciseKind.hold,
    description: 'stand on one leg, facing the camera',
  ),
};
