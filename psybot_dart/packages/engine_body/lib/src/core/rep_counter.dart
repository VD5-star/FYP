import 'dart:math' as math;

enum RepState {
  up,

  down,

  between,
}

class HysteresisConfig {
  HysteresisConfig({
    required this.name,
    required this.downBelow,
    required this.upAbove,
    this.confirmFrames = 3,
    this.minDwell = const Duration(milliseconds: 300),
  }) {
    if (upAbove <= downBelow) {
      throw ArgumentError(
        '$name: upAbove must exceed downBelow, or there is no dead zone and '
        'the state will chatter',
      );
    }
    if (deadZone < 20.0) {
      throw ArgumentError(
        '$name: dead zone of ${deadZone.toStringAsFixed(0)}° is inside the '
        'measured noise band (published limits of agreement are -6.7° to '
        '+11.9° on knee angle). Widen it to at least 20°.',
      );
    }
  }

  final String name;

  final double downBelow;

  final double upAbove;

  final int confirmFrames;

  final Duration minDwell;

  double get deadZone => upAbove - downBelow;
}

final squatConfig = HysteresisConfig(
  name: 'squat',
  downBelow: 100,
  upAbove: 160,
);

final pushupConfig = HysteresisConfig(
  name: 'pushup',
  downBelow: 100,
  upAbove: 155,
);

final armRaiseConfig = HysteresisConfig(
  name: 'armRaise',
  downBelow: 60,
  upAbove: 140,
);

class RepResult {
  const RepResult({
    required this.state,
    required this.count,
    required this.partialCount,
    this.completed = false,
    this.partial = false,
  });

  final RepState state;

  final bool completed;

  final bool partial;

  final int count;
  final int partialCount;
}

class RepCounter {
  RepCounter(this.config, {this.inverted = false});

  final HysteresisConfig config;

  final bool inverted;

  int count = 0;
  int partialCount = 0;

  RepState _state = RepState.between;
  RepState? _pending;
  int _pendingFrames = 0;
  Duration? _stateEnteredAt;
  Duration? _downEnteredAt;
  bool _reachedDown = false;
  bool _leftUp = false;
  Duration? _lastTime;
  int _blockedAttempts = 0;
  RepState? _blockedTarget;

  RepState get state => _state;

  static const gapReset = Duration(milliseconds: 100);

  static const blockedBeforePartial = 3;

  RepState _classify(double angle) {
    if (inverted) {
      if (angle > config.upAbove) return RepState.down;
      if (angle < config.downBelow) return RepState.up;
      return RepState.between;
    }
    if (angle < config.downBelow) return RepState.down;
    if (angle > config.upAbove) return RepState.up;
    return RepState.between;
  }

  RepResult update(double? angle, Duration timestamp) {
    final last = _lastTime;
    if (last != null && timestamp - last > gapReset) {
      _pending = null;
      _pendingFrames = 0;
    }
    _lastTime = timestamp;

    if (angle == null) return _unchanged();

    final observed = _classify(angle);

    if (observed == _state) {
      _pending = null;
      _pendingFrames = 0;
      if (_blockedTarget != null && _noteAbandonedAttempt()) {
        partialCount++;
        return RepResult(
          state: _state,
          partial: true,
          count: count,
          partialCount: partialCount,
        );
      }
      return _unchanged();
    }

    if (observed == _pending) {
      _pendingFrames++;
    } else {
      _pending = observed;
      _pendingFrames = 1;
    }

    if (_pendingFrames < config.confirmFrames) return _unchanged();

    final entered = _stateEnteredAt;
    if (entered != null &&
        _state != RepState.between &&
        timestamp - entered < config.minDwell) {
      _blockedTarget = observed;
      return _unchanged();
    }

    _blockedTarget = null;
    _blockedAttempts = 0;
    return _transition(observed, timestamp);
  }

  bool _noteAbandonedAttempt() {
    _blockedTarget = null;
    _blockedAttempts++;
    if (_blockedAttempts >= blockedBeforePartial) {
      _blockedAttempts = 0;
      return true;
    }
    return false;
  }

  RepResult _unchanged() => RepResult(
    state: _state,
    count: count,
    partialCount: partialCount,
  );

  RepResult _transition(RepState newState, Duration timestamp) {
    final previous = _state;
    _state = newState;
    _stateEnteredAt = timestamp;
    _pending = null;
    _pendingFrames = 0;

    var completed = false;
    var partial = false;

    if (newState == RepState.down) {
      _reachedDown = true;
      _downEnteredAt = timestamp;
    } else if (newState == RepState.up) {
      final downAt = _downEnteredAt;
      final held = _reachedDown &&
          downAt != null &&
          timestamp - downAt >= config.minDwell;

      if (held && _leftUp) {
        count++;
        completed = true;
      } else if (_leftUp) {
        partialCount++;
        partial = true;
      }
      _reachedDown = false;
      _downEnteredAt = null;
      _leftUp = false;
    }

    if (previous == RepState.up) _leftUp = true;

    return RepResult(
      state: _state,
      completed: completed,
      partial: partial,
      count: count,
      partialCount: partialCount,
    );
  }

  void reset({bool keepCounts = true}) {
    _state = RepState.between;
    _pending = null;
    _pendingFrames = 0;
    _stateEnteredAt = null;
    _downEnteredAt = null;
    _reachedDown = false;
    _leftUp = false;
    _lastTime = null;
    _blockedAttempts = 0;
    _blockedTarget = null;
    if (!keepCounts) {
      count = 0;
      partialCount = 0;
    }
  }
}

class RangeCalibrator {
  RangeCalibrator(this.name, {this.minSamples = 60});

  final String name;

  final int minSamples;

  final List<double> _samples = [];

  int get sampleCount => _samples.length;

  void add(double? angle) {
    if (angle != null) _samples.add(angle);
  }

  static const minUsableRange = 35.0;

  static const deadZoneFraction = 0.5;

  HysteresisConfig? result() {
    failureReason = null;

    if (_samples.length < minSamples) {
      failureReason =
          'only ${_samples.length} usable frames, need $minSamples — '
          'move into full view and try again';
      return null;
    }

    final sorted = List<double>.from(_samples)..sort();
    final low = _percentile(sorted, 5);
    final high = _percentile(sorted, 95);
    observedMin = low;
    observedMax = high;
    final span = high - low;

    if (span < minUsableRange) {
      failureReason =
          'range of motion was only ${span.toStringAsFixed(0)}°, which is too '
          'small to set a reliable threshold — the movement may not have been '
          'performed, or the camera may not have seen it clearly';
      return null;
    }

    final margin = span * (1 - deadZoneFraction) / 2;
    try {
      return HysteresisConfig(
        name: name,
        downBelow: low + margin,
        upAbove: high - margin,
      );
    } on ArgumentError catch (e) {
      failureReason = e.message.toString();
      return null;
    }
  }

  String? failureReason;

  double? observedMin;
  double? observedMax;

  static double _percentile(List<double> sorted, double p) {
    if (sorted.isEmpty) return 0;
    final rank = (p / 100) * (sorted.length - 1);
    final lower = rank.floor();
    final upper = math.min(lower + 1, sorted.length - 1);
    final weight = rank - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  void reset() {
    _samples.clear();
    failureReason = null;
    observedMin = null;
    observedMax = null;
  }
}
