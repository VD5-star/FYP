import 'dart:math' as math;

class SignedVelocity {
  static const alpha = 0.17;

  static const gapReset = Duration(milliseconds: 100);

  double? _previous;
  Duration? _previousTime;
  double? _rate;

  double? get rate => _rate;

  double? update(double? value, Duration timestamp) {
    if (value == null || !value.isFinite) {
      _previous = null;
      _previousTime = null;
      return _rate;
    }

    final prev = _previous;
    final prevTime = _previousTime;
    if (prev == null || prevTime == null) {
      _previous = value;
      _previousTime = timestamp;
      return _rate;
    }

    final dt = timestamp - prevTime;
    if (dt <= Duration.zero || dt > gapReset) {
      _previous = value;
      _previousTime = timestamp;
      _rate = null;
      return null;
    }

    final seconds = dt.inMicroseconds / 1e6;
    final raw = (value - prev) / seconds;
    _rate = _rate == null ? raw : _rate! + alpha * (raw - _rate!);
    _previous = value;
    _previousTime = timestamp;
    return _rate;
  }

  void reset() {
    _previous = null;
    _previousTime = null;
    _rate = null;
  }
}

class Turnaround {
  const Turnaround({
    required this.timestamp,
    required this.atMaximum,
    required this.value,
    required this.travelled,
  });

  final Duration timestamp;

  final bool atMaximum;

  final double value;

  final double travelled;
}

class VelocityRepResult {
  const VelocityRepResult({
    required this.count,
    this.completed = false,
    this.turnaround,
    this.depthReached,
    this.velocity,
    this.rejected,
  });

  final int count;
  final bool completed;
  final Turnaround? turnaround;

  final double? depthReached;

  final double? velocity;

  final String? rejected;
}

class VelocityRepCounter {
  VelocityRepCounter({
    this.minTravel = 25.0,
    this.minSpeed = 15.0,
    this.minInterval = const Duration(milliseconds: 250),
  });

  final double minTravel;

  final double minSpeed;

  final Duration minInterval;

  int count = 0;

  final _velocity = SignedVelocity();
  double? _lastReversalValue;
  Duration? _lastReversalTime;
  bool? _lastWasMaximum;
  double _peakSpeed = 0;
  int _previousSign = 0;
  double? _pendingExtreme;
  bool? _restingExtreme;
  double? _firstValue;

  VelocityRepResult update(double? angle, Duration timestamp) {
    final rate = _velocity.update(angle, timestamp);

    if (angle == null || rate == null) {
      return VelocityRepResult(count: count, velocity: rate);
    }

    _firstValue ??= angle;

    _peakSpeed = math.max(_peakSpeed, rate.abs());

    final sign = rate.abs() < 1e-9 ? 0 : (rate > 0 ? 1 : -1);
    if (sign == 0) return VelocityRepResult(count: count, velocity: rate);

    if (_previousSign == 0) {
      _restingExtreme ??= sign < 0;
      _previousSign = sign;
      _pendingExtreme = angle;
      return VelocityRepResult(count: count, velocity: rate);
    }

    if (sign == _previousSign) {
      _pendingExtreme = angle;
      return VelocityRepResult(count: count, velocity: rate);
    }

    final reversedAtMaximum = _previousSign > 0;
    _previousSign = sign;
    final extreme = _pendingExtreme ?? angle;
    _pendingExtreme = angle;

    final peakSpeed = _peakSpeed;
    _peakSpeed = 0;

    if (peakSpeed < minSpeed) {
      return VelocityRepResult(
        count: count,
        velocity: rate,
        rejected: 'movement too slow to be deliberate',
      );
    }

    final lastTime = _lastReversalTime;
    if (lastTime != null && timestamp - lastTime < minInterval) {
      return VelocityRepResult(
        count: count,
        velocity: rate,
        rejected: 'reversals too close together',
      );
    }

    final reference = _lastReversalValue ?? _firstValue;
    if (reference == null) {
      return VelocityRepResult(
        count: count,
        velocity: rate,
        rejected: 'no reference to measure travel from',
      );
    }

    final travelled = (extreme - reference).abs();
    if (travelled < minTravel) {
      return VelocityRepResult(
        count: count,
        velocity: rate,
        rejected: 'not enough movement between turns',
      );
    }

    final turnaround = Turnaround(
      timestamp: timestamp,
      atMaximum: reversedAtMaximum,
      value: extreme,
      travelled: travelled,
    );

    var completed = false;
    double? depth;

    if (_restingExtreme == null) {
      _restingExtreme = reversedAtMaximum;
    } else if (reversedAtMaximum != _restingExtreme) {
      count++;
      completed = true;
      depth = extreme;
    }

    _lastWasMaximum = reversedAtMaximum;
    _lastReversalValue = extreme;
    _lastReversalTime = timestamp;

    return VelocityRepResult(
      count: count,
      completed: completed,
      turnaround: turnaround,
      depthReached: depth,
      velocity: rate,
    );
  }

  bool? get restingExtreme => _restingExtreme;

  bool? get lastWasMaximum => _lastWasMaximum;

  void reset({bool keepCounts = true}) {
    _velocity.reset();
    _lastReversalValue = null;
    _lastReversalTime = null;
    _lastWasMaximum = null;
    _peakSpeed = 0;
    _previousSign = 0;
    _pendingExtreme = null;
    _restingExtreme = null;
    _firstValue = null;
    if (!keepCounts) count = 0;
  }
}
