import 'dart:math' as math;

class OneEuroFilter {
  OneEuroFilter({
    this.minCutoff = 1.0,
    this.beta = 0.007,
    this.derivativeCutoff = 1.0,
  });

  final double minCutoff;

  final double beta;

  final double derivativeCutoff;

  double? _previousValue;
  double? _previousDerivative;
  double? _previousTimestampSeconds;

  double filter(double value, DateTime timestamp) {
    final now = timestamp.microsecondsSinceEpoch / 1e6;
    final previousValue = _previousValue;
    final previousTime = _previousTimestampSeconds;

    if (previousValue == null || previousTime == null) {
      _previousValue = value;
      _previousDerivative = 0;
      _previousTimestampSeconds = now;
      return value;
    }

    var dt = now - previousTime;
    if (dt <= 0) dt = 1 / 30;

    final rate = 1 / dt;

    final rawDerivative = (value - previousValue) * rate;
    final derivative = _lowPass(
      rawDerivative,
      _previousDerivative ?? 0,
      _alpha(rate, derivativeCutoff),
    );

    final cutoff = minCutoff + beta * derivative.abs();
    final filtered = _lowPass(value, previousValue, _alpha(rate, cutoff));

    _previousValue = filtered;
    _previousDerivative = derivative;
    _previousTimestampSeconds = now;
    return filtered;
  }

  void reset() {
    _previousValue = null;
    _previousDerivative = null;
    _previousTimestampSeconds = null;
  }

  static double _lowPass(double value, double previous, double alpha) =>
      alpha * value + (1 - alpha) * previous;

  static double _alpha(double rate, double cutoff) {
    final tau = 1 / (2 * math.pi * cutoff);
    final dt = 1 / rate;
    return 1 / (1 + tau / dt);
  }
}

class OneEuroPoint {
  OneEuroPoint({double minCutoff = 1.0, double beta = 0.007})
    : _x = OneEuroFilter(minCutoff: minCutoff, beta: beta),
      _y = OneEuroFilter(minCutoff: minCutoff, beta: beta);

  final OneEuroFilter _x;
  final OneEuroFilter _y;

  (double, double) filter(double x, double y, DateTime timestamp) =>
      (_x.filter(x, timestamp), _y.filter(y, timestamp));

  void reset() {
    _x.reset();
    _y.reset();
  }
}
