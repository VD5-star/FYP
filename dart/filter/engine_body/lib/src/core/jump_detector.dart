import 'dart:collection';
import 'dart:math' as math;

import 'body_frame.dart';
import 'body_guards.dart';
import 'reliability.dart';

const gravity = 9.81;

class JumpEvent {
  const JumpEvent({
    required this.takeoff,
    required this.landing,
    required this.peakRiseTorsoUnits,
    required this.fittedGravityRatio,
  });

  final Duration takeoff;
  final Duration landing;

  Duration get flight => landing - takeoff;

  final double peakRiseTorsoUnits;

  final double fittedGravityRatio;
}

class JumpDetector {
  JumpDetector({
    this.riseThreshold = 0.08,
    this.minFlightFrames = 5,
    this.gravityTolerance = 0.20,
    this.history = const Duration(seconds: 3),
  });

  final double riseThreshold;

  final int minFlightFrames;

  final double gravityTolerance;

  final Duration history;

  final _times = Queue<Duration>();
  final _heights = Queue<double>();
  final List<JumpEvent> jumps = [];

  double? _baseline;
  bool _airborne = false;
  Duration? _takeoff;
  double _peak = 0;

  int get count => jumps.length;

  JumpEvent? update(BodyFrame frame, Duration timestamp) {
    final com = centreOfMass(frame);
    final torso = torsoLength(frame);

    if (com == null || torso == null || torso <= 0) {
      _abort();
      return null;
    }

    if (_times.isNotEmpty && timestamp - _times.last > gapResetThreshold) {
      _abort();
    }

    final height = -com.y / torso;
    if (!height.isFinite) {
      _abort();
      return null;
    }

    _times.addLast(timestamp);
    _heights.addLast(height);
    while (_times.isNotEmpty && timestamp - _times.first > history) {
      _times.removeFirst();
      _heights.removeFirst();
    }

    if (_heights.length >= 5) {
      _baseline = _median(_heights);
    }
    final baseline = _baseline;
    if (baseline == null) return null;

    final rise = height - baseline;

    if (!_airborne) {
      if (rise > riseThreshold) {
        _airborne = true;
        _takeoff = timestamp;
        _peak = rise;
      }
      return null;
    }

    _peak = math.max(_peak, rise);
    if (rise > riseThreshold * 0.5) return null;

    final event = _finish(timestamp);
    _airborne = false;
    _takeoff = null;
    _peak = 0;
    return event;
  }

  JumpEvent? _finish(Duration landing) {
    final takeoff = _takeoff;
    if (takeoff == null) return null;

    final times = <double>[];
    final heights = <double>[];
    var index = 0;
    for (final t in _times) {
      if (t >= takeoff && t <= landing) {
        times.add((t - takeoff).inMicroseconds / 1e6);
        heights.add(_heights.elementAt(index));
      }
      index++;
    }

    if (times.length < minFlightFrames) return null;

    final coefficients = _fitQuadratic(times, heights);
    if (coefficients == null) return null;

    final accel = 2 * coefficients.a * 0.5;
    final ratio = accel.abs() / gravity;

    if (coefficients.a >= 0) return null;
    if ((ratio - 1.0).abs() > gravityTolerance) return null;

    final flightSeconds = (landing - takeoff).inMicroseconds / 1e6;
    final expectedRise = gravity * flightSeconds * flightSeconds / 8.0 / 0.5;
    if (expectedRise > 1e-9) {
      final riseRatio = _peak / expectedRise;
      if (riseRatio <= 0.4 || riseRatio >= 2.5) return null;
    }

    final event = JumpEvent(
      takeoff: takeoff,
      landing: landing,
      peakRiseTorsoUnits: _peak,
      fittedGravityRatio: ratio,
    );
    jumps.add(event);
    return event;
  }

  void _abort() {
    _airborne = false;
    _takeoff = null;
    _peak = 0;
  }

  void reset({bool keepJumps = true}) {
    _times.clear();
    _heights.clear();
    _baseline = null;
    _abort();
    if (!keepJumps) jumps.clear();
  }

  static double _median(Iterable<double> values) {
    final sorted = List<double>.from(values)..sort();
    final middle = sorted.length ~/ 2;
    if (sorted.length.isOdd) return sorted[middle];
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  static ({double a, double b, double c})? _fitQuadratic(
      List<double> t, List<double> y) {
    final n = t.length;
    if (n < 3) return null;

    var s0 = n.toDouble(), s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    var t0 = 0.0, t1 = 0.0, t2 = 0.0;
    for (var i = 0; i < n; i++) {
      final x = t[i];
      final x2 = x * x;
      s1 += x;
      s2 += x2;
      s3 += x2 * x;
      s4 += x2 * x2;
      t0 += y[i];
      t1 += x * y[i];
      t2 += x2 * y[i];
    }

    final det = s4 * (s2 * s0 - s1 * s1) -
        s3 * (s3 * s0 - s1 * s2) +
        s2 * (s3 * s1 - s2 * s2);
    if (det.abs() < 1e-12 || !det.isFinite) return null;

    final detA = t2 * (s2 * s0 - s1 * s1) -
        s3 * (t1 * s0 - t0 * s1) +
        s2 * (t1 * s1 - t0 * s2);
    final detB = s4 * (t1 * s0 - t0 * s1) -
        t2 * (s3 * s0 - s1 * s2) +
        s2 * (s3 * t0 - t1 * s2);
    final detC = s4 * (s2 * t0 - s1 * t1) -
        s3 * (s3 * t0 - s1 * t2) +
        t2 * (s3 * s1 - s2 * s2);

    final a = detA / det;
    final b = detB / det;
    final c = detC / det;
    if (!a.isFinite || !b.isFinite || !c.isFinite) return null;
    return (a: a, b: b, c: c);
  }
}
