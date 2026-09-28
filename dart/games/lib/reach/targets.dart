import 'dart:math' as math;

import 'points.dart';

const String near = 'near';
const String far = 'far';
const String veryFar = 'very_far';
const String bonus = 'bonus';

const double armLength = 0.97;
const double defaultBodyGap = 0.95;

const Map<String, List<double>> bands = <String, List<double>>{
  near: <double>[0.95, 1.15, 1],
  far: <double>[1.15, 1.40, 2],
  veryFar: <double>[1.40, 99.0, 3],
};

const double reachSide = 2.30;
const double reachUp = 2.50;
const double reachDown = 1.80;

const double edgeStep = 0.25;

const double baseSize = 0.42;
const double minSizePx = 22.0;

const double defaultLifetime = 1.5;
const double shrinkFloor = 0.35;

const int bonusEvery = 5;
const double bonusSeconds = 3.0;
const double bonusSize = 1.25;

class Target {
  Target({
    required this.x,
    required this.y,
    required this.radius,
    required this.band,
    required this.points,
    required this.born,
    required this.lifetime,
  });

  final double x;
  final double y;
  final double radius;
  final String band;
  final int points;
  double born;
  final double lifetime;

  bool get isBonus => band == bonus;

  double age(double now) => now - born;

  double remaining(double now) => math.max(0.0, lifetime - age(now));

  double life(double now) =>
      (remaining(now) / math.max(lifetime, 1e-6)).clamp(0.0, 1.0);

  double scale(double now) => shrinkFloor + (1.0 - shrinkFloor) * life(now);

  double currentRadius(double now) => radius * scale(now);

  bool contains(double px, double py, double now) {
    if (!px.isFinite || !py.isFinite) return false;
    final double r = currentRadius(now);
    return (px - x).abs() <= r && (py - y).abs() <= r;
  }

  bool containsAny(List<double> xs, List<double> ys, double now) {
    if (xs.isEmpty) return false;
    final double r = currentRadius(now);
    for (int i = 0; i < xs.length; i++) {
      if ((xs[i] - x).abs() <= r && (ys[i] - y).abs() <= r) return true;
    }
    return false;
  }

  bool expired(double now) => age(now) >= lifetime;
}

class EdgeFence {
  const EdgeFence(this.xs, this.ys);
  final List<double> xs;
  final List<double> ys;

  bool get isEmpty => xs.isEmpty;
  int get length => xs.length;
}

EdgeFence edgePoints(List<double> bodyX, List<double> bodyY, double torso,
    double width, double height,
    {double margin = 10.0}) {
  if (bodyX.isEmpty || torso <= 1e-6) {
    return const EdgeFence(<double>[], <double>[]);
  }

  final double step = math.max(8.0, edgeStep * torso);
  double x0 = bodyX.first;
  double x1 = bodyX.first;
  double y0 = bodyY.first;
  double y1 = bodyY.first;
  for (int i = 1; i < bodyX.length; i++) {
    if (bodyX[i] < x0) x0 = bodyX[i];
    if (bodyX[i] > x1) x1 = bodyX[i];
    if (bodyY[i] < y0) y0 = bodyY[i];
    if (bodyY[i] > y1) y1 = bodyY[i];
  }
  final double pad = torso * 0.5;
  final List<double> outX = <double>[];
  final List<double> outY = <double>[];

  if (y0 <= margin) {
    for (final double x in arange(x0 - pad, x1 + pad + step, step)) {
      outX.add(x);
      outY.add(0.0);
    }
  }
  if (y1 >= height - margin) {
    for (final double x in arange(x0 - pad, x1 + pad + step, step)) {
      outX.add(x);
      outY.add(height);
    }
  }
  if (x0 <= margin) {
    for (final double y in arange(y0 - pad, y1 + pad + step, step)) {
      outX.add(0.0);
      outY.add(y);
    }
  }
  if (x1 >= width - margin) {
    for (final double y in arange(y0 - pad, y1 + pad + step, step)) {
      outX.add(width);
      outY.add(y);
    }
  }

  return EdgeFence(outX, outY);
}

class TargetSpawner {
  TargetSpawner({
    this.lifetime = defaultLifetime,
    this.bodyGap = defaultBodyGap,
    int? seed,
  })  : _seed = seed,
        _random = math.Random(seed);

  final double lifetime;
  final double bodyGap;
  final int? _seed;
  math.Random _random;

  int rejected = 0;
  int relaxed = 0;

  void reset() {
    _random = math.Random(_seed);
    rejected = 0;
    relaxed = 0;
  }

  List<Object> _bandFor(double gap) {
    for (final MapEntry<String, List<double>> e in bands.entries) {
      if (gap >= e.value[0] && gap < e.value[1]) {
        return <Object>[e.key, e.value[2].toInt()];
      }
    }
    return <Object>[veryFar, bands[veryFar]![2].toInt()];
  }

  double _limit(double dy) {
    if (dy > 0) return reachSide * (1 - dy) + reachUp * dy;
    return reachSide * (1 + dy) + reachDown * -dy;
  }

  Target? spawn(double anchorX, double anchorY, double torso, double now,
      double width, double height,
      {List<double>? bodyX, List<double>? bodyY, bool bonusTarget = false}) {
    if (torso <= 1e-6 || !anchorX.isFinite || !anchorY.isFinite) return null;

    final double base = math.max(minSizePx, torso * baseSize * 0.5);
    final double radius = bonusTarget ? base * bonusSize : base;
    final double margin = radius * 1.05;

    List<double> avoidX;
    List<double> avoidY;
    if (bodyX == null || bodyX.isEmpty) {
      avoidX = <double>[anchorX];
      avoidY = <double>[anchorY];
    } else {
      avoidX = <double>[];
      avoidY = <double>[];
      for (int i = 0; i < bodyX.length; i++) {
        if (bodyX[i].isFinite && bodyY![i].isFinite) {
          avoidX.add(bodyX[i]);
          avoidY.add(bodyY[i]);
        }
      }
      if (avoidX.isEmpty) {
        avoidX = <double>[anchorX];
        avoidY = <double>[anchorY];
      } else {
        final EdgeFence fence =
            edgePoints(avoidX, avoidY, torso, width, height);
        if (!fence.isEmpty) {
          avoidX = <double>[...avoidX, ...fence.xs];
          avoidY = <double>[...avoidY, ...fence.ys];
        }
      }
    }

    Target? best;
    double bestGap = -1.0;

    for (int attempt = 0; attempt < 220; attempt++) {
      final double fraction = _random.nextDouble();
      final double angle = _random.nextDouble() * 2.0 * math.pi;
      final double dx = math.cos(angle);
      final double dy = math.sin(angle);
      final double span = _limit(dy) * torso;
      final double close = bodyGap * torso;
      if (span <= close) continue;
      final double distance = close + fraction * (span - close);
      final double x = anchorX + distance * dx;
      final double y = anchorY - distance * dy;

      if (!(x >= margin && x <= width - margin)) continue;
      if (!(y >= margin && y <= height - margin)) continue;

      double gap = double.infinity;
      for (int i = 0; i < avoidX.length; i++) {
        final double d = hypot(avoidX[i] - x, avoidY[i] - y);
        if (d < gap) gap = d;
      }
      final double units = gap / torso;

      if (gap >= bodyGap * torso) {
        final List<Object> b = _bandFor(units);
        return Target(
          x: x,
          y: y,
          radius: radius,
          band: bonusTarget ? bonus : b[0] as String,
          points: bonusTarget ? 0 : b[1] as int,
          born: now,
          lifetime: lifetime,
        );
      }

      if (gap > bestGap) {
        bestGap = gap;
        final List<Object> b = _bandFor(units);
        best = Target(
          x: x,
          y: y,
          radius: radius,
          band: bonusTarget ? bonus : b[0] as String,
          points: bonusTarget ? 0 : b[1] as int,
          born: now,
          lifetime: lifetime,
        );
      }
    }

    if (best != null && bestGap >= bodyGap * torso * 0.6) {
      relaxed += 1;
      best.born = now;
      return best;
    }

    rejected += 1;
    return null;
  }
}
