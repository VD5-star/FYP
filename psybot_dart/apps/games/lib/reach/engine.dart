import 'dart:math' as math;

import 'points.dart';
import 'targets.dart';

const String running = 'running';
const String paused = 'paused';
const String ending = 'ending';
const String finished = 'finished';

const double endingSeconds = 1.1;

const String timed = 'timed';
const String calm = 'calm';

const List<List<String>> limbs = <List<String>>[
  <String>['leftShoulder', 'leftElbow'],
  <String>['leftElbow', 'leftWrist'],
  <String>['rightShoulder', 'rightElbow'],
  <String>['rightElbow', 'rightWrist'],
  <String>['leftHip', 'leftKnee'],
  <String>['leftKnee', 'leftAnkle'],
  <String>['rightHip', 'rightKnee'],
  <String>['rightKnee', 'rightAnkle'],
  <String>['leftShoulder', 'rightShoulder'],
  <String>['leftHip', 'rightHip'],
  <String>['leftShoulder', 'leftHip'],
  <String>['rightShoulder', 'rightHip'],
  <String>['leftWrist', 'leftIndex'],
  <String>['rightWrist', 'rightIndex'],
  <String>['leftAnkle', 'leftFootIndex'],
  <String>['rightAnkle', 'rightFootIndex'],
];

const int limbSteps = 4;

const double touchConfidence = 0.15;
const double anchorConfidence = 0.15;

class Hit {
  const Hit({
    required this.band,
    required this.points,
    required this.at,
    this.bonus = false,
  });

  final String band;
  final int points;
  final double at;
  final bool bonus;
}

class GameState {
  GameState({this.mode = timed});

  final String mode;
  String phase = running;
  int score = 0;
  List<Hit> hits = <Hit>[];
  int misses = 0;
  int streak = 0;
  double bonusTime = 0.0;
  int bonusTaken = 0;
  double? started;
  double? ended;
  double? pausedAt;
  double pausedTotal = 0.0;
  double? lastHit;
  String? lastBand;
  double? endingAt;
}

class Summary {
  const Summary({
    required this.score,
    required this.hits,
    required this.misses,
    required this.bonus,
    required this.seconds,
    required this.bands,
  });

  final int score;
  final int hits;
  final int misses;
  final int bonus;
  final double seconds;
  final Map<String, int> bands;
}

class ReachEngine {
  ReachEngine({
    this.mode = timed,
    this.duration = 60.0,
    double targetLifetime = defaultLifetime,
    double gap = defaultBodyGap,
    int? seed,
  })  : spawner =
            TargetSpawner(lifetime: targetLifetime, bodyGap: gap, seed: seed),
        state = GameState(mode: mode);

  String mode;
  double duration;
  final TargetSpawner spawner;
  GameState state;
  Target? target;
  bool _bonusDue = false;

  List<Target> get targets =>
      target == null ? <Target>[] : <Target>[target!];

  double elapsed(double now) {
    final double? started = state.started;
    if (started == null) return 0.0;
    double end = state.ended ?? now;
    final double? pausedAt = state.pausedAt;
    if (state.phase == paused && pausedAt != null) end = pausedAt;
    final double spent =
        math.max(0.0, end - started - state.pausedTotal);
    if (mode == timed) {
      return math.min(spent, duration + state.bonusTime);
    }
    return spent;
  }

  double totalTime() => duration + state.bonusTime;

  double remaining(double now) {
    if (mode != timed) return double.infinity;
    return math.max(0.0, totalTime() - elapsed(now));
  }

  void pause(double now) {
    if (state.phase != running) return;
    state.phase = paused;
    state.pausedAt = now;
  }

  void resume(double now) {
    if (state.phase != paused) return;
    final double? pausedAt = state.pausedAt;
    if (pausedAt != null) {
      final double shift = now - pausedAt;
      state.pausedTotal += shift;
      target?.born += shift;
    }
    state.pausedAt = null;
    state.phase = running;
  }

  void beginEnding(double now) {
    if (state.phase == ending || state.phase == finished) return;
    if (target != null) state.misses += 1;
    state.phase = ending;
    state.endingAt = now;
    state.ended = now;
  }

  void finish(double now) {
    if (state.phase == finished) return;
    state.ended ??= now;
    state.phase = finished;
    target = null;
  }

  double endingProgress(double now) {
    final double? at = state.endingAt;
    if (at == null) return 1.0;
    return ((now - at) / endingSeconds).clamp(0.0, 1.0);
  }

  List<Hit> update(Points? points, List<double>? visibility,
      Map<String, int> index, double? torso, double now, double width,
      double height) {
    if (state.phase == ending) {
      if (endingProgress(now) >= 1.0) finish(now);
      return <Hit>[];
    }
    if (state.phase == finished || state.phase == paused) return <Hit>[];
    if (points == null || visibility == null || torso == null ||
        torso <= 1e-6) {
      return <Hit>[];
    }

    state.started ??= now;

    if (mode == timed && remaining(now) <= 0.0) {
      beginEnding(now);
      return <Hit>[];
    }

    final List<double>? anchor = _anchor(points, visibility, index);
    if (anchor == null) return <Hit>[];

    final List<List<double>> body = _body(points, visibility);
    final List<Hit> scored = <Hit>[];

    final Target? live = target;
    if (live != null && live.expired(now)) {
      if (!live.isBonus) {
        state.misses += 1;
        state.streak = 0;
      }
      target = null;
    }

    if (target == null) {
      _spawn(anchor[0], anchor[1], torso, now, width, height, body);
    }

    final Target? current = target;
    if (current == null) return scored;

    final List<List<double>> touched =
        touchPoints(points, visibility, index);
    if (current.containsAny(touched[0], touched[1], now)) {
      final Hit hit = Hit(
        band: current.band,
        points: current.points,
        at: now,
        bonus: current.isBonus,
      );
      if (current.isBonus) {
        state.bonusTime += bonusSeconds;
        state.bonusTaken += 1;
      } else {
        state.score += current.points;
        state.streak += 1;
        if (state.streak % bonusEvery == 0 && mode == timed) {
          _bonusDue = true;
        }
      }
      state.hits.add(hit);
      state.lastHit = now;
      state.lastBand = current.band;
      scored.add(hit);
      target = null;
      _spawn(anchor[0], anchor[1], torso, now, width, height, body);
    }

    return scored;
  }

  void _spawn(double ax, double ay, double torso, double now, double width,
      double height, List<List<double>> body) {
    final bool wantBonus = _bonusDue && mode == timed;
    final Target? made = spawner.spawn(ax, ay, torso, now, width, height,
        bodyX: body[0], bodyY: body[1], bonusTarget: wantBonus);
    if (made == null) return;
    if (wantBonus) _bonusDue = false;
    target = made;
  }

  List<List<double>> _body(Points points, List<double> visibility) {
    final List<double> xs = <double>[];
    final List<double> ys = <double>[];
    for (int i = 0; i < points.count; i++) {
      if (visibility[i] < touchConfidence) continue;
      if (!points.finite(i)) continue;
      xs.add(points.xs[i]);
      ys.add(points.ys[i]);
    }
    return <List<double>>[xs, ys];
  }

  List<double>? _anchor(
      Points points, List<double> visibility, Map<String, int> index) {
    const List<String> names = <String>[
      'leftShoulder',
      'rightShoulder',
      'leftHip',
      'rightHip'
    ];
    double sx = 0.0;
    double sy = 0.0;
    int n = 0;
    for (final String name in names) {
      final int? i = index[name];
      if (i == null) continue;
      if (visibility[i] < anchorConfidence) continue;
      if (!points.finite(i)) continue;
      sx += points.xs[i];
      sy += points.ys[i];
      n += 1;
    }
    if (n < 2) return null;
    return <double>[sx / n, sy / n];
  }

  List<List<double>> touchPoints(
      Points points, List<double> visibility, Map<String, int> index) {
    final List<double> xs = <double>[];
    final List<double> ys = <double>[];
    final List<bool> usable = List<bool>.filled(points.count, false);
    for (int i = 0; i < points.count; i++) {
      if (visibility[i] < touchConfidence) continue;
      if (!points.finite(i)) continue;
      usable[i] = true;
      xs.add(points.xs[i]);
      ys.add(points.ys[i]);
    }

    for (final List<String> limb in limbs) {
      final int? ia = index[limb[0]];
      final int? ib = index[limb[1]];
      if (ia == null || ib == null) continue;
      if (!usable[ia] || !usable[ib]) continue;
      final double ax = points.xs[ia];
      final double ay = points.ys[ia];
      final double bx = points.xs[ib];
      final double by = points.ys[ib];
      for (int k = 1; k < limbSteps; k++) {
        final double f = k / limbSteps;
        xs.add(ax + (bx - ax) * f);
        ys.add(ay + (by - ay) * f);
      }
    }

    return <List<double>>[xs, ys];
  }

  Summary summary(double now) {
    final Map<String, int> byBand = <String, int>{};
    int touched = 0;
    for (final Hit hit in state.hits) {
      byBand[hit.band] = (byBand[hit.band] ?? 0) + 1;
      if (!hit.bonus) touched += 1;
    }
    return Summary(
      score: state.score,
      hits: touched,
      misses: state.misses,
      bonus: state.bonusTaken,
      seconds: elapsed(now),
      bands: byBand,
    );
  }

  void reset() {
    state = GameState(mode: mode);
    target = null;
    _bonusDue = false;
    spawner.reset();
  }
}
