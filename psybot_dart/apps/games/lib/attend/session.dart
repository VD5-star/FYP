import 'dart:math' as math;

import 'sounds.dart';

const String stageFocus = 'focus';
const String stageSwitch = 'switch';
const String stageDivide = 'divide';
const String stageSettle = 'settle';
const String stageOver = 'over';

const int minMinutes = 3;
const int maxMinutes = 10;
const int defaultMinutes = 5;

const double settleFrac = 0.06;
const double settleMin = 10.0;
const double settleMax = 26.0;
const double divideFrac = 0.28;
const double focusFrac = 0.34;
const double switchHold = 6.0;
const int minSwitches = 6;
const double crossfade = 1.1;
const double maxStep = 0.25;

class ShapeBand {
  const ShapeBand(this.top, this.name, this.detail);

  final int top;
  final String name;
  final String detail;
}

const List<ShapeBand> shapes = <ShapeBand>[
  ShapeBand(4, 'light', 'a short pass, one look at each sound'),
  ShapeBand(6, 'steady', 'long enough to settle into it'),
  ShapeBand(8, 'full', 'every sound gets real time'),
  ShapeBand(10, 'deep', 'a long hold at the end'),
];

int clampMinutes(int m) => math.min(maxMinutes, math.max(minMinutes, m));

ShapeBand shapeFor(int minutes) {
  final int m = clampMinutes(minutes);
  for (final ShapeBand s in shapes) {
    if (m <= s.top) return s;
  }
  return shapes.last;
}

String clockText(double seconds) {
  final int s = seconds.round();
  final int mm = s ~/ 60;
  final int ss = s % 60;
  return '$mm:${ss.toString().padLeft(2, '0')}';
}

class Step {
  Step(this.stage, this.target, this.hold);

  final String stage;
  final String? target;
  double hold;
}

class _Rand {
  _Rand(int? seed) : _r = math.Random(seed ?? 20240917);

  final math.Random _r;

  int nextInt(int n) => _r.nextInt(n);

  void shuffle(List<String> a) {
    for (int i = a.length - 1; i > 0; i--) {
      final int j = _r.nextInt(i + 1);
      final String t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
}

class Session {
  Session({int minutes = defaultMinutes, this.seed})
      : minutes = clampMinutes(minutes) {
    build();
  }

  int minutes;
  final int? seed;
  List<Step> steps = <Step>[];
  int index = 0;
  double elapsed = 0.0;
  bool paused = false;
  bool finished = false;

  void build() {
    minutes = clampMinutes(minutes);
    final double total = minutes * 60.0;
    final _Rand rng = _Rand(seed);
    final List<String> names = List<String>.from(soundNames);

    final double settle =
        math.min(settleMax, math.max(settleMin, total * settleFrac));
    final double divide = total * divideFrac;
    final double focusAll = total * focusFrac;
    final double focusHold = focusAll / names.length;

    final double moving = total - settle - divide - focusAll;
    final int switches =
        math.max(minSwitches, (moving / switchHold).round());
    final double swHold = moving / switches;

    final List<Step> out = <Step>[];
    final List<String> order = List<String>.from(names);
    rng.shuffle(order);
    for (final String n in order) {
      out.add(Step(stageFocus, n, focusHold));
    }

    String? last = out.isEmpty ? null : out.last.target;
    for (int i = 0; i < switches; i++) {
      final List<String> choices =
          names.where((String n) => n != last).toList();
      final String n = choices[rng.nextInt(choices.length)];
      out.add(Step(stageSwitch, n, swHold));
      last = n;
    }

    out.add(Step(stageDivide, null, divide));
    out.add(Step(stageSettle, null, settle));

    double sum = 0;
    for (final Step s in out) {
      sum += s.hold;
    }
    final double drift = total - sum;
    out.last.hold = math.max(1.0, out.last.hold + drift);

    steps = out;
    index = 0;
    elapsed = 0.0;
    finished = false;
    paused = false;
  }

  double get total {
    double sum = 0;
    for (final Step s in steps) {
      sum += s.hold;
    }
    return sum;
  }

  Step? get step {
    if (finished || index >= steps.length) return null;
    return steps[index];
  }

  String get stage => step?.stage ?? stageOver;

  String? get target => step?.target;

  double get doneBefore {
    double sum = 0;
    for (int i = 0; i < index && i < steps.length; i++) {
      sum += steps[i].hold;
    }
    return sum;
  }

  double get position => doneBefore + elapsed;

  double get progress {
    final double t = total;
    return t <= 0 ? 0.0 : math.min(1.0, position / t);
  }

  double get remaining => math.max(0.0, total - position);

  double get stepProgress {
    final Step? s = step;
    if (s == null || s.hold <= 0) return 1.0;
    return math.min(1.0, elapsed / s.hold);
  }

  Map<String, double> gains() {
    final Step? s = step;
    return <String, double>{
      for (final String n in soundNames) n: s == null ? 0.0 : 1.0
    };
  }

  Map<String, double> sceneWeights() {
    final Map<String, double> out = <String, double>{
      for (final String n in soundNames) n: 0.0
    };
    final Step? s = step;
    if (s == null) return out;
    if (s.stage == stageDivide || s.stage == stageSettle) {
      final double w = 1.0 / out.length;
      for (final String k in out.keys.toList()) {
        out[k] = w;
      }
      return out;
    }
    String? prev;
    for (int k = index - 1; k >= 0; k--) {
      if (steps[k].target != null) {
        prev = steps[k].target;
        break;
      }
    }
    final double t =
        crossfade > 0 ? math.min(1.0, elapsed / crossfade) : 1.0;
    final String? cur = s.target;
    if (prev != null && prev != cur && t < 1.0) {
      out[prev] = 1.0 - t;
      if (cur != null) out[cur] = t;
    } else if (cur != null) {
      out[cur] = 1.0;
    }
    return out;
  }

  bool advance(double dt, {bool clamp = true}) {
    if (finished || paused || dt <= 0) return false;
    bool moved = false;
    double left = clamp ? math.min(dt, maxStep) : dt;
    while (left > 0 && !finished) {
      final Step? s = step;
      if (s == null) {
        finished = true;
        break;
      }
      final double room = s.hold - elapsed;
      if (left < room) {
        elapsed += left;
        left = 0.0;
      } else {
        left -= room;
        index += 1;
        elapsed = 0.0;
        moved = true;
        if (index >= steps.length) finished = true;
      }
    }
    return moved;
  }

  bool togglePause() {
    if (!finished) paused = !paused;
    return paused;
  }

  void restart() => build();
}
