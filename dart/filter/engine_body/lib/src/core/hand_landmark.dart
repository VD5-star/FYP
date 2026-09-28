import 'dart:math' as math;

const int handWrist = 0;

enum Finger { thumb, indexFinger, middle, ring, pinky }

class FingerJoints {
  const FingerJoints(this.tip, this.middle, this.base);

  final int tip;
  final int middle;
  final int base;
}

const fingerJoints = <Finger, FingerJoints>{
  Finger.thumb: FingerJoints(4, 3, 2),
  Finger.indexFinger: FingerJoints(8, 6, 5),
  Finger.middle: FingerJoints(12, 10, 9),
  Finger.ring: FingerJoints(16, 14, 13),
  Finger.pinky: FingerJoints(20, 18, 17),
};

const fingerNumbers = <Finger, int>{
  Finger.pinky: 1,
  Finger.ring: 2,
  Finger.middle: 3,
  Finger.indexFinger: 4,
  Finger.thumb: 5,
};

const handBones = <(int, int)>[
  (0, 1), (1, 2), (2, 3), (3, 4),
  (0, 5), (5, 6), (6, 7), (7, 8),
  (0, 9), (9, 10), (10, 11), (11, 12),
  (0, 13), (13, 14), (14, 15), (15, 16),
  (0, 17), (17, 18), (18, 19), (19, 20),
  (5, 9), (9, 13), (13, 17),
];

class HandPoint {
  const HandPoint(this.x, this.y);

  final double x;
  final double y;

  double distanceTo(HandPoint other) {
    final dx = x - other.x;
    final dy = y - other.y;
    return math.sqrt(dx * dx + dy * dy);
  }
}

bool fingerExtended(List<HandPoint> points, Finger finger) {
  final joints = fingerJoints[finger]!;
  if (points.length <= joints.tip || points.length <= joints.middle) {
    return false;
  }
  final wrist = points[handWrist];
  return points[joints.tip].distanceTo(wrist) >
      points[joints.middle].distanceTo(wrist);
}

Map<Finger, bool> readFingers(List<HandPoint> points) => {
      for (final finger in Finger.values)
        finger: fingerExtended(points, finger),
    };

List<int> fingerNumbersOf(Map<Finger, bool> states) {
  final out = <int>[
    for (final entry in states.entries)
      if (entry.value) fingerNumbers[entry.key]!,
  ]..sort();
  return out;
}

bool isStopGesture(Map<Finger, bool> states) =>
    (states[Finger.pinky] ?? false) &&
    (states[Finger.thumb] ?? false) &&
    !(states[Finger.indexFinger] ?? true) &&
    !(states[Finger.middle] ?? true) &&
    !(states[Finger.ring] ?? true);

class HandReading {
  HandReading({
    required this.points,
    required this.handedness,
    required this.states,
  });

  final List<HandPoint> points;

  final String handedness;

  final Map<Finger, bool> states;

  List<int> get numbers => fingerNumbersOf(states);

  bool get isStop => isStopGesture(states);

  static HandReading? fromPoints(
    List<HandPoint> points, {
    String handedness = 'unknown',
  }) {
    if (points.length < 21) return null;
    return HandReading(
      points: points,
      handedness: handedness,
      states: readFingers(points),
    );
  }

  static HandReading? fromMap(Map<Object?, Object?> map) {
    final raw = map['landmarks'];
    if (raw is! List) return null;

    final points = <HandPoint>[];
    for (final entry in raw) {
      if (entry is! Map) continue;
      final x = entry['x'];
      final y = entry['y'];
      if (x is! num || y is! num) continue;
      points.add(HandPoint(x.toDouble(), y.toDouble()));
    }

    final side = map['handedness'];
    return fromPoints(
      points,
      handedness: side is String ? side : 'unknown',
    );
  }
}

class GestureState {
  const GestureState({
    this.progress = 0,
    this.fired = false,
  });

  final double progress;

  final bool fired;
}

class HeldGesture {
  HeldGesture({
    this.hold = const Duration(milliseconds: 1500),
    this.grace = const Duration(milliseconds: 400),
  });

  final Duration hold;

  final Duration grace;

  Duration? _started;
  Duration? _lastSeen;
  bool _fired = false;

  GestureState update(bool present, Duration timestamp) {
    if (present) {
      _lastSeen = timestamp;
      _started ??= timestamp;
    } else {
      final started = _started;
      final lastSeen = _lastSeen;
      if (started != null && lastSeen != null) {
        if (timestamp - lastSeen > grace) {
          _started = null;
          _fired = false;
          return const GestureState();
        }
      }
    }

    final started = _started;
    if (started == null) return const GestureState();

    final held = timestamp - started;
    final progress =
        (held.inMicroseconds / hold.inMicroseconds).clamp(0.0, 1.0);
    final fired = progress >= 1.0 && !_fired;
    if (fired) _fired = true;
    return GestureState(progress: progress, fired: fired);
  }

  void reset() {
    _started = null;
    _lastSeen = null;
    _fired = false;
  }
}

class HandSampler {
  HandSampler({int everyNFrames = 4})
      : everyNFrames = everyNFrames < 1 ? 1 : everyNFrames;

  final int everyNFrames;

  int _frame = 0;

  bool tick() {
    final due = _frame % everyNFrames == 0;
    _frame++;
    return due;
  }

  void reset() => _frame = 0;
}
