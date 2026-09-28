import 'dart:collection';

import 'config.dart';
import 'maths.dart';

String compoundKey(String a, String b) {
  final pair = <String>[a, b]..sort();
  return '${pair[0]}|${pair[1]}';
}

final Map<String, List<String>> compoundNames = <String, List<String>>{
  compoundKey('happy', 'surprise'): const <String>['delighted', 'مبتهج'],
  compoundKey('happy', 'fear'): const <String>['nervous joy', 'فرح متوتر'],
  compoundKey('happy', 'sad'): const <String>['bittersweet', 'فرح ممزوج بحزن'],
  compoundKey('sad', 'anger'): const <String>['frustrated', 'محبط'],
  compoundKey('sad', 'fear'): const <String>['anxious', 'قلق'],
  compoundKey('anger', 'disgust'): const <String>['contempt', 'ازدراء'],
  compoundKey('fear', 'surprise'): const <String>['alarmed', 'مفزوع'],
  compoundKey('anger', 'fear'): const <String>['threatened', 'متوتر ومتوجس'],
  compoundKey('neutral', 'sad'): const <String>['subdued', 'فاتر'],
  compoundKey('neutral', 'happy'): const <String>['content', 'مرتاح'],
};

class AffectReading {
  AffectReading({
    this.duchenne = 0.0,
    this.smileType = 'none',
    this.compound,
    this.compoundAr,
    List<MapEntry<String, double>>? mixture,
    this.engagement = 0.0,
    this.fatigue = 0.0,
    this.tension = 0.0,
    this.volatility = 0.0,
    this.blinkRate = 0.0,
    this.expressiveness = 0.0,
    this.transition,
  }) : mixture = mixture ?? <MapEntry<String, double>>[];

  double duchenne;
  String smileType;
  String? compound;
  String? compoundAr;
  List<MapEntry<String, double>> mixture;
  double engagement;
  double fatigue;
  double tension;
  double volatility;
  double blinkRate;
  double expressiveness;
  String? transition;

  Map<String, Object?> toMap() => <String, Object?>{
    'duchenne': roundTo(duchenne, 4),
    'smile_type': smileType,
    'compound': compound,
    'compound_ar': compoundAr,
    'mixture': <List<Object?>>[
      for (final entry in mixture)
        <Object?>[entry.key, roundTo(entry.value, 4)],
    ],
    'engagement': roundTo(engagement, 4),
    'fatigue': roundTo(fatigue, 4),
    'tension': roundTo(tension, 4),
    'volatility': roundTo(volatility, 4),
    'blink_rate': roundTo(blinkRate, 2),
    'expressiveness': roundTo(expressiveness, 4),
    'transition': transition,
  };
}

class AffectSignals {
  const AffectSignals({
    this.label = 'neutral',
    this.valence = 0.0,
    this.arousal = 0.0,
    this.probs = const <String, double>{},
  });

  final String label;
  final double valence;
  final double arousal;
  final Map<String, double> probs;
}

class AffectPose {
  const AffectPose({
    this.eyeOpenness = 0.0,
    this.pitch = 0.0,
    this.attention = 0.0,
    this.isBlinking = false,
  });

  final double eyeOpenness;
  final double pitch;
  final double attention;
  final bool isBlinking;
}

class AffectAnalyzer {
  AffectAnalyzer({this.window = 150});

  final int window;

  final ListQueue<double> _valence = ListQueue<double>();
  final ListQueue<double> _arousal = ListQueue<double>();
  final ListQueue<String> _labels = ListQueue<String>();
  final ListQueue<double> _blinks = ListQueue<double>();
  final ListQueue<double> _eyeOpen = ListQueue<double>();
  final ListQueue<double> _pitch = ListQueue<double>();
  final ListQueue<double> _attention = ListQueue<double>();

  String? _lastLabel;
  bool _wasBlinking = false;

  int get blinkCount => _blinks.length;

  void _push<T>(ListQueue<T> buffer, T value, int limit) {
    buffer.addLast(value);
    while (buffer.length > limit) {
      buffer.removeFirst();
    }
  }

  AffectReading update(
    AffectSignals emotion,
    AffectPose? pose,
    Map<String, double> actionUnits, {
    double? now,
  }) {
    final reading = AffectReading();
    final stamp = now ?? DateTime.now().millisecondsSinceEpoch / 1000.0;

    final probs = emotion.probs;
    final label = emotion.label;

    _push(_valence, emotion.valence, window);
    _push(_arousal, emotion.arousal, window);
    _push(_labels, label, window);

    if (pose != null) {
      _push(_eyeOpen, pose.eyeOpenness, window);
      _push(_pitch, pose.pitch, window);
      _push(_attention, pose.attention, window);
      final blinking = pose.isBlinking;
      if (blinking && !_wasBlinking) {
        _push(_blinks, stamp, 60);
      }
      _wasBlinking = blinking;
    }

    final smile = _duchenne(probs, actionUnits);
    reading.duchenne = smile.$1;
    reading.smileType = smile.$2;

    final compound = _compound(probs);
    reading.mixture = compound.$1;
    reading.compound = compound.$2;
    reading.compoundAr = compound.$3;

    reading.expressiveness = _expressiveness(probs);
    reading.engagement = _engagement(reading.expressiveness);
    reading.blinkRate = _blinkRate(stamp);
    reading.fatigue = _fatigue(reading.blinkRate);
    reading.tension = _tension(actionUnits, probs);
    reading.volatility = _volatility();
    reading.transition = _transition(label);

    return reading;
  }

  static (double, String) _duchenne(
    Map<String, double> probs,
    Map<String, double> units,
  ) {
    final happy = probs['happy'] ?? 0.0;
    final smile = units['smile'] ?? 0.0;
    if (happy < 0.30 || smile <= 0.005) return (0.0, 'none');

    final eyeOpen = units['eye_open'] ?? 0.30;
    final narrowing = clampd((0.30 - eyeOpen) / 0.12, 0.0, 1.0);
    final mouth = clampd(smile / 0.045, 0.0, 1.0);

    final score = clampd(0.45 * mouth + 0.55 * narrowing, 0.0, 1.0);
    if (score >= 0.55) return (score, 'genuine');
    return (score, 'social');
  }

  static (List<MapEntry<String, double>>, String?, String?) _compound(
    Map<String, double> probs,
  ) {
    if (probs.isEmpty) return (<MapEntry<String, double>>[], null, null);

    final ranked = rankedDescending(probs);
    final top = ranked.first;
    final MapEntry<String, double>? second =
        ranked.length > 1 ? ranked[1] : null;

    final mixture = <MapEntry<String, double>>[
      for (final entry in lastNFirst(ranked, 3))
        if (entry.value >= 0.12) entry,
    ];

    if (second == null || second.value < 0.22 || top.value > 0.80) {
      return (mixture, null, null);
    }

    final named = compoundNames[compoundKey(top.key, second.key)];
    if (named == null) return (mixture, null, null);
    return (mixture, named[0], named[1]);
  }

  static List<T> lastNFirst<T>(List<T> values, int count) =>
      values.length <= count ? values : values.sublist(0, count);

  static double _expressiveness(Map<String, double> probs) {
    if (probs.isEmpty) return 0.0;
    return clampd(1.0 - (probs['neutral'] ?? 0.0), 0.0, 1.0);
  }

  double _engagement(double expressiveness) {
    if (_attention.isEmpty) return 0.0;
    final attention = mean(lastN(_attention.toList(), 30));
    return clampd(0.65 * attention + 0.35 * expressiveness, 0.0, 1.0);
  }

  double _blinkRate(double now) {
    final cutoff = now - 60.0;
    while (_blinks.isNotEmpty && _blinks.first < cutoff) {
      _blinks.removeFirst();
    }
    if (_blinks.isEmpty) return 0.0;
    final span = (now - _blinks.first) < 1.0 ? 1.0 : now - _blinks.first;
    return _blinks.length * 60.0 / span;
  }

  double _fatigue(double blinkRate) {
    if (_eyeOpen.length < 10) return 0.0;
    final eye = mean(lastN(_eyeOpen.toList(), 60));
    var droop = 0.0;
    if (_pitch.isNotEmpty) {
      final pitch = mean(lastN(_pitch.toList(), 60));
      droop = clampd((-pitch - 8.0) / 22.0, 0.0, 1.0);
    }

    final closed = clampd((0.42 - eye) / 0.22, 0.0, 1.0);
    final excessBlink = clampd((blinkRate - 22.0) / 26.0, 0.0, 1.0);
    return clampd(
      0.45 * closed + 0.30 * excessBlink + 0.25 * droop,
      0.0,
      1.0,
    );
  }

  static double _tension(
    Map<String, double> units,
    Map<String, double> probs,
  ) {
    if (units.isEmpty) return 0.0;
    final knit = clampd((0.95 - (units['brow_knit'] ?? 0.95)) / 0.35, 0.0, 1.0);
    final lips = clampd((0.28 - (units['mouth_open'] ?? 0.28)) / 0.20, 0.0, 1.0);
    var negative = 0.0;
    for (final key in const <String>['anger', 'fear', 'disgust', 'sad']) {
      negative += probs[key] ?? 0.0;
    }
    return clampd(0.40 * knit + 0.25 * lips + 0.35 * negative, 0.0, 1.0);
  }

  double _volatility() {
    if (_valence.length < 8) return 0.0;
    final recentV = lastN(_valence.toList(), 45);
    final recentA = lastN(_arousal.toList(), 45);
    final movement = mean(_absDiff(recentV)) + mean(_absDiff(recentA));
    return clampd(movement / 0.16, 0.0, 1.0);
  }

  static List<double> _absDiff(List<double> values) => <double>[
    for (var i = 1; i < values.length; i++) (values[i] - values[i - 1]).abs(),
  ];

  String? _transition(String label) {
    if (_lastLabel == null) {
      _lastLabel = label;
      return null;
    }
    if (label == _lastLabel) return null;
    final previous = _lastLabel;
    _lastLabel = label;
    return '$previous -> $label';
  }

  Map<String, Object?> summary() {
    if (_labels.isEmpty) return <String, Object?>{'samples': 0};
    final labels = _labels.toList();
    final dominant = mostCommon(labels)!;
    return <String, Object?>{
      'samples': labels.length,
      'dominant': dominant,
      'dominant_ar': emotionsAr[dominant] ?? dominant,
      'stability': countOf(labels, dominant) / labels.length,
      'mean_valence': mean(_valence),
      'mean_arousal': mean(_arousal),
      'valence_range': _valence.length > 1 ? peakToPeak(_valence) : 0.0,
      'distinct_emotions': labels.toSet().length,
    };
  }

  void reset() {
    _valence.clear();
    _arousal.clear();
    _labels.clear();
    _eyeOpen.clear();
    _pitch.clear();
    _attention.clear();
    _blinks.clear();
    _lastLabel = null;
    _wasBlinking = false;
  }
}
