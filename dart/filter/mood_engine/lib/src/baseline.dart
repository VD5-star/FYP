import 'dart:collection';

import 'config.dart';
import 'maths.dart';

const Map<String, List<String>> moodStates = <String, List<String>>{
  'positive': <String>['Positive', 'إيجابي'],
  'content': <String>['Content', 'مرتاح'],
  'neutral': <String>['Neutral', 'محايد'],
  'withdrawn': <String>['Withdrawn', 'منسحب'],
  'tense': <String>['Tense', 'متوتر'],
  'negative': <String>['Negative', 'سلبي'],
  'volatile': <String>['Volatile', 'متقلّب'],
};

class MoodSummary {
  MoodSummary({
    this.state = 'neutral',
    this.stateEn = 'Neutral',
    this.stateAr = 'محايد',
    this.confidence = 0.0,
    this.valence = 0.0,
    this.energy = 0.0,
    this.stability = 1.0,
    this.samples = 0,
    this.calibrated = false,
    this.baselineProgress = 0.0,
  });

  String state;
  String stateEn;
  String stateAr;
  double confidence;
  double valence;
  double energy;
  double stability;
  int samples;
  bool calibrated;
  double baselineProgress;

  Map<String, Object?> toMap() => <String, Object?>{
    'state': state,
    'state_en': stateEn,
    'state_ar': stateAr,
    'confidence': roundTo(confidence, 4),
    'valence': roundTo(valence, 4),
    'energy': roundTo(energy, 4),
    'stability': roundTo(stability, 4),
    'samples': samples,
    'calibrated': calibrated,
    'baseline_progress': roundTo(baselineProgress, 3),
  };
}

class PersonBaseline {
  PersonBaseline({
    List<double>? probs,
    Map<String, double>? actionUnits,
    this.samples = 0,
    double? updatedAt,
  }) : probs = probs ?? List<double>.filled(emotions.length, 0.0),
       actionUnits = actionUnits ?? <String, double>{},
       updatedAt = updatedAt ?? _nowSeconds();

  static const int required = 500;

  static const double maxAgeS = 7 * 24 * 3600.0;

  static const int emaCap = 400;

  List<double> probs;
  Map<String, double> actionUnits;
  int samples;
  double updatedAt;

  static double _nowSeconds() =>
      DateTime.now().millisecondsSinceEpoch / 1000.0;

  bool get ready => samples >= required;

  double get progress {
    final value = samples / required;
    return value < 1.0 ? value : 1.0;
  }

  double get ageS {
    final value = _nowSeconds() - updatedAt;
    return value > 0.0 ? value : 0.0;
  }

  bool get expired => ready && ageS > maxAgeS;

  void observe(List<double> incoming, Map<String, double> units) {
    final capped = samples + 1 < emaCap ? samples + 1 : emaCap;
    final alpha = 1.0 / capped;
    probs = <double>[
      for (var i = 0; i < probs.length; i++)
        (1 - alpha) * probs[i] + alpha * incoming[i],
    ];
    for (final entry in units.entries) {
      final previous = actionUnits[entry.key] ?? entry.value;
      actionUnits[entry.key] = (1 - alpha) * previous + alpha * entry.value;
    }
    samples += 1;
    updatedAt = _nowSeconds();
  }

  void reset() {
    probs = List<double>.filled(emotions.length, 0.0);
    actionUnits = <String, double>{};
    samples = 0;
    updatedAt = _nowSeconds();
  }

  Map<String, Object?> toMap() => <String, Object?>{
    'probs': <String, double>{
      for (var i = 0; i < emotions.length; i++)
        emotions[i]: roundTo(probs[i], 5),
    },
    'action_units': <String, double>{
      for (final entry in actionUnits.entries)
        entry.key: roundTo(entry.value, 5),
    },
    'samples': samples,
    'updated_at': updatedAt,
  };

  static PersonBaseline fromMap(Map<String, Object?> payload) {
    final rawProbs = payload['probs'];
    final probsMap = rawProbs is Map ? rawProbs : const <Object?, Object?>{};
    final rawUnits = payload['action_units'];
    final unitsMap = rawUnits is Map ? rawUnits : const <Object?, Object?>{};
    return PersonBaseline(
      probs: <double>[
        for (final name in emotions)
          (probsMap[name] as num?)?.toDouble() ?? 0.0,
      ],
      actionUnits: <String, double>{
        for (final entry in unitsMap.entries)
          entry.key.toString(): (entry.value as num).toDouble(),
      },
      samples: (payload['samples'] as num?)?.toInt() ?? 0,
      updatedAt: (payload['updated_at'] as num?)?.toDouble() ?? _nowSeconds(),
    );
  }
}

class BaselineTracker {
  BaselineTracker({this.window = 240});

  static const double excludeAboveConfidence = 0.80;

  final int window;

  final Map<int, PersonBaseline> _baselines = <int, PersonBaseline>{};
  PersonBaseline _anonymous = PersonBaseline();
  int? _current;

  final ListQueue<double> _valence = ListQueue<double>();
  final ListQueue<double> _arousal = ListQueue<double>();
  final ListQueue<String> _states = ListQueue<String>();

  Iterable<int> get knownPersonIds => _baselines.keys;

  PersonBaseline baselineFor(int? personId) {
    if (personId == null) return _anonymous;
    return _baselines.putIfAbsent(personId, PersonBaseline.new);
  }

  void load(int personId, Map<String, Object?> payload) {
    try {
      _baselines[personId] = PersonBaseline.fromMap(payload);
    } on Object {
      return;
    }
  }

  Map<String, Object?>? export(int personId) => _baselines[personId]?.toMap();

  void switchPerson(int? personId) {
    if (personId == _current) return;
    _current = personId;
    _valence.clear();
    _arousal.clear();
    _states.clear();
  }

  void _push<T>(ListQueue<T> buffer, T value) {
    buffer.addLast(value);
    while (buffer.length > window) {
      buffer.removeFirst();
    }
  }

  (List<double>, PersonBaseline) adjust(
    List<double> probs,
    Map<String, double> units,
    int? personId, {
    bool learn = true,
  }) {
    final baseline = baselineFor(personId);

    if (learn &&
        (!baseline.ready || maxOf(probs) < excludeAboveConfidence)) {
      baseline.observe(probs, units);
    }

    if (!baseline.ready) return (probs, baseline);

    final neutralIndex = emotionIndex('neutral');
    const floor = 0.06;

    final neutralMass = probs[neutralIndex];
    var expressiveMass = 0.0;
    for (var i = 0; i < probs.length; i++) {
      if (i != neutralIndex) expressiveMass += probs[i];
    }
    if (expressiveMass <= 0 || !expressiveMass.isFinite) {
      return (probs, baseline);
    }

    final ratio = <int, double>{};
    var ratioTotal = 0.0;
    for (var i = 0; i < probs.length; i++) {
      if (i == neutralIndex) continue;
      final denominator =
          baseline.probs[i] > floor ? baseline.probs[i] : floor;
      final value = probs[i] / denominator;
      ratio[i] = value;
      ratioTotal += value;
    }
    if (ratioTotal <= 0 || !ratioTotal.isFinite) return (probs, baseline);

    final adjusted = List<double>.filled(probs.length, 0.0);
    adjusted[neutralIndex] = neutralMass;
    for (final entry in ratio.entries) {
      adjusted[entry.key] = entry.value / ratioTotal * expressiveMass;
    }

    const strength = 0.5;
    final blended = <double>[
      for (var i = 0; i < probs.length; i++)
        strength * adjusted[i] + (1.0 - strength) * probs[i],
    ];
    final total = sumOf(blended);
    if (total <= 0 || !total.isFinite) return (probs, baseline);
    return (<double>[for (final value in blended) value / total], baseline);
  }

  MoodSummary updateMood(
    double valence,
    double arousal,
    double tension,
    double volatility,
    int? personId,
  ) {
    switchPerson(personId);
    _push(_valence, valence);
    _push(_arousal, arousal);

    final baseline = baselineFor(personId);
    final summary = MoodSummary(
      samples: _valence.length,
      calibrated: baseline.ready,
      baselineProgress: baseline.progress,
    );
    if (_valence.length < 10) {
      summary.stateEn = moodStates['neutral']![0];
      summary.stateAr = moodStates['neutral']![1];
      return summary;
    }

    final meanValence = mean(_valence);
    final meanArousal = mean(_arousal);
    final swing = populationStd(_valence);

    summary.valence = meanValence;
    summary.energy = meanArousal;
    summary.stability = clampd(1.0 - swing / 0.5, 0.0, 1.0);

    final state = _classify(
      meanValence,
      meanArousal,
      swing,
      tension,
      volatility,
    );
    _push(_states, state);

    final states = _states.toList();
    final dominant = mostCommon(states)!;
    summary.state = dominant;
    summary.stateEn = moodStates[dominant]![0];
    summary.stateAr = moodStates[dominant]![1];
    summary.confidence = countOf(states, dominant) / states.length;
    return summary;
  }

  static String _classify(
    double valence,
    double arousal,
    double swing,
    double tension,
    double volatility,
  ) {
    if (swing > 0.34 || volatility > 0.72) return 'volatile';
    if (valence >= 0.30) return arousal >= 0.45 ? 'positive' : 'content';
    if (valence <= -0.28) {
      if (tension >= 0.55 || arousal >= 0.55) return 'tense';
      return 'negative';
    }
    if (valence <= -0.10 && arousal < 0.35) return 'withdrawn';
    if (tension >= 0.62) return 'tense';
    return 'neutral';
  }

  void reset() {
    _valence.clear();
    _arousal.clear();
    _states.clear();
    _current = null;
  }

  void forget([int? personId]) {
    if (personId == null) {
      _baselines.clear();
      _anonymous = PersonBaseline();
    } else {
      _baselines.remove(personId);
    }
  }
}
