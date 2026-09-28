import 'maths.dart';

const Map<String, List<String>> sessionStates = <String, List<String>>{
  'positive': <String>['Positive', 'إيجابي'],
  'content': <String>['Content', 'مرتاح'],
  'calm': <String>['Calm', 'هادئ'],
  'neutral': <String>['Neutral', 'محايد'],
  'tired': <String>['Tired', 'متعب'],
  'withdrawn': <String>['Withdrawn', 'منسحب'],
  'tense': <String>['Tense', 'متوتر'],
  'restless': <String>['Restless', 'مضطرب'],
  'distressed': <String>['Distressed', 'قلق'],
  'low': <String>['Low', 'حزين'],
};

class SessionReport {
  const SessionReport({
    required this.personId,
    required this.personName,
    required this.startedAt,
    required this.endedAt,
    required this.samples,
    required this.state,
    required this.confidence,
    required this.dominance,
    required this.valence,
    required this.arousal,
    required this.stability,
    required this.engagement,
    required this.fatigue,
    required this.tension,
    this.emotionMix = const <String, double>{},
    this.notes = const <String>[],
  });

  final int? personId;
  final String? personName;
  final double startedAt;
  final double endedAt;
  final int samples;
  final String state;
  final double confidence;
  final double dominance;
  final double valence;
  final double arousal;
  final double stability;
  final double engagement;
  final double fatigue;
  final double tension;
  final Map<String, double> emotionMix;
  final List<String> notes;

  double get durationS {
    final value = endedAt - startedAt;
    return value > 0.0 ? value : 0.0;
  }

  String label({bool arabic = false}) {
    final names = sessionStates[state];
    if (names == null) return state;
    return arabic ? names[1] : names[0];
  }

  Map<String, Object?> toMap() => <String, Object?>{
    'person_id': personId,
    'person_name': personName,
    'started_at': startedAt,
    'ended_at': endedAt,
    'duration_s': roundTo(durationS, 1),
    'samples': samples,
    'state': state,
    'label_en': label(),
    'label_ar': label(arabic: true),
    'confidence': roundTo(confidence, 4),
    'dominance': roundTo(dominance, 4),
    'valence': roundTo(valence, 4),
    'arousal': roundTo(arousal, 4),
    'stability': roundTo(stability, 4),
    'engagement': roundTo(engagement, 4),
    'fatigue': roundTo(fatigue, 4),
    'tension': roundTo(tension, 4),
    'emotion_mix': <String, double>{
      for (final entry in emotionMix.entries) entry.key: roundTo(entry.value, 4),
    },
    'notes': List<String>.from(notes),
  };
}

class SessionAnalyser {
  SessionAnalyser({
    double periodS = defaultPeriodS,
    int minSamples = minSamplesDefault,
    double? now,
  }) : periodS = periodS < 30.0 ? 30.0 : periodS,
       minSamples = minSamples < 5 ? 5 : minSamples,
       _started = now ?? _nowSeconds();

  static const double defaultPeriodS = 300.0;

  static const int minSamplesDefault = 60;

  static const double minSampleConfidence = 0.35;

  final double periodS;
  final int minSamples;

  int? _person;
  String? _name;
  double _started;

  final List<double> _valence = <double>[];
  final List<double> _arousal = <double>[];
  final Map<String, int> _states = <String, int>{};
  final Map<String, int> _emotions = <String, int>{};
  final Map<String, List<double>> _units = <String, List<double>>{};
  final List<double> _confidences = <double>[];
  int _weak = 0;

  static double _nowSeconds() =>
      DateTime.now().millisecondsSinceEpoch / 1000.0;

  SessionReport? observe({
    required int? personId,
    required String? personName,
    required double valence,
    required double arousal,
    required String? moodState,
    required String? emotion,
    required double confidence,
    Map<String, double>? units,
    double? now,
  }) {
    final stamp = now ?? _nowSeconds();
    SessionReport? report;

    if (personId != _person) {
      report = _finalise(stamp);
      _reset(personId, personName, stamp);
    } else if (personName != null && _name == null) {
      _name = personName;
    }

    if (confidence >= minSampleConfidence) {
      _valence.add(valence);
      _arousal.add(arousal);
      if (moodState != null) {
        _states[moodState] = (_states[moodState] ?? 0) + 1;
      }
      if (emotion != null) {
        _emotions[emotion] = (_emotions[emotion] ?? 0) + 1;
      }
      for (final entry in (units ?? const <String, double>{}).entries) {
        _units.putIfAbsent(entry.key, () => <double>[]).add(entry.value);
      }
    } else {
      _weak += 1;
    }
    _confidences.add(confidence);

    if (report == null && stamp - _started >= periodS) {
      report = _finalise(stamp);
      _reset(personId, personName, stamp);
    }
    return report;
  }

  SessionReport? flush({double? now}) {
    final stamp = now ?? _nowSeconds();
    final report = _finalise(stamp);
    _reset(_person, _name, stamp);
    return report;
  }

  void reset() => _reset(null, null, _nowSeconds());

  Map<String, Object?> progress({double? now}) {
    final stamp = now ?? _nowSeconds();
    final samples = _valence.length;
    final elapsedRaw = stamp - _started;
    final elapsed = elapsedRaw > 0.0 ? elapsedRaw : 0.0;
    final byTime = elapsed / periodS;
    final byEvidence = samples / (minSamples > 1 ? minSamples : 1);
    return <String, Object?>{
      'person_id': _person,
      'person_name': _name,
      'elapsed_s': roundTo(elapsed, 1),
      'period_s': periodS,
      'samples': samples,
      'min_samples': minSamples,
      'progress': roundTo(byTime < byEvidence ? byTime : byEvidence, 4),
      'will_report': samples >= minSamples,
    };
  }

  void _reset(int? personId, String? name, double now) {
    _person = personId;
    _name = name;
    _started = now;
    _valence.clear();
    _arousal.clear();
    _states.clear();
    _emotions.clear();
    _units.clear();
    _confidences.clear();
    _weak = 0;
  }

  double _unit(String key) {
    final values = _units[key];
    return values == null || values.isEmpty ? 0.0 : mean(values);
  }

  SessionReport? _finalise(double now) {
    final samples = _valence.length;
    if (samples < minSamples) return null;

    final valence = mean(_valence);
    final arousal = mean(_arousal);
    final swing = populationStd(_valence);
    final stability = clampd(1.0 - swing * 2.2, 0.0, 1.0);
    final engagement = _unit('engagement');
    final fatigue = _unit('fatigue');
    final tension = _unit('tension');

    final verdict = _verdict(valence, arousal, stability, fatigue, tension);
    final state = verdict.$1;
    final dominance = verdict.$2;

    var total = 0.0;
    for (final value in _emotions.values) {
      total += value;
    }
    if (total == 0.0) total = 1.0;
    final mix = <String, double>{
      for (final entry in rankedByCount(_emotions)) entry.key: entry.value / total,
    };

    final denominator = minSamples * 3 > 1 ? (minSamples * 3).toDouble() : 1.0;
    final rawCoverage = samples / denominator;
    final coverage = rawCoverage < 1.0 ? rawCoverage : 1.0;
    final quality = _confidences.isEmpty ? 0.0 : mean(_confidences);
    final confidence = clampd(
      0.45 * coverage + 0.30 * dominance + 0.25 * quality,
      0.0,
      1.0,
    );

    return SessionReport(
      personId: _person,
      personName: _name,
      startedAt: _started,
      endedAt: now,
      samples: samples,
      state: state,
      confidence: confidence,
      dominance: dominance,
      valence: valence,
      arousal: arousal,
      stability: stability,
      engagement: engagement,
      fatigue: fatigue,
      tension: tension,
      emotionMix: mix,
      notes: _notes(samples, stability, fatigue, tension, engagement),
    );
  }

  (String, double) _verdict(
    double valence,
    double arousal,
    double stability,
    double fatigue,
    double tension,
  ) {
    var votes = 0;
    for (final value in _states.values) {
      votes += value;
    }
    if (votes > 0) {
      final top = rankedByCount(_states).first;
      final share = top.value / votes;
      if (share >= 0.55) {
        return (_mapState(top.key, fatigue, tension), share);
      }
    }

    var dominance = 0.0;
    if (votes > 0) {
      dominance = rankedByCount(_states).first.value / votes;
    }

    double best(double a, double b) => a > b ? a : b;
    double atMostOne(double value) => value < 1.0 ? value : 1.0;

    if (stability < 0.35) {
      return ('restless', best(dominance, 1.0 - stability));
    }
    if (tension > 0.60 && valence < 0.05) {
      return ('distressed', best(dominance, tension));
    }
    if (fatigue > 0.60 && arousal < 0.40) {
      return ('tired', best(dominance, fatigue));
    }
    if (valence <= -0.30) {
      return ('low', best(dominance, atMostOne(valence.abs())));
    }
    if (valence < -0.08) {
      return (
        arousal < 0.45 ? 'withdrawn' : 'tense',
        best(dominance, atMostOne(valence.abs() * 2)),
      );
    }
    if (valence >= 0.30) {
      return ('positive', best(dominance, atMostOne(valence)));
    }
    if (valence > 0.08) {
      return ('content', best(dominance, atMostOne(valence * 2)));
    }
    if (arousal < 0.35) return ('calm', best(dominance, 1.0 - arousal));
    return ('neutral', best(dominance, 0.5));
  }

  static String _mapState(String frameState, double fatigue, double tension) {
    if (frameState == 'volatile') return 'restless';
    if (frameState == 'negative') return 'low';
    if (frameState == 'tense') {
      return tension > 0.65 ? 'distressed' : 'tense';
    }
    if (frameState == 'neutral' && fatigue > 0.60) return 'tired';
    return sessionStates.containsKey(frameState) ? frameState : 'neutral';
  }

  List<String> _notes(
    int samples,
    double stability,
    double fatigue,
    double tension,
    double engagement,
  ) {
    final notes = <String>[];
    if (_weak > samples) notes.add('face often unclear');
    if (stability < 0.40) {
      notes.add('mood shifted repeatedly');
    } else if (stability > 0.85) {
      notes.add('steady throughout');
    }
    if (fatigue > 0.60) notes.add('signs of tiredness');
    if (tension > 0.60) notes.add('sustained tension');
    if (engagement < 0.30) notes.add('low engagement');
    return notes;
  }
}
