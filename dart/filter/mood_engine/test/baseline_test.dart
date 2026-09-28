import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

final int neutralIndex = emotionIndex('neutral');
final int happyIndex = emotionIndex('happy');
final int angerIndex = emotionIndex('anger');

List<double> probs(Map<String, double> weights) {
  final vector = List<double>.filled(emotions.length, 0.02);
  for (final entry in weights.entries) {
    vector[emotionIndex(entry.key)] = entry.value;
  }
  final total = sumOf(vector);
  return <double>[for (final value in vector) value / total];
}

void calibrate(
  BaselineTracker tracker,
  List<double> resting, {
  int personId = 1,
  int? frames,
}) {
  final count = frames ?? PersonBaseline.required;
  for (var i = 0; i < count; i++) {
    tracker.adjust(resting, <String, double>{}, personId);
  }
}

void main() {
  test('baseline needs enough samples before it is trusted', () {
    final tracker = BaselineTracker();
    final reading = probs(<String, double>{'anger': 0.6, 'neutral': 0.2});

    final result = tracker.adjust(reading, <String, double>{}, 1);
    expect(result.$2.ready, isFalse);
    for (var i = 0; i < emotions.length; i++) {
      expect(result.$1[i], closeTo(reading[i], 1e-12),
          reason: 'readings must pass through untouched');
    }
  });

  test('baseline forms from ordinary frames', () {
    final tracker = BaselineTracker();
    calibrate(tracker, probs(<String, double>{'anger': 0.55, 'neutral': 0.08}));
    expect(tracker.baselineFor(1).ready, isTrue);
  });

  test('extreme frames stop shifting an established baseline', () {
    final tracker = BaselineTracker();
    calibrate(
      tracker,
      probs(<String, double>{'neutral': 0.35, 'happy': 0.25, 'sad': 0.15}),
    );

    final settled = List<double>.from(tracker.baselineFor(1).probs);
    final samplesBefore = tracker.baselineFor(1).samples;

    final laughing = probs(<String, double>{'happy': 0.97});
    for (var i = 0; i < 60; i++) {
      tracker.adjust(laughing, <String, double>{}, 1);
    }

    expect(tracker.baselineFor(1).samples, samplesBefore,
        reason: 'extreme frames should not be recorded once calibrated');
    for (var i = 0; i < emotions.length; i++) {
      expect(tracker.baselineFor(1).probs[i], closeTo(settled[i], 1e-12),
          reason: 'a long laugh must not become the resting face');
    }
  });

  test('progress is reported while calibrating', () {
    final tracker = BaselineTracker();
    final resting = probs(<String, double>{'neutral': 0.3, 'anger': 0.3});

    for (var i = 0; i < 10; i++) {
      tracker.adjust(resting, <String, double>{}, 1);
    }
    final baseline = tracker.baselineFor(1);

    expect(baseline.progress, greaterThan(0.0));
    expect(baseline.progress, lessThan(1.0));
    calibrate(tracker, resting);
    expect(tracker.baselineFor(1).progress, 1.0);
  });

  test('a persons resting bias is suppressed', () {
    final tracker = BaselineTracker();
    final resting = probs(
      <String, double>{'anger': 0.50, 'neutral': 0.10, 'sad': 0.15},
    );
    calibrate(tracker, resting);

    final adjusted = tracker.adjust(resting, <String, double>{}, 1).$1;
    expect(adjusted[angerIndex], lessThan(resting[angerIndex]),
        reason: 'the habitual anger reading must be reduced');
  });

  test('a real expression still registers', () {
    final tracker = BaselineTracker();
    calibrate(
      tracker,
      probs(<String, double>{'anger': 0.45, 'neutral': 0.12}),
    );

    final smiling = probs(<String, double>{'happy': 0.70, 'neutral': 0.10});
    final adjusted = tracker.adjust(smiling, <String, double>{}, 1).$1;

    expect(emotions[argMax(adjusted)], 'happy');
    expect(adjusted[happyIndex], greaterThan(0.3));
  });

  test('an unusual emotion is amplified', () {
    final tracker = BaselineTracker();
    calibrate(
      tracker,
      probs(<String, double>{'neutral': 0.4, 'happy': 0.3, 'anger': 0.2}),
    );

    final surprised = probs(
      <String, double>{'surprise': 0.35, 'neutral': 0.3, 'happy': 0.2},
    );
    final adjusted = tracker.adjust(surprised, <String, double>{}, 1).$1;

    final rawRank = _rankOf(surprised, emotionIndex('surprise'));
    final adjRank = _rankOf(adjusted, emotionIndex('surprise'));
    expect(adjRank, lessThanOrEqualTo(rawRank),
        reason: 'a rare class should rank at least as high');
  });

  test('output stays a valid distribution', () {
    final tracker = BaselineTracker();
    final resting = probs(<String, double>{'anger': 0.5, 'neutral': 0.05});
    calibrate(tracker, resting);

    final readings = <List<double>>[
      probs(<String, double>{'happy': 0.8}),
      probs(<String, double>{'fear': 0.4, 'sad': 0.4}),
      probs(<String, double>{'neutral': 0.9}),
      resting,
    ];
    for (final reading in readings) {
      final adjusted = tracker.adjust(reading, <String, double>{}, 1).$1;
      expect(adjusted.every((double v) => v >= 0), isTrue);
      expect(sumOf(adjusted), closeTo(1.0, 1e-9));
      expect(adjusted.every((double v) => v.isFinite), isTrue);
    }
  });

  test('baselines are kept separate per person', () {
    final tracker = BaselineTracker();
    calibrate(
      tracker,
      probs(<String, double>{'anger': 0.55, 'neutral': 0.10}),
    );
    calibrate(
      tracker,
      probs(<String, double>{'happy': 0.55, 'neutral': 0.10}),
      personId: 2,
    );

    final one = tracker.baselineFor(1).probs;
    final two = tracker.baselineFor(2).probs;
    expect(one[angerIndex], greaterThan(two[angerIndex]));
    expect(two[happyIndex], greaterThan(one[happyIndex]));
  });

  test('a baseline survives a round trip', () {
    final tracker = BaselineTracker();
    calibrate(
      tracker,
      probs(<String, double>{'anger': 0.45, 'neutral': 0.20, 'sad': 0.15}),
    );

    final payload = tracker.export(1);
    expect(payload, isNotNull);
    expect(payload!['samples'] as int,
        greaterThanOrEqualTo(PersonBaseline.required));

    final restored = BaselineTracker()..load(1, payload);
    expect(restored.baselineFor(1).ready, isTrue);
    for (var i = 0; i < emotions.length; i++) {
      expect(restored.baselineFor(1).probs[i],
          closeTo(tracker.baselineFor(1).probs[i], 1e-4));
    }
  });

  test('forget clears calibration', () {
    final tracker = BaselineTracker();
    calibrate(
      tracker,
      probs(<String, double>{'anger': 0.45, 'neutral': 0.25, 'sad': 0.15}),
    );
    expect(tracker.baselineFor(1).ready, isTrue);

    tracker.forget(1);
    expect(tracker.baselineFor(1).ready, isFalse);
  });

  test('mood needs a few frames before committing', () {
    final tracker = BaselineTracker();
    final summary = tracker.updateMood(0.8, 0.7, 0.1, 0.1, 1);
    expect(summary.state, 'neutral');
    expect(summary.confidence, 0.0);
  });

  test('sustained positive affect reads positive', () {
    final tracker = BaselineTracker();
    late MoodSummary summary;
    for (var i = 0; i < 40; i++) {
      summary = tracker.updateMood(0.65, 0.60, 0.1, 0.05, 1);
    }
    expect(<String>['positive', 'content'], contains(summary.state));
    expect(summary.valence, greaterThan(0.5));
  });

  test('calm positive affect reads content', () {
    final tracker = BaselineTracker();
    late MoodSummary summary;
    for (var i = 0; i < 40; i++) {
      summary = tracker.updateMood(0.45, 0.20, 0.1, 0.05, 1);
    }
    expect(summary.state, 'content');
  });

  test('sustained negative affect reads negative or tense', () {
    final tracker = BaselineTracker();
    late MoodSummary summary;
    for (var i = 0; i < 40; i++) {
      summary = tracker.updateMood(-0.55, 0.30, 0.2, 0.05, 1);
    }
    expect(<String>['negative', 'withdrawn', 'tense'], contains(summary.state));
  });

  test('high tension reads tense', () {
    final tracker = BaselineTracker();
    late MoodSummary summary;
    for (var i = 0; i < 40; i++) {
      summary = tracker.updateMood(-0.45, 0.65, 0.8, 0.1, 1);
    }
    expect(summary.state, 'tense');
  });

  test('swinging affect reads volatile', () {
    final tracker = BaselineTracker();
    late MoodSummary summary;
    for (var i = 0; i < 60; i++) {
      final sign = i % 2 == 0 ? 1.0 : -1.0;
      summary = tracker.updateMood(0.75 * sign, 0.5, 0.2, 0.8, 1);
    }
    expect(summary.state, 'volatile');
    expect(summary.stability, lessThan(0.5));
  });

  test('mood window resets when the person changes', () {
    final tracker = BaselineTracker();
    for (var i = 0; i < 40; i++) {
      tracker.updateMood(0.8, 0.7, 0.1, 0.05, 1);
    }

    final summary = tracker.updateMood(-0.5, 0.3, 0.2, 0.1, 2);
    expect(summary.samples, 1, reason: 'the window should restart');
  });

  test('summary serialises cleanly', () {
    final tracker = BaselineTracker();
    late MoodSummary summary;
    for (var i = 0; i < 30; i++) {
      summary = tracker.updateMood(0.5, 0.5, 0.2, 0.1, 1);
    }

    final payload = summary.toMap();
    for (final key in const <String>[
      'state',
      'state_en',
      'state_ar',
      'confidence',
      'valence',
      'energy',
      'stability',
      'calibrated',
      'baseline_progress',
    ]) {
      expect(payload.containsKey(key), isTrue, reason: key);
    }
    expect(payload['confidence'] as double, inInclusiveRange(0.0, 1.0));
    expect(payload['state_ar'] as String, isNotEmpty,
        reason: 'Arabic label must be present');
  });

  test('mood states are all bilingual', () {
    for (final entry in moodStates.entries) {
      expect(entry.value[0], isNotEmpty);
      expect(entry.value[1], isNotEmpty);
      expect(
        entry.value[1].runes.any((int r) => r >= 0x600 && r <= 0x6ff),
        isTrue,
        reason: entry.key,
      );
    }
  });
}

int _rankOf(List<double> values, int index) {
  final sorted = List<int>.generate(values.length, (int i) => i)
    ..sort((int a, int b) => values[b].compareTo(values[a]));
  return sorted.indexOf(index);
}
