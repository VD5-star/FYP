import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

List<SessionReport> feed(
  SessionAnalyser analyser,
  int count, {
  int? person = 1,
  double valence = 0.0,
  double arousal = 0.4,
  String state = 'neutral',
  String emotion = 'neutral',
  double confidence = 0.8,
  double start = 0.0,
  double step = 0.11,
  Map<String, double>? units,
}) {
  final resolved = units ??
      const <String, double>{
        'engagement': 0.5,
        'fatigue': 0.2,
        'tension': 0.2,
      };
  final reports = <SessionReport>[];
  for (var i = 0; i < count; i++) {
    final report = analyser.observe(
      personId: person,
      personName: 'P$person',
      valence: valence,
      arousal: arousal,
      moodState: state,
      emotion: emotion,
      confidence: confidence,
      units: resolved,
      now: start + i * step,
    );
    if (report != null) reports.add(report);
  }
  return reports;
}

void main() {
  test('thin window reports nothing', () {
    final analyser = SessionAnalyser(periodS: 30, minSamples: 60, now: 0.0);
    feed(analyser, 20);
    expect(analyser.flush(now: 40.0), isNull);
  });

  test('period closes on time', () {
    final analyser = SessionAnalyser(periodS: 30, minSamples: 50, now: 0.0);
    final reports = feed(analyser, 400);
    expect(reports.length, greaterThanOrEqualTo(1));
    expect(reports.first.durationS, closeTo(30.0, 1.0));
  });

  test('person change closes the window', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    feed(analyser, 100, valence: -0.5, state: 'negative', emotion: 'sad');
    final report = analyser.observe(
      personId: 2,
      personName: 'P2',
      valence: 0.5,
      arousal: 0.5,
      moodState: 'positive',
      emotion: 'happy',
      confidence: 0.8,
      now: 11.0,
    );
    expect(report, isNotNull);
    expect(report!.personId, 1);
    expect(report.valence, lessThan(0));
  });

  test('previous person does not leak into the next', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    feed(analyser, 120, valence: -0.7, state: 'negative', emotion: 'sad');
    analyser.observe(
      personId: 2,
      personName: 'P2',
      valence: 0.6,
      arousal: 0.5,
      moodState: 'positive',
      emotion: 'happy',
      confidence: 0.8,
      now: 14.0,
    );
    feed(
      analyser,
      120,
      person: 2,
      valence: 0.6,
      state: 'positive',
      emotion: 'happy',
      start: 15.0,
    );
    final report = analyser.flush(now: 60.0);
    expect(report, isNotNull);
    expect(report!.personId, 2);
    expect(report.valence, greaterThan(0.4),
        reason: "previous person's readings leaked");
    expect(report.emotionMix.containsKey('sad'), isFalse);
  });

  test('swinging person is restless not neutral', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    for (var i = 0; i < 160; i++) {
      final positive = (i ~/ 10) % 2 == 0;
      analyser.observe(
        personId: 5,
        personName: 'P5',
        valence: positive ? 0.7 : -0.7,
        arousal: 0.6,
        moodState: positive ? 'positive' : 'negative',
        emotion: positive ? 'happy' : 'sad',
        confidence: 0.8,
        units: const <String, double>{
          'engagement': 0.5,
          'fatigue': 0.2,
          'tension': 0.3,
        },
        now: i * 0.11,
      );
    }
    final report = analyser.flush(now: 30.0);
    expect(report, isNotNull);
    expect(report!.state, 'restless');
    expect(report.stability, lessThan(0.3));
    expect(report.notes, contains('mood shifted repeatedly'));
  });

  test('low confidence frames do not vote', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    feed(analyser, 100, valence: 0.5, state: 'positive', emotion: 'happy');
    feed(
      analyser,
      100,
      valence: -0.9,
      state: 'negative',
      emotion: 'sad',
      confidence: 0.1,
      start: 12.0,
    );
    final report = analyser.flush(now: 30.0);
    expect(report, isNotNull);
    expect(report!.valence, greaterThan(0.3));
    expect(report.emotionMix.containsKey('sad'), isFalse);
  });

  test('sustained state wins over derived rules', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    feed(
      analyser,
      120,
      valence: -0.5,
      arousal: 0.3,
      state: 'negative',
      emotion: 'sad',
    );
    final report = analyser.flush(now: 20.0);
    expect(report, isNotNull);
    expect(report!.state, 'low');
    expect(report.dominance, greaterThan(0.9));
  });

  test('tiredness is distinguished from sadness', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    feed(
      analyser,
      120,
      arousal: 0.2,
      units: const <String, double>{
        'engagement': 0.2,
        'fatigue': 0.8,
        'tension': 0.1,
      },
    );
    final report = analyser.flush(now: 20.0);
    expect(report, isNotNull);
    expect(report!.state, 'tired');
    expect(report.notes, contains('signs of tiredness'));
  });

  test('every state has a bilingual label', () {
    for (final entry in sessionStates.entries) {
      final english = entry.value[0];
      final arabic = entry.value[1];
      expect(english, isNotEmpty, reason: '${entry.key} is missing a label');
      expect(arabic, isNotEmpty, reason: '${entry.key} is missing a label');
      expect(english, isNot(arabic),
          reason: '${entry.key} was never translated');
      expect(
        arabic.runes.any((int r) => r >= 0x600 && r <= 0x6ff),
        isTrue,
        reason: entry.key,
      );
    }
  });

  test('report is serialisable', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    feed(analyser, 120, valence: 0.5, state: 'positive', emotion: 'happy');
    final report = analyser.flush(now: 20.0);
    expect(report, isNotNull);
    final payload = report!.toMap();
    expect(payload['state'], 'positive');
    expect(payload['label_en'] as String, isNotEmpty);
    expect(payload['label_ar'] as String, isNotEmpty);
    expect(payload['emotion_mix'], isA<Map<String, double>>());
    expect(payload['notes'], isA<List<String>>());
    expect(payload['confidence'], closeTo(0.86, 1e-9));
    expect(payload['dominance'], closeTo(1.0, 1e-9));
    expect(payload['samples'], 120);
    expect(payload['duration_s'], closeTo(20.0, 1e-9));
    expect(payload['notes'], <String>['steady throughout']);
  });

  test('progress reports the slower of time and evidence', () {
    final analyser = SessionAnalyser(periodS: 100, minSamples: 100, now: 0.0);
    feed(analyser, 10, step: 1.0);
    final progress = analyser.progress(now: 10.0);
    expect(progress['progress'] as double, closeTo(0.1, 0.02));
    expect(progress['will_report'], isFalse);
  });

  test('flush restarts cleanly', () {
    final analyser = SessionAnalyser(periodS: 600, minSamples: 50, now: 0.0);
    feed(analyser, 120, valence: 0.5, state: 'positive', emotion: 'happy');
    expect(analyser.flush(now: 20.0), isNotNull);
    expect(analyser.flush(now: 21.0), isNull,
        reason: 'evidence was not cleared');
  });
}
