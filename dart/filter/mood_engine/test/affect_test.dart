import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

AffectSignals fakeEmotion({
  String label = 'neutral',
  double valence = 0.0,
  double arousal = 0.3,
  Map<String, double>? probs,
}) => AffectSignals(
  label: label,
  valence: valence,
  arousal: arousal,
  probs: probs ?? const <String, double>{'neutral': 1.0},
);

AffectPose fakePose({
  double attention = 0.8,
  double eyeOpenness = 0.4,
  double pitch = 0.0,
  bool isBlinking = false,
}) => AffectPose(
  attention: attention,
  eyeOpenness: eyeOpenness,
  pitch: pitch,
  isBlinking: isBlinking,
);

void main() {
  test('genuine smile scores above social smile', () {
    final analyzer = AffectAnalyzer();

    final social = analyzer.update(
      fakeEmotion(label: 'happy', probs: <String, double>{'happy': 0.9}),
      fakePose(),
      <String, double>{'smile': 0.05, 'eye_open': 0.34},
    );

    analyzer.reset();
    final genuine = analyzer.update(
      fakeEmotion(label: 'happy', probs: <String, double>{'happy': 0.9}),
      fakePose(),
      <String, double>{'smile': 0.05, 'eye_open': 0.19},
    );

    expect(genuine.duchenne, greaterThan(social.duchenne));
    expect(genuine.smileType, 'genuine');
    expect(social.smileType, 'social');
    expect(social.duchenne, closeTo(0.45, 1e-12));
    expect(genuine.duchenne, closeTo(0.9541666666666666, 1e-12));
  });

  test('no smile reports none', () {
    final analyzer = AffectAnalyzer();
    final reading = analyzer.update(
      fakeEmotion(label: 'neutral', probs: <String, double>{'neutral': 0.9}),
      fakePose(),
      <String, double>{'smile': 0.0, 'eye_open': 0.30},
    );
    expect(reading.smileType, 'none');
    expect(reading.duchenne, 0.0);
  });

  test('compound emotion is named for a real mixture', () {
    final analyzer = AffectAnalyzer();
    final reading = analyzer.update(
      fakeEmotion(
        label: 'sad',
        probs: <String, double>{'sad': 0.45, 'fear': 0.35, 'neutral': 0.20},
      ),
      fakePose(),
      <String, double>{},
    );
    expect(reading.compound, 'anxious');
    expect(reading.compoundAr, isNotNull);
    expect(reading.compoundAr, isNotEmpty);
  });

  test('dominant emotion reports no compound', () {
    final analyzer = AffectAnalyzer();
    final reading = analyzer.update(
      fakeEmotion(
        label: 'happy',
        probs: <String, double>{'happy': 0.95, 'neutral': 0.05},
      ),
      fakePose(),
      <String, double>{},
    );
    expect(reading.compound, isNull);
  });

  test('mixture lists meaningful components only', () {
    final analyzer = AffectAnalyzer();
    final reading = analyzer.update(
      fakeEmotion(
        probs: <String, double>{
          'happy': 0.50,
          'surprise': 0.30,
          'sad': 0.15,
          'fear': 0.05,
        },
      ),
      fakePose(),
      <String, double>{},
    );
    final names = <String>[for (final entry in reading.mixture) entry.key];
    expect(names, contains('happy'));
    expect(names, contains('surprise'));
    expect(names, isNot(contains('fear')),
        reason: 'negligible components should be dropped');
  });

  test('engagement needs more than a blank stare', () {
    final blank = AffectAnalyzer().update(
      fakeEmotion(probs: <String, double>{'neutral': 1.0}),
      fakePose(attention: 0.95),
      <String, double>{},
    );

    final lively = AffectAnalyzer().update(
      fakeEmotion(
        label: 'happy',
        probs: <String, double>{'happy': 0.9, 'neutral': 0.1},
      ),
      fakePose(attention: 0.95),
      <String, double>{},
    );

    expect(lively.engagement, greaterThan(blank.engagement));
    expect(blank.engagement, closeTo(0.6175, 1e-9));
    expect(lively.engagement, closeTo(0.9325, 1e-9));
  });

  test('fatigue rises with closed eyes and head droop', () {
    final alert = AffectAnalyzer();
    final tired = AffectAnalyzer();
    for (var i = 0; i < 30; i++) {
      alert.update(
        fakeEmotion(),
        fakePose(eyeOpenness: 0.55),
        <String, double>{},
      );
      tired.update(
        fakeEmotion(),
        fakePose(eyeOpenness: 0.15, pitch: -28.0),
        <String, double>{},
      );
    }

    final a = alert.update(
      fakeEmotion(),
      fakePose(eyeOpenness: 0.55),
      <String, double>{},
    );
    final t = tired.update(
      fakeEmotion(),
      fakePose(eyeOpenness: 0.15, pitch: -28.0),
      <String, double>{},
    );
    expect(t.fatigue, greaterThan(a.fatigue));
    expect(t.fatigue, inInclusiveRange(0.0, 1.0));
    expect(a.fatigue, closeTo(0.0, 1e-12));
    expect(t.fatigue, closeTo(0.6772727272727272, 1e-12));
  });

  test('tension rises with knitted brow and negative affect', () {
    final calm = AffectAnalyzer().update(
      fakeEmotion(probs: <String, double>{'neutral': 1.0}),
      fakePose(),
      <String, double>{'brow_knit': 1.10, 'mouth_open': 0.30},
    );
    final tense = AffectAnalyzer().update(
      fakeEmotion(
        label: 'anger',
        probs: <String, double>{'anger': 0.7, 'fear': 0.3},
      ),
      fakePose(),
      <String, double>{'brow_knit': 0.62, 'mouth_open': 0.12},
    );
    expect(tense.tension, greaterThan(calm.tension));
    expect(calm.tension, closeTo(0.0, 1e-12));
    expect(tense.tension, closeTo(0.9271428571428572, 1e-12));
  });

  test('blink rate counts only transitions', () {
    final analyzer = AffectAnalyzer();
    for (var i = 0; i < 10; i++) {
      analyzer.update(
        fakeEmotion(),
        fakePose(isBlinking: true),
        <String, double>{},
      );
    }
    final reading = analyzer.update(
      fakeEmotion(),
      fakePose(isBlinking: true),
      <String, double>{},
    );
    expect(analyzer.blinkCount, 1);
    expect(reading.blinkRate, greaterThanOrEqualTo(0.0));
  });

  test('volatility separates steady from swinging affect', () {
    final steady = AffectAnalyzer();
    final swinging = AffectAnalyzer();
    for (var i = 0; i < 40; i++) {
      steady.update(
        fakeEmotion(valence: 0.5, arousal: 0.5),
        fakePose(),
        <String, double>{},
      );
      final sign = i % 2 == 0 ? 1.0 : -1.0;
      swinging.update(
        fakeEmotion(valence: 0.8 * sign, arousal: 0.5 + 0.4 * sign),
        fakePose(),
        <String, double>{},
      );
    }

    final a = steady.update(
      fakeEmotion(valence: 0.5, arousal: 0.5),
      fakePose(),
      <String, double>{},
    );
    final b = swinging.update(
      fakeEmotion(valence: -0.8, arousal: 0.1),
      fakePose(),
      <String, double>{},
    );
    expect(b.volatility, greaterThan(a.volatility));
    expect(a.volatility, closeTo(0.0, 1e-12));
    expect(b.volatility, closeTo(1.0, 1e-12));
  });

  test('transition is reported once per change', () {
    final analyzer = AffectAnalyzer();
    analyzer.update(
      fakeEmotion(),
      fakePose(),
      <String, double>{},
    );
    final changed = analyzer.update(
      fakeEmotion(label: 'happy'),
      fakePose(),
      <String, double>{},
    );
    final same = analyzer.update(
      fakeEmotion(label: 'happy'),
      fakePose(),
      <String, double>{},
    );

    expect(changed.transition, 'neutral -> happy');
    expect(same.transition, isNull);
  });

  test('summary reports session shape', () {
    final analyzer = AffectAnalyzer();
    for (var i = 0; i < 12; i++) {
      analyzer.update(
        fakeEmotion(label: 'happy', valence: 0.7),
        fakePose(),
        <String, double>{},
      );
    }
    for (var i = 0; i < 4; i++) {
      analyzer.update(
        fakeEmotion(label: 'sad', valence: -0.5),
        fakePose(),
        <String, double>{},
      );
    }

    final summary = analyzer.summary();
    expect(summary['samples'], 16);
    expect(summary['dominant'], 'happy');
    expect(summary['distinct_emotions'], 2);
    expect(summary['stability'] as double, inInclusiveRange(0.0, 1.0));
    expect(summary['valence_range'] as double, greaterThan(0));
    expect(summary['stability'], closeTo(0.75, 1e-12));
    expect(summary['mean_valence'] as double, closeTo(0.4, 1e-12));
    expect(summary['valence_range'] as double, closeTo(1.2, 1e-12));
    expect(summary['dominant_ar'], emotionsAr['happy']);
  });

  test('reset clears all state', () {
    final analyzer = AffectAnalyzer();
    for (var i = 0; i < 10; i++) {
      analyzer.update(
        fakeEmotion(label: 'anger'),
        fakePose(),
        <String, double>{},
      );
    }
    analyzer.reset();
    expect(analyzer.summary()['samples'], 0);
  });

  test('reading serialises every documented key', () {
    final analyzer = AffectAnalyzer();
    final reading = analyzer.update(
      fakeEmotion(label: 'happy', probs: <String, double>{'happy': 0.9}),
      fakePose(),
      <String, double>{'smile': 0.05, 'eye_open': 0.19},
    );
    final payload = reading.toMap();
    for (final key in const <String>[
      'duchenne',
      'smile_type',
      'engagement',
      'fatigue',
      'tension',
      'volatility',
      'expressiveness',
    ]) {
      expect(payload.containsKey(key), isTrue, reason: key);
    }
    for (final key in const <String>[
      'duchenne',
      'engagement',
      'fatigue',
      'tension',
      'volatility',
      'expressiveness',
    ]) {
      expect(payload[key] as double, inInclusiveRange(0.0, 1.0),
          reason: '$key out of range');
    }
  });
}
