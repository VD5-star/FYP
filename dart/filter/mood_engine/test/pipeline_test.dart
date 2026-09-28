import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

List<double> probsOf(Map<String, double> values) {
  final v = List<double>.filled(emotions.length, 0.0);
  values.forEach((String name, double value) {
    v[emotions.indexOf(name)] = value;
  });
  final total = v.fold<double>(0.0, (double a, double b) => a + b);
  if (total == 0) return v;
  return <double>[for (final x in v) x / total];
}

void main() {
  group('the label does not flicker', () {
    test('a marginal lead never steals the label', () {
      final a = EmotionAnalyzer()..reset();
      expect(a.stableLabel(probsOf(<String, double>{
        'happy': 0.50,
        'sad': 0.30,
      })), 'happy');
      expect(a.stableLabel(probsOf(<String, double>{
        'sad': 0.41,
        'happy': 0.39,
      })), 'happy');
      expect(a.stableLabel(probsOf(<String, double>{
        'sad': 0.42,
        'happy': 0.40,
      })), 'happy');
    });

    test('a clear lead held long enough does take over', () {
      final a = EmotionAnalyzer()..reset();
      a.stableLabel(probsOf(<String, double>{'happy': 0.60, 'sad': 0.20}));
      final strong =
          probsOf(<String, double>{'sad': 0.70, 'happy': 0.10});
      for (var i = 0; i < a.config.switchFrames; i++) {
        a.stableLabel(strong);
      }
      expect(a.stableLabel(strong), 'sad');
    });

    test('resetting forgets the sticky label', () {
      final a = EmotionAnalyzer();
      a.stableLabel(probsOf(<String, double>{'anger': 0.9}));
      a.reset();
      expect(a.stickyLabel, isNull);
      expect(a.stableLabel(probsOf(<String, double>{'happy': 0.9})),
          'happy');
    });
  });

  group('smoothing respects how good the frame was', () {
    test('a poor frame moves the estimate less', () {
      final start = probsOf(<String, double>{'neutral': 1.0});
      final shock = probsOf(<String, double>{'anger': 1.0});
      final index = emotions.indexOf('anger');

      final good = EmotionAnalyzer()..smoothedProbs = List<double>.from(start);
      good.smoothProbs(shock, quality: 1.0);

      final poor = EmotionAnalyzer()..smoothedProbs = List<double>.from(start);
      poor.smoothProbs(shock, quality: 0.0);

      expect(good.smoothedProbs![index],
          greaterThan(poor.smoothedProbs![index]),
          reason: 'frame quality is not affecting the smoothing rate');
    });

    test('the smoothed output is still a distribution', () {
      final a = EmotionAnalyzer();
      for (final q in <double>[0.0, 0.5, 1.0]) {
        final out = a.smoothProbs(
            probsOf(<String, double>{'happy': 0.6, 'sad': 0.4}),
            quality: q);
        final sum = out.fold<double>(0.0, (double x, double y) => x + y);
        expect((sum - 1.0).abs(), lessThan(1e-6));
        for (final v in out) {
          expect(v, greaterThanOrEqualTo(0.0));
        }
      }
    });
  });
}
