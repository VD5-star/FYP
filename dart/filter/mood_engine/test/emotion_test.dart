import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

List<double> probsOf(Map<String, double> values) {
  final vector = List<double>.filled(emotions.length, 0.0);
  for (final entry in values.entries) {
    vector[emotionIndex(entry.key)] = entry.value;
  }
  final total = sumOf(vector);
  return total == 0
      ? vector
      : <double>[for (final value in vector) value / total];
}

void main() {
  test('label does not flip on a marginal lead', () {
    final analyzer = EmotionAnalyzer()..reset();

    expect(
      analyzer.stableLabel(probsOf(<String, double>{
        'happy': 0.50,
        'sad': 0.30,
      })),
      'happy',
    );
    expect(
      analyzer.stableLabel(probsOf(<String, double>{
        'sad': 0.41,
        'happy': 0.39,
      })),
      'happy',
    );
    expect(
      analyzer.stableLabel(probsOf(<String, double>{
        'sad': 0.42,
        'happy': 0.40,
      })),
      'happy',
    );
  });

  test('label switches on a sustained clear lead', () {
    final analyzer = EmotionAnalyzer()..reset();
    analyzer.stableLabel(
      probsOf(<String, double>{'happy': 0.60, 'sad': 0.20}),
    );

    final strong = probsOf(<String, double>{'sad': 0.70, 'happy': 0.10});
    for (var i = 0; i < analyzer.config.switchFrames; i++) {
      analyzer.stableLabel(strong);
    }
    expect(analyzer.stableLabel(strong), 'sad');
  });

  test('reset clears sticky label', () {
    final analyzer = EmotionAnalyzer();
    analyzer.stableLabel(probsOf(<String, double>{'anger': 0.9}));
    analyzer.reset();
    expect(analyzer.stickyLabel, isNull);
    expect(
      analyzer.stableLabel(probsOf(<String, double>{'happy': 0.9})),
      'happy',
    );
  });

  test('low quality frames move the estimate less', () {
    final start = probsOf(<String, double>{'neutral': 1.0});
    final shock = probsOf(<String, double>{'anger': 1.0});

    final good = EmotionAnalyzer()..smoothedProbs = List<double>.from(start);
    good.smoothProbs(shock);

    final poor = EmotionAnalyzer()..smoothedProbs = List<double>.from(start);
    poor.smoothProbs(shock, quality: 0.0);

    final index = emotionIndex('anger');
    expect(good.smoothedProbs![index], greaterThan(poor.smoothedProbs![index]),
        reason: 'frame quality is not affecting the smoothing rate');
    expect(good.smoothedProbs![index], closeTo(0.35, 1e-12));
    expect(poor.smoothedProbs![index], closeTo(0.12, 1e-12));
  });

  test('smoothed output stays a distribution', () {
    final analyzer = EmotionAnalyzer();
    for (final quality in <double>[0.0, 0.5, 1.0]) {
      final out = analyzer.smoothProbs(
        probsOf(<String, double>{'happy': 0.6, 'sad': 0.4}),
        quality: quality,
      );
      expect((sumOf(out) - 1.0).abs(), lessThan(1e-6));
      expect(out.every((double v) => v >= 0), isTrue);
    }
  });

  test('probabilities form a valid distribution', () {
    final model = FakeEmotionModel()..setLabel('happy');
    final analyzer = EmotionAnalyzer(model: model);
    final result = analyzer.analyse(
      crop: FaceImage.blank(224),
      smooth: false,
    );

    expect(result.probs.keys.toSet(), emotions.toSet());
    expect((sumOf(result.probs.values) - 1.0).abs(), lessThan(1e-5));
    expect(result.probs.values.every((double p) => p >= 0.0 && p <= 1.0),
        isTrue);
    expect(result.label, 'happy');
    expect(result.confidence, closeTo(result.probs[result.label]!, 1e-12));
  });

  test('contempt is folded into disgust', () {
    final model = FakeEmotionModel()..setLabel('contempt', strength: 9.0);
    final analyzer = EmotionAnalyzer(model: model);
    final result = analyzer.analyse(
      crop: FaceImage.blank(224),
      smooth: false,
    );
    expect(result.label, 'disgust');
    expect(result.probs.containsKey('contempt'), isFalse);
  });

  test('valence and arousal stay within bounds', () {
    for (final label in emotions) {
      final va = EmotionAnalyzer.valenceArousal(
        probsOf(<String, double>{label: 1.0}),
        const <String, double>{},
      );
      expect(va.$1, inInclusiveRange(-1.0, 1.0));
      expect(va.$2, inInclusiveRange(0.0, 1.0));
    }
  });

  test('valence and arousal match the python mapping', () {
    final neutral = EmotionAnalyzer.valenceArousal(
      probsOf(<String, double>{'neutral': 1.0}),
      const <String, double>{},
    );
    expect(neutral.$1, closeTo(0.0, 1e-12));
    expect(neutral.$2, closeTo(0.2, 1e-12));

    final happy = EmotionAnalyzer.valenceArousal(
      probsOf(<String, double>{'happy': 1.0}),
      const <String, double>{},
    );
    expect(happy.$1, closeTo(0.8, 1e-12));
    expect(happy.$2, closeTo(0.65, 1e-12));

    final withUnits = EmotionAnalyzer.valenceArousal(
      probsOf(<String, double>{'happy': 1.0}),
      const <String, double>{
        'mouth_open': 0.5,
        'eye_open': 0.5,
        'smile': 0.08,
      },
    );
    expect(withUnits.$1, closeTo(0.88, 1e-12));
    expect(withUnits.$2, closeTo(0.7190000000000001, 1e-12));

    final mix = EmotionAnalyzer.valenceArousal(
      probsOf(<String, double>{'happy': 0.5, 'sad': 0.5}),
      const <String, double>{},
    );
    expect(mix.$1, closeTo(0.050000000000000044, 1e-12));
    expect(mix.$2, closeTo(0.45, 1e-12));
  });

  test('geometric fallback is neutral without action units', () {
    final probs = EmotionAnalyzer.inferGeometric(const <String, double>{});
    expect(emotions[argMax(probs)], 'neutral');
    expect(probs[emotionIndex('neutral')], closeTo(1.0, 1e-12));
  });

  test('geometric fallback matches the python distribution', () {
    final probs = EmotionAnalyzer.inferGeometric(const <String, double>{
      'smile': 0.06,
      'mouth_open': 0.2,
      'brow_raise': 0.35,
      'brow_knit': 0.7,
      'eye_open': 0.3,
    });
    final expected = <double>[
      0.569897,
      0.114372,
      0.063146,
      0.063146,
      0.063146,
      0.063146,
      0.063146,
    ];
    for (var i = 0; i < emotions.length; i++) {
      expect(probs[i], closeTo(expected[i], 1e-6), reason: emotions[i]);
    }
    expect(emotions[argMax(probs)], 'neutral');
  });

  test('arabic label is present', () {
    final model = FakeEmotionModel()..setLabel('sad');
    final analyzer = EmotionAnalyzer(model: model);
    final result = analyzer.analyse(
      crop: FaceImage.blank(224),
      smooth: false,
    );
    expect(result.labelAr, emotionsAr[result.label]);
    expect(result.labelAr.trim(), isNotEmpty);
  });

  test('smoothing reduces jitter', () {
    final model = FakeEmotionModel()..setLabel('happy');
    final analyzer = EmotionAnalyzer(model: model);

    for (var i = 0; i < 6; i++) {
      analyzer.analyse(crop: FaceImage.blank(224));
    }
    final settled = analyzer.analyse(crop: FaceImage.blank(224));

    model.setLabel('sad');
    final perturbed = analyzer.analyse(crop: FaceImage.blank(224));

    final fresh = EmotionAnalyzer(model: FakeEmotionModel()..setLabel('sad'));
    final unsmoothed = fresh.analyse(
      crop: FaceImage.blank(224),
      smooth: false,
    );

    final drift =
        (perturbed.probs[settled.label]! - settled.probs[settled.label]!)
            .abs();
    final rawDrift =
        (unsmoothed.probs[settled.label]! - settled.probs[settled.label]!)
            .abs();
    expect(drift, lessThanOrEqualTo(rawDrift + 1e-9));
  });

  test('trend reports dominant emotion', () {
    final analyzer = EmotionAnalyzer(
      model: FakeEmotionModel()..setLabel('happy'),
    );

    for (var i = 0; i < 8; i++) {
      analyzer.analyse(crop: FaceImage.blank(224));
    }

    final trend = analyzer.trend();
    expect(trend['samples'], 8);
    expect(emotions, contains(trend['dominant']));
    expect(trend['stability'] as double, inInclusiveRange(0.0, 1.0));
  });

  test('empty trend is well formed', () {
    final trend = EmotionAnalyzer().trend();
    expect(trend['samples'], 0);
    expect(trend['dominant'], isNull);
    expect(trend['stability'], 0.0);
  });

  test('backend reports whether a model is attached', () {
    expect(EmotionAnalyzer().backend, 'geometric');
    expect(EmotionAnalyzer(model: FakeEmotionModel()).backend, 'onnx');
  });

  test('rebalance hook is applied and tagged', () {
    final analyzer = EmotionAnalyzer(
      model: FakeEmotionModel()..setLabel('happy'),
    );
    final result = analyzer.analyse(
      crop: FaceImage.blank(224),
      smooth: false,
      rebalance: (List<double> p, Map<String, double> aus) =>
          List<double>.filled(emotions.length, 1.0),
    );
    expect(result.source, 'onnx+baseline');
    expect(sumOf(result.probs.values), closeTo(1.0, 1e-9));
  });

  test('action units need a full landmark set', () {
    expect(EmotionAnalyzer.actionUnits(null), isEmpty);
    expect(
      EmotionAnalyzer.actionUnits(
        List<Point2>.filled(20, const Point2(0, 0)),
      ),
      isEmpty,
    );
  });

  test('action units are derived from landmark geometry', () {
    final landmarks = List<Point2>.generate(
      106,
      (int i) => Point2(i.toDouble(), (i % 7).toDouble()),
    );
    final units = EmotionAnalyzer.actionUnits(landmarks);
    for (final key in const <String>[
      'eye_open',
      'brow_raise',
      'brow_knit',
      'mouth_open',
      'mouth_width',
      'smile',
    ]) {
      expect(units.containsKey(key), isTrue, reason: key);
      expect(units[key]!.isFinite, isTrue, reason: key);
    }
  });
}
