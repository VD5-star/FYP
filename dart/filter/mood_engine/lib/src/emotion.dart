import 'dart:collection';
import 'dart:math' as math;

import 'config.dart';
import 'geometry.dart';
import 'maths.dart';

const List<String> hse8 = <String>[
  'anger',
  'contempt',
  'disgust',
  'fear',
  'happy',
  'neutral',
  'sad',
  'surprise',
];

const List<String> hse7 = <String>[
  'anger',
  'disgust',
  'fear',
  'happy',
  'neutral',
  'sad',
  'surprise',
];

const String contemptTarget = 'disgust';

const List<double> imagenetMean = <double>[0.485, 0.456, 0.406];
const List<double> imagenetStd = <double>[0.229, 0.224, 0.225];

class EmotionResult {
  const EmotionResult({
    required this.label,
    required this.labelAr,
    required this.confidence,
    required this.probs,
    required this.valence,
    required this.arousal,
    required this.source,
    this.actionUnits = const <String, double>{},
  });

  final String label;
  final String labelAr;
  final double confidence;
  final Map<String, double> probs;
  final double valence;
  final double arousal;
  final String source;
  final Map<String, double> actionUnits;

  Map<String, Object?> toMap() => <String, Object?>{
    'label': label,
    'label_ar': labelAr,
    'confidence': roundTo(confidence, 4),
    'probs': <String, double>{
      for (final entry in probs.entries) entry.key: roundTo(entry.value, 4),
    },
    'valence': roundTo(valence, 4),
    'arousal': roundTo(arousal, 4),
    'source': source,
    'action_units': <String, double>{
      for (final entry in actionUnits.entries)
        entry.key: roundTo(entry.value, 4),
    },
  };
}

abstract class EmotionModel {
  List<double> logits(FaceImage crop);

  List<String> get classOrder;

  int get inputSize;

  String get name;
}

class FaceImage {
  const FaceImage({
    required this.width,
    required this.height,
    required this.pixels,
    this.channels = 3,
  });

  final int width;
  final int height;
  final int channels;
  final List<int> pixels;

  static FaceImage blank(int size) => FaceImage(
    width: size,
    height: size,
    pixels: List<int>.filled(size * size * 3, 0),
  );
}

class FakeEmotionModel implements EmotionModel {
  FakeEmotionModel({
    List<double>? scores,
    this.classOrder = hse8,
    this.inputSize = 224,
  }) : _scores = scores ?? List<double>.filled(hse8.length, 0.0);

  List<double> _scores;

  @override
  final List<String> classOrder;

  @override
  final int inputSize;

  @override
  String get name => 'fake';

  int calls = 0;

  void setScores(List<double> scores) => _scores = scores;

  void setLabel(String label, {double strength = 6.0}) {
    final scores = List<double>.filled(classOrder.length, 0.0);
    final index = classOrder.indexOf(label);
    if (index >= 0) scores[index] = strength;
    _scores = scores;
  }

  @override
  List<double> logits(FaceImage crop) {
    calls += 1;
    return List<double>.from(_scores);
  }
}

class EmotionAnalyzer {
  EmotionAnalyzer({EmotionConfig? config, this.model})
    : config = config ?? const EmotionConfig(),
      history = ListQueue<EmotionSample>();

  final EmotionConfig config;
  final EmotionModel? model;

  final ListQueue<EmotionSample> history;

  List<double>? _smoothed;
  String? _stickyLabel;
  String? _challenger;
  int _challengerFrames = 0;

  List<double>? get smoothedProbs => _smoothed;

  String? get stickyLabel => _stickyLabel;

  set smoothedProbs(List<double>? value) => _smoothed = value;

  String get backend => model == null ? 'geometric' : 'onnx';

  String? get modelName => model?.name;

  EmotionResult analyse({
    FaceImage? crop,
    List<Point2>? landmarks,
    bool smooth = true,
    double quality = 1.0,
    List<double> Function(List<double>, Map<String, double>)? rebalance,
  }) {
    var aus = <String, double>{};
    List<double> probs;
    var source = 'geometric';

    final active = model;
    if (active != null && crop != null) {
      probs = probabilitiesFromLogits(active.logits(crop), active.classOrder);
      source = 'onnx';
      if (landmarks != null) aus = actionUnits(landmarks);
    } else {
      aus = landmarks != null ? actionUnits(landmarks) : <String, double>{};
      probs = inferGeometric(aus);
    }

    if (rebalance != null) {
      try {
        final adjusted = rebalance(probs, aus);
        final total = sumOf(adjusted);
        if (adjusted.length == probs.length &&
            adjusted.every((double v) => v.isFinite) &&
            total > 0) {
          probs = <double>[for (final value in adjusted) value / total];
          source += '+baseline';
        }
      } on Object {
        source = source;
      }
    }

    String label;
    if (smooth) {
      probs = smoothProbs(probs, quality: quality);
      label = stableLabel(probs);
    } else {
      label = emotions[argMax(probs)];
    }

    final confidence = probs[emotionIndex(label)];
    final va = valenceArousal(probs, aus);

    history.addLast(
      EmotionSample(label: label, valence: va.$1, arousal: va.$2),
    );
    while (history.length > config.historyLen) {
      history.removeFirst();
    }

    return EmotionResult(
      label: label,
      labelAr: emotionsAr[label]!,
      confidence: confidence,
      probs: <String, double>{
        for (var i = 0; i < emotions.length; i++) emotions[i]: probs[i],
      },
      valence: va.$1,
      arousal: va.$2,
      source: source,
      actionUnits: aus,
    );
  }

  static List<double> probabilitiesFromLogits(
    List<double> logits,
    List<String> order,
  ) {
    final raw = softmax(logits);

    final head = raw.length < order.length ? raw.length : order.length;
    var winnerIndex = 0;
    for (var i = 1; i < head; i++) {
      if (raw[i] > raw[winnerIndex]) winnerIndex = i;
    }
    var winner = order[winnerIndex];
    if (winner == 'contempt') winner = contemptTarget;

    final out = List<double>.filled(emotions.length, 0.0);
    for (var i = 0; i < order.length; i++) {
      if (i >= raw.length) break;
      final cls = order[i];
      final target = cls == 'contempt' ? contemptTarget : cls;
      final index = emotionIndex(target);
      if (index >= 0) out[index] += raw[i];
    }

    final total = sumOf(out);
    final normalised = total > 0
        ? <double>[for (final value in out) value / total]
        : List<double>.filled(emotions.length, 1.0 / emotions.length);

    final w = emotionIndex(winner);
    if (argMax(normalised) != w) {
      normalised[w] = maxOf(normalised) + 1e-6;
      final retotal = sumOf(normalised);
      return <double>[for (final value in normalised) value / retotal];
    }
    return normalised;
  }

  static Map<String, double> actionUnits(List<Point2>? landmarks) {
    if (landmarks == null || landmarks.length < 106) {
      return <String, double>{};
    }
    final p = landmarks;

    final leftEye = <Point2>[for (final i in leftEyeIndices) p[i]];
    final rightEye = <Point2>[for (final i in rightEyeIndices) p[i]];
    final leftBrow = <Point2>[for (final i in leftBrowIndices) p[i]];
    final rightBrow = <Point2>[for (final i in rightBrowIndices) p[i]];
    final mouthOuter = <Point2>[for (final i in mouthOuterIndices) p[i]];

    final lc = centroid(leftEye);
    final rc = centroid(rightEye);
    var iod = rc.distanceTo(lc);
    if (iod == 0.0) iod = 1.0;

    final eyeOpen =
        (verticalSpread(leftEye) + verticalSpread(rightEye)) / 2.0 / iod;
    final browY = (meanY(leftBrow) + meanY(rightBrow)) / 2.0;
    final eyeY = (lc.y + rc.y) / 2.0;
    final browRaise = (eyeY - browY) / iod;

    final mouthH = verticalSpread(mouthOuter) / iod;
    final mouthW = horizontalSpread(mouthOuter) / iod;

    final cornersY = (p[52].y + p[61].y) / 2.0;
    final lipsY = meanY(mouthOuter);
    final smile = (lipsY - cornersY) / iod;

    final knit = leftBrow.last.distanceTo(rightBrow.last) / iod;

    return <String, double>{
      'eye_open': eyeOpen,
      'brow_raise': browRaise,
      'brow_knit': knit,
      'mouth_open': mouthH,
      'mouth_width': mouthW,
      'smile': smile,
    };
  }

  static List<double> inferGeometric(Map<String, double> aus) {
    if (aus.isEmpty) {
      final probs = List<double>.filled(emotions.length, 0.0);
      probs[emotionIndex('neutral')] = 1.0;
      return probs;
    }

    final smile = aus['smile'] ?? 0.0;
    final mouthOpen = aus['mouth_open'] ?? 0.0;
    final browRaise = aus['brow_raise'] ?? 0.0;
    final browKnit = aus['brow_knit'] ?? 0.0;
    final eyeOpen = aus['eye_open'] ?? 0.0;

    double atLeastZero(double value) => value > 0.0 ? value : 0.0;

    final s = <String, double>{for (final e in emotions) e: 0.0};
    s['neutral'] = 1.0;
    s['happy'] = 6.0 * atLeastZero(smile - 0.015);
    s['sad'] = 5.0 * atLeastZero(-smile - 0.010) +
        1.5 * atLeastZero(0.30 - browRaise);
    s['surprise'] = 4.0 * atLeastZero(mouthOpen - 0.28) +
        3.0 * atLeastZero(browRaise - 0.42);
    s['fear'] = 3.0 * atLeastZero(eyeOpen - 0.34) +
        2.0 * atLeastZero(browRaise - 0.40);
    s['anger'] = 4.0 * atLeastZero(0.62 - browKnit) +
        2.0 * atLeastZero(0.26 - browRaise);
    s['disgust'] = 3.0 * atLeastZero(0.60 - browKnit) +
        2.0 * atLeastZero(-smile - 0.02);

    return softmax(<double>[for (final e in emotions) s[e]! * 2.2]);
  }

  List<double> smoothProbs(List<double> probs, {double quality = 1.0}) {
    final q = clampd(quality, 0.0, 1.0);
    final a = config.smoothingAlphaMin +
        (config.smoothingAlpha - config.smoothingAlphaMin) * q;
    final previous = _smoothed;
    if (previous == null) {
      _smoothed = List<double>.from(probs);
    } else {
      _smoothed = <double>[
        for (var i = 0; i < probs.length; i++)
          a * probs[i] + (1.0 - a) * previous[i],
      ];
    }
    final current = _smoothed!;
    final total = sumOf(current);
    return total > 0
        ? <double>[for (final value in current) value / total]
        : current;
  }

  String stableLabel(List<double> probs) {
    final top = emotions[argMax(probs)];

    if (_stickyLabel == null) {
      _stickyLabel = top;
      _challenger = null;
      _challengerFrames = 0;
      return top;
    }

    if (top == _stickyLabel) {
      _challenger = null;
      _challengerFrames = 0;
      return _stickyLabel!;
    }

    final lead =
        probs[emotionIndex(top)] - probs[emotionIndex(_stickyLabel!)];
    if (lead < config.switchMargin) return _stickyLabel!;

    if (top == _challenger) {
      _challengerFrames += 1;
    } else {
      _challenger = top;
      _challengerFrames = 1;
    }

    if (_challengerFrames >= config.switchFrames) {
      _stickyLabel = top;
      _challenger = null;
      _challengerFrames = 0;
    }
    return _stickyLabel!;
  }

  static (double, double) valenceArousal(
    List<double> probs,
    Map<String, double> aus,
  ) {
    var v = 0.0;
    var a = 0.0;
    for (var i = 0; i < emotions.length; i++) {
      v += probs[i] * emotionVa[emotions[i]]![0];
      a += probs[i] * emotionVa[emotions[i]]![1];
    }
    if (aus.isNotEmpty) {
      a += 0.18 * clampd((aus['mouth_open'] ?? 0.0) - 0.25, 0.0, 0.6);
      a += 0.12 * clampd((aus['eye_open'] ?? 0.0) - 0.30, 0.0, 0.4);
      v += 0.20 * clampd(aus['smile'] ?? 0.0, -0.10, 0.10) * 5.0;
    }
    return (clampd(v, -1.0, 1.0), clampd(a, 0.0, 1.0));
  }

  void reset() {
    _smoothed = null;
    _stickyLabel = null;
    _challenger = null;
    _challengerFrames = 0;
    history.clear();
  }

  Map<String, Object?> trend() {
    if (history.isEmpty) {
      return <String, Object?>{
        'dominant': null,
        'avg_valence': 0.0,
        'avg_arousal': 0.0,
        'samples': 0,
        'stability': 0.0,
      };
    }
    final labels = <String>[for (final sample in history) sample.label];
    final dominant = mostCommon(labels)!;
    return <String, Object?>{
      'dominant': dominant,
      'dominant_ar': emotionsAr[dominant],
      'avg_valence': mean(<double>[for (final s in history) s.valence]),
      'avg_arousal': mean(<double>[for (final s in history) s.arousal]),
      'samples': history.length,
      'stability': countOf(labels, dominant) / labels.length,
    };
  }
}

class EmotionSample {
  const EmotionSample({
    required this.label,
    required this.valence,
    required this.arousal,
  });

  final String label;
  final double valence;
  final double arousal;
}

List<double> preprocessRgb(FaceImage image, int size) {
  final out = List<double>.filled(size * size * 3, 0.0);
  final plane = size * size;
  for (var y = 0; y < size; y++) {
    final sy = (y * image.height / size).floor();
    for (var x = 0; x < size; x++) {
      final sx = (x * image.width / size).floor();
      final base = (sy * image.width + sx) * image.channels;
      for (var c = 0; c < 3; c++) {
        final raw = image.pixels[base + (2 - c)] / 255.0;
        out[c * plane + y * size + x] =
            (raw - imagenetMean[c]) / imagenetStd[c];
      }
    }
  }
  return out;
}

double laplacianVariance(List<int> grey, int width, int height) {
  if (width < 3 || height < 3) return 0.0;
  final values = <double>[];
  for (var y = 1; y < height - 1; y++) {
    for (var x = 1; x < width - 1; x++) {
      final centre = grey[y * width + x] * 4;
      final sum = grey[(y - 1) * width + x] +
          grey[(y + 1) * width + x] +
          grey[y * width + x - 1] +
          grey[y * width + x + 1];
      values.add((sum - centre).toDouble());
    }
  }
  final average = mean(values);
  var total = 0.0;
  for (final value in values) {
    final delta = value - average;
    total += delta * delta;
  }
  return total / values.length;
}

double degrees(double radians) => radians * 180.0 / math.pi;
