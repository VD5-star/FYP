import 'dart:math' as math;

import 'body_frame.dart';
import 'body_landmark.dart';

const double targetSubjectHeight = 500.0;

const double maxUpscale = 8.0;

const List<(double, double)> searchAttempts = <(double, double)>[
  (0.7, 1.0),
  (0.7, 0.55),
  (0.7, 0.3),
  (2.0, 1.0),
];

const Duration maxPrediction = Duration(milliseconds: 500);

const Duration eagerSearchWindow = Duration(seconds: 1);

const Duration searchInterval = Duration(milliseconds: 250);

const Duration maxSearchInterval = Duration(seconds: 1);

class SearchRegion {
  const SearchRegion({
    required this.x0,
    required this.y0,
    required this.x1,
    required this.y1,
    required this.scale,
  });

  final int x0;
  final int y0;
  final int x1;
  final int y1;

  final double scale;

  int get width => x1 - x0;
  int get height => y1 - y0;

  bool get isValid => width >= 16 && height >= 16;

  double get centreX => (x0 + x1) / 2;
  double get centreY => (y0 + y1) / 2;

  ({double x, double y}) toFrame(double x, double y) =>
      (x: x0 + x / scale, y: y0 + y / scale);

  @override
  String toString() =>
      'SearchRegion($x0, $y0, $x1, $y1, scale ${scale.toStringAsFixed(2)})';
}

class SubjectLocator {
  SubjectLocator({this.confidenceThreshold = 0.5});

  final double confidenceThreshold;

  double? _centreX;
  double? _centreY;
  double? _size;
  double _velocityX = 0;
  double _velocityY = 0;
  Duration? _seenAt;
  Duration? _lastSearch;
  Duration _interval = searchInterval;

  bool get hasSubject =>
      _centreX != null && _centreY != null && _size != null;

  Duration get interval => _interval;

  void seen(BodyFrame frame, Duration timestamp) {
    final good = <BodyLandmark>[
      for (final l in frame.landmarks)
        if (l.likelihood >= confidenceThreshold &&
            l.x.isFinite &&
            l.y.isFinite)
          l,
    ];
    if (good.length < 4) return;

    var lowX = good.first.x;
    var highX = lowX;
    var lowY = good.first.y;
    var highY = lowY;
    for (final l in good) {
      if (l.x < lowX) lowX = l.x;
      if (l.x > highX) highX = l.x;
      if (l.y < lowY) lowY = l.y;
      if (l.y > highY) highY = l.y;
    }

    final centreX = (lowX + highX) / 2;
    final centreY = (lowY + highY) / 2;
    final size = math.max(highX - lowX, highY - lowY);
    if (!centreX.isFinite || !centreY.isFinite) return;
    if (!size.isFinite || size <= 0) return;

    final previousX = _centreX;
    final previousY = _centreY;
    final previousTime = _seenAt;
    if (previousX != null && previousY != null && previousTime != null) {
      final dt = (timestamp - previousTime).inMicroseconds / 1e6;
      if (dt > 1e-3 && dt < 0.5) {
        _velocityX = 0.5 * _velocityX + 0.5 * (centreX - previousX) / dt;
        _velocityY = 0.5 * _velocityY + 0.5 * (centreY - previousY) / dt;
      }
    }

    _centreX = centreX;
    _centreY = centreY;
    _size = size;
    _seenAt = timestamp;
    _lastSearch = null;
    _interval = searchInterval;
  }

  bool shouldSearch(Duration timestamp) {
    final seenAt = _seenAt;
    if (!hasSubject || seenAt == null) return false;

    final missing = timestamp - seenAt;
    if (missing <= eagerSearchWindow) return true;

    final lastSearch = _lastSearch;
    if (lastSearch == null) return true;
    return timestamp - lastSearch >= _interval;
  }

  void searched(Duration timestamp) {
    _lastSearch = timestamp;
    final seenAt = _seenAt;
    if (seenAt == null || timestamp - seenAt <= eagerSearchWindow) {
      _interval = searchInterval;
      return;
    }
    final doubled = _interval * 2;
    _interval = doubled > maxSearchInterval ? maxSearchInterval : doubled;
  }

  int get attempts => searchAttempts.length;

  SearchRegion? region(
    int attempt,
    int frameWidth,
    int frameHeight,
    Duration timestamp,
  ) {
    final centreX = _centreX;
    final centreY = _centreY;
    final size = _size;
    if (centreX == null || centreY == null || size == null) return null;
    if (attempt < 0 || attempt >= searchAttempts.length) return null;

    final (padding, sizeFactor) = searchAttempts[attempt];
    final assumedSize = size * sizeFactor * frameHeight;

    final seenAt = _seenAt;
    var elapsed = seenAt == null
        ? 0.0
        : math.max(0.0, (timestamp - seenAt).inMicroseconds / 1e6);
    final horizon = math.min(elapsed, maxPrediction.inMicroseconds / 1e6);

    final predictedX = (centreX + _velocityX * horizon) * frameWidth;
    final predictedY = (centreY + _velocityY * horizon) * frameHeight;

    final half = assumedSize * (0.5 + padding);
    final x0 = math.max(0, (predictedX - half).floor());
    final y0 = math.max(0, (predictedY - half).floor());
    final x1 = math.min(frameWidth, (predictedX + half).ceil());
    final y1 = math.min(frameHeight, (predictedY + half).ceil());

    if (x1 - x0 < 16 || y1 - y0 < 16) return null;

    var scale = targetSubjectHeight / math.max(assumedSize, 1.0);
    scale = scale.clamp(1.0, maxUpscale);

    if (x1 - x0 >= frameWidth * 0.9 &&
        y1 - y0 >= frameHeight * 0.9 &&
        scale <= 1.01) {
      return null;
    }

    return SearchRegion(x0: x0, y0: y0, x1: x1, y1: y1, scale: scale);
  }

  void reset() {
    _centreX = null;
    _centreY = null;
    _size = null;
    _velocityX = 0;
    _velocityY = 0;
    _seenAt = null;
    _lastSearch = null;
    _interval = searchInterval;
  }
}
