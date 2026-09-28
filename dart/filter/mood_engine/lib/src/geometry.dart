import 'dart:math' as math;

class Point2 {
  const Point2(this.x, this.y);

  final double x;
  final double y;

  double distanceTo(Point2 other) {
    final dx = x - other.x;
    final dy = y - other.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  Point2 operator -(Point2 other) => Point2(x - other.x, y - other.y);

  Point2 operator +(Point2 other) => Point2(x + other.x, y + other.y);

  Point2 operator *(double factor) => Point2(x * factor, y * factor);

  @override
  String toString() => 'Point2($x, $y)';
}

const List<int> leftEyeIndices = <int>[35, 36, 37, 38, 39, 40, 41, 42];
const List<int> rightEyeIndices = <int>[89, 90, 91, 92, 93, 94, 95, 96];
const List<int> leftBrowIndices = <int>[43, 44, 45, 46, 47];
const List<int> rightBrowIndices = <int>[97, 98, 99, 100, 101];
const List<int> mouthOuterIndices = <int>[
  52,
  55,
  56,
  53,
  59,
  58,
  61,
  68,
  67,
  71,
  63,
  64,
];

const Map<String, int> landmarkIndex106 = <String, int>{
  'nose': 86,
  'chin': 0,
  'eye_l': 35,
  'eye_r': 93,
  'mouth_l': 52,
  'mouth_r': 61,
};

Point2 centroid(List<Point2> points) {
  var sx = 0.0;
  var sy = 0.0;
  for (final point in points) {
    sx += point.x;
    sy += point.y;
  }
  return Point2(sx / points.length, sy / points.length);
}

double verticalSpread(List<Point2> points) {
  var low = points.first.y;
  var high = points.first.y;
  for (final point in points) {
    if (point.y < low) low = point.y;
    if (point.y > high) high = point.y;
  }
  return high - low;
}

double horizontalSpread(List<Point2> points) {
  var low = points.first.x;
  var high = points.first.x;
  for (final point in points) {
    if (point.x < low) low = point.x;
    if (point.x > high) high = point.x;
  }
  return high - low;
}

double meanY(List<Point2> points) {
  var total = 0.0;
  for (final point in points) {
    total += point.y;
  }
  return total / points.length;
}

class BoundingBox {
  const BoundingBox(this.x1, this.y1, this.x2, this.y2);

  final double x1;
  final double y1;
  final double x2;
  final double y2;

  double get width => x2 - x1;

  double get height => y2 - y1;

  double get area {
    final w = width > 0.0 ? width : 0.0;
    final h = height > 0.0 ? height : 0.0;
    return w * h;
  }

  Point2 get centre => Point2((x1 + x2) / 2.0, (y1 + y2) / 2.0);

  BoundingBox clampTo(double width, double height) => BoundingBox(
    _clamp(x1, 0.0, width - 1.0),
    _clamp(y1, 0.0, height - 1.0),
    _clamp(x2, 0.0, width - 1.0),
    _clamp(y2, 0.0, height - 1.0),
  );

  BoundingBox scaled(double factor) =>
      BoundingBox(x1 * factor, y1 * factor, x2 * factor, y2 * factor);

  BoundingBox shifted(double dx, double dy) =>
      BoundingBox(x1 + dx, y1 + dy, x2 + dx, y2 + dy);

  List<double> toList() => <double>[x1, y1, x2, y2];

  static double _clamp(double value, double low, double high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
  }

  @override
  String toString() => 'BoundingBox($x1, $y1, $x2, $y2)';
}
