import 'dart:math' as math;
import 'dart:typed_data';

class Points {
  Points(this.count)
      : xs = Float64List(count),
        ys = Float64List(count);

  Points.of(Float64List x, Float64List y)
      : xs = x,
        ys = y,
        count = x.length;

  final int count;
  final Float64List xs;
  final Float64List ys;

  static Points nan(int n) {
    final Points p = Points(n);
    for (int i = 0; i < n; i++) {
      p.xs[i] = double.nan;
      p.ys[i] = double.nan;
    }
    return p;
  }

  static Points fromPairs(List<List<double>> rows) {
    final Points p = Points(rows.length);
    for (int i = 0; i < rows.length; i++) {
      p.xs[i] = rows[i][0];
      p.ys[i] = rows[i][1];
    }
    return p;
  }

  double x(int i) => xs[i];

  double y(int i) => ys[i];

  void set(int i, double px, double py) {
    xs[i] = px;
    ys[i] = py;
  }

  bool finite(int i) => xs[i].isFinite && ys[i].isFinite;

  Points copy() {
    final Points p = Points(count);
    p.xs.setAll(0, xs);
    p.ys.setAll(0, ys);
    return p;
  }

  double distanceTo(int i, double px, double py) =>
      hypot(xs[i] - px, ys[i] - py);

  double gap(int a, int b) => hypot(xs[a] - xs[b], ys[a] - ys[b]);
}

double hypot(double dx, double dy) => math.sqrt(dx * dx + dy * dy);

double median(List<double> values) {
  if (values.isEmpty) return double.nan;
  final List<double> s = List<double>.from(values)..sort();
  final int n = s.length;
  final int mid = n ~/ 2;
  if (n.isOdd) return s[mid];
  return (s[mid - 1] + s[mid]) / 2.0;
}

List<double> arange(double start, double stop, double step) {
  final List<double> out = <double>[];
  if (step <= 0) return out;
  final int n = math.max(0, ((stop - start) / step).ceil());
  for (int i = 0; i < n; i++) {
    out.add(start + i * step);
  }
  return out;
}
