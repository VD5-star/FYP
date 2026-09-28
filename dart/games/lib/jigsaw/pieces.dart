import 'dart:math' as math;
import 'dart:typed_data';

import 'raster.dart';

const String styleSquare = 'square';
const String styleInterlock = 'interlock';

const double tabRatio = 0.22;
const int curveSteps = 22;
const int ss = 4;

const List<List<List<double>>> _tabPath = <List<List<double>>>[
  <List<double>>[
    <double>[0.00, 0.00],
    <double>[0.20, 0.00],
    <double>[0.33, 0.00],
    <double>[0.40, 0.00],
  ],
  <List<double>>[
    <double>[0.46, 0.00],
    <double>[0.28, 0.62],
    <double>[0.40, 0.78],
  ],
  <List<double>>[
    <double>[0.52, 0.94],
    <double>[0.48, 0.94],
    <double>[0.60, 0.78],
  ],
  <List<double>>[
    <double>[0.72, 0.62],
    <double>[0.54, 0.00],
    <double>[0.60, 0.00],
  ],
  <List<double>>[
    <double>[0.67, 0.00],
    <double>[0.80, 0.00],
    <double>[1.00, 0.00],
  ],
];

void _cubic(List<double> out, List<double> p0, List<double> c1,
    List<double> c2, List<double> p3, int steps) {
  for (int i = 0; i < steps; i++) {
    final double t = i / steps;
    final double u = 1.0 - t;
    final double a = u * u * u;
    final double b = 3 * u * u * t;
    final double c = 3 * u * t * t;
    final double d = t * t * t;
    out.add(a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0]);
    out.add(a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1]);
  }
}

Float64List tabProfile({int steps = curveSteps}) {
  final List<double> pts = <double>[];
  List<double> current = _tabPath[0][0];
  for (final List<List<double>> seg in _tabPath) {
    if (seg.length == 4) {
      _cubic(pts, seg[0], seg[1], seg[2], seg[3], steps);
      current = seg[3];
    } else {
      _cubic(pts, current, seg[0], seg[1], seg[2], steps);
      current = seg[2];
    }
  }
  pts.add(1.0);
  pts.add(0.0);
  return Float64List.fromList(pts);
}

class Cut {
  Cut({
    required this.rows,
    required this.cols,
    required this.width,
    required this.height,
    required this.style,
    required this.horizontal,
    required this.vertical,
    required this.amp,
  });

  final int rows;
  final int cols;
  final int width;
  final int height;
  final String style;
  final Int8List horizontal;
  final Int8List vertical;
  final double amp;

  double get cellW => width / cols;
  double get cellH => height / rows;

  int hAt(int r, int c) => horizontal[r * cols + c];
  int vAt(int r, int c) => vertical[r * (cols + 1) + c];
}

class _Lcg {
  _Lcg(int seed) : _s = (seed & 0x7FFFFFFF) | 1;
  int _s;
  int next() {
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF;
    return _s;
  }

  int sign() => (next() >> 8) % 2 == 0 ? -1 : 1;
  int range(int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (next() >> 8) % (hi - lo + 1);
  }

  double unit() => (next() >> 7) / 16777216.0;
}

Cut makeCut(int rows, int cols, int width, int height,
    {String style = styleInterlock, int? seed}) {
  final _Lcg rng = _Lcg(seed ?? 1);
  final Int8List horizontal = Int8List((rows + 1) * cols);
  final Int8List vertical = Int8List(rows * (cols + 1));
  if (style == styleInterlock) {
    for (int r = 1; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        horizontal[r * cols + c] = rng.sign();
      }
    }
    for (int r = 0; r < rows; r++) {
      for (int c = 1; c < cols; c++) {
        vertical[r * (cols + 1) + c] = rng.sign();
      }
    }
  }
  final double amp =
      math.min(width / cols, height / rows) * tabRatio;
  return Cut(
    rows: rows,
    cols: cols,
    width: width,
    height: height,
    style: style,
    horizontal: horizontal,
    vertical: vertical,
    amp: amp,
  );
}

List<double> hEdge(Cut cut, int r, int c, Float64List profile) {
  final double x0 = c * cut.cellW;
  final double x1 = (c + 1) * cut.cellW;
  final double y = r * cut.cellH;
  final int sign = cut.hAt(r, c);
  if (sign == 0) return <double>[x0, y, x1, y];
  final List<double> out = <double>[];
  for (int i = 0; i < profile.length; i += 2) {
    out.add(x0 + profile[i] * (x1 - x0));
    out.add(y + profile[i + 1] * cut.amp * sign);
  }
  return out;
}

List<double> vEdge(Cut cut, int r, int c, Float64List profile) {
  final double y0 = r * cut.cellH;
  final double y1 = (r + 1) * cut.cellH;
  final double x = c * cut.cellW;
  final int sign = cut.vAt(r, c);
  if (sign == 0) return <double>[x, y0, x, y1];
  final List<double> out = <double>[];
  for (int i = 0; i < profile.length; i += 2) {
    out.add(x + profile[i + 1] * cut.amp * sign);
    out.add(y0 + profile[i] * (y1 - y0));
  }
  return out;
}

List<double> _reverse(List<double> pts) {
  final List<double> out = <double>[];
  for (int i = pts.length - 2; i >= 0; i -= 2) {
    out.add(pts[i]);
    out.add(pts[i + 1]);
  }
  return out;
}

List<double> pieceOutline(Cut cut, int r, int c, [Float64List? profile]) {
  final Float64List p = profile ?? tabProfile();
  final List<double> top = hEdge(cut, r, c, p);
  final List<double> right = vEdge(cut, r, c + 1, p);
  final List<double> bottom = _reverse(hEdge(cut, r + 1, c, p));
  final List<double> left = _reverse(vEdge(cut, r, c, p));
  final List<double> out = <double>[];
  out.addAll(top);
  out.addAll(right.sublist(2));
  out.addAll(bottom.sublist(2));
  if (left.length > 4) out.addAll(left.sublist(2, left.length - 2));
  return out;
}

void fillPoly(Uint8List mask, int w, int h, List<double> poly, int value) {
  final int n = poly.length ~/ 2;
  if (n < 3) return;
  double minY = poly[1];
  double maxY = poly[1];
  for (int i = 1; i < n; i++) {
    final double y = poly[i * 2 + 1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  int y0 = minY.floor();
  int y1 = maxY.ceil();
  if (y0 < 0) y0 = 0;
  if (y1 > h - 1) y1 = h - 1;
  final List<double> xs = <double>[];
  for (int y = y0; y <= y1; y++) {
    xs.clear();
    final double cy = y + 0.5;
    for (int i = 0; i < n; i++) {
      final int j = (i + 1) % n;
      final double ay = poly[i * 2 + 1];
      final double by = poly[j * 2 + 1];
      if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
        final double ax = poly[i * 2];
        final double bx = poly[j * 2];
        xs.add(ax + (cy - ay) / (by - ay) * (bx - ax));
      }
    }
    if (xs.isEmpty) continue;
    xs.sort();
    for (int k = 0; k + 1 < xs.length; k += 2) {
      int sx = (xs[k] + 0.5).floor();
      int ex = (xs[k + 1] - 0.5).ceil();
      if (sx < 0) sx = 0;
      if (ex > w - 1) ex = w - 1;
      final int row = y * w;
      for (int x = sx; x <= ex; x++) {
        mask[row + x] = value;
      }
    }
  }
}

class Piece {
  Piece({
    required this.row,
    required this.col,
    required this.sprite,
    required this.alpha,
    required this.homeX,
    required this.homeY,
  })  : x = homeX,
        y = homeY;

  final int row;
  final int col;
  final Raster sprite;
  final Uint8List alpha;
  final int homeX;
  final int homeY;

  int x;
  int y;
  bool placed = false;

  int get w => sprite.w;
  int get h => sprite.h;

  bool hits(num px, num py) {
    final int lx = (px - x).floor();
    final int ly = (py - y).floor();
    if (lx < 0 || ly < 0 || lx >= w || ly >= h) return false;
    return alpha[ly * w + lx] > 96;
  }
}

List<Piece> cutImage(Raster image, Cut cut) {
  final int w = image.w;
  final int h = image.h;
  final Float64List profile = tabProfile();
  final int pad = cut.amp.ceil() + 2;
  final int canvasW = w + 2 * pad;
  final int canvasH = h + 2 * pad;
  final Raster padded = _replicateBorder(image, pad);

  final List<Piece> out = <Piece>[];
  for (int r = 0; r < cut.rows; r++) {
    for (int c = 0; c < cut.cols; c++) {
      final List<double> poly = pieceOutline(cut, r, c, profile);
      for (int i = 0; i < poly.length; i++) {
        poly[i] += pad;
      }
      double mnx = poly[0];
      double mny = poly[1];
      double mxx = poly[0];
      double mxy = poly[1];
      for (int i = 0; i < poly.length; i += 2) {
        final double px = poly[i].roundToDouble();
        final double py = poly[i + 1].roundToDouble();
        if (px < mnx) mnx = px;
        if (px > mxx) mxx = px;
        if (py < mny) mny = py;
        if (py > mxy) mxy = py;
      }
      final int x0 = math.max(0, mnx.toInt());
      final int y0 = math.max(0, mny.toInt());
      final int x1 = math.min(canvasW, mxx.toInt() + 1);
      final int y1 = math.min(canvasH, mxy.toInt() + 1);
      final int pw = x1 - x0;
      final int ph = y1 - y0;

      final int hw = pw * ss;
      final int hh = ph * ss;
      final Uint8List hi = Uint8List(hw * hh);
      final List<double> fine = List<double>.filled(poly.length, 0);
      for (int i = 0; i < poly.length; i += 2) {
        fine[i] = ((poly[i] - x0) * ss).roundToDouble();
        fine[i + 1] = ((poly[i + 1] - y0) * ss).roundToDouble();
      }
      fillPoly(hi, hw, hh, fine, 255);

      final Uint8List mask = Uint8List(pw * ph);
      for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
          int acc = 0;
          for (int sy = 0; sy < ss; sy++) {
            final int row = (y * ss + sy) * hw + x * ss;
            for (int sx = 0; sx < ss; sx++) {
              acc += hi[row + sx];
            }
          }
          mask[y * pw + x] = (acc / (ss * ss)).round();
        }
      }

      final Raster sprite = cropRaster(padded, x0, y0, pw, ph);
      out.add(Piece(
        row: r,
        col: c,
        sprite: sprite,
        alpha: mask,
        homeX: x0 - pad,
        homeY: y0 - pad,
      ));
    }
  }
  return out;
}

Raster _replicateBorder(Raster src, int pad) {
  final int w = src.w + pad * 2;
  final int h = src.h + pad * 2;
  final Uint8List out = Uint8List(w * h * 4);
  for (int y = 0; y < h; y++) {
    int sy = y - pad;
    if (sy < 0) sy = 0;
    if (sy > src.h - 1) sy = src.h - 1;
    for (int x = 0; x < w; x++) {
      int sx = x - pad;
      if (sx < 0) sx = 0;
      if (sx > src.w - 1) sx = src.w - 1;
      final int si = (sy * src.w + sx) * 4;
      final int di = (y * w + x) * 4;
      out[di] = src.rgba[si];
      out[di + 1] = src.rgba[si + 1];
      out[di + 2] = src.rgba[si + 2];
      out[di + 3] = 255;
    }
  }
  return Raster(w, h, out);
}

Int32List coverage(Cut cut) {
  final Float64List profile = tabProfile();
  final Int32List count = Int32List(cut.width * cut.height);
  final Uint8List layer = Uint8List(cut.width * cut.height);
  for (int r = 0; r < cut.rows; r++) {
    for (int c = 0; c < cut.cols; c++) {
      layer.fillRange(0, layer.length, 0);
      final List<double> poly = pieceOutline(cut, r, c, profile);
      for (int i = 0; i < poly.length; i++) {
        poly[i] = poly[i].roundToDouble();
      }
      fillPoly(layer, cut.width, cut.height, poly, 1);
      for (int i = 0; i < layer.length; i++) {
        count[i] += layer[i];
      }
    }
  }
  return count;
}
