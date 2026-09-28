import 'dart:math' as math;
import 'dart:typed_data';

class Raster {
  Raster(this.w, this.h, this.rgba);

  factory Raster.blank(int w, int h) => Raster(w, h, Uint8List(w * h * 4));

  final int w;
  final int h;
  final Uint8List rgba;
}

class Frame {
  Frame(this.w, this.h) : d = Float32List(w * h * 3);

  final int w;
  final int h;
  final Float32List d;

  void addAt(int x, int y, double r, double g, double b) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    final int i = (y * w + x) * 3;
    d[i] += r;
    d[i + 1] += g;
    d[i + 2] += b;
  }

  void setAt(int x, int y, double r, double g, double b) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    final int i = (y * w + x) * 3;
    d[i] = r;
    d[i + 1] = g;
    d[i + 2] = b;
  }

  Raster toRaster() {
    final Uint8List out = Uint8List(w * h * 4);
    for (int i = 0, o = 0; i < d.length; i += 3, o += 4) {
      out[o] = _clamp255(d[i]);
      out[o + 1] = _clamp255(d[i + 1]);
      out[o + 2] = _clamp255(d[i + 2]);
      out[o + 3] = 255;
    }
    return Raster(w, h, out);
  }
}

int _clamp255(double v) {
  if (v <= 0) return 0;
  if (v >= 255) return 255;
  return v.toInt();
}

Frame frameFromRaster(Raster r) {
  final Frame f = Frame(r.w, r.h);
  for (int i = 0, o = 0; o < f.d.length; i += 4, o += 3) {
    f.d[o] = r.rgba[i].toDouble();
    f.d[o + 1] = r.rgba[i + 1].toDouble();
    f.d[o + 2] = r.rgba[i + 2].toDouble();
  }
  return f;
}

Float32List resizePlane(
    Float32List src, int sw, int sh, int dw, int dh) {
  if (sw == dw && sh == dh) return src;
  final Float32List out = Float32List(dw * dh);
  final double fx = sw / dw;
  final double fy = sh / dh;
  for (int y = 0; y < dh; y++) {
    double sy = (y + 0.5) * fy - 0.5;
    if (sy < 0) sy = 0;
    if (sy > sh - 1) sy = (sh - 1).toDouble();
    final int y0 = sy.floor();
    final int y1 = math.min(y0 + 1, sh - 1);
    final double wy = sy - y0;
    final int r0 = y0 * sw;
    final int r1 = y1 * sw;
    for (int x = 0; x < dw; x++) {
      double sx = (x + 0.5) * fx - 0.5;
      if (sx < 0) sx = 0;
      if (sx > sw - 1) sx = (sw - 1).toDouble();
      final int x0 = sx.floor();
      final int x1 = math.min(x0 + 1, sw - 1);
      final double wx = sx - x0;
      final double a = src[r0 + x0];
      final double b = src[r0 + x1];
      final double c = src[r1 + x0];
      final double e = src[r1 + x1];
      out[y * dw + x] =
          (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + e * wx) * wy;
    }
  }
  return out;
}

Float32List areaPlane(Float32List src, int sw, int sh, int dw, int dh) {
  if (sw == dw && sh == dh) return src;
  final Float32List out = Float32List(dw * dh);
  final double fx = sw / dw;
  final double fy = sh / dh;
  for (int y = 0; y < dh; y++) {
    final int y0 = (y * fy).floor();
    int y1 = ((y + 1) * fy).ceil();
    if (y1 <= y0) y1 = y0 + 1;
    if (y1 > sh) y1 = sh;
    for (int x = 0; x < dw; x++) {
      final int x0 = (x * fx).floor();
      int x1 = ((x + 1) * fx).ceil();
      if (x1 <= x0) x1 = x0 + 1;
      if (x1 > sw) x1 = sw;
      double acc = 0;
      int n = 0;
      for (int yy = y0; yy < y1; yy++) {
        final int row = yy * sw;
        for (int xx = x0; xx < x1; xx++) {
          acc += src[row + xx];
          n++;
        }
      }
      out[y * dw + x] = n == 0 ? 0 : acc / n;
    }
  }
  return out;
}

Float32List boxBlurPlane(
    Float32List src, int w, int h, int kw, int kh) {
  final Float32List mid = Float32List(w * h);
  final int hx = kw ~/ 2;
  for (int y = 0; y < h; y++) {
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      double acc = 0;
      int n = 0;
      for (int k = -hx; k <= hx; k++) {
        int sx = x + k;
        if (sx < 0) sx = 0;
        if (sx >= w) sx = w - 1;
        acc += src[row + sx];
        n++;
      }
      mid[row + x] = acc / n;
    }
  }
  final Float32List out = Float32List(w * h);
  final int hy = kh ~/ 2;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      double acc = 0;
      int n = 0;
      for (int k = -hy; k <= hy; k++) {
        int sy = y + k;
        if (sy < 0) sy = 0;
        if (sy >= h) sy = h - 1;
        acc += src[sy * w + x];
        n++;
      }
      out[y * w + x] = acc / n;
    }
  }
  return out;
}

Float32List gradientY(Float32List src, int w, int h) {
  final Float32List out = Float32List(w * h);
  for (int y = 0; y < h; y++) {
    final int up = y == 0 ? 0 : y - 1;
    final int dn = y == h - 1 ? h - 1 : y + 1;
    final double div = (dn - up) == 0 ? 1.0 : (dn - up).toDouble();
    for (int x = 0; x < w; x++) {
      out[y * w + x] = (src[dn * w + x] - src[up * w + x]) / div;
    }
  }
  return out;
}

void blurFrame(Frame f, double sigma) {
  final int r = math.max(1, (sigma * 1.6).round());
  final int k = r * 2 + 1;
  for (int c = 0; c < 3; c++) {
    Float32List plane = Float32List(f.w * f.h);
    for (int i = 0; i < plane.length; i++) {
      plane[i] = f.d[i * 3 + c];
    }
    for (int pass = 0; pass < 3; pass++) {
      plane = boxBlurPlane(plane, f.w, f.h, k, k);
    }
    for (int i = 0; i < plane.length; i++) {
      f.d[i * 3 + c] = plane[i];
    }
  }
}

void fillRectF(Frame f, int x0, int y0, int x1, int y1, List<double> col) {
  final int ax = math.max(0, math.min(x0, x1));
  final int ay = math.max(0, math.min(y0, y1));
  final int bx = math.min(f.w - 1, math.max(x0, x1));
  final int by = math.min(f.h - 1, math.max(y0, y1));
  for (int y = ay; y <= by; y++) {
    for (int x = ax; x <= bx; x++) {
      f.setAt(x, y, col[0], col[1], col[2]);
    }
  }
}

void strokeRectF(Frame f, int x0, int y0, int x1, int y1, List<double> col) {
  for (int x = x0; x <= x1; x++) {
    f.setAt(x, y0, col[0], col[1], col[2]);
    f.setAt(x, y1, col[0], col[1], col[2]);
  }
  for (int y = y0; y <= y1; y++) {
    f.setAt(x0, y, col[0], col[1], col[2]);
    f.setAt(x1, y, col[0], col[1], col[2]);
  }
}

void fillCircleF(Frame f, int cx, int cy, double r, List<double> col,
    {int thickness = -1}) {
  final int rr = r.ceil() + (thickness > 0 ? thickness : 0);
  final double inner = thickness > 0 ? r - thickness : -1.0;
  for (int y = cy - rr; y <= cy + rr; y++) {
    if (y < 0 || y >= f.h) continue;
    for (int x = cx - rr; x <= cx + rr; x++) {
      if (x < 0 || x >= f.w) continue;
      final double dx = (x - cx).toDouble();
      final double dy = (y - cy).toDouble();
      final double dd = math.sqrt(dx * dx + dy * dy);
      if (dd > r + 0.5) continue;
      if (thickness > 0 && dd < inner) continue;
      f.setAt(x, y, col[0], col[1], col[2]);
    }
  }
}

void fillEllipseF(Frame f, int cx, int cy, double ax, double ay,
    double angleDeg, List<double> col,
    {int thickness = -1}) {
  final double a = angleDeg * math.pi / 180.0;
  final double ca = math.cos(a);
  final double sa = math.sin(a);
  final double rad = math.max(ax, ay) + (thickness > 0 ? thickness : 0) + 1;
  final double band =
      thickness > 0 ? thickness / math.max(2.0, math.min(ax, ay) * 2) : 0.0;
  final int r = rad.ceil();
  for (int y = cy - r; y <= cy + r; y++) {
    if (y < 0 || y >= f.h) continue;
    for (int x = cx - r; x <= cx + r; x++) {
      if (x < 0 || x >= f.w) continue;
      final double dx = (x - cx).toDouble();
      final double dy = (y - cy).toDouble();
      final double u = (dx * ca + dy * sa) / math.max(0.5, ax);
      final double v = (-dx * sa + dy * ca) / math.max(0.5, ay);
      final double dd = math.sqrt(u * u + v * v);
      if (thickness > 0) {
        if ((dd - 1.0).abs() > band) continue;
      } else if (dd > 1.0) {
        continue;
      }
      f.setAt(x, y, col[0], col[1], col[2]);
    }
  }
}

void lineF(Frame f, int x0, int y0, int x1, int y1, List<double> col) {
  int dx = (x1 - x0).abs();
  int dy = -(y1 - y0).abs();
  int sx = x0 < x1 ? 1 : -1;
  int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  int x = x0;
  int y = y0;
  int guard = 0;
  while (guard++ < 1 << 16) {
    f.setAt(x, y, col[0], col[1], col[2]);
    if (x == x1 && y == y1) break;
    final int e2 = err * 2;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
}

Raster resizeRaster(Raster src, int dw, int dh) {
  if (src.w == dw && src.h == dh) return src;
  final bool shrink = dw < src.w || dh < src.h;
  final Uint8List out = Uint8List(dw * dh * 4);
  if (shrink) {
    final double fx = src.w / dw;
    final double fy = src.h / dh;
    for (int y = 0; y < dh; y++) {
      final int y0 = (y * fy).floor();
      int y1 = ((y + 1) * fy).ceil();
      if (y1 <= y0) y1 = y0 + 1;
      if (y1 > src.h) y1 = src.h;
      for (int x = 0; x < dw; x++) {
        final int x0 = (x * fx).floor();
        int x1 = ((x + 1) * fx).ceil();
        if (x1 <= x0) x1 = x0 + 1;
        if (x1 > src.w) x1 = src.w;
        double r = 0;
        double g = 0;
        double b = 0;
        int n = 0;
        for (int yy = y0; yy < y1; yy++) {
          int i = (yy * src.w + x0) * 4;
          for (int xx = x0; xx < x1; xx++) {
            r += src.rgba[i];
            g += src.rgba[i + 1];
            b += src.rgba[i + 2];
            i += 4;
            n++;
          }
        }
        final int o = (y * dw + x) * 4;
        out[o] = (r / n).round();
        out[o + 1] = (g / n).round();
        out[o + 2] = (b / n).round();
        out[o + 3] = 255;
      }
    }
    return Raster(dw, dh, out);
  }
  final double fx = src.w / dw;
  final double fy = src.h / dh;
  for (int y = 0; y < dh; y++) {
    double sy = (y + 0.5) * fy - 0.5;
    if (sy < 0) sy = 0;
    if (sy > src.h - 1) sy = (src.h - 1).toDouble();
    final int y0 = sy.floor();
    final int y1 = math.min(y0 + 1, src.h - 1);
    final double wy = sy - y0;
    for (int x = 0; x < dw; x++) {
      double sx = (x + 0.5) * fx - 0.5;
      if (sx < 0) sx = 0;
      if (sx > src.w - 1) sx = (src.w - 1).toDouble();
      final int x0 = sx.floor();
      final int x1 = math.min(x0 + 1, src.w - 1);
      final double wx = sx - x0;
      final int ia = (y0 * src.w + x0) * 4;
      final int ib = (y0 * src.w + x1) * 4;
      final int ic = (y1 * src.w + x0) * 4;
      final int id = (y1 * src.w + x1) * 4;
      final int o = (y * dw + x) * 4;
      for (int c = 0; c < 3; c++) {
        final double top =
            src.rgba[ia + c] * (1 - wx) + src.rgba[ib + c] * wx;
        final double bot =
            src.rgba[ic + c] * (1 - wx) + src.rgba[id + c] * wx;
        out[o + c] = (top * (1 - wy) + bot * wy).round();
      }
      out[o + 3] = 255;
    }
  }
  return Raster(dw, dh, out);
}

Raster cropRaster(Raster src, int x, int y, int w, int h) {
  final Uint8List out = Uint8List(w * h * 4);
  for (int r = 0; r < h; r++) {
    final int si = ((y + r) * src.w + x) * 4;
    out.setRange(r * w * 4, (r + 1) * w * 4, src.rgba, si);
  }
  return Raster(w, h, out);
}

Raster makeThumb(Raster src, int w, int h) {
  final double s = math.max(w / src.w, h / src.h);
  final int nw = math.max(w, (src.w * s + 0.5).toInt());
  final int nh = math.max(h, (src.h * s + 0.5).toInt());
  final Raster r = resizeRaster(src, nw, nh);
  return cropRaster(r, (nw - w) ~/ 2, (nh - h) ~/ 2, w, h);
}
