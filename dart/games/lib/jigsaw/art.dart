import 'dart:math' as math;
import 'dart:typed_data';

import '../core/synth.dart';
import 'raster.dart';

const int artWidth = 1200;
const int artHeight = 800;
const int fbmWork = 460;
const double minTileStd = 7.0;
const int artMax = 760;

double smoothEdge(double a, double b, double x) {
  final double d = b - a;
  double t = d == 0 ? 0.0 : (x - a) / d;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  return t * t * (3 - 2 * t);
}

Float32List noisePlane(int w, int h, int seed, int cx, int cy) {
  final int gx = math.max(2, cx);
  final int gy = math.max(2, cy);
  final Rng rng = Rng(seed * 2654435761 + 1);
  final Float32List g = Float32List((gy + 1) * (gx + 1));
  for (int i = 0; i < g.length; i++) {
    g[i] = rng.nextDouble();
  }
  final Float32List out = Float32List(w * h);
  final Float32List fxs = Float32List(w);
  final Int32List xi = Int32List(w);
  for (int x = 0; x < w; x++) {
    final double s = x * gx / w;
    final int i0 = s.floor();
    double f = s - i0;
    f = f * f * (3 - 2 * f);
    xi[x] = i0;
    fxs[x] = f;
  }
  for (int y = 0; y < h; y++) {
    final double s = y * gy / h;
    final int j0 = s.floor();
    double fy = s - j0;
    fy = fy * fy * (3 - 2 * fy);
    final int r0 = j0 * (gx + 1);
    final int r1 = (j0 + 1) * (gx + 1);
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      final int i0 = xi[x];
      final double fx = fxs[x];
      final double a = g[r0 + i0];
      final double b = g[r0 + i0 + 1];
      final double c = g[r1 + i0];
      final double d = g[r1 + i0 + 1];
      out[row + x] =
          (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    }
  }
  return out;
}

Float32List fbm(int w, int h, int seed,
    {int octaves = 5, double cx = 4.0, double stretch = 1.0}) {
  final double top = cx * math.pow(2, octaves - 1);
  final int need = math.min(w, math.max(64, (top * 3.0).toInt()));
  final int ww = w > fbmWork ? math.min(w, math.max(fbmWork, need)) : w;
  final int hh = math.max(2, (h * ww / w).round());

  Float32List out = Float32List(ww * hh);
  double amp = 1.0;
  double tot = 0.0;
  double c = cx;
  for (int i = 0; i < octaves; i++) {
    final int cy = math.max(2, (c * hh / ww / stretch).round());
    final Float32List n = noisePlane(ww, hh, seed + i * 101, c.toInt(), cy);
    for (int k = 0; k < out.length; k++) {
      out[k] += n[k] * amp;
    }
    tot += amp;
    amp *= 0.5;
    c *= 2;
  }
  for (int k = 0; k < out.length; k++) {
    out[k] /= tot;
  }
  if (hh != h || ww != w) out = resizePlane(out, ww, hh, w, h);
  return out;
}

Float32List fbm1(int w, int seed, int octaves, double cx) {
  final Float32List out = Float32List(w);
  double amp = 1.0;
  double tot = 0.0;
  double c = cx;
  for (int i = 0; i < octaves; i++) {
    final int cc = math.max(2, c.toInt());
    final Rng rng = Rng((seed + i * 71) * 2654435761 + 7);
    final Float32List g = Float32List(cc + 1);
    for (int k = 0; k <= cc; k++) {
      g[k] = rng.nextDouble();
    }
    for (int x = 0; x < w; x++) {
      final double s = x * cc / w;
      final int x0 = s.floor();
      double f = s - x0;
      f = f * f * (3 - 2 * f);
      out[x] += (g[x0] * (1 - f) + g[x0 + 1] * f) * amp;
    }
    tot += amp;
    amp *= 0.5;
    c *= 2;
  }
  for (int x = 0; x < w; x++) {
    out[x] /= tot;
  }
  return out;
}

void grain(Frame f, int seed, double amount) {
  final Rng rng = Rng(seed * 2654435761 + 13);
  for (int i = 0; i < f.d.length; i += 3) {
    final double v = (rng.nextDouble() - 0.5) * amount;
    f.d[i] += v;
    f.d[i + 1] += v;
    f.d[i + 2] += v;
  }
}

void detail(Frame f, int seed,
    {double amount = 17.0,
    double cx = 9.0,
    List<double> tint = const <double>[1.0, 1.0, 1.0]}) {
  final Float32List d = fbm(f.w, f.h, seed, octaves: 6, cx: cx);
  for (int i = 0, p = 0; i < d.length; i++, p += 3) {
    final double v = d[i] - 0.5;
    f.d[p] += v * tint[0] * amount;
    f.d[p + 1] += v * tint[1] * amount;
    f.d[p + 2] += v * tint[2] * amount;
  }
}

void specks(Frame f, int seed, int count, double lo, double hi,
    List<double> tint,
    {int rmin = 1, int rmax = 3}) {
  final Rng rng = Rng(seed * 2654435761 + 19);
  final int n =
      (count * (f.w * f.h) / (artWidth * artHeight)).toInt() + 20;
  for (int i = 0; i < n; i++) {
    final int x = (rng.nextDouble() * f.w).toInt();
    final int y = (rng.nextDouble() * f.h).toInt();
    final double m = lo + rng.nextDouble() * (hi - lo);
    final int r = rmin + (rng.nextDouble() * (rmax - rmin + 1)).toInt();
    final List<double> col = <double>[
      tint[0] * m,
      tint[1] * m,
      tint[2] * m,
    ];
    if (r <= 1) {
      f.addAt(x, y, col[0], col[1], col[2]);
    } else {
      _addCircle(f, x, y, r.toDouble(), col);
    }
  }
}

void _addCircle(Frame f, int cx, int cy, double r, List<double> col) {
  final int rr = r.ceil();
  for (int y = cy - rr; y <= cy + rr; y++) {
    for (int x = cx - rr; x <= cx + rr; x++) {
      final double dx = (x - cx).toDouble();
      final double dy = (y - cy).toDouble();
      if (dx * dx + dy * dy > r * r) continue;
      f.addAt(x, y, col[0], col[1], col[2]);
    }
  }
}

Frame dusk(int w, int h) {
  final Frame f = Frame(w, h);
  const double horizon = 0.60;
  const List<double> top = <double>[38, 44, 92];
  const List<double> low = <double>[244, 158, 92];
  const double sx = 0.68;
  const double sy = 0.545;

  final Float32List d2 = Float32List(w * h);
  final Float32List lit = Float32List(w * h);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    double t = yy / horizon;
    if (t > 1) t = 1;
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : x / (w - 1);
      final double a = (xx - sx) * 1.9;
      final double b = (yy - sy) * 2.9;
      final double dd = a * a + b * b;
      d2[row + x] = dd;
      final double la = (xx - sx) * 1.5;
      final double lb = (yy - sy) * 2.0;
      lit[row + x] = math.exp(-(la * la + lb * lb) * 3.0);
      final double glow = math.exp(-dd * 9.0);
      final double disc = smoothEdge(0.0032, 0.0016, dd);
      f.setAt(
        x,
        y,
        top[0] * (1 - t) + low[0] * t + glow * 120 + disc * 90,
        top[1] * (1 - t) + low[1] * t + glow * 74 + disc * 70,
        top[2] * (1 - t) + low[2] * t + glow * 20 + disc * 40,
      );
    }
  }

  const List<List<double>> layers = <List<double>>[
    <double>[5.0, 3.4, 0.50, 0.85],
    <double>[11.0, 5.0, 0.55, 0.55],
    <double>[21.0, 7.0, 0.58, 0.38],
  ];
  for (int i = 0; i < layers.length; i++) {
    final double cs = layers[i][0];
    final double st = layers[i][1];
    final double thr = layers[i][2];
    final double op = layers[i][3];
    final Float32List cloud =
        fbm(w, h, 21 + i * 37, octaves: 5, cx: cs, stretch: st);
    for (int y = 0; y < h; y++) {
      final double yy = h <= 1 ? 0.0 : y / (h - 1);
      final double fade = smoothEdge(0.70, 0.30, yy);
      final int row = y * w;
      for (int x = 0; x < w; x++) {
        final double band =
            smoothEdge(thr, thr + 0.24, cloud[row + x]) * fade;
        final double a = band * op;
        if (a <= 0) continue;
        final double l = lit[row + x];
        final double c0 = 88 + i * 18 + l * 150;
        final double c1 = 64 + i * 10 + l * 90;
        final double c2 = 78 + i * 6 + l * 30;
        final int p = (row + x) * 3;
        f.d[p] = f.d[p] * (1 - a) + c0 * a;
        f.d[p + 1] = f.d[p + 1] * (1 - a) + c1 * a;
        f.d[p + 2] = f.d[p + 2] * (1 - a) + c2 * a;
      }
    }
  }

  final Float32List streak =
      fbm(w, h, 900, octaves: 5, cx: 40.0, stretch: 9.0);
  final Float32List fine = fbm(w, h, 910, octaves: 6, cx: 12.0);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    final double s = smoothEdge(0.72, 0.18, yy);
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      final double a = (streak[row + x] - 0.5) * s;
      final double b = fine[row + x] - 0.5;
      final int p = (row + x) * 3;
      f.d[p] += a * 34 + b * 20;
      f.d[p + 1] += a * 24 + b * 16;
      f.d[p + 2] += a * 20 + b * 18;
    }
  }

  final List<List<Object>> ridges = <List<Object>>[
    <Object>[0.615, fbm1(w, 5, 4, 3.0), 0.050, <double>[96, 92, 124]],
    <Object>[0.685, fbm1(w, 9, 4, 5.0), 0.070, <double>[62, 58, 88]],
    <Object>[0.790, fbm1(w, 13, 5, 7.0), 0.095, <double>[32, 30, 50]],
  ];
  final Float32List rock = fbm(w, h, 44, octaves: 6, cx: 14.0);
  for (final List<Object> rg in ridges) {
    final double base = rg[0] as double;
    final Float32List prof = rg[1] as Float32List;
    final double amp = rg[2] as double;
    final List<double> col = rg[3] as List<double>;
    for (int y = 0; y < h; y++) {
      final double yy = h <= 1 ? 0.0 : y / (h - 1);
      final int row = y * w;
      for (int x = 0; x < w; x++) {
        final double line = base + (prof[x] - 0.5) * amp;
        if (yy < line) continue;
        double depth = (yy - line) * 5.0;
        if (depth > 1) depth = 1;
        final double k = 1.0 - depth * 0.35;
        final double n = (rock[row + x] - 0.5) * 46.0;
        final int p = (row + x) * 3;
        f.d[p] = col[0] * k + n;
        f.d[p + 1] = col[1] * k + n;
        f.d[p + 2] = col[2] * k + n;
      }
    }
  }

  detail(f, 88,
      amount: 15.0, cx: 10.0, tint: <double>[1.0, 0.92, 0.86]);
  specks(f, 66, 150, 0.30, 0.95, <double>[150, 130, 160], rmax: 2);
  grain(f, 77, 11.0);
  return f;
}

Frame water(int w, int h) {
  final Frame f = Frame(w, h);
  const List<double> top = <double>[16, 62, 84];
  const List<double> bot = <double>[44, 128, 140];
  final Float32List warp = fbm(w, h, 33, octaves: 4, cx: 6.0);
  final Float32List glit = fbm(w, h, 51, octaves: 6, cx: 22.0);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    final int row = y * w;
    final double sinY = math.sin(yy * 14.0) * 2.4;
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : x / (w - 1);
      final int i = row + x;
      double r = top[0] * (1 - yy) + bot[0] * yy;
      double g = top[1] * (1 - yy) + bot[1] * yy;
      double b = top[2] * (1 - yy) + bot[2] * yy;
      final double phase =
          (xx * 30.0 + sinY + (warp[i] - 0.5) * 5.0 + yy * 6.0) * math.pi;
      final double ripple = math.sin(phase) * 0.5 + 0.5;
      final double crest =
          smoothEdge(0.62, 0.99, ripple) * (0.35 + yy * 0.65);
      r += crest * 120;
      g += crest * 150;
      b += crest * 130;
      final double trough = smoothEdge(0.42, 0.02, ripple);
      r -= trough * 12;
      g -= trough * 34;
      b -= trough * 38;
      final double pa = (xx - 0.52) * 3.1;
      final double path = math.exp(-(pa * pa) * 2.2);
      final double spark =
          smoothEdge(0.78, 0.94, glit[i]) * path * (0.25 + yy * 0.75);
      r += spark * 180;
      g += spark * 190;
      b += spark * 150;
      f.setAt(x, y, r, g, b);
    }
  }
  grain(f, 91, 9.0);
  return f;
}

Frame leaves(int w, int h) {
  final Frame f = Frame(w, h);
  final Float32List base = fbm(w, h, 7, octaves: 5, cx: 3.0);
  for (int i = 0, p = 0; i < base.length; i++, p += 3) {
    f.d[p] = 26 + base[i] * 34;
    f.d[p + 1] = 54 + base[i] * 58;
    f.d[p + 2] = 28 + base[i] * 26;
  }

  final Rng rng = Rng(19 * 2654435761 + 3);
  final int n = (210 * (w * h) / (artWidth * artHeight)).toInt() + 60;
  for (int i = 0; i < n; i++) {
    final int cx = ((-0.05 + rng.nextDouble() * 1.10) * w).toInt();
    final int cy = ((-0.05 + rng.nextDouble() * 1.10) * h).toInt();
    final double ln = (0.045 + rng.nextDouble() * 0.070) * w;
    final double ax = math.max(3.0, ln);
    final double ay =
        math.max(2.0, ln * (0.34 + rng.nextDouble() * 0.18));
    final double ang = rng.nextDouble() * 180.0;
    final double lift = rng.nextDouble();
    final double jit = -14 + rng.nextDouble() * 28;
    final List<double> col = <double>[
      44 + lift * 92 + jit,
      96 + lift * 96 + jit,
      38 + lift * 40 + jit,
    ];
    fillEllipseF(f, cx, cy, ax, ay, ang, col);
    final double r = ang * math.pi / 180.0;
    final double dx = math.cos(r) * ax;
    final double dy = math.sin(r) * ax;
    final List<double> vein = <double>[
      col[0] * 0.78,
      col[1] * 0.78,
      col[2] * 0.78,
    ];
    lineF(f, (cx - dx).toInt(), (cy - dy).toInt(), (cx + dx).toInt(),
        (cy + dy).toInt(), vein);
  }

  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    final double b = (yy - 0.22) * 2.0;
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : x / (w - 1);
      final double a = (xx - 0.32) * 2.0;
      final double sun = math.exp(-(a * a + b * b) * 2.4);
      f.addAt(x, y, sun * 70, sun * 74, sun * 30);
    }
  }
  detail(f, 27, amount: 22.0, cx: 14.0, tint: <double>[0.8, 1.0, 0.7]);
  specks(f, 31, 240, 0.30, 0.95, <double>[90, 130, 70], rmax: 2);
  grain(f, 23, 12.0);
  return f;
}

Frame sand(int w, int h) {
  final Frame f = Frame(w, h);
  final Float32List dune =
      fbm(w, h, 61, octaves: 4, cx: 3.0, stretch: 2.2);
  final Float32List slope = gradientY(dune, w, h);
  final Float32List grit = fbm(w, h, 84, octaves: 6, cx: 30.0);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    final double depth = smoothEdge(0.10, 0.95, yy);
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : x / (w - 1);
      final int i = row + x;
      final double k = 1 - yy * 0.28;
      double r = 226 * k;
      double g = 196 * k;
      double b = 148 * k;
      double sl = slope[i] * 260.0;
      if (sl > 60) sl = 60;
      if (sl < -60) sl = -60;
      r += sl;
      g += sl * 0.86;
      b += sl * 0.62;
      final double rip =
          math.sin((xx * 46.0 + dune[i] * 9.0 + yy * 3.0) * math.pi) *
              depth *
              13.0;
      r += rip;
      g += rip * 0.88;
      b += rip * 0.70;
      final double sh = smoothEdge(0.45, 0.85, dune[i]);
      r -= sh * 26;
      g -= sh * 30;
      b -= sh * 34;
      final double gr = grit[i] - 0.5;
      r += gr * 40;
      g += gr * 34;
      b += gr * 26;
      f.setAt(x, y, r, g, b);
    }
  }

  final Rng rng = Rng(87 * 2654435761 + 5);
  final int n = (90 * (w * h) / (artWidth * artHeight)).toInt() + 30;
  for (int i = 0; i < n; i++) {
    final int cx = (rng.nextDouble() * w).toInt();
    final int cy = (rng.nextDouble() * h).toInt();
    final int r = ((0.004 + rng.nextDouble() * 0.012) * w).toInt();
    final double tone = -46 + rng.nextDouble() * 80;
    fillCircleF(f, cx, cy, math.max(1, r).toDouble(),
        <double>[tone, tone * 0.9, tone * 0.74]);
    fillEllipseF(
      f,
      cx,
      cy + math.max(1, r ~/ 2),
      math.max(2, (r * 1.5).toInt()).toDouble(),
      math.max(1, (r * 0.5).toInt()).toDouble(),
      rng.nextDouble() * 180.0,
      const <double>[-18.0, -16.0, -12.0],
    );
  }

  detail(f, 85, amount: 16.0, cx: 13.0, tint: <double>[1.0, 0.90, 0.72]);
  specks(f, 86, 260, 0.35, 1.0, <double>[60, 48, 34], rmax: 2);
  grain(f, 83, 20.0);
  return f;
}

Frame stones(int w, int h) {
  final Frame f = Frame(w, h);
  final Float32List bed = fbm(w, h, 29, octaves: 4, cx: 6.0);
  for (int i = 0, p = 0; i < bed.length; i++, p += 3) {
    f.d[p] = 74 + bed[i] * 30;
    f.d[p + 1] = 70 + bed[i] * 28;
    f.d[p + 2] = 66 + bed[i] * 26;
  }

  final Rng rng = Rng(43 * 2654435761 + 11);
  final List<List<double>> spots = <List<double>>[];
  int tries = 0;
  final int want = (120 * (w * h) / (artWidth * artHeight)).toInt() + 40;
  while (spots.length < want && tries < want * 40) {
    tries++;
    final double r = (0.028 + rng.nextDouble() * 0.044) * w;
    final double cx = (-0.02 + rng.nextDouble() * 1.04) * w;
    final double cy = (-0.02 + rng.nextDouble() * 1.04) * h;
    bool clash = false;
    for (final List<double> o in spots) {
      final double dx = cx - o[0];
      final double dy = cy - o[1];
      if (dx * dx + dy * dy < (r + o[2]) * (r + o[2]) * 0.62) {
        clash = true;
        break;
      }
    }
    if (!clash) spots.add(<double>[cx, cy, r]);
  }

  for (final List<double> s in spots) {
    final double cx = s[0];
    final double cy = s[1];
    final double r = s[2];
    final double ang = rng.nextDouble() * 180.0;
    final double ax = math.max(3.0, r);
    final double ay =
        math.max(3.0, r * (0.68 + rng.nextDouble() * 0.26));
    final double tone = 58 + rng.nextDouble() * 110;
    final List<double> tint = <double>[
      tone,
      tone * (0.94 + rng.nextDouble() * 0.09),
      tone * (0.90 + rng.nextDouble() * 0.12),
    ];
    fillEllipseF(f, cx.toInt(), cy.toInt(), ax, ay, ang, tint);
    fillEllipseF(f, cx.toInt(), cy.toInt(), ax, ay, ang,
        <double>[tint[0] * 0.55, tint[1] * 0.55, tint[2] * 0.55],
        thickness: 2);
    fillEllipseF(
      f,
      (cx - ax * 0.30).toInt(),
      (cy - ay * 0.34).toInt(),
      math.max(2, (ax * 0.46).toInt()).toDouble(),
      math.max(2, (ay * 0.40).toInt()).toDouble(),
      ang,
      <double>[
        math.min(255.0, tint[0] * 1.28),
        math.min(255.0, tint[1] * 1.28),
        math.min(255.0, tint[2] * 1.28),
      ],
    );
  }

  blurFrame(f, math.max(0.6, w / 900.0));
  grain(f, 37, 10.0);
  return f;
}

Frame petals(int w, int h) {
  final Frame f = Frame(w, h);
  final Float32List base = fbm(w, h, 15, octaves: 5, cx: 4.0);
  for (int i = 0, p = 0; i < base.length; i++, p += 3) {
    f.d[p] = 96 + base[i] * 48;
    f.d[p + 1] = 112 + base[i] * 52;
    f.d[p + 2] = 86 + base[i] * 40;
  }

  final Rng rng = Rng(67 * 2654435761 + 17);
  final int n = (26 * (w * h) / (artWidth * artHeight)).toInt() + 12;
  for (int i = 0; i < n; i++) {
    final double cx = (-0.04 + rng.nextDouble() * 1.08) * w;
    final double cy = (-0.04 + rng.nextDouble() * 1.08) * h;
    final double r = (0.045 + rng.nextDouble() * 0.053) * w;
    final double hue = rng.nextDouble();
    final List<double> col = <double>[
      236 * (1 - hue) + 196 * hue,
      172 * (1 - hue) + 150 * hue,
      196 * (1 - hue) + 232 * hue,
    ];
    final int petalN = 5 + (rng.nextDouble() * 3).toInt();
    final double off = rng.nextDouble() * 360.0;
    for (int k = 0; k < petalN; k++) {
      final double a = off + 360.0 * k / petalN;
      final double ra = a * math.pi / 180.0;
      final double px = cx + math.cos(ra) * r * 0.56;
      final double py = cy + math.sin(ra) * r * 0.56;
      final double sh = 0.82 + rng.nextDouble() * 0.24;
      fillEllipseF(
        f,
        px.toInt(),
        py.toInt(),
        math.max(2, (r * 0.52).toInt()).toDouble(),
        math.max(2, (r * 0.30).toInt()).toDouble(),
        a,
        <double>[col[0] * sh, col[1] * sh, col[2] * sh],
      );
    }
    final double cr = math.max(2, (r * 0.24).toInt()).toDouble();
    fillCircleF(f, cx.toInt(), cy.toInt(), cr,
        const <double>[250.0, 206.0, 96.0]);
    fillCircleF(f, cx.toInt(), cy.toInt(), cr,
        const <double>[214.0, 162.0, 64.0],
        thickness: 1);
  }

  final Rng rng2 = Rng(68 * 2654435761 + 23);
  final int nb = (70 * (w * h) / (artWidth * artHeight)).toInt() + 24;
  for (int i = 0; i < nb; i++) {
    final int cx = (rng2.nextDouble() * w).toInt();
    final int cy = (rng2.nextDouble() * h).toInt();
    final double ln = (0.02 + rng2.nextDouble() * 0.05) * w;
    final double a = rng2.nextDouble() * 2 * math.pi;
    final double k = 0.7 + rng2.nextDouble() * 0.6;
    final List<double> col = <double>[58 * k, 92 * k, 52 * k];
    lineF(f, cx, cy, (cx + math.cos(a) * ln).toInt(),
        (cy + math.sin(a) * ln).toInt(), col);
    fillEllipseF(
      f,
      cx,
      cy,
      math.max(2, (ln * 0.22).toInt()).toDouble(),
      math.max(1, (ln * 0.10).toInt()).toDouble(),
      a * 180.0 / math.pi,
      <double>[col[0] * 1.2, col[1] * 1.2, col[2] * 1.2],
    );
  }

  detail(f, 62, amount: 20.0, cx: 12.0, tint: <double>[0.9, 1.0, 0.9]);
  specks(f, 64, 200, 0.30, 0.9, <double>[120, 140, 110], rmax: 2);
  grain(f, 59, 12.0);
  return f;
}

Frame aurora(int w, int h) {
  final Frame f = Frame(w, h);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    for (int x = 0; x < w; x++) {
      f.setAt(x, y, 26 + yy * 26, 28 + yy * 30, 52 + yy * 34);
    }
  }

  final Rng rng = Rng(101 * 2654435761 + 29);
  final int nStars =
      (320 * (w * h) / (artWidth * artHeight)).toInt() + 80;
  for (int i = 0; i < nStars; i++) {
    final int sx = (rng.nextDouble() * w).toInt();
    final int sy = (math.pow(rng.nextDouble(), 1.7) * h).toInt();
    final double b = 70 + rng.nextDouble() * 160;
    f.addAt(sx, sy, b, b, b * 0.95);
  }

  const List<List<Object>> bands = <List<Object>>[
    <Object>[0.30, 0.30, 0.16, <double>[60, 224, 150]],
    <Object>[0.54, 0.24, 0.12, <double>[110, 236, 180]],
    <Object>[0.74, 0.34, 0.20, <double>[96, 150, 232]],
  ];
  for (int i = 0; i < bands.length; i++) {
    final double bcx = bands[i][0] as double;
    final double amp = bands[i][1] as double;
    final double wid = bands[i][2] as double;
    final List<double> col = bands[i][3] as List<double>;
    final Float32List drift = fbm1(h, 200 + i * 17, 4, 3.0);
    final Float32List streak =
        fbm(w, h, 300 + i * 23, octaves: 4, cx: 26.0, stretch: 6.0);
    for (int y = 0; y < h; y++) {
      final double yy = h <= 1 ? 0.0 : y / (h - 1);
      final double centre = bcx + (drift[y] - 0.5) * amp;
      final double fade = smoothEdge(0.86, 0.10, yy);
      final int row = y * w;
      for (int x = 0; x < w; x++) {
        final double xs = w <= 1 ? 0.0 : x / (w - 1);
        final double t = (xs - centre) / wid;
        final double band = math.exp(-(t * t) * 2.6);
        final double k =
            band * fade * (0.55 + 0.45 * streak[row + x]);
        f.addAt(x, y, k * col[0], k * col[1], k * col[2]);
      }
    }
  }

  blurFrame(f, math.max(0.5, w / 1400.0));

  final Float32List dust = fbm(w, h, 117, octaves: 6, cx: 16.0);
  final Float32List cloud = fbm(w, h, 121, octaves: 5, cx: 7.0);
  final Float32List band2 =
      fbm(w, h, 123, octaves: 5, cx: 30.0, stretch: 8.0);
  final Float32List grains = fbm(w, h, 127, octaves: 6, cx: 34.0);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    final double s1 = smoothEdge(0.88, 0.16, yy);
    final double s2 = smoothEdge(0.80, 0.22, yy);
    final double hz = smoothEdge(0.72, 1.0, yy);
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : x / (w - 1);
      final int i = row + x;
      final double du = dust[i] - 0.5;
      final double veil = smoothEdge(0.52, 0.88, cloud[i]) * 0.62;
      final double bb = (band2[i] - 0.5) * s1;
      final double mt = (xx * 1.1 - yy * 0.7 - 0.18) / 0.16;
      final double milky = math.exp(-(mt * mt) * 2.0);
      final double mk = milky * s2 * (0.45 + grains[i] * 0.55);
      final int p = i * 3;
      f.d[p] += du * 34 + veil * 34 + bb * 26 + mk * 36 + hz * 24;
      f.d[p + 1] += du * 38 + veil * 52 + bb * 40 + mk * 34 + hz * 30;
      f.d[p + 2] += du * 50 + veil * 70 + bb * 30 + mk * 44 + hz * 22;
    }
  }

  final Float32List ground = fbm1(w, 133, 4, 4.0);
  final Float32List tree = fbm(w, h, 137, octaves: 6, cx: 40.0);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    final int row = y * w;
    for (int x = 0; x < w; x++) {
      final double line = 0.90 + (ground[x] - 0.5) * 0.05;
      if (yy < line) continue;
      final double d = (tree[row + x] - 0.5) * 40.0;
      final int p = (row + x) * 3;
      f.d[p] = 24 + d;
      f.d[p + 1] = 30 + d;
      f.d[p + 2] = 34 + d;
    }
  }

  specks(f, 141, 200, 0.25, 0.8, <double>[90, 120, 150], rmax: 2);
  grain(f, 113, 9.0);
  return f;
}

Frame mosaic(int w, int h) {
  final Frame f = Frame(w, h);
  for (int i = 0; i < f.d.length; i += 3) {
    f.d[i] = 38;
    f.d[i + 1] = 36;
    f.d[i + 2] = 40;
  }
  const int cols = 26;
  final int rows = math.max(6, (cols * h / w).round());
  final double cw = w / cols;
  final double ch = h / rows;
  final Float32List field = fbm(w, h, 131, octaves: 4, cx: 3.0);
  final Rng rng = Rng(73 * 2654435761 + 31);
  const List<List<double>> palette = <List<double>>[
    <double>[206, 92, 74],
    <double>[236, 176, 78],
    <double>[76, 148, 166],
    <double>[54, 96, 138],
    <double>[214, 206, 182],
    <double>[132, 84, 146],
  ];
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      final int x0 = (c * cw).toInt() + 1;
      final int y0 = (r * ch).toInt() + 1;
      final int x1 = ((c + 1) * cw).toInt() - 1;
      final int y1 = ((r + 1) * ch).toInt() - 1;
      if (x1 <= x0 || y1 <= y0) continue;
      final int fy = math.min(h - 1, ((r + 0.5) * ch).toInt());
      final int fx = math.min(w - 1, ((c + 0.5) * cw).toInt());
      final double v = field[fy * w + fx];
      int k = (v * palette.length).toInt();
      if (k < 0) k = 0;
      if (k > palette.length - 1) k = palette.length - 1;
      final double j = 0.82 + rng.nextDouble() * 0.32;
      final List<double> col = <double>[
        palette[k][0] * j,
        palette[k][1] * j,
        palette[k][2] * j,
      ];
      fillRectF(f, x0, y0, x1, y1, col);
      strokeRectF(f, x0, y0, x1, y1,
          <double>[col[0] * 0.74, col[1] * 0.74, col[2] * 0.74]);
      if (rng.nextDouble() < 0.30) {
        fillCircleF(
          f,
          (x0 + x1) ~/ 2,
          (y0 + y1) ~/ 2,
          math.max(1, (math.min(cw, ch) * 0.18).toInt()).toDouble(),
          <double>[col[0] * 1.22, col[1] * 1.22, col[2] * 1.22],
        );
      }
    }
  }
  detail(f, 151, amount: 12.0, cx: 8.0);
  grain(f, 149, 12.0);
  return f;
}

typedef ArtGen = Frame Function(int w, int h);

const Map<String, ArtGen> gallery = <String, ArtGen>{
  'dusk': dusk,
  'water': water,
  'leaves': leaves,
  'sand': sand,
  'stones': stones,
  'petals': petals,
  'aurora': aurora,
  'mosaic': mosaic,
};

final List<String> artNames = gallery.keys.toList(growable: false);

final Map<String, int> artSeeds = <String, int>{
  for (int i = 0; i < artNames.length; i++) artNames[i]: 1000 + i * 137,
};

Float32List _luma(Frame f) {
  final Float32List g = Float32List(f.w * f.h);
  for (int i = 0, p = 0; i < g.length; i++, p += 3) {
    double r = f.d[p];
    double gg = f.d[p + 1];
    double b = f.d[p + 2];
    if (r < 0) r = 0;
    if (r > 255) r = 255;
    if (gg < 0) gg = 0;
    if (gg > 255) gg = 255;
    if (b < 0) b = 0;
    if (b > 255) b = 255;
    g[i] = 0.299 * r + 0.587 * gg + 0.114 * b;
  }
  return g;
}

void ensureTexture(Frame f, int seed,
    {double target = minTileStd, int grid = 12, int rounds = 6}) {
  final int w = f.w;
  final int h = f.h;
  final double scale = math.min(1.0, 420.0 / math.max(w, h));
  final int sw = math.max(8, (w * scale).toInt());
  final int sh = math.max(8, (h * scale).toInt());
  final int kx = math.max(3, sw ~/ grid);
  final int ky = math.max(3, sh ~/ grid);
  for (int k = 0; k < rounds; k++) {
    Float32List g = _luma(f);
    if (sw != w || sh != h) g = areaPlane(g, w, h, sw, sh);
    final Float32List sq = Float32List(g.length);
    for (int i = 0; i < g.length; i++) {
      sq[i] = g[i] * g[i];
    }
    final Float32List mean = boxBlurPlane(g, sw, sh, kx, ky);
    final Float32List msq = boxBlurPlane(sq, sw, sh, kx, ky);
    final Float32List need = Float32List(g.length);
    double peak = 0;
    for (int i = 0; i < g.length; i++) {
      final double v = msq[i] - mean[i] * mean[i];
      final double local = v <= 0 ? 0.0 : math.sqrt(v);
      double n = (target * 1.9 - local) / (target * 1.9);
      if (n < 0) n = 0;
      if (n > 1) n = 1;
      need[i] = n;
      if (n > peak) peak = n;
    }
    if (peak <= 0.01) break;
    final Float32List full =
        (sw != w || sh != h) ? resizePlane(need, sw, sh, w, h) : need;
    final Float32List d =
        fbm(w, h, seed + k * 211, octaves: 6, cx: 16.0 + k * 9.0);
    final double amount = 34.0 + k * 10.0;
    for (int i = 0, p = 0; i < full.length; i++, p += 3) {
      final double v = (d[i] - 0.5) * full[i] * amount;
      f.d[p] += v;
      f.d[p + 1] += v;
      f.d[p + 2] += v;
    }
  }
}

Raster renderArt(String name, {int w = artWidth, int h = artHeight}) {
  int rw = w;
  int rh = h;
  if (w > artMax) {
    rw = artMax;
    rh = math.max(2, (h * artMax / w).round());
  }
  Frame f = gallery[name]!(rw, rh);
  if (rw != w || rh != h) {
    f = frameFromRaster(resizeRaster(f.toRaster(), w, h));
  }
  ensureTexture(f, artSeeds[name]!);
  return f.toRaster();
}

double tileStd(Raster r, int x0, int y0, int tw, int th) {
  double sum = 0;
  double sq = 0;
  int n = 0;
  for (int y = y0; y < y0 + th; y++) {
    for (int x = x0; x < x0 + tw; x++) {
      final int i = (y * r.w + x) * 4;
      final double g = 0.299 * r.rgba[i] +
          0.587 * r.rgba[i + 1] +
          0.114 * r.rgba[i + 2];
      sum += g;
      sq += g * g;
      n++;
    }
  }
  if (n == 0) return 0;
  final double m = sum / n;
  final double v = sq / n - m * m;
  return v <= 0 ? 0 : math.sqrt(v);
}
