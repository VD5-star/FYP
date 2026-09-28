import 'dart:math' as math;
import 'dart:typed_data';

import '../core/synth.dart';
import '../jigsaw/raster.dart';
import 'sounds.dart';

const int sceneW = 1100;
const int sceneH = 700;
const double maxLuma = 96.0;

const Map<String, String> sceneGallery = <String, String>{
  'stream': 'rain',
  'bowl': 'chime',
  'breeze': 'wind',
  'strings': 'wood',
  'deep': 'hum',
};

Float32List _noise(int w, int h, int seed, int cells) {
  final int cx = math.max(2, cells);
  final int cy = math.max(2, (cells * h / w).round());
  final Rng rng = Rng(seed);
  final Float32List g = Float32List((cy + 1) * (cx + 1));
  for (int i = 0; i < g.length; i++) {
    g[i] = rng.nextDouble();
  }
  return resizePlane(g, cx + 1, cy + 1, w, h);
}

Float32List _fbm(int w, int h, int seed, int octaves, double cells) {
  final Float32List out = Float32List(w * h);
  double amp = 1.0;
  double tot = 0.0;
  double c = cells;
  for (int i = 0; i < octaves; i++) {
    final Float32List layer = _noise(w, h, seed + i * 97, c.toInt());
    for (int k = 0; k < out.length; k++) {
      out[k] += layer[k] * amp;
    }
    tot += amp;
    amp *= 0.5;
    c *= 2;
  }
  for (int k = 0; k < out.length; k++) {
    out[k] /= tot;
  }
  return out;
}

Frame _vgrad(int w, int h, List<double> top, List<double> bottom) {
  final Frame f = Frame(w, h);
  for (int y = 0; y < h; y++) {
    final double t = h <= 1 ? 0.0 : y / (h - 1);
    final double r = top[0] * (1 - t) + bottom[0] * t;
    final double g = top[1] * (1 - t) + bottom[1] * t;
    final double b = top[2] * (1 - t) + bottom[2] * t;
    for (int x = 0; x < w; x++) {
      f.setAt(x, y, r, g, b);
    }
  }
  return f;
}

void _vignette(Frame f, double amount) {
  final int w = f.w;
  final int h = f.h;
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : -1.0 + 2.0 * y / (h - 1);
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : -1.0 + 2.0 * x / (w - 1);
      final double r = math.sqrt(xx * xx + yy * yy) / 1.414;
      final double k = 1.0 - amount * r * r;
      final int i = (y * w + x) * 3;
      f.d[i] *= k;
      f.d[i + 1] *= k;
      f.d[i + 2] *= k;
    }
  }
}

void _addPlane(Frame f, Float32List p, List<double> tint,
    {double bias = 0.0, double scale = 1.0}) {
  for (int i = 0, o = 0; i < p.length; i++, o += 3) {
    final double v = (p[i] + bias) * scale;
    f.d[o] += v * tint[0];
    f.d[o + 1] += v * tint[1];
    f.d[o + 2] += v * tint[2];
  }
}

Raster rain(int w, int h) {
  final Frame f = _vgrad(w, h, <double>[56, 46, 38], <double>[36, 28, 22]);
  _addPlane(f, _fbm(w, h, 5, 5, 3.0), <double>[22, 18, 16], bias: -0.5);

  final Rng rng = Rng(9);
  final Float32List layer = Float32List(w * h);
  final int drops = (w * h / 5200).toInt();
  for (int i = 0; i < drops; i++) {
    final int x0 = (rng.nextDouble() * w).toInt();
    final int y0 = (rng.nextDouble() * h).toInt();
    final int ln = (h * (0.04 + rng.nextDouble() * 0.09)).toInt();
    final int lean = (ln * 0.14).toInt();
    final double v = 0.25 + rng.nextDouble() * 0.50;
    for (int k = 0; k <= ln; k++) {
      final int x = x0 + (lean * k / math.max(1, ln)).toInt();
      final int y = y0 + k;
      if (x >= 0 && x < w && y >= 0 && y < h) layer[y * w + x] += v;
    }
  }
  final Float32List soft = boxBlurPlane(layer, w, h, 3, 3);
  _addPlane(f, soft, <double>[48, 40, 34]);

  for (int y = 0; y < h; y++) {
    final double gy = (-1.4 + 2.0 * y / math.max(1, h - 1)) * 1.3;
    for (int x = 0; x < w; x++) {
      final double gx = (-1.0 + 2.0 * x / math.max(1, w - 1)) * 1.6;
      final double g = math.exp(-(gx * gx + gy * gy) * 1.6);
      final int i = (y * w + x) * 3;
      f.d[i] += g * 28;
      f.d[i + 1] += g * 22;
      f.d[i + 2] += g * 18;
    }
  }
  _vignette(f, 0.34);
  return f.toRaster();
}

Raster wood(int w, int h) {
  final Frame f = _vgrad(w, h, <double>[34, 46, 62], <double>[19, 26, 36]);
  final Float32List warp = _fbm(w, h, 21, 4, 3.0);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    for (int x = 0; x < w; x++) {
      final int p = y * w + x;
      final double ring =
          math.sin((yy * 13.0 + warp[p] * 2.6) * math.pi * 2.0) * 0.5;
      final int i = p * 3;
      f.d[i] += ring * 9;
      f.d[i + 1] += ring * 14;
      f.d[i + 2] += ring * 20;
    }
  }
  _addPlane(f, _fbm(w, h, 23, 5, 40.0), <double>[8, 12, 16], bias: -0.5);

  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : x / (w - 1);
      final double knot = math.exp(-(math.pow((xx - 0.72) * 5.0, 2) +
                  math.pow((yy - 0.36) * 6.2, 2))
              .toDouble() *
          2.0);
      final double lamp = math.exp(-(math.pow((xx - 0.30) * 1.5, 2) +
                  math.pow((yy - 0.24) * 1.8, 2))
              .toDouble() *
          1.5);
      final int i = (y * w + x) * 3;
      f.d[i] += lamp * 12 - knot * 14;
      f.d[i + 1] += lamp * 22 - knot * 20;
      f.d[i + 2] += lamp * 30 - knot * 26;
    }
  }
  _vignette(f, 0.36);
  return f.toRaster();
}

Raster wind(int w, int h) {
  final Frame f = _vgrad(w, h, <double>[62, 52, 44], <double>[44, 40, 30]);
  final Float32List streak = _fbm(w, h, 31, 5, 9.0);
  final int sh = math.max(2, h ~/ 8);
  final Float32List squashed = areaPlane(streak, w, h, w, sh);
  final Float32List stretched = resizePlane(squashed, w, sh, w, h);
  _addPlane(f, stretched, <double>[30, 28, 26], bias: -0.5);

  final Float32List field = _fbm(w, h, 33, 4, 5.0);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    final double band = ((yy - 0.62) * 3.2).clamp(0.0, 1.0);
    for (int x = 0; x < w; x++) {
      final int p = y * w + x;
      final double g = band * (0.4 + field[p] * 0.6);
      final int i = p * 3;
      f.d[i] -= g * 14;
      f.d[i + 1] -= g * 16;
      f.d[i + 2] -= g * 18;
    }
  }

  final Rng rng = Rng(35);
  final Float32List blades = Float32List(w * h);
  final int count = w ~/ 7;
  for (int i = 0; i < count; i++) {
    final int x0 = (rng.nextDouble() * w).toInt();
    final int base = (h * (0.72 + rng.nextDouble() * 0.30)).toInt();
    final int ln = (h * (0.08 + rng.nextDouble() * 0.14)).toInt();
    final int bend = (6 + rng.nextDouble() * 20).toInt();
    final double v = 0.2 + rng.nextDouble() * 0.35;
    for (int k = 0; k <= ln; k++) {
      final double t = ln <= 0 ? 0.0 : k / ln;
      final int x = x0 + (bend * t).toInt();
      final int y = base - k;
      if (x >= 0 && x < w && y >= 0 && y < h) blades[y * w + x] += v;
    }
  }
  _addPlane(f, blades, <double>[22, 26, 22]);
  _vignette(f, 0.32);
  return f.toRaster();
}

Raster chime(int w, int h) {
  final Frame f = _vgrad(w, h, <double>[54, 34, 30], <double>[52, 40, 46]);
  for (int y = 0; y < h; y++) {
    final double gy = (-0.4 + 2.2 * y / math.max(1, h - 1)) * 1.1;
    for (int x = 0; x < w; x++) {
      final double gx = (-1.0 + 2.0 * x / math.max(1, w - 1)) * 1.2;
      final double g = math.exp(-(gx * gx + gy * gy) * 1.4);
      final int i = (y * w + x) * 3;
      f.d[i] += g * 30;
      f.d[i + 1] += g * 34;
      f.d[i + 2] += g * 40;
    }
  }

  final Rng rng = Rng(41);
  final Float32List spec = Float32List(w * h);
  for (int i = 0; i < 90; i++) {
    final int x = (rng.nextDouble() * w).toInt();
    final int y = (rng.nextDouble() * h * 0.8).toInt();
    final int r = (1 + rng.nextDouble() * 2).toInt();
    final double v = 0.3 + rng.nextDouble() * 0.7;
    for (int dy = -r; dy <= r; dy++) {
      for (int dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r * r) continue;
        final int px = x + dx;
        final int py = y + dy;
        if (px >= 0 && px < w && py >= 0 && py < h) {
          spec[py * w + px] += v;
        }
      }
    }
  }
  _addPlane(f, boxBlurPlane(spec, w, h, 5, 5), <double>[50, 44, 46]);
  _addPlane(f, _fbm(w, h, 43, 5, 4.0), <double>[16, 13, 14], bias: -0.5);
  _vignette(f, 0.38);
  return f.toRaster();
}

Raster hum(int w, int h) {
  final Frame f = _vgrad(w, h, <double>[38, 30, 26], <double>[30, 30, 34]);
  _addPlane(f, _fbm(w, h, 51, 5, 2.5), <double>[20, 18, 18], bias: -0.5);

  final double ar = w / math.max(1, h);
  for (int y = 0; y < h; y++) {
    final double yy = h <= 1 ? 0.0 : y / (h - 1);
    for (int x = 0; x < w; x++) {
      final double xx = w <= 1 ? 0.0 : x / (w - 1);
      final double dx = (xx - 0.5) * ar;
      final double dy = yy - 0.52;
      final double d = math.sqrt(dx * dx + dy * dy);
      double add = 0;
      add += math.exp(-math.pow(d - 0.55, 2).toDouble() / 0.010) * 1.0;
      add += math.exp(-math.pow(d - 0.30, 2).toDouble() / 0.010) * 0.6;
      final double core = math.exp(-(dx * dx + dy * dy) * 9.0);
      final int i = (y * w + x) * 3;
      f.d[i] += add * 18 + core * 26;
      f.d[i + 1] += add * 15 + core * 22;
      f.d[i + 2] += add * 16 + core * 24;
    }
  }
  _vignette(f, 0.40);
  return f.toRaster();
}

Raster _cap(Raster r, {double ceiling = maxLuma}) {
  double sum = 0;
  final int n = r.w * r.h;
  for (int i = 0; i < n; i++) {
    final int o = i * 4;
    sum += 0.114 * r.rgba[o] + 0.587 * r.rgba[o + 1] + 0.299 * r.rgba[o + 2];
  }
  final double mean = sum / math.max(1, n);
  if (mean <= ceiling) return r;
  final double g = ceiling / mean;
  final Uint8List out = Uint8List(r.rgba.length);
  for (int i = 0; i < n; i++) {
    final int o = i * 4;
    for (int c = 0; c < 3; c++) {
      final double v = r.rgba[o + c] * g;
      out[o + c] = v <= 0 ? 0 : (v >= 255 ? 255 : v.toInt());
    }
    out[o + 3] = 255;
  }
  return Raster(r.w, r.h, out);
}

Raster renderScene(String sound, {int w = sceneW, int h = sceneH}) {
  final String which = sceneGallery[sound] ?? 'hum';
  switch (which) {
    case 'rain':
      return _cap(rain(w, h));
    case 'wood':
      return _cap(wood(w, h));
    case 'wind':
      return _cap(wind(w, h));
    case 'chime':
      return _cap(chime(w, h));
    case 'hum':
      return _cap(hum(w, h));
  }
  return _cap(hum(w, h));
}

Map<String, Raster> renderAllScenes({int w = sceneW, int h = sceneH}) {
  return <String, Raster>{
    for (final String n in soundNames) n: renderScene(n, w: w, h: h)
  };
}

Raster blendScenes(Map<String, Raster> scenes, Map<String, double> weights,
    int w, int h) {
  final Uint8List out = Uint8List(w * h * 4);
  final List<MapEntry<String, double>> live = weights.entries
      .where((MapEntry<String, double> e) => e.value > 0.001)
      .toList();
  if (live.isEmpty) {
    for (int i = 3; i < out.length; i += 4) {
      out[i] = 255;
    }
    return Raster(w, h, out);
  }
  double total = 0;
  for (final MapEntry<String, double> e in live) {
    total += e.value;
  }
  final Float32List acc = Float32List(w * h * 3);
  for (final MapEntry<String, double> e in live) {
    final Raster? src = scenes[e.key];
    if (src == null) continue;
    final Raster fit = (src.w == w && src.h == h)
        ? src
        : resizeRaster(src, w, h);
    final double k = e.value / total;
    for (int i = 0, o = 0; o < acc.length; i += 4, o += 3) {
      acc[o] += fit.rgba[i] * k;
      acc[o + 1] += fit.rgba[i + 1] * k;
      acc[o + 2] += fit.rgba[i + 2] * k;
    }
  }
  for (int i = 0, o = 0; o < acc.length; i += 4, o += 3) {
    out[i] = acc[o].clamp(0, 255).toInt();
    out[i + 1] = acc[o + 1].clamp(0, 255).toInt();
    out[i + 2] = acc[o + 2].clamp(0, 255).toInt();
    out[i + 3] = 255;
  }
  return Raster(w, h, out);
}
