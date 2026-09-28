import 'dart:math' as math;
import 'dart:typed_data';

import '../core/synth.dart';
import 'fft.dart';

const Map<String, double> psychoLimits = <String, double>{
  'sharp': 0.80,
  'treble': 25.0,
  'harsh': 6.0,
};

const double roughTonal = 12.0;
const double roughNoise = 45.0;
const double tonalCrest = 3.6;

const List<List<double>> bands = <List<double>>[
  <double>[50, 150],
  <double>[150, 300],
  <double>[300, 600],
  <double>[600, 1200],
  <double>[1200, 2400],
  <double>[2400, 4800],
  <double>[4800, 9600],
];

const double modMinHz = 18.0;
const double modPeak = 70.0;
const double modWid = 0.65;

double _bark(double f) =>
    13.0 * math.atan(0.00076 * f) +
    3.5 * math.atan(math.pow(f / 7500.0, 2).toDouble());

Float64List _head(Float64List x, int sr) {
  final int n = pow2Floor(math.min(x.length, sr * 4));
  return Float64List.fromList(x.sublist(0, n));
}

double sharpness(Float64List x, {int sr = sampleRate}) {
  final Float64List seg = _head(x, sr);
  final Float64List mag = magnitudeSpectrum(seg);
  final Float64List f = rfftFreq(seg.length, 1.0 / sr);
  double tot = 0;
  double acc = 0;
  for (int i = 0; i < mag.length; i++) {
    final double p = mag[i] * mag[i];
    final double z = _bark(f[i]);
    final double w =
        z < 15.8 ? 1.0 : 0.15 * math.exp(0.42 * (z - 15.8)) + 0.85;
    tot += p;
    acc += p * z * w;
  }
  if (tot <= 0) return 0.0;
  return 0.11 * acc / tot;
}

double trebleShare(Float64List x,
    {int sr = sampleRate, double cut = 1500.0}) {
  final Float64List seg = _head(x, sr);
  final Float64List mag = magnitudeSpectrum(seg);
  final Float64List f = rfftFreq(seg.length, 1.0 / sr);
  double up = 0;
  double all = 0;
  for (int i = 0; i < mag.length; i++) {
    all += mag[i];
    if (f[i] > cut) up += mag[i];
  }
  return up / (all + 1e-9) * 100.0;
}

double harshBand(Float64List x, {int sr = sampleRate}) {
  final Float64List seg = _head(x, sr);
  final Float64List mag = magnitudeSpectrum(seg);
  final Float64List f = rfftFreq(seg.length, 1.0 / sr);
  double up = 0;
  double all = 0;
  for (int i = 0; i < mag.length; i++) {
    all += mag[i];
    if (f[i] >= 2000.0 && f[i] <= 5000.0) up += mag[i];
  }
  return up / (all + 1e-9) * 100.0;
}

double tonality(Float64List x, {int sr = sampleRate}) {
  final Float64List seg = _head(x, sr);
  final Float64List mag = magnitudeSpectrum(seg);
  double tot = 0;
  for (final double m in mag) {
    tot += m * m;
  }
  if (tot <= 0) return 0.0;
  double ent = 0;
  int count = 0;
  for (final double m in mag) {
    final double p = m * m / tot;
    if (p > 0) {
      ent -= p * math.log(p);
      count++;
    }
  }
  if (count <= 1) return 1.0;
  return 1.0 - ent / math.log(count);
}

double crest(Float64List x) {
  double acc = 0;
  double peak = 0;
  for (final double v in x) {
    acc += v * v;
    if (v.abs() > peak) peak = v.abs();
  }
  final double r = math.sqrt(acc / x.length);
  if (r <= 0) return 0.0;
  return peak / r;
}

double medianFreq(Float64List x, {int sr = sampleRate}) {
  final int n = pow2Floor(x.length);
  final Float64List seg = Float64List.fromList(x.sublist(0, n));
  final Float64List mag = magnitudeSpectrum(seg);
  final Float64List f = rfftFreq(seg.length, 1.0 / sr);
  double tot = 0;
  for (final double m in mag) {
    tot += m;
  }
  if (tot <= 0) return 0.0;
  double acc = 0;
  for (int i = 0; i < mag.length; i++) {
    acc += mag[i];
    if (acc >= 0.5 * tot) return f[i];
  }
  return f[f.length - 1];
}

Float64List rmsEnvelope(Float64List x, {int sr = sampleRate}) {
  final int win = sr;
  final int hop = win ~/ 2;
  final List<double> out = <double>[];
  for (int i = 0; i + win < x.length; i += hop) {
    double acc = 0;
    for (int k = i; k < i + win; k++) {
      acc += x[k] * x[k];
    }
    out.add(math.sqrt(acc / win));
  }
  return Float64List.fromList(out);
}

double percentile(Float64List v, double p) {
  if (v.isEmpty) return 0.0;
  final List<double> s = List<double>.from(v)..sort();
  final double idx = (s.length - 1) * p / 100.0;
  final int lo = idx.floor();
  final int hi = idx.ceil();
  if (lo == hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

double centroid(Float64List x, {int sr = sampleRate}) {
  final Float64List seg = _head(x, sr);
  final Float64List mag = magnitudeSpectrum(seg);
  final Float64List f = rfftFreq(seg.length, 1.0 / sr);
  double num = 0;
  double den = 0;
  for (int i = 0; i < mag.length; i++) {
    final double p = mag[i] * mag[i];
    num += f[i] * p;
    den += p;
  }
  if (den <= 0) return 0.0;
  return num / den;
}

Float64List _bandSignal(Float64List x, double lo, double hi, int sr) {
  final int n = x.length;
  final Spectrum sp = rfft(x);
  final Float64List f = rfftFreq(n, 1.0 / sr);
  for (int i = 0; i < sp.length; i++) {
    if (f[i] < lo || f[i] >= hi) {
      sp.re[i] = 0;
      sp.im[i] = 0;
    }
  }
  return irfft(sp, n);
}

Float64List _hilbertEnv(Float64List x) => analyticEnvelope(x);

double roughness(Float64List x, {int sr = sampleRate}) {
  final Float64List seg = _head(x, sr);
  double peak = 0;
  for (final double v in seg) {
    if (v.abs() > peak) peak = v.abs();
  }
  if (peak <= 1e-12) return 0.0;
  double power = 0;
  for (final double v in seg) {
    power += v * v;
  }
  power = power / seg.length + 1e-15;

  double total = 0;
  for (final List<double> b in bands) {
    final Float64List bs = _bandSignal(seg, b[0], b[1], sr);
    double bp = 0;
    for (final double v in bs) {
      bp += v * v;
    }
    bp /= bs.length;
    if (bp / power < 0.01) continue;

    final Float64List env = _hilbertEnv(bs);
    final int dec = math.max(1, sr ~/ 2000);
    final int m = pow2Floor(env.length ~/ dec);
    if (m < 8) continue;
    final Float64List e = Float64List(m);
    double mean = 0;
    for (int i = 0; i < m; i++) {
      e[i] = env[i * dec];
      mean += e[i];
    }
    mean = mean / m + 1e-15;
    for (int i = 0; i < m; i++) {
      e[i] = (e[i] - mean) * (0.5 - 0.5 * math.cos(2 * math.pi * i / (m - 1)));
    }
    final double esr = sr / dec;
    final Float64List mag = magnitudeSpectrum(e);
    final Float64List f = rfftFreq(m, 1.0 / esr);
    double depBand = 0;
    for (int i = 0; i < mag.length; i++) {
      final double amp = mag[i] / m * 2.0;
      if (f[i] < modMinHz || f[i] > 300.0) continue;
      final double lg = math.log(math.max(f[i], 1e-6) / modPeak);
      depBand += amp * math.exp(-(lg * lg) / modWid);
    }
    total += depBand * (bp / power);
  }
  return total * 100.0;
}

class PsychoReport {
  PsychoReport(this.name, this.sharp, this.rough, this.treble, this.harsh,
      this.tonal, this.crestFactor, this.centre);

  final String name;
  final double sharp;
  final double rough;
  final double treble;
  final double harsh;
  final double tonal;
  final double crestFactor;
  final double centre;

  double get roughLimit => tonal > 0.62 ? roughTonal : roughNoise;

  List<String> get verdict {
    final List<String> bad = <String>[];
    if (sharp > psychoLimits['sharp']!) {
      bad.add('sharp ${sharp.toStringAsFixed(2)}');
    }
    if (treble > psychoLimits['treble']!) {
      bad.add('treble ${treble.toStringAsFixed(2)}');
    }
    if (harsh > psychoLimits['harsh']!) {
      bad.add('harsh ${harsh.toStringAsFixed(2)}');
    }
    if (rough > roughLimit) {
      bad.add('rough ${rough.toStringAsFixed(1)}');
    }
    return bad;
  }
}

PsychoReport reportOn(String name, Float64List x, {int sr = sampleRate}) {
  return PsychoReport(
    name,
    sharpness(x, sr: sr),
    roughness(x, sr: sr),
    trebleShare(x, sr: sr),
    harshBand(x, sr: sr),
    tonality(x, sr: sr),
    crest(x),
    centroid(x, sr: sr),
  );
}
