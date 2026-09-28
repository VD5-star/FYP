import 'dart:math' as math;
import 'dart:typed_data';

import '../core/synth.dart';
import 'fft.dart';

const double loopSeconds = 48.0;
const double targetLoud = 0.115;
const double blockMs = 400.0;
const int envDecim = 64;

const List<String> soundNames = <String>[
  'stream',
  'bowl',
  'breeze',
  'strings',
  'deep',
];

const Map<String, String> soundLabels = <String, String>{
  'stream': 'the stream',
  'bowl': 'the bowl',
  'breeze': 'the breeze',
  'strings': 'the strings',
  'deep': 'the deep note',
};

const Map<String, double> soundPans = <String, double>{
  'stream': -0.50,
  'bowl': 0.45,
  'breeze': 0.62,
  'strings': -0.28,
  'deep': 0.00,
};

int frameCount(double seconds, int sr) =>
    pow2Floor((seconds * sr).round());

double _bin(double freq, int n, int sr) {
  final double step = sr / n;
  final double k = math.max(1.0, (freq / step).roundToDouble());
  return k * step;
}

Float64List _smoothEnv(int n, double rate, double depth, int seed, int sr) {
  final int m = math.max(64, n ~/ envDecim);
  final Rng rng = Rng(seed);
  final Float64List out = Float64List(m);
  for (int k = 0; k < 2; k++) {
    final double f = _bin(rate * (1.0 + k * 0.6), n, sr);
    final double cyc = f * n / sr;
    final double ph = rng.nextDouble() * 6.283;
    final double amp = 1.0 / (k + 1);
    for (int i = 0; i < m; i++) {
      out[i] += math.sin(2 * math.pi * cyc * (i / m) + ph) * amp;
    }
  }
  double peak = 0;
  for (final double v in out) {
    if (v.abs() > peak) peak = v.abs();
  }
  if (peak > 0) {
    for (int i = 0; i < m; i++) {
      out[i] /= peak;
    }
  }
  final Float64List env = Float64List(n);
  for (int i = 0; i < n; i++) {
    final double u = i * m / n;
    final int i0 = u.floor();
    final double fr = u - i0;
    final double a = out[i0 % m];
    final double b = out[(i0 + 1) % m];
    final double v = a + (b - a) * fr;
    env[i] = 1.0 - depth + depth * (0.5 + 0.5 * v);
  }
  return env;
}

Float64List _lowpass(Float64List x, double cut, double order, int sr) {
  final int n = x.length;
  final Spectrum sp = rfft(x);
  final Float64List f = rfftFreq(n, 1.0 / sr);
  for (int i = 0; i < sp.length; i++) {
    final double g =
        1.0 / (1.0 + math.pow(math.max(f[i], 1e-6) / cut, order));
    sp.re[i] *= g;
    sp.im[i] *= g;
  }
  return irfft(sp, n);
}

Float64List _pad(int n, double freq, List<List<double>> partials,
    double detune, double rate, int seed, int sr) {
  final Rng rng = Rng(seed);
  final Float64List out = Float64List(n);
  final List<Float64List> sways = <Float64List>[
    for (int k = 0; k < 3; k++)
      _smoothEnv(n, rate * (1.0 + k * 0.31), 0.10, seed + k * 17, sr)
  ];
  for (final List<double> p in partials) {
    final double mult = p[0];
    final double amp = p[1];
    final double base = freq * mult;
    final List<double> ds = <double>[-detune, 0.0, detune];
    for (int k = 0; k < 3; k++) {
      final double f = _bin(base * (1.0 + ds[k]), n, sr);
      final double cyc = f * n / sr;
      final double ph = rng.nextDouble() * 6.283;
      final Float64List sw = sways[k];
      final double w = 2 * math.pi * cyc;
      for (int i = 0; i < n; i++) {
        out[i] += math.sin(w * (i / n) + ph) * amp * sw[i] / 3.0;
      }
    }
  }
  double m = 0;
  for (final double v in out) {
    if (v.abs() > m) m = v.abs();
  }
  if (m > 0) {
    for (int i = 0; i < n; i++) {
      out[i] /= m;
    }
  }
  return out;
}

double _cycles(double period, double seconds) =>
    math.max(1.0, (seconds / math.max(0.01, period)).roundToDouble());

Float64List phrase(int n, double period, double depth, double floor,
    double skew, double seconds) {
  final double c = _cycles(period, seconds);
  final Float64List out = Float64List(n);
  for (int i = 0; i < n; i++) {
    final double t = i / n;
    final double ph = (t * c) % 1.0;
    double s = 0.5 - 0.5 * math.cos(2 * math.pi * ph);
    s = math.pow(s, skew).toDouble();
    out[i] = floor + (1.0 - floor) * (1.0 - depth + depth * s);
  }
  return out;
}

double loudness(Float64List x, int sr) {
  final int n = math.max(1, (sr * blockMs / 1000.0).round());
  final int hop = math.max(1, n ~/ 4);
  double meanSq(int from, int len) {
    double acc = 0;
    for (int i = from; i < from + len; i++) {
      acc += x[i] * x[i];
    }
    return acc / len;
  }

  if (x.length < n) return math.sqrt(meanSq(0, x.length));
  final List<double> p = <double>[];
  for (int i = 0; i + n <= x.length; i += hop) {
    p.add(meanSq(i, n));
  }
  if (p.isEmpty) return math.sqrt(meanSq(0, x.length));
  double mx = 0;
  for (final double v in p) {
    if (v > mx) mx = v;
  }
  final List<double> loud =
      p.where((double v) => v > mx * 0.01).toList();
  final List<double> use = loud.isEmpty ? p : loud;
  double acc = 0;
  for (final double v in use) {
    acc += v;
  }
  return math.sqrt(acc / use.length);
}

void _softLimit(Float64List x, {double ceiling = 0.90}) {
  final double knee = ceiling * 0.60;
  final double room = ceiling - knee;
  for (int i = 0; i < x.length; i++) {
    final double a = x[i].abs();
    if (a > knee) {
      final double s = x[i] < 0 ? -1.0 : 1.0;
      x[i] = s * (knee + room * _tanh((a - knee) / room));
    }
  }
}

double _tanh(double v) {
  if (v > 20) return 1.0;
  if (v < -20) return -1.0;
  final double e = math.exp(2 * v);
  return (e - 1) / (e + 1);
}

Float64List _norm(Float64List x, int sr) {
  double mean = 0;
  for (final double v in x) {
    mean += v;
  }
  mean /= x.length;
  for (int i = 0; i < x.length; i++) {
    x[i] -= mean;
  }
  for (int pass = 0; pass < 2; pass++) {
    final double lo = loudness(x, sr);
    if (lo > 0) {
      final double g = targetLoud / lo;
      for (int i = 0; i < x.length; i++) {
        x[i] *= g;
      }
    }
    _softLimit(x);
  }
  return x;
}

Float64List stream({double seconds = loopSeconds, int sr = sampleRate}) {
  final int n = frameCount(seconds, sr);
  final Float64List x = _pad(
      n,
      587.33,
      const <List<double>>[
        <double>[1.0, 1.0],
        <double>[2.0, 0.30],
        <double>[3.0, 0.13],
        <double>[4.0, 0.05],
      ],
      0.0016,
      0.052,
      101,
      sr);
  final Float64List ph = phrase(n, 5.3, 0.55, 0.30, 1.0, seconds);
  for (int i = 0; i < n; i++) {
    x[i] *= ph[i];
  }
  return _norm(_lowpass(x, 2200.0, 2.4, sr), sr);
}

Float64List bowl({double seconds = loopSeconds, int sr = sampleRate}) {
  final int n = frameCount(seconds, sr);
  final Float64List x = Float64List(n);
  const List<List<double>> tones = <List<double>>[
    <double>[329.63, 0.55],
    <double>[392.00, 0.42],
    <double>[440.00, 0.30],
  ];
  final int strikes = math.max(4, (seconds / 2.4).round());
  for (int k = 0; k < strikes; k++) {
    final int i = ((k + 0.35) * n / strikes).toInt() % n;
    final double f0 = tones[k % tones.length][0];
    final double wobble = tones[k % tones.length][1];
    final int dn = math.min(n, n ~/ strikes);
    final Float64List body = Float64List(dn);
    const List<List<double>> parts = <List<double>>[
      <double>[1.0, 1.0, 1.15],
      <double>[2.40, 0.45, 1.70],
      <double>[3.85, 0.22, 2.40],
      <double>[6.20, 0.10, 3.40],
      <double>[9.10, 0.05, 4.80],
    ];
    for (final List<double> p in parts) {
      final double f = _bin(f0 * p[0], n, sr);
      final double amp = p[1];
      final double dec = p[2];
      for (int t = 0; t < dn; t++) {
        final double tt = t / sr;
        final double w = 2 * math.pi * 1.7 * tt;
        body[t] += math.sin(2 * math.pi * f * tt + wobble * 0.06 * w) *
            amp *
            math.exp(-tt * dec);
      }
    }
    final int a = math.max(64, (sr * 0.045).toInt());
    for (int t = 0; t < a && t < dn; t++) {
      body[t] *= 0.5 - 0.5 * math.cos(math.pi * t / a);
    }
    double m = 0;
    for (final double v in body) {
      if (v.abs() > m) m = v.abs();
    }
    if (m > 0) {
      for (int t = 0; t < dn; t++) {
        body[t] /= m;
      }
    }
    for (final int start in <int>[i, i - n]) {
      final int e = start + dn;
      if (e <= 0) continue;
      final int s2 = math.max(0, start);
      final int e2 = math.min(n, e);
      for (int t = s2; t < e2; t++) {
        x[t] += body[t - start] * 0.9;
      }
    }
  }
  return _norm(_lowpass(x, 1500.0, 2.8, sr), sr);
}

Float64List breeze({double seconds = loopSeconds, int sr = sampleRate}) {
  final int n = frameCount(seconds, sr);
  final Float64List x = _pad(
      n,
      783.99,
      const <List<double>>[
        <double>[1.0, 1.0],
        <double>[2.0, 0.03],
        <double>[3.0, 0.006],
      ],
      0.0026,
      0.041,
      121,
      sr);
  final Float64List ph = phrase(n, 8.0, 0.70, 0.22, 1.7, seconds);
  for (int i = 0; i < n; i++) {
    x[i] *= ph[i];
  }
  return _norm(_lowpass(x, 1150.0, 3.8, sr), sr);
}

Float64List strings({double seconds = loopSeconds, int sr = sampleRate}) {
  final int n = frameCount(seconds, sr);
  final Float64List x = _pad(
      n,
      220.0,
      const <List<double>>[
        <double>[1.0, 1.0],
        <double>[2.0, 0.30],
        <double>[3.0, 0.14],
        <double>[4.0, 0.06],
        <double>[6.0, 0.02],
      ],
      0.0015,
      0.036,
      131,
      sr);
  final Float64List ph = phrase(n, 16.0, 0.42, 0.45, 2.4, seconds);
  for (int i = 0; i < n; i++) {
    x[i] *= ph[i];
  }
  return _norm(_lowpass(x, 2400.0, 2.2, sr), sr);
}

Float64List deep({double seconds = loopSeconds, int sr = sampleRate}) {
  final int n = frameCount(seconds, sr);
  final Float64List x = _pad(
      n,
      65.41,
      const <List<double>>[
        <double>[1.0, 1.0],
        <double>[2.0, 0.26],
        <double>[3.0, 0.09],
        <double>[4.0, 0.03],
      ],
      0.0011,
      0.028,
      141,
      sr);
  final Float64List ph = phrase(n, 10.7, 0.30, 0.60, 1.0, seconds);
  for (int i = 0; i < n; i++) {
    x[i] *= ph[i];
  }
  return _norm(_lowpass(x, 520.0, 2.4, sr), sr);
}

Float64List renderSound(String name,
    {double seconds = loopSeconds, int sr = sampleRate}) {
  switch (name) {
    case 'stream':
      return stream(seconds: seconds, sr: sr);
    case 'bowl':
      return bowl(seconds: seconds, sr: sr);
    case 'breeze':
      return breeze(seconds: seconds, sr: sr);
    case 'strings':
      return strings(seconds: seconds, sr: sr);
    case 'deep':
      return deep(seconds: seconds, sr: sr);
  }
  throw ArgumentError('unknown sound $name');
}

Map<String, Float64List> renderAllSounds(
    {double seconds = loopSeconds, int sr = sampleRate}) {
  return <String, Float64List>{
    for (final String n in soundNames)
      n: renderSound(n, seconds: seconds, sr: sr)
  };
}

Float64List phraseFor(String name, int n, double seconds) {
  switch (name) {
    case 'stream':
      return phrase(n, 5.3, 0.55, 0.30, 1.0, seconds);
    case 'breeze':
      return phrase(n, 8.0, 0.70, 0.22, 1.7, seconds);
    case 'strings':
      return phrase(n, 16.0, 0.42, 0.45, 2.4, seconds);
    case 'deep':
      return phrase(n, 10.7, 0.30, 0.60, 1.0, seconds);
    case 'bowl':
      return Float64List(n);
  }
  throw ArgumentError('unknown sound $name');
}
