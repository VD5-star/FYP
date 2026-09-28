import 'dart:math' as math;
import 'dart:typed_data';

const int sampleRate = 44100;

const int volMin = 0;
const int volMax = 100;
const int volDefault = 55;


double volumeCurve(int v) {
  if (v <= volMin) return 0.0;
  final double f = v.clamp(volMin, volMax) / volMax;
  return math.pow(f, 2.2).toDouble();
}

class Rng {
  Rng(int seed) : _s = (seed == 0 ? 0x9E3779B9 : seed) & 0xFFFFFFFF;

  int _s;

  int nextInt() {
    _s ^= (_s << 13) & 0xFFFFFFFF;
    _s ^= _s >> 17;
    _s ^= (_s << 5) & 0xFFFFFFFF;
    _s &= 0xFFFFFFFF;
    return _s;
  }

  double nextDouble() => nextInt() / 0x100000000;

  
  double normal() {
    final double u1 = math.max(1e-9, nextDouble());
    final double u2 = nextDouble();
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
  }
}

Float32List smoothBox(Float32List x, int k) {
  if (k < 2) return x;
  final int n = x.length;
  final Float32List out = Float32List(n);
  double acc = 0;
  final int half = k ~/ 2;
  for (int i = 0; i < n + half; i++) {
    if (i < n) acc += x[i];
    if (i - k >= 0) acc -= x[i - k];
    final int o = i - half;
    if (o >= 0 && o < n) {
      final int lo = i - k + 1 < 0 ? 0 : i - k + 1;
      final int hi = i < n - 1 ? i : n - 1;
      final int span = hi - lo + 1;
      out[o] = acc / (span < 1 ? 1 : span);
    }
  }
  return out;
}

void normalise(Float32List x) {
  double peak = 0;
  for (final double v in x) {
    final double a = v.abs();
    if (a > peak) peak = a;
  }
  if (peak > 0) {
    for (int i = 0; i < x.length; i++) {
      x[i] = x[i] / peak;
    }
  }
}


Float32List shapedEnvelope(int n, double decay,
    {double attackMs = 9.0, double tailFrac = 0.35}) {
  final Float32List env = Float32List(n);
  int a = math.max(4, (sampleRate * attackMs / 1000.0).round());
  a = math.min(a, n ~/ 3);
  for (int i = 0; i < n; i++) {
    final double t = i / sampleRate;
    double e = math.exp(-t * decay);
    if (i < a) e *= 0.5 - 0.5 * math.cos(math.pi * i / a);
    env[i] = e;
  }
  final int tail = math.max(4, (n * tailFrac).round());
  for (int i = 0; i < tail; i++) {
    final double f = 1.0 - i / tail;
    env[n - tail + i] *= f * f;
  }
  return env;
}

Float32List clickTone({
  double freq = 247.0,
  double ms = 150.0,
  int seed = 0,
  double attackMs = 9.0,
  double decay = 7.5,
  bool withThud = true,
}) {
  final int n = math.max(16, (sampleRate * ms / 1000.0).round());
  final Float32List body = Float32List(n);
  const List<List<double>> parts = <List<double>>[
    <double>[1.0, 1.00, 5.5],
    <double>[2.0, 0.16, 9.0],
    <double>[3.01, 0.05, 14.0],
  ];
  for (final List<double> p in parts) {
    final double mult = p[0], amp = p[1], rate = p[2];
    for (int i = 0; i < n; i++) {
      final double t = i / sampleRate;
      body[i] += math.sin(2 * math.pi * freq * mult * t) *
          amp *
          math.exp(-t * rate);
    }
  }

  if (withThud) {
    final Rng rng = Rng(seed + 1);
    Float32List thud = Float32List(n);
    for (int i = 0; i < n; i++) {
      thud[i] = rng.normal();
    }
    final int k = math.max(4, (sampleRate / math.max(60.0, freq * 1.4)).round());
    thud = smoothBox(smoothBox(thud, k), k);
    for (int i = 0; i < n; i++) {
      final double t = i / sampleRate;
      body[i] += thud[i] * 0.22 * math.exp(-t * 34.0);
    }
  }

  final Float32List env = shapedEnvelope(n, decay, attackMs: attackMs);
  for (int i = 0; i < n; i++) {
    body[i] *= env[i];
  }
  body[0] = 0;
  body[n - 1] = 0;
  normalise(body);
  return body;
}


Float32List toStereo(Float32List mono, {double pan = 0.0, double gain = 1.0}) {
  final double p = pan.clamp(-1.0, 1.0);
  final double l = gain * math.sqrt(0.5 * (1.0 - p));
  final double r = gain * math.sqrt(0.5 * (1.0 + p));
  final Float32List out = Float32List(mono.length * 2);
  for (int i = 0; i < mono.length; i++) {
    out[i * 2] = mono[i] * l;
    out[i * 2 + 1] = mono[i] * r;
  }
  return out;
}


Uint8List wavFromFloat(Float32List samples, {int channels = 2}) {
  final int frames = samples.length ~/ channels;
  final int dataBytes = frames * channels * 2;
  final ByteData out = ByteData(44 + dataBytes);
  void tag(int off, String s) {
    for (int i = 0; i < s.length; i++) {
      out.setUint8(off + i, s.codeUnitAt(i));
    }
  }

  tag(0, 'RIFF');
  out.setUint32(4, 36 + dataBytes, Endian.little);
  tag(8, 'WAVE');
  tag(12, 'fmt ');
  out.setUint32(16, 16, Endian.little);
  out.setUint16(20, 1, Endian.little);
  out.setUint16(22, channels, Endian.little);
  out.setUint32(24, sampleRate, Endian.little);
  out.setUint32(28, sampleRate * channels * 2, Endian.little);
  out.setUint16(32, channels * 2, Endian.little);
  out.setUint16(34, 16, Endian.little);
  tag(36, 'data');
  out.setUint32(40, dataBytes, Endian.little);

  int off = 44;
  for (int i = 0; i < frames * channels; i++) {
    final double v = samples[i].clamp(-1.0, 1.0);
    out.setInt16(off, (v * 32767).round(), Endian.little);
    off += 2;
  }
  return out.buffer.asUint8List();
}
