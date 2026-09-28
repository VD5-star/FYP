import 'dart:math' as math;
import 'dart:typed_data';

int pow2Floor(int n) {
  int p = 1;
  while (p * 2 <= n) {
    p *= 2;
  }
  return p;
}

void _fftInPlace(Float64List re, Float64List im, bool inverse) {
  final int n = re.length;
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      final double tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      final double ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    final double ang = 2 * math.pi / len * (inverse ? 1 : -1);
    final double wr = math.cos(ang);
    final double wi = math.sin(ang);
    final int half = len >> 1;
    for (int i = 0; i < n; i += len) {
      double cr = 1.0;
      double ci = 0.0;
      for (int k = 0; k < half; k++) {
        final int a = i + k;
        final int b = a + half;
        final double xr = re[b] * cr - im[b] * ci;
        final double xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
        final double ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  if (inverse) {
    for (int i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

class Spectrum {
  Spectrum(this.re, this.im);

  final Float64List re;
  final Float64List im;

  int get length => re.length;
}

Float64List rfftFreq(int n, double d) {
  final int half = n ~/ 2 + 1;
  final Float64List f = Float64List(half);
  for (int i = 0; i < half; i++) {
    f[i] = i / (n * d);
  }
  return f;
}

Spectrum rfft(Float64List x) {
  final int n = x.length;
  final Float64List re = Float64List(n);
  final Float64List im = Float64List(n);
  for (int i = 0; i < n; i++) {
    re[i] = x[i];
  }
  _fftInPlace(re, im, false);
  final int half = n ~/ 2 + 1;
  return Spectrum(
    Float64List.fromList(re.sublist(0, half)),
    Float64List.fromList(im.sublist(0, half)),
  );
}

Float64List irfft(Spectrum s, int n) {
  final Float64List re = Float64List(n);
  final Float64List im = Float64List(n);
  final int half = n ~/ 2 + 1;
  for (int i = 0; i < half && i < s.length; i++) {
    re[i] = s.re[i];
    im[i] = s.im[i];
  }
  for (int i = 1; i <= n - half; i++) {
    re[n - i] = s.re[i];
    im[n - i] = -s.im[i];
  }
  _fftInPlace(re, im, true);
  return re;
}

void fftComplex(Float64List re, Float64List im, {bool inverse = false}) {
  _fftInPlace(re, im, inverse);
}

Float64List analyticEnvelope(Float64List x) {
  final int n = x.length;
  final Float64List re = Float64List(n);
  final Float64List im = Float64List(n);
  for (int i = 0; i < n; i++) {
    re[i] = x[i];
  }
  _fftInPlace(re, im, false);
  final int half = n ~/ 2;
  for (int i = 1; i < half; i++) {
    re[i] *= 2;
    im[i] *= 2;
  }
  for (int i = half + 1; i < n; i++) {
    re[i] = 0;
    im[i] = 0;
  }
  _fftInPlace(re, im, true);
  final Float64List env = Float64List(n);
  for (int i = 0; i < n; i++) {
    env[i] = math.sqrt(re[i] * re[i] + im[i] * im[i]);
  }
  return env;
}

Float64List magnitudeSpectrum(Float64List x) {
  final Spectrum s = rfft(x);
  final Float64List m = Float64List(s.length);
  for (int i = 0; i < s.length; i++) {
    m[i] = math.sqrt(s.re[i] * s.re[i] + s.im[i] * s.im[i]);
  }
  return m;
}
