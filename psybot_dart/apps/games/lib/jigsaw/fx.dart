import 'dart:math' as math;
import 'dart:typed_data';

const double sparkLife = 0.62;
const int sparkCount = 14;
const double sparkSpeed = 170.0;
const double sparkDrag = 3.1;
const double gravity = 210.0;

const double ringLife = 0.46;
const double ringGrow = 2.1;
const int maxRings = 6;

const int dustCount = 5;
const double dustLife = 0.9;

const int maxParticles = 260;

class Particles {
  Particles({this.cap = maxParticles})
      : px = Float32List(cap),
        py = Float32List(cap),
        vx = Float32List(cap),
        vy = Float32List(cap),
        life = Float32List(cap),
        full = Float32List(cap),
        size = Float32List(cap),
        col = Float32List(cap * 3),
        grav = Float32List(cap);

  final int cap;
  int n = 0;
  final Float32List px;
  final Float32List py;
  final Float32List vx;
  final Float32List vy;
  final Float32List life;
  final Float32List full;
  final Float32List size;
  final Float32List col;
  final Float32List grav;

  void clear() => n = 0;

  int _room(int k) {
    if (n + k > cap) {
      final int drop = math.min(n, n + k - cap);
      final int keep = n - drop;
      for (final Float32List a in <Float32List>[
        px,
        py,
        vx,
        vy,
        life,
        full,
        size,
        grav
      ]) {
        a.setRange(0, keep, a, drop);
      }
      col.setRange(0, keep * 3, col, drop * 3);
      n = keep;
    }
    final int take = math.min(k, cap - n);
    final int s = n;
    n += take;
    return s;
  }

  void emit(double x, double y, int count, math.Random rng,
      List<double> colour,
      {double speed = sparkSpeed,
      double life0 = sparkLife,
      double gravityA = gravity,
      double sizeA = 2.2,
      double spread = 1.0}) {
    if (count <= 0) return;
    final int s = _room(count);
    final int k = n - s;
    if (k <= 0) return;
    for (int i = s; i < n; i++) {
      final double ang = rng.nextDouble() * 2 * math.pi;
      final double mag =
          math.pow(0.25 + rng.nextDouble() * 0.75, 0.6).toDouble() *
              speed *
              spread;
      final double jr = math.sqrt(rng.nextDouble()) *
          math.max(1.0, sizeA * 2.2);
      final double ja = rng.nextDouble() * 2 * math.pi;
      px[i] = x + math.cos(ja) * jr;
      py[i] = y + math.sin(ja) * jr;
      vx[i] = math.cos(ang) * mag;
      vy[i] = math.sin(ang) * mag - speed * 0.25;
      final double lf = life0 * (0.65 + rng.nextDouble() * 0.60);
      life[i] = lf;
      full[i] = lf;
      size[i] = sizeA * (0.6 + rng.nextDouble() * 0.9);
      grav[i] = gravityA;
      final double tint = 0.82 + rng.nextDouble() * 0.36;
      for (int c = 0; c < 3; c++) {
        col[i * 3 + c] = (colour[c] * tint).clamp(0.0, 255.0);
      }
    }
  }

  void step(double dt) {
    if (n == 0) return;
    final double d0 = dt.clamp(0.0, 0.05);
    final double d = math.exp(-sparkDrag * d0);
    int k = 0;
    for (int i = 0; i < n; i++) {
      vx[i] *= d;
      vy[i] = vy[i] * d + grav[i] * d0;
      px[i] += vx[i] * d0;
      py[i] += vy[i] * d0;
      life[i] -= d0;
      if (life[i] <= 0.0) continue;
      if (k != i) {
        px[k] = px[i];
        py[k] = py[i];
        vx[k] = vx[i];
        vy[k] = vy[i];
        life[k] = life[i];
        full[k] = full[i];
        size[k] = size[i];
        grav[k] = grav[i];
        col[k * 3] = col[i * 3];
        col[k * 3 + 1] = col[i * 3 + 1];
        col[k * 3 + 2] = col[i * 3 + 2];
      }
      k++;
    }
    n = k;
  }

  double alphaOf(int i) =>
      (life[i] / math.max(full[i], 1e-6)).clamp(0.0, 1.0);

  double drawSize(int i) => size[i] * (0.35 + alphaOf(i) * 0.65);
}

class Ring {
  Ring(this.x, this.y, this.r0, this.life, this.full, this.colour);
  final double x;
  final double y;
  final double r0;
  double life;
  final double full;
  final List<double> colour;
}

class Effects {
  Effects({int seed = 7}) : rng = math.Random(seed);

  final math.Random rng;
  final Particles parts = Particles();
  final List<Ring> rings = <Ring>[];

  void clear() {
    parts.clear();
    rings.clear();
  }

  void place(double x, double y, double radius,
      {List<double> colour = const <double>[196, 224, 236],
      double strength = 1.0}) {
    parts.emit(x, y, (sparkCount * strength).toInt(), rng, colour,
        speed: sparkSpeed * (0.8 + strength * 0.3),
        sizeA: 2.1 * strength);
    rings.add(Ring(x, y, radius * 0.45, ringLife, ringLife, colour));
    if (rings.length > maxRings) {
      rings.removeRange(0, rings.length - maxRings);
    }
  }

  void lift(double x, double y) {
    parts.emit(x, y, dustCount, rng, const <double>[150, 150, 146],
        speed: 52.0,
        life0: dustLife,
        gravityA: -26.0,
        sizeA: 1.5,
        spread: 0.7);
  }

  void finish(int w, int h, {int n = 90}) {
    const int spots = 7;
    for (int i = 0; i < spots; i++) {
      final double x = w * 0.18 + rng.nextDouble() * w * 0.64;
      final double y = h * 0.22 + rng.nextDouble() * h * 0.44;
      parts.emit(x, y, math.max(4, n ~/ spots), rng,
          const <double>[188, 206, 214],
          speed: 190.0, life0: 1.6, gravityA: 110.0, sizeA: 2.0);
    }
  }

  void step(double dt) {
    parts.step(dt);
    if (rings.isEmpty) return;
    rings.removeWhere((Ring r) {
      r.life -= dt;
      return r.life <= 0;
    });
  }

  bool get busy => parts.n > 0 || rings.isNotEmpty;
}
