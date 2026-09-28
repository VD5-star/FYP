import 'dart:math' as math;

import 'package:flutter/painting.dart';

const double touchMin = 44.0;
const double sliderPad = 22.0;
const double tapSlop = 8.0;

class Zone {
  const Zone(this.x, this.y, this.w, this.h);

  final double x;
  final double y;
  final double w;
  final double h;

  double get x1 => x + w;
  double get y1 => y + h;
  double get cx => x + w / 2;
  double get cy => y + h / 2;

  bool get touchable => w >= touchMin && h >= touchMin;

  bool hits(double px, double py) =>
      px >= x && px < x1 && py >= y && py < y1;

  bool overlaps(Zone o) => x < o.x1 && o.x < x1 && y < o.y1 && o.y < y1;

  Rect get rect => Rect.fromLTWH(x, y, w, h);
  Offset get centre => Offset(cx, cy);
}

class Zones {
  final Map<String, Zone> items = <String, Zone>{};

  void clear() => items.clear();

  void add(String key, Zone r) => items[key] = r;

  bool has(String key) => items.containsKey(key);

  Zone? operator [](String key) => items[key];

  Iterable<String> get keys => items.keys;

  String? at(double x, double y) {
    String? found;
    for (final MapEntry<String, Zone> e in items.entries) {
      if (e.value.hits(x, y)) found = e.key;
    }
    return found;
  }
}

int sliderValue(Zone r, double x, {int lo = 0, int hi = 100}) {
  final double inner = math.max(1.0, r.w - 2 * sliderPad);
  final double f = (x - (r.x + sliderPad)) / inner;
  final double c = f.clamp(0.0, 1.0);
  return (lo + (hi - lo) * c).round();
}

double sliderKnob(Zone r, num value, {int lo = 0, int hi = 100}) {
  final double inner = math.max(1.0, r.w - 2 * sliderPad);
  final double span = (hi - lo).toDouble();
  final double f = span <= 0 ? 0.0 : (value - lo) / span;
  return r.x + sliderPad + inner * f.clamp(0.0, 1.0);
}
