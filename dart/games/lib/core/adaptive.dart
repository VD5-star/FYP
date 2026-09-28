import 'dart:math' as math;

import 'package:flutter/widgets.dart';

class Adaptive {
  const Adaptive._();

  static double shortest(BuildContext c) =>
      MediaQuery.sizeOf(c).shortestSide;

  static double longest(BuildContext c) => MediaQuery.sizeOf(c).longestSide;

  static bool isPhone(BuildContext c) => shortest(c) < 600;

  static bool isTablet(BuildContext c) => shortest(c) >= 600;

  static bool isSmallPhone(BuildContext c) => shortest(c) < 360;

  static double scale(BuildContext c) {
    final double s = shortest(c) / 400.0;
    return s.clamp(0.85, 1.35);
  }

  static double font(BuildContext c, double base) =>
      (base * scale(c)).clamp(10.0, 32.0);

  static EdgeInsets safePad(BuildContext c) => MediaQuery.paddingOf(c);

  static double availableWidth(BuildContext c, double pad) =>
      MediaQuery.sizeOf(c).width - pad * 2;

  static double gridForWidth(double w) {
    if (w < 360) return 3;
    if (w < 420) return 4;
    if (w < 520) return 5;
    return 6;
  }

  static int jigsawSideForWidth(double w, double h) {
    final double s = math.min(w, h);
    if (s < 360) return 3;
    if (s < 500) return 4;
    if (s < 650) return 5;
    if (s < 800) return 6;
    if (s < 1000) return 8;
    return 10;
  }

  static double touchTarget(BuildContext c) {
    final double base = isSmallPhone(c) ? 44.0 : 48.0;
    return base * (isTablet(c) ? 1.12 : 1.0);
  }

  static double panelWidth(BuildContext c, double maxW) {
    final double w = MediaQuery.sizeOf(c).width;
    return math.min(maxW, w - 32.0);
  }
}
