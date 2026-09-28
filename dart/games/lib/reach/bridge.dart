import 'package:engine_body/engine_body.dart';

import 'angles.dart';
import 'points.dart';

class FrameScale {
  const FrameScale(this.width, this.height);

  final double width;
  final double height;

  double get unit => height;

  bool get usable => width > 1e-9 && height > 1e-9;

  double toIsoX(double px) => px / unit;

  double toIsoY(double py) => py / unit;

  double toPixelX(double ix) => ix * unit;

  double toPixelY(double iy) => iy * unit;

  double get isoWidth => width / unit;

  double get isoHeight => height / unit;
}

Points pixelsToIso(Points pixels, FrameScale scale) {
  final Points out = Points(pixels.count);
  final double u = scale.unit;
  for (int i = 0; i < pixels.count; i++) {
    out.xs[i] = pixels.xs[i] / u;
    out.ys[i] = pixels.ys[i] / u;
  }
  return out;
}

Points isoToPixels(Points iso, FrameScale scale) {
  final Points out = Points(iso.count);
  final double u = scale.unit;
  for (int i = 0; i < iso.count; i++) {
    out.xs[i] = iso.xs[i] * u;
    out.ys[i] = iso.ys[i] * u;
  }
  return out;
}

BodyFrame frameFromIso(
  Points iso,
  List<double> visibility,
  FrameScale scale,
  double timestamp, {
  bool mirrored = false,
}) {
  final double sx = scale.isoWidth;
  final double sy = scale.isoHeight;
  final List<BodyLandmark> marks = <BodyLandmark>[];
  final List<LandmarkType> types = LandmarkType.values;
  final int n = iso.count < types.length ? iso.count : types.length;

  for (int i = 0; i < n; i++) {
    final double v = i < visibility.length ? visibility[i] : 0.0;
    marks.add(BodyLandmark(
      type: types[i],
      x: sx <= 1e-9 ? iso.xs[i] : iso.xs[i] / sx,
      y: sy <= 1e-9 ? iso.ys[i] : iso.ys[i] / sy,
      z: 0.0,
      likelihood: v,
      inFrameLikelihood: v,
    ));
  }

  return BodyFrame(
    timestamp: DateTime.fromMicrosecondsSinceEpoch(
        (timestamp * 1e6).round(),
        isUtc: true),
    landmarks: marks,
    framing: const FramingGate().assess(marks),
    isMirrored: mirrored,
  );
}

Points isoFromFrame(BodyFrame frame, FrameScale scale, int count) {
  final double sx = scale.isoWidth;
  final double sy = scale.isoHeight;
  final Points out = Points.nan(count);
  for (final BodyLandmark mark in frame.landmarks) {
    final int i = mark.type.index;
    if (i >= count) continue;
    out.xs[i] = mark.x * sx;
    out.ys[i] = mark.y * sy;
  }
  return out;
}

List<double> confidenceFromFrame(BodyFrame frame, int count) {
  final List<double> out = List<double>.filled(count, 0.0);
  for (final BodyLandmark mark in frame.landmarks) {
    final int i = mark.type.index;
    if (i >= count) continue;
    out[i] = mark.likelihood;
  }
  return out;
}

double isoTorso(Points iso) {
  final int ls = idx['leftShoulder']!;
  final int rs = idx['rightShoulder']!;
  final int lh = idx['leftHip']!;
  final int rh = idx['rightHip']!;
  final double sx = (iso.xs[ls] + iso.xs[rs]) / 2;
  final double sy = (iso.ys[ls] + iso.ys[rs]) / 2;
  final double hx = (iso.xs[lh] + iso.xs[rh]) / 2;
  final double hy = (iso.ys[lh] + iso.ys[rh]) / 2;
  return hypot(sx - hx, sy - hy);
}
