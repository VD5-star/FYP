import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/body_frame.dart';
import '../core/body_landmark.dart';
import '../core/framing.dart';
import '../core/landmark_type.dart';

const _angleJoints = <(LandmarkType, LandmarkType, LandmarkType)>[
  (LandmarkType.leftElbow, LandmarkType.leftShoulder, LandmarkType.leftWrist),
  (
    LandmarkType.rightElbow,
    LandmarkType.rightShoulder,
    LandmarkType.rightWrist,
  ),
  (LandmarkType.leftShoulder, LandmarkType.leftHip, LandmarkType.leftElbow),
  (LandmarkType.rightShoulder, LandmarkType.rightHip, LandmarkType.rightElbow),
  (LandmarkType.leftKnee, LandmarkType.leftHip, LandmarkType.leftAnkle),
  (LandmarkType.rightKnee, LandmarkType.rightHip, LandmarkType.rightAnkle),
  (LandmarkType.leftHip, LandmarkType.leftShoulder, LandmarkType.leftKnee),
  (LandmarkType.rightHip, LandmarkType.rightShoulder, LandmarkType.rightKnee),
];

const _headLandmarks = <LandmarkType>{
  LandmarkType.nose,
  LandmarkType.leftEye,
  LandmarkType.rightEye,
  LandmarkType.leftEyeInner,
  LandmarkType.rightEyeInner,
  LandmarkType.leftEyeOuter,
  LandmarkType.rightEyeOuter,
  LandmarkType.leftEar,
  LandmarkType.rightEar,
  LandmarkType.leftMouth,
  LandmarkType.rightMouth,
};

class SkeletonPainter extends CustomPainter {
  SkeletonPainter({
    required this.frame,
    this.threshold = 0.5,
    this.color = const Color(0xFF4CD3C2),
    this.lowConfidenceColor = const Color(0xFFFFA726),
    this.showAngles = true,
    this.showBounds = true,
  });

  final BodyFrame frame;

  final double threshold;

  final Color color;
  final Color lowConfidenceColor;
  final bool showAngles;
  final bool showBounds;

  @override
  void paint(Canvas canvas, Size size) {
    if (!frame.hasSubject) return;

    final byType = {for (final l in frame.landmarks) l.type: l};

    final unit = math.min(size.width, size.height) / 100;

    if (showBounds) _paintBounds(canvas, size, byType, unit);
    _paintBones(canvas, size, byType, unit);
    _paintHead(canvas, size, byType, unit);
    _paintJoints(canvas, size, byType, unit);
    if (showAngles) _paintAngles(canvas, size, byType, unit);
  }

  void _paintBones(
    Canvas canvas,
    Size size,
    Map<LandmarkType, BodyLandmark> byType,
    double unit,
  ) {
    for (final (a, b) in skeletonBones) {
      if (_headLandmarks.contains(a) && _headLandmarks.contains(b)) continue;

      final from = byType[a];
      final to = byType[b];
      if (from == null || to == null) continue;

      final p1 = _toCanvas(from, size);
      final p2 = _toCanvas(to, size);

      final reliable = from.isReliable(threshold) && to.isReliable(threshold);
      final confidence = math.min(_confidenceOf(from), _confidenceOf(to));
      final alpha = (0.2 + confidence * 0.8).clamp(0.0, 1.0);
      final width = _boneWidth(a, b) * unit;

      canvas.drawLine(
        p1,
        p2,
        Paint()
          ..color = Colors.black.withValues(alpha: alpha * 0.55)
          ..strokeWidth = width + unit * 0.55
          ..strokeCap = StrokeCap.round,
      );

      canvas.drawLine(
        p1,
        p2,
        Paint()
          ..color = (reliable ? color : lowConfidenceColor).withValues(
            alpha: alpha,
          )
          ..strokeWidth = width
          ..strokeCap = StrokeCap.round,
      );
    }
  }

  double _boneWidth(LandmarkType a, LandmarkType b) {
    const trunk = {
      LandmarkType.leftShoulder,
      LandmarkType.rightShoulder,
      LandmarkType.leftHip,
      LandmarkType.rightHip,
    };
    if (trunk.contains(a) && trunk.contains(b)) return 1.5;

    const extremity = {
      LandmarkType.leftThumb,
      LandmarkType.rightThumb,
      LandmarkType.leftIndex,
      LandmarkType.rightIndex,
      LandmarkType.leftPinky,
      LandmarkType.rightPinky,
      LandmarkType.leftHeel,
      LandmarkType.rightHeel,
      LandmarkType.leftFootIndex,
      LandmarkType.rightFootIndex,
    };
    if (extremity.contains(a) || extremity.contains(b)) return 0.7;

    return 1.1;
  }

  void _paintHead(
    Canvas canvas,
    Size size,
    Map<LandmarkType, BodyLandmark> byType,
    double unit,
  ) {
    final points = <Offset>[];
    var reliable = false;
    for (final type in _headLandmarks) {
      final landmark = byType[type];
      if (landmark == null) continue;
      if (landmark.inFrameLikelihood < 0.2) continue;
      points.add(_toCanvas(landmark, size));
      if (landmark.isReliable(threshold)) reliable = true;
    }
    if (points.length < 3) return;

    var rect = _boundsOf(points);

    final padX = rect.width * 0.35;
    final padY = rect.height * 0.55;
    rect = Rect.fromLTRB(
      rect.left - padX,
      rect.top - padY,
      rect.right + padX,
      rect.bottom + padY * 0.45,
    );

    final paintColor = reliable ? color : lowConfidenceColor;
    final rrect = RRect.fromRectAndRadius(rect, Radius.circular(unit * 0.8));

    canvas.drawRRect(
      rrect,
      Paint()
        ..color = Colors.black.withValues(alpha: 0.5)
        ..style = PaintingStyle.stroke
        ..strokeWidth = unit * 0.9,
    );
    canvas.drawRRect(
      rrect,
      Paint()
        ..color = paintColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = unit * 0.5,
    );

    final tick = math.min(rect.width, rect.height) * 0.22;
    final corner = Paint()
      ..color = paintColor
      ..strokeWidth = unit * 0.7
      ..strokeCap = StrokeCap.round;

    for (final (origin, dx, dy) in [
      (rect.topLeft, 1.0, 1.0),
      (rect.topRight, -1.0, 1.0),
      (rect.bottomLeft, 1.0, -1.0),
      (rect.bottomRight, -1.0, -1.0),
    ]) {
      canvas.drawLine(origin, origin.translate(tick * dx, 0), corner);
      canvas.drawLine(origin, origin.translate(0, tick * dy), corner);
    }
  }

  void _paintJoints(
    Canvas canvas,
    Size size,
    Map<LandmarkType, BodyLandmark> byType,
    double unit,
  ) {
    for (final landmark in frame.landmarks) {
      if (_headLandmarks.contains(landmark.type)) continue;
      if (landmark.inFrameLikelihood < 0.15) continue;

      final centre = _toCanvas(landmark, size);
      final reliable = landmark.isReliable(threshold);
      final alpha = (0.25 + _confidenceOf(landmark) * 0.75).clamp(0.0, 1.0);
      final major = _majorJoints.contains(landmark.type);
      final radius = (major ? 1.5 : 0.85) * unit;
      final paintColor = reliable ? color : lowConfidenceColor;

      canvas.drawCircle(
        centre,
        radius + unit * 0.28,
        Paint()..color = Colors.black.withValues(alpha: alpha * 0.6),
      );
      canvas.drawCircle(
        centre,
        radius,
        Paint()..color = paintColor.withValues(alpha: alpha),
      );
      if (major) {
        canvas.drawCircle(
          centre,
          radius * 0.42,
          Paint()..color = Colors.white.withValues(alpha: alpha),
        );
      }
    }
  }

  static const _majorJoints = <LandmarkType>{
    LandmarkType.leftShoulder,
    LandmarkType.rightShoulder,
    LandmarkType.leftElbow,
    LandmarkType.rightElbow,
    LandmarkType.leftWrist,
    LandmarkType.rightWrist,
    LandmarkType.leftHip,
    LandmarkType.rightHip,
    LandmarkType.leftKnee,
    LandmarkType.rightKnee,
    LandmarkType.leftAnkle,
    LandmarkType.rightAnkle,
  };

  void _paintAngles(
    Canvas canvas,
    Size size,
    Map<LandmarkType, BodyLandmark> byType,
    double unit,
  ) {
    for (final (vertex, a, b) in _angleJoints) {
      final angle = frame.jointAngle(vertex, a, b, threshold: threshold);
      if (angle == null) continue;

      final landmark = byType[vertex];
      if (landmark == null) continue;

      final bent = angle < _bentThreshold;
      final centre = _toCanvas(landmark, size);

      if (bent) {
        canvas.drawCircle(
          centre,
          unit * 2.8,
          Paint()
            ..color = _bentColor.withValues(alpha: 0.9)
            ..style = PaintingStyle.stroke
            ..strokeWidth = unit * 0.45,
        );
      }

      _paintLabel(
        canvas,
        '${angle.round()}',
        centre.translate(unit * 3.2, -unit * 1.6),
        unit,
        bent ? _bentColor : Colors.white,
      );
    }
  }

  static const _bentThreshold = 160.0;
  static const _bentColor = Color(0xFFFFD54F);

  void _paintLabel(
    Canvas canvas,
    String text,
    Offset at,
    double unit,
    Color textColor,
  ) {
    final painter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: textColor,
          fontSize: unit * 2.6,
          fontWeight: FontWeight.w600,
          shadows: const [
            Shadow(blurRadius: 3, color: Colors.black),
            Shadow(blurRadius: 6, color: Colors.black),
          ],
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    painter.paint(canvas, at);
  }

  void _paintBounds(
    Canvas canvas,
    Size size,
    Map<LandmarkType, BodyLandmark> byType,
    double unit,
  ) {
    final points = <Offset>[];
    for (final landmark in frame.landmarks) {
      if (!landmark.isReliable(threshold)) continue;
      points.add(_toCanvas(landmark, size));
    }
    if (points.length < 4) return;

    final rect = _boundsOf(points).inflate(unit * 1.5);
    final ok = frame.framing.framing == Framing.full;
    final paintColor = (ok ? color : lowConfidenceColor).withValues(alpha: 0.5);

    final length = math.min(rect.width, rect.height) * 0.16;
    final paint = Paint()
      ..color = paintColor
      ..strokeWidth = unit * 0.5
      ..strokeCap = StrokeCap.round;

    for (final (origin, dx, dy) in [
      (rect.topLeft, 1.0, 1.0),
      (rect.topRight, -1.0, 1.0),
      (rect.bottomLeft, 1.0, -1.0),
      (rect.bottomRight, -1.0, -1.0),
    ]) {
      canvas.drawLine(origin, origin.translate(length * dx, 0), paint);
      canvas.drawLine(origin, origin.translate(0, length * dy), paint);
    }
  }

  Rect _boundsOf(List<Offset> points) {
    var left = points.first.dx;
    var top = points.first.dy;
    var right = left;
    var bottom = top;
    for (final p in points) {
      if (p.dx < left) left = p.dx;
      if (p.dx > right) right = p.dx;
      if (p.dy < top) top = p.dy;
      if (p.dy > bottom) bottom = p.dy;
    }
    return Rect.fromLTRB(left, top, right, bottom);
  }

  double _confidenceOf(BodyLandmark l) => l.likelihood * l.inFrameLikelihood;

  Offset _toCanvas(BodyLandmark landmark, Size size) =>
      Offset(landmark.x * size.width, landmark.y * size.height);

  @override
  bool shouldRepaint(SkeletonPainter oldDelegate) =>
      !identical(oldDelegate.frame, frame) ||
      oldDelegate.threshold != threshold ||
      oldDelegate.showAngles != showAngles ||
      oldDelegate.showBounds != showBounds ||
      oldDelegate.color != color;
}

({Offset from, Offset to})? clipToFrame(Offset a, Offset b, Size size) {
  final dx = b.dx - a.dx;
  final dy = b.dy - a.dy;

  var t0 = 0.0;
  var t1 = 1.0;

  for (final (p, q) in <(double, double)>[
    (-dx, a.dx),
    (dx, size.width - a.dx),
    (-dy, a.dy),
    (dy, size.height - a.dy),
  ]) {
    if (p.abs() < 1e-12) {
      if (q < 0) return null;
      continue;
    }
    final r = q / p;
    if (p < 0) {
      if (r > t1) return null;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return null;
      if (r < t1) t1 = r;
    }
  }

  if (t0 > t1) return null;
  return (
    from: Offset(a.dx + dx * t0, a.dy + dy * t0),
    to: Offset(a.dx + dx * t1, a.dy + dy * t1),
  );
}

String framingAdviceText(FramingAdvice advice) => switch (advice) {
  FramingAdvice.none => '',
  FramingAdvice.moveBack => 'Move back so your legs are in view',
  FramingAdvice.stepIntoView => 'Step into view of the camera',
  FramingAdvice.faceTheCamera => 'Turn to face the camera',
};
