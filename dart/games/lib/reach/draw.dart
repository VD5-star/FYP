import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../core/paint.dart';
import '../core/zones.dart';
import 'angles.dart';
import 'body.dart';
import 'engine.dart';
import 'logic.dart';
import 'points.dart';
import 'targets.dart';

const Skin reachSkin = Skin(
  bg: Color(0xFF1D1F22),
  text: Color(0xFFE2DED8),
  dim: Color(0xFF968D8A),
  faint: Color(0xFF685F5E),
  card: Color(0xFF32363A),
  cardOn: Color(0xFF546064),
  edge: Color(0xFF5A544C),
  accent: Color(0xFFC6D6DA),
  panel: Color(0xFF342F2B),
);

const Color colorBody = Color(0xFFEBEBEB);
const Color colorFaint = Color(0xFF8C8C8C);
const Color colorShadow = Color(0xFF000000);
const Color colorEnding = Color(0xFF7878FF);

const Map<String, Color> bandColors = <String, Color>{
  near: Color(0xFF78DC78),
  far: Color(0xFFF0C85A),
  veryFar: Color(0xFF78A0FF),
  bonus: Color(0xFFFF96FF),
};

const double flashSeconds = 0.35;

List<Offset>? clipToFrame(
    double ax, double ay, double bx, double by, double w, double h) {
  final double dx = bx - ax;
  final double dy = by - ay;
  double t0 = 0.0;
  double t1 = 1.0;
  final List<List<double>> edges = <List<double>>[
    <double>[-dx, ax],
    <double>[dx, w - ax],
    <double>[-dy, ay],
    <double>[dy, h - ay],
  ];
  for (final List<double> e in edges) {
    final double p = e[0];
    final double q = e[1];
    if (p.abs() < 1e-12) {
      if (q < 0) return null;
      continue;
    }
    final double r = q / p;
    if (p < 0) {
      if (r > t1) return null;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return null;
      if (r < t1) t1 = r;
    }
  }
  if (t0 > t1) return null;
  return <Offset>[
    Offset(ax + dx * t0, ay + dy * t0),
    Offset(ax + dx * t1, ay + dy * t1),
  ];
}

void drawBody(Canvas canvas, Size size, Points? points,
    List<double>? visibility, List<bool>? discarded, double unit) {
  if (points == null || visibility == null) return;
  final List<bool> gone =
      discarded ?? List<bool>.filled(points.count, false);
  final double thickness = math.max(2.0, unit * 1.1);

  void stroke(double ax, double ay, double bx, double by) {
    final List<Offset>? seg =
        clipToFrame(ax, ay, bx, by, size.width, size.height);
    if (seg == null) return;
    canvas.drawLine(
      seg[0],
      seg[1],
      Paint()
        ..color = colorShadow
        ..strokeWidth = thickness + 4
        ..strokeCap = StrokeCap.round,
    );
    canvas.drawLine(
      seg[0],
      seg[1],
      Paint()
        ..color = colorBody
        ..strokeWidth = thickness
        ..strokeCap = StrokeCap.round,
    );
  }

  for (final List<String> bone in bones) {
    final int ia = idx[bone[0]]!;
    final int ib = idx[bone[1]]!;
    if (gone[ia] || gone[ib]) continue;
    if (visibility[ia] < drawThreshold || visibility[ib] < drawThreshold) {
      continue;
    }
    if (!points.finite(ia) || !points.finite(ib)) continue;
    stroke(points.xs[ia], points.ys[ia], points.xs[ib], points.ys[ib]);
  }

  final int ls = idx['leftShoulder']!;
  final int rs = idx['rightShoulder']!;
  final int lh = idx['leftHip']!;
  final int rh = idx['rightHip']!;
  final List<int> core = <int>[ls, rs, lh, rh];
  final bool coreOk = core.every((int i) =>
      !gone[i] && visibility[i] >= drawThreshold && points.finite(i));
  if (coreOk) {
    final double msx = (points.xs[ls] + points.xs[rs]) / 2;
    final double msy = (points.ys[ls] + points.ys[rs]) / 2;
    final double mhx = (points.xs[lh] + points.xs[rh]) / 2;
    final double mhy = (points.ys[lh] + points.ys[rh]) / 2;
    stroke(msx, msy, mhx, mhy);
    stroke(points.xs[ls], points.ys[ls], points.xs[rs], points.ys[rs]);
    stroke(points.xs[lh], points.ys[lh], points.xs[rh], points.ys[rh]);
  }

  final List<int> headPoints = <int>[];
  for (final String name in head) {
    final int i = idx[name]!;
    if (!gone[i] && visibility[i] >= drawThreshold && points.finite(i)) {
      headPoints.add(i);
    }
  }
  if (headPoints.isNotEmpty) {
    double minX = points.xs[headPoints.first];
    double maxX = minX;
    double minY = points.ys[headPoints.first];
    double maxY = minY;
    double sumX = 0;
    double sumY = 0;
    for (final int i in headPoints) {
      final double x = points.xs[i];
      final double y = points.ys[i];
      minX = math.min(minX, x);
      maxX = math.max(maxX, x);
      minY = math.min(minY, y);
      maxY = math.max(maxY, y);
      sumX += x;
      sumY += y;
    }
    final Offset c =
        Offset(sumX / headPoints.length, sumY / headPoints.length);
    final double r =
        math.max(unit * 2.2, math.max(maxX - minX, maxY - minY) * 0.9);
    canvas.drawCircle(
      c,
      r,
      Paint()
        ..color = colorShadow
        ..style = PaintingStyle.stroke
        ..strokeWidth = thickness + 4,
    );
    canvas.drawCircle(
      c,
      r,
      Paint()
        ..color = colorBody
        ..style = PaintingStyle.stroke
        ..strokeWidth = thickness,
    );
  }

  const List<String> joints = <String>[
    'leftWrist',
    'rightWrist',
    'leftAnkle',
    'rightAnkle',
    'leftElbow',
    'rightElbow',
    'leftKnee',
    'rightKnee',
  ];
  for (final String name in joints) {
    final int i = idx[name]!;
    if (gone[i] || visibility[i] < drawThreshold) continue;
    if (!points.finite(i)) continue;
    canvas.drawCircle(
      Offset(points.xs[i], points.ys[i]),
      math.max(3.0, unit * 0.8),
      Paint()..color = colorBody,
    );
  }
}

void drawTargets(Canvas canvas, List<Target> targets, double now,
    {double fade = 1.0, Color? override}) {
  for (final Target target in targets) {
    final Color colour =
        override ?? bandColors[target.band] ?? colorBody;
    final double r = target.currentRadius(now) * fade;
    if (r < 2) continue;
    final double life = target.life(now);
    final Rect box = Rect.fromCenter(
      center: Offset(target.x, target.y),
      width: r * 2,
      height: r * 2,
    );
    final double alpha = 0.30 * fade * (0.30 + 0.70 * life);
    canvas.drawRect(box, Paint()..color = colour.withValues(alpha: alpha));
    canvas.drawRect(
      box,
      Paint()
        ..color = colorShadow
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6,
    );
    canvas.drawRect(
      box,
      Paint()
        ..color = colour
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3,
    );
    final String label =
        target.isBonus ? '+${bonusSeconds.toInt()}s' : '${target.points}';
    final double size = math.max(14.0, r * 0.62);
    textCentre(canvas, label, target.x, target.y, size, colour, bold: true);
  }
}

void drawFlash(Canvas canvas, Size size, GameState state, double now) {
  final double? last = state.lastHit;
  if (last == null) return;
  final double age = now - last;
  if (age > flashSeconds) return;
  final double strength = 1.0 - age / flashSeconds;
  final Color colour = bandColors[state.lastBand] ?? colorBody;
  canvas.drawRect(
    Offset.zero & size,
    Paint()..color = colour.withValues(alpha: 0.18 * strength),
  );
}

void drawHud(Canvas canvas, Size size, ReachEngine engine, double now) {
  textAt(canvas, '${engine.state.score}', 24, 34, 34, reachSkin.text,
      bold: true);

  final int streak = engine.state.streak % 5;
  for (int i = 0; i < 5; i++) {
    canvas.drawCircle(
      Offset(30 + i * 18, 96),
      5,
      Paint()
        ..color = i < streak
            ? bandColors[bonus]!
            : const Color(0xFF464646),
    );
  }

  if (engine.mode != timed) return;

  final double left = engine.remaining(now);
  final Color colour =
      left <= 10.0 ? const Color(0xFF6E6EFF) : reachSkin.text;
  final String label = left.toStringAsFixed(1).padLeft(4, '0');
  final TextPainter tp = layoutText(label, 26, colour, bold: true);
  tp.paint(canvas, Offset(size.width - tp.width - 24, 34));

  final double frac =
      math.min(1.0, left / math.max(engine.totalTime(), 1e-6));
  canvas.drawRect(
    Rect.fromLTWH(24, 78, size.width - 48, 8),
    Paint()..color = const Color(0xFF3C3C3C),
  );
  canvas.drawRect(
    Rect.fromLTWH(24, 78, (size.width - 48) * frac, 8),
    Paint()..color = colour,
  );
}

void drawWaiting(Canvas canvas, Size size, String note) {
  textCentre(canvas, note, size.width / 2, size.height - 60, 18, colorFaint);
}

void drawMenu(Canvas canvas, Size size, ReachLogic logic, bool ready) {
  dimScreen(canvas, size, 0.62);
  final double cx = size.width / 2;

  textCentre(canvas, 'reach', cx, 84, 40, reachSkin.text, bold: true);
  textCentre(canvas, 'move your body to meet the targets', cx, 132, 18,
      reachSkin.dim);
  textCentre(canvas, 'stand back so your whole body fits', cx, 162, 16,
      reachSkin.faint);

  final Zone? minutes = logic.zones['minutes'];
  if (minutes != null) {
    _slider(canvas, minutes, logic.minutes, minMinutes, maxMinutes);
    final String word = logic.minutes == 1 ? 'minute' : 'minutes';
    final List<String> words = logic.shapeWords;
    textCentre(canvas, '${logic.minutes} $word', minutes.cx,
        minutes.cy - 44, 26, reachSkin.text);
    textCentre(
        canvas, words[0], minutes.cx, minutes.cy + 52, 19, reachSkin.accent);
    textCentre(
        canvas, words[1], minutes.cx, minutes.cy + 80, 15, reachSkin.faint);
  } else {
    final Zone? t = logic.zones['timed'];
    if (t != null) {
      textCentre(canvas, 'no clock', cx, t.y - 64, 25, reachSkin.text);
      textCentre(canvas, 'stop whenever you want', cx, t.y - 34, 15,
          reachSkin.faint);
    }
  }

  const List<List<String>> modes = <List<String>>[
    <String>[timed, 'timed', 'the clock runs'],
    <String>[calm, 'calm', 'no clock at all'],
  ];
  for (final List<String> row in modes) {
    final Zone? z = logic.zones[row[0]];
    if (z == null) continue;
    final bool on = logic.mode == row[0];
    roundRect(canvas, z, on ? reachSkin.cardOn : reachSkin.card,
        edge: on ? reachSkin.accent : reachSkin.edge);
    textCentre(canvas, row[1], z.cx, z.cy - 8, 19,
        on ? reachSkin.text : reachSkin.dim);
    textCentre(canvas, row[2], z.cx, z.cy + 15, 13,
        on ? reachSkin.dim : reachSkin.faint);
  }

  final Zone? begin = logic.zones['begin'];
  if (begin != null) {
    roundRect(canvas, begin, ready ? reachSkin.cardOn : reachSkin.card,
        edge: ready ? reachSkin.accent : reachSkin.edge);
    textMid(canvas, ready ? 'begin' : 'finding the camera', begin,
        ready ? 20 : 15, ready ? reachSkin.text : reachSkin.faint);
  }

  final Zone? camera = logic.zones['camera'];
  if (camera != null) {
    roundRect(canvas, camera, reachSkin.card, edge: reachSkin.edge);
    textCentre(canvas, 'camera', camera.cx, camera.cy - 9, 13,
        reachSkin.faint);
    textCentre(canvas, logic.cameraLabel, camera.cx, camera.cy + 13, 17,
        reachSkin.text);
  }

  textCentre(canvas, logic.trackingNote, cx, size.height - 46, 14,
      reachSkin.faint);
}

void _slider(Canvas canvas, Zone r, int value, int lo, int hi) {
  final double y = r.cy;
  final double x0 = r.x + sliderPad;
  final double x1 = r.x1 - sliderPad;
  canvas.drawLine(
    Offset(x0, y),
    Offset(x1, y),
    Paint()
      ..color = const Color(0xFF484440)
      ..strokeWidth = 5
      ..strokeCap = StrokeCap.round,
  );
  final double kx = sliderKnob(r, value, lo: lo, hi: hi);
  if (kx > x0) {
    canvas.drawLine(
      Offset(x0, y),
      Offset(kx, y),
      Paint()
        ..color = reachSkin.accent
        ..strokeWidth = 5
        ..strokeCap = StrokeCap.round,
    );
  }
  for (int m = lo; m <= hi; m++) {
    canvas.drawCircle(
      Offset(sliderKnob(r, m, lo: lo, hi: hi), y + 20),
      2,
      Paint()..color = reachSkin.faint,
    );
  }
  canvas.drawCircle(Offset(kx, y), 14, Paint()..color = reachSkin.bg);
  canvas.drawCircle(Offset(kx, y), 12, Paint()..color = reachSkin.accent);
}

void drawPause(Canvas canvas, Size size, Zones zones) {
  dimScreen(canvas, size, 0.55);
  final Zone? panel = zones['panel'];
  if (panel != null) {
    roundRect(canvas, panel, reachSkin.panel,
        radius: 18, edge: reachSkin.edge);
  }
  final double top = panel == null ? size.height / 2 : panel.y + 76;
  textCentre(canvas, 'paused', size.width / 2, top, 30, reachSkin.text);
  textCentre(canvas, 'take your time', size.width / 2, top + 36, 16,
      reachSkin.faint);

  const List<List<Object>> buttons = <List<Object>>[
    <Object>['resume', 'continue', true],
    <Object>['finish', 'finish', false],
  ];
  for (final List<Object> row in buttons) {
    final Zone? z = zones[row[0] as String];
    if (z == null) continue;
    final bool lead = row[2] as bool;
    roundRect(canvas, z, lead ? reachSkin.cardOn : reachSkin.card,
        edge: lead ? reachSkin.accent : reachSkin.edge);
    textMid(canvas, row[1] as String, z, 19,
        lead ? reachSkin.text : reachSkin.dim);
  }
}

void drawSummary(Canvas canvas, Size size, Zones zones, Summary summary,
    String mode, double alpha) {
  dimScreen(canvas, size, 0.62 * alpha.clamp(0.0, 1.0));
  if (alpha < 0.35) return;

  final double cx = size.width / 2;
  final double top = size.height * 0.26;
  textCentre(canvas, 'well done', cx, top, 32, reachSkin.text, bold: true);
  textCentre(canvas, 'nothing is saved, nothing is ranked', cx, top + 34, 15,
      reachSkin.faint);

  final List<List<String>> stats = summaryStats(summary, mode);
  final int n = stats.length;
  final double span = math.min(size.width - 120, 190.0 * n);
  final double step = span / math.max(1, n);
  final double x0 = (size.width - span) / 2 + step / 2;
  final double ys = size.height * 0.46;
  for (int i = 0; i < n; i++) {
    final double sx = x0 + i * step;
    textCentre(canvas, stats[i][1], sx, ys, 32, reachSkin.text);
    textCentre(canvas, stats[i][0], sx, ys + 30, 14, reachSkin.faint);
  }

  final String line = bandBreakdown(summary);
  if (line.isNotEmpty) {
    textCentre(canvas, line, cx, ys + 76, 16, reachSkin.dim);
  }

  const List<List<Object>> buttons = <List<Object>>[
    <Object>['again', 'again', true],
    <Object>['menu', 'main menu', false],
    <Object>['close', 'close', false],
  ];
  for (final List<Object> row in buttons) {
    final Zone? z = zones[row[0] as String];
    if (z == null) continue;
    final bool lead = row[2] as bool;
    roundRect(canvas, z, lead ? reachSkin.cardOn : reachSkin.card,
        edge: lead ? reachSkin.accent : reachSkin.edge);
    textMid(canvas, row[1] as String, z, 18,
        lead ? reachSkin.text : reachSkin.dim);
  }
}

class ReachPainter extends CustomPainter {
  ReachPainter({
    required this.logic,
    required this.now,
    required this.cameraReady,
    required this.repaint,
  }) : super(repaint: repaint);

  final ReachLogic logic;
  final double now;
  final bool cameraReady;
  final Listenable repaint;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = reachSkin.bg);

    drawBody(canvas, size, logic.drawPoints, logic.drawVisibility,
        logic.drawDiscarded, logic.unit());

    if (logic.inMenu) {
      drawMenu(canvas, size, logic, cameraReady);
      return;
    }

    final ReachEngine engine = logic.engine;
    final String phase = engine.state.phase;

    if (phase == finished) {
      drawSummary(canvas, size, logic.zones, logic.summary(), engine.mode,
          logic.summaryAlpha(now));
      return;
    }
    if (phase == ending) {
      drawTargets(canvas, engine.targets, now,
          fade: 1.0 - engine.endingProgress(now), override: colorEnding);
      drawHud(canvas, size, engine, now);
      return;
    }
    if (phase == paused) {
      drawPause(canvas, size, logic.zones);
      return;
    }

    drawTargets(canvas, engine.targets, now);
    drawFlash(canvas, size, engine.state, now);
    drawHud(canvas, size, engine, now);
    if (!logic.tracking) {
      drawWaiting(canvas, size,
          logic.trackingNote.isEmpty
              ? 'step back so I can see you'
              : logic.trackingNote);
    }
    final Zone? stop = logic.zones['stop'];
    if (stop != null) drawStopButton(canvas, stop, reachSkin);
  }

  @override
  bool shouldRepaint(covariant ReachPainter old) => true;
}
