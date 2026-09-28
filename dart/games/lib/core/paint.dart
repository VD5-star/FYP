import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'zones.dart';

class Skin {
  const Skin({
    required this.bg,
    required this.text,
    required this.dim,
    required this.faint,
    required this.card,
    required this.cardOn,
    required this.edge,
    required this.accent,
    required this.panel,
  });

  final Color bg;
  final Color text;
  final Color dim;
  final Color faint;
  final Color card;
  final Color cardOn;
  final Color edge;
  final Color accent;
  final Color panel;
}

TextPainter layoutText(String s, double size, Color c, {bool bold = false}) {
  final TextPainter tp = TextPainter(
    text: TextSpan(
      text: s,
      style: TextStyle(
        color: c,
        fontSize: size,
        height: 1.0,
        fontWeight: bold ? FontWeight.w600 : FontWeight.w400,
        letterSpacing: 0.2,
      ),
    ),
    textDirection: TextDirection.ltr,
  );
  tp.layout();
  return tp;
}

void textAt(Canvas canvas, String s, double x, double y, double size, Color c,
    {bool bold = false}) {
  layoutText(s, size, c, bold: bold).paint(canvas, Offset(x, y));
}

void textCentre(Canvas canvas, String s, double cx, double cy, double size,
    Color c, {bool bold = false}) {
  final TextPainter tp = layoutText(s, size, c, bold: bold);
  tp.paint(canvas, Offset(cx - tp.width / 2, cy - tp.height / 2));
}

void textMid(Canvas canvas, String s, Zone r, double size, Color c,
    {bool bold = false}) {
  textCentre(canvas, s, r.cx, r.cy, size, c, bold: bold);
}

void roundRect(Canvas canvas, Zone r, Color fill,
    {double radius = 12, Color? edge, double edgeWidth = 1}) {
  final RRect rr =
      RRect.fromRectAndRadius(r.rect, Radius.circular(radius));
  canvas.drawRRect(rr, Paint()..color = fill);
  if (edge != null) {
    canvas.drawRRect(
      rr,
      Paint()
        ..color = edge
        ..style = PaintingStyle.stroke
        ..strokeWidth = edgeWidth,
    );
  }
}

void dimScreen(Canvas canvas, Size size, double alpha) {
  canvas.drawRect(
    Offset.zero & size,
    Paint()..color = Colors.black.withValues(alpha: alpha),
  );
}


const double stopRadius = 34;
const double stopPad = 26;

Zone stopZone(double w, double h) => Zone(
      w - stopPad - stopRadius * 2,
      h / 2 - stopRadius,
      stopRadius * 2,
      stopRadius * 2,
    );

void drawStopButton(Canvas canvas, Zone r, Skin skin) {
  final Offset c = r.centre;
  final double rad = r.w / 2;
  canvas.drawCircle(c, rad, Paint()..color = Colors.black.withValues(alpha: 0.45));
  canvas.drawCircle(c, rad - 2, Paint()..color = skin.card);
  canvas.drawCircle(
    c,
    rad - 2,
    Paint()
      ..color = skin.edge
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1,
  );
  final double s = math.max(8, rad / 3);
  canvas.drawRect(
    Rect.fromCenter(center: c, width: s * 2, height: s * 2),
    Paint()..color = skin.dim,
  );
}


class PauseLayout {
  PauseLayout(double w, double h) {
    final double pw = math.min(440.0, w - 80);
    const double ph = 286.0;
    final double px = (w - pw) / 2;
    final double py = (h - ph) / 2;
    const double pad = 28;
    const double gap = 14;
    final double bw = (pw - pad * 2 - gap) / 2;
    const double bh = 52.0;
    final double x0 = px + pad;
    final double x1 = x0 + bw + gap;
    panel = Zone(px, py, pw, ph);
    volume = Zone(x0, py + 56, pw - pad * 2, 52);
    final double y = py + 138;
    topLeft = Zone(x0, y, bw, bh);
    topRight = Zone(x1, y, bw, bh);
    bottomLeft = Zone(x0, y + bh + gap, bw, bh);
    bottomRight = Zone(x1, y + bh + gap, bw, bh);
  }

  late final Zone panel;
  late final Zone volume;
  late final Zone topLeft;
  late final Zone topRight;
  late final Zone bottomLeft;
  late final Zone bottomRight;
}

void drawVolumeSlider(
  Canvas canvas,
  Zone r,
  int volume,
  bool mute,
  Skin skin, {
  Color? trackColor,
  Color? knobHole,
}) {
  final double y = r.cy + 6;
  final double x0 = r.x + sliderPad;
  final double x1 = r.x1 - sliderPad;
  final String label = mute ? 'sound off' : 'sound $volume';
  final TextPainter tp =
      layoutText(label, 15, mute ? skin.faint : skin.dim);
  tp.paint(canvas, Offset(r.cx - tp.width / 2, r.y + 2));

  final Paint track = Paint()
    ..color = trackColor ?? const Color(0xFF48443E)
    ..strokeWidth = 5
    ..strokeCap = StrokeCap.round;
  canvas.drawLine(Offset(x0, y), Offset(x1, y), track);

  final double kx = sliderKnob(r, volume);
  if (kx > x0 && !mute) {
    canvas.drawLine(
      Offset(x0, y),
      Offset(kx, y),
      Paint()
        ..color = skin.accent
        ..strokeWidth = 5
        ..strokeCap = StrokeCap.round,
    );
  }
  canvas.drawCircle(
      Offset(kx, y), 13, Paint()..color = knobHole ?? skin.panel);
  canvas.drawCircle(
    Offset(kx, y),
    11,
    Paint()..color = mute ? skin.edge : skin.accent,
  );
}

class PauseButton {
  const PauseButton(this.key, this.label, this.zone, {this.lead = false});

  final String key;
  final String label;
  final Zone zone;
  final bool lead;
}

void drawPausePanel(
  Canvas canvas,
  Size size,
  Zones zones,
  List<PauseButton> buttons,
  int volume,
  bool mute,
  Skin skin, {
  double dim = 0.42,
}) {
  dimScreen(canvas, size, dim);
  final Zone? panel = zones['panel'];
  if (panel != null) {
    roundRect(canvas, panel, skin.panel, radius: 18, edge: skin.edge);
  }
  final Zone? vol = zones['volume'];
  if (vol != null) {
    drawVolumeSlider(canvas, vol, volume, mute, skin);
  }
  for (final PauseButton b in buttons) {
    roundRect(canvas, b.zone, b.lead ? skin.cardOn : skin.card,
        edge: b.lead ? skin.accent : skin.edge);
    textMid(canvas, b.label, b.zone, 16, b.lead ? skin.text : skin.dim);
  }
}

Future<ui.Image> imageFromPixels(Uint8List rgba, int w, int h) {
  final Completer<ui.Image> done = Completer<ui.Image>();
  ui.decodeImageFromPixels(rgba, w, h, ui.PixelFormat.rgba8888, done.complete);
  return done.future;
}