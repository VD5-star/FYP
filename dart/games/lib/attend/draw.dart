import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../core/paint.dart';
import '../core/zones.dart';
import 'logic.dart';
import 'session.dart';
import 'sounds.dart';

const Skin attendSkin = Skin(
  bg: Color(0xFF171614),
  text: Color(0xFFD8DEE2),
  dim: Color(0xFF9CA3A8),
  faint: Color(0xFF676C70),
  card: Color(0xFF32363A),
  cardOn: Color(0xFF646056),
  edge: Color(0xFF5E635E),
  accent: Color(0xFFD8D4C6),
  panel: Color(0xFF26292C),
);

const double veil = 0.34;
const int driftOrbs = 3;
const double driftSpeed = 0.10;

const Map<String, String> stagePrompts = <String, String>{
  stageFocus: 'stay with',
  stageSwitch: 'now move to',
  stageDivide: 'hold them all at once',
  stageSettle: 'let it all go',
};

class AttendPainter extends CustomPainter {
  AttendPainter({
    required this.logic,
    required this.now,
    required this.scenes,
    required this.repaint,
  }) : super(repaint: repaint);

  final AttendLogic logic;
  final double now;
  final Map<String, ui.Image> scenes;
  final Listenable repaint;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = attendSkin.bg);
    if (logic.page == pageMenu) {
      _menu(canvas, size);
      return;
    }
    if (logic.page == pageSettings) {
      _menu(canvas, size);
      _settings(canvas, size);
      return;
    }
    final Session? s = logic.sess;
    if (s == null) {
      _menu(canvas, size);
      return;
    }
    _session(canvas, size, s);
    if (logic.page == pageOver) {
      final double a =
          math.min(1.0, (now - logic.overAt) / overFade);
      _over(canvas, size, a);
      return;
    }
    if (logic.inStopMenu) {
      drawPausePanel(canvas, size, logic.zones, logic.pauseButtons(),
          logic.volume, logic.muted, attendSkin);
      return;
    }
    final Zone? open = logic.zones['open'];
    if (open != null) drawStopButton(canvas, open, attendSkin);
  }

  void _backdrop(Canvas canvas, Size size, Map<String, double> weights) {
    double total = 0;
    for (final double v in weights.values) {
      total += v;
    }
    if (total <= 0 || scenes.isEmpty) return;
    final Rect dst = Offset.zero & size;
    for (final MapEntry<String, double> e in weights.entries) {
      if (e.value <= 0.001) continue;
      final ui.Image? img = scenes[e.key];
      if (img == null) continue;
      final Paint p = Paint()
        ..color = Colors.white.withValues(alpha: e.value / total)
        ..filterQuality = FilterQuality.low;
      canvas.drawImageRect(
        img,
        Rect.fromLTWH(0, 0, img.width.toDouble(), img.height.toDouble()),
        dst,
        p,
      );
    }
  }

  void _drift(Canvas canvas, Size size, double t) {
    for (int i = 0; i < driftOrbs; i++) {
      final double p = t * driftSpeed + i * (2 * math.pi / driftOrbs);
      final double rx = size.width * (0.20 + 0.05 * math.sin(p * 0.37 + i));
      final double ry =
          size.height * (0.13 + 0.04 * math.cos(p * 0.29 + i * 1.7));
      final double x = size.width * 0.5 + rx * math.sin(p);
      final double y =
          size.height * 0.5 + ry * math.sin(p * 0.61 + i * 0.9);
      final double rad = math.min(size.width, size.height) *
          (0.085 + 0.022 * math.sin(p * 0.53 + i));
      canvas.drawCircle(
        Offset(x, y),
        rad * 2.2,
        Paint()
          ..color = const Color(0xFF0D0C0A).withValues(alpha: 0.5)
          ..maskFilter =
              MaskFilter.blur(BlurStyle.normal, math.max(8.0, rad * 0.8))
          ..blendMode = BlendMode.plus,
      );
    }
  }

  void _session(Canvas canvas, Size size, Session s) {
    _backdrop(canvas, size, s.sceneWeights());
    dimScreen(canvas, size, veil);
    _drift(canvas, size, s.position);

    final String stage = s.stage;
    final String? t = s.target;
    final double cy = size.height / 2;
    if (stage == stageFocus || stage == stageSwitch) {
      textCentre(canvas, stagePrompts[stage] ?? '', size.width / 2,
          cy - 34, 19, attendSkin.dim);
      textCentre(canvas, soundLabels[t] ?? t ?? '', size.width / 2,
          cy + 26, 38, attendSkin.text);
    } else if (stage == stageDivide) {
      textCentre(canvas, stagePrompts[stage] ?? '', size.width / 2, cy + 8,
          28, attendSkin.text);
    } else if (stage == stageSettle) {
      textCentre(canvas, stagePrompts[stage] ?? '', size.width / 2, cy + 8,
          28, attendSkin.dim);
    }

    _breath(canvas, size, s);
    _bar(canvas, size, s);
  }

  void _breath(Canvas canvas, Size size, Session s) {
    final double ph = (s.position / 9.0) % 1.0;
    final double k = 0.5 - 0.5 * math.cos(2 * math.pi * ph);
    final double r = math.min(size.width, size.height) * (0.030 + 0.010 * k);
    canvas.drawCircle(
      Offset(size.width / 2, size.height * 0.80),
      r,
      Paint()
        ..color = attendSkin.faint.withValues(alpha: 0.30 + 0.25 * k)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  void _bar(Canvas canvas, Size size, Session s) {
    final double w = size.width * 0.52;
    final double x0 = (size.width - w) / 2;
    final double y = size.height - 42;
    canvas.drawLine(
      Offset(x0, y),
      Offset(x0 + w, y),
      Paint()
        ..color = attendSkin.card
        ..strokeWidth = 3,
    );
    canvas.drawLine(
      Offset(x0, y),
      Offset(x0 + w * s.progress, y),
      Paint()
        ..color = attendSkin.accent
        ..strokeWidth = 3,
    );
    textCentre(canvas, clockText(s.remaining), size.width / 2, y - 22, 14,
        attendSkin.faint);
  }

  void _menu(Canvas canvas, Size size) {
    final ui.Image? first = scenes[soundNames.first];
    if (first != null) {
      canvas.drawImageRect(
        first,
        Rect.fromLTWH(0, 0, first.width.toDouble(), first.height.toDouble()),
        Offset.zero & size,
        Paint()..filterQuality = FilterQuality.low,
      );
      dimScreen(canvas, size, 0.58);
    }
    textCentre(canvas, 'attend', size.width / 2, size.height * 0.22, 40,
        attendSkin.text);
    textCentre(canvas, 'five sounds, one at a time', size.width / 2,
        size.height * 0.22 + 38, 16, attendSkin.faint);

    final Zone? mins = logic.zones['minutes'];
    if (mins != null) {
      final double y = mins.cy;
      final double x0 = mins.x + sliderPad;
      final double x1 = mins.x1 - sliderPad;
      canvas.drawLine(
        Offset(x0, y),
        Offset(x1, y),
        Paint()
          ..color = attendSkin.card
          ..strokeWidth = 5
          ..strokeCap = StrokeCap.round,
      );
      final double kx = sliderKnob(mins, logic.minutes,
          lo: minMinutes, hi: maxMinutes);
      canvas.drawLine(
        Offset(x0, y),
        Offset(kx, y),
        Paint()
          ..color = attendSkin.accent
          ..strokeWidth = 5
          ..strokeCap = StrokeCap.round,
      );
      canvas.drawCircle(Offset(kx, y), 13, Paint()..color = attendSkin.bg);
      canvas.drawCircle(
          Offset(kx, y), 11, Paint()..color = attendSkin.accent);
      textCentre(canvas, '${logic.minutes} minutes', mins.cx, y - 42, 24,
          attendSkin.text);
      final ShapeBand sh = logic.shape;
      textCentre(canvas, sh.name, mins.cx, y + 44, 17, attendSkin.accent);
      textCentre(canvas, sh.detail, mins.cx, y + 70, 14, attendSkin.faint);
    }

    final Zone? begin = logic.zones['begin'];
    if (begin != null) {
      final bool on = logic.ready;
      roundRect(canvas, begin, on ? attendSkin.cardOn : attendSkin.card,
          edge: on ? attendSkin.accent : attendSkin.edge);
      textMid(canvas, on ? 'begin' : 'getting ready', begin, 19,
          on ? attendSkin.text : attendSkin.faint);
    }
    final Zone? st = logic.zones['settings'];
    if (st != null && logic.page != pageSettings) {
      roundRect(canvas, st, attendSkin.card, edge: attendSkin.edge);
      textMid(canvas, 'sound', st, 15, attendSkin.dim);
    }
  }

  void _settings(Canvas canvas, Size size) {
    dimScreen(canvas, size, 0.45);
    final Zone? panel = logic.zones['panel'];
    if (panel != null) {
      roundRect(canvas, panel, attendSkin.panel,
          radius: 18, edge: attendSkin.edge);
      textCentre(canvas, 'sound', panel.cx, panel.y + 44, 24,
          attendSkin.text);
    }
    final Zone? vol = logic.zones['volume'];
    if (vol != null) {
      drawVolumeSlider(
          canvas, vol, logic.volume, logic.muted, attendSkin);
    }
    final Zone? close = logic.zones['close'];
    if (close != null) {
      roundRect(canvas, close, attendSkin.cardOn, edge: attendSkin.accent);
      textMid(canvas, 'done', close, 18, attendSkin.text);
    }
  }

  void _over(Canvas canvas, Size size, double a) {
    dimScreen(canvas, size, 0.55 * a);
    textCentre(canvas, 'that is the end of it', size.width / 2,
        size.height * 0.38, 30, attendSkin.text.withValues(alpha: a));
    textCentre(canvas, 'nothing was scored, nothing was kept',
        size.width / 2, size.height * 0.38 + 34, 15,
        attendSkin.faint.withValues(alpha: a));
    for (final String k in <String>['again', 'stop']) {
      final Zone? r = logic.zones[k];
      if (r == null) continue;
      final bool lead = k == 'again';
      roundRect(canvas, r, lead ? attendSkin.cardOn : attendSkin.card,
          edge: lead ? attendSkin.accent : attendSkin.edge);
      textMid(canvas, lead ? 'again' : 'back', r, 17,
          lead ? attendSkin.text : attendSkin.dim);
    }
  }

  @override
  bool shouldRepaint(covariant AttendPainter old) => true;
}
