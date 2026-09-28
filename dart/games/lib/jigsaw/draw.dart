import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../core/paint.dart';
import '../core/zones.dart';
import 'board.dart';
import 'fx.dart';
import 'logic.dart';
import 'pieces.dart';

const Skin jigsawSkin = Skin(
  bg: Color(0xFF222A2A),
  text: Color(0xFFCAD1D6),
  dim: Color(0xFF8A9196),
  faint: Color(0xFF5E6368),
  card: Color(0xFF32373C),
  cardOn: Color(0xFF646054),
  edge: Color(0xFF545A60),
  accent: Color(0xFFDAD6C6),
  panel: Color(0xFF2F3338),
);

const Color trayBg = Color(0xFF2B2F34);
const Color frameCol = Color(0xFF42484E);
const Color sparkCol = Color(0xFFE6DCC8);
const double ghostMix = 0.17;

class PieceArt {
  const PieceArt(this.image, this.w, this.h);
  final ui.Image image;
  final int w;
  final int h;
}

class JigsawPainter extends CustomPainter {
  JigsawPainter({
    required this.logic,
    required this.now,
    required this.pieceArt,
    required this.reference,
    required this.thumbs,
    required this.setupThumb,
    required this.repaint,
  }) : super(repaint: repaint);

  final JigsawLogic logic;
  final double now;
  final List<PieceArt?> pieceArt;
  final ui.Image? reference;
  final Map<String, ui.Image> thumbs;
  final ui.Image? setupThumb;
  final Listenable repaint;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = jigsawSkin.bg);
    if (logic.page == pagePick) {
      _menu(canvas, size);
    } else if (logic.page == pageSetup) {
      _setup(canvas, size);
    } else {
      _game(canvas, size);
    }
    if (logic.settings) _settings(canvas, size);
  }

  void _menu(Canvas canvas, Size size) {
    textCentre(canvas, 'jigsaw', size.width / 2, 44, 30, jigsawSkin.text,
        bold: true);
    textCentre(
        canvas, 'pick a picture', size.width / 2, 76, 15, jigsawSkin.faint);

    for (int i = 0; i < logic.names.length; i++) {
      final Zone? r = logic.zones['pic:$i'];
      if (r == null) continue;
      final ui.Image? t = thumbs[logic.names[i]];
      if (t != null) {
        canvas.drawImageRect(
          t,
          Rect.fromLTWH(0, 0, t.width.toDouble(), t.height.toDouble()),
          r.rect,
          Paint()..filterQuality = FilterQuality.medium,
        );
      } else {
        roundRect(canvas, r, jigsawSkin.card, radius: 6);
        textMid(canvas, logic.names[i], r, 15, jigsawSkin.faint);
      }
      final bool on = i == logic.pick;
      canvas.drawRect(
        r.rect.inflate(on ? 2 : 1),
        Paint()
          ..color = on ? jigsawSkin.accent : frameCol
          ..style = PaintingStyle.stroke
          ..strokeWidth = on ? 2 : 1,
      );
    }

    final Zone? own = logic.zones['own'];
    if (own != null) {
      roundRect(canvas, own, jigsawSkin.card, edge: jigsawSkin.edge);
      textMid(canvas, 'your picture', own, 15, jigsawSkin.text);
    }
    final Zone? st = logic.zones['settings'];
    if (st != null) {
      roundRect(canvas, st, jigsawSkin.card, edge: jigsawSkin.edge);
      textMid(canvas, 'settings', st, 15, jigsawSkin.dim);
    }
    final Zone? go = logic.zones['start'];
    if (go != null) {
      roundRect(canvas, go, jigsawSkin.cardOn, edge: jigsawSkin.accent);
      textMid(canvas, 'start', go, 18, jigsawSkin.text, bold: true);
    }
  }

  void _setup(Canvas canvas, Size size) {
    textCentre(canvas, 'how do you want it', size.width / 2, 52, 22,
        jigsawSkin.text);

    final ui.Image? t = setupThumb;
    if (t != null) {
      const double top = 96;
      final double room = math.max(0.0, size.height * 0.40 - top);
      double tw = t.width.toDouble();
      double th = t.height.toDouble();
      if (th > room && th > 0) {
        final double sc = room / th;
        tw = math.max(1.0, tw * sc);
        th = math.max(1.0, room);
      }
      canvas.drawImageRect(
        t,
        Rect.fromLTWH(0, 0, t.width.toDouble(), t.height.toDouble()),
        Rect.fromLTWH((size.width - tw) / 2, top, tw, th),
        Paint()
          ..color = Colors.white.withValues(alpha: 0.5)
          ..filterQuality = FilterQuality.medium,
      );
    }

    final Zone? r = logic.zones['pieces'];
    if (r != null) {
      final List<int> opts = logic.sizeOptions();
      final int idx =
          opts.contains(logic.side) ? opts.indexOf(logic.side) : 0;
      final int top = math.max(1, opts.length - 1);
      final double y = r.cy;
      final double x0 = r.x + sliderPad;
      final double x1 = r.x1 - sliderPad;
      final Paint track = Paint()
        ..color = const Color(0xFF40444A)
        ..strokeWidth = 5
        ..strokeCap = StrokeCap.round;
      canvas.drawLine(Offset(x0, y), Offset(x1, y), track);
      final double kx = sliderKnob(r, idx, lo: 0, hi: top);
      if (kx > x0) {
        canvas.drawLine(
          Offset(x0, y),
          Offset(kx, y),
          Paint()
            ..color = jigsawSkin.accent
            ..strokeWidth = 5
            ..strokeCap = StrokeCap.round,
        );
      }
      for (int i = 0; i < opts.length; i++) {
        canvas.drawCircle(
          Offset(sliderKnob(r, i, lo: 0, hi: top), y + 20),
          2,
          Paint()..color = jigsawSkin.faint,
        );
      }
      canvas.drawCircle(Offset(kx, y), 14, Paint()..color = jigsawSkin.bg);
      canvas.drawCircle(
          Offset(kx, y), 12, Paint()..color = jigsawSkin.accent);

      final int count = logic.side * logic.side;
      final List<String> words = pieceShape(count);
      textCentre(canvas, '$count pieces', r.cx, y - 44, 24, jigsawSkin.text);
      textCentre(canvas, words[0], r.cx, y + 52, 17, jigsawSkin.accent);
      textCentre(canvas, words[1], r.cx, y + 78, 13, jigsawSkin.faint);
    }

    const List<List<String>> cuts = <List<String>>[
      <String>['square', 'straight', 'plain square edges'],
      <String>['interlock', 'interlocking', 'tabs that hook'],
    ];
    for (final List<String> row in cuts) {
      final Zone? z = logic.zones[row[0]];
      if (z == null) continue;
      final bool on = logic.style == row[0];
      roundRect(canvas, z, on ? jigsawSkin.cardOn : jigsawSkin.card,
          edge: on ? jigsawSkin.accent : jigsawSkin.edge);
      textCentre(canvas, row[1], z.cx, z.cy - 8, 17,
          on ? jigsawSkin.text : jigsawSkin.dim);
      textCentre(canvas, row[2], z.cx, z.cy + 14, 12,
          on ? jigsawSkin.dim : jigsawSkin.faint);
    }

    final Zone? back = logic.zones['back'];
    if (back != null) {
      roundRect(canvas, back, jigsawSkin.card, edge: jigsawSkin.edge);
      textMid(canvas, 'back', back, 15, jigsawSkin.dim);
    }
    final Zone? go = logic.zones['start'];
    if (go != null) {
      roundRect(canvas, go, jigsawSkin.cardOn, edge: jigsawSkin.accent);
      textMid(canvas, 'start', go, 19, jigsawSkin.text, bold: true);
    }
  }

  void _game(Canvas canvas, Size size) {
    final Board? b = logic.board;
    if (b == null) return;

    canvas.drawRect(
      Rect.fromLTWH(b.tray.x.toDouble(), b.tray.y.toDouble(),
          b.tray.w.toDouble(), b.tray.h.toDouble()),
      Paint()..color = trayBg,
    );

    final ui.Image? ref = reference;
    if (ref != null) {
      final Rect dst = Rect.fromLTWH(
        b.originX.toDouble(),
        b.originY.toDouble(),
        ref.width.toDouble(),
        ref.height.toDouble(),
      );
      canvas.drawImageRect(
        ref,
        Rect.fromLTWH(0, 0, ref.width.toDouble(), ref.height.toDouble()),
        dst,
        Paint()..color = Colors.white.withValues(alpha: ghostMix),
      );
      canvas.drawRect(
        dst.inflate(1),
        Paint()
          ..color = frameCol
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1,
      );
    }

    for (int i = 0; i < b.pieces.length; i++) {
      final Piece p = b.pieces[i];
      if (!p.placed) continue;
      _blit(canvas, i, p.x.toDouble(), p.y.toDouble());
      final double age = b.settleAge(p, now);
      if (age < settleSeconds) {
        _blit(canvas, i, p.x.toDouble(), p.y.toDouble(),
            tint: jigsawSkin.accent,
            alpha: 0.30 * (1.0 - age / settleSeconds));
      }
    }

    for (final int idx in b.order) {
      final Piece p = b.pieces[idx];
      if (p.placed) continue;
      final double lift = idx == b.held ? 3.0 : 0.0;
      if (lift > 0) {
        _blit(canvas, idx, p.x + 5.0, p.y + 6.0,
            tint: Colors.black, alpha: 0.26);
      }
      _blit(canvas, idx, p.x - lift, p.y - lift);
    }

    _particles(canvas);

    if (b.solved && logic.doneAt > 0.0) {
      _done(canvas, size);
      return;
    }
    if (logic.paused) {
      _pause(canvas, size);
      return;
    }
    final Zone? stop = logic.zones['stop'];
    if (stop != null) drawStopButton(canvas, stop, jigsawSkin);
  }

  void _blit(Canvas canvas, int index, double x, double y,
      {Color? tint, double alpha = 1.0}) {
    if (index >= pieceArt.length) return;
    final PieceArt? a = pieceArt[index];
    if (a == null) return;
    final Paint paint = Paint()..isAntiAlias = false;
    if (tint != null) {
      paint.colorFilter = ColorFilter.mode(tint, BlendMode.srcATop);
    }
    if (alpha < 1.0) {
      paint.color = Colors.white.withValues(alpha: alpha);
    }
    canvas.drawImage(a.image, Offset(x, y), paint);
  }

  void _particles(Canvas canvas) {
    final Effects f = logic.fx;
    final Paint add = Paint()..blendMode = BlendMode.plus;
    for (final Ring r in f.rings) {
      final double t = r.life / math.max(1e-6, r.full);
      final double rad = r.r0 * (1.0 + (1.0 - t) * ringGrow);
      final double a = t * t * 0.5;
      if (rad < 1 || a <= 0.01) continue;
      canvas.drawCircle(
        Offset(r.x, r.y),
        rad,
        Paint()
          ..blendMode = BlendMode.plus
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = Color.fromARGB(
            255,
            (r.colour[0] * a).round().clamp(0, 255),
            (r.colour[1] * a).round().clamp(0, 255),
            (r.colour[2] * a).round().clamp(0, 255),
          ),
      );
    }
    final Particles p = f.parts;
    for (int i = 0; i < p.n; i++) {
      final double t = p.alphaOf(i);
      final double a = math.pow(t, 1.4).toDouble();
      add.color = Color.fromARGB(
        255,
        (p.col[i * 3] * a).round().clamp(0, 255),
        (p.col[i * 3 + 1] * a).round().clamp(0, 255),
        (p.col[i * 3 + 2] * a).round().clamp(0, 255),
      );
      final double s = math.max(1.0, p.drawSize(i));
      canvas.drawRect(
        Rect.fromLTWH(p.px[i], p.py[i], s, s),
        add,
      );
    }
  }

  void _pause(Canvas canvas, Size size) {
    final List<PauseButton> buttons = <PauseButton>[];
    void addButton(String key, String label, {bool lead = false}) {
      final Zone? z = logic.zones[key];
      if (z != null) {
        buttons.add(PauseButton(key, label, z, lead: lead));
      }
    }

    addButton('resume', 'keep going', lead: true);
    addButton('again', 'start over');
    addButton('shuffle', 'shuffle');
    addButton('home', 'pictures');
    drawPausePanel(canvas, size, logic.zones, buttons, logic.volume,
        logic.volume <= 0, jigsawSkin);
  }

  void _done(Canvas canvas, Size size) {
    final double a = logic.doneAlpha(now);
    dimScreen(canvas, size, a * 0.42);
    final int c = (140 * a).round() + 90;
    textCentre(canvas, 'done', size.width / 2, size.height / 2 - 48, 34,
        Color.fromARGB(255, c, c, c));
    if (a <= 0.55) return;
    const List<List<String>> rows = <List<String>>[
      <String>['again', 'again'],
      <String>['back', 'pictures'],
    ];
    for (final List<String> row in rows) {
      final Zone? z = logic.zones[row[0]];
      if (z == null) continue;
      roundRect(canvas, z, jigsawSkin.card, edge: jigsawSkin.edge);
      textMid(canvas, row[1], z, 16, jigsawSkin.text);
    }
  }

  void _settings(Canvas canvas, Size size) {
    dimScreen(canvas, size, 0.45);
    final Zone? panel = logic.zones['panel'];
    if (panel != null) {
      roundRect(canvas, panel, jigsawSkin.panel,
          radius: 18, edge: jigsawSkin.edge);
    }
    textCentre(canvas, 'settings', size.width / 2, size.height * 0.30, 23,
        jigsawSkin.text);
    final Zone? vol = logic.zones['volume'];
    if (vol != null) {
      drawVolumeSlider(
          canvas, vol, logic.volume, logic.volume <= 0, jigsawSkin,
          knobHole: jigsawSkin.bg);
    }
    final Zone? close = logic.zones['close'];
    if (close != null) {
      roundRect(canvas, close, jigsawSkin.cardOn, edge: jigsawSkin.accent);
      textMid(canvas, 'done', close, 17, jigsawSkin.text);
    }
  }

  @override
  bool shouldRepaint(covariant JigsawPainter old) => true;
}
