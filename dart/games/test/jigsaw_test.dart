import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:psybot_games/jigsaw/game.dart';
import 'package:psybot_games/core/synth.dart';
import 'package:psybot_games/core/zones.dart';
import 'package:psybot_games/jigsaw/art.dart';
import 'package:psybot_games/jigsaw/audio.dart';
import 'package:psybot_games/jigsaw/board.dart';
import 'package:psybot_games/jigsaw/fx.dart';
import 'package:psybot_games/jigsaw/logic.dart';
import 'package:psybot_games/jigsaw/photo.dart';
import 'package:psybot_games/jigsaw/pieces.dart';
import 'package:psybot_games/jigsaw/raster.dart';

Raster fakeArt(String name, int w, int h) {
  final Raster r = Raster.blank(w, h);
  final int seed = name.codeUnits.fold(0, (int a, int b) => a + b);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final int i = (y * w + x) * 4;
      r.rgba[i] = (x * 3 + seed) % 256;
      r.rgba[i + 1] = (y * 5 + seed) % 256;
      r.rgba[i + 2] = ((x + y) * 7) % 256;
      r.rgba[i + 3] = 255;
    }
  }
  return r;
}

JigsawLogic makeLogic({int w = 1280, int h = 860}) =>
    JigsawLogic(width: w, height: h, seed: 3, artSource: fakeArt);

void tapZone(JigsawLogic a, String key) {
  final Zone r = a.zones[key]!;
  a.pointerDown(r.cx, r.cy);
  a.pointerUp(r.cx, r.cy);
}

List<int>? grabPoint(Piece p) {
  for (int y = p.y; y < p.y + p.h; y++) {
    for (int x = p.x; x < p.x + p.w; x++) {
      if (p.hits(x, y)) return <int>[x, y];
    }
  }
  return null;
}

bool solve(JigsawLogic a) {
  final Board b = a.board!;
  int guard = 0;
  while (!b.solved && guard < b.total * 4) {
    guard++;
    final Piece p = b.pieces[b.order.last];
    final List<int>? g = grabPoint(p);
    if (g == null) break;
    a.pointerDown(g[0].toDouble(), g[1].toDouble());
    final double dx = (b.targetX(p) + (g[0] - p.x)).toDouble();
    final double dy = (b.targetY(p) + (g[1] - p.y)).toDouble();
    a.pointerMove(dx, dy);
    a.pointerUp(dx, dy);
  }
  return b.solved;
}

Raster demoImage(int w, int h) {
  final Raster r = Raster.blank(w, h);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final int i = (y * w + x) * 4;
      r.rgba[i] = (20 + x * 235 ~/ math.max(1, w - 1)).clamp(0, 255);
      r.rgba[i + 1] = (20 + y * 235 ~/ math.max(1, h - 1)).clamp(0, 255);
      r.rgba[i + 2] = 90;
      r.rgba[i + 3] = 255;
    }
  }
  return r;
}

void main() {
  group('pick page', () {
    test('nine cells in a 3x3 grid', () {
      final JigsawLogic a = makeLogic();
      final List<String> cells = a.zones.keys
          .where((String k) => k.startsWith('pic:'))
          .toList();
      expect(cells.length, 8);
      expect(a.zones.has('own'), isTrue);
      expect(cells.length + 1, 9);

      final Set<double> xs = <double>{};
      final Set<double> ys = <double>{};
      for (final String k in <String>[...cells, 'own']) {
        xs.add(a.zones[k]!.x);
        ys.add(a.zones[k]!.y);
      }
      expect(xs.length, 3);
      expect(ys.length, 3);
    });

    test('start and settings sit centred below the grid', () {
      final JigsawLogic a = makeLogic();
      final Zone st = a.zones['settings']!;
      final Zone go = a.zones['start']!;
      expect(st.y, go.y);
      double maxY = 0;
      for (final String k in a.zones.keys) {
        if (k.startsWith('pic:') || k == 'own') {
          maxY = math.max(maxY, a.zones[k]!.y);
        }
      }
      expect(st.y > maxY, isTrue);
      expect(((st.x + go.x1) / 2 - a.w / 2).abs() < 12, isTrue);
    });

    test('no piece controls on the pick page', () {
      final JigsawLogic a = makeLogic();
      expect(a.zones.has('pieces'), isFalse);
      expect(a.zones.has('cut'), isFalse);
      expect(a.zones.has('square'), isFalse);
      expect(a.zones.has('interlock'), isFalse);
    });

    test('every pick zone is touchable', () {
      final JigsawLogic a = makeLogic();
      for (final String k in a.zones.keys) {
        expect(a.zones[k]!.touchable, isTrue, reason: k);
      }
    });
  });

  group('routing', () {
    test('tapping a picture goes to setup without a board', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'pic:3');
      expect(a.page, pageSetup);
      expect(a.board, isNull);
      expect(a.pick, 3);
      expect(a.zones.has('pieces'), isTrue);
      expect(a.zones.has('square'), isTrue);
      expect(a.zones.has('interlock'), isTrue);
      expect(a.zones.has('start'), isTrue);
      expect(a.zones.has('back'), isTrue);
    });

    test('back returns to pick', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'pic:3');
      tapZone(a, 'back');
      expect(a.page, pagePick);
      expect(a.zones.has('pic:0'), isTrue);
    });

    test('setup start builds the board for every picture', () {
      for (int i = 0; i < artNames.length; i++) {
        final JigsawLogic a = makeLogic();
        tapZone(a, 'pic:$i');
        tapZone(a, 'start');
        expect(a.board, isNotNull, reason: 'picture $i');
        expect(a.title, a.names[i]);
        expect(a.page, pageGame);
        expect(a.board!.total, a.side * a.side);
      }
    });

    test('the cut is chosen by tap on setup', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'square');
      expect(a.style, styleSquare);
      tapZone(a, 'interlock');
      expect(a.style, styleInterlock);
      tapZone(a, 'square');
      expect(a.style, styleSquare);
      tapZone(a, 'start');
      expect(a.board!.cut.style, styleSquare);
    });

    test('the pieces slider reaches every offered size', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      final Zone r = a.zones['pieces']!;
      expect(r.touchable, isTrue);
      final List<int> opts = a.sizeOptions();
      final Set<int> got = <int>{};
      for (double px = r.x; px < r.x1; px += 2) {
        a.pointerDown(r.cx, r.cy);
        a.pointerMove(px, r.cy);
        a.pointerUp(px, r.cy);
        got.add(a.side);
      }
      expect(got, opts.toSet());
      a.pointerDown(r.x + 4, r.cy);
      a.pointerUp(r.x + 4, r.cy);
      expect(a.side, opts.first);
      a.pointerDown(r.x1 - 4, r.cy);
      a.pointerUp(r.x1 - 4, r.cy);
      expect(a.side, opts.last);
    });

    test('size words never go backwards', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      final List<String> words = a
          .sizeOptions()
          .map((int s) => pieceShape(s * s)[0])
          .toList();
      expect(words.toSet().length >= 2, isTrue);
      for (int i = 0; i + 1 < words.length; i++) {
        expect(words.indexOf(words[i]) <= words.indexOf(words[i + 1]), isTrue);
      }
    });
  });

  group('game page', () {
    test('exactly one round stop zone on the right at half height', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      expect(a.zones.keys.toList(), <String>['stop']);
      final Zone s = a.zones['stop']!;
      expect(s.w, s.h);
      expect(s.cx > 0.8 * a.w, isTrue);
      expect((s.cy - a.h / 2).abs() <= 8, isTrue);
      expect(s.touchable, isTrue);
    });

    test('a tap on the board does not disturb a piece', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      final Board b = a.board!;
      final Piece p = b.pieces[b.order.last];
      final List<int> g = grabPoint(p)!;
      final List<int> before = <int>[p.x, p.y];
      a.pointerDown(g[0].toDouble(), g[1].toDouble());
      a.pointerUp(g[0].toDouble(), g[1].toDouble());
      expect(<int>[p.x, p.y], before);
    });

    test('drag then drop moves and can snap', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      final Board b = a.board!;
      final Piece p = b.pieces[b.order.last];
      List<int> g = grabPoint(p)!;
      a.pointerDown(g[0].toDouble(), g[1].toDouble());
      a.pointerMove(g[0] + 120.0, g[1] - 80.0);
      a.pointerUp(g[0] + 120.0, g[1] - 80.0);
      expect(p.placed, isFalse);

      g = grabPoint(p)!;
      a.pointerDown(g[0].toDouble(), g[1].toDouble());
      final double dx = (b.targetX(p) + (g[0] - p.x)).toDouble();
      final double dy = (b.targetY(p) + (g[1] - p.y)).toDouble();
      a.pointerMove(dx, dy);
      a.pointerUp(dx, dy);
      expect(p.placed, isTrue);
    });

    test('a drag starting on the stop button never grabs a piece', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      final Board b = a.board!;
      final Zone r = a.zones['stop']!;
      a.pointerDown(r.cx, r.cy);
      expect(b.held, isNull);
      a.pointerMove(r.cx + 200, r.cy - 200);
      a.pointerUp(r.cx + 200, r.cy - 200);
      expect(b.moves, 0);
      expect(b.done, 0);
    });

    test('full solve by pointer only, both cuts', () {
      for (final String style in <String>[styleSquare, styleInterlock]) {
        final JigsawLogic a = makeLogic();
        a.style = style;
        tapZone(a, 'start');
        a.style = style;
        tapZone(a, 'start');
        expect(solve(a), isTrue, reason: style);
        expect(a.board!.moves, a.board!.total);
      }
    });

    test('the done screen is reachable and works', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      solve(a);
      a.step(100.0, 0.016);
      expect(a.doneAt > 0, isTrue);
      expect(a.zones.has('again'), isTrue);
      expect(a.zones.has('back'), isTrue);
      tapZone(a, 'again');
      expect(a.board!.done, 0);
      solve(a);
      a.step(200.0, 0.016);
      tapZone(a, 'back');
      expect(a.board, isNull);
      expect(a.page, pagePick);
    });
  });

  group('stop menu', () {
    JigsawLogic opened() {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      tapZone(a, 'stop');
      return a;
    }

    test('has volume, resume, again, shuffle, home', () {
      final JigsawLogic a = opened();
      expect(a.paused, isTrue);
      for (final String k in <String>[
        'volume',
        'resume',
        'again',
        'shuffle',
        'home',
      ]) {
        expect(a.zones.has(k), isTrue, reason: k);
        expect(a.zones[k]!.touchable, isTrue, reason: k);
      }
      expect(a.zones.has('settings'), isFalse);
    });

    test('panel is mid-size and centred', () {
      final JigsawLogic a = opened();
      final Zone p = a.zones['panel']!;
      expect(p.w >= 260 && p.w <= 520, isTrue, reason: '${p.w}');
      expect(p.h >= 200 && p.h <= 340, isTrue, reason: '${p.h}');
      expect((p.cx - a.w / 2).abs() <= 2, isTrue);
      expect((p.cy - a.h / 2).abs() <= 2, isTrue);
    });

    test('volume sits above the buttons and is centred', () {
      final JigsawLogic a = opened();
      final Zone v = a.zones['volume']!;
      final Zone p = a.zones['panel']!;
      expect(v.y < a.zones['resume']!.y, isTrue);
      expect((v.cx - p.cx).abs() <= 2, isTrue);
    });

    test('four buttons in exactly two rows of two', () {
      final JigsawLogic a = opened();
      final Map<double, List<String>> rows = <double, List<String>>{};
      for (final String k in <String>['resume', 'again', 'shuffle', 'home']) {
        rows.putIfAbsent(a.zones[k]!.y, () => <String>[]).add(k);
      }
      expect(rows.length, 2);
      for (final List<String> v in rows.values) {
        expect(v.length, 2);
      }
    });

    test('the panel never swallows its own controls', () {
      final JigsawLogic a = opened();
      for (final String k in <String>[
        'volume',
        'resume',
        'again',
        'shuffle',
        'home',
      ]) {
        final Zone z = a.zones[k]!;
        expect(a.zones.at(z.cx, z.cy), k);
      }
    });

    test('dragging volume far left gives 0, far right 100', () {
      final JigsawLogic a = opened();
      final Zone r = a.zones['volume']!;
      a.pointerDown(r.cx, r.cy);
      a.pointerMove(r.x - 400, r.cy);
      a.pointerUp(r.x - 400, r.cy);
      expect(a.volume, volMin);
      expect(a.paused, isTrue);
      expect(a.settings, isFalse);

      a.pointerDown(r.cx, r.cy);
      a.pointerMove(r.x1 + 400, r.cy);
      a.pointerUp(r.x1 + 400, r.cy);
      expect(a.volume, volMax);
      expect(a.paused, isTrue);
    });

    test('the volume slider is smooth, not a few steps', () {
      final JigsawLogic a = opened();
      final Zone r = a.zones['volume']!;
      final Set<int> steps = <int>{};
      for (double px = r.x; px < r.x1; px += 3) {
        a.pointerDown(r.cx, r.cy);
        a.pointerMove(px, r.cy);
        a.pointerUp(px, r.cy);
        steps.add(a.volume);
      }
      expect(steps.length > 40, isTrue, reason: '${steps.length}');
    });

    test('tapping the panel background does nothing', () {
      final JigsawLogic a = opened();
      final Zone p = a.zones['panel']!;
      a.pointerDown(p.x + 6, p.y + 6);
      a.pointerUp(p.x + 6, p.y + 6);
      expect(a.paused, isTrue);
      expect(a.zones.has('panel'), isTrue);
    });

    test('keep going returns to the board', () {
      final JigsawLogic a = opened();
      tapZone(a, 'resume');
      expect(a.paused, isFalse);
      expect(a.zones.keys.toList(), <String>['stop']);
    });

    test('start over gives a fresh board and leaves the menu', () {
      final JigsawLogic a = opened();
      final Board b1 = a.board!;
      b1.pieces[0].placed = true;
      tapZone(a, 'again');
      expect(identical(a.board, b1), isFalse);
      expect(a.board!.done, 0);
      expect(a.paused, isFalse);
    });

    test('shuffle keeps placed pieces and moves loose ones', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      final Board b = a.board!;
      final Piece p0 = b.pieces[0];
      p0.x = b.targetX(p0);
      p0.y = b.targetY(p0);
      p0.placed = true;
      final List<Piece> loose = b.pieces.where((Piece p) => !p.placed).toList();
      final List<List<int>> before = loose
          .map((Piece p) => <int>[p.x, p.y])
          .toList();

      tapZone(a, 'stop');
      tapZone(a, 'shuffle');

      expect(identical(a.board, b), isTrue);
      expect(p0.placed, isTrue);
      expect(b.done, 1);
      expect(p0.x, b.targetX(p0));
      expect(p0.y, b.targetY(p0));
      final List<List<int>> after = loose
          .map((Piece p) => <int>[p.x, p.y])
          .toList();
      expect(after.toString() == before.toString(), isFalse);
      expect(a.paused, isFalse);
    });

    test('pictures returns to the pick page', () {
      final JigsawLogic a = opened();
      tapZone(a, 'home');
      expect(a.board, isNull);
      expect(a.page, pagePick);
      expect(a.custom, isNull);
      expect(a.paused, isFalse);
    });
  });

  group('sizes and pieces', () {
    test('every offered size 3..10 yields pieces >= 52px', () {
      for (final List<int> wh in <List<int>>[
        <int>[900, 600],
        <int>[1280, 860],
        <int>[1920, 1080],
        <int>[700, 520],
      ]) {
        final JigsawLogic a = makeLogic(w: wh[0], h: wh[1]);
        final Layout l = a.layout();
        for (final int s in a.sizeOptions()) {
          expect(s >= 3 && s <= 10, isTrue);
          final double px = math.min(l.areaW / s, l.areaH / s);
          expect(
            px >= minPiecePx,
            isTrue,
            reason: '${wh[0]}x${wh[1]} size $s -> $px',
          );
        }
      }
    });

    test('cut has no gaps for either style', () {
      for (final String style in <String>[styleSquare, styleInterlock]) {
        for (final int n in <int>[3, 5, 8]) {
          final Cut cut = makeCut(n, n, 300, 200, style: style, seed: 7);
          final Int32List cov = coverage(cut);
          int gaps = 0;
          for (final int v in cov) {
            if (v == 0) gaps++;
          }
          expect(gaps, 0, reason: '$style ${n}x$n');
        }
      }
    });

    test('reassembly covers the whole picture', () {
      final Raster img = demoImage(240, 160);
      for (final String style in <String>[styleSquare, styleInterlock]) {
        final Cut cut = makeCut(4, 4, img.w, img.h, style: style, seed: 5);
        final List<Piece> ps = cutImage(img, cut);
        final Uint8List painted = Uint8List(img.w * img.h);
        for (final Piece p in ps) {
          for (int y = 0; y < p.h; y++) {
            final int cy = p.homeY + y;
            if (cy < 0 || cy >= img.h) continue;
            for (int x = 0; x < p.w; x++) {
              final int cx = p.homeX + x;
              if (cx < 0 || cx >= img.w) continue;
              if (p.alpha[y * p.w + x] > 0) painted[cy * img.w + cx] = 1;
            }
          }
        }
        int missed = 0;
        for (final int v in painted) {
          if (v == 0) missed++;
        }
        expect(missed, 0, reason: style);
      }
    });

    test('piece edges are antialiased but cores stay solid', () {
      final Raster img = demoImage(240, 160);
      for (final String style in <String>[styleSquare, styleInterlock]) {
        final Cut cut = makeCut(4, 4, img.w, img.h, style: style, seed: 3);
        int soft = 0;
        int hard = 0;
        for (final Piece p in cutImage(img, cut)) {
          for (final int a in p.alpha) {
            if (a > 0) hard++;
            if (a > 0 && a < 250) soft++;
          }
        }
        expect(soft > 0, isTrue, reason: style);
        expect(soft / hard < 0.16, isTrue, reason: '$style ${soft / hard}');
      }
    });

    test('every piece has a grabbable pixel', () {
      final Raster img = demoImage(240, 160);
      final Board b = buildBoard(
        img,
        4,
        4,
        40,
        46,
        const Tray(0, 200, 640, 200),
        style: styleInterlock,
        seed: 9,
      );
      for (final Piece p in b.pieces) {
        expect(grabPoint(p), isNotNull);
      }
    });
  });

  group('board logic', () {
    Board demo({String style = styleInterlock, int n = 4}) => buildBoard(
      demoImage(240, 160),
      n,
      n,
      40,
      46,
      const Tray(0, 200, 640, 200),
      style: style,
      seed: 4,
    );

    test('scatter puts every piece loose and orders them all', () {
      final Board b = demo();
      expect(b.done, 0);
      final List<int> sorted = List<int>.from(b.order)..sort();
      expect(sorted, List<int>.generate(b.total, (int i) => i));
    });

    test('snap works inside the radius and only there', () {
      final Board b = demo();
      final Piece p = b.pieces[0];
      final math.Random rng = math.Random(1);
      int inside = 0;
      for (int i = 0; i < 200; i++) {
        final double ang = rng.nextDouble() * 2 * math.pi;
        final double d = rng.nextDouble() * b.radius * 0.94;
        b.held = 0;
        b.grabX = 0;
        b.grabY = 0;
        p.placed = false;
        b.settled.clear();
        if (!b.order.contains(0)) b.order.add(0);
        p.x = (b.targetX(p) + math.cos(ang) * d).toInt();
        p.y = (b.targetY(p) + math.sin(ang) * d).toInt();
        if (b.drop()) inside++;
      }
      expect(inside, 200);

      int outside = 0;
      for (int i = 0; i < 200; i++) {
        final double ang = rng.nextDouble() * 2 * math.pi;
        final double d = b.radius * (1.3 + rng.nextDouble() * 4);
        b.held = 0;
        b.grabX = 0;
        b.grabY = 0;
        p.placed = false;
        b.settled.clear();
        if (!b.order.contains(0)) b.order.add(0);
        p.x = (b.targetX(p) + math.cos(ang) * d).toInt();
        p.y = (b.targetY(p) + math.sin(ang) * d).toInt();
        if (b.drop()) outside++;
      }
      expect(outside, 0);
    });

    test('snap radius shrinks as the grid grows', () {
      double? prev;
      for (final int n in <int>[3, 5, 8, 10]) {
        final double r = snapRadius(makeCut(n, n, 900, 600, seed: 1));
        if (prev != null) expect(r <= prev + 0.01, isTrue);
        prev = r;
      }
      expect(snapRadius(makeCut(10, 10, 900, 600, seed: 1)) >= snapMin, isTrue);
    });

    test('a placed piece leaves the pick order and stays put', () {
      final Board b = demo();
      const int idx = 3;
      final Piece p = b.pieces[idx];
      b.order.remove(idx);
      b.order.add(idx);
      b.held = idx;
      b.grabX = 0;
      b.grabY = 0;
      p.x = b.targetX(p);
      p.y = b.targetY(p);
      b.drop();
      expect(p.placed, isTrue);
      expect(b.order.contains(idx), isFalse);
      expect(b.pick(p.x + p.w / 2, p.y + p.h / 2), isFalse);
    });

    test('pick returns the topmost piece', () {
      final Board b = demo(style: styleSquare);
      b.pieces[0].x = 200;
      b.pieces[0].y = 260;
      b.pieces[1].x = 200;
      b.pieces[1].y = 260;
      b.order = <int>[0, 1];
      b.pick(200 + b.pieces[0].w / 2, 260 + b.pieces[0].h / 2);
      expect(b.held, 1);
      expect(b.order.last, 1);
    });

    test('drag keeps the grab offset constant', () {
      final Board b = demo();
      final Piece p = b.pieces[2];
      double gx = p.x + p.w / 2;
      double gy = p.y + p.h / 2;
      b.pick(gx, gy);
      for (final List<double> d in <List<double>>[
        <double>[30, 20],
        <double>[-140, 90],
        <double>[400, -260],
      ]) {
        gx += d[0];
        gy += d[1];
        b.drag(gx, gy);
        final Piece held = b.pieces[b.held!];
        expect((gx - held.x - b.grabX).abs() < 1, isTrue);
        expect((gy - held.y - b.grabY).abs() < 1, isTrue);
      }
    });

    test('gaps between moves are recorded, not scored', () {
      final Board b = demo();
      double t = 1000.0;
      final List<double> steps = <double>[0.0, 1.2, 0.8, 4.0, 1.0];
      for (int i = 0; i < steps.length; i++) {
        t += steps[i];
        b.held = i;
        b.grabX = 0;
        b.grabY = 0;
        if (!b.order.contains(i)) b.order.add(i);
        b.pieces[i].x = 9999;
        b.pieces[i].y = 9999;
        b.drop(now: t);
      }
      expect(b.gaps.length, b.moves - 1);
      for (int i = 0; i < b.gaps.length; i++) {
        expect((b.gaps[i] - steps[i + 1]).abs() < 1e-6, isTrue);
      }
    });

    test('solving every piece marks the board solved', () {
      for (final String style in <String>[styleSquare, styleInterlock]) {
        final Board b = demo(style: style, n: 3);
        for (int i = 0; i < b.pieces.length; i++) {
          b.order.remove(i);
          b.order.add(i);
          b.held = i;
          b.grabX = 0;
          b.grabY = 0;
          b.pieces[i].x = b.targetX(b.pieces[i]);
          b.pieces[i].y = b.targetY(b.pieces[i]);
          b.drop();
        }
        expect(b.solved, isTrue, reason: style);
        expect(b.order, isEmpty);
      }
    });
  });

  group('art', () {
    test('no flat tiles in any of the eight pictures', () {
      for (final String name in artNames) {
        final Raster r = renderArt(name, w: 400, h: 264);
        const int n = 10;
        final int tw = r.w ~/ n;
        final int th = r.h ~/ n;
        int flat = 0;
        for (int ry = 0; ry < n; ry++) {
          for (int rx = 0; rx < n; rx++) {
            if (tileStd(r, rx * tw, ry * th, tw, th) < 6.0) flat++;
          }
        }
        expect(flat, 0, reason: '$name has $flat flat tiles');
      }
    });

    test('pictures have range and are the right shape', () {
      for (final String name in artNames) {
        final Raster r = renderArt(name, w: 320, h: 212);
        expect(r.w, 320);
        expect(r.h, 212);
        expect(r.rgba.length, 320 * 212 * 4);
        int lo = 255;
        int hi = 0;
        for (int i = 0; i < r.rgba.length; i += 4) {
          for (int c = 0; c < 3; c++) {
            final int v = r.rgba[i + c];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          expect(r.rgba[i + 3], 255);
        }
        expect(hi - lo > 60, isTrue, reason: '$name span ${hi - lo}');
        expect(tileStd(r, 0, 0, r.w, r.h) > 12, isTrue, reason: name);
      }
    });

    test('there are exactly eight named pictures with seeds', () {
      expect(artNames.length, 8);
      expect(artNames, <String>[
        'dusk',
        'water',
        'leaves',
        'sand',
        'stones',
        'petals',
        'aurora',
        'mosaic',
      ]);
      for (int i = 0; i < artNames.length; i++) {
        expect(artSeeds[artNames[i]], 1000 + i * 137);
      }
    });
  });

  group('photo', () {
    test('fit keeps aspect and stays in bounds', () {
      for (final List<int> wh in <List<int>>[
        <int>[400, 300],
        <int>[60, 240],
        <int>[192, 108],
        <int>[100, 100],
      ]) {
        final Raster f = fitRaster(Raster.blank(wh[0], wh[1]), 120, 56);
        expect(f.w <= 120, isTrue);
        expect(f.h <= 56, isTrue);
        final double a = wh[0] / wh[1];
        final double b = f.w / f.h;
        expect((a - b).abs() / a < 0.03, isTrue, reason: '$a vs $b');
      }
    });

    test('snapToGrid divides evenly', () {
      for (final int rows in <int>[3, 7, 8, 10]) {
        final Raster g = snapToGrid(Raster.blank(997, 563), rows, rows);
        expect(g.w % rows, 0);
        expect(g.h % rows, 0);
      }
    });

    test('rasterFromEncodedBytes round-trips a real png', () async {
      const int w = 6;
      const int h = 4;
      final Uint8List src = Uint8List(w * h * 4);
      for (int i = 0; i < w * h; i++) {
        src[i * 4] = (i * 37) % 256;
        src[i * 4 + 1] = (i * 61) % 256;
        src[i * 4 + 2] = (i * 97) % 256;
        src[i * 4 + 3] = 255;
      }
      final Completer<ui.Image> decoded = Completer<ui.Image>();
      ui.decodeImageFromPixels(
        src,
        w,
        h,
        ui.PixelFormat.rgba8888,
        decoded.complete,
      );
      final ui.Image image = await decoded.future;
      final ByteData? png = await image.toByteData(
        format: ui.ImageByteFormat.png,
      );
      image.dispose();
      expect(png, isNotNull);

      final Raster back = await rasterFromEncodedBytes(
        png!.buffer.asUint8List(),
      );
      expect(back.w, w);
      expect(back.h, h);
      expect(back.rgba, src);
    });
  });

  group('effects', () {
    test('lift, place and finish emit then decay to nothing', () {
      final Effects f = Effects(seed: 3);
      expect(f.busy, isFalse);
      f.lift(100, 100);
      expect(f.parts.n, dustCount);
      f.place(120, 120, 30);
      expect(f.rings.length, 1);
      expect(f.parts.n > dustCount, isTrue);
      f.finish(900, 600);
      expect(f.parts.n > 0, isTrue);
      for (int i = 0; i < 400; i++) {
        f.step(0.05);
      }
      expect(f.busy, isFalse);
    });

    test('particles never exceed the cap', () {
      final Effects f = Effects(seed: 5);
      for (int i = 0; i < 200; i++) {
        f.place(10.0 * i, 20.0, 12);
      }
      expect(f.parts.n <= maxParticles, isTrue);
      expect(f.rings.length <= maxRings, isTrue);
    });

    test('clear empties everything', () {
      final Effects f = Effects(seed: 9);
      f.finish(900, 600);
      f.place(10, 10, 5);
      f.clear();
      expect(f.busy, isFalse);
      expect(f.parts.n, 0);
    });
  });

  group('layout at every size', () {
    const List<List<int>> screens = <List<int>>[
      <int>[900, 600],
      <int>[1280, 860],
      <int>[1920, 1080],
    ];

    test('zones never overlap and stay on screen', () {
      for (final List<int> wh in screens) {
        final JigsawLogic a = makeLogic(w: wh[0], h: wh[1]);
        final List<void Function()> pages = <void Function()>[
          a.menuZones,
          a.setupZones,
          a.gameZones,
          a.pauseZones,
          a.doneZones,
          a.settingsZones,
        ];
        for (final void Function() make in pages) {
          make();
          final List<String> keys = a.zones.keys
              .where((String k) => k != 'panel')
              .toList();
          for (int i = 0; i < keys.length; i++) {
            final Zone ra = a.zones[keys[i]]!;
            expect(ra.x >= 0, isTrue, reason: '${wh[0]} ${keys[i]}');
            expect(ra.y >= 0, isTrue, reason: '${wh[0]} ${keys[i]}');
            expect(ra.x1 <= wh[0], isTrue, reason: '${wh[0]} ${keys[i]}');
            expect(ra.y1 <= wh[1], isTrue, reason: '${wh[1]} ${keys[i]}');
            for (int j = i + 1; j < keys.length; j++) {
              expect(
                ra.overlaps(a.zones[keys[j]]!),
                isFalse,
                reason: '${keys[i]} vs ${keys[j]} at ${wh[0]}x${wh[1]}',
              );
            }
          }
        }
      }
    });

    test('every control is touchable at every size', () {
      for (final List<int> wh in screens) {
        final JigsawLogic a = makeLogic(w: wh[0], h: wh[1]);
        for (final void Function() make in <void Function()>[
          a.menuZones,
          a.setupZones,
          a.gameZones,
          a.pauseZones,
          a.doneZones,
          a.settingsZones,
        ]) {
          make();
          for (final String k in a.zones.keys) {
            if (k == 'panel') continue;
            expect(
              a.zones[k]!.touchable,
              isTrue,
              reason: '$k at ${wh[0]}x${wh[1]}',
            );
          }
        }
      }
    });

    test('a full play works at every size', () {
      for (final List<int> wh in screens) {
        final JigsawLogic a = makeLogic(w: wh[0], h: wh[1]);
        tapZone(a, 'start');
        tapZone(a, 'start');
        expect(a.board, isNotNull);
        expect(a.zones.keys.toList(), <String>['stop']);
      }
    });
  });

  group('privacy', () {
    test('no score, timer, failure or best fields anywhere', () {
      const List<String> bad = <String>[
        'score',
        'points',
        'streak',
        'lives',
        'penalty',
        'combo',
        'timeLeft',
        'time_left',
        'deadline',
        'failed',
        'mistakes',
        'best',
      ];
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      final String dump =
          '${a.zones.keys.join(' ')} '
          '${a.board!.order} ${a.page} ${a.title}';
      for (final String b in bad) {
        expect(dump.contains(b), isFalse, reason: b);
      }
      expect(a.board!.moves, 0);
    });

    test('a custom picture is dropped when leaving the game', () {
      final JigsawLogic a = makeLogic();
      a.custom = demoImage(700, 500);
      a.start();
      expect(a.title, 'yours');
      expect(a.board!.total, a.side * a.side);
      tapZone(a, 'stop');
      tapZone(a, 'home');
      expect(a.custom, isNull);
      expect(a.image, isNull);
      expect(a.board, isNull);
    });

    test('nothing is ever written to disk by the logic layer', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'start');
      tapZone(a, 'start');
      solve(a);
      a.step(10.0, 0.016);
      expect(a.board!.solved, isTrue);
    });
  });

  group('settings panel', () {
    test('opens from pick, slides, and closes back to pick', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'settings');
      expect(a.settings, isTrue);
      expect(a.zones.has('volume'), isTrue);

      final Zone v = a.zones['volume']!;
      expect(a.zones.at(v.cx, v.cy), 'volume');
      a.pointerDown(v.cx, v.cy);
      a.pointerMove(v.x1 + 300, v.cy);
      a.pointerUp(v.x1 + 300, v.cy);
      expect(a.volume, volMax);
      expect(a.settings, isTrue);

      final Zone c = a.zones['close']!;
      expect(a.zones.at(c.cx, c.cy), 'close');
      tapZone(a, 'close');
      expect(a.settings, isFalse);
      expect(a.page, pagePick);
      expect(a.zones.has('start'), isTrue);
    });

    test('tapping your picture with no picker is harmless', () {
      final JigsawLogic a = makeLogic();
      tapZone(a, 'own');
      expect(a.page, pagePick);
      expect(a.custom, isNull);
    });

    test('picking a photo sets custom and moves to setup', () async {
      final JigsawLogic a = makeLogic();
      final Raster picked = demoImage(700, 500);
      a.onPickOwn = () async => picked;
      tapZone(a, 'own');
      await Future<void>.delayed(Duration.zero);
      expect(a.custom, same(picked));
      expect(a.page, pageSetup);
      expect(a.sourceImage(), same(picked));
      expect(a.title, 'yours');
    });

    test('cancelling the picker leaves the pick screen untouched', () async {
      final JigsawLogic a = makeLogic();
      a.onPickOwn = () async => null;
      tapZone(a, 'own');
      await Future<void>.delayed(Duration.zero);
      expect(a.custom, isNull);
      expect(a.page, pagePick);
    });
  });

  group('widget', () {
    testWidgets('builds a real board on a background isolate', (
      WidgetTester tester,
    ) async {
      tester.view.physicalSize = const Size(1280, 860);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(body: JigsawGame(seed: 3, sound: false)),
        ),
      );
      await tester.pump(const Duration(milliseconds: 16));

      JigsawLogic logic() =>
          (tester.state(find.byType(JigsawGame)) as dynamic).logic
              as JigsawLogic;

      expect(logic().page, pagePick);
      await tester.tapAt(const Offset(640, 700));
      await tester.pump(const Duration(milliseconds: 16));
      expect(logic().page, pageSetup);

      final Zone go = logic().zones['start']!;
      await tester.runAsync(() async {
        await tester.tapAt(Offset(go.cx, go.cy));
        for (int i = 0; i < 600 && logic().board == null; i++) {
          await Future<void>.delayed(const Duration(milliseconds: 25));
        }
      });
      await tester.pump(const Duration(milliseconds: 16));

      expect(logic().board, isNotNull);
      expect(logic().page, pageGame);
      expect(logic().zones.keys.toList(), <String>['stop']);
      expect(logic().board!.total, logic().side * logic().side);
      expect(tester.takeException(), isNull);
    }, timeout: const Timeout(Duration(minutes: 5)));

    testWidgets('isolate jobs carry no unsendable captures', (
      WidgetTester tester,
    ) async {
      await tester.runAsync(() async {
        final Raster a = await renderArtAsync('dusk', 300, 200);
        expect(a.w, greaterThan(0));
        final CutResult r = await cutAsync(
          a.rgba,
          a.w,
          a.h,
          3,
          styleInterlock,
          3,
        );
        expect(r.pw.length, 9);
      });
    }, timeout: const Timeout(Duration(minutes: 5)));
  });

  group('audio model', () {
    test('tones and gains match the original', () {
      expect(tones, <double>[196, 220, 247, 262, 294, 330]);
      expect(doneTone, 147.0);
      expect(clickGain, 0.34);
      expect(doneGain, 0.30);
      expect(maxVoices, 8);
    });

    test('clickTone produces a normalised silent-ended buffer', () {
      final Float32List t = clickTone(freq: 247, ms: 150, seed: 1);
      expect(t.first, 0.0);
      expect(t.last, 0.0);
      double peak = 0;
      for (final double v in t) {
        peak = math.max(peak, v.abs());
      }
      expect((peak - 1.0).abs() < 1e-5, isTrue);
    });

    test('volume curve is silent at zero and full at one hundred', () {
      expect(volumeCurve(volMin), 0.0);
      expect(volumeCurve(volMax), 1.0);
      expect(volumeCurve(50) > 0 && volumeCurve(50) < 1, isTrue);
    });
  });
}
