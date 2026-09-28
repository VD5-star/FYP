import 'dart:math' as math;

import 'pieces.dart';
import 'raster.dart';

const double snapRatio = 0.34;
const double snapMin = 14.0;
const double settleSeconds = 0.22;

double snapRadius(Cut cut) =>
    math.max(snapMin, math.min(cut.cellW, cut.cellH) * snapRatio);

class Placement {
  Placement(this.piece, this.at);
  final Piece piece;
  final double at;
}

class Tray {
  const Tray(this.x, this.y, this.w, this.h);
  final int x;
  final int y;
  final int w;
  final int h;
}

class _Rand {
  _Rand(int seed) : _s = (seed & 0x7FFFFFFF) | 1;
  int _s;
  int next() {
    _s = (_s * 1103515245 + 12345) & 0x7FFFFFFF;
    return _s;
  }

  int range(int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (next() >> 8) % (hi - lo + 1);
  }

  void shuffle(List<int> a) {
    for (int i = a.length - 1; i > 0; i--) {
      final int j = (next() >> 8) % (i + 1);
      final int t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
}

class Board {
  Board({
    required this.cut,
    required this.pieces,
    required this.originX,
    required this.originY,
    required this.tray,
  });

  final Cut cut;
  final List<Piece> pieces;
  final int originX;
  final int originY;
  final Tray tray;

  List<int> order = <int>[];
  int? held;
  int grabX = 0;
  int grabY = 0;
  final List<Placement> settled = <Placement>[];
  int moves = 0;
  double lastMove = 0.0;
  final List<double> gaps = <double>[];

  double get radius => snapRadius(cut);
  int get total => pieces.length;
  int get done => pieces.where((Piece p) => p.placed).length;
  bool get solved => done == total;

  int targetX(Piece p) => originX + p.homeX;
  int targetY(Piece p) => originY + p.homeY;

  void scatter({int? seed}) {
    final _Rand rng = _Rand(seed ?? 1);
    for (final Piece p in pieces) {
      p.x = rng.range(tray.x, math.max(tray.x, tray.x + tray.w - p.w));
      p.y = rng.range(tray.y, math.max(tray.y, tray.y + tray.h - p.h));
      p.placed = false;
    }
    order = List<int>.generate(pieces.length, (int i) => i);
    rng.shuffle(order);
  }

  int reshuffle({int? seed}) {
    final _Rand rng = _Rand(seed ?? 1);
    int moved = 0;
    held = null;
    for (final Piece p in pieces) {
      if (p.placed) continue;
      p.x = rng.range(tray.x, math.max(tray.x, tray.x + tray.w - p.w));
      p.y = rng.range(tray.y, math.max(tray.y, tray.y + tray.h - p.h));
      moved++;
    }
    final List<int> loose =
        order.where((int i) => !pieces[i].placed).toList();
    rng.shuffle(loose);
    int k = 0;
    order = order
        .map((int i) => pieces[i].placed ? i : loose[k++])
        .toList();
    return moved;
  }

  bool pick(num x, num y) {
    for (int i = order.length - 1; i >= 0; i--) {
      final int idx = order[i];
      final Piece p = pieces[idx];
      if (p.placed) continue;
      if (p.hits(x, y)) {
        held = idx;
        grabX = (x - p.x).round();
        grabY = (y - p.y).round();
        order.removeAt(i);
        order.add(idx);
        return true;
      }
    }
    return false;
  }

  void drag(num x, num y) {
    final int? h = held;
    if (h == null) return;
    final Piece p = pieces[h];
    p.x = (x - grabX).round();
    p.y = (y - grabY).round();
  }

  bool drop({double? now}) {
    final int? h = held;
    if (h == null) return false;
    final double t = now ?? _clock();
    final Piece p = pieces[h];
    held = null;
    moves++;
    if (lastMove != 0.0) gaps.add(t - lastMove);
    lastMove = t;

    final double dx = (p.x - targetX(p)).toDouble();
    final double dy = (p.y - targetY(p)).toDouble();
    if (math.sqrt(dx * dx + dy * dy) <= radius) {
      p.x = targetX(p);
      p.y = targetY(p);
      p.placed = true;
      settled.add(Placement(p, t));
      order.remove(h);
      return true;
    }
    return false;
  }

  double settleAge(Piece piece, double now) {
    for (final Placement s in settled) {
      if (identical(s.piece, piece)) return now - s.at;
    }
    return 999.0;
  }
}

double _clock() =>
    DateTime.now().microsecondsSinceEpoch / 1000000.0;

Board buildBoard(
  Raster image,
  int rows,
  int cols,
  int originX,
  int originY,
  Tray tray, {
  String style = styleInterlock,
  int? seed,
}) {
  final Cut cut =
      makeCut(rows, cols, image.w, image.h, style: style, seed: seed);
  final List<Piece> ps = cutImage(image, cut);
  final Board b = Board(
    cut: cut,
    pieces: ps,
    originX: originX,
    originY: originY,
    tray: tray,
  );
  b.scatter(seed: seed);
  return b;
}
