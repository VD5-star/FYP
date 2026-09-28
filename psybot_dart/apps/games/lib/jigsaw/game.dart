import 'dart:async';
import 'dart:isolate';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import '../core/paint.dart';
import 'audio.dart';
import 'board.dart';
import 'draw.dart';
import 'logic.dart';
import 'own_photo.dart';
import 'pieces.dart';
import 'raster.dart';

class ArtRequest {
  const ArtRequest(this.name, this.w, this.h);
  final String name;
  final int w;
  final int h;
}

class ArtResult {
  const ArtResult(this.w, this.h, this.rgba);
  final int w;
  final int h;
  final Uint8List rgba;
}

ArtResult renderArtJob(ArtRequest req) {
  final Raster r = defaultArtSource(req.name, req.w, req.h);
  return ArtResult(r.w, r.h, r.rgba);
}

Future<Raster> renderArtAsync(String name, int w, int h) async {
  final String n = name;
  final int aw = w;
  final int ah = h;
  final ArtResult r = await Isolate.run(
    () => renderArtJob(ArtRequest(n, aw, ah)),
  );
  return Raster(r.w, r.h, r.rgba);
}

class CutRequest {
  const CutRequest(this.rgba, this.w, this.h, this.side, this.style, this.seed);
  final Uint8List rgba;
  final int w;
  final int h;
  final int side;
  final String style;
  final int? seed;
}

class CutResult {
  const CutResult(
    this.rgba,
    this.w,
    this.h,
    this.sprites,
    this.alphas,
    this.pw,
    this.ph,
    this.homeX,
    this.homeY,
  );
  final Uint8List rgba;
  final int w;
  final int h;
  final List<Uint8List> sprites;
  final List<Uint8List> alphas;
  final List<int> pw;
  final List<int> ph;
  final List<int> homeX;
  final List<int> homeY;
}

Future<CutResult> cutAsync(
  Uint8List rgba,
  int w,
  int h,
  int side,
  String style,
  int? seed,
) {
  return Isolate.run(() => cutJob(CutRequest(rgba, w, h, side, style, seed)));
}

CutResult cutJob(CutRequest r) {
  final Raster img = Raster(r.w, r.h, r.rgba);
  final Cut cut = makeCut(
    r.side,
    r.side,
    img.w,
    img.h,
    style: r.style,
    seed: r.seed,
  );
  final List<Piece> ps = cutImage(img, cut);
  return CutResult(
    img.rgba,
    img.w,
    img.h,
    ps.map((Piece p) => p.sprite.rgba).toList(),
    ps.map((Piece p) => p.alpha).toList(),
    ps.map((Piece p) => p.w).toList(),
    ps.map((Piece p) => p.h).toList(),
    ps.map((Piece p) => p.homeX).toList(),
    ps.map((Piece p) => p.homeY).toList(),
  );
}

class JigsawGame extends StatefulWidget {
  const JigsawGame({super.key, this.seed, this.sound = true});

  final int? seed;
  final bool sound;

  @override
  State<JigsawGame> createState() => _JigsawGameState();
}

class _JigsawGameState extends State<JigsawGame>
    with SingleTickerProviderStateMixin {
  late final JigsawLogic logic;
  late final Ticker _ticker;
  final ValueNotifier<int> _tick = ValueNotifier<int>(0);

  JigsawAudio? audio;
  List<PieceArt?> pieceArt = <PieceArt?>[];
  ui.Image? reference;
  ui.Image? setupThumb;
  final Map<String, ui.Image> thumbs = <String, ui.Image>{};
  final Set<String> _warming = <String>{};
  int _boardStamp = 0;
  double _last = 0.0;
  Size _size = Size.zero;
  bool building = false;

  @override
  void initState() {
    super.initState();
    logic = JigsawLogic(seed: widget.seed);
    logic.onChange = () => setState(() {});
    logic.onVolume = (int v) => audio?.setVolume(v);
    logic.onPlaceSound = (double f, double pan) =>
        audio?.place(doneFrac: f, pan: pan);
    logic.onDoneSound = () => audio?.done();
    logic.onStart = _startAsync;
    logic.onPickOwn = pickOwnPhoto;
    _ticker = createTicker(_frame)..start();
    if (widget.sound) _initAudio();
    _buildThumbs();
  }

  Future<void> _startAsync() async {
    if (building) return;
    building = true;
    final Layout l = logic.layout();
    final int side = logic.side;
    final String style = logic.style;
    final int? seed = widget.seed;
    try {
      Raster src;
      final Raster? c = logic.custom;
      if (c != null) {
        logic.title = 'yours';
        src = c;
      } else {
        final String name = logic.names[logic.pick];
        logic.title = name;
        src = logic.artReady(name, l.areaW, l.areaH)
            ? logic.artAt(name, l.areaW, l.areaH)
            : await renderArtAsync(name, l.areaW, l.areaH);
        logic.cacheArt(name, l.areaW, l.areaH, src);
      }
      final Raster fitted = logic.fitForCut(src);
      final CutResult r = await cutAsync(
        fitted.rgba,
        fitted.w,
        fitted.h,
        side,
        style,
        seed,
      );
      if (!mounted) return;
      final Raster img = Raster(r.w, r.h, r.rgba);
      final List<Piece> ps = <Piece>[];
      for (int i = 0; i < r.pw.length; i++) {
        ps.add(
          Piece(
            row: i ~/ side,
            col: i % side,
            sprite: Raster(r.pw[i], r.ph[i], r.sprites[i]),
            alpha: r.alphas[i],
            homeX: r.homeX[i],
            homeY: r.homeY[i],
          ),
        );
      }
      final Cut cut = makeCut(
        side,
        side,
        img.w,
        img.h,
        style: style,
        seed: seed,
      );
      final Board b = Board(
        cut: cut,
        pieces: ps,
        originX: (logic.w - img.w) ~/ 2,
        originY: 52,
        tray: l.tray,
      );
      b.scatter(seed: seed);
      setState(() => logic.startWith(img, b));
      _after();
    } finally {
      building = false;
    }
  }

  Future<void> _initAudio() async {
    final JigsawAudio a = JigsawAudio(seed: widget.seed ?? 0);
    await a.load();
    a.setVolume(logic.volume);
    if (mounted) audio = a;
  }

  @override
  void dispose() {
    _ticker.dispose();
    _tick.dispose();
    audio?.dispose();
    for (final PieceArt? p in pieceArt) {
      p?.image.dispose();
    }
    reference?.dispose();
    setupThumb?.dispose();
    for (final ui.Image i in thumbs.values) {
      i.dispose();
    }
    super.dispose();
  }

  void _frame(Duration d) {
    final double now = d.inMicroseconds / 1000000.0;
    final double dt = _last == 0.0 ? 0.0 : now - _last;
    _last = now;
    final bool wasDone = logic.doneAt > 0.0;
    logic.step(now, dt);
    if (logic.doneAt > 0.0 != wasDone) setState(() {});
    _tick.value = _tick.value + 1;
  }

  Future<void> _buildThumbs() async {
    for (final String n in logic.names) {
      if (thumbs.containsKey(n)) continue;
      final Raster r = await renderArtAsync(n, 260, 162);
      final ui.Image img = await imageFromPixels(r.rgba, r.w, r.h);
      if (!mounted) {
        img.dispose();
        return;
      }
      setState(() => thumbs[n] = img);
    }
  }

  void _warm(String name) {
    final Layout l = logic.layout();
    final String key = '$name:${l.areaW}:${l.areaH}';
    if (logic.artReady(name, l.areaW, l.areaH)) return;
    if (_warming.contains(key)) return;
    _warming.add(key);
    renderArtAsync(name, l.areaW, l.areaH).then((Raster r) {
      _warming.remove(key);
      if (!mounted) return;
      logic.cacheArt(name, l.areaW, l.areaH, r);
    });
  }

  Future<void> _syncBoard() async {
    final Board? b = logic.board;
    final int stamp = identityHashCode(b);
    if (stamp == _boardStamp) return;
    _boardStamp = stamp;
    for (final PieceArt? p in pieceArt) {
      p?.image.dispose();
    }
    pieceArt = <PieceArt?>[];
    reference?.dispose();
    reference = null;
    if (b == null) return;

    final Raster? img = logic.image;
    if (img != null) {
      reference = await imageFromPixels(img.rgba, img.w, img.h);
    }
    final List<PieceArt?> made = List<PieceArt?>.filled(b.pieces.length, null);
    for (int i = 0; i < b.pieces.length; i++) {
      final Piece p = b.pieces[i];
      final Uint8List rgba = Uint8List(p.w * p.h * 4);
      for (int k = 0; k < p.w * p.h; k++) {
        rgba[k * 4] = p.sprite.rgba[k * 4];
        rgba[k * 4 + 1] = p.sprite.rgba[k * 4 + 1];
        rgba[k * 4 + 2] = p.sprite.rgba[k * 4 + 2];
        rgba[k * 4 + 3] = p.alpha[k];
      }
      made[i] = PieceArt(await imageFromPixels(rgba, p.w, p.h), p.w, p.h);
    }
    if (!mounted) {
      for (final PieceArt? p in made) {
        p?.image.dispose();
      }
      return;
    }
    setState(() => pieceArt = made);
  }

  Future<void> _syncSetupThumb() async {
    if (logic.page != pageSetup) return;
    final Raster? c = logic.custom;
    if (c != null) {
      setupThumb?.dispose();
      setupThumb = await imageFromPixels(c.rgba, c.w, c.h);
      if (mounted) setState(() {});
      return;
    }
    final ui.Image? t = thumbs[logic.names[logic.pick]];
    if (t != null && setupThumb != t) {
      setupThumb = t;
      if (mounted) setState(() {});
    }
  }

  void _after() {
    _syncBoard();
    _syncSetupThumb();
    if (logic.page == pagePick || logic.page == pageSetup) {
      if (logic.custom == null) _warm(logic.names[logic.pick]);
    }
  }

  void _down(PointerDownEvent e) {
    setState(() => logic.pointerDown(e.localPosition.dx, e.localPosition.dy));
    _after();
  }

  void _move(PointerMoveEvent e) {
    logic.pointerMove(e.localPosition.dx, e.localPosition.dy);
  }

  void _up(PointerUpEvent e) {
    setState(() => logic.pointerUp(e.localPosition.dx, e.localPosition.dy));
    _after();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints c) {
        final Size s = Size(c.maxWidth, c.maxHeight);
        if (s != _size && s.width > 0 && s.height > 0) {
          _size = s;
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (!mounted) return;
            setState(() => logic.resize(s.width.toInt(), s.height.toInt()));
          });
        }
        return Listener(
          behavior: HitTestBehavior.opaque,
          onPointerDown: _down,
          onPointerMove: _move,
          onPointerUp: _up,
          onPointerCancel: (PointerCancelEvent e) =>
              logic.pointerUp(e.localPosition.dx, e.localPosition.dy),
          child: CustomPaint(
            size: s,
            painter: JigsawPainter(
              logic: logic,
              now: _last,
              pieceArt: pieceArt,
              reference: reference,
              thumbs: thumbs,
              setupThumb: setupThumb,
              repaint: _tick,
            ),
          ),
        );
      },
    );
  }
}

class JigsawApp extends StatelessWidget {
  const JigsawApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'jigsaw',
      debugShowCheckedModeBanner: false,
      home: const Scaffold(
        backgroundColor: Color(0xFF222A2A),
        body: JigsawGame(),
      ),
    );
  }
}
