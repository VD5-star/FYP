import 'dart:math' as math;

import '../core/paint.dart';
import '../core/synth.dart';
import '../core/zones.dart';
import 'art.dart';
import 'board.dart';
import 'fx.dart';
import 'photo.dart';
import 'pieces.dart';
import 'raster.dart';

const String pagePick = 'pick';
const String pageSetup = 'setup';
const String pageGame = 'game';

const List<int> sizeSteps = <int>[3, 4, 5, 6, 7, 8, 10];
const double trayFrac = 0.30;
const double doneFade = 1.6;
const double minPiecePx = 52.0;

const List<List<Object>> pieceWords = <List<Object>>[
  <Object>[9, 'gentle', 'a few big pieces'],
  <Object>[25, 'easy', 'room to see the picture'],
  <Object>[49, 'steady', 'a real puzzle'],
  <Object>[64, 'busy', 'small pieces, long sit'],
  <Object>[100, 'deep', 'the longest one'],
];

List<String> pieceShape(int count) {
  for (final List<Object> row in pieceWords) {
    if (count <= (row[0] as int)) {
      return <String>[row[1] as String, row[2] as String];
    }
  }
  final List<Object> last = pieceWords.last;
  return <String>[last[1] as String, last[2] as String];
}

typedef ArtSource = Raster Function(String name, int w, int h);

Raster defaultArtSource(String name, int w, int h) {
  const double ratio = artWidth / artHeight;
  int rw;
  int rh;
  if (w / h > ratio) {
    rw = w;
    rh = (w / ratio).round();
  } else {
    rw = (h * ratio).round();
    rh = h;
  }
  return renderArt(name, w: rw, h: rh);
}

class Layout {
  const Layout(this.tray, this.areaW, this.areaH);
  final Tray tray;
  final int areaW;
  final int areaH;
}

class JigsawLogic {
  JigsawLogic({
    int width = 1280,
    int height = 860,
    this.style = styleInterlock,
    this.seed,
    ArtSource? artSource,
  })  : w = width,
        h = height,
        _art = artSource ?? defaultArtSource {
    menuZones();
  }

  int w;
  int h;
  String style;
  final int? seed;
  final ArtSource _art;

  final List<String> names = List<String>.from(artNames);
  int pick = 0;
  int sizePick = 2;
  String page = pagePick;
  String title = '';

  Board? board;
  Raster? image;
  Raster? custom;

  final Zones zones = Zones();
  final Effects fx = Effects(seed: 7);
  final Map<String, Raster> thumbs = <String, Raster>{};
  final Map<String, Raster> artCache = <String, Raster>{};

  bool paused = false;
  bool settings = false;
  bool sliding = false;
  bool dragging = false;
  double doneAt = 0.0;
  int volume = volDefault;

  double? pressX;
  double? pressY;
  String? pressKey;

  void resize(int width, int height) {
    if (width == w && height == h) return;
    w = width;
    h = height;
    rebuildZones();
  }

  Layout layout() {
    final int trayH = (h * trayFrac).toInt();
    final int areaH = h - trayH - 76;
    final int areaW = w - 80;
    return Layout(Tray(0, h - trayH, w, trayH), areaW, areaH);
  }

  List<int> sizeOptions() {
    final Layout l = layout();
    final List<int> out = <int>[];
    for (final int s in sizeSteps) {
      if (math.min(l.areaW / s, l.areaH / s) >= minPiecePx) out.add(s);
    }
    return out.isEmpty ? <int>[sizeSteps.first] : out;
  }

  int get side => sizeSteps[sizePick];

  void menuZones() {
    zones.clear();
    const int cols = 3;
    const int rows = 3;
    const int pad = 20;
    const int bh = 58;
    const int top = 96;
    const int foot = bh + 34;
    final int roomH = h - top - foot - 24;
    final int roomW = w - 60;
    int ch = (roomH - pad * (rows - 1)) ~/ rows;
    int cw = (ch / 0.62).toInt();
    if (cw * cols + pad * (cols - 1) > roomW) {
      cw = (roomW - pad * (cols - 1)) ~/ cols;
      ch = (cw * 0.62).toInt();
    }
    final int gw = cw * cols + pad * (cols - 1);
    final int gh = ch * rows + pad * (rows - 1);
    final int x0 = (w - gw) ~/ 2;

    final int n = math.min(names.length, cols * rows - 1);
    for (int i = 0; i < n; i++) {
      final int r = i ~/ cols;
      final int c = i % cols;
      zones.add(
        'pic:$i',
        Zone((x0 + c * (cw + pad)).toDouble(),
            (top + r * (ch + pad)).toDouble(), cw.toDouble(),
            ch.toDouble()),
      );
    }
    final int r = n ~/ cols;
    final int c = n % cols;
    zones.add(
      'own',
      Zone((x0 + c * (cw + pad)).toDouble(),
          (top + r * (ch + pad)).toDouble(), cw.toDouble(),
          ch.toDouble()),
    );

    final int y = top + gh + 24;
    const int bw = 190;
    const int gap = 20;
    final int sx = (w - (bw * 2 + gap)) ~/ 2;
    zones.add('settings',
        Zone(sx.toDouble(), y.toDouble(), bw.toDouble(), bh.toDouble()));
    zones.add(
        'start',
        Zone((sx + bw + gap).toDouble(), y.toDouble(), bw.toDouble(),
            bh.toDouble()));
  }

  void setupZones() {
    zones.clear();
    final List<int> avail = sizeOptions();
    if (sizePick >= sizeSteps.length || !avail.contains(side)) {
      sizePick = sizeSteps.indexOf(avail.last);
    }
    final double sw = math.min(460.0, w - 140.0);
    final double y = (h * 0.44).floorToDouble();
    zones.add('pieces', Zone((w - sw) / 2, y, sw, 56));
    const double bw = 180;
    const double bh = 56;
    const double gap = 20;
    final double x0 = (w - (bw * 2 + gap)) / 2;
    zones.add('square', Zone(x0, y + 128, bw, bh));
    zones.add('interlock', Zone(x0 + bw + gap, y + 128, bw, bh));
    zones.add('start', Zone((w - 220) / 2, y + 206, 220, 62));
    zones.add('back', const Zone(26, 26, 140, 50));
  }

  void gameZones() {
    zones.clear();
    zones.add('stop', stopZone(w.toDouble(), h.toDouble()));
  }

  void pauseZones() {
    zones.clear();
    final PauseLayout p = PauseLayout(w.toDouble(), h.toDouble());
    zones.add('panel', p.panel);
    zones.add('volume', p.volume);
    zones.add('resume', p.topLeft);
    zones.add('again', p.topRight);
    zones.add('shuffle', p.bottomLeft);
    zones.add('home', p.bottomRight);
  }

  void doneZones() {
    zones.clear();
    zones.add('again', Zone(w / 2 - 200, h / 2 + 16, 180, 54));
    zones.add('back', Zone(w / 2 + 20, h / 2 + 16, 180, 54));
  }

  void settingsZones() {
    zones.clear();
    final double pw = math.min(520.0, w - 80.0);
    const double ph = 260;
    final double px = (w - pw) / 2;
    final double py = (h - ph) / 2;
    zones.add('panel', Zone(px, py, pw, ph));
    zones.add('volume', Zone(px + 30, py + 112, pw - 60, 56));
    zones.add('close', Zone(w / 2 - 85, py + ph - 76, 170, 54));
  }

  void rebuildZones() {
    if (settings) {
      settingsZones();
    } else if (paused) {
      pauseZones();
    } else if (page == pagePick) {
      menuZones();
    } else if (page == pageSetup) {
      setupZones();
    } else if (solvedView()) {
      doneZones();
    } else {
      gameZones();
    }
  }

  Raster artAt(String name, int aw, int ah) {
    final String key = '$name:$aw:$ah';
    final Raster? hit = artCache[key];
    if (hit != null) return hit;
    final Raster made = _art(name, aw, ah);
    artCache.clear();
    artCache[key] = made;
    return made;
  }

  void cacheArt(String name, int aw, int ah, Raster r) {
    artCache['$name:$aw:$ah'] = r;
  }

  bool artReady(String name, int aw, int ah) =>
      artCache.containsKey('$name:$aw:$ah');

  Raster sourceImage() {
    final Layout l = layout();
    if (custom != null) {
      title = 'yours';
      return custom!;
    }
    title = names[pick];
    return artAt(title, l.areaW, l.areaH);
  }

  Raster fitForCut(Raster src) {
    final Layout l = layout();
    return snapToGrid(fitRaster(src, l.areaW, l.areaH), side, side);
  }

  void startWith(Raster img, Board made) {
    image = img;
    board = made;
    doneAt = 0.0;
    paused = false;
    sliding = false;
    fx.clear();
    page = pageGame;
    gameZones();
  }

  void start() {
    final Raster img = fitForCut(sourceImage());
    startWith(
      img,
      buildBoard(img, side, side, (w - img.w) ~/ 2, 52, layout().tray,
          style: style, seed: seed),
    );
  }

  void Function()? onStart;

  void _requestStart() {
    final void Function()? f = onStart;
    if (f != null) {
      f();
      return;
    }
    start();
  }

  void toMenu() {
    board = null;
    image = null;
    custom = null;
    doneAt = 0.0;
    paused = false;
    sliding = false;
    page = pagePick;
    fx.clear();
    menuZones();
  }

  void toSetup() {
    board = null;
    image = null;
    doneAt = 0.0;
    paused = false;
    sliding = false;
    page = pageSetup;
    fx.clear();
    setupZones();
  }

  void openPause() {
    final Board? b = board;
    if (paused || b == null || b.solved) return;
    paused = true;
    sliding = false;
    pauseZones();
  }

  void closePause() {
    paused = false;
    sliding = false;
    gameZones();
  }

  void openSettings() {
    if (settings) return;
    settings = true;
    sliding = false;
    settingsZones();
  }

  void closeSettings() {
    settings = false;
    sliding = false;
    rebuildZones();
  }

  bool solvedView() =>
      board != null && board!.solved && doneAt > 0.0;

  void setSize(double x) {
    final Zone? r = zones['pieces'];
    if (r == null) return;
    final List<int> opts = sizeOptions();
    final int i = sliderValue(r, x, lo: 0, hi: opts.length - 1);
    sizePick = sizeSteps.indexOf(opts[i]);
  }

  void setVolume(double x) {
    final Zone? r = zones['volume'];
    if (r == null) return;
    volume = sliderValue(r, x, lo: volMin, hi: volMax);
    onVolume?.call(volume);
  }

  void setVolumeValue(int v) {
    volume = v.clamp(volMin, volMax);
    onVolume?.call(volume);
  }

  set side(int s) {
    final int idx = sizeSteps.indexOf(s);
    if (idx != -1) sizePick = idx;
  }

  void Function(int v)? onVolume;
  void Function(double frac, double pan)? onPlaceSound;
  void Function()? onDoneSound;
  Future<Raster?> Function()? onPickOwn;

  String get soundName => volume <= volMin ? 'sound off' : 'sound $volume';

  void tapMenu(String key) {
    if (key.startsWith('pic:')) {
      pick = int.parse(key.split(':')[1]);
      custom = null;
      toSetup();
    } else if (key == 'settings') {
      openSettings();
    } else if (key == 'start') {
      custom = null;
      toSetup();
    } else if (key == 'own') {
      openOwn();
    }
  }

  void openOwn() {
    final Future<Raster?> Function()? f = onPickOwn;
    if (f == null) return;
    f().then((Raster? r) {
      if (r == null) return;
      custom = r;
      toSetup();
      onChange?.call();
    });
  }

  void Function()? onChange;

  void tapSetup(String key) {
    if (key == 'square') {
      style = styleSquare;
    } else if (key == 'interlock') {
      style = styleInterlock;
    } else if (key == 'start') {
      _requestStart();
    } else if (key == 'back') {
      toMenu();
    }
  }

  void tapPause(String key) {
    if (key == 'resume') {
      closePause();
    } else if (key == 'again') {
      _requestStart();
    } else if (key == 'shuffle') {
      board?.reshuffle(seed: seed);
      closePause();
    } else if (key == 'home') {
      toMenu();
    }
  }

  void tapGame(String key) {
    if (key == 'stop') openPause();
  }

  void tapDone(String key) {
    if (key == 'again') {
      _requestStart();
    } else if (key == 'back') {
      toMenu();
    }
  }

  void pointerDown(double x, double y) {
    pressX = x;
    pressY = y;
    pressKey = zones.at(x, y);
    dragging = false;

    if (settings) {
      if (pressKey == 'volume') {
        sliding = true;
        setVolume(x);
      }
      return;
    }
    if (page == pageSetup && pressKey == 'pieces') {
      sliding = true;
      setSize(x);
      return;
    }
    if (paused) {
      if (pressKey == 'volume') {
        sliding = true;
        setVolume(x);
      }
      return;
    }
    final Board? b = board;
    if (b != null && !b.solved && pressKey == null) {
      dragging = b.pick(x, y);
      if (dragging) fx.lift(x, y);
    }
  }

  void pointerMove(double x, double y) {
    if (sliding) {
      if (settings || paused) {
        setVolume(x);
      } else if (page == pageSetup) {
        setSize(x);
      }
      return;
    }
    if (dragging) board?.drag(x, y);
  }

  void pointerUp(double x, double y) {
    if (sliding) {
      if (settings || paused) {
        setVolume(x);
      } else if (page == pageSetup) {
        setSize(x);
      }
      sliding = false;
      pressX = null;
      pressY = null;
      pressKey = null;
      return;
    }

    if (dragging) {
      final Board? b = board;
      if (b != null) {
        final int? h0 = b.held;
        final Piece? held = h0 == null ? null : b.pieces[h0];
        if (b.drop()) {
          final double frac = b.done / math.max(1, b.total);
          double pan = 0.0;
          if (held != null) {
            final double cx = held.x + held.w * 0.5;
            final double cy = held.y + held.h * 0.5;
            final double rad =
                math.min(b.cut.cellW, b.cut.cellH) * 0.5;
            fx.place(cx, cy, rad, strength: 0.8 + frac * 0.5);
            pan = ((cx / math.max(1, w)) * 2.0 - 1.0).clamp(-1.0, 1.0);
          }
          onPlaceSound?.call(frac, pan * 0.6);
        }
      }
      dragging = false;
      pressX = null;
      pressY = null;
      pressKey = null;
      return;
    }

    final double? px = pressX;
    final double? py = pressY;
    if (px == null || py == null) return;
    final double moved = (x - px).abs() + (y - py).abs();
    final String? key = zones.at(x, y);
    if (moved <= tapSlop && key != null && key == pressKey) {
      if (settings) {
        if (key == 'close') closeSettings();
      } else if (paused) {
        if (key != 'panel') tapPause(key);
      } else if (page == pagePick) {
        tapMenu(key);
      } else if (page == pageSetup) {
        tapSetup(key);
      } else if (solvedView()) {
        tapDone(key);
      } else {
        tapGame(key);
      }
    }
    pressX = null;
    pressY = null;
    pressKey = null;
  }

  void step(double now, double dt) {
    fx.step(dt.clamp(0.0, 0.05));
    final Board? b = board;
    if (b == null || !b.solved || page != pageGame) return;
    if (doneAt == 0.0) {
      doneAt = now;
      doneZones();
      fx.finish(w, h);
      onDoneSound?.call();
    }
  }

  double doneAlpha(double now) =>
      doneAt == 0.0 ? 0.0 : math.min(1.0, (now - doneAt) / doneFade);
}
