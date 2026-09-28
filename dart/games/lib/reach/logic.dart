import 'dart:math' as math;

import '../core/paint.dart';
import '../core/synth.dart';
import '../core/zones.dart';
import 'angles.dart';
import 'body.dart';
import 'engine.dart';
import 'points.dart';
import 'cameras.dart';
import 'pose_source.dart';
import 'targets.dart';

const int minMinutes = 1;
const int maxMinutes = 10;
const int defaultMinutes = 1;

const double summaryFade = 0.9;

const List<List<Object>> shapes = <List<Object>>[
  <Object>[2, 'quick', 'a short warm up'],
  <Object>[4, 'steady', 'long enough to find a rhythm'],
  <Object>[7, 'full', 'a proper session'],
  <Object>[10, 'long', 'settle in and keep moving'],
];

int clampMinutes(int m) => m.clamp(minMinutes, maxMinutes);

List<String> shape(int minutes) {
  final int m = clampMinutes(minutes);
  for (final List<Object> row in shapes) {
    if (m <= (row[0] as int)) {
      return <String>[row[1] as String, row[2] as String];
    }
  }
  final List<Object> last = shapes.last;
  return <String>[last[1] as String, last[2] as String];
}

const String pageMenu = 'menu';
const String pagePlay = 'play';

String clockText(double seconds) {
  final int s = seconds.toInt();
  return '${s ~/ 60}:${(s % 60).toString().padLeft(2, '0')}';
}

List<List<String>> summaryStats(Summary summary, String mode) {
  final List<List<String>> stats = <List<String>>[
    <String>['touched', '${summary.hits}'],
    <String>['time', clockText(summary.seconds)],
  ];
  if (mode == timed) {
    stats.insert(0, <String>['score', '${summary.score}']);
    stats.add(<String>['bonus', '${summary.bonus}']);
  }
  return stats;
}

String bandBreakdown(Summary summary) {
  if (summary.bands.isEmpty) return '';
  final List<String> keys = summary.bands.keys.toList()..sort();
  return keys.map((String k) => '$k ${summary.bands[k]}').join('   ');
}

const double cameraRowHeight = 52;
const double cameraRowGap = 12;
const double menuBottomPad = 20;

Zones menuZones(double w, double h, String mode, {bool camera = false}) {
  final Zones z = Zones();
  final double sw = math.min(460.0, w - 120.0);
  final double needed = camera
      ? 252 + cameraRowGap + cameraRowHeight + menuBottomPad
      : 252 + menuBottomPad;
  final double y =
      math.max(0.0, math.min((h * 0.42).floorToDouble(), h - needed));
  if (mode != calm) {
    z.add('minutes', Zone((w - sw) / 2, y, sw, 56));
  }
  const double bw = 150;
  const double bh = 54;
  const double gap = 20;
  final double x0 = (w - (bw * 2 + gap)) / 2;
  z.add('timed', Zone(x0, y + 112, bw, bh));
  z.add('calm', Zone(x0 + bw + gap, y + 112, bw, bh));
  z.add('begin', Zone((w - 220) / 2, y + 190, 220, 62));
  if (camera) {
    final double cw = math.min(260.0, w - 80.0);
    z.add(
      'camera',
      Zone((w - cw) / 2, y + 252 + cameraRowGap, cw, cameraRowHeight),
    );
  }
  return z;
}

Zones playZones(double w, double h) {
  final Zones z = Zones();
  z.add('stop', stopZone(w, h));
  return z;
}

Zones pauseZones(double w, double h) {
  final Zones z = Zones();
  final double pw = math.min(520.0, w - 80.0);
  const double ph = 250;
  final double px = (w - pw) / 2;
  final double py = (h - ph) / 2;
  z.add('panel', Zone(px, py, pw, ph));
  final double bw = math.min(190.0, (pw - 66) / 2);
  const double bh = 58;
  const double gap = 22;
  final double x0 = (w - (bw * 2 + gap)) / 2;
  z.add('resume', Zone(x0, py + ph - 96, bw, bh));
  z.add('finish', Zone(x0 + bw + gap, py + ph - 96, bw, bh));
  return z;
}

Zones summaryZones(double w, double h) {
  final Zones z = Zones();
  const double gap = 20;
  final double bw = math.min(200.0, (w - 80 - gap * 2) / 3);
  const double bh = 58;
  final double total = bw * 3 + gap * 2;
  final double x0 = (w - total) / 2;
  final double y = math.min((h * 0.72).floorToDouble(), h - bh - 20);
  z.add('again', Zone(x0, y, bw, bh));
  z.add('menu', Zone(x0 + bw + gap, y, bw, bh));
  z.add('close', Zone(x0 + (bw + gap) * 2, y, bw, bh));
  return z;
}

class ReachLogic {
  ReachLogic({
    double width = 1280,
    double height = 720,
    int minutes = defaultMinutes,
    String mode = timed,
    int? seed,
    this.trackingNote = '',
  })  : w = width,
        h = height,
        _minutes = clampMinutes(minutes),
        _mode = mode,
        engine = ReachEngine(
            mode: mode, duration: clampMinutes(minutes) * 60.0, seed: seed) {
    rebuildZones();
  }

  double w;
  double h;
  int _minutes;
  String _mode;
  String trackingNote;

  final ReachEngine engine;
  final BodyTracker tracker = BodyTracker();
  final Zones zones = Zones();

  String page = pageMenu;
  bool sliding = false;
  int volume = volDefault;
  double now = 0.0;
  double? finishedAt;
  bool closed = false;
  bool tracking = false;

  List<CameraChoice> cameras = const <CameraChoice>[];
  int cameraIndex = -1;
  Future<bool> Function(int index)? onPickCamera;

  bool get hasCameraChoice => cameras.length > 1;

  String get cameraLabel {
    if (cameraIndex < 0 || cameraIndex >= cameras.length) return 'camera';
    return labelFor(cameras, cameras[cameraIndex]);
  }

  void setCameras(List<CameraChoice> found, int index) {
    cameras = found;
    cameraIndex = index;
    rebuildZones();
  }

  Future<void> cycleCamera() async {
    if (!hasCameraChoice) return;
    final int wanted = nextIndex(cameras, cameraIndex);
    final Future<bool> Function(int index)? pick = onPickCamera;
    if (pick == null) {
      cameraIndex = wanted;
      rebuildZones();
      return;
    }
    if (await pick(wanted)) {
      cameraIndex = wanted;
      tracker.reset();
      rebuildZones();
    }
  }

  double? pressX;
  double? pressY;
  String? pressKey;

  int get minutes => _minutes;
  String get mode => _mode;

  void Function(int v)? onVolume;
  void Function(Hit hit)? onHitSound;
  void Function()? onDoneSound;
  void Function()? onClose;

  bool get inMenu => page == pageMenu;

  bool get showSummary =>
      page == pagePlay && engine.state.phase == finished;

  bool get showPause => page == pagePlay && engine.state.phase == paused;

  void resize(double width, double height) {
    if (width == w && height == h) return;
    w = width;
    h = height;
    rebuildZones();
  }

  void rebuildZones() {
    zones.clear();
    Zones built;
    if (page == pageMenu) {
      built = menuZones(w, h, _mode, camera: hasCameraChoice);
    } else if (showSummary) {
      built = summaryZones(w, h);
    } else if (showPause) {
      built = pauseZones(w, h);
    } else {
      built = playZones(w, h);
    }
    for (final String key in built.keys) {
      zones.add(key, built[key]!);
    }
  }

  void setMode(String mode) {
    _mode = mode;
    engine.mode = mode;
    rebuildZones();
  }

  void setMinutes(double x) {
    final Zone? r = zones['minutes'];
    if (r == null) return;
    _minutes = sliderValue(r, x, lo: minMinutes, hi: maxMinutes);
  }

  void setVolume(int v) {
    volume = v.clamp(volMin, volMax);
    onVolume?.call(volume);
  }

  List<String> get shapeWords => shape(_minutes);

  void begin() {
    engine.mode = _mode;
    engine.duration = _minutes * 60.0;
    engine.reset();
    tracker.reset();
    finishedAt = null;
    page = pagePlay;
    rebuildZones();
  }

  void toMenu() {
    page = pageMenu;
    finishedAt = null;
    rebuildZones();
  }

  void step(PoseFrame? frame, double t) {
    now = t;
    if (page != pagePlay) return;

    final String before = engine.state.phase;

    if (frame != null) {
      tracker.update(frame);
      tracking = tracker.torso != null;
      final List<Hit> scored = engine.update(
        tracker.points,
        tracker.visibility,
        idx,
        tracker.torso,
        t,
        frame.width,
        frame.height,
      );
      for (final Hit hit in scored) {
        onHitSound?.call(hit);
      }
    } else {
      tracking = false;
      engine.update(null, null, idx, null, t, w, h);
    }

    if (engine.state.phase != before) rebuildZones();
    if (engine.state.phase == finished && finishedAt == null) {
      finishedAt = t;
      onDoneSound?.call();
      rebuildZones();
    }
  }

  double summaryAlpha(double t) {
    final double? at = finishedAt;
    if (at == null) return 0.0;
    return ((t - at) / summaryFade).clamp(0.0, 1.0);
  }

  Summary summary() => engine.summary(now);

  void tapMenu(String key) {
    if (key == timed || key == calm) {
      setMode(key);
    } else if (key == 'begin') {
      begin();
    } else if (key == 'camera') {
      cycleCamera();
    }
  }

  void tapPlay(String key) {
    if (key == 'stop') {
      engine.pause(now);
      rebuildZones();
    }
  }

  void tapPause(String key) {
    if (key == 'resume') {
      engine.resume(now);
      rebuildZones();
    } else if (key == 'finish') {
      engine.beginEnding(now);
      engine.finish(now);
      finishedAt ??= now;
      onDoneSound?.call();
      rebuildZones();
    }
  }

  void tapSummary(String key) {
    if (key == 'again') {
      begin();
    } else if (key == 'menu') {
      toMenu();
    } else if (key == 'close') {
      closed = true;
      onClose?.call();
    }
  }

  void pointerDown(double x, double y) {
    pressX = x;
    pressY = y;
    pressKey = zones.at(x, y);
    if (page == pageMenu && pressKey == 'minutes') {
      sliding = true;
      setMinutes(x);
    }
  }

  void pointerMove(double x, double y) {
    if (sliding) setMinutes(x);
  }

  void pointerUp(double x, double y) {
    if (sliding) {
      setMinutes(x);
      sliding = false;
      pressX = null;
      pressY = null;
      pressKey = null;
      return;
    }

    final double? px = pressX;
    final double? py = pressY;
    if (px == null || py == null) {
      pressKey = null;
      return;
    }

    final double moved = (x - px).abs() + (y - py).abs();
    final String? key = zones.at(x, y);
    if (moved <= tapSlop && key != null && key == pressKey &&
        key != 'panel') {
      if (page == pageMenu) {
        tapMenu(key);
      } else if (showSummary) {
        tapSummary(key);
      } else if (showPause) {
        tapPause(key);
      } else {
        tapPlay(key);
      }
    }

    pressX = null;
    pressY = null;
    pressKey = null;
  }

  double unit() {
    final double? t = tracker.torso;
    return t == null ? 4.0 : t / 22.0;
  }

  List<Target> get liveTargets => engine.targets;

  Points? get drawPoints => tracker.points;

  List<double>? get drawVisibility => tracker.visibility;

  List<bool>? get drawDiscarded => tracker.discarded;
}
